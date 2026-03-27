"""
Prefill/Decode拆分导出器

参考export_qwen3/post_process_onnx.py的实现
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from copy import deepcopy

import onnx
from onnx import TensorProto, helper
import numpy as np
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from .base import BaseExporter
from ..config import ExportConfig
from ..utils import (
    console, ProgressTracker, load_external_tensor, save_external_tensor,
    load_manifest, save_manifest, format_size
)
from shutil import copyfile
import onnx_graphsurgeon as gs


class SplitExporter(BaseExporter):
    """
    将完整的生成模型拆分为Prefill和Decode两个子模型
    
    Prefill: 处理输入prompt的并行计算
    Decode: 处理单token自回归生成
    """
    
    def __init__(self, config: ExportConfig):
        super().__init__(config)
        self.original_model: Optional[onnx.ModelProto] = None
        self.original_manifest: Dict[str, Any] = {}
        self.original_data_path: Optional[Path] = None
        self.initializers = None
    
    def export(self) -> Path:
        """执行拆分导出"""
        input_dir = self.config.input_dir
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 加载原始模型
        self._load_original_model(input_dir)
        
        # 2. 提取权重和manifest
        data_path, manifest = self._extract_weights()
        
        # 3. 创建Prefill/Decode模型
        prefill_model = self._create_prefill_model()
        
        # 4. 创建Decode模型
        decode_model = self._create_decode_model()
        
        # 5. 保存模型
        self._save_weightless_onnx(prefill_model, output_dir / "prefill.onnx", manifest)
        self._save_weightless_onnx(decode_model, output_dir / "decode.onnx", manifest)
        
        # 6. 复制配置文件
        copyfile(input_dir / "config.json", output_dir / "config.json")
        
        # 7. 打印摘要
        self._print_summary(output_dir, prefill_model, decode_model, manifest)
        
        return output_dir
    
    def _load_original_model(self, input_dir: Path):
        """加载原始模型"""
        model_path = input_dir / "model.onnx"
        manifest_path = input_dir / "manifest.json"
        data_path = input_dir / "data.bin"
        
        if not model_path.exists():
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        
        with ProgressTracker("加载原始模型"):
            self.original_model = onnx.load(model_path, load_external_data=False)
            
        self.initializers = {init.name: init for init in self.original_model.graph.initializer}
            
    def _extract_weights(self) -> str:
        """
        提取所有 initializer 到 data.bin，返回 manifest
        """
        data_bin_path = self.config.output_dir / "data.bin"
        manifest = {}
        offset = 0
        
        # 按名称排序确保确定性
        weight_names = sorted(self.initializers.keys())
        
        with open(data_bin_path, 'wb') as f:
            with ProgressTracker("提取权重", total=len(weight_names)):
                for name in weight_names:
                    tensor = self.initializers[name]
                    # 解析 raw_data
                    if tensor.raw_data:
                        data = tensor.raw_data
                    else:
                        # 使用 numpy_helper 转换
                        arr = onnx.numpy_helper.to_array(tensor)
                        data = arr.tobytes()
                    
                    # 128字节对齐
                    padding = (128 - (len(data) % 128)) % 128
                    padded_data = data + b'\x00' * padding
                    
                    # 记录元数据
                    arr = onnx.numpy_helper.to_array(tensor)
                    manifest[name] = {
                        "offset": offset,
                        "size": len(data),
                        "padded_size": len(padded_data),
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                        "onnx_dtype": tensor.data_type
                    }
                    
                    f.write(padded_data)
                    offset += len(padded_data)
        
        console.print(f"Extracted {len(weight_names)} weights to {data_bin_path} ({offset/1024**2:.1f} MB)")
        
        # 保存 manifest
        import json
        manifest_path = self.config.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return str(data_bin_path), manifest
    
    def _create_manifest_from_model(self, model: onnx.ModelProto) -> Dict[str, Any]:
        """从模型创建manifest（内部数据情况）"""
        manifest = {}
        offset = 0
        
        for init in model.graph.initializer:
            size = len(init.raw_data)
            manifest[init.name] = {
                "offset": offset,
                "size": size,
                "dtype": "fp16",  # 假设FP16
                "shape": list(init.dims)
            }
            offset += size
        
        return manifest
    
    def _create_prefill_model(self):
        """创建Prefill模型：移除past_key_values输入"""
        import onnx_graphsurgeon as gs
        
        with ProgressTracker("创建Prefill模型"):
            graph = gs.import_onnx(self.original_model)
            
            # 识别KV cache输入
            kv_inputs = [inp for inp in graph.inputs 
                        if 'past_key_values' in inp.name or 'past_key_' in inp.name]
            
            # 先断开与消费节点的连接（必须在从graph.inputs移除之前）
            for kv_inp in kv_inputs:
                # kv_inp.outputs 是消费这个tensor的节点列表
                for node in list(kv_inp.outputs):
                    if kv_inp in node.inputs:
                        node.inputs.remove(kv_inp)
            
            # 再从图的输入列表中移除
            for kv_inp in kv_inputs:
                if kv_inp in graph.inputs:
                    graph.inputs.remove(kv_inp)
            
            # 清理未使用的节点（包括原来消费KV输入的节点，如果它们没有其他输入了）
            graph.cleanup()
            
            prefill_model = gs.export_onnx(graph)
            prefill_model.ir_version = self.original_model.ir_version
            prefill_model.opset_import.extend(self.original_model.opset_import)
            
            console.print(f"  [dim]移除 {len(kv_inputs)} 个KV cache输入, 剩余 {len(graph.inputs)} 个输入[/dim]")
            return prefill_model
    
    def _create_decode_model(self) -> onnx.ModelProto:
        """创建Decode模型"""
        with ProgressTracker("创建Decode模型"):
            decode_model = onnx.ModelProto()
            decode_model.CopyFrom(self.original_model)
            decode_model.ir_version = self.original_model.ir_version
            decode_model.opset_import.extend(self.original_model.opset_import)
            
            return decode_model
    
    def _save_weightless_onnx(self, model: onnx.ModelProto, output_path: Path, manifest: dict):
        """
        保存 ONNX 模型，但将所有 initializer 转为外部引用，指向 data.bin
        """
        new_graph = helper.GraphProto()
        new_graph.name = model.graph.name
        
        # 复制节点
        new_graph.node.extend(model.graph.node)
        
        # 处理 initializer：清空 raw_data，设置 external_data
        for init in model.graph.initializer:
            new_init = helper.TensorProto()
            new_init.CopyFrom(init)
            
            if init.name in manifest:
                info = manifest[init.name]
                new_init.raw_data = b""  # 清空
                
                # 设置外部数据属性
                del new_init.external_data[:]
                
                loc = new_init.external_data.add()
                loc.key = "location"
                loc.value = "data.bin"
                
                off = new_init.external_data.add()
                off.key = "offset"
                off.value = str(info["offset"])
                
                length = new_init.external_data.add()
                length.key = "length"
                length.value = str(info["size"])
                
                new_init.data_location = TensorProto.EXTERNAL
            
            new_graph.initializer.append(new_init)
        
        # 复制输入输出（此时仍是56个张量形式，或如果调用了merge则为2个）
        new_graph.input.extend(model.graph.input)
        new_graph.output.extend(model.graph.output)
        
        # 复制其他信息
        if model.graph.doc_string:
            new_graph.doc_string = model.graph.doc_string
        
        new_model = helper.make_model(new_graph)
        new_model.ir_version = model.ir_version
        new_model.opset_import.extend(model.opset_import)
        
        onnx.save(new_model, str(output_path))
        console.print(f"Saved weightless ONNX: {output_path} ({output_path.stat().st_size/1024**2:.1f} MB structure)")
    
    def _print_summary(
        self,
        output_dir: Path,
        prefill_model: onnx.ModelProto,
        decode_model: onnx.ModelProto,
        manifest: Dict[str, Any]
    ):
        """打印导出摘要"""
        # 统计信息
        table = Table(title="拆分导出结果", show_header=True)
        table.add_column("模型", style="cyan")
        table.add_column("输入数", justify="right")
        table.add_column("输出数", justify="right")
        table.add_column("权重数", justify="right")
        table.add_column("文件大小", justify="right")
        
        prefill_path = output_dir / "prefill.onnx"
        decode_path = output_dir / "decode.onnx"
        data_path = output_dir / "data.bin"
        
        table.add_row(
            "Prefill",
            str(len(prefill_model.graph.input)),
            str(len(prefill_model.graph.output)),
            str(len(prefill_model.graph.initializer)),
            format_size(prefill_path.stat().st_size)
        )
        
        table.add_row(
            "Decode",
            str(len(decode_model.graph.input)),
            str(len(decode_model.graph.output)),
            str(len(decode_model.graph.initializer)),
            format_size(decode_path.stat().st_size)
        )
        
        table.add_row(
            "Data",
            "-",
            "-",
            str(len(manifest)),
            format_size(data_path.stat().st_size)
        )
        
        console.print(Panel(table, title="导出完成", border_style="green"))
        
        # 详细树状结构
        tree = Tree(f"[bold]{output_dir}[/bold]")
        tree.add(f"prefill.onnx ({format_size(prefill_path.stat().st_size)})")
        tree.add(f"decode.onnx ({format_size(decode_path.stat().st_size)})")
        tree.add(f"data.bin ({format_size(data_path.stat().st_size)})")
        tree.add("manifest.json")
        
        console.print(tree)
