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
            
            # 合并KV输出（56个 -> 2个）
            self._merge_kv_outputs(graph)
            
            graph.cleanup()
            
            prefill_model = gs.export_onnx(graph)
            prefill_model.ir_version = self.original_model.ir_version
            prefill_model.opset_import.extend(self.original_model.opset_import)
            
            console.print(f"  [dim]移除 {len(kv_inputs)} 个KV cache输入, 剩余 {len(graph.inputs)} 个输入[/dim]")
            console.print(f"  [dim]合并KV输出: 56个 -> 2个堆叠张量[/dim]")
            return prefill_model
    
    def _create_decode_model(self) -> onnx.ModelProto:
        """创建Decode模型：优化KV Cache结构"""

        with ProgressTracker("创建Decode模型"):
            graph = gs.import_onnx(self.original_model)
            
            # 1. 合并KV输入（56个 -> 2个）
            num_layers, num_heads = self._merge_kv_inputs(graph)
            
            # 2. 合并KV输出（56个 -> 2个）
            self._merge_kv_outputs(graph)
            
            graph.cleanup()
            
            decode_model = gs.export_onnx(graph)
            decode_model.ir_version = self.original_model.ir_version
            decode_model.opset_import.extend(self.original_model.opset_import)
            
            if num_layers > 0:
                console.print(f"  [dim]合并 {num_layers * 2} 个KV张量 -> 2个堆叠张量[/dim]")
            return decode_model

    def _merge_kv_inputs(self, graph: gs.Graph) -> Tuple[int, int]:
        """
        将分散的KV输入合并为2个堆叠张量
        
        原始: past_key_values.0.key, past_key_values.0.value, ..., past_key_values.{n-1}.value (2n个)
        合并: past_keys, past_values (2个)
        
        并在图中插入Split节点将堆叠张量分解为原始格式供后续节点使用
        
        Returns:
            (num_layers, num_heads) 用于后续处理
        """
        # 识别KV输入（格式: past_key_values.{layer}.key 或 past_key_values.{layer}.value）
        kv_key_inputs = []
        kv_value_inputs = []
        
        for inp in graph.inputs:
            if 'past_key_values.' in inp.name and '.key' in inp.name:
                # 提取层号
                try:
                    layer_idx = int(inp.name.split('past_key_values.')[1].split('.')[0])
                    kv_key_inputs.append((layer_idx, inp))
                except (IndexError, ValueError):
                    continue
            elif 'past_key_values.' in inp.name and '.value' in inp.name:
                try:
                    layer_idx = int(inp.name.split('past_key_values.')[1].split('.')[0])
                    kv_value_inputs.append((layer_idx, inp))
                except (IndexError, ValueError):
                    continue
        
        if not kv_key_inputs or not kv_value_inputs:
            console.print("  [yellow]警告: 未找到KV cache输入，跳过合并[/yellow]")
            return 0, 0
        
        # 按层号排序
        kv_key_inputs.sort(key=lambda x: x[0])
        kv_value_inputs.sort(key=lambda x: x[0])
        num_layers = len(kv_key_inputs)
        
        # 获取样本输入以推断形状
        sample_key_inp = kv_key_inputs[0][1]
        # 原始形状为 [batch, num_heads, past_seq_len, head_dim]
        # 其中 past_seq_len 是动态维度
        # 我们需要创建 [batch, num_layers, num_heads, past_seq_len, head_dim] (batch_first)
        
        # 从原始输入获取形状信息，保持动态维度
        # 假设原始形状: [batch, num_heads, seq_len, head_dim]
        original_shape = sample_key_inp.shape
        if len(original_shape) >= 4:
            # 保持batch和seq_len维度（可以是整数或字符串符号如 "batch_size"）
            batch_dim = original_shape[0]  # 可能是 "batch_size" 或具体数值
            num_heads_dim = original_shape[1] if len(original_shape) > 1 else self.config.num_heads  # 从原始输入获取num_heads
            seq_dim = original_shape[2] if len(original_shape) > 2 else None  # 可能是动态符号
            head_dim = original_shape[3] if len(original_shape) > 3 else self.config.head_dim  # 从原始输入获取head_dim
        else:
            batch_dim = None  # 保持动态
            num_heads_dim = self.config.num_heads
            seq_dim = None  # 保持动态
            head_dim = self.config.head_dim
        
        # 创建新的堆叠输入形状 [batch, num_layers, num_heads, seq_len, head_dim] (batch_first)
        # 保留原始的batch_dim（可能是字符串如 "batch_size"）
        stacked_shape = [batch_dim, num_layers, num_heads_dim, seq_dim, head_dim]
        
        past_keys_input = gs.Variable(
            name="past_keys",
            dtype=sample_key_inp.dtype,
            shape=stacked_shape
        )
        past_values_input = gs.Variable(
            name="past_values",
            dtype=kv_value_inputs[0][1].dtype,
            shape=stacked_shape
        )
        
        # 添加新输入到图
        graph.inputs.append(past_keys_input)
        graph.inputs.append(past_values_input)
        
        # 为每一层创建Split节点来提取对应的KV
        # 使用INT64_MAX表示动态维度的结束（ONNX Slice语义）
        INT64_MAX = 9223372036854775807
        
        for layer_idx in range(num_layers):
            # 获取原始输入的形状用于Squeeze输出
            old_key_inp = kv_key_inputs[layer_idx][1]
            old_value_inp = kv_value_inputs[layer_idx][1]
            original_key_shape = old_key_inp.shape if old_key_inp.shape else [None, self.config.num_heads, None, self.config.head_dim]
            
            # 创建key的Split/Squeeze操作
            # 使用Slice来提取: past_keys[:, layer_idx:layer_idx+1, ...] (batch_first)
            key_slice = gs.Node(
                op="Slice",
                inputs=[
                    past_keys_input,
                    gs.Constant(name=f"key_start_{layer_idx}", values=np.array([0, layer_idx, 0, 0, 0], dtype=np.int64)),
                    gs.Constant(name=f"key_end_{layer_idx}", values=np.array([INT64_MAX, layer_idx + 1, INT64_MAX, INT64_MAX, INT64_MAX], dtype=np.int64)),
                    gs.Constant(name=f"key_axes_{layer_idx}", values=np.array([0, 1, 2, 3, 4], dtype=np.int64)),
                    gs.Constant(name=f"key_steps_{layer_idx}", values=np.array([1, 1, 1, 1, 1], dtype=np.int64)),
                ],
                outputs=[gs.Variable(name=f"past_key_values.{layer_idx}.key_sliced", dtype=sample_key_inp.dtype)],
                name=f"Slice_Key_{layer_idx}"
            )
            graph.nodes.append(key_slice)
            
            # Squeeze去掉num_layers维度 (现在是第1维)
            key_squeeze = gs.Node(
                op="Squeeze",
                inputs=[key_slice.outputs[0], gs.Constant(name=f"squeeze_axes_key_{layer_idx}", values=np.array([1], dtype=np.int64))],
                outputs=[gs.Variable(name=f"past_key_values.{layer_idx}.key", dtype=sample_key_inp.dtype,
                                     shape=original_key_shape)],
                name=f"Squeeze_Key_{layer_idx}"
            )
            graph.nodes.append(key_squeeze)
            
            # 同样的处理 for value
            value_slice = gs.Node(
                op="Slice",
                inputs=[
                    past_values_input,
                    gs.Constant(name=f"value_start_{layer_idx}", values=np.array([0, layer_idx, 0, 0, 0], dtype=np.int64)),
                    gs.Constant(name=f"value_end_{layer_idx}", values=np.array([INT64_MAX, layer_idx + 1, INT64_MAX, INT64_MAX, INT64_MAX], dtype=np.int64)),
                    gs.Constant(name=f"value_axes_{layer_idx}", values=np.array([0, 1, 2, 3, 4], dtype=np.int64)),
                    gs.Constant(name=f"value_steps_{layer_idx}", values=np.array([1, 1, 1, 1, 1], dtype=np.int64)),
                ],
                outputs=[gs.Variable(name=f"past_key_values.{layer_idx}.value_sliced", dtype=kv_value_inputs[0][1].dtype)],
                name=f"Slice_Value_{layer_idx}"
            )
            graph.nodes.append(value_slice)
            
            value_squeeze = gs.Node(
                op="Squeeze",
                inputs=[value_slice.outputs[0], gs.Constant(name=f"squeeze_axes_value_{layer_idx}", values=np.array([1], dtype=np.int64))],
                outputs=[gs.Variable(name=f"past_key_values.{layer_idx}.value", dtype=kv_value_inputs[0][1].dtype,
                                     shape=original_key_shape)],
                name=f"Squeeze_Value_{layer_idx}"
            )
            graph.nodes.append(value_squeeze)
            
            # 将原来使用旧KV输入的节点重定向到新的Squeeze输出
            for node in graph.nodes:
                for i, inp in enumerate(node.inputs):
                    if inp == old_key_inp:
                        node.inputs[i] = key_squeeze.outputs[0]
                    elif inp == old_value_inp:
                        node.inputs[i] = value_squeeze.outputs[0]
            
            # 从图的输入列表中移除旧的KV输入
            if old_key_inp in graph.inputs:
                graph.inputs.remove(old_key_inp)
            if old_value_inp in graph.inputs:
                graph.inputs.remove(old_value_inp)
        
        console.print(f"  [dim]合并 {num_layers * 2} 个KV输入 -> 2个堆叠张量[/dim]")
        return num_layers, self.config.num_heads

    def _merge_kv_outputs(self, graph: gs.Graph) -> None:
        """
        将分散的KV输出合并为2个堆叠张量
        
        原始: present.0.key, present.0.value, ..., present.{n-1}.value (2n个)
        合并: present_keys, present_values (2个)
        
        在图中插入Concat节点将分散的输出合并为堆叠格式
        """
        # 识别KV输出（格式: present.{layer}.key 或 present.{layer}.value）
        kv_key_outputs = []
        kv_value_outputs = []
        
        for out in graph.outputs:
            if 'present.' in out.name and '.key' in out.name:
                try:
                    layer_idx = int(out.name.split('present.')[1].split('.')[0])
                    kv_key_outputs.append((layer_idx, out))
                except (IndexError, ValueError):
                    continue
            elif 'present.' in out.name and '.value' in out.name:
                try:
                    layer_idx = int(out.name.split('present.')[1].split('.')[0])
                    kv_value_outputs.append((layer_idx, out))
                except (IndexError, ValueError):
                    continue
        
        if not kv_key_outputs or not kv_value_outputs:
            console.print("  [yellow]警告: 未找到KV cache输出，跳过合并[/yellow]")
            return
        
        # 按层号排序
        kv_key_outputs.sort(key=lambda x: x[0])
        kv_value_outputs.sort(key=lambda x: x[0])
        num_layers = len(kv_key_outputs)
        
        # 首先为每个输出添加Unsqueeze节点，增加num_layers维度
        unsqueezed_keys = []
        unsqueezed_values = []
        
        # 获取样本输出的形状用于构建堆叠输出形状
        sample_key_out = kv_key_outputs[0][1]
        original_out_shape = sample_key_out.shape if sample_key_out.shape else [None, self.config.num_heads, None, self.config.head_dim]
        # 构建堆叠形状: [batch, num_layers, num_heads, seq_len, head_dim] (batch_first)
        # 原始形状是 [batch, num_heads, seq_len, head_dim]
        out_batch_dim = original_out_shape[0] if len(original_out_shape) > 0 else None
        out_num_heads = original_out_shape[1] if len(original_out_shape) > 1 else self.config.num_heads
        out_seq_dim = original_out_shape[2] if len(original_out_shape) > 2 else None
        out_head_dim = original_out_shape[3] if len(original_out_shape) > 3 else self.config.head_dim
        stacked_out_shape = [out_batch_dim, num_layers, out_num_heads, out_seq_dim, out_head_dim]
        
        for layer_idx in range(num_layers):
            old_key_out = kv_key_outputs[layer_idx][1]
            old_value_out = kv_value_outputs[layer_idx][1]
            
            # Unsqueeze key: [batch, heads, seq, dim] -> [batch, 1, heads, seq, dim] (在第1维插入)
            key_unsqueeze = gs.Node(
                op="Unsqueeze",
                inputs=[old_key_out, gs.Constant(name=f"unsqueeze_axes_key_out_{layer_idx}", values=np.array([1], dtype=np.int64))],
                outputs=[gs.Variable(name=f"present.{layer_idx}.key_unsqueezed", dtype=old_key_out.dtype)],
                name=f"Unsqueeze_Key_Out_{layer_idx}"
            )
            graph.nodes.append(key_unsqueeze)
            unsqueezed_keys.append(key_unsqueeze.outputs[0])
            
            # Unsqueeze value
            value_unsqueeze = gs.Node(
                op="Unsqueeze",
                inputs=[old_value_out, gs.Constant(name=f"unsqueeze_axes_value_out_{layer_idx}", values=np.array([1], dtype=np.int64))],
                outputs=[gs.Variable(name=f"present.{layer_idx}.value_unsqueezed", dtype=old_value_out.dtype)],
                name=f"Unsqueeze_Value_Out_{layer_idx}"
            )
            graph.nodes.append(value_unsqueeze)
            unsqueezed_values.append(value_unsqueeze.outputs[0])
            
            # 从图的输出列表中移除旧的分散输出
            if old_key_out in graph.outputs:
                graph.outputs.remove(old_key_out)
            if old_value_out in graph.outputs:
                graph.outputs.remove(old_value_out)
        
        # 创建Concat节点将所有unsqueezed张量沿第1维（num_layers维度）拼接
        # Concat keys
        concat_keys = gs.Node(
            op="Concat",
            inputs=unsqueezed_keys,
            outputs=[gs.Variable(name="present_keys", dtype=kv_key_outputs[0][1].dtype,
                                 shape=stacked_out_shape)],
            attrs={"axis": 1},  # 在第1维拼接 (batch_first)
            name="Concat_Present_Keys"
        )
        graph.nodes.append(concat_keys)
        graph.outputs.append(concat_keys.outputs[0])
        
        # Concat values
        concat_values = gs.Node(
            op="Concat",
            inputs=unsqueezed_values,
            outputs=[gs.Variable(name="present_values", dtype=kv_value_outputs[0][1].dtype,
                                 shape=stacked_out_shape)],
            attrs={"axis": 1},  # 在第1维拼接 (batch_first)
            name="Concat_Present_Values"
        )
        graph.nodes.append(concat_values)
        graph.outputs.append(concat_values.outputs[0])
        
        console.print(f"  [dim]合并 {num_layers * 2} 个KV输出 -> 2个堆叠张量[/dim]")
    
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
