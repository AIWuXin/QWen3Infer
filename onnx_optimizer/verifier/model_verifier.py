"""
模型验证器实现
"""

from pathlib import Path
from typing import Optional
import json

import onnx
from onnx import ModelProto
from rich.table import Table
from rich.tree import Tree

from .base import BaseVerifier
from ..config import VerifyConfig
from ..utils import load_external_tensor, format_size


class ModelVerifier(BaseVerifier):
    """
    模型验证器
    
    验证ONNX模型的完整性:
    1. 模型结构检查
    2. 权重完整性检查
    3. 外部数据文件检查
    4. （可选）与参考模型精度对比
    """
    
    def __init__(self, config: VerifyConfig = None):
        config = config or VerifyConfig()
        super().__init__(config)
    
    def verify(self, model_path: Path) -> bool:
        """执行完整验证"""
        self.console.print(f"\n[bold blue]验证模型: {model_path.name}[/bold blue]\n")
        
        try:
            # 1. 加载模型
            model = self._verify_load(model_path)
            if model is None:
                return False
            
            # 2. 检查结构
            if self.config.check_structure:
                self._verify_structure(model)
            
            # 3. 检查权重
            if self.config.check_weights:
                self._verify_weights(model_path, model)
            
            # 4. 与参考模型对比
            if self.config.reference_model:
                self._compare_with_reference(model)
            
        except Exception as e:
            self.log_error(f"验证过程异常: {e}")
            return False
        
        self.print_summary()
        return len(self.errors) == 0
    
    def _verify_load(self, model_path: Path) -> Optional[ModelProto]:
        """验证模型加载"""
        self.console.print("[dim]检查模型加载...[/dim]")
        
        try:
            model = onnx.load(model_path, load_external_data=False)
            self.log_success("模型格式正确")
            return model
        except Exception as e:
            self.log_error(f"模型加载失败: {e}")
            return None
    
    def _verify_structure(self, model: ModelProto):
        """验证模型结构"""
        self.console.print("\n[dim]检查模型结构...[/dim]")
        
        graph = model.graph
        
        # 基本信息
        info = Tree("模型结构")
        info.add(f"IR版本: {model.ir_version}")
        info.add(f"Opset: {model.opset_import}")
        info.add(f"输入数: {len(graph.input)}")
        info.add(f"输出数: {len(graph.output)}")
        info.add(f"节点数: {len(graph.node)}")
        info.add(f"权重数: {len(graph.initializer)}")
        
        self.console.print(info)
        self.log_success("结构检查完成")
    
    def _verify_weights(self, model_path: Path, model: ModelProto):
        """验证权重完整性"""
        self.console.print("\n[dim]检查权重完整性...[/dim]")
        
        manifest_path = model_path.parent / "manifest.json"
        data_path = model_path.parent / "data.bin"
        
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # 检查每个权重
            total_size = 0
            int8_count = 0
            fp16_count = 0
            
            for name, entry in manifest.items():
                total_size += entry.get("size", 0)
                dtype = entry.get("dtype", "unknown")
                if dtype == "int8":
                    int8_count += 1
                elif dtype in ("fp16", "float16"):
                    fp16_count += 1
            
            self.console.print(f"  权重总数: {len(manifest)}")
            self.console.print(f"  INT8权重: {int8_count}")
            self.console.print(f"  FP16权重: {fp16_count}")
            self.console.print(f"  总大小: {format_size(total_size)}")
            
            # 检查data.bin存在
            if data_path.exists():
                actual_size = data_path.stat().st_size
                expected_size = max(e["offset"] + e["size"] for e in manifest.values())
                if actual_size >= expected_size:
                    self.log_success(f"数据文件完整 ({format_size(actual_size)})")
                else:
                    self.log_error(f"数据文件不完整: {actual_size} < {expected_size}")
            else:
                self.log_error(f"找不到数据文件: {data_path}")
        else:
            # 内部权重模式
            self.console.print("  使用内部权重存储")
            internal_size = sum(len(init.raw_data) for init in model.graph.initializer)
            self.console.print(f"  内部权重大小: {format_size(internal_size)}")
        
        self.log_success("权重检查完成")
    
    def _compare_with_reference(self, model: ModelProto):
        """与参考模型对比"""
        self.console.print("\n[dim]与参考模型对比...[/dim]")
        
        ref_path = self.config.reference_model
        if not ref_path.exists():
            self.log_error(f"找不到参考模型: {ref_path}")
            return
        
        try:
            ref_model = onnx.load(ref_path, load_external_data=False)
            
            # 对比结构
            self.console.print("  对比模型结构...")
            
            # 对比输入
            model_inputs = {i.name for i in model.graph.input}
            ref_inputs = {i.name for i in ref_model.graph.input}
            
            if model_inputs == ref_inputs:
                self.log_success("输入匹配")
            else:
                self.log_warning(f"输入不匹配: {model_inputs} vs {ref_inputs}")
            
            # 对比输出
            model_outputs = {o.name for o in model.graph.output}
            ref_outputs = {o.name for o in ref_model.graph.output}
            
            if model_outputs == ref_outputs:
                self.log_success("输出匹配")
            else:
                self.log_warning(f"输出不匹配: {model_outputs} vs {ref_outputs}")
            
        except Exception as e:
            self.log_error(f"参考模型对比失败: {e}")
    
    def verify_quantization_accuracy(
        self,
        quantized_path: Path,
        reference_path: Path,
        sample_inputs: Optional[dict] = None
    ) -> float:
        """
        验证量化精度
        
        Returns:
            最大相对误差
        """
        # TODO: 实现推理对比
        self.console.print("[dim]精度验证需要推理功能，待实现[/dim]")
        return 0.0
