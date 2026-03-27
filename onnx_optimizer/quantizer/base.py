"""
量化器基类和执行器接口
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import ModelProto

from ..config import QuantizationConfig
from ..utils import console


@dataclass
class QuantizationResult:
    """量化结果"""
    quantized_weights: Dict[str, np.ndarray]  # 量化后的权重
    scales: Dict[str, np.ndarray]             # 缩放因子
    zero_points: Optional[Dict[str, np.ndarray]] = None  # 零点（对称量化为None）


class QuantizationExecutor(ABC):
    """
    量化执行器基类
    
    每个具体的量化算法需要实现这个接口
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """执行器名称"""
        pass
    
    @property
    @abstractmethod
    def supported_dtypes(self) -> Tuple[str, ...]:
        """支持的数据类型"""
        pass
    
    @abstractmethod
    def quantize(
        self,
        weights: np.ndarray,
        tensor_name: str,
        config: QuantizationConfig
    ) -> QuantizationResult:
        """
        量化单个权重张量
        
        Args:
            weights: 原始权重 [out_features, in_features]
            tensor_name: 张量名称（用于日志）
            config: 量化配置
        
        Returns:
            量化结果
        """
        pass
    
    @abstractmethod
    def dequantize(
        self,
        q_weights: np.ndarray,
        scales: np.ndarray,
        zero_points: Optional[np.ndarray],
        group_size: Optional[int] = None
    ) -> np.ndarray:
        """
        反量化
        
        Args:
            q_weights: 量化权重
            scales: 缩放因子
            zero_points: 零点
            group_size: 分组大小（如果是分组量化）
        
        Returns:
            反量化后的FP16权重
        """
        pass
    
    def should_quantize(self, tensor_name: str, config: QuantizationConfig) -> bool:
        """
        判断是否应该量化该张量
        
        Args:
            tensor_name: 张量名称
            config: 量化配置
        
        Returns:
            是否应该量化
        """
        # 检查保留的层
        if config.preserve_embedding and "embed_tokens" in tensor_name:
            return False
        
        if config.preserve_norm and "norm" in tensor_name and "layernorm" not in tensor_name:
            return False
        
        if config.preserve_lm_head and "lm_head" in tensor_name:
            return False
        
        # 只量化2D矩阵（线性层权重）
        # 实际shape检查在调用者处进行
        return True


class BaseQuantizer(ABC):
    """
    量化器基类
    
    负责整体量化流程管理，调用具体的Executor进行实际量化
    """
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.executor = self._create_executor()
        self.console = console
    
    @abstractmethod
    def _create_executor(self) -> QuantizationExecutor:
        """创建量化执行器"""
        pass
    
    def quantize(
        self,
        model: ModelProto,
        manifest: Dict[str, Any],
        data_path: Path,
        output_dir: Path
    ) -> Tuple[ModelProto, Dict[str, Any], Path]:
        """
        执行模型量化
        
        Args:
            model: ONNX模型
            manifest: 权重manifest
            data_path: 原始数据文件路径
            output_dir: 输出目录
        
        Returns:
            (量化后的模型, 新的manifest, 新的数据文件路径)
        """
        from ..utils import (
            load_external_tensor, save_external_tensor, 
            save_manifest, ProgressTracker
        )
        from onnx import TensorProto, helper
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_data_path = output_dir / "data.bin"
        
        quantized_manifest = {}
        offset = 0
        
        with ProgressTracker(
            f"量化权重 ({self.executor.name})",
            total=len(model.graph.initializer)
        ) as progress:
            for init in model.graph.initializer:
                name = init.name
                progress.set_description(f"量化: {name[:50]}...")
                
                # 加载原始权重
                if name in manifest:
                    entry = manifest[name]
                    weights = load_external_tensor(entry, data_path)
                else:
                    # 内部数据
                    dtype = np.float16 if init.data_type == TensorProto.FLOAT16 else np.float32
                    weights = np.frombuffer(init.raw_data, dtype=dtype).reshape(list(init.dims))
                
                # 判断是否应该量化
                if len(weights.shape) == 2 and self.executor.should_quantize(name, self.config):
                    # 执行量化
                    result = self.executor.quantize(weights, name, self.config)
                    
                    # 保存量化权重
                    q_weights = result.quantized_weights
                    offset = save_external_tensor(
                        q_weights, name, output_data_path, quantized_manifest, offset
                    )
                    
                    # 保存scale
                    if result.scales:
                        scale_name = f"{name}_scale"
                        offset = save_external_tensor(
                            result.scales, scale_name, output_data_path, 
                            quantized_manifest, offset
                        )
                    
                    # 更新initializer
                    self._update_initializer_quantized(init, q_weights.dtype)
                    
                else:
                    # 不量化，转为FP16保存
                    weights_fp16 = weights.astype(np.float16)
                    offset = save_external_tensor(
                        weights_fp16, name, output_data_path, quantized_manifest, offset
                    )
                    
                    # 更新initializer
                    self._update_initializer_fp16(init)
                
                progress.update(1)
        
        # 保存manifest
        save_manifest(quantized_manifest, output_dir / "manifest.json")
        
        # 更新模型中的外部数据引用
        self._update_model_external_data(model, quantized_manifest, output_data_path.name)
        
        return model, quantized_manifest, output_data_path
    
    def _update_initializer_quantized(self, init, dtype):
        """更新initializer为量化格式"""
        init.ClearField('raw_data')
        init.data_type = TensorProto.INT8 if dtype == np.int8 else TensorProto.FLOAT16
    
    def _update_initializer_fp16(self, init):
        """更新initializer为FP16格式"""
        init.ClearField('raw_data')
        init.data_type = TensorProto.FLOAT16
    
    def _update_model_external_data(
        self, 
        model: ModelProto, 
        manifest: Dict[str, Any],
        data_file_name: str
    ):
        """更新模型的外部数据引用"""
        from onnx import TensorProto
        
        for init in model.graph.initializer:
            name = init.name
            if name in manifest:
                init.data_location = TensorProto.EXTERNAL
                
                # 添加外部数据信息
                init.external_data.clear()
                entries = [
                    ("location", data_file_name),
                    ("offset", str(manifest[name]["offset"])),
                    ("length", str(manifest[name]["size"]))
                ]
                for key, value in entries:
                    entry = init.external_data.add()
                    entry.key = key
                    entry.value = value
