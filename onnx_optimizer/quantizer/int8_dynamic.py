"""
INT8动态量化执行器 (预留)

动态量化在推理时根据激活值动态计算scale
适用于对延迟敏感的场景
"""

import numpy as np
from typing import Optional

from .base import QuantizationExecutor, QuantizationResult
from .registry import register_quantizer
from ..config import QuantizationMode, QuantizationConfig


@register_quantizer(QuantizationMode.INT8_DYNAMIC)
class Int8DynamicExecutor(QuantizationExecutor):
    """
    INT8动态量化 (预留实现)
    
    动态量化在推理时根据输入数据动态计算scale和zero_point，
    不需要校准数据集，但可能精度略低于静态量化。
    
    TODO: 实现动态量化逻辑
    """
    
    @property
    def name(self) -> str:
        return "Int8Dynamic"
    
    @property
    def supported_dtypes(self) -> tuple:
        return ("float16", "float32")
    
    def quantize(
        self,
        weights: np.ndarray,
        tensor_name: str,
        config: QuantizationConfig
    ) -> QuantizationResult:
        """执行动态量化（当前回退到静态per-channel）"""
        # TODO: 实现真正的动态量化
        # 当前简单实现：使用per-channel静态量化作为占位
        
        if len(weights.shape) != 2:
            raise ValueError(f"只支持2D权重，得到 {weights.shape}")
        
        out_features, in_features = weights.shape
        
        # per-channel量化
        abs_max = np.abs(weights).max(axis=1, keepdims=True)
        abs_max_safe = np.where(abs_max == 0, 1.0, abs_max)
        
        scales = (abs_max_safe / 127.0).astype(np.float16)
        quantized = np.clip(np.round(weights / scales), -127, 127).astype(np.int8)
        
        return QuantizationResult(
            quantized_weights={tensor_name: quantized},
            scales={tensor_name: scales},
            zero_points=None
        )
    
    def dequantize(
        self,
        q_weights: np.ndarray,
        scales: np.ndarray,
        zero_points: Optional[np.ndarray],
        group_size: Optional[int] = None
    ) -> np.ndarray:
        """反量化"""
        scales_expanded = scales.reshape(-1, 1)
        return (q_weights.astype(np.float32) * scales_expanded).astype(np.float16)
