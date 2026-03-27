"""
FP16转换器

将FP32权重转换为FP16格式（非量化，只是精度转换）
"""

import numpy as np
from typing import Optional

from .base import QuantizationExecutor, QuantizationResult
from .registry import register_quantizer
from ..config import QuantizationMode, QuantizationConfig


@register_quantizer(QuantizationMode.FP16)
class FP16Converter(QuantizationExecutor):
    """
    FP16精度转换器
    
    将FP32权重转换为FP16格式，不执行量化，
    只是简单的精度降低，用于减少模型大小。
    """
    
    @property
    def name(self) -> str:
        return "FP16Converter"
    
    @property
    def supported_dtypes(self) -> tuple:
        return ("float32",)
    
    def quantize(
        self,
        weights: np.ndarray,
        tensor_name: str,
        config: QuantizationConfig
    ) -> QuantizationResult:
        """
        转换权重为FP16
        
        实际上不是量化，只是精度转换
        """
        # 转换为FP16
        weights_fp16 = weights.astype(np.float16)
        
        # 返回结果（用quantized_weights存储，但实际是FP16）
        return QuantizationResult(
            quantized_weights={tensor_name: weights_fp16},
            scales={},  # FP16转换不需要scale
            zero_points=None
        )
    
    def dequantize(
        self,
        q_weights: np.ndarray,
        scales: np.ndarray,
        zero_points: Optional[np.ndarray],
        group_size: Optional[int] = None
    ) -> np.ndarray:
        """
        反量化（对于FP16转换就是原样返回）
        """
        # FP16转换无需反量化
        return q_weights
    
    def should_quantize(self, tensor_name: str, config: QuantizationConfig) -> bool:
        """
        FP16转换应该应用于所有FP32权重
        """
        # 转换所有权重，不管维度
        return True
