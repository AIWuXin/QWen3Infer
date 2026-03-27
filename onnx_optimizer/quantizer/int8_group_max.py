"""
INT8分组最大值量化执行器

实现分组最大值对称量化算法
"""

import numpy as np
from typing import Optional

from .base import QuantizationExecutor, QuantizationResult
from .registry import register_quantizer
from ..config import QuantizationMode, QuantizationConfig


@register_quantizer(QuantizationMode.INT8_GROUP_MAX)
class Int8GroupMaxExecutor(QuantizationExecutor):
    """
    INT8分组最大值量化
    
    将权重按指定group_size分组，每组使用最大值计算scale
    支持对称量化（zero_point=0）
    
    量化公式:
        scale = max(abs(group)) / 127
        int8 = round(fp16 / scale)
    
    Attributes:
        group_size: 分组大小，默认128
    """
    
    def __init__(self, group_size: int = 128):
        self.group_size = group_size
    
    @property
    def name(self) -> str:
        return "Int8GroupMax"
    
    @property
    def supported_dtypes(self) -> tuple:
        return ("float16", "float32")
    
    def quantize(
        self,
        weights: np.ndarray,
        tensor_name: str,
        config: QuantizationConfig
    ) -> QuantizationResult:
        """
        执行分组最大值量化
        
        Args:
            weights: 原始权重 [out_features, in_features]
            tensor_name: 张量名称（用于日志）
            config: 量化配置
        
        Returns:
            QuantizationResult
        """
        group_size = config.group_size
        original_shape = weights.shape
        
        # 确保是2D
        if len(original_shape) != 2:
            raise ValueError(f"只支持2D权重，得到 {original_shape}")
        
        out_features, in_features = original_shape
        
        # 对in_features维度进行padding，使其能被group_size整除
        pad_len = (group_size - in_features % group_size) % group_size
        if pad_len > 0:
            weights_padded = np.pad(weights, ((0, 0), (0, pad_len)))
        else:
            weights_padded = weights
        
        num_groups = weights_padded.shape[1] // group_size
        
        # reshape为 [out_features, num_groups, group_size]
        weights_grouped = weights_padded.reshape(out_features, num_groups, group_size)
        
        # 计算每组的绝对最大值
        abs_max = np.abs(weights_grouped).max(axis=2, keepdims=True)
        
        # 避免除零：全零组的scale设为1
        abs_max_safe = np.where(abs_max == 0, 1.0, abs_max)
        
        # 计算scale: max / 127 (INT8最大值)
        scales = (abs_max_safe / 127.0).astype(np.float16)
        
        # 量化
        quantized_grouped = np.clip(
            np.round(weights_grouped / scales),
            -127, 127
        ).astype(np.int8)
        
        # reshape回原始形状（去除padding）
        quantized_padded = quantized_grouped.reshape(out_features, -1)
        quantized = quantized_padded[:, :in_features]
        
        # squeeze scales: [out_features, num_groups, 1] -> [out_features, num_groups]
        scales = np.squeeze(scales, axis=2)
        
        return QuantizationResult(
            quantized_weights={tensor_name: quantized},
            scales={tensor_name: scales},
            zero_points=None  # 对称量化，zero_point=0
        )
    
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
            q_weights: INT8量化权重 [out_features, in_features]
            scales: 缩放因子 [out_features, num_groups]
            zero_points: 未使用（对称量化）
            group_size: 分组大小
        
        Returns:
            反量化后的FP16权重
        """
        if group_size is None:
            group_size = self.group_size
        
        out_features, in_features = q_weights.shape
        
        # padding
        pad_len = (group_size - in_features % group_size) % group_size
        if pad_len > 0:
            q_weights_padded = np.pad(q_weights, ((0, 0), (0, pad_len)))
        else:
            q_weights_padded = q_weights
        
        num_groups = q_weights_padded.shape[1] // group_size
        
        # reshape
        q_grouped = q_weights_padded.reshape(out_features, num_groups, group_size)
        
        # 检查scales形状
        if scales.shape != (out_features, num_groups):
            raise ValueError(
                f"scales形状不匹配: 期望 {(out_features, num_groups)}, 得到 {scales.shape}"
            )
        
        # 反量化
        scales_expanded = scales.reshape(out_features, num_groups, 1)
        fp_grouped = q_grouped.astype(np.float32) * scales_expanded
        
        # reshape back
        fp_weights_padded = fp_grouped.reshape(out_features, -1)
        fp_weights = fp_weights_padded[:, :in_features]
        
        return fp_weights.astype(np.float16)
