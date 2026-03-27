"""
量化模块 - 提供多种量化算法实现

支持的量化算法:
- Int8GroupMax: 分组最大值量化 (默认)
- Int8Dynamic: 动态INT8量化
- FP16: FP16半精度转换
- (预留) Int4GPTQ: GPTQ 4bit量化
- (预留) Int4AWQ: AWQ 4bit量化

使用示例:
    >>> from onnx_optimizer.quantizer import QuantizerRegistry, QuantizationConfig
    >>> config = QuantizationConfig(mode='int8_group_max', group_size=128)
    >>> quantizer = QuantizerRegistry.get_quantizer(config)
    >>> quantizer.quantize(model, manifest, output_dir)
"""

from .base import BaseQuantizer, QuantizationExecutor
from .registry import QuantizerRegistry
from .int8_group_max import Int8GroupMaxExecutor
from .int8_dynamic import Int8DynamicExecutor
from .fp16_converter import FP16Converter
from ..config import QuantizationConfig, QuantizationMode

__all__ = [
    'BaseQuantizer',
    'QuantizationExecutor',
    'QuantizerRegistry',
    'Int8GroupMaxExecutor',
    'Int8DynamicExecutor', 
    'FP16Converter',
    'QuantizationConfig',
    'QuantizationMode',
]
