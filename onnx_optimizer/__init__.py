"""
ONNX Optimizer - 通用大模型ONNX优化工具

提供大语言模型(如Qwen3、Llama等)的ONNX转换、优化和量化功能。

主要功能模块:
- optimizer: 图级别优化
- quantizer: 量化(INT8/FP16)
- splitter: Prefill/Decode拆分
- verifier: 模型验证
- analyzer: 模型分析

示例:
    >>> from onnx_optimizer import QuantizationConfig, QuantizationMode
    >>> config = QuantizationConfig(mode=QuantizationMode.INT8_GROUP_MAX, group_size=128)
"""

__version__ = "0.1.0"
__author__ = "ONNX Optimizer Team"

from onnx_optimizer.cli import main
from onnx_optimizer.config import (
    ExportConfig,
    QuantizationConfig,
    GraphOptimizationConfig,
    FusionConfig,
    VerifyConfig,
    OptimizationLevel,
    QuantizationMode,
)

__all__ = [
    # CLI
    'main',

    # Config
    'ExportConfig',
    'QuantizationConfig',
    'GraphOptimizationConfig',
    'FusionConfig',
    'VerifyConfig',
    'OptimizationLevel',
    'QuantizationMode',
]

# 可选导入（延迟加载）
def _get_analyzer():
    from .analyzer import ModelAnalyzer, auto_detect_config, ModelConfig
    return ModelAnalyzer, auto_detect_config, ModelConfig

# 导出analyzer相关类
from .analyzer import ModelAnalyzer, auto_detect_config, ModelConfig
__all__.extend(['ModelAnalyzer', 'auto_detect_config', 'ModelConfig'])
