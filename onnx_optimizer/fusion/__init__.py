"""
层融合模块 (O1~O3级别)

提供基于YAML配置的子图匹配和替换功能

优化级别:
- O1: 基础融合（标准ONNX算子）
- O2: 激进融合（ORT支持的非标准算子）
- O3: 自定义算子（需要推理框架实现）
"""

from .pass_manager import FusionPassManager, FusionPass
from .config_loader import FusionConfigLoader
from .graph_matcher import (
    GraphMatcher, SequenceMatcher, MatchResult,
    load_onnx_to_gs, save_gs_to_onnx
)
from .graph_replacer import (
    SubgraphReplacer, ReplacementBuilder, apply_fusion_pass
)

__all__ = [
    'FusionPassManager', 'FusionConfigLoader', 'FusionPass',
    'GraphMatcher', 'SequenceMatcher', 'MatchResult',
    'SubgraphReplacer', 'ReplacementBuilder', 'apply_fusion_pass',
    'load_onnx_to_gs', 'save_gs_to_onnx'
]
