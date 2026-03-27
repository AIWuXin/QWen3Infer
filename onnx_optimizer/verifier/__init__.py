"""
验证模块

提供模型验证功能:
- 结构完整性检查
- 权重验证
- 精度对比
- 推理测试
"""

from .base import BaseVerifier
from .model_verifier import ModelVerifier

__all__ = ['BaseVerifier', 'ModelVerifier']
