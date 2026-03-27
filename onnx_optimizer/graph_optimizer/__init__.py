"""
图优化模块 (O0级别)

提供基础的图级别优化，不修改模型结构，只进行:
- 常量折叠
- 死代码消除
- 输入静态化
- 形状推导
- Identity节点移除
"""

from .pipeline import GraphOptimizationPipeline
from .base import BaseGraphPass
from .passes.constant_folding import ConstantFoldingPass
from .passes.dead_code_elimination import DeadCodeEliminationPass
from .passes.input_statication import InputStaticationPass
from .passes.identity_removal import IdentityRemovalPass

__all__ = [
    'GraphOptimizationPipeline',
    'BaseGraphPass',
    'ConstantFoldingPass',
    'DeadCodeEliminationPass',
    'InputStaticationPass',
    'IdentityRemovalPass',
]
