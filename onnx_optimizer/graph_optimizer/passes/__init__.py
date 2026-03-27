"""
图优化Pass实现
"""

from .constant_folding import ConstantFoldingPass
from .dead_code_elimination import DeadCodeEliminationPass
from .input_statication import InputStaticationPass
from .identity_removal import IdentityRemovalPass

__all__ = [
    'ConstantFoldingPass',
    'DeadCodeEliminationPass',
    'InputStaticationPass',
    'IdentityRemovalPass',
]
