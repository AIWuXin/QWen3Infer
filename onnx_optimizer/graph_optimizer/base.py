"""
图优化Pass基类
"""

from abc import ABC, abstractmethod
from typing import Tuple

from onnx import ModelProto


class BaseGraphPass(ABC):
    """图优化Pass基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.change_count = 0
    
    @abstractmethod
    def run(self, model: ModelProto) -> Tuple[ModelProto, bool]:
        """
        执行优化
        
        Args:
            model: ONNX模型
        
        Returns:
            (优化后的模型, 是否做了修改)
        """
        pass
    
    def log(self, message: str):
        """记录日志"""
        print(f"[{self.name}] {message}")


class OptimizationResult:
    """优化结果"""
    
    def __init__(self):
        self.passes_applied = []
        self.total_changes = 0
        self.initial_node_count = 0
        self.final_node_count = 0
    
    def add_pass(self, name: str, changes: int):
        """添加Pass结果"""
        self.passes_applied.append({
            'name': name,
            'changes': changes
        })
        self.total_changes += changes
