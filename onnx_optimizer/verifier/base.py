"""
验证器基类
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

import onnx
from onnx import ModelProto

from ..config import VerifyConfig
from ..utils import console


class BaseVerifier(ABC):
    """验证器基类"""
    
    def __init__(self, config: VerifyConfig):
        self.config = config
        self.console = console
        self.errors = []
        self.warnings = []
    
    @abstractmethod
    def verify(self, model_path: Path) -> bool:
        """
        执行验证
        
        Args:
            model_path: 模型路径
        
        Returns:
            验证是否通过
        """
        pass
    
    def load_model(self, model_path: Path) -> ModelProto:
        """加载模型"""
        try:
            return onnx.load(model_path, load_external_data=False)
        except Exception as e:
            self.errors.append(f"模型加载失败: {e}")
            raise
    
    def log_error(self, message: str):
        """记录错误"""
        self.errors.append(message)
        self.console.print(f"  [red]✗[/red] {message}")
    
    def log_warning(self, message: str):
        """记录警告"""
        self.warnings.append(message)
        self.console.print(f"  [yellow]![/yellow] {message}")
    
    def log_success(self, message: str):
        """记录成功"""
        self.console.print(f"  [green]✓[/green] {message}")
    
    def print_summary(self):
        """打印验证摘要"""
        from rich.table import Table
        from rich.panel import Panel
        
        table = Table(show_header=False)
        table.add_column("项目")
        table.add_column("数量", justify="right")
        
        table.add_row("错误", str(len(self.errors)))
        table.add_row("警告", str(len(self.warnings)))
        
        status = "[green]通过[/green]" if not self.errors else "[red]失败[/red]"
        table.add_row("状态", status)
        
        self.console.print(Panel(table, title="验证结果"))
