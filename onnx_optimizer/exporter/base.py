"""
导出器基类
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

import onnx
from rich.console import Console

from ..config import ExportConfig
from ..utils import console, ProgressTracker


class BaseExporter(ABC):
    """导出器基类"""
    
    def __init__(self, config: ExportConfig):
        self.config = config
        self.console = console
    
    @abstractmethod
    def export(self) -> Path:
        """
        执行导出
        
        Returns:
            输出目录路径
        """
        pass
    
    def load_model(self, model_path: Path) -> onnx.ModelProto:
        """加载ONNX模型"""
        with ProgressTracker(f"加载模型: {model_path.name}"):
            return onnx.load(model_path, load_external_data=False)
    
    def save_model(
        self,
        model: onnx.ModelProto,
        output_path: Path,
        external_data_path: Optional[Path] = None
    ):
        """
        保存ONNX模型
        
        Args:
            model: ONNX模型
            output_path: 输出路径
            external_data_path: 外部数据路径（如果使用外部存储）
        """
        with ProgressTracker(f"保存模型: {output_path.name}"):
            if external_data_path:
                # 使用外部数据格式
                onnx.save_model(
                    model,
                    output_path,
                    save_as_external_data=True,
                    location=external_data_path.name,
                    size_threshold=self.config.external_data_threshold * 1024 * 1024
                )
            else:
                onnx.save(model, output_path)
    
    def print_summary(self, output_dir: Path, manifest: Dict[str, Any]):
        """打印导出摘要"""
        from rich.table import Table
        from rich.panel import Panel
        
        table = Table(title="导出结果", show_header=True)
        table.add_column("项目", style="cyan")
        table.add_column("值", style="green")
        
        table.add_row("输出目录", str(output_dir))
        table.add_row("权重数量", str(len(manifest)))
        
        # 计算总大小
        total_size = sum(entry.get("size", 0) for entry in manifest.values())
        from ..utils import format_size
        table.add_row("总数据大小", format_size(total_size))
        
        self.console.print(Panel(table, title="导出完成", border_style="green"))
