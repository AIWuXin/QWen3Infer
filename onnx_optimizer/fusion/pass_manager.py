"""
层融合Pass管理器

管理所有融合Pass的加载和执行
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

import onnx
import onnx_graphsurgeon as gs
from rich.table import Table
from rich.panel import Panel

from ..config import FusionConfig
from ..utils import console
from .config_loader import FusionConfigLoader
from .graph_matcher import load_onnx_to_gs, save_gs_to_onnx
from .graph_replacer import apply_fusion_pass


class FusionPass:
    """
    融合Pass
    
    封装单个融合规则（模式匹配+替换）
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.name = config.get('name', 'unnamed_pass')
        self.description = config.get('description', '')
        self.enabled = config.get('enabled', True)
        self.pattern_config = config.get('pattern', {})
        self.replacement_config = config.get('replacement', {})
        self.fusion_count = 0
    
    def run(self, graph: gs.Graph) -> int:
        """
        执行融合Pass
        
        Args:
            graph: onnx-graphsurgeon图
            
        Returns:
            融合次数
        """
        if not self.enabled:
            return 0
        
        self.fusion_count = apply_fusion_pass(
            graph, 
            self.pattern_config, 
            self.replacement_config
        )
        
        return self.fusion_count
    
    def __repr__(self):
        return f"FusionPass({self.name})"


class FusionPassManager:
    """
    层融合Pass管理器
    
    根据配置加载和执行融合Pass
    """
    
    def __init__(self, config: FusionConfig):
        self.config = config
        self.loader = FusionConfigLoader()
        self.passes: List[FusionPass] = []
        self.fusion_stats: Dict[str, int] = {}
    
    def load_passes(self):
        """从配置加载所有融合Pass"""
        level_map = {
            0: None,  # O0: 不加载
            1: 'o1',
            2: 'o2',
            3: 'o3'
        }
        
        level_name = level_map.get(self.config.level.value)
        if level_name is None:
            return
        
        # 如果有自定义配置文件，使用它
        if self.config.config_file:
            config = self.loader.load(str(self.config.config_file))
        else:
            try:
                config = self.loader.load(level_name)
            except FileNotFoundError:
                console.print(f"[yellow]警告: 找不到{level_name}配置文件[/yellow]")
                return
        
        self._create_passes_from_config(config)
    
    def _create_passes_from_config(self, config: Dict[str, Any]):
        """根据配置创建Pass实例"""
        passes_config = config.get('passes', [])
        
        for pass_config in passes_config:
            name = pass_config.get('name', 'unnamed')
            enabled = pass_config.get('enabled', True)
            
            if not enabled:
                continue
            
            # 创建Pass实例
            pass_instance = FusionPass(pass_config)
            self.passes.append(pass_instance)
            
            console.print(f"[dim]  加载Pass: {name}[/dim]")
    
    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """
        执行所有融合Pass
        
        Args:
            model: 输入ONNX模型
        
        Returns:
            融合后的模型
        """
        if not self.passes:
            console.print("[dim]没有可用的融合Pass[/dim]")
            return model
        
        console.print(f"[bold blue]开始层融合 (O{self.config.level.value}级别)[/bold blue]")
        
        # 转换为onnx-graphsurgeon图
        graph = gs.import_onnx(model)
        
        # 执行每个Pass
        total_fusions = 0
        for pass_instance in self.passes:
            console.print(f"  执行: {pass_instance.name}...", end=" ")
            count = pass_instance.run(graph)
            self.fusion_stats[pass_instance.name] = count
            total_fusions += count
            console.print(f"[green]✓[/green] ({count}次融合)")
        
        # 转换回ONNX模型
        if total_fusions > 0:
            graph.cleanup()
            new_model = gs.export_onnx(graph)
            # 复制元数据
            new_model.ir_version = model.ir_version
            new_model.opset_import.extend(model.opset_import)
            model = new_model
            console.print(f"\n[green]共融合 {total_fusions} 处[/green]")
        else:
            console.print("\n[yellow]没有可以融合的子图[/yellow]")
        
        self._print_summary()
        
        return model
    
    def run_on_file(self, input_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        直接在文件上执行融合
        
        Args:
            input_path: 输入模型路径
            output_path: 输出模型路径（可选）
            
        Returns:
            输出模型路径
        """
        # 加载模型
        with console.status("[dim]加载模型...[/dim]"):
            graph, model = load_onnx_to_gs(input_path)
        
        if not self.passes:
            console.print("[dim]没有可用的融合Pass[/dim]")
            return input_path
        
        console.print(f"[bold blue]开始层融合 (O{self.config.level.value}级别)[/bold blue]")
        
        # 执行每个Pass
        total_fusions = 0
        for pass_instance in self.passes:
            console.print(f"  执行: {pass_instance.name}...", end=" ")
            count = pass_instance.run(graph)
            self.fusion_stats[pass_instance.name] = count
            total_fusions += count
            console.print(f"[green]✓[/green] ({count}次融合)")
        
        # 保存结果
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_fused{input_path.suffix}"
        
        if total_fusions > 0:
            with console.status(f"[dim]保存到 {output_path}...[/dim]"):
                new_model = save_gs_to_onnx(graph, model)
                onnx.save(new_model, output_path)
            console.print(f"\n[bold green]✓ 融合完成:[/bold green] {output_path}")
        else:
            console.print("\n[yellow]没有可以融合的子图[/yellow]")
        
        self._print_summary()
        
        return output_path
    
    def _print_summary(self):
        """打印融合摘要"""
        if not self.fusion_stats:
            return
        
        table = Table(title="层融合结果", show_header=True)
        table.add_column("Pass名称", style="cyan")
        table.add_column("融合次数", justify="right")
        
        total = 0
        for name, count in self.fusion_stats.items():
            table.add_row(name, str(count))
            total += count
        
        if total > 0:
            table.add_row("总计", str(total), style="bold")
        
        console.print(Panel(table, border_style="green"))
