"""
图优化管道

管理所有O0级别优化Pass的执行
"""

from typing import List, Type
from onnx import ModelProto
from rich.table import Table
from rich.panel import Panel

from ..config import GraphOptimizationConfig
from ..utils import console
from .base import BaseGraphPass, OptimizationResult
from .passes import (
    ConstantFoldingPass,
    DeadCodeEliminationPass,
    InputStaticationPass,
    IdentityRemovalPass,
)


class GraphOptimizationPipeline:
    """
    图优化管道
    
    按顺序执行所有O0级别优化
    """
    
    def __init__(self, config: GraphOptimizationConfig = None):
        self.config = config or GraphOptimizationConfig()
        self.passes: List[BaseGraphPass] = []
        self.result = OptimizationResult()
        
        self._build_pipeline()
    
    def _build_pipeline(self):
        """构建优化管道"""
        if self.config.constant_folding:
            self.passes.append(ConstantFoldingPass())
        
        if self.config.identity_removal:
            self.passes.append(IdentityRemovalPass())
        
        if self.config.dead_code_elimination:
            self.passes.append(DeadCodeEliminationPass())
        
        if self.config.input_statication:
            self.passes.append(InputStaticationPass())
    
    def run(self, model: ModelProto, iterations: int = 100) -> ModelProto:
        """
        执行优化管道
        
        Args:
            model: 输入模型
            iterations: 迭代次数（有些优化需要多次迭代才能达到稳定）
        
        Returns:
            优化后的模型
        """
        console.print("[bold blue]开始图优化 (O0级别)[/bold blue]")
        
        # 记录初始节点数
        self.result.initial_node_count = len(model.graph.node)
        
        # 迭代优化直到收敛
        for iteration in range(iterations):
            console.print(f"\n[dim]迭代 {iteration + 1}/{iterations}[/dim]")
            
            any_modified = False
            
            for opt_pass in self.passes:
                model, modified = opt_pass.run(model)
                
                if opt_pass.change_count > 0:
                    console.print(
                        f"  [green]✓[/green] {opt_pass.name}: "
                        f"{opt_pass.change_count} 处修改"
                    )
                    self.result.add_pass(opt_pass.name, opt_pass.change_count)
                    any_modified = True
            
            if not any_modified:
                console.print("[dim]  已达收敛状态[/dim]")
                break
        
        self.result.final_node_count = len(model.graph.node)
        
        # 打印摘要
        self._print_summary()
        
        return model
    
    def _print_summary(self):
        """打印优化摘要"""
        table = Table(title="图优化结果", show_header=True)
        table.add_column("指标", style="cyan")
        table.add_column("数值", justify="right")
        
        table.add_row("初始节点数", str(self.result.initial_node_count))
        table.add_row("最终节点数", str(self.result.final_node_count))
        
        reduction = (
            (self.result.initial_node_count - self.result.final_node_count) /
            self.result.initial_node_count * 100
            if self.result.initial_node_count > 0 else 0
        )
        table.add_row("节点减少", f"{reduction:.1f}%")
        table.add_row("总修改数", str(self.result.total_changes))
        
        console.print(Panel(table, border_style="green"))
