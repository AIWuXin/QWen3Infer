#!/usr/bin/env python3
"""
ONNX Optimizer CLI - 通用大模型ONNX优化工具

完整功能:
- export: 模型导出与拆分
- optimize: 图优化 (O0)
- fuse: 层融合 (O1~O3)
- quantize: 模型量化
- verify: 模型验证
- analyze: 模型分析
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich import box

from onnx_optimizer.config import (
    ExportConfig, QuantizationConfig, GraphOptimizationConfig,
    FusionConfig, VerifyConfig, OptimizationLevel, QuantizationMode,
)
from onnx_optimizer.exporter import SplitExporter
from onnx_optimizer.graph_optimizer import GraphOptimizationPipeline
from onnx_optimizer.fusion import FusionPassManager
from onnx_optimizer.quantizer import QuantizerRegistry
from onnx_optimizer.verifier import ModelVerifier


console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="onnx-opt")
@click.option('--verbose', '-v', is_flag=True, help='启用详细输出')
@click.pass_context
def cli(ctx: click.Context, verbose: bool):
    """
    [bold blue]ONNX Optimizer[/bold blue] - 大模型ONNX优化工具
    
    支持Qwen3、Llama等主流大语言模型的ONNX转换与优化。
    
    [dim]示例:[/dim]
        onnx-opt export model_dir/ -o output/ --num-heads 8 --head-dim 128
        onnx-opt optimize model.onnx -o optimized.onnx
        onnx-opt quantize model.onnx --mode int8_group_max
        onnx-opt fuse model.onnx --level O2
        onnx-opt verify model_int8.onnx --reference model_fp16.onnx
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--output', '-o', 'output_dir', type=click.Path(path_type=Path),
              required=True, help='输出目录')
@click.option('--num-heads', type=int, help='注意力头数（自动检测）')
@click.option('--head-dim', type=int, help='头维度（自动检测）')
@click.option('--num-layers', type=int, help='层数（自动检测）')
@click.option('--batch-size', default=1, help='批大小 (默认: 1)')
@click.option('--max-seq-len', default=32768, help='最大序列长度 (默认: 32768)')
@click.pass_context
def export(
    ctx: click.Context,
    input_dir: Path,
    output_dir: Path,
    num_heads: Optional[int],
    head_dim: Optional[int],
    num_layers: Optional[int],
    batch_size: int,
    max_seq_len: int
):
    """
    [bold green]导出并拆分模型[/bold green]
    
    将optimum-cli导出的模型拆分为prefill/decode格式。
    模型参数（num-heads, head-dim等）会自动检测，无需手动指定。
    
    [dim]参数:[/dim]
        INPUT_DIR: optimum-cli导出的模型目录
    
    [dim]示例:[/dim]
        onnx-opt export models/qwen3_0.6b/ -o output/
        onnx-opt export models/qwen3_0.6b/ -o output/ --num-heads 8  # 强制指定
    """
    # 自动检测模型参数
    from .analyzer import auto_detect_config
    
    config_path = input_dir / "config.json"
    if not config_path.exists():
        console.print(f"[bold red]✗ 错误:[/bold red] 找不到模型配置文件: {config_path}")
        sys.exit(1)
    
    # 自动检测或使用用户指定的参数
    if num_heads is None or head_dim is None or num_layers is None:
        detected = auto_detect_config(config_path)
        
        if num_heads is None:
            num_heads = detected.num_heads
            console.print(f"[dim]自动检测 num_heads: {num_heads}[/dim]")
        
        if head_dim is None:
            head_dim = detected.head_dim
            console.print(f"[dim]自动检测 head_dim: {head_dim}[/dim]")
        
        if num_layers is None:
            num_layers = detected.num_layers
            console.print(f"[dim]自动检测 num_layers: {num_layers}[/dim]")
    
    config = ExportConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        batch_size=batch_size,
        max_seq_len=max_seq_len
    )
    
    exporter = SplitExporter(config)
    
    try:
        result_dir = exporter.export()
        console.print(f"\n[bold green]✓ 导出完成:[/bold green] {result_dir}")
    except Exception as e:
        console.print(f"[bold red]✗ 导出失败:[/bold red] {e}")
        if ctx.obj.get('verbose'):
            raise
        sys.exit(1)


@cli.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path),
              required=True, help='输出路径')
@click.option('--no-constant-folding', is_flag=True, help='禁用常量折叠')
@click.option('--no-dead-code-elim', is_flag=True, help='禁用死代码消除')
@click.option('--no-identity-removal', is_flag=True, help='禁用Identity移除')
@click.option('--batch-size', type=int, help='固定batch size')
@click.option('--seq-len', type=int, help='固定序列长度')
@click.pass_context
def optimize(
    ctx: click.Context,
    input_path: Path,
    output: Path,
    no_constant_folding: bool,
    no_dead_code_elim: bool,
    no_identity_removal: bool,
    batch_size: Optional[int],
    seq_len: Optional[int]
):
    """
    [bold cyan]图优化 (O0)[/bold cyan]
    
    执行基础图优化：常量折叠、死代码消除、输入静态化等。
    
    [dim]示例:[/dim]
        onnx-opt optimize model.onnx -o optimized.onnx
        onnx-opt optimize model.onnx -o opt.onnx --batch-size 1 --seq-len 128
    """
    import onnx
    
    config = GraphOptimizationConfig(
        constant_folding=not no_constant_folding,
        dead_code_elimination=not no_dead_code_elim,
        input_statication=batch_size is not None or seq_len is not None,
        identity_removal=not no_identity_removal
    )
    
    try:
        # 加载模型
        with console.status("[dim]加载模型...[/dim]"):
            model = onnx.load(input_path, load_external_data=False)
        
        # 执行优化
        pipeline = GraphOptimizationPipeline(config)
        
        # 如果指定了静态化参数，创建静态化pass
        if config.input_statication:
            from .graph_optimizer.passes import InputStaticationPass
            static_pass = InputStaticationPass(
                batch_size=batch_size or 1,
                seq_len=seq_len
            )
            pipeline.passes.append(static_pass)
        
        model = pipeline.run(model)
        
        # 保存
        with console.status(f"[dim]保存到 {output}...[/dim]"):
            onnx.save(model, output)
        
        console.print(f"\n[bold green]✓ 优化完成:[/bold green] {output}")
        
    except Exception as e:
        console.print(f"[bold red]✗ 优化失败:[/bold red] {e}")
        if ctx.obj.get('verbose'):
            raise
        sys.exit(1)


@cli.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='输出目录 (默认: 输入目录)')
@click.option('--mode', '-m', 
              type=click.Choice(['int8_group_max', 'int8_dynamic', 'fp16']),
              default='int8_group_max', help='量化模式')
@click.option('--group-size', '-g', default=128, help='分组大小 (默认: 128)')
@click.option('--preserve-embedding/--no-preserve-embedding', default=True,
              help='保留Embedding层')
@click.option('--preserve-norm/--no-preserve-norm', default=True,
              help='保留Norm层')
@click.option('--preserve-lm-head/--no-preserve-lm-head', default=True,
              help='保留LM Head层')
@click.pass_context
def quantize(
    ctx: click.Context,
    input_path: Path,
    output: Optional[Path],
    mode: str,
    group_size: int,
    preserve_embedding: bool,
    preserve_norm: bool,
    preserve_lm_head: bool
):
    """
    [bold magenta]模型量化[/bold magenta]
    
    将模型量化为INT8或FP16格式。
    
    [dim]示例:[/dim]
        onnx-opt quantize model.onnx --mode int8_group_max
        onnx-opt quantize model.onnx --mode int8_group_max -g 64
        onnx-opt quantize model.onnx --mode fp16
    """
    import onnx
    
    # 确定输出目录
    if output is None:
        output = input_path.parent / f"{input_path.stem}_quantized"
    output.mkdir(parents=True, exist_ok=True)
    
    # 构建配置
    config = QuantizationConfig(
        mode=QuantizationMode(mode),
        group_size=group_size,
        preserve_embedding=preserve_embedding,
        preserve_norm=preserve_norm,
        preserve_lm_head=preserve_lm_head
    )
    
    try:
        # 加载模型
        with console.status("[dim]加载模型...[/dim]"):
            model = onnx.load(input_path, load_external_data=False)
        
        # 查找manifest和data
        manifest_path = input_path.parent / "manifest.json"
        data_path = input_path.parent / "data.bin"
        
        if manifest_path.exists():
            from .utils import load_manifest
            manifest = load_manifest(manifest_path)
        else:
            manifest = {}
        
        # 获取量化器
        quantizer = QuantizerRegistry.get_quantizer(config)
        
        console.print(f"[dim]使用量化器: {quantizer.executor.name}[/dim]\n")
        
        # 执行量化
        model, new_manifest, new_data_path = quantizer.quantize(
            model, manifest, data_path, output
        )
        
        # 保存模型
        onnx.save(model, output / f"{input_path.stem}.onnx")
        
        console.print(f"\n[bold green]✓ 量化完成:[/bold green] {output}")
        
    except Exception as e:
        console.print(f"[bold red]✗ 量化失败:[/bold red] {e}")
        if ctx.obj.get('verbose'):
            raise
        sys.exit(1)


@cli.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--level', type=click.Choice(['O1', 'O2', 'O3']),
              default='O1', help='融合级别 (默认: O1)')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='输出路径')
@click.option('--config', type=click.Path(path_type=Path),
              help='自定义YAML配置文件')
@click.pass_context
def fuse(
    ctx: click.Context,
    input_path: Path,
    level: str,
    output: Optional[Path],
    config: Optional[Path]
):
    """
    [bold yellow]层融合 (O1~O3)[/bold yellow]
    
    根据配置执行层融合优化。
    
    [dim]级别说明:[/dim]
        O1: 基础融合（标准ONNX算子）
        O2: 激进融合（ORT特定算子）
        O3: 自定义算子（需推理框架支持）
    
    [dim]示例:[/dim]
        onnx-opt fuse model.onnx --level O1
        onnx-opt fuse model.onnx --level O2 -o fused.onnx
        onnx-opt fuse model.onnx --config custom_fusion.yaml
    """
    level_map = {'O1': 1, 'O2': 2, 'O3': 3}
    
    fusion_config = FusionConfig(
        level=OptimizationLevel(level_map[level]),
        config_file=config
    )
    
    try:
        manager = FusionPassManager(fusion_config)
        manager.load_passes()
        
        console.print(f"[yellow]⚠ 层融合功能尚未完全实现[/yellow]")
        console.print(f"[dim]配置级别: {level}[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]✗ 融合失败:[/bold red] {e}")
        if ctx.obj.get('verbose'):
            raise
        sys.exit(1)


@cli.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.option('--reference', '-r', type=click.Path(exists=True, path_type=Path),
              help='参考模型路径')
@click.option('--tolerance', default=0.01, help='误差容忍度 (默认: 0.01)')
@click.option('--no-structure-check', is_flag=True, help='跳过结构检查')
@click.option('--no-weight-check', is_flag=True, help='跳过权重检查')
@click.pass_context
def verify(
    ctx: click.Context,
    model_path: Path,
    reference: Optional[Path],
    tolerance: float,
    no_structure_check: bool,
    no_weight_check: bool
):
    """
    [bold white]验证模型[/bold white]
    
    验证ONNX模型的完整性和正确性。
    
    [dim]示例:[/dim]
        onnx-opt verify model.onnx
        onnx-opt verify model_int8.onnx -r model_fp16.onnx
    """
    config = VerifyConfig(
        reference_model=reference,
        tolerance=tolerance,
        check_structure=not no_structure_check,
        check_weights=not no_weight_check
    )
    
    verifier = ModelVerifier(config)
    
    try:
        success = verifier.verify(model_path)
        sys.exit(0 if success else 1)
    except Exception as e:
        console.print(f"[bold red]✗ 验证失败:[/bold red] {e}")
        if ctx.obj.get('verbose'):
            raise
        sys.exit(1)


@cli.command()
def list_quantizers():
    """
    [bold blue]列出可用的量化器[/bold blue]
    """
    from rich.table import Table
    
    table = Table(title="可用量化器", show_header=True)
    table.add_column("模式", style="cyan")
    table.add_column("名称", style="green")
    table.add_column("支持类型", style="dim")
    
    quantizers = QuantizerRegistry.list_quantizers()
    
    for mode, name in quantizers.items():
        executor_class = QuantizerRegistry.get_executor(mode)
        dtypes = ", ".join(executor_class().supported_dtypes)
        table.add_row(mode.value, name, dtypes)
    
    console.print(table)


# 错误处理
def main():
    """主入口"""
    try:
        cli()
    except Exception as e:
        console.print(f"[bold red]错误:[/bold red] {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
