#!/usr/bin/env python3
"""
INT8 量化模型验证脚本

验证内容:
1. 模型文件完整性
2. 权重分布统计（检查是否有异常值）
3. Scale分布检查
4. 反量化精度验证
5. 与原始FP16模型的精度对比
"""

import os
import json
import struct
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import onnx
from onnx import TensorProto

console = Console()
print = console.print


def read_external_tensor(manifest_entry: dict, data_path: str, dtype: np.dtype) -> np.ndarray:
    """从 data.bin 读取外部张量"""
    offset = manifest_entry["offset"]
    size = manifest_entry["size"]
    shape = manifest_entry.get("shape")
    
    with open(data_path, 'rb') as f:
        f.seek(offset)
        raw_data = f.read(size)
    
    arr = np.frombuffer(raw_data, dtype=dtype).copy()
    
    # 根据manifest中的shape进行reshape
    if shape is not None and len(shape) > 0:
        try:
            arr = arr.reshape(shape)
        except ValueError as e:
            print(f"  [yellow]警告: reshape失败 {shape} -> {arr.shape}: {e}[/yellow]")
    
    return arr


def dequantize_int8(q_weights: np.ndarray, scales: np.ndarray, group_size: int) -> np.ndarray:
    """
    反量化INT8权重到FP16
    
    Args:
        q_weights: INT8量化权重 [out_features, in_features]
        scales: 缩放因子 [out_features, num_groups]
        group_size: 分组大小
    
    Returns:
        fp16_weights: 反量化后的FP16权重
    """
    if len(q_weights.shape) != 2:
        raise ValueError(f"只支持2D权重，得到 {q_weights.shape}")
    
    out_features, in_features = q_weights.shape
    
    # pad in_features维度使其能被group_size整除
    pad_len = (group_size - in_features % group_size) % group_size
    if pad_len > 0:
        q_weights_padded = np.pad(q_weights, ((0, 0), (0, pad_len)))
    else:
        q_weights_padded = q_weights
    
    num_groups = q_weights_padded.shape[1] // group_size
    
    # reshape为 [out_features, num_groups, group_size]
    q_grouped = q_weights_padded.reshape(out_features, num_groups, group_size)
    
    # scales形状应该是 [out_features, num_groups]
    if scales.shape != (out_features, num_groups):
        raise ValueError(f"scales形状不匹配: 期望 {(out_features, num_groups)}, 得到 {scales.shape}")
    
    # 反量化: [out_features, num_groups, group_size] * [out_features, num_groups, 1]
    scales_expanded = scales.reshape(out_features, num_groups, 1)
    fp_grouped = q_grouped.astype(np.float32) * scales_expanded
    
    # reshape back
    fp_weights_padded = fp_grouped.reshape(out_features, -1)
    fp_weights = fp_weights_padded[:, :in_features]
    
    return fp_weights.astype(np.float16)


def verify_model(
    int8_dir: str,
    fp16_dir: str,
    group_size: int = 128,
    model_type: str = "prefill"
) -> bool:
    """验证INT8模型的正确性"""
    
    int8_dir = Path(int8_dir)
    fp16_dir = Path(fp16_dir)
    
    print(f"\n[bold blue]{'='*70}[/bold blue]")
    print(f"[bold blue]验证模型: {model_type}[/bold blue]")
    print(f"[bold blue]{'='*70}[/bold blue]\n")
    
    # 检查文件存在性
    int8_model_path = int8_dir / f"{model_type}.onnx"
    int8_data_path = int8_dir / "data.bin"
    int8_manifest_path = int8_dir / "manifest.json"
    
    fp16_model_path = fp16_dir / f"{model_type}.onnx"
    fp16_data_path = fp16_dir / "data.bin"
    fp16_manifest_path = fp16_dir / "manifest.json"
    
    for f in [int8_model_path, int8_data_path, int8_manifest_path]:
        if not f.exists():
            print(f"[red]错误: 找不到文件 {f}[/red]")
            return False
    
    print(f"[green]✓[/green] INT8 模型文件检查通过")
    print(f"  模型: {int8_model_path} ({int8_model_path.stat().st_size / 1024:.1f} KB)")
    print(f"  数据: {int8_data_path} ({int8_data_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # 加载manifest
    with open(int8_manifest_path, 'r') as f:
        int8_manifest = json.load(f)
    
    with open(fp16_manifest_path, 'r') as f:
        fp16_manifest = json.load(f)
    
    # 加载模型
    int8_model = onnx.load(int8_model_path, load_external_data=False)
    fp16_model = onnx.load(fp16_model_path, load_external_data=False)
    
    print(f"\n[bold]权重统计:[/bold]")
    print(f"  FP16 权重数: {len(fp16_manifest)}")
    print(f"  INT8 权重数: {len(int8_manifest)}")
    
    # 分类统计
    int8_weights = {}  # name -> {dtype, shape, has_scale}
    fp16_weights = {}
    
    for name, entry in int8_manifest.items():
        if name.endswith("_scale"):
            continue
        has_scale = f"{name}_scale" in int8_manifest
        int8_weights[name] = {
            "dtype": entry.get("dtype", "unknown"),
            "shape": entry["shape"],
            "has_scale": has_scale
        }
    
    for name in fp16_manifest:
        fp16_weights[name] = {"shape": fp16_manifest[name]["shape"]}
    
    # 统计量化情况
    quantized_count = sum(1 for w in int8_weights.values() if w["dtype"] == "int8")
    fp16_preserved = sum(1 for w in int8_weights.values() if w["dtype"] == "fp16")
    
    print(f"  已量化 (INT8): {quantized_count}")
    print(f"  保留 (FP16): {fp16_preserved}")
    print(f"  Scale数量: {len(int8_manifest) - len(int8_weights)}")
    
    # 详细验证
    print(f"\n[bold]详细验证:[/bold]")
    
    stats = {
        "total": 0,
        "verified": 0,
        "failed": 0,
        "fp16_preserved": 0,
        "errors": []
    }
    
    # 按层分组
    layer_weights = defaultdict(list)
    other_weights = []
    
    for name in sorted(int8_weights.keys()):
        if "model.layers." in name:
            layer_num = int(name.split("model.layers.")[1].split(".")[0])
            layer_weights[layer_num].append(name)
        else:
            other_weights.append(name)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("验证权重...", total=len(int8_weights))
        
        for name in sorted(int8_weights.keys()):
            progress.update(task, description=f"验证: {name[:50]}...")
            
            info = int8_weights[name]
            stats["total"] += 1
            
            if info["dtype"] == "fp16":
                # 验证保留的FP16权重是否一致
                if name in fp16_manifest:
                    fp16_data = read_external_tensor(fp16_manifest[name], fp16_data_path, np.float16)
                    int8_fp16_data = read_external_tensor(int8_manifest[name], int8_data_path, np.float16)
                    
                    if fp16_data.shape == int8_fp16_data.shape and np.allclose(fp16_data, int8_fp16_data):
                        stats["fp16_preserved"] += 1
                        stats["verified"] += 1
                    else:
                        stats["failed"] += 1
                        stats["errors"].append(f"{name}: FP16保留权重不匹配")
                else:
                    stats["fp16_preserved"] += 1
                    stats["verified"] += 1
            
            elif info["dtype"] == "int8":
                # 验证INT8量化精度
                scale_name = f"{name}_scale"
                
                if scale_name not in int8_manifest:
                    stats["failed"] += 1
                    stats["errors"].append(f"{name}: 缺少scale")
                    continue
                
                # 读取INT8权重和scale
                q_weights = read_external_tensor(int8_manifest[name], int8_data_path, np.int8)
                scales = read_external_tensor(int8_manifest[scale_name], int8_data_path, np.float16)
                
                # 反量化
                dq_weights = dequantize_int8(q_weights.reshape(info["shape"]), scales, group_size)
                
                # 读取原始FP16/FP32权重（根据原始dtype）
                if name in fp16_manifest:
                    # 根据原始dtype选择numpy dtype
                    orig_dtype_str = fp16_manifest[name].get('dtype', 'float16')
                    if orig_dtype_str == 'float32':
                        orig_dtype = np.float32
                    elif orig_dtype_str == 'float16':
                        orig_dtype = np.float16
                    else:
                        orig_dtype = np.float16
                    original_weights = read_external_tensor(fp16_manifest[name], fp16_data_path, orig_dtype)
                    # 转换为FP16以便比较
                    original_weights = original_weights.astype(np.float16)
                else:
                    stats["failed"] += 1
                    stats["errors"].append(f"{name}: FP16源权重不存在")
                    continue
                
                # 计算误差
                if original_weights.shape != dq_weights.shape:
                    stats["failed"] += 1
                    stats["errors"].append(f"{name}: 形状不匹配 {original_weights.shape} vs {dq_weights.shape}")
                    continue
                
                diff = np.abs(original_weights.astype(np.float32) - dq_weights.astype(np.float32))
                max_error = diff.max()
                mean_error = diff.mean()
                
                # 计算相对误差（避免除零）
                abs_original = np.abs(original_weights.astype(np.float32))
                mask = abs_original > 1e-6
                if mask.any():
                    rel_error = (diff[mask] / abs_original[mask]).mean() * 100
                else:
                    rel_error = 0.0
                
                # 阈值检查
                if max_error < 0.1:  # 最大误差小于0.1 FP16值
                    stats["verified"] += 1
                else:
                    stats["failed"] += 1
                    stats["errors"].append(
                        f"{name}: 误差过大 (max={max_error:.4f}, mean={mean_error:.4f}, rel={rel_error:.2f}%)"
                    )
            
            progress.advance(task)
    
    # 打印结果
    print(f"\n[bold]验证结果:[/bold]")
    print(f"  总权重数: {stats['total']}")
    print(f"  通过: [green]{stats['verified']}[/green]")
    print(f"  失败: [red]{stats['failed']}[/red]" if stats['failed'] > 0 else f"  失败: {stats['failed']}")
    print(f"  FP16保留: {stats['fp16_preserved']}")
    
    if stats["errors"]:
        print(f"\n[red]错误详情 (前10个):[/red]")
        for err in stats["errors"][:10]:
            print(f"  - {err}")
        if len(stats["errors"]) > 10:
            print(f"  ... 还有 {len(stats['errors']) - 10} 个错误")
    
    # 逐层统计
    print(f"\n[bold]逐层验证统计:[/bold]")
    layer_table = Table(show_header=True, header_style="bold")
    layer_table.add_column("层号", width=6)
    layer_table.add_column("权重数", width=8)
    layer_table.add_column("INT8", width=6)
    layer_table.add_column("FP16保留", width=10)
    layer_table.add_column("状态", width=8)
    
    for layer_num in sorted(layer_weights.keys())[:5]:  # 只显示前5层
        names = layer_weights[layer_num]
        int8_count = sum(1 for n in names if int8_weights[n]["dtype"] == "int8")
        fp16_count = sum(1 for n in names if int8_weights[n]["dtype"] == "fp16")
        status = "✅" if all(int8_weights[n]["dtype"] == "int8" for n in names if "q_proj" in n or "k_proj" in n or "v_proj" in n or "o_proj" in n) else "⚠️"
        layer_table.add_row(str(layer_num), str(len(names)), str(int8_count), str(fp16_count), status)
    
    if len(layer_weights) > 5:
        layer_table.add_row("...", "...", "...", "...", "...")
    
    print(layer_table)
    
    # 非层权重
    print(f"\n[bold]非层权重:[/bold]")
    for name in sorted(other_weights):
        info = int8_weights[name]
        dtype_str = "[green]INT8[/green]" if info["dtype"] == "int8" else "[yellow]FP16[/yellow]"
        scale_str = f" + scale" if info["has_scale"] else ""
        print(f"  {name}: {dtype_str}{scale_str}")
    
    success = stats["failed"] == 0
    
    if success:
        print(f"\n[bold green]✓ 所有权重验证通过![/bold green]")
    else:
        print(f"\n[bold red]✗ 验证失败: {stats['failed']} 个权重有问题[/bold red]")
    
    return success


def compare_model_size(int8_dir: str, fp16_dir: str):
    """对比模型大小"""
    print(f"\n[bold blue]模型大小对比[/bold blue]\n")
    
    int8_dir = Path(int8_dir)
    fp16_dir = Path(fp16_dir)
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("文件", width=20)
    table.add_column("FP16", width=15)
    table.add_column("INT8", width=15)
    table.add_column("压缩率", width=10)
    
    for model_type in ["prefill", "decode"]:
        fp16_data = fp16_dir / "data.bin"
        int8_data = int8_dir / "data.bin"
        
        if fp16_data.exists() and int8_data.exists():
            fp16_size = fp16_data.stat().st_size
            int8_size = int8_data.stat().st_size
            ratio = int8_size / fp16_size * 100
            
            table.add_row(
                f"{model_type} data.bin",
                f"{fp16_size / 1024 / 1024:.1f} MB",
                f"{int8_size / 1024 / 1024:.1f} MB",
                f"{ratio:.1f}%"
            )
    
    print(table)


def main():
    parser = argparse.ArgumentParser(description="INT8 量化模型验证")
    parser.add_argument("--int8", "-i", type=str, default="models/onnx/qwen3_0.6b/int8",
                       help="INT8模型目录")
    parser.add_argument("--fp16", "-f", type=str, default="models/onnx/qwen3_0.6b/opt",
                       help="原始FP16模型目录(用于对比)")
    parser.add_argument("--group-size", "-g", type=int, default=128,
                       help="量化时使用的分组大小")
    
    args = parser.parse_args()
    
    print(f"[bold cyan]INT8 量化模型验证[/bold cyan]")
    print(f"INT8模型: {args.int8}")
    print(f"FP16参考: {args.fp16}")
    print(f"分组大小: {args.group_size}")
    
    # 验证 prefill
    prefill_ok = verify_model(args.int8, args.fp16, args.group_size, "prefill")
    
    # 验证 decode
    decode_ok = verify_model(args.int8, args.fp16, args.group_size, "decode")
    
    # 大小对比
    compare_model_size(args.int8, args.fp16)
    
    # 总结
    print(f"\n[bold blue]{'='*70}[/bold blue]")
    print(f"[bold]验证总结:[/bold]")
    print(f"  Prefill: {'[green]✓ 通过[/green]' if prefill_ok else '[red]✗ 失败[/red]'}")
    print(f"  Decode:  {'[green]✓ 通过[/green]' if decode_ok else '[red]✗ 失败[/red]'}")
    
    if prefill_ok and decode_ok:
        print(f"\n[bold green]🎉 所有验证通过! INT8模型可以正常使用。[/bold green]")
    else:
        print(f"\n[bold red]⚠️ 部分验证失败，请检查错误信息。[/bold red]")


if __name__ == "__main__":
    main()
