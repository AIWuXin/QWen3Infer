#!/usr/bin/env python3
"""
分组最大值量化脚本 - 将 FP16 模型转换为 INT8

量化策略:
- 分组最大值量化 (Per-group Max Quantization)
- 默认分组大小: 128 (可配置)
- 量化公式: int8 = round(fp16 / scale * 127), scale = max(abs(group)) / 127

支持的量化粒度:
- per_channel: 每个输出通道独立量化
- per_group: 固定大小的块量化 (默认, 推荐)
"""

import os
import json
import struct
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Literal
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import onnx
from onnx import TensorProto, helper

console = Console()
print = console.print


@dataclass
class QuantConfig:
    """量化配置"""
    group_size: int = 128  # 分组大小
    quant_mode: Literal["per_channel", "per_group"] = "per_group"
    preserve_embedding: bool = True  # 保留embedding为FP16（避免精度损失）
    preserve_norm: bool = True  # 保留norm层为FP16
    preserve_lm_head: bool = True  # 保留lm_head为FP16


def read_external_tensor(manifest_entry: dict, data_path: str, dtype: np.dtype) -> np.ndarray:
    """从 data.bin 读取外部张量"""
    offset = manifest_entry["offset"]
    size = manifest_entry["size"]
    
    with open(data_path, 'rb') as f:
        f.seek(offset)
        raw_data = f.read(size)
    
    # 解析FP16数据
    return np.frombuffer(raw_data, dtype=dtype).copy()


def compute_group_scales(weights: np.ndarray, group_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算分组量化参数
    
    Args:
        weights: 原始权重 [..., K] (K是最后一个维度，通常是in_features)
        group_size: 每组元素数量
    
    Returns:
        scales: 每组缩放因子
        quantized: 量化后的INT8权重
    """
    original_shape = weights.shape
    K = original_shape[-1]
    
    # 确保K是group_size的倍数（pad如果需要）
    pad_len = (group_size - K % group_size) % group_size
    if pad_len > 0:
        weights_padded = np.pad(weights, [(0, 0)] * (len(original_shape) - 1) + [(0, pad_len)])
    else:
        weights_padded = weights
    
    # reshape成 [..., num_groups, group_size]
    new_shape = list(original_shape[:-1]) + [-1, group_size]
    weights_grouped = weights_padded.reshape(new_shape)
    
    # 计算每组的绝对最大值作为scale
    abs_max = np.abs(weights_grouped).max(axis=-1, keepdims=True)
    
    # 避免除零：将0替换为1（全零权重的scale设为1，量化后仍为0）
    abs_max_safe = np.where(abs_max == 0, 1.0, abs_max)
    scales = abs_max_safe / 127.0  # INT8 最大值
    scales = np.squeeze(scales, axis=-1)  # [..., num_groups]
    
    # 量化（使用safe的abs_max避免除零）
    quantized = np.clip(np.round(weights_grouped / (abs_max_safe / 127.0)), -127, 127).astype(np.int8)
    
    # reshape回原始形状（去除padding）
    quantized = quantized.reshape(weights_padded.shape)[..., :K]
    
    return scales.astype(np.float16), quantized.reshape(original_shape)


def compute_per_channel_scales(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    按输出通道量化 (per-channel)
    
    假设权重形状为 [out_features, in_features]
    每个输出通道独立量化
    """
    # 计算每个输出通道的绝对最大值
    abs_max = np.abs(weights).max(axis=-1, keepdims=True)
    
    # 避免除零
    abs_max_safe = np.where(abs_max == 0, 1.0, abs_max)
    scales = (abs_max_safe / 127.0).astype(np.float16)
    
    # 量化
    quantized = np.clip(np.round(weights / (abs_max_safe / 127.0)), -127, 127).astype(np.int8)
    
    return scales, quantized


def quantize_tensor(
    weights: np.ndarray,
    config: QuantConfig,
    tensor_name: str
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    量化单个张量
    
    Returns:
        quantized_weights: 量化后的INT8权重
        scales: 缩放因子（None表示未量化）
    """
    # 检查是否应该跳过量化
    if config.preserve_embedding and "embed_tokens" in tensor_name:
        print(f"  [yellow]跳过 embedding: {tensor_name}[/yellow]")
        return weights.astype(np.float16), None
    
    if config.preserve_norm and "norm" in tensor_name and "layernorm" not in tensor_name:
        print(f"  [yellow]跳过 norm: {tensor_name}[/yellow]")
        return weights.astype(np.float16), None
    
    if config.preserve_lm_head and "lm_head" in tensor_name:
        print(f"  [yellow]跳过 lm_head: {tensor_name}[/yellow]")
        return weights.astype(np.float16), None
    
    # 只有2D矩阵才量化（线性层权重）
    if len(weights.shape) != 2:
        print(f"  [dim]跳过非2D权重: {tensor_name}, shape={weights.shape}[/dim]")
        return weights.astype(np.float16), None
    
    # 执行量化
    if config.quant_mode == "per_channel":
        scales, quantized = compute_per_channel_scales(weights)
        print(f"  [green]量化 {tensor_name}: {weights.shape} -> per-channel scales {scales.shape}[/green]")
    else:  # per_group
        scales, quantized = compute_group_scales(weights, config.group_size)
        num_groups = scales.shape[-1]
        print(f"  [green]量化 {tensor_name}: {weights.shape} -> {num_groups} groups (group_size={config.group_size})[/green]")
    
    return quantized, scales


def create_quantized_onnx(
    original_model: onnx.ModelProto,
    quantized_weights: Dict[str, np.ndarray],
    scales_map: Dict[str, np.ndarray],
    config: QuantConfig
) -> onnx.ModelProto:
    """创建量化后的ONNX模型"""
    
    # 复制模型结构
    new_model = onnx.ModelProto()
    new_model.CopyFrom(original_model)
    new_model.graph.ClearField('initializer')
    
    # 添加量化后的权重
    for init in original_model.graph.initializer:
        name = init.name
        
        if name in quantized_weights:
            q_weights = quantized_weights[name]
            
            if q_weights.dtype == np.int8:
                # INT8 权重
                tensor = helper.make_tensor(
                    name=name,
                    data_type=TensorProto.INT8,
                    dims=list(init.dims),
                    vals=q_weights.tobytes(),
                    raw=True
                )
            else:
                # FP16 权重（未量化的）
                tensor = helper.make_tensor(
                    name=name,
                    data_type=TensorProto.FLOAT16,
                    dims=list(init.dims),
                    vals=q_weights.astype(np.float16).tobytes(),
                    raw=True
                )
            
            new_model.graph.initializer.append(tensor)
            
            # 如果有scale，添加scale张量
            if name in scales_map and scales_map[name] is not None:
                scales = scales_map[name]
                scale_name = f"{name}_scale"
                scale_tensor = helper.make_tensor(
                    name=scale_name,
                    data_type=TensorProto.FLOAT16,
                    dims=list(scales.shape),
                    vals=scales.astype(np.float16).tobytes(),
                    raw=True
                )
                new_model.graph.initializer.append(scale_tensor)
        else:
            # 保持原样
            new_init = new_model.graph.initializer.add()
            new_init.CopyFrom(init)
    
    return new_model


def save_external_data(
    quantized_weights: Dict[str, np.ndarray],
    scales_map: Dict[str, np.ndarray],
    output_data_path: str,
    output_manifest_path: str
):
    """保存量化后的权重到外部数据文件"""
    
    manifest = {}
    offset = 0
    
    with open(output_data_path, 'wb') as f:
        for name in sorted(quantized_weights.keys()):
            weights = quantized_weights[name]
            
            # 写入权重
            data = weights.tobytes()
            f.write(data)
            
            manifest[name] = {
                "offset": offset,
                "size": len(data),
                "dtype": "int8" if weights.dtype == np.int8 else "fp16",
                "shape": list(weights.shape)
            }
            offset += len(data)
            
            # 写入scale（如果有）
            if name in scales_map and scales_map[name] is not None:
                scales = scales_map[name]
                scale_name = f"{name}_scale"
                scale_data = scales.astype(np.float16).tobytes()
                f.write(scale_data)
                
                manifest[scale_name] = {
                    "offset": offset,
                    "size": len(scale_data),
                    "dtype": "fp16",
                    "shape": list(scales.shape)
                }
                offset += len(scale_data)
    
    # 保存manifest
    with open(output_manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return manifest


def quantize_model(
    input_dir: str,
    output_dir: str,
    config: QuantConfig,
    model_type: str = "prefill"  # "prefill" or "decode"
):
    """量化单个模型文件"""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = input_dir / f"{model_type}.onnx"
    data_path = input_dir / "data.bin"
    manifest_path = input_dir / "manifest.json"
    
    print(f"\n[bold blue]{'='*60}[/bold blue]")
    print(f"[bold blue]量化模型: {model_type}[/bold blue]")
    print(f"[bold blue]{'='*60}[/bold blue]\n")
    
    # 加载模型
    print(f"[dim]加载模型: {model_path}[/dim]")
    model = onnx.load(model_path, load_external_data=False)
    
    # 加载manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print(f"权重总数: {len(model.graph.initializer)}")
    print(f"量化模式: {config.quant_mode}")
    if config.quant_mode == "per_group":
        print(f"分组大小: {config.group_size}")
    print()
    
    # 量化每个权重
    quantized_weights = {}
    scales_map = {}
    
    dtype_map = {
        TensorProto.FLOAT16: np.float16,
        TensorProto.FLOAT: np.float32,
    }
    
    total_size_before = 0
    total_size_after = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("量化权重...", total=len(model.graph.initializer))
        
        for init in model.graph.initializer:
            name = init.name
            progress.update(task, description=f"处理: {name[:50]}...")
            
            # 计算原始大小
            tensor_size = 1
            for dim in init.dims:
                tensor_size *= dim
            
            dtype = dtype_map.get(init.data_type, np.float16)
            size_before = tensor_size * 2  # FP16 = 2 bytes
            total_size_before += size_before
            
            # 读取权重数据
            if name in manifest:
                # 从外部数据读取
                weights = read_external_tensor(manifest[name], data_path, dtype)
                weights = weights.reshape(list(init.dims))
            elif len(init.raw_data) > 0:
                # 从内部数据读取
                weights = np.frombuffer(init.raw_data, dtype=dtype).copy()
                weights = weights.reshape(list(init.dims))
            else:
                print(f"  [red]警告: 无法读取 {name}[/red]")
                progress.advance(task)
                continue
            
            # 量化
            q_weights, scales = quantize_tensor(weights, config, name)
            quantized_weights[name] = q_weights
            scales_map[name] = scales
            
            # 计算量化后大小
            if scales is not None:
                # INT8 权重 + FP16 scales
                size_after = q_weights.nbytes + scales.nbytes
            else:
                # 未量化的FP16
                size_after = q_weights.nbytes
            total_size_after += size_after
            
            progress.advance(task)
    
    # 保存量化后的模型
    print(f"\n[dim]保存量化模型...[/dim]")
    
    output_model_path = output_dir / f"{model_type}.onnx"
    output_data_path = output_dir / "data.bin"
    output_manifest_path = output_dir / "manifest.json"
    
    # 创建新的ONNX模型（结构保持不变）
    new_model = create_quantized_onnx(model, quantized_weights, scales_map, config)
    
    # 保存外部数据
    manifest = save_external_data(quantized_weights, scales_map, output_data_path, output_manifest_path)
    
    # 修改模型以引用外部数据
    for init in new_model.graph.initializer:
        name = init.name
        if name in manifest:
            # 设置外部数据位置
            init.ClearField('raw_data')
            
            # 添加外部数据信息
            ext_data = init.external_data.add()
            ext_data.key = "location"
            ext_data.value = "data.bin"
            
            ext_data_offset = init.external_data.add()
            ext_data_offset.key = "offset"
            ext_data_offset.value = str(manifest[name]["offset"])
            
            ext_data_length = init.external_data.add()
            ext_data_length.key = "length"
            ext_data_length.value = str(manifest[name]["size"])
            
            init.data_location = TensorProto.EXTERNAL
    
    # 保存ONNX模型
    onnx.save(new_model, output_model_path)
    
    # 打印统计信息
    print(f"\n[bold green]✓ 量化完成![/bold green]")
    print(f"\n输出路径: {output_dir}")
    print(f"  模型文件: {output_model_path.name} ({output_model_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  数据文件: {output_data_path.name} ({output_data_path.stat().st_size / 1024 / 1024:.2f} MB)")
    
    print(f"\n[bold]压缩统计:[/bold]")
    print(f"  原始大小: {total_size_before / 1024 / 1024:.2f} MB")
    print(f"  量化后大小: {total_size_after / 1024 / 1024:.2f} MB")
    print(f"  压缩率: {total_size_after / total_size_before * 100:.1f}%")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="FP16 到 INT8 分组最大值量化")
    parser.add_argument("--input", "-i", type=str, default="models/onnx/qwen3_0.6b/opt",
                       help="输入模型目录 (包含 prefill.onnx/decode.onnx/data.bin)")
    parser.add_argument("--output", "-o", type=str, default="models/onnx/qwen3_0.6b/int8",
                       help="输出模型目录")
    parser.add_argument("--group-size", "-g", type=int, default=128,
                       help="分组大小 (默认: 128)")
    parser.add_argument("--mode", "-m", type=str, default="per_group",
                       choices=["per_group", "per_channel"],
                       help="量化模式 (默认: per_group)")
    parser.add_argument("--quant-embedding", action="store_true",
                       help="是否量化 embedding 层 (默认保留FP16)")
    parser.add_argument("--quant-norm", action="store_true",
                       help="是否量化 norm 层 (默认保留FP16)")
    parser.add_argument("--quant-lm-head", action="store_true",
                       help="是否量化 lm_head 层 (默认保留FP16)")
    
    args = parser.parse_args()
    
    # 创建配置
    config = QuantConfig(
        group_size=args.group_size,
        quant_mode=args.mode,
        preserve_embedding=not args.quant_embedding,
        preserve_norm=not args.quant_norm,
        preserve_lm_head=not args.quant_lm_head
    )
    
    print(f"[bold cyan]FP16 -> INT8 分组最大值量化[/bold cyan]")
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    
    # 量化 prefill 和 decode
    input_path = Path(args.input)
    
    if (input_path / "prefill.onnx").exists():
        quantize_model(args.input, args.output, config, "prefill")
    else:
        print(f"[red]错误: 找不到 prefill.onnx[/red]")
    
    if (input_path / "decode.onnx").exists():
        quantize_model(args.input, args.output, config, "decode")
    else:
        print(f"[yellow]警告: 找不到 decode.onnx[/yellow]")
    
    print(f"\n[bold green]🎉 全部量化完成![/bold green]")


if __name__ == "__main__":
    main()
