"""
ONNX Optimizer 工具函数
"""

import json
import struct
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import asdict

import numpy as np
import onnx
from onnx import TensorProto, ModelProto
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def load_external_tensor(
    manifest_entry: Dict[str, Any],
    data_path: Path,
    dtype: Optional[np.dtype] = None
) -> np.ndarray:
    """
    从外部数据文件加载张量
    
    Args:
        manifest_entry: manifest中的条目
        data_path: data.bin文件路径
        dtype: 数据类型，None时自动推断
    
    Returns:
        加载的numpy数组
    """
    offset = manifest_entry["offset"]
    size = manifest_entry["size"]
    shape = manifest_entry.get("shape", [])
    entry_dtype = manifest_entry.get("dtype", "fp16")
    
    # 自动推断dtype
    if dtype is None:
        dtype_map = {
            "fp16": np.float16,
            "float16": np.float16,
            "fp32": np.float32,
            "float32": np.float32,
            "int8": np.int8,
            "int32": np.int32,
            "int64": np.int64,
        }
        dtype = dtype_map.get(entry_dtype, np.float16)
    
    with open(data_path, 'rb') as f:
        f.seek(offset)
        raw_data = f.read(size)
    
    arr = np.frombuffer(raw_data, dtype=dtype).copy()
    
    # reshape
    if shape:
        try:
            arr = arr.reshape(shape)
        except ValueError:
            pass  # reshape失败时保持一维
    
    return arr


def save_external_tensor(
    tensor: np.ndarray,
    name: str,
    data_path: Path,
    manifest: Dict[str, Any],
    offset: int
) -> int:
    """
    保存张量到外部数据文件
    
    Args:
        tensor: 要保存的numpy数组
        name: 张量名称
        data_path: data.bin文件路径
        manifest: manifest字典
        offset: 写入偏移量
    
    Returns:
        新的偏移量
    """
    dtype_map = {
        np.float16: "fp16",
        np.float32: "fp32",
        np.int8: "int8",
        np.int32: "int32",
        np.int64: "int64",
    }
    
    data = tensor.tobytes()
    
    # 追加写入
    mode = 'ab' if data_path.exists() else 'wb'
    with open(data_path, mode) as f:
        f.write(data)
    
    # 更新manifest
    manifest[name] = {
        "offset": offset,
        "size": len(data),
        "dtype": dtype_map.get(tensor.dtype.type, "unknown"),
        "shape": list(tensor.shape)
    }
    
    return offset + len(data)


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """加载manifest文件"""
    with open(manifest_path, 'r') as f:
        return json.load(f)


def save_manifest(manifest: Dict[str, Any], manifest_path: Path):
    """保存manifest文件"""
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def get_tensor_dtype_size(tensor_type: int) -> int:
    """获取ONNX张量类型的字节大小"""
    dtype_sizes = {
        TensorProto.FLOAT: 4,
        TensorProto.FLOAT16: 2,
        TensorProto.DOUBLE: 8,
        TensorProto.INT8: 1,
        TensorProto.INT16: 2,
        TensorProto.INT32: 4,
        TensorProto.INT64: 8,
        TensorProto.UINT8: 1,
        TensorProto.UINT16: 2,
        TensorProto.UINT32: 4,
        TensorProto.UINT64: 8,
        TensorProto.BOOL: 1,
    }
    return dtype_sizes.get(tensor_type, 4)


def format_size(size_bytes: int) -> str:
    """格式化字节大小为人类可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def copy_model_structure(model: ModelProto) -> ModelProto:
    """深度复制模型结构（不包含权重数据）"""
    new_model = ModelProto()
    new_model.CopyFrom(model)
    return new_model


class ProgressTracker:
    """进度追踪器"""
    
    def __init__(self, description: str, total: Optional[int] = None):
        self.description = description
        self.total = total
        self.progress = None
        self.task = None
    
    def __enter__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        )
        self.progress.__enter__()
        self.task = self.progress.add_task(self.description, total=self.total)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.__exit__(exc_type, exc_val, exc_tb)
    
    def update(self, advance: int = 1, description: Optional[str] = None):
        """更新进度"""
        if self.task is not None:
            self.progress.update(
                self.task,
                advance=advance,
                description=description or self.description
            )
    
    def set_description(self, description: str):
        """设置描述"""
        self.description = description
        if self.task is not None:
            self.progress.update(self.task, description=description)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并配置字典"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result
