"""
ONNX Optimizer 配置模块

定义所有配置类和枚举类型
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum


class OptimizationLevel(Enum):
    """优化等级"""
    O0 = 0  # 基础图优化（常量折叠、死代码消除等）
    O1 = 1  # 基础层融合（标准ONNX算子）
    O2 = 2  # 激进融合（ORT支持的非标准算子）
    O3 = 3  # 自定义算子（需要推理框架实现）


class QuantizationMode(Enum):
    """量化模式"""
    INT8_GROUP_MAX = "int8_group_max"      # 分组最大值量化
    INT8_DYNAMIC = "int8_dynamic"          # 动态INT8量化
    FP16 = "fp16"                          # FP16半精度
    INT4_GPTQ = "int4_gptq"                # GPTQ 4bit量化（预留）
    INT4_AWQ = "int4_awq"                  # AWQ 4bit量化（预留）


@dataclass
class ExportConfig:
    """导出配置"""
    input_dir: Path                       # 输入模型目录
    output_dir: Path                      # 输出目录
    num_heads: int                        # 注意力头数
    head_dim: int                         # 头维度
    num_layers: Optional[int] = None      # 层数（自动检测）
    max_seq_len: int = 32768              # 最大序列长度
    batch_size: int = 1                   # 批大小
    use_external_data: bool = True        # 使用外部数据存储
    external_data_threshold: int = 1024   # 外部数据阈值(MB)


@dataclass
class QuantizationConfig:
    """量化配置"""
    mode: QuantizationMode = QuantizationMode.INT8_GROUP_MAX
    group_size: int = 128                 # 分组大小
    
    # 保留精度层
    preserve_embedding: bool = True
    preserve_norm: bool = True
    preserve_lm_head: bool = True
    
    # 校准配置（静态量化用）
    calibration_data: Optional[Path] = None
    calibration_samples: int = 128
    
    # 额外参数
    extra_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphOptimizationConfig:
    """图优化配置 (O0级别)"""
    constant_folding: bool = True         # 常量折叠
    dead_code_elimination: bool = True    # 死代码消除
    input_statication: bool = True        # 输入静态化
    shape_inference: bool = True          # 形状推导
    identity_removal: bool = True         # 移除Identity节点
    unused_input_removal: bool = True     # 移除未使用的输入


@dataclass
class FusionConfig:
    """层融合配置 (O1~O3级别)"""
    level: OptimizationLevel = OptimizationLevel.O1
    config_file: Optional[Path] = None    # 自定义YAML配置文件
    
    # O1: 基础融合
    fuse_attention: bool = True           # 融合Attention子图
    fuse_ffn: bool = True                 # 融合FFN子图
    fuse_layernorm: bool = True           # 融合LayerNorm
    
    # O2: 激进融合（ORT特定）
    fuse_multi_head_attention: bool = True # 融合MHA（ORT支持）
    fuse_gelu: bool = True                # 融合GELU激活
    
    # O3: 自定义算子
    use_custom_ops: bool = False          # 使用自定义算子
    custom_op_domain: str = "onnx_opt.custom"  # 自定义算子域


@dataclass
class OptimizationPipelineConfig:
    """完整优化管道配置"""
    # 各阶段开关
    do_export: bool = True
    do_optimize: bool = True
    do_fusion: bool = True
    do_quantize: bool = False
    do_verify: bool = True
    
    # 各阶段配置
    export: ExportConfig = field(default_factory=lambda: ExportConfig(
        input_dir=Path("."),
        output_dir=Path("output"),
        num_heads=8,
        head_dim=128
    ))
    graph_opt: GraphOptimizationConfig = field(default_factory=GraphOptimizationConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    quantize: QuantizationConfig = field(default_factory=QuantizationConfig)
    
    # 日志配置
    verbose: bool = False
    log_file: Optional[Path] = None


@dataclass
class VerifyConfig:
    """验证配置"""
    reference_model: Optional[Path] = None    # 参考模型
    test_data: Optional[Path] = None          # 测试数据
    tolerance: float = 0.01                   # 误差容忍度
    check_weights: bool = True                # 检查权重
    check_structure: bool = True              # 检查结构
    check_inference: bool = False             # 检查推理
