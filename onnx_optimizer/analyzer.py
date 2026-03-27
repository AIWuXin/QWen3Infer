"""
模型分析器

自动检测LLM模型的结构参数
"""

import re
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from dataclasses import dataclass

import onnx
import numpy as np
from rich.table import Table
from rich.panel import Panel
import json

from .utils import console


@dataclass
class ModelConfig:
    """检测到的模型配置"""
    hidden_size: int
    num_heads: int
    head_dim: int
    num_layers: int
    intermediate_size: Optional[int] = None
    vocab_size: Optional[int] = None
    model_type: str = "unknown"  # qwen3, llama, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'num_layers': self.num_layers,
            'intermediate_size': self.intermediate_size,
            'vocab_size': self.vocab_size,
            'model_type': self.model_type,
        }


class ModelAnalyzer:
    """
    ONNX模型分析器
    
    通过分析模型结构和权重，自动推断LLM配置参数
    """
    
    def __init__(self, config_path: Optional[Path] = None, model: Optional[onnx.ModelProto] = None):
        """
        Args:
            model_path: 模型文件路径
            model: 已加载的ONNX模型
        """
        self.model = None
        self.config = None
        
        if model is not None:
            self.model = model
        elif config_path is not None:
           with open(config_path) as f:
               self.config = json.load(f)
        else:
            raise ValueError("必须提供model_path或model")
        
        if self.model is not None:
            self.initializers = {init.name: init for init in self.model.graph.initializer}
    
    def analyze(self) -> ModelConfig:
        """
        分析模型并返回配置
        
        Returns:
            ModelConfig
        """
        console.print("[dim]分析模型结构...[/dim]")
        
        # 1. 检测hidden_size
        hidden_size = self._detect_hidden_size()
        
        # 2. 检测num_heads和head_dim
        num_heads, head_dim = self._detect_attention_config(hidden_size)
        
        # 3. 检测num_layers
        num_layers = self._detect_num_layers()
        
        # 4. 检测intermediate_size
        intermediate_size = self._detect_intermediate_size()
        
        # 5. 检测vocab_size
        vocab_size = self._detect_vocab_size()
        
        # 6. 推断模型类型
        model_type = self._detect_model_type()
        
        config = ModelConfig(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            model_type=model_type
        )
        
        self._print_config(config)
        
        return config
    
    def _detect_hidden_size(self) -> int:
        """
        检测hidden_size
        
        策略：查找embedding权重或layernorm权重的形状
        """
        if self.config is not None:
            return self.config["hidden_size"]
        
        # 1. 尝试从embedding权重检测
        for name, init in self.initializers.items():
            # 查找embed_tokens或word_embeddings
            if 'embed' in name.lower() or 'token' in name.lower():
                shape = list(init.dims)
                if len(shape) == 2:
                    # [vocab_size, hidden_size] 或 [hidden_size, vocab_size]
                    # 通常vocab_size > hidden_size
                    return min(shape[0], shape[1])
        
        # 2. 尝试从layernorm/rmsnorm权重检测
        for name, init in self.initializers.items():
            if any(x in name.lower() for x in ['norm', 'ln', 'rms']):
                shape = list(init.dims)
                if len(shape) == 1:
                    return shape[0]
        
        # 3. 尝试从Q/K/V投影权重检测
        for name, init in self.initializers.items():
            if any(x in name.lower() for x in ['q_proj', 'query']):
                shape = list(init.dims)
                if len(shape) == 2:
                    # 通常是 [hidden_size, hidden_size] 或 [num_heads*head_dim, hidden_size]
                    return shape[1]  # 输入维度
        
        # 4. 尝试从MLP权重检测
        for name, init in self.initializers.items():
            if any(x in name.lower() for x in ['gate_proj', 'up_proj', 'fc1']):
                shape = list(init.dims)
                if len(shape) == 2:
                    return shape[1]  # 输入维度
        
        raise ValueError("无法自动检测hidden_size，请手动指定")
    
    def _detect_attention_config(self, hidden_size: int) -> Tuple[int, int]:
        """
        检测注意力配置
        
        Returns:
            (num_heads, head_dim)
        """
        
        if self.config is not None:
            return self.config["num_attention_heads"], self.config["head_dim"]
        
        # 策略：分析Q/K/V投影权重的形状
        q_proj_shapes = []
        k_proj_shapes = []
        
        for name, init in self.initializers.items():
            lower_name = name.lower()
            shape = list(init.dims)
            
            if len(shape) != 2:
                continue
            
            # Q投影
            if any(x in lower_name for x in ['q_proj', 'query', 'q_weight']):
                q_proj_shapes.append(shape)
            # K投影（用于GQA/MQA检测）
            elif any(x in lower_name for x in ['k_proj', 'key', 'k_weight']):
                k_proj_shapes.append(shape)
        
        if not q_proj_shapes:
            raise ValueError("无法找到Q投影权重，请手动指定num_heads")
        
        # 分析Q投影形状 [out_dim, in_dim]
        # out_dim = num_heads * head_dim
        q_out_dim = q_proj_shapes[0][0]
        
        # 尝试从K投影推断num_kv_heads（GQA）
        if k_proj_shapes:
            k_out_dim = k_proj_shapes[0][0]
            # 如果K的维度是Q的一半，可能是GQA with num_kv_heads = num_heads // 2
            if k_out_dim * 2 == q_out_dim:
                # 可能是MHA，也可能是GQA
                pass
        
        # 常见配置映射
        common_configs = {
            # hidden_size -> (num_heads, head_dim)
            1024: (16, 64),    # 小型模型 (e.g., qwen3-0.6b)
            1536: (12, 128),   # 可能的配置
            2048: (32, 64),    # 中型模型
            2560: (32, 80),    # 某些模型
            3072: (24, 128),   # Llama 7B style
            3584: (28, 128),   # Qwen 7B
            4096: (32, 128),   # 标准7B/8B
            5120: (40, 128),   # 13B
            6144: (48, 128),   # 某些模型
            8192: (64, 128),   # 70B
            16384: (128, 128), # 大模型
        }
        
        if hidden_size in common_configs:
            return common_configs[hidden_size]
        
        # 推断head_dim通常是64或128
        if q_out_dim % 128 == 0:
            head_dim = 128
        elif q_out_dim % 64 == 0:
            head_dim = 64
        else:
            # 尝试找到能整除的head_dim
            for dim in [128, 64, 96, 80]:
                if q_out_dim % dim == 0:
                    head_dim = dim
                    break
            else:
                head_dim = 128  # 默认值
        
        num_heads = q_out_dim // head_dim
        
        return num_heads, head_dim
    
    def _detect_num_layers(self) -> int:
        """
        检测层数
        
        策略：从层命名模式计数，如 model.layers.0, model.layers.1, ...
        """
        
        if self.config is not None:
            return self.config["num_hidden_layers"]
        
        layer_indices = set()
        
        # 常见的层命名模式
        patterns = [
            r'layers?\.(\d+)',  # layers.0, layer_1, etc.
            r'encoder\.layer\.(\d+)',
            r'decoder\.layer\.(\d+)',
            r'transformer\.h\.(\d+)',
            r'model\.layers\.(\d+)',
        ]
        
        for name in self.initializers.keys():
            for pattern in patterns:
                match = re.search(pattern, name)
                if match:
                    layer_idx = int(match.group(1))
                    layer_indices.add(layer_idx)
        
        if layer_indices:
            return max(layer_indices) + 1  # 层数 = 最大索引 + 1
        
        # 如果没有找到层索引，尝试其他策略
        # 计算self-attention模块的数量
        attn_count = 0
        for name in self.initializers.keys():
            if any(x in name.lower() for x in ['q_proj', 'query', 'self_attn']):
                attn_count += 1
        
        # 每个层通常有1个self-attention
        # 但有些模型可能有多个（如cross-attention）
        if attn_count > 0:
            # 假设每层1个self-attention
            return attn_count
        
        return 1  # 默认1层
    
    def _detect_intermediate_size(self) -> Optional[int]:
        """
        检测FFN中间层维度
        
        策略：分析gate_proj或up_proj的权重形状
        """
        
        if self.config is not None:
            return self.config["intermediate_size"]
        
        for name, init in self.initializers.items():
            lower_name = name.lower()
            shape = list(init.dims)
            
            if len(shape) != 2:
                continue
            
            # gate_proj通常形状为 [intermediate_size, hidden_size]
            if any(x in lower_name for x in ['gate_proj', 'w1', 'fc1']):
                return shape[0]
            # up_proj
            elif any(x in lower_name for x in ['up_proj', 'w3']):
                return shape[0]
        
        return None
    
    def _detect_vocab_size(self) -> Optional[int]:
        """
        检测词表大小
        
        策略：从embedding权重或lm_head权重检测
        """
        
        if self.config is not None:
            return self.config["vocab_size"]
        
        for name, init in self.initializers.items():
            lower_name = name.lower()
            shape = list(init.dims)
            
            if len(shape) != 2:
                continue
            
            if any(x in lower_name for x in ['embed', 'token', 'word']):
                # 假设vocab_size > hidden_size
                return max(shape[0], shape[1])
            elif any(x in lower_name for x in ['lm_head', 'output', 'classifier']):
                return shape[0]
        
        return None
    
    def _detect_model_type(self) -> str:
        """
        检测模型类型
        
        根据权重命名模式推断
        """
        
        if self.config is not None:
            return self.config["model_type"]
        
        names = [n.lower() for n in self.initializers.keys()]
        
        # Qwen特征
        if any('qwen' in n for n in names):
            return 'qwen'
        if any('rmsnorm' in n or 'rms_norm' in n for n in names):
            if any('q_proj' in n for n in names):
                return 'qwen2/3'  # 有QKV投影的RMSNorm模型
        
        # Llama特征
        if any('llama' in n for n in names):
            return 'llama'
        if any('rotary' in n or 'rope' in n for n in names):
            return 'llama/rope'
        
        # GPT特征
        if any('gpt' in n for n in names):
            return 'gpt'
        if any('attn.c_attn' in n for n in names):
            return 'gpt2'
        
        # BERT特征
        if any('bert' in n for n in names):
            return 'bert'
        
        # 通用检测
        if any('layer_norm' in n for n in names):
            if any('q_proj' in n for n in names):
                return 'transformer'
            return 'bert-like'
        
        return 'unknown'
    
    def _print_config(self, config: ModelConfig):
        """打印检测到的配置"""
        table = Table(title="检测到的模型配置", show_header=True)
        table.add_column("参数", style="cyan")
        table.add_column("值", style="green")
        
        table.add_row("Model Type", config.model_type)
        table.add_row("Hidden Size", str(config.hidden_size))
        table.add_row("Num Heads", str(config.num_heads))
        table.add_row("Head Dim", str(config.head_dim))
        table.add_row("Num Layers", str(config.num_layers))
        
        if config.intermediate_size:
            table.add_row("Intermediate Size", str(config.intermediate_size))
        if config.vocab_size:
            table.add_row("Vocab Size", str(config.vocab_size))
        
        console.print(Panel(table, border_style="blue"))
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        return {
            'num_initializers': len(self.model.graph.initializer),
            'num_nodes': len(self.model.graph.node),
            'num_inputs': len(self.model.graph.input),
            'num_outputs': len(self.model.graph.output),
            'op_types': list(set(node.op_type for node in self.model.graph.node)),
        }


def auto_detect_config(config_path: Path) -> ModelConfig:
    """
    便捷函数：自动检测模型配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        ModelConfig
    """
    analyzer = ModelAnalyzer(config_path=config_path)
    return analyzer.analyze()
