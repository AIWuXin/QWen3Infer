"""
量化器注册表

管理所有可用的量化执行器
"""

from typing import Dict, Type

from .base import QuantizationExecutor, BaseQuantizer
from ..config import QuantizationConfig, QuantizationMode


class QuantizerRegistry:
    """量化执行器注册表"""
    
    _executors: Dict[QuantizationMode, Type[QuantizationExecutor]] = {}
    _quantizers: Dict[QuantizationMode, Type[BaseQuantizer]] = {}
    
    @classmethod
    def register(
        cls, 
        mode: QuantizationMode,
        executor_class: Type[QuantizationExecutor] = None,
        quantizer_class: Type[BaseQuantizer] = None
    ):
        """
        注册量化执行器
        
        Args:
            mode: 量化模式
            executor_class: 执行器类
            quantizer_class: 量化器类（可选，默认使用通用BaseQuantizer）
        """
        if executor_class:
            cls._executors[mode] = executor_class
        if quantizer_class:
            cls._quantizers[mode] = quantizer_class
    
    @classmethod
    def get_executor(cls, mode: QuantizationMode) -> Type[QuantizationExecutor]:
        """获取执行器类"""
        if mode not in cls._executors:
            raise KeyError(f"未注册的量化模式: {mode}")
        return cls._executors[mode]
    
    @classmethod
    def get_quantizer(cls, config: QuantizationConfig) -> BaseQuantizer:
        """
        获取量化器实例
        
        Args:
            config: 量化配置
        
        Returns:
            量化器实例
        """
        mode = config.mode
        
        if mode in cls._quantizers:
            # 使用自定义量化器
            return cls._quantizers[mode](config)
        elif mode in cls._executors:
            # 使用通用量化器
            return GenericQuantizer(config)
        else:
            raise KeyError(f"未注册的量化模式: {mode}")
    
    @classmethod
    def list_quantizers(cls) -> Dict[QuantizationMode, str]:
        """列出所有可用的量化器"""
        result = {}
        for mode, executor_class in cls._executors.items():
            result[mode] = executor_class.__doc__ or executor_class.__name__
        return result


class GenericQuantizer(BaseQuantizer):
    """通用量化器 - 使用注册的Executor"""
    
    def _create_executor(self) -> QuantizationExecutor:
        """创建配置指定的执行器"""
        executor_class = QuantizerRegistry.get_executor(self.config.mode)
        return executor_class()


def register_quantizer(mode: QuantizationMode):
    """装饰器：注册量化执行器"""
    def decorator(executor_class: Type[QuantizationExecutor]):
        QuantizerRegistry.register(mode, executor_class=executor_class)
        return executor_class
    return decorator
