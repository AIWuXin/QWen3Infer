"""
YAML配置加载器

加载层融合的配置文件
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List
from rich.console import Console

console = Console()


class FusionConfigLoader:
    """
    融合配置加载器
    
    从YAML文件加载融合规则
    """
    
    def __init__(self, config_dir: Path = None):
        if config_dir is None:
            config_dir = Path(__file__).parent / "configs"
        self.config_dir = config_dir
    
    def load(self, level: str) -> Dict[str, Any]:
        """
        加载指定级别的配置
        
        Args:
            level: 'o1', 'o2', 'o3' 或自定义配置文件路径
        
        Returns:
            配置字典
        """
        if level in ('o1', 'o2', 'o3'):
            config_file = self.config_dir / f"{level}_basic.yaml"
        else:
            config_file = Path(level)
        
        if not config_file.exists():
            raise FileNotFoundError(f"找不到配置文件: {config_file}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        console.print(f"[dim]加载融合配置: {config_file.name}[/dim]")
        return config
    
    def list_available_configs(self) -> List[Path]:
        """列出可用的配置文件"""
        if not self.config_dir.exists():
            return []
        return list(self.config_dir.glob("*.yaml"))


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
