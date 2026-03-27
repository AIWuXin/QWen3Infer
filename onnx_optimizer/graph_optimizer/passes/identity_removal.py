"""
Identity节点移除Pass

移除Identity节点，直接连接输入输出
"""

from typing import Tuple, Dict
from onnx import ModelProto

from ..base import BaseGraphPass


class IdentityRemovalPass(BaseGraphPass):
    """
    Identity节点移除
    
    Identity节点只是将输入复制到输出，没有实际计算，可以移除
    同时修复输入输出的引用关系
    """
    
    def __init__(self):
        super().__init__("IdentityRemoval")
    
    def run(self, model: ModelProto) -> Tuple[ModelProto, bool]:
        """执行Identity节点移除"""
        graph = model.graph
        modified = False
        self.change_count = 0
        
        # 构建tensor重命名映射
        # Identity(x) -> y, 则将所有使用y的地方替换为x
        rename_map: Dict[str, str] = {}
        
        for node in graph.node:
            if node.op_type == "Identity":
                input_name = node.input[0]
                output_name = node.output[0]
                rename_map[output_name] = input_name
        
        # 如果没有Identity节点，直接返回
        if not rename_map:
            return model, False
        
        # 重命名所有节点中的引用
        for node in graph.node:
            # 重命名输入
            for i, input_name in enumerate(node.input):
                if input_name in rename_map:
                    node.input[i] = rename_map[input_name]
            
            # 重命名输出
            for i, output_name in enumerate(node.output):
                if output_name in rename_map:
                    # 如果这个输出也被重命名，需要传递
                    pass
        
        # 重命名graph输入
        for input_info in graph.input:
            if input_info.name in rename_map:
                input_info.name = rename_map[input_info.name]
        
        # 重命名graph输出
        for output_info in graph.output:
            if output_info.name in rename_map:
                output_info.name = rename_map[output_info.name]
        
        # 移除Identity节点
        nodes_to_remove = []
        for node in graph.node:
            if node.op_type == "Identity":
                nodes_to_remove.append(node)
        
        for node in nodes_to_remove:
            graph.node.remove(node)
            self.change_count += 1
            modified = True
        
        return model, modified
