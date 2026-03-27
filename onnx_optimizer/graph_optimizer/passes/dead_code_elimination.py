"""
死代码消除Pass

移除对输出没有贡献的节点
"""

from typing import Tuple, Set, List
from collections import deque
from onnx import ModelProto

from ..base import BaseGraphPass


class DeadCodeEliminationPass(BaseGraphPass):
    """
    死代码消除
    
    移除所有无法到达输出的节点，包括：
    - 未使用的initializer
    - 无法到达graph输出的中间节点
    """
    
    def __init__(self):
        super().__init__("DeadCodeElimination")
    
    def run(self, model: ModelProto) -> Tuple[ModelProto, bool]:
        """执行死代码消除"""
        graph = model.graph
        modified = False
        self.change_count = 0
        
        # 收集所有graph输出
        required_outputs = set()
        for output in graph.output:
            required_outputs.add(output.name)
        
        # 逆向遍历找到所有必需的节点
        required_nodes = set()
        queue = deque(required_outputs)
        
        # 构建节点输出到节点的映射
        output_to_node = {}
        for i, node in enumerate(graph.node):
            for output in node.output:
                output_to_node[output] = i
        
        # BFS找到所有必需节点
        while queue:
            tensor_name = queue.popleft()
            
            if tensor_name in output_to_node:
                node_idx = output_to_node[tensor_name]
                if node_idx not in required_nodes:
                    required_nodes.add(node_idx)
                    node = graph.node[node_idx]
                    
                    # 添加该节点的所有输入到队列
                    for input_name in node.input:
                        if input_name:
                            queue.append(input_name)
        
        # 收集需要删除的节点（逆序删除）
        nodes_to_remove = []
        for i in range(len(graph.node) - 1, -1, -1):
            if i not in required_nodes:
                nodes_to_remove.append(graph.node[i])
        
        # 删除节点
        for node in nodes_to_remove:
            graph.node.remove(node)
            self.change_count += 1
            modified = True
        
        # 清理未使用的initializer
        used_initializers = set()
        for node in graph.node:
            for input_name in node.input:
                used_initializers.add(input_name)
        
        initializers_to_remove = []
        for init in graph.initializer:
            if init.name not in used_initializers:
                initializers_to_remove.append(init)
        
        for init in initializers_to_remove:
            graph.initializer.remove(init)
            self.change_count += 1
            modified = True
        
        return model, modified
