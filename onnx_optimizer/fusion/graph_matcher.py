"""
基于onnx-graphsurgeon的子图匹配引擎

支持多种匹配模式：
- sequence: 线性节点序列匹配
- parallel: 并行分支匹配  
- subgraph: 复杂拓扑子图匹配
"""

import re
from typing import List, Dict, Any, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path

import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon import Node, Tensor, Graph


@dataclass
class MatchResult:
    """匹配结果"""
    matched: bool
    nodes: List[Node] = field(default_factory=list)
    inputs: List[Tensor] = field(default_factory=list)
    outputs: List[Tensor] = field(default_factory=list)
    bindings: Dict[str, Any] = field(default_factory=dict)  # 变量绑定
    
    def __bool__(self):
        return self.matched


class PatternNode:
    """模式节点定义"""
    
    def __init__(self, config: Dict[str, Any], index: int):
        self.index = index
        self.op = config.get('op', config.get('op_type'))
        self.name_pattern = config.get('name_pattern')
        self.attrs = config.get('attrs', {})
        self.inputs_spec = config.get('inputs', [])  # 输入连接规范
        self.outputs_spec = config.get('outputs', [])  # 输出连接规范
        
    def match(self, node: Node, graph: Graph) -> bool:
        """检查节点是否匹配当前模式"""
        # 检查算子类型
        if self.op and node.op != self.op:
            return False
        
        # 检查名称模式
        if self.name_pattern:
            if not re.match(self.name_pattern.replace('*', '.*'), node.name):
                return False
        
        # 检查属性
        for attr_name, expected_value in self.attrs.items():
            if attr_name not in node.attrs:
                return False
            actual_value = node.attrs[attr_name]
            if isinstance(expected_value, list):
                if list(actual_value) != expected_value:
                    return False
            elif actual_value != expected_value:
                return False
        
        return True
    
    def __repr__(self):
        return f"PatternNode({self.op}, idx={self.index})"


class SequenceMatcher:
    """
    序列匹配器
    
    匹配线性的节点序列，如：
    ReduceMean -> Sub -> Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul -> Add (LayerNorm)
    """
    
    def __init__(self, pattern_nodes: List[Dict[str, Any]]):
        self.pattern_nodes = [PatternNode(cfg, i) for i, cfg in enumerate(pattern_nodes)]
        
    def match(self, graph: Graph, start_node: Node) -> MatchResult:
        """
        从start_node开始尝试匹配序列
        
        Args:
            graph: onnx-graphsurgeon图
            start_node: 起始节点
            
        Returns:
            MatchResult
        """
        matched_nodes = []
        current_node = start_node
        bindings = {}
        
        for i, pattern_node in enumerate(self.pattern_nodes):
            if current_node is None:
                return MatchResult(False)
            
            # 检查当前节点是否匹配模式
            if not pattern_node.match(current_node, graph):
                return MatchResult(False)
            
            matched_nodes.append(current_node)
            bindings[f"node{i}"] = current_node
            bindings[f"node{i}.name"] = current_node.name
            bindings[f"node{i}.op"] = current_node.op
            
            # 导出属性绑定
            for attr_name, attr_value in current_node.attrs.items():
                bindings[f"node{i}.attrs.{attr_name}"] = attr_value
            
            # 导出输入绑定
            for j, inp in enumerate(current_node.inputs):
                bindings[f"node{i}.input[{j}]"] = inp
                if inp.name:
                    bindings[f"node{i}.input['{inp.name}']"] = inp
            
            # 导出输出绑定
            for j, out in enumerate(current_node.outputs):
                bindings[f"node{i}.output[{j}]"] = out
                if out.name:
                    bindings[f"node{i}.output['{out.name}']"] = out
            
            # 查找下一个节点（通过第一个输出）
            if i < len(self.pattern_nodes) - 1:
                next_node = self._find_next_node(current_node, pattern_node, self.pattern_nodes[i + 1])
                current_node = next_node
        
        # 确定整体输入输出
        inputs = self._determine_inputs(matched_nodes)
        outputs = self._determine_outputs(matched_nodes)
        bindings['pattern.input[0]'] = inputs[0] if inputs else None
        bindings['last.output[0]'] = outputs[0] if outputs else None
        
        return MatchResult(
            matched=True,
            nodes=matched_nodes,
            inputs=inputs,
            outputs=outputs,
            bindings=bindings
        )
    
    def _find_next_node(self, current_node: Node, current_pattern: PatternNode, 
                        next_pattern: PatternNode) -> Optional[Node]:
        """查找序列中的下一个节点"""
        # 默认策略：找当前节点的第一个输出的消费者
        if current_node.outputs:
            first_output = current_node.outputs[0]
            for consumer in first_output.outputs:
                if next_pattern.match(consumer, None):
                    return consumer
        return None
    
    def _determine_inputs(self, nodes: List[Node]) -> List[Tensor]:
        """确定整个模式的输入（不被模式内其他节点消费的输入）"""
        node_set = set(id(n) for n in nodes)
        external_inputs = []
        
        for node in nodes:
            for inp in node.inputs:
                # 输入是图输入，或者不是由模式内节点产生的
                if not inp.inputs or not any(id(producer) in node_set for producer in inp.inputs):
                    if inp not in external_inputs:
                        external_inputs.append(inp)
        
        return external_inputs
    
    def _determine_outputs(self, nodes: List[Node]) -> List[Tensor]:
        """确定整个模式的输出（被模式外节点消费的输出）"""
        node_set = set(id(n) for n in nodes)
        external_outputs = []
        
        for node in nodes:
            for out in node.outputs:
                # 输出被模式外的节点消费
                for consumer in out.outputs:
                    if id(consumer) not in node_set:
                        if out not in external_outputs:
                            external_outputs.append(out)
                        break
        
        return external_outputs
    
    def find_all_matches(self, graph: Graph) -> List[MatchResult]:
        """在图中找到所有匹配的序列"""
        matches = []
        matched_node_ids = set()  # 避免重叠匹配，使用id
        
        for node in graph.nodes:
            if id(node) in matched_node_ids:
                continue
            
            result = self.match(graph, node)
            if result:
                # 检查是否有重叠
                result_node_ids = {id(n) for n in result.nodes}
                if not result_node_ids & matched_node_ids:
                    matches.append(result)
                    matched_node_ids.update(result_node_ids)
        
        return matches


class ParallelMatcher:
    """
    并行分支匹配器
    
    匹配共享同一输入的多个并行分支，如Q/K/V投影：
    input -> MatMul(Q) -> ...
         -> MatMul(K) -> ...
         -> MatMul(V) -> ...
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.input_spec = config.get('input')
        self.branches = config.get('branches', [])
        self.merge_config = config.get('merge', {})
        
    def match(self, graph: Graph) -> List[MatchResult]:
        """匹配并行分支"""
        matches = []
        # 实现并行分支匹配逻辑
        return matches


class TopologyMatcher:
    """
    拓扑匹配器
    
    匹配复杂的子图拓扑结构，支持分支、合并、多输出等
    """
    
    def __init__(self, pattern: Dict[str, Any]):
        self.pattern = pattern
        self.nodes_config = pattern.get('nodes', [])
        self.edges_config = pattern.get('edges', [])
        
    def match(self, graph: Graph) -> List[MatchResult]:
        """在图中匹配拓扑结构"""
        matches = []
        # 实现拓扑匹配逻辑
        return matches


class GraphMatcher:
    """
    主匹配器
    
    根据配置创建相应的匹配器并执行匹配
    """
    
    def __init__(self, pattern_config: Dict[str, Any]):
        self.config = pattern_config
        self.pattern_type = pattern_config.get('type', 'sequence')
        self.matcher = self._create_matcher()
        
    def _create_matcher(self):
        """根据配置创建对应的匹配器"""
        if self.pattern_type == 'sequence':
            return SequenceMatcher(self.config.get('nodes', []))
        elif self.pattern_type == 'parallel':
            return ParallelMatcher(self.config)
        elif self.pattern_type == 'subgraph':
            return TopologyMatcher(self.config)
        else:
            raise ValueError(f"未知的匹配类型: {self.pattern_type}")
    
    def find_matches(self, graph: Graph) -> List[MatchResult]:
        """在图中找到所有匹配"""
        if isinstance(self.matcher, SequenceMatcher):
            return self.matcher.find_all_matches(graph)
        else:
            return self.matcher.match(graph)


def load_onnx_to_gs(model_path: Path) -> Tuple[Graph, onnx.ModelProto]:
    """
    加载ONNX模型并转换为onnx-graphsurgeon图
    
    Args:
        model_path: ONNX模型路径
        
    Returns:
        (gs.Graph, onnx.ModelProto)
    """
    model = onnx.load(model_path, load_external_data=False)
    graph = gs.import_onnx(model)
    return graph, model


def save_gs_to_onnx(graph: Graph, original_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    将onnx-graphsurgeon图转换回ONNX模型
    
    Args:
        graph: gs.Graph
        original_model: 原始ONNX模型（用于复制元数据）
        
    Returns:
        onnx.ModelProto
    """
    # 清理未使用的节点和tensor
    graph.cleanup()
    
    # 导出回ONNX
    new_model = gs.export_onnx(graph)
    
    # 复制原始模型的元数据
    new_model.ir_version = original_model.ir_version
    new_model.opset_import.extend(original_model.opset_import)
    if original_model.doc_string:
        new_model.doc_string = original_model.doc_string
    
    return new_model
