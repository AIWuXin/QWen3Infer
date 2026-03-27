"""
子图替换引擎

将匹配的子图替换为目标算子或子图
"""

from typing import List, Dict, Any, Optional, Union
from copy import deepcopy

import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon import Node, Tensor, Graph

from .graph_matcher import MatchResult


class ReplacementBuilder:
    """
    替换节点构建器
    
    根据YAML配置构建替换节点
    """
    
    def __init__(self, replacement_config: Dict[str, Any], match_result: MatchResult):
        self.config = replacement_config
        self.match = match_result
        
    def build(self, graph: Graph) -> Optional[Node]:
        """
        根据配置构建替换节点
        
        Args:
            graph: 目标图
            
        Returns:
            新建的节点
        """
        op_type = self.config.get('op')
        if not op_type:
            return None
        
        # 解析输入
        inputs = self._resolve_inputs()
        
        # 解析输出
        outputs = self._resolve_outputs()
        
        # 解析属性
        attrs = self._resolve_attrs()
        
        # 解析domain
        domain = self.config.get('domain', '')
        
        # 创建新节点
        new_node = gs.Node(
            op=op_type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            domain=domain if domain else None
        )
        
        return new_node
    
    def _resolve_inputs(self) -> List[Tensor]:
        """解析输入tensor"""
        inputs_config = self.config.get('inputs', [])
        inputs = []
        
        for inp_spec in inputs_config:
            tensor = self._resolve_tensor_reference(inp_spec)
            if tensor:
                inputs.append(tensor)
        
        return inputs
    
    def _resolve_outputs(self) -> List[Tensor]:
        """解析输出tensor"""
        outputs_config = self.config.get('outputs', [])
        outputs = []
        
        for out_spec in outputs_config:
            # 如果是字符串引用，解析它
            if isinstance(out_spec, str) and out_spec.startswith('${'):
                tensor = self._resolve_tensor_reference(out_spec)
                if tensor:
                    outputs.append(tensor)
            else:
                # 创建新的输出tensor
                tensor = gs.Variable(name=str(out_spec))
                outputs.append(tensor)
        
        # 如果没有指定输出，使用原模式的输出
        if not outputs and self.match.outputs:
            outputs = [gs.Variable(name=f"{out.name}_fused" if out.name else "fused_out") 
                      for out in self.match.outputs]
        
        return outputs
    
    def _resolve_attrs(self) -> Dict[str, Any]:
        """解析属性"""
        attrs_config = self.config.get('attrs', {})
        attrs = {}
        
        for attr_name, attr_value in attrs_config.items():
            if isinstance(attr_value, str) and attr_value.startswith('${'):
                # 解析变量引用
                resolved = self._resolve_value(attr_value)
                attrs[attr_name] = resolved
            else:
                attrs[attr_name] = attr_value
        
        return attrs
    
    def _resolve_tensor_reference(self, ref: str) -> Optional[Tensor]:
        """
        解析tensor引用
        
        支持的格式：
        - ${pattern.input[0]} - 模式的首个输入
        - ${node0.input[0]} - 第1个节点的第1个输入
        - ${node0.output[0]} - 第1个节点的第1个输出
        - ${last.output[0]} - 最后一个节点的第1个输出
        """
        if not isinstance(ref, str) or not ref.startswith('${'):
            return None
        
        # 去掉${和}
        expr = ref[2:-1].strip()
        
        # 从bindings中查找
        if expr in self.match.bindings:
            result = self.match.bindings[expr]
            if isinstance(result, Tensor):
                return result
        
        # 特殊处理
        if expr == 'pattern.input[0]':
            if self.match.inputs:
                return self.match.inputs[0]
        elif expr == 'last.output[0]':
            if self.match.outputs:
                return self.match.outputs[0]
        elif 'input[' in expr or 'output[' in expr:
            # 尝试从bindings中查找
            parts = expr.split('.')
            if len(parts) == 2:
                node_ref, io_ref = parts
                if node_ref in self.match.bindings:
                    node = self.match.bindings[node_ref]
                    if isinstance(node, Node):
                        # 解析input[N]或output[N]
                        if 'input[' in io_ref:
                            idx = int(io_ref[6:-1])
                            if idx < len(node.inputs):
                                return node.inputs[idx]
                        elif 'output[' in io_ref:
                            idx = int(io_ref[7:-1])
                            if idx < len(node.outputs):
                                return node.outputs[idx]
        
        return None
    
    def _resolve_value(self, ref: str) -> Any:
        """
        解析值引用
        
        支持从bindings中解析各种值
        """
        if not isinstance(ref, str) or not ref.startswith('${'):
            return ref
        
        expr = ref[2:-1].strip()
        
        if expr in self.match.bindings:
            return self.match.bindings[expr]
        
        # 处理嵌套属性访问，如 node4.attrs.epsilon
        parts = expr.split('.')
        if len(parts) >= 2 and parts[0].startswith('node'):
            node_key = parts[0]
            if node_key in self.match.bindings:
                node = self.match.bindings[node_key]
                if isinstance(node, Node) and len(parts) >= 3 and parts[1] == 'attrs':
                    attr_name = parts[2]
                    return node.attrs.get(attr_name)
        
        return ref


class SubgraphReplacer:
    """
    子图替换器
    
    负责将匹配的子图替换为新的节点或子图
    """
    
    def __init__(self, graph: Graph):
        self.graph = graph
        
    def replace(self, match_result: MatchResult, replacement_config: Dict[str, Any]) -> Optional[Node]:
        """
        替换匹配的子图
        
        Args:
            match_result: 匹配结果
            replacement_config: 替换配置
            
        Returns:
            新建的替换节点
        """
        # 1. 构建替换节点
        builder = ReplacementBuilder(replacement_config, match_result)
        new_node = builder.build(self.graph)
        
        if new_node is None:
            return None
        
        # 2. 先记录图输出关系（在断开连接前）
        self._record_graph_outputs(match_result)
        
        # 3. 使用GS的API断开并移除原节点
        for old_node in list(match_result.nodes):  # 使用list复制避免修改迭代
            # 断开所有输入连接
            for inp_tensor in list(old_node.inputs):
                if old_node in inp_tensor.outputs:
                    inp_tensor.outputs.remove(old_node)
            
            # 断开所有输出连接
            for out_tensor in list(old_node.outputs):
                # 清空此tensor的生产者（即该节点）
                if old_node in out_tensor.inputs:
                    out_tensor.inputs.remove(old_node)
                # 清空此tensor的消费者引用
                out_tensor.outputs.clear()
            
            # 清空节点的输入输出引用
            old_node.inputs.clear()
            old_node.outputs.clear()
            
            # 从图中移除节点
            if old_node in self.graph.nodes:
                self.graph.nodes.remove(old_node)
        
        # 3. 添加新节点到图
        self.graph.nodes.append(new_node)
        
        # 4. 处理输入连接
        self._connect_inputs(new_node, match_result)
        
        # 5. 处理输出连接
        self._connect_outputs(new_node, match_result)
        
        # 6. 处理图输出（如果匹配的输出是图输出）
        self._update_graph_outputs(new_node, match_result)
        
        return new_node
    
    def _record_graph_outputs(self, match_result: MatchResult) -> Dict[int, str]:
        """
        记录哪些匹配节点的输出是图输出
        
        Returns:
            {output_index_in_matched_node: graph_output_name}
        """
        self._graph_output_mapping = {}
        
        for node_idx, node in enumerate(match_result.nodes):
            for out_idx, output in enumerate(node.outputs):
                for graph_out in self.graph.outputs:
                    if graph_out.name == output.name:
                        self._graph_output_mapping[out_idx] = output.name
                        break
        
        return self._graph_output_mapping
    
    def _update_graph_outputs(self, new_node: Node, match_result: MatchResult):
        """更新图的输出（如果匹配的输出是图输出）"""
        if not hasattr(self, '_graph_output_mapping'):
            return
        
        # 更新图输出
        for out_idx, graph_out_name in self._graph_output_mapping.items():
            if out_idx < len(new_node.outputs):
                new_output = new_node.outputs[out_idx]
                new_output.name = graph_out_name  # 保持输出名一致
                
                # 替换图输出列表中的对应项
                for i, graph_out in enumerate(self.graph.outputs):
                    if graph_out.name == graph_out_name:
                        self.graph.outputs[i] = new_output
                        break
    
    def _connect_inputs(self, new_node: Node, match_result: MatchResult):
        """连接新节点的输入"""
        # 输入已经在build时解析好了
        # 这里确保输入tensor的消费者被正确更新
        for inp in new_node.inputs:
            if new_node not in inp.outputs:
                inp.outputs.append(new_node)
    
    def _connect_outputs(self, new_node: Node, match_result: MatchResult):
        """连接新节点的输出到原有消费者"""
        # 收集原模式输出的所有外部消费者
        external_consumers = {}  # output_tensor -> [consumers]
        
        for old_output in match_result.outputs:
            consumers = []
            for consumer in old_output.outputs:
                if consumer not in match_result.nodes:
                    consumers.append(consumer)
            if consumers:
                external_consumers[old_output] = consumers
        
        # 将新节点的输出连接到这些消费者
        for i, new_output in enumerate(new_node.outputs):
            if i < len(match_result.outputs):
                old_output = match_result.outputs[i]
                if old_output in external_consumers:
                    for consumer in external_consumers[old_output]:
                        # 更新消费者的输入
                        for j, consumer_input in enumerate(consumer.inputs):
                            if consumer_input == old_output:
                                consumer.inputs[j] = new_output
                        # 更新输出的消费者列表
                        if consumer not in new_output.outputs:
                            new_output.outputs.append(consumer)


def apply_fusion_pass(graph: Graph, pattern_config: Dict[str, Any], 
                      replacement_config: Dict[str, Any]) -> int:
    """
    应用单个融合pass
    
    Args:
        graph: 目标图
        pattern_config: 匹配模式配置
        replacement_config: 替换配置
        
    Returns:
        融合次数
    """
    from .graph_matcher import GraphMatcher
    
    matcher = GraphMatcher(pattern_config)
    matches = matcher.find_matches(graph)
    
    replacer = SubgraphReplacer(graph)
    fusion_count = 0
    
    for match in matches:
        new_node = replacer.replace(match, replacement_config)
        if new_node:
            fusion_count += 1
    
    return fusion_count
