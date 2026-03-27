"""
常量折叠Pass

将可以预先计算的常量表达式在编译期求值
"""

from typing import Tuple, List, Dict
import numpy as np
from onnx import ModelProto, TensorProto, numpy_helper
from onnx.helper import make_tensor

from ..base import BaseGraphPass


class ConstantFoldingPass(BaseGraphPass):
    """
    常量折叠优化
    
    识别并计算常量子图，将结果替换为常量节点
    例如: Mul(2, 3) -> Constant(6)
    """
    
    def __init__(self):
        super().__init__("ConstantFolding")
        self.constant_tensors: Dict[str, np.ndarray] = {}
    
    def run(self, model: ModelProto) -> Tuple[ModelProto, bool]:
        """执行常量折叠"""
        graph = model.graph
        modified = False
        self.change_count = 0
        
        # 第一遍：收集所有常量
        self._collect_constants(graph)
        
        # 第二遍：识别可折叠的节点
        nodes_to_remove = []
        
        for node in graph.node:
            if self._can_fold(node, graph):
                # 执行折叠
                result = self._fold_node(node)
                if result is not None:
                    # 创建新的常量tensor
                    tensor_name = node.output[0]
                    new_tensor = make_tensor(
                        name=tensor_name,
                        data_type=TensorProto.FLOAT,
                        dims=result.shape,
                        vals=result.flatten().tolist()
                    )
                    
                    # 添加到initializer
                    graph.initializer.append(new_tensor)
                    self.constant_tensors[tensor_name] = result
                    
                    # 标记节点待删除
                    nodes_to_remove.append(node)
                    self.change_count += 1
                    modified = True
        
        # 删除已折叠的节点
        for node in nodes_to_remove:
            graph.node.remove(node)
        
        return model, modified
    
    def _collect_constants(self, graph):
        """收集图中的所有常量"""
        # 从initializer收集
        for init in graph.initializer:
            arr = numpy_helper.to_array(init)
            self.constant_tensors[init.name] = arr
        
        # 从Constant节点收集
        for node in graph.node:
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.type == TensorProto.TENSOR:
                        arr = numpy_helper.to_array(attr.t)
                        self.constant_tensors[node.output[0]] = arr
    
    def _can_fold(self, node, graph) -> bool:
        """判断节点是否可以折叠"""
        # 只处理纯计算节点
        foldable_ops = {
            'Add', 'Sub', 'Mul', 'Div', 'Pow', 'Sqrt', 'Exp', 'Log',
            'Abs', 'Neg', 'Floor', 'Ceil', 'Round',
            'Reshape', 'Expand', 'Gather', 'Concat',
            'Unsqueeze', 'Squeeze', 'Slice', 'Transpose',
            'Cast', 'Clip', 'Relu', 'Sigmoid', 'Tanh'
        }
        
        if node.op_type not in foldable_ops:
            return False
        
        # 检查所有输入是否都是常量
        for input_name in node.input:
            if input_name and input_name not in self.constant_tensors:
                return False
        
        return True
    
    def _fold_node(self, node) -> np.ndarray:
        """
        执行节点计算
        
        Returns:
            计算结果，失败返回None
        """
        try:
            # 获取输入
            inputs = []
            for input_name in node.input:
                if input_name:
                    inputs.append(self.constant_tensors[input_name])
            
            op_type = node.op_type
            
            # 执行计算
            if op_type == 'Add':
                return np.add(inputs[0], inputs[1])
            elif op_type == 'Sub':
                return np.subtract(inputs[0], inputs[1])
            elif op_type == 'Mul':
                return np.multiply(inputs[0], inputs[1])
            elif op_type == 'Div':
                return np.divide(inputs[0], inputs[1])
            elif op_type == 'Pow':
                return np.power(inputs[0], inputs[1])
            elif op_type == 'Sqrt':
                return np.sqrt(inputs[0])
            elif op_type == 'Exp':
                return np.exp(inputs[0])
            elif op_type == 'Log':
                return np.log(inputs[0])
            elif op_type == 'Abs':
                return np.abs(inputs[0])
            elif op_type == 'Neg':
                return np.negative(inputs[0])
            elif op_type == 'Floor':
                return np.floor(inputs[0])
            elif op_type == 'Ceil':
                return np.ceil(inputs[0])
            elif op_type == 'Round':
                return np.round(inputs[0])
            elif op_type == 'Reshape':
                new_shape = inputs[1].astype(np.int64)
                return inputs[0].reshape(new_shape.tolist())
            elif op_type == 'Transpose':
                perm = None
                for attr in node.attribute:
                    if attr.name == 'perm':
                        perm = list(attr.ints)
                return np.transpose(inputs[0], axes=perm)
            elif op_type == 'Relu':
                return np.maximum(inputs[0], 0)
            elif op_type == 'Sigmoid':
                return 1 / (1 + np.exp(-inputs[0]))
            elif op_type == 'Tanh':
                return np.tanh(inputs[0])
            # ... 更多算子
            
            else:
                return None
                
        except Exception as e:
            self.log(f"折叠失败 {node.op_type}: {e}")
            return None
