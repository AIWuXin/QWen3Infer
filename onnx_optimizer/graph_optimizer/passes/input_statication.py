"""
输入静态化Pass

将动态维度（如batch_size, seq_len）固定为静态值
"""

from typing import Tuple, Optional
from onnx import ModelProto, TensorProto, ValueInfoProto
from onnx.helper import make_tensor_value_info

from ..base import BaseGraphPass


class InputStaticationPass(BaseGraphPass):
    """
    输入静态化
    
    将动态输入维度固定为指定值，例如:
    - batch_size: 从'dim_value'变为1
    - seq_len: 从'dim_param'变为固定长度
    """
    
    def __init__(
        self,
        batch_size: int = 1,
        seq_len: Optional[int] = None,
        input_names: Optional[list] = None
    ):
        super().__init__("InputStatication")
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.input_names = input_names  # 如果指定，只处理这些输入
    
    def run(self, model: ModelProto) -> Tuple[ModelProto, bool]:
        """执行输入静态化"""
        graph = model.graph
        modified = False
        self.change_count = 0
        
        for input_info in graph.input:
            if self.input_names and input_info.name not in self.input_names:
                continue
            
            tensor_type = input_info.type.tensor_type
            if not tensor_type.HasField('shape'):
                continue
            
            shape = tensor_type.shape
            
            # 修改batch维度（第一个维度）
            if len(shape.dim) > 0:
                dim = shape.dim[0]
                if dim.HasField('dim_param') or not dim.HasField('dim_value'):
                    dim.dim_value = self.batch_size
                    dim.ClearField('dim_param')
                    self.change_count += 1
                    modified = True
            
            # 修改序列维度（第二个维度，通常是seq_len）
            if self.seq_len is not None and len(shape.dim) > 1:
                dim = shape.dim[1]
                if dim.HasField('dim_param') or not dim.HasField('dim_value'):
                    dim.dim_value = self.seq_len
                    dim.ClearField('dim_param')
                    self.change_count += 1
                    modified = True
        
        return model, modified
