import onnx
import json
import struct
from pathlib import Path
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree

console = Console()
print = console.print

def check_external_data(model_path, data_path, manifest_path):
    """检查外部数据文件中的权重"""
    # 加载模型（不加载外部数据）
    model = onnx.load(model_path, load_external_data=False)
    
    # 读取 manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # 获取 data.bin 文件大小
    data_size = Path(data_path).stat().st_size
    
    # 统计信息
    total_weights = len(model.graph.initializer)
    external_weights = {}  # 存储在 data.bin 中的权重
    internal_weights = {}  # 存储在 onnx 文件中的权重
    
    # 按层分类
    layer_weights = defaultdict(lambda: {"external": [], "internal": [], "empty": []})
    non_layer_weights = {"external": [], "internal": [], "empty": []}
    
    # 检查每个权重
    for init in model.graph.initializer:
        name = init.name
        has_raw_data = len(init.raw_data) > 0
        
        # 计算期望的数据大小
        tensor_size = 1
        for dim in init.dims:
            tensor_size *= dim
        
        # 确定数据类型大小
        dtype_size = {
            onnx.TensorProto.FLOAT: 4,
            onnx.TensorProto.FLOAT16: 2,
            onnx.TensorProto.INT64: 8,
            onnx.TensorProto.INT32: 4,
            onnx.TensorProto.INT16: 2,
            onnx.TensorProto.INT8: 1,
            onnx.TensorProto.UINT8: 1,
        }.get(init.data_type, 4)
        
        expected_bytes = tensor_size * dtype_size
        
        # 检查是否在 manifest 中
        if name in manifest:
            external_weights[name] = {
                "offset": manifest[name]["offset"],
                "size": manifest[name]["size"],
                "expected": expected_bytes,
                "dims": list(init.dims),
                "dtype": onnx.TensorProto.DataType.Name(init.data_type)
            }
            category = "external"
        elif has_raw_data:
            internal_weights[name] = {
                "size": len(init.raw_data),
                "expected": expected_bytes,
                "dims": list(init.dims),
                "dtype": onnx.TensorProto.DataType.Name(init.data_type)
            }
            category = "internal"
        else:
            category = "empty"
        
        # 按层分类
        if "model.layers." in name:
            try:
                layer_num = int(name.split("model.layers.")[1].split(".")[0])
                layer_weights[layer_num][category].append(name)
            except (IndexError, ValueError):
                non_layer_weights[category].append(name)
        else:
            non_layer_weights[category].append(name)
    
    return {
        "model": model,
        "manifest": manifest,
        "data_size": data_size,
        "total_weights": total_weights,
        "external_weights": external_weights,
        "internal_weights": internal_weights,
        "layer_weights": layer_weights,
        "non_layer_weights": non_layer_weights
    }


def main():
    model_path = "models/onnx/qwen3_0.6b/opt/prefill.onnx"
    data_path = "models/onnx/qwen3_0.6b/opt/data.bin"
    manifest_path = "models/onnx/qwen3_0.6b/opt/manifest.json"
    
    print(Panel.fit(f"[bold cyan]模型结构分析: {model_path}[/bold cyan]"))
    
    try:
        result = check_external_data(model_path, data_path, manifest_path)
        
        # 打印文件信息
        print(f"\n[bold yellow]📁 文件信息:[/bold yellow]")
        print(f"  ONNX 文件: {Path(model_path).stat().st_size / 1024 / 1024:.2f} MB")
        print(f"  Data 文件: {result['data_size'] / 1024 / 1024:.2f} MB")
        print(f"  Manifest 条目: {len(result['manifest'])}")
        
        # 逐层权重统计表格
        print(f"\n[bold yellow]📊 逐层权重统计:[/bold yellow]")
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("层号", style="cyan", justify="center", width=6)
        table.add_column("外部数据", justify="right", style="green", width=10)
        table.add_column("内部数据", justify="right", style="blue", width=10)
        table.add_column("缺失", justify="right", style="red", width=8)
        table.add_column("状态", justify="center", width=10)
        
        layer_keys = sorted(result['layer_weights'].keys())
        complete_layers = 0
        partial_layers = 0
        empty_layers = 0
        
        for layer_num in layer_keys:
            stats = result['layer_weights'][layer_num]
            ext_count = len(stats["external"])
            int_count = len(stats["internal"])
            empty_count = len(stats["empty"])
            total = ext_count + int_count + empty_count
            
            if ext_count == total:
                status = "[green]✅ 完整[/green]"
                complete_layers += 1
            elif ext_count + int_count > 0:
                status = "[yellow]⚠️ 部分[/yellow]"
                partial_layers += 1
            else:
                status = "[red]❌ 全空[/red]"
                empty_layers += 1
            
            table.add_row(
                str(layer_num),
                str(ext_count),
                str(int_count),
                str(empty_count),
                status
            )
        
        print(table)
        
        # 非层权重统计
        non_layer = result['non_layer_weights']
        print(f"\n[bold yellow]📋 非层权重统计:[/bold yellow]")
        print(f"  外部数据: {len(non_layer['external'])} 个")
        print(f"  内部数据: {len(non_layer['internal'])} 个")
        print(f"  缺失: {len(non_layer['empty'])} 个")
        
        # 分类显示
        embed_ext = [n for n in non_layer['external'] if 'embed' in n.lower()]
        norm_ext = [n for n in non_layer['external'] if 'norm' in n.lower()]
        lm_head_ext = [n for n in non_layer['external'] if 'lm_head' in n.lower() or 'head' in n.lower()]
        
        if embed_ext:
            print(f"    └─ Embedding: {len(embed_ext)} 个 ✅")
        if norm_ext:
            print(f"    └─ Norm: {len(norm_ext)} 个 ✅")
        if lm_head_ext:
            print(f"    └─ LM Head: {len(lm_head_ext)} 个 ✅")
        
        # 总结
        total_external = len(result['external_weights'])
        total_internal = len(result['internal_weights'])
        total_empty = result['total_weights'] - total_external - total_internal
        
        print(f"\n[bold yellow]📈 总结:[/bold yellow]")
        summary_table = Table(show_header=False, box=None)
        summary_table.add_column("项目", style="cyan")
        summary_table.add_column("数量", justify="right")
        summary_table.add_row("总权重数", str(result['total_weights']))
        summary_table.add_row("外部数据 (data.bin)", f"[green]{total_external}[/green]")
        summary_table.add_row("内部数据 (onnx文件)", f"[blue]{total_internal}[/blue]")
        summary_table.add_row("缺失权重", f"[red]{total_empty}[/red]" if total_empty > 0 else "[green]0[/green]")
        summary_table.add_row("", "")
        summary_table.add_row("完整层数", f"[green]{complete_layers}[/green]/{len(layer_keys)}")
        summary_table.add_row("部分层数", f"[yellow]{partial_layers}[/yellow]/{len(layer_keys)}")
        summary_table.add_row("空层数", f"[red]{empty_layers}[/red]/{len(layer_keys)}")
        print(summary_table)
        
        # 权重详情树状图
        if result['external_weights']:
            print(f"\n[bold yellow]💾 外部数据详情 (前5个):[/bold yellow]")
            tree = Tree(f"[bold]data.bin[/bold] ({result['data_size']/1024/1024:.1f} MB)")
            
            for i, (name, info) in enumerate(list(result['external_weights'].items())[:5]):
                size_kb = info['size'] / 1024
                node = tree.add(f"[cyan]{name}[/cyan]")
                node.add(f"偏移: {info['offset']:,} | 大小: {size_kb:.1f} KB | 形状: {info['dims']} | 类型: {info['dtype']}")
            
            if len(result['external_weights']) > 5:
                tree.add(f"[dim]... 还有 {len(result['external_weights']) - 5} 个权重 ...[/dim]")
            
            print(tree)
        
        # 验证 data.bin 完整性
        print(f"\n[bold yellow]🔍 Data.bin 完整性检查:[/bold yellow]")
        last_offset = 0
        max_offset_end = 0
        
        for name, info in result['external_weights'].items():
            offset_end = info['offset'] + info['size']
            if offset_end > max_offset_end:
                max_offset_end = offset_end
        
        if max_offset_end <= result['data_size']:
            print(f"  [green]✅ 数据范围有效[/green]")
            print(f"     最大使用偏移: {max_offset_end:,} bytes")
            print(f"     文件总大小: {result['data_size']:,} bytes")
            print(f"     利用率: {max_offset_end/result['data_size']*100:.1f}%")
        else:
            print(f"  [red]❌ 数据超出范围![/red]")
            print(f"     最大使用偏移: {max_offset_end:,} bytes")
            print(f"     文件总大小: {result['data_size']:,} bytes")
        
    except Exception as e:
        print(f"[bold red]❌ 错误: {e}[/bold red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
