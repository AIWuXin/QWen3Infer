"""
使用 optimum-cli 导出 Qwen3 ONNX 模型
这是 HuggingFace 推荐的导出方式，支持 KV Cache 和动态轴
"""
import subprocess
import os
from rich.console import Console

console = Console()


if __name__ == '__main__':
    model_name = "Qwen/Qwen3-0.6B"
    output_dir = "./models/onnx/qwen3_0.6b"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    console.print(f"[green]开始导出 {model_name} 到 ONNX 格式...[/green]")
    console.print(f"[yellow]输出目录: {output_dir}[/yellow]")
    
    # 使用 optimum-cli 导出
    # 支持的选项：
    # --task text-generation: 文本生成任务
    # --fp16: 使用 FP16 精度
    # --optimize O1/O2/O3/O4: 优化级别
    cmd = [
        "optimum-cli", "export", "onnx",
        "--model", model_name,
        output_dir,
        "--task", "text-generation-with-past",  # 带 KV Cache 的文本生成
        "--trust-remote-code",
        "--no-post-process",  # 跳过 post-processing（避免 protobuf 2GB 限制）
        "--dtype", "fp16",
        "--opset", "18",
        "--batch_size", "1",
    ]
    
    console.print(f"[blue]执行命令: {' '.join(cmd)}[/blue]")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        console.print(result.stdout)
        console.print("🎉 导出成功！")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]导出失败: {e}[/red]")
        console.print(f"[red]错误输出: {e.stderr}[/red]")
        
        # 如果 optimum-cli 不可用，提供手动安装建议
        console.print("\n[yellow]请确保已安装 optimum：[yellow]")
        console.print("  uv pip install optimum[onnxruntime-gpu]")
