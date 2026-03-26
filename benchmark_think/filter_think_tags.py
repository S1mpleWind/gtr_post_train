"""
过滤脚本：从输出中移除<think>标签
用途：在评测前清理思考模式的输出，只保留最终答案
"""

import json
import re
import sys
from pathlib import Path


def remove_think_tags(text):
    """
    移除<think>...</think>标签及其内容，保留最终答案
    
    Args:
        text: 包含<think>标签的文本
        
    Returns:
        清理后的文本（仅保留答案部分）
    """
    # 移除<think>...</think>及其内容
    # 使用非贪心匹配确保正确处理嵌套情况
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 清理多余的空白和换行
    cleaned = cleaned.strip()
    
    return cleaned


def filter_json_outputs(input_file, output_file):
    """
    过滤JSON格式的输出文件
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
    """
    print(f"Reading from: {input_file}")
    
    # 读取原始JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        if input_file.endswith('.json'):
            outputs = json.load(f)
        else:  # jsonl格式
            outputs = [json.loads(line) for line in f]
    
    # 过滤每个输出
    filtered_outputs = []
    think_count = 0
    
    for i, output in enumerate(outputs):
        if isinstance(output, str):
            if '<think>' in output:
                think_count += 1
            filtered = remove_think_tags(output)
        elif isinstance(output, dict):
            # 如果是字典，过滤每个值
            filtered = {}
            for key, value in output.items():
                if isinstance(value, str):
                    if '<think>' in value:
                        think_count += 1
                    filtered[key] = remove_think_tags(value)
                else:
                    filtered[key] = value
        else:
            filtered = output
        
        filtered_outputs.append(filtered)
    
    # 保存过滤后的输出
    print(f"Writing to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        if input_file.endswith('.json'):
            json.dump(filtered_outputs, f, ensure_ascii=False, indent=2)
        else:
            for output in filtered_outputs:
                f.write(json.dumps(output, ensure_ascii=False) + '\n')
    
    print(f"✓ 处理完成：共处理 {len(filtered_outputs)} 条输出")
    print(f"✓ 含有<think>标签的输出数：{think_count}")
    print(f"✓ 已保存到：{output_file}")


def batch_filter_directory(directory, pattern="outputs_*.json"):
    """
    批量过滤目录中的所有输出文件
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式
    """
    directory = Path(directory)
    
    # 查找所有匹配的输出文件
    output_files = list(directory.glob(pattern))
    
    if not output_files:
        print(f"未找到匹配 '{pattern}' 的文件在 {directory}")
        return
    
    print(f"找到 {len(output_files)} 个文件")
    print("=" * 60)
    
    for output_file in sorted(output_files):
        # 生成输出文件名（添加_filtered后缀）
        filtered_file = output_file.parent / f"{output_file.stem}_filtered.json"
        
        filter_json_outputs(str(output_file), str(filtered_file))
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.json', '_filtered.json')
        
        if Path(input_file).is_dir():
            # 批量处理目录
            batch_filter_directory(input_file)
        else:
            # 处理单个文件
            filter_json_outputs(input_file, output_file)
    else:
        print("使用方法：")
        print("  单个文件：python filter_think_tags.py <input.json> [output.json]")
        print("  整个目录：python filter_think_tags.py <directory>")
        print("\n示例：")
        print("  python filter_think_tags.py outputs_gsm8k.json")
        print("  python filter_think_tags.py ./temp_0.1/")
