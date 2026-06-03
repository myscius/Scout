import argparse
import os
import json
from pathlib import Path
import openai
import re
from typing import Dict, List, Optional

def format_time(time_value):
    """
    将时间值（无论是秒数浮点型还是 'HH:MM:SS.ms' 字符串）统一格式化为 'HH:MM:SS.ms' 字符串。
    """
    # 如果值已经是字符串，直接返回，假定其格式正确
    if isinstance(time_value, str):
        return time_value
    
    # 如果是数字（秒数），则进行转换
    if isinstance(time_value, (int, float)):
        seconds = time_value
        # 计算小时、分钟、秒和毫秒
        milliseconds = round((seconds - int(seconds)) * 1000)
        total_seconds = int(seconds)
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        # 格式化字符串
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    
    # 对于其他未知类型，返回其字符串表示形式
    return str(time_value)

def parse_srt(file_path):
    """
    解析SRT文件并提取所有字幕文本。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误: 文件未找到 '{file_path}'")

    text_content = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 忽略序号、时间戳和空行，只提取文本行
            if not line.strip().isdigit() and '-->' not in line and line.strip():
                text_content.append(line.strip())
    
    return " ".join(text_content)

def load_jsonl_subtitles(jsonl_path: str) -> List[Dict]:
    """读取 jsonl 字幕，要求每行是一个 JSON 对象。"""
    items: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {line_no} 行不是合法 JSON: {e}") from e

            if not isinstance(obj, dict):
                raise ValueError(f"第 {line_no} 行不是 JSON 对象。")
            if "text" not in obj or "start_time" not in obj or "end_time" not in obj:
                raise ValueError(f"第 {line_no} 行缺少必要字段 text/start_time/end_time。")

            items.append(obj)
    return items


def subtitles_to_time_text_lines(
    subtitles: List[Dict],
    merge: bool = True,
    max_gap: float = 0.6,
) -> List[str]:
    """
    将字幕对象列表转换为:
    [HH:MM:SS.mmm-HH:MM:SS.mmm] 字幕内容

    merge=True 时会按时间连续性做轻量合并，减少“逐词一行”噪声。
    """
    if not subtitles:
        return []

    normalized = []
    for i, item in enumerate(subtitles, 1):
        text = str(item["text"]).strip()
        start = float(item["start_time"])
        end = float(item["end_time"])
        if end < start:
            start, end = end, start
        if not text:
            continue
        normalized.append({"text": text, "start": start, "end": end, "index": i})

    if not normalized:
        return []

    normalized.sort(key=lambda x: (x["start"], x["end"], x["index"]))

    if not merge:
        return [
            f"[{format_time(x['start'])}-{format_time(x['end'])}] {x['text']}"
            for x in normalized
        ]

    merged: List[Dict] = []
    current = normalized[0].copy()

    for nxt in normalized[1:]:
        gap = nxt["start"] - current["end"]
        same_or_overlap = gap <= max_gap
        if same_or_overlap:
            current["text"] = f"{current['text']} {nxt['text']}".strip()
            current["end"] = max(current["end"], nxt["end"])
        else:
            merged.append(current)
            current = nxt.copy()
    merged.append(current)

    return [
        f"[{format_time(x['start'])}-{format_time(x['end'])}] {x['text']}"
        for x in merged
    ]

def get_subtitle(file_path, num_samples=None):
    """
    查看字幕文件样例，支持 .srt 和 .json 格式。
    仅提取时间戳和文本进行格式化展示。
    """
    path = Path(file_path)
    if not path.exists():
        print(f"警告: 文件不存在 '{file_path}'")
        return

    print(f"\n>>> 查看字幕文件样例: {path.name}")
    try:
        if path.suffix.lower() == '.srt':
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            # SRT通常由空行分隔块
            blocks = content.split('\n\n')
            # print(f"格式: SRT | 总块数: {len(blocks)} | 展示前 {num_samples} 个:")
            
            # subtitle content
            subtitle_content = ""
            for i, block in enumerate(blocks[:num_samples]):
                lines = block.split('\n')
                if len(lines) >= 3:
                    # lines[0] 是序号, lines[1] 是时间戳, lines[2:] 是文本
                    timestamp = lines[1].replace(' --> ', '-').replace(',', '.')
                    text = " ".join(lines[2:])
                    # 清理HTML标签
                    text = re.sub(r'<[^>]+>', '', text)
                    # print(f"[{timestamp}] {text}")
                    subtitle_content += f"[{timestamp}] {text} "
                else:
                    print(f"--- [Sample {i+1}] (格式异常) ---\n{block}")
                
        elif path.suffix.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            items = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                # 尝试寻找包含列表的字段
                for k, v in data.items():
                    if isinstance(v, list) and len(v) > 0:
                        items = v
                        break
                if not items:
                    print("未找到列表数据")
                    return

            # print(f"列表长度: {len(items)} | 展示前 {num_samples} 个:")
            # subtitle content
            subtitle_content = ""
            for i, item in enumerate(items[:num_samples]):
                # 尝试智能提取常见字段
                if 'timestamp' in item:
                    start = format_time(item['timestamp'][0])
                    end = format_time(item['timestamp'][1])
                    timestamp = f"{start}-{end}"
                elif 'start' in item:
                    start = format_time(item.get('start'))
                    end = format_time(item.get('end'))
                    timestamp = f"{start}-{end}"
                else:
                    print("Find no timestamp field")
                    timestamp = "N/A"
                
                text = item.get('text') or item.get('content') or item.get('line') or str(item)
                # 清理HTML标签
                # text = re.sub(r'<[^>]+>', '', str(text))
                
                # print(f"[{timestamp}] {text}")
                subtitle_content += f"[{timestamp}] {text} "
        elif path.suffix.lower() == '.jsonl':
            content_lines = subtitles_to_time_text_lines(load_jsonl_subtitles(str(path)))
            subtitle_content = " ".join(content_lines[:num_samples])
        else:
            print(f"不支持的查看格式: {path.suffix}")
            
    except Exception as e:
        print(f"读取样例出错: {e}")
    print("-" * 30 + "\n")
    return subtitle_content

def reorganize_with_openai(text, api_key, base_url, model_name):
    """
    使用OpenAI格式的API重新组织文本内容。
    """
    print("原始字幕拼接", text)
    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        prompt = f"Please organize the following subtitle content, reassemble the time segments and sentence segments according to semantics, merge the time segments corresponding to different segments of the complete sentence, and reorganize them into pairs of time segments and sentences. Please keep your reply concise, retaining only the necessary timestamps [start-end] and stitched subtitles without any extra information. \n\nOriginal subtitles：\n---\n{text}\n---\n\nReorganized content："

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an assistant who is good at summarizing and organizing texts."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"调用OpenAI API时出错: {e}"

def subtitle_rebuild(subtitle_file):
    """
    主函数
    """
    # --- 在此处配置您的模型参数 ---
    # 您的API密钥，如果不需要可以留空或设为 "dummy-key"
    API_KEY = "sk-CC31kkIMY3uMNhLxyYubMzSGOoFqhq9BH30vkyiTyYHIbkW2" 
    # 您的模型调用网址 (例如: "http://localhost:8000/v1")
    BASE_URL = "http://43.162.122.167:42013/v1" 
    # 您要使用的模型名称
    MODEL_NAME = "gpt-4o-mini"
    # --------------------------------
    
    try:
        print(f"正在解析字幕文件: {subtitle_file}...")
        concatenated_text = parse_srt(subtitle_file)
        
        if not concatenated_text.strip():
            print("错误：未能从文件中提取任何文本内容。")
            return

        print(f"字幕内容拼接完成，正在通过 {BASE_URL} 调用模型 {MODEL_NAME}...")
        reorganized_text = reorganize_with_openai(concatenated_text, API_KEY, BASE_URL, MODEL_NAME)
        
        print("\n--- 模型重组后的内容 ---")
        print(reorganized_text)
        
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")

if __name__ == "__main__":
    # file_path = "/data1/yangyan/workdir/example/kj3Po7zUeyw_en.json"
    file_path = "/data1/yangyan/benchmark/Video-MME-v2/subtitle/001.jsonl"
    
    # 在重组前查看样例
    # inspect_subtitle_file(file_path)
    subtitle_content = get_subtitle(file_path, num_samples=5)
    print("提取的字幕内容:")
    print(subtitle_content)
    
    # subtitle_rebuild(file_path)