import json
import os

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

def reorganize_json_subtitles(input_file_path, output_file_path=None):
    """
    重新组织JSON格式的字幕文件，按完整句子合并字幕片段。
    同时生成一个带时间戳的、用换行符分隔的完整字符串。

    Args:
        input_file_path (str): 输入的JSON字幕文件路径。
        output_file_path (str): 输出的重组后JSON文件的路径。
    
    Returns:
        str: 格式化后的完整字幕字符串。
    """
    # 1. 读取并解析JSON文件
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            subtitles = json.load(f)
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 '{input_file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"错误: 文件 '{input_file_path}' 不是有效的JSON格式。")
        return None

    if not subtitles:
        print("警告: 输入文件为空。")
        return ""

    # 2. 格式检测与数据归一化
    normalized_subtitles = []
    first_segment = subtitles[0]
    
    # 检测格式1: {"start": "...", "end": "...", "line": "..."}
    if 'start' in first_segment and 'line' in first_segment:
        print("检测到格式: {'start', 'end', 'line'}")
        # 时间戳已经是字符串格式，直接使用
        normalized_subtitles = subtitles
    # 检测格式2: {"timestamp": [start, end], "text": "..."}
    elif 'timestamp' in first_segment and 'text' in first_segment:
        print("检测到格式: {'timestamp', 'text'}")
        for segment in subtitles:
            normalized_subtitles.append({
                # 将浮点数秒转换为 HH:MM:SS.ms 格式
                'start': format_time(segment['timestamp'][0]),
                'end': format_time(segment['timestamp'][1]),
                'line': segment['text']
            })
    else:
        print("错误: 未知的JSON字幕格式。")
        return None

    # --- 从这里开始，核心逻辑使用归一化后的 `normalized_subtitles` ---

    reorganized_subtitles = []
    current_sentence = ""
    sentence_start_time = None

    for segment in normalized_subtitles:
        # 3. 将字幕中的\n替换为空格
        cleaned_line = segment['line'].replace('\n', ' ').strip()

        if not current_sentence:
            # 开始一个新句子
            sentence_start_time = segment['start']
            current_sentence = cleaned_line
        else:
            # 拼接句子
            current_sentence += " " + cleaned_line

        # 4. 检查句子是否以句号结尾
        if current_sentence.endswith('.'):
            # 句子完整，创建新的合并后的片段
            new_segment = {
                "start": sentence_start_time,
                "end": segment['end'],  # 使用当前片段的结束时间
                "line": current_sentence
            }
            reorganized_subtitles.append(new_segment)
            
            # 重置，准备下一个新句子
            current_sentence = ""
            sentence_start_time = None

    # 处理文件末尾可能剩余的、不以句号结尾的最后一句
    if current_sentence:
        last_segment_end_time = normalized_subtitles[-1]['end']
        new_segment = {
            "start": sentence_start_time,
            "end": last_segment_end_time,
            "line": current_sentence
        }
        reorganized_subtitles.append(new_segment)

    # # 将重组后的内容写入新的JSON文件
    # with open(output_file_path, 'w', encoding='utf-8') as f:
    #     json.dump(reorganized_subtitles, f, indent=2, ensure_ascii=False)
    # print(f"字幕已成功重组并保存到: '{output_file_path}'")

    # --- 新增功能：生成带时间戳的拼接字符串 ---
    formatted_lines = []
    for segment in reorganized_subtitles:
        # 格式：[开始时间 - 结束时间] 字幕内容
        formatted_line = f"[{segment['start']} - {segment['end']}] {segment['line']}"
        formatted_lines.append(formatted_line)
    
    if len(formatted_lines)==1:
        print("警告: 重组后的字幕内容过少，启用ai重组。")
        # 如果只有一个片段，直接返回文本内容，不带时间戳
        return reorganized_subtitles[0]['line']
    # 用换行符 \n 分隔
    final_string = "\n".join(formatted_lines)
    
    return final_string


if __name__ == "__main__":
    # --- 配置输入和输出文件 ---
    INPUT_JSON_FILE = "/data1/yangyan/workdir/kj3Po7zUeyw_en.json"
    
    # 自动生成输出文件名
    base, ext = os.path.splitext(INPUT_JSON_FILE)
    OUTPUT_JSON_FILE = f"{base}_reorganized.json"
    OUTPUT_TXT_FILE = f"{base}_reorganized.txt"
    # --------------------------

    # 执行重组并获取格式化后的字符串
    reorganized_string = reorganize_json_subtitles(INPUT_JSON_FILE, OUTPUT_JSON_FILE)

    if reorganized_string is not None:
        # 将字符串保存到 .txt 文件
        with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as f:
            f.write(reorganized_string)
        print(f"拼接后的字符串已保存到: '{OUTPUT_TXT_FILE}'")

        # 打印拼接后的字符串到控制台
        print("\n--- 拼接后的字幕字符串 ---")
        print(reorganized_string)