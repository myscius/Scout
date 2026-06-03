import os
import re
import ast
import json

def parse_log_distribution(log_file, output_file):
    """
    从日志文件中提取 Distribution 信息并保存为 JSON。
    """
    if not os.path.exists(log_file):
        print(f"Error: Log file '{log_file}' not found.")
        return

    # 用于存储最终结果的字典
    collected_data = {}
    
    # 状态变量
    current_video = None
    current_question = None
    current_qstype = None # 新增：用于存储当前问题的类型

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # 1. 提取视频路径
                # 示例: ... - INFO - Processing video: /path/to/video.mp4
                video_match = re.search(r'Processing video:\s+(.+)', line)
                if video_match:
                    current_video = video_match.group(1).strip()
                    # 每次遇到新的视频处理流，重置状态
                    current_question = None 
                    current_qstype = None
                    continue

                # 2. 提取问题内容
                # 示例: ... - INFO - Question: What is...?
                question_match = re.search(r' - INFO - Question:\s+(.+)', line)
                if question_match:
                    current_question = question_match.group(1).strip()
                    continue

                # 3. 新增：提取问题类型
                # 示例: ... - INFO - Question type detected: Counting/Ordinal
                type_match = re.search(r' - INFO - Question type detected:\s+(.+)', line)
                if type_match:
                    current_qstype = type_match.group(1).strip()
                    continue

                # 4. 提取 Distribution 数据
                # 示例: ... - INFO - Distribution: {'estimated_time_range': ...}
                dist_match = re.search(r' - INFO - Distribution:\s+(.+)', line)
                if dist_match:
                    # 只有当上下文（视频和问题）都存在时才进行提取
                    if current_video and current_question:
                        dist_str = dist_match.group(1).strip()
                        try:
                            # 使用 ast.literal_eval 安全地解析 Python 风格的字典字符串
                            distribution_dict = ast.literal_eval(dist_str)
                            
                            # 检查规则：如果 distribution 中不存在 estimated_time_range 或者 为空，则认为是失败的，不统计
                            if 'estimated_time_range' not in distribution_dict or not distribution_dict['estimated_time_range']:
                                continue

                            # 构造 Key
                            video_basename = os.path.basename(current_video)
                            query_clean = current_question.split("\nOptions")[0]
                            cache_key = f'{video_basename}#{query_clean}'
                            
                            # 构造符合要求的 Value 结构
                            # 包含 qstype 和 distribution (distribution内含 estimated_time_range 和 distribution 列表)
                            entry_data = {
                                "qstype": current_qstype,
                                "distribution": distribution_dict
                            }
                            
                            # 存入字典
                            collected_data[cache_key] = entry_data
                            
                        except Exception as e:
                            print(f"[Warning] Failed to parse distribution for key: {video_basename}#{current_question}. Error: {e}")
                    else:
                        pass

        # 5. 保存为 JSON 文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(collected_data, f, indent=4, ensure_ascii=False)
        
        print(f"Successfully processed log.")
        print(f"Total entries collected: {len(collected_data)}")
        print(f"Saved to: {output_file}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # 配置输入日志文件和输出 JSON 文件路径
    log_filename = 'LongVideoBench_qwen3.5_cache_frame20.log'
    output_filename = 'collect_dis.json'
    
    parse_log_distribution(log_filename, output_filename)