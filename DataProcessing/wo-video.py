import json
import os
import cv2
import base64
import numpy as np
from openai import OpenAI
from tqdm import tqdm

# 初始化 OpenAI 客户端
# 请确保设置了环境变量 OPENAI_API_KEY，或者在这里直接赋值 (不推荐直接硬编码)
client = OpenAI(
    base_url="http://43.131.235.107:45101/v1",
    api_key="sk-CC31kkIMY3uMNhLxyYubMzSGOoFqhq9BH30vkyiTyYHIbkW2"
)

def load_data(filepath, limit=None):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:limit]

def extract_frames(video_path, num_frames=10):
    """均匀采样视频帧并转换为base64编码"""
    if not video_path or not os.path.exists(video_path):
        return None
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return None
        
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # 压缩图片以适应token限制并加快传输
            _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            b64_frame = base64.b64encode(buffer).decode("utf-8")
            frames.append(b64_frame)
            
    cap.release()
    return frames

def analyze_content(item, mode):
    question = item.get('question', '')
    candidates = item.get('candidates', [])
    
    candidates_text = ""
    if candidates:
        candidates_text = "\n选项:\n" + "\n".join([f"{i}. {c}" for i, c in enumerate(candidates)])
    
    # 处理字幕路径
    subtitle_filename = item.get('subtitle_path', '')
    subtitles = ""
    if subtitle_filename:
        sub_full_path = os.path.join("/data1/yangyan/benchmark/LongVideoBench/subtitles", subtitle_filename)
        if os.path.exists(sub_full_path):
            try:
                with open(sub_full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    subtitles = f.read()
            except Exception:
                subtitles = "[读取字幕失败]"
    
    # 处理视频路径
    video_filename = item.get('video_path', '') 
    video_full_path = os.path.join("/data1/yangyan/benchmark/LongVideoBench/videos", video_filename)
    
    messages = []
    
    # 基础规则部分
    base_rules = (
        "请遵循以下规则：\n"
        "1. 进行深入的逻辑推理，寻找时间线索、事件顺序或特定场景描述。\n"
        "2. 不要仅仅依赖简单的关键词匹配。\n"
        "3. 如果确实没有足够的证据或信息来推断位置，必须诚实地回答无法判断，绝对不要伪造证据。\n"
        "4. 输出格式要求：首先输出精炼的【推理过程】，详细描述你的分析；最后输出一行【结论】，仅包含'是'或'否'。\n"
    )

    system_prompt = ""

    if mode == "q_video":
        system_prompt = (
            "你是一个视频视觉分析专家。你的任务是结合给定的问题、选项和视频采样帧，判断是否能够定位到对应视频片段的大致位置。\n"
            "你需要仔细观察图像中的场景、人物动作、物体状态以及光影变化等视觉信息，分析其是否是问题对应的视频帧或者相关可以辅助问答的帧。\n"
            + base_rules
        )
        frames = extract_frames(video_full_path)
        if not frames:
            return f"Error: 无法读取视频文件或提取帧: {video_full_path}"
            
        content_payload = [{"type": "text", "text": f"问题: {question}{candidates_text}\n以下是视频的均匀采样帧："}]
        for frame in frames:
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
            })
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content_payload})
        
    elif mode == "q_sub":
        system_prompt = (
            "你是一个视频字幕分析专家。你的任务是结合给定的问题、选项和视频字幕文本，判断是否能够定位到对应视频片段的大致位置。\n"
            "你需要分析字幕上下文逻辑，寻找是否存在与问题或选项相关的线索。\n"
            "每一段字幕都有其时间戳，如果你可以找到和问题相关的字幕，就可以利用这些时间戳进行大致的定位，可以是多个相关的片段。\n"
            + base_rules
        )
        # 截断过长的字幕以防止超出token限制
        if len(subtitles) > 50000: 
            subtitles = subtitles[:50000] + "...(truncated)"
        prompt_content = f"问题: {question}{candidates_text}\n字幕: {subtitles}"
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_content})
        
    elif mode == "q_only":
        system_prompt = (
            "你是一个逻辑推理专家。你的任务是仅根据给定的问题和选项本身，判断该问题是否包含足够的信息（如明确的时间点、独特的事件序列、极具辨识度的单一场景描述等），使得在观看完整视频之前，理论上可以判断对应视频帧的大致位置。\n"
            "可以将视频顺序地分为几个分段，判断问题是否有倾向性，可以特定地倾向于在某几个分段中（视频开头、中间或者结尾）。\n"
            + base_rules
        )
        prompt_content = f"问题: {question}{candidates_text}"
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_content})

    try:
        response = client.chat.completions.create(
            model="gemini-3-flash-preview", 
            messages=messages,
            temperature=0.2 
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def main(json_path = '/home/yangyan/FastGrouding/lvbench_subset_50.json',limit=None):
    json_path = json_path
    
    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"文件不存在: {json_path}")
        return

    data = load_data(json_path, limit=limit)
    
    modes = ["q_video", "q_sub", "q_only"]
    stats = {mode: {"是": 0, "否": 0, "其他": 0} for mode in modes}
    
    # 定义结果文件路径
    output_files = {
        "q_video": "result_q_video.json",
        "q_sub": "result_q_sub.json",
        "q_only": "result_q_only.json"
    }
    
    # 缓存结果
    results_cache = {mode: [] for mode in modes}

    print(f"开始处理 {len(data)} 条数据...")
    
    for i, item in enumerate(tqdm(data, desc="总体进度")):
        tqdm.write(f"\n--- 处理第 {i+1} 条数据 ---")
        for mode in modes:
            tqdm.write(f"[模式: {mode}]")
            result = analyze_content(item, mode)
            # 简略打印结果，避免刷屏
            tqdm.write(result[:200] + "..." if len(result) > 200 else result)
            
            # 构造结果对象
            result_entry = {
                "index": i + 1,
                "question": item.get('question', ''),
                "candidates": item.get('candidates', []),
                "video_path": item.get('video_path', ''),
                "subtitle_path": item.get('subtitle_path', ''),
                "analysis_result": result
            }

            # 添加到缓存
            results_cache[mode].append(result_entry)

            # 统计结果
            lines = result.strip().split('\n')
            last_line = lines[-1].strip() if lines else ""
            if "是" in last_line:
                stats[mode]["是"] += 1
            elif "否" in last_line:
                stats[mode]["否"] += 1
            else:
                stats[mode]["其他"] += 1
            
            # 计算并输出当前精度 (假设'是'为正确预测)
            current_total = stats[mode]["是"] + stats[mode]["否"] + stats[mode]["其他"]
            current_acc = stats[mode]["是"] / current_total if current_total > 0 else 0.0
            tqdm.write(f"Mode: {mode} | 当前 '是' 比例 (精度): {current_acc:.2%} ({stats[mode]['是']}/{current_total})")
                
        tqdm.write("="*50)

    # 处理完成后统一保存
    print("\n正在保存结果到文件...")
    for mode, filepath in output_files.items():
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_cache[mode], f, ensure_ascii=False, indent=4)
            print(f"已保存: {filepath}")
        except Exception as e:
            print(f"保存文件 {filepath} 失败: {e}")

    print("\n" + "="*20 + " 统计结果 " + "="*20)
    for mode, counts in stats.items():
        print(f"模式: {mode}")
        print(f"  是: {counts['是']}")
        print(f"  否: {counts['否']}")
        print(f"  其他/错误: {counts['其他']}")
        print("-" * 20)

if __name__ == "__main__":
    json_path = '/home/yangyan/FastGrouding/lvbench_subset_200.json'
    main(json_path = json_path, limit=100)
