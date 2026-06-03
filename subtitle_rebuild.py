import argparse
import os
import openai

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

def reorganize_with_openai(text, api_key, base_url, model_name):
    """
    使用OpenAI格式的API重新组织文本内容。
    """
    # print("原始字幕拼接", text)
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
    API_KEY = "$API_KEY" 
    # 您的模型调用网址 (例如: "http://localhost:8000/v1")
    BASE_URL = "$BASE_URL/v1" 
    # 您要使用的模型名称
    MODEL_NAME = "gpt-4o-mini"
    # --------------------------------
    
    try:
        # print(f"正在解析字幕文件: {subtitle_file}...")
        concatenated_text = parse_srt(subtitle_file)
        
        if not concatenated_text.strip():
            print("错误：未能从文件中提取任何文本内容。")
            return

        # print(f"字幕内容拼接完成，正在通过 {BASE_URL} 调用模型 {MODEL_NAME}...")
        reorganized_text = reorganize_with_openai(concatenated_text, API_KEY, BASE_URL, MODEL_NAME)
        
        # print("\n--- 模型重组后的内容 ---")
        # print(reorganized_text)
        return reorganized_text
        
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")

if __name__ == "__main__":
    subtitle_rebuild("/data1/yangyan/workdir/kj3Po7zUeyw_en.json")