import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def extract_and_translate_questions(input_file, output_file_en, output_file_zh):
    """
    从一个JSON文件中提取所有'question'字段，翻译成中文，并分别保存英文和中文版本。

    Args:
        input_file (str): 输入的JSON文件路径。
        output_file_en (str): 输出的英文问题JSON文件路径。
        output_file_zh (str): 输出的中文问题JSON文件路径。
    """
    try:
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取所有 'question' 字段的值
        questions_en = [item.get('question') for item in data if 'question' in item]
        
        print(f"成功从 '{input_file}' 中提取了 {len(questions_en)} 个问题。")
        print("正在加载翻译模型...")

        # 加载翻译模型 (使用Helsinki-NLP的英译中模型)
        model_name = "Helsinki-NLP/opus-mt-en-zh"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        print("模型加载完成，开始翻译...")

        # 翻译问题
        questions_zh = []
        for i, question in enumerate(questions_en):
            # 对每个问题进行翻译
            inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model.generate(**inputs, max_length=512)
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            questions_zh.append(translated)
            
            # 显示进度
            if (i + 1) % 10 == 0:
                print(f"已翻译 {i + 1}/{len(questions_en)} 个问题")

        # 保存英文版本
        with open(output_file_en, 'w', encoding='utf-8') as f:
            json.dump(questions_en, f, indent=4, ensure_ascii=False)
        
        # 保存中文版本
        with open(output_file_zh, 'w', encoding='utf-8') as f:
            json.dump(questions_zh, f, indent=4, ensure_ascii=False)
        
        print(f"\n翻译完成！")
        print(f"英文版本已保存到：'{output_file_en}'")
        print(f"中文版本已保存到：'{output_file_zh}'")

    except FileNotFoundError:
        print(f"错误：输入文件 '{input_file}' 未找到。")
    except json.JSONDecodeError:
        print(f"错误：无法解析 '{input_file}'。请检查文件是否为有效的JSON格式。")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    # 定义输入和输出文件路径
    input_json_path = 'lvbench_subset_50.json'
    output_json_path_en = 'questions_english.json'
    output_json_path_zh = 'questions_chinese.json'
    
    # 调用函数执行提取和翻译操作
    extract_and_translate_questions(input_json_path, output_json_path_en, output_json_path_zh)