from openai import OpenAI
import json
from typing import List, Dict, Tuple
import os
import re

class QuestionClassifier:
    """视频理解问题定位类型分类器"""
    
    def __init__(self, client=None, api_key: str = None, base_url: str = None, model: str = "gpt-4"):
        """
        初始化分类器
        
        Args:
            api_key: OpenAI API密钥，如果不提供则从环境变量读取
            base_url: API调用地址
            model: 使用的模型名称，默认为gpt-4
        """
        if client:
            self.client = client
            self.model = model
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.model = model
            
            # 使用新版 OpenAI API
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=base_url
            )
        
        # 定义已知类型（按优先级排序）
        self.known_types = {
            "ExplicitReference": "Explicitly mentions reference markers, e.g., 'when someone mentions/when subtitles show xxx'",
            "Counting/Ordinal": "Involves order or counting, e.g., 'the first/x-th person/object to appear'",
            "Descriptive": "Locates via feature description, e.g., 'what is the person in black clothes doing in the white room'",
            "Holistic": "Requires synthesizing global information, e.g., 'which of the following descriptions is correct'"
        }
        
        # 存储新发现的类型
        self.new_types = []
    
    def _create_classification_prompt(self, question: str) -> str:
        """创建分类提示词"""
        prompt = f"""You are an expert specializing in analyzing grounding information in video understanding questions. Please analyze which grounding type the following question belongs to.

        Known Grounding Types (in descending order of priority):
        1. Explicit Reference: The question explicitly mentions time reference markers, e.g., 'when someone mentions xxx', 'when subtitles show xxx', etc.
        2. Holistic: Requires synthesizing global information, e.g., 'which of the following is correct'.
        3. Counting/Ordinal: Involves order or counting for grounding, e.g., 'the first/x-th person/object to appear'.
        4. Descriptive: Locates via feature description, e.g., 'the person in black clothes in the white room'.
        

        Classification Rules:
        - If the question contains features of multiple types, select the type with the highest priority.
        - If the question does not belong to any of the above types, please return as a new type.

        Question to analyze:
        "{question}"

        Please return the result in JSON format:
        {{
            "type": "Classification Type",
            "confidence": "Confidence (High/Medium/Low)",
            "reason": "Reason for classification",
            "is_new_type": false,
            "new_type_suggestion": {{
            "name": "New Type Name (if applicable)",
            "description": "Description of the new type"
            }}
        }}
        Do not include Markdown blocks (```json) or any other text.
        """
        return prompt
    
    def classify_question(self, question: str) -> Dict:
        """
        对单个问题进行分类
        
        Args:
            question: 待分类的问题文本
            
        Returns:
            包含分类结果的字典
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional video question analysis expert, specializing in identifying the type of grounding information in questions."},
                    {"role": "user", "content": self._create_classification_prompt(question)}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            # re.DOTALL 确保 . 可以匹配换行符
            raw_content = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            
            result = json.loads(json_match.group(0))
            result["original_question"] = question
            
            # 如果发现新类型，记录下来
            if result.get("is_new_type") and result.get("new_type_suggestion"):
                self.new_types.append({
                    "question": question,
                    "suggestion": result["new_type_suggestion"]
                })
            
            return result
            
        except Exception as e:
            return {
                "original_question": question,
                "type": "error",
                "error": str(e)
            }
    
    def classify_questions(self, questions: List[str]) -> List[Dict]:
        """
        批量分类问题
        
        Args:
            questions: 问题列表
            
        Returns:
            分类结果列表
        """
        results = []
        for i, question in enumerate(questions, 1):
            print(f"Processing question {i}/{len(questions)}...")
            result = self.classify_question(question)
            results.append(result)
        
        return results
    
    def generate_report(self, results: List[Dict]) -> str:
        """
        生成分类报告
        
        Args:
            results: 分类结果列表
            
        Returns:
            报告文本
        """
        # 统计各类型数量
        type_counts = {}
        for result in results:
            type_name = result.get("type", "未知")
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        report = "=" * 60 + "\n"
        report += "视频问题定位类型分类报告\n"
        report += "=" * 60 + "\n\n"
        
        report += f"总问题数：{len(results)}\n\n"
        
        report += "类型分布：\n"
        for type_name, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(results)) * 100
            report += f"  - {type_name}: {count} ({percentage:.1f}%)\n"
        
        if self.new_types:
            report += f"\n发现新类型：{len(self.new_types)} 个\n"
            report += "-" * 60 + "\n"
            for i, new_type in enumerate(self.new_types, 1):
                report += f"\n新类型 {i}:\n"
                report += f"  名称：{new_type['suggestion'].get('name', '未命名')}\n"
                report += f"  描述：{new_type['suggestion'].get('description', '无描述')}\n"
                report += f"  触发问题：{new_type['question']}\n"
        
        return report
    
    def save_results(self, results: List[Dict], output_file: str):
        """保存结果到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到：{output_file}")


def load_questions_from_file(file_path: str) -> List[str]:
    """
    从JSON文件加载问题列表
    
    Args:
        file_path: 问题文件路径
        
    Returns:
        问题列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 文件结构是一个字符串列表
    if isinstance(data, list):
        questions = [item for item in data if isinstance(item, str)]
        return questions
    
    return []


def main():
    """主函数示例"""
    # API配置
    API_KEY = "$API_KEY"
    BASE_URL = "$BASE_URL/v1"
    # QUESTIONS_FILE = "/data1/yangyan/workdir/questions_lvbench_subset_50_en.json"
    QUESTIONS_FILE = "/data1/yangyan/workdir/questions_vmme_en.json"
    OUTPUT_FILE = "/data1/yangyan/workdir/class_vmme_results.json"
    REPORT_FILE = "/data1/yangyan/workdir/class_vmme_report.txt"
    
    # 初始化分类器
    print("初始化分类器...")
    classifier = QuestionClassifier(
        client=None,
        api_key=API_KEY,
        base_url=BASE_URL,
        model="gpt-4o-mini"
    )
    
    # 加载问题
    print(f"从文件加载问题: {QUESTIONS_FILE}")
    try:
        questions = load_questions_from_file(QUESTIONS_FILE)
        print(f"成功加载 {len(questions)} 个问题\n")
    except Exception as e:
        print(f"加载问题文件失败: {e}")
        return
    
    if not questions:
        print("未找到问题，请检查文件格式")
        return
    
    # 执行分类
    print("开始分类问题...\n")
    results = classifier.classify_questions(questions)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("分类结果预览（前5个）：")
    print("=" * 60)
    for result in results[:5]:
        print(f"\n问题：{result['original_question']}")
        print(f"类型：{result.get('type', '未知')}")
        print(f"置信度：{result.get('confidence', '未知')}")
        print(f"理由：{result.get('reason', '无')}")
    
    # 生成报告
    report = classifier.generate_report(results)
    print("\n" + report)
    
    # 保存结果
    classifier.save_results(results, OUTPUT_FILE)
    
    # 保存报告
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"报告已保存到：{REPORT_FILE}")


if __name__ == "__main__":
    main()
