from openai import OpenAI
import json
from typing import List, Dict, Tuple
import os
import re
from .QuesPrompt import QuesPrompt
from .QuesClassifier import QuestionClassifier
from .utils import get_subtitle
import traceback


class QuestionDistributor:

    def __init__(self, client, model_name="gemini-3-flash-preview"):
        self.client = client
        self.model_name = model_name
        self.prompt_template = QuesPrompt()
        self.classifier = QuestionClassifier(client=client, model=model_name)  

    def classify_question(self, question):
        result = self.classifier.classify_question(question)
        question_type = result.get("type", "is_new_type")  # default to temporal if not found
        if question_type == "error":
            print(f"Error classifying question: {result.get('error')}")
            question_type = "uniform"  # Fallback to uniform on error
        return question_type
    
    def get_distribution_typely(self, question, subtitle_path=None):
        print("Getting distribution based on question type...")
        question_type = self.classify_question(question)
        prompt_qstype = self.prompt_template.get_prompt(question_type)
        if question_type=='is_new_type' and prompt_qstype == "Unknown problem type, with uniform distribution.":
            print("New question type, return with uniform distribution...")
            # return [0.1] * 5
            return [0.1] * 10  # Uniform distribution for unknown types, or use default of the specific method.
        else:
            print("Specific the pipline with different stratyge for different question...")

        # if subtitles
        subtitle_text = ""
        if subtitle_path and os.path.exists(subtitle_path):
            try:
                # with open(subtitle_path, 'r', encoding='utf-8') as f:
                #     subtitle_text = f.read()
                subtitle_text = get_subtitle(subtitle_path)
            except Exception as e:
                print(f"Error reading subtitle: {e}")

        prompt = f"""
        You are a helpful assistant for video understanding.
        The user has a question about a video: "{question}\n"
        """
        if subtitle_text:
            prompt += f"""
            Here are the subtitles for the video.\n
            NOTE that you do not need to analyze all of the subtitles. Search according to your needs to avoid too much redundant information affecting your normal work.\n
            <Additional subtitles, available as needed>\n
            {subtitle_text}
            </Additional subtitles, available as needed>\n
            Based on the question and the optional subtitles, please predict the temporal distribution of the relevant video segments that might contain the answer.\n
            """
        else:
            prompt += "Please predict the temporal distribution of the relevant video segments that might contain the answer.\n"

        prompt += prompt_qstype
        prompt += f"""
        Please analyze the context and return the result strictly in the following JSON format:
        {{
            "estimated_time_range": ["string", "string"],
            "distribution": [float, float, ..., float]
        }}

        Detailed Instructions:
        1. "estimated_time_range": 
           - If the answer can be explicitly located based on the subtitle timestamps, provide the time range of the corresponding subtitle segment (e.g., ["00:01:20-00:02:10"], ["00:01:20-00:02:10","00:03:30-00:04:10"]).
           - The data format is a list containing strings, which can contain one string or multiple strings. For problems of Explicit Reference type, there is usually only one string corresponding to the timestamp. For problems of the Holistic type, it is likely that each option can correspond to a timestamp string.
           - Only give a time period when the evidence is clear, otherwise it may lead to misleading. If it cannot be determined, set this value to "N/A".

        2. "distribution":
           - Divide the total video duration into 10 equal segments.
           - Provide a list of 10 probability values [p0, p1, ..., p9] corresponding to each segment. Unless absolutely certain, try not to have a complete zero probability.
           - The sum of these 10 probabilities must equal 1.0.
           - The more confident you are, the more you tend towards a unimodal distribution. Conversely, the less confident you are, the more you should tend towards a uniform or multimodal distribution.
           - Use your best judgment to assign higher probabilities to relevant segments. If you are not absolutely confident, I hope you can provide a uniform or multimodal distribution as much as possible.

        Constraint: Return ONLY the raw JSON object. Do not include Markdown blocks (```json) or any other text.
        """# 这里指定范围应该可以不止一个

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            content = response.choices[0].message.content.strip()
            # print(f"Raw response content: {content}")  # Debugging line
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            content = json.loads(content)
            # Extract list using regex or json
            total = sum(content["distribution"])
            if total > 0:
                probs = [p/total for p in content["distribution"]]
                content["distribution"] = probs
            else:
                probs = [0.1] * 10
                content["distribution"] = probs
            print(f"Processed content: {content}")  # Debugging line
            return question_type, content
        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"Error getting distribution: {e}")
            print(f"Traceback:\n{error_msg}")
            
        
        return "uniform", [0.1] * 10 # Fallback to uniform

    def get_distribution_directly(self, question, uniform=False, subtitle_path=None):
        if uniform:
            return [0.1] * 10
        else:
            print("Getting temporal distribution from distributer...")
        
        subtitle_text = ""
        if subtitle_path and os.path.exists(subtitle_path):
            try:
                # with open(subtitle_path, 'r', encoding='utf-8') as f:
                #     subtitle_text = f.read()
                subtitle_text = get_subtitle(subtitle_path)
            except Exception as e:
                print(f"Error reading subtitle: {e}")

        prompt = f"""
        You are a helpful assistant for video understanding.
        The user has a question about a video: "{question}"
        """
        if subtitle_text:
            prompt += f"""
            Here are the subtitles for the video:
            <Additional subtitles, available as needed>
            {subtitle_text}
            </Additional subtitles, available as needed>
            Based on the question and the optional subtitles, please predict the temporal distribution of the relevant video segments that might contain the answer.
            """
        else:
            prompt += "Please predict the temporal distribution of the relevant video segments that might contain the answer.\n"

        prompt += f"""
        Divide the video into 10 equal segments.
        Return a list of 10 probabilities [p0, p1, ..., p9] corresponding to these segments. The sum must be 1.
        Please provide your most confident response. If there is no clear evidence for the question itself, please lean towards a more even response.
        Provide ONLY the list in JSON format, no other text.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            content = response.choices[0].message.content.strip()
            # Extract list using regex or json
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                probs = json.loads(match.group(0))
                if len(probs) == 10:
                    # Normalize just in case
                    total = sum(probs)
                    if total > 0:
                        probs = [p/total for p in probs]
                    else:
                        probs = [0.1] * 10
                    return probs
        except Exception as e:
            print(f"Error getting distribution: {e}")
        
        return [0.1] * 10 # Fallback to uniform
    
class OpenDistributor:

    def __init__(self, client):
        self.client = client
        self.prompt_template = QuesPrompt()
        self.classifier = QuestionClassifier(client=client)  

    def classify_question(self, question):
        result = self.classifier.classify_question(question)
        question_type = result.get("type", "is_new_type")  # default to temporal if not found
        if question_type == "error":
            print(f"Error classifying question: {result.get('error')}")
            question_type = "uniform"  # Fallback to uniform on error
        return question_type
    
    def get_distribution_typely(self, question, subtitle_path=None):
        print("Getting distribution based on question type...")
        question_type = self.classify_question(question)
        prompt_qstype = self.prompt_template.get_prompt(question_type)
        if question_type=='is_new_type' and prompt_qstype == "Unknown problem type, with uniform distribution.":
            print("New question type, return with uniform distribution...")
            return "uniform", [0.1] * 10  # Uniform distribution for unknown types, or use default of the specific method.
        else:
            print("Specific the pipline with different stratyge for different question...")

        # if subtitles
        subtitle_text = ""
        if subtitle_path and os.path.exists(subtitle_path):
            try:
                # with open(subtitle_path, 'r', encoding='utf-8') as f:
                #     subtitle_text = f.read()
                subtitle_text = get_subtitle(subtitle_path)
            except Exception as e:
                print(f"Error reading subtitle: {e}")

        prompt = f"""
        You are a helpful assistant for video understanding.
        The user has a question about a video: "{question}\n"
        """
        if subtitle_text:
            prompt += f"""
            Here are the subtitles for the video.\n
            NOTE that you do not need to analyze all of the subtitles. Search according to your needs to avoid too much redundant information affecting your normal work.\n
            <Additional subtitles, available as needed>\n
            {subtitle_text}
            </Additional subtitles, available as needed>\n
            Based on the question and the optional subtitles, please predict the temporal distribution of the relevant video segments that might contain the answer.\n
            """
        else:
            prompt += "Please predict the temporal distribution of the relevant video segments that might contain the answer.\n"

        prompt += prompt_qstype
        prompt += f"""
        Please analyze the context and return the result strictly in the following JSON format:
        {{
            "estimated_time_range": ["string", "string"],
            "distribution": [float, float, ..., float]
        }}

        Detailed Instructions:
        1. "estimated_time_range": 
           - If the answer can be explicitly located based on the subtitle timestamps, provide the time range of the corresponding subtitle segment (e.g., ["00:01:20-00:02:10"], ["00:01:20-00:02:10","00:03:30-00:04:10"]).
           - The data format is a list containing strings, which can contain one string or multiple strings. For problems of Explicit Reference type, there is usually only one string corresponding to the timestamp. For problems of the Holistic type, it is likely that each option can correspond to a timestamp string.
           - Only give a time period when the evidence is clear, otherwise it may lead to misleading. If it cannot be determined, set this value to "N/A".

        2. "distribution":
           - Divide the total video duration into 10 equal segments.
           - Provide a list of 10 probability values [p0, p1, ..., p9] corresponding to each segment.
           - The sum of these 10 probabilities must equal 1.0.
           - Use your best judgment to assign higher probabilities to relevant segments. If the evidence is ambiguous, lean towards a more uniform distribution.

        Constraint: Return ONLY the raw JSON object. Do not include Markdown blocks (```json) or any other text.
        """# 这里指定范围应该可以不止一个

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            content = response.choices[0].message.content.strip()
            content = json.loads(content)
            # Extract list using regex or json
            total = sum(content["distribution"])
            if total > 0:
                probs = [p/total for p in content["distribution"]]
                content["distribution"] = probs
            else:
                probs = [0.1] * 10
                content["distribution"] = probs
            return question_type, content
        except Exception as e:
            print(f"Error getting distribution: {e}")
        
        return "uniform", [0.1] * 10 # Fallback to uniform

    def get_distribution_directly(self, question, uniform=False, subtitle_path=None):
        if uniform:
            return [0.1] * 10
        else:
            print("Getting temporal distribution from distributer...")
        
        subtitle_text = ""
        if subtitle_path and os.path.exists(subtitle_path):
            try:
                # with open(subtitle_path, 'r', encoding='utf-8') as f:
                #     subtitle_text = f.read()
                subtitle_text = get_subtitle(subtitle_path)
            except Exception as e:
                print(f"Error reading subtitle: {e}")

        prompt = f"""
        You are a helpful assistant for video understanding.
        The user has a question about a video: "{question}"
        """
        if subtitle_text:
            prompt += f"""
            Here are the subtitles for the video:
            <Additional subtitles, available as needed>
            {subtitle_text}
            </Additional subtitles, available as needed>
            Based on the question and the optional subtitles, please predict the temporal distribution of the relevant video segments that might contain the answer.
            """
        else:
            prompt += "Please predict the temporal distribution of the relevant video segments that might contain the answer.\n"

        prompt += f"""
        Divide the video into 10 equal segments.
        Return a list of 10 probabilities [p0, p1, ..., p9] corresponding to these segments. The sum must be 1.
        Please provide your most confident response. If there is no clear evidence for the question itself, please lean towards a more even response.
        Provide ONLY the list in JSON format, no other text.
        """
        try:
            # response = self.client.chat.completions.create(
            #     model=self.model_name,
            #     messages=[{"role": "user", "content": prompt}],
            #     temperature=0.2
            # )
            response = self.client.get_text_answer(prompt)
            content = response.choices[0].message.content.strip()
            # Extract list using regex or json
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                probs = json.loads(match.group(0))
                if len(probs) == 10:
                    # Normalize just in case
                    total = sum(probs)
                    if total > 0:
                        probs = [p/total for p in probs]
                    else:
                        probs = [0.1] * 10
                    return probs
        except Exception as e:
            print(f"Error getting distribution: {e}")
        
        return [0.1] * 10 # Fallback to uniform