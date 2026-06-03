import os
import json
import numpy as np
import torch
from decord import VideoReader, cpu
from openai import OpenAI
from all_model_agent import BaseAgent
import re
from tqdm import tqdm
from QuesGrouding import QuestionClassifier, QuestionDistributor

import random
from scipy import interpolate

import logging
import sys
# 配置日志系统
log_tag = 'LongVideoBench_gemma4-31b'  # 可以根据需要修改标签
Nf = 20
log_file = f'{log_tag}_frame{Nf}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),  # 输出到文件
        logging.StreamHandler(sys.stdout)  # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)

import httpx

# 自定义 HTTP 客户端，强制不使用任何代理
custom_http_client = httpx.Client(
    proxy=None,  # 禁用代理
    trust_env=False, # 忽略操作系统的 HTTP_PROXY 等环境变量
    timeout=300.0  # 虽然去掉了代理，但依然建议给一个较长的超时时间
)

class SelfAgent(BaseAgent):
    def __init__(self, model_name='Qwen/Qwen2.5-VL-7B-Instruct'):
        super().__init__(model_name)
        self.client = OpenAI(
            # base_url="http://automl.aiserverai.online/v1",
            # api_key="sk-GP9H3FNTMcu8asZ3xU0gmJyDnKmKgCmLVByMLdhHHde0Gw2f"
            base_url = "https://integrate.api.nvidia.com/v1",
            api_key = "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",
            # base_url="http://127.0.0.1:11434/v1",
            # api_key="sk-CC31kkIMY3uMNhLxyYubMzSGOoFqhq9BH30vkyiTyYHIbkW2"
            http_client=custom_http_client
        )
        # gemini-3-flash-preview
        self.model_name = "google/gemma-4-31b-it" # Qwen/Qwen3.5-9B
        # self.model_name = "deepseek-ai/DeepSeek-V3" 
        # self.model_name = "qwen3:32b" 
        self.distributor = QuestionDistributor(client=self.client, model_name=self.model_name)
        self.qstype = None
        self.type_acc = {
            "explicitreference": [0,0],
            "counting/ordinal": [0,0],
            "descriptive": [0,0],
            "holistic": [0,0],
            "uniform":[0,0],
            "direct":[0,0]
        }
        self.dis_cache = {}
        self.id = 0

    def load_cache(self, cache_path):
        if cache_path is None:
            return
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self.dis_cache = json.load(f)
            print(f"Loaded distribution cache from {cache_path}, {len(self.dis_cache)} entries.")
        else:
            print(f"No existing cache found at {cache_path}. Starting fresh.")
    def save_cache(self, cache_path="./test_distribution_cache.json"):
        if cache_path is None:
            return
        with open(cache_path, 'w') as f:
            json.dump(self.dis_cache, f)
        print(f"Saved distribution cache to {cache_path}, {len(self.dis_cache)} entries.")


    def sample_frames(self, video_path, estimated_time_range=None, distribution=None, num_frames=20):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        # 固定采帧间隔
        interval = 120.0  # seconds
        # 获取视频帧率，用于将时间转换为帧索引
        fps = vr.get_avg_fps()
        
        if total_frames < num_frames:
            return sorted(list(range(total_frames)))
            
        # -----------------------------
        # 1. 处理 Explicit Time Range 采样
        # -----------------------------
        explicit_samples = []
        
        # 规范化输入：确保 time_ranges 是一个列表
        time_ranges = []
        if estimated_time_range:
            if isinstance(estimated_time_range, list):
                time_ranges = estimated_time_range
            elif isinstance(estimated_time_range, str):
                time_ranges = [estimated_time_range]
        
        # 定义内部辅助函数：时间字符串 "00:01:20" 转秒
        def time_str_to_sec(t_str):
            try:
                parts = t_str.strip().split(':')
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            except:
                return None

        # 遍历列表中的每个时间范围字符串
        for t_range in time_ranges:
            if isinstance(t_range, str) and t_range.lower() != "n/a":
                try:
                    if '-' not in t_range:
                        continue
                        
                    start_str, end_str = t_range.split('-')
                    start_sec = time_str_to_sec(start_str)
                    end_sec = time_str_to_sec(end_str)
                    
                    if start_sec is None or end_sec is None:
                        continue

                    duration = end_sec - start_sec
                    

                    if duration > 0:
                        time_points = []
                        if duration <= interval:
                            # 不足10s，取中间一帧
                            time_points.append(start_sec + duration / 2)
                        else:
                            # 每隔10s分配一帧
                            curr = start_sec
                            while curr <= end_sec:
                                time_points.append(curr)
                                curr += interval
                        
                        # 将时间点转换为帧索引
                        for t in time_points:
                            f_idx = int(t * fps)
                            # 确保不越界
                            if 0 <= f_idx < total_frames:
                                explicit_samples.append(f_idx)
                except Exception as e:
                    print(f"Error parsing estimated_time_range '{t_range}': {e}")
                    continue
        
        # 去重并排序
        explicit_samples = sorted(list(set(explicit_samples)))

        # -----------------------------
        # 2. 根据 Distribution 分配剩余帧
        # -----------------------------
        # 计算剩余可用的配额
        remaining_num_frames = num_frames - len(explicit_samples)
        dist_samples = []

        # 只有当还有剩余配额且提供了分布时才执行分布采样
        if remaining_num_frames > 0 and distribution:
            segment_size = total_frames / len(distribution)
            
            # 使用 remaining_num_frames 而不是 num_frames 来计算每一段应分配的帧数
            counts = [int(p * remaining_num_frames) for p in distribution]
            diff = remaining_num_frames - sum(counts)
            
            # Distribute remaining frames to segments with highest probabilities
            if diff > 0:
                sorted_indices = np.argsort(distribution)[::-1]
                for i in range(diff):
                    counts[sorted_indices[i]] += 1
            
            for i in range(len(distribution)):
                count = counts[i]
                if count > 0:
                    start = int(i * segment_size)
                    end = int((i + 1) * segment_size)
                    end = min(end, total_frames)
                    if start >= end:
                        continue
                        
                    # Uniform sample within segment
                    if count == 1:
                        seg_indices = [start + (end - start) // 2]
                    else:
                        seg_indices = np.linspace(start, end - 1, count, dtype=int).tolist()
                    dist_samples.extend(seg_indices)

        # -----------------------------
        # 3. 合并、填充与截断
        # -----------------------------
        # 合并两部分采样结果
        samples = sorted(list(set(explicit_samples + dist_samples)))
        
        # Fill up if needed (如果两部分加起来还不够 num_frames，随机填充)
        while len(samples) < num_frames:
            missing = num_frames - len(samples)
            possible_indices = list(set(range(total_frames)) - set(samples))
            if not possible_indices:
                break
            new_samples = np.random.choice(possible_indices, min(missing, len(possible_indices)), replace=False)
            samples.extend(new_samples)
            samples = sorted(list(set(samples)))
            
        # Trim if needed (如果 Explicit 采样过多导致总数超过 num_frames)
        if len(samples) > num_frames:
             samples = samples[:num_frames]

        return samples

    def weighted_sampling(self,dist_list, sample_max, sample_num):
        """
        加权采样函数（封装完整版，支持外推+插值+缩放+采样）
        
        参数：
            dist_list (list): 长度为10的分布控制点数组 [w1,w2,...w10]
            sample_max (int): 采样范围最大值，范围为 1 ~ sample_max
            sample_num (int): 需要采样的数量
        
        返回：
            list: 加权采样后的索引列表
        """
        sample_max = int(sample_max/10)  # 缩放sample_max 较少计算
        # --------------------------
        # 1. 基础参数构建
        # --------------------------
        n_control = len(dist_list)  # 控制点数量（10）
        x_control = np.linspace(1, sample_max, n_control)  # 1 ~ sample_max 的10个控制点
        
        # --------------------------
        # 2. 区间外推：前后各扩0.5
        # --------------------------
        range_start = 1 - 0.5
        range_end = sample_max + 0.5
        
        # --------------------------
        # 3. 三次插值拟合分布函数
        # --------------------------
        interp_func = interpolate.interp1d(
            x_control, dist_list,
            kind="quadratic",  # 平滑插值：linear/quadratic/cubic
            fill_value="extrapolate"
        )
        
        # --------------------------
        # 4. 生成全范围均匀索引（覆盖所有区间）
        # --------------------------
        # 生成 0.5 ~ (sample_max+0.5) 的所有位置点
        x_indices = np.linspace(range_start, range_end, sample_max)
        # 计算每个点的权重（非负保证）
        weights = np.maximum(interp_func(x_indices), 0)
        
        # --------------------------
        # 5. 加权随机采样
        # --------------------------
        # 抽取带权重的样本
        samples = random.choices(x_indices, weights=weights, k=sample_num)
        
        # --------------------------
        # 6. 映射回原始整数索引（输出最终采样位置）
        # --------------------------
        # 将采样值 → 对应到 1~sample_max 的整数索引
        sample_indexes = [int(round(x)) for x in samples]
        # 边界修正（防止越界）
        sample_indexes = [max(1, min(sample_max, idx)) for idx in sample_indexes]
        
        return [i*10 for i in sample_indexes]

    # def sample_frames(self, video_path, estimated_time_range=None, distribution=None, num_frames=20):
    #     vr = VideoReader(video_path, ctx=cpu(0))
    #     total_frames = len(vr)
    #     # 固定采帧间隔
    #     interval = 120.0  # seconds
    #     # 获取视频帧率，用于将时间转换为帧索引
    #     fps = vr.get_avg_fps()
        
    #     if total_frames < num_frames:
    #         return sorted(list(range(total_frames)))
            
    #     # -----------------------------
    #     # 1. 处理 Explicit Time Range 采样
    #     # -----------------------------
    #     explicit_samples = []
        
    #     # 规范化输入：确保 time_ranges 是一个列表
    #     time_ranges = []
    #     if estimated_time_range:
    #         if isinstance(estimated_time_range, list):
    #             time_ranges = estimated_time_range
    #         elif isinstance(estimated_time_range, str):
    #             time_ranges = [estimated_time_range]
        
    #     # 定义内部辅助函数：时间字符串 "00:01:20" 转秒
    #     def time_str_to_sec(t_str):
    #         try:
    #             parts = t_str.strip().split(':')
    #             return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    #         except:
    #             return None

    #     # 遍历列表中的每个时间范围字符串
    #     for t_range in time_ranges:
    #         if isinstance(t_range, str) and t_range.lower() != "n/a":
    #             try:
    #                 if '-' not in t_range:
    #                     continue
                        
    #                 start_str, end_str = t_range.split('-')
    #                 start_sec = time_str_to_sec(start_str)
    #                 end_sec = time_str_to_sec(end_str)
                    
    #                 if start_sec is None or end_sec is None:
    #                     continue

    #                 duration = end_sec - start_sec
                    

    #                 if duration > 0:
    #                     time_points = []
    #                     if duration <= interval:
    #                         # 不足10s，取中间一帧
    #                         time_points.append(start_sec + duration / 2)
    #                     else:
    #                         # 每隔10s分配一帧
    #                         curr = start_sec
    #                         while curr <= end_sec:
    #                             time_points.append(curr)
    #                             curr += interval
                        
    #                     # 将时间点转换为帧索引
    #                     for t in time_points:
    #                         f_idx = int(t * fps)
    #                         # 确保不越界
    #                         if 0 <= f_idx < total_frames:
    #                             explicit_samples.append(f_idx)
    #             except Exception as e:
    #                 print(f"Error parsing estimated_time_range '{t_range}': {e}")
    #                 continue
        
    #     # 去重并排序
    #     explicit_samples = sorted(list(set(explicit_samples)))

    #     # -----------------------------
    #     # 2. 根据 Distribution 分配剩余帧
    #     # -----------------------------
    #     # 计算剩余可用的配额
    #     remaining_num_frames = num_frames - len(explicit_samples)
    #     dist_samples = []

    #     # 只有当还有剩余配额且提供了分布时才执行分布采样
    #     if remaining_num_frames > 0 and distribution:
    #         dist_samples = self.weighted_sampling(distribution, total_frames, remaining_num_frames)
            

    #     # -----------------------------
    #     # 3. 合并、填充与截断
    #     # -----------------------------
    #     # 合并两部分采样结果
    #     samples = sorted(list(set(explicit_samples + dist_samples)))
        
    #     # Fill up if needed (如果两部分加起来还不够 num_frames，随机填充)
    #     while len(samples) < num_frames:
    #         missing = num_frames - len(samples)
    #         possible_indices = list(set(range(total_frames)) - set(samples))
    #         if not possible_indices:
    #             break
    #         new_samples = np.random.choice(possible_indices, min(missing, len(possible_indices)), replace=False)
    #         samples.extend(new_samples)
    #         samples = sorted(list(set(samples)))
            
    #     # Trim if needed (如果 Explicit 采样过多导致总数超过 num_frames)
    #     if len(samples) > num_frames:
    #          samples = samples[:num_frames]

    #     return samples

    def get_answer(self, video_path, query, sorted_frame_idx=None, subtitle_path=None, subtitle_mode=0, distribution_mode=0):
        distribution = None
        if self.bench_name == 'LongVideoBench':
            print("Using video ID as cache key for LongVideoBench.")
            cache_key = str(self.id)
        else:
            print("Using video name and question as cache key for other benchmarks.")
            cache_key = f'{os.path.basename(video_path)}#{query.split("\nOptions")[0]}'

        if cache_key in self.dis_cache and distribution_mode!=0:
            self.qstype = self.dis_cache[cache_key]["qstype"]
            distribution = self.dis_cache[cache_key]["distribution"]
            print(f"Using cached distribution: {cache_key} -> {self.qstype}, {distribution}")
        else:
            print(f"No cached distribution found for key: {cache_key}. Computing distribution...")
        # # Always use smart sampling based on query
        

        # distribution = self.distributor.get_distribution_directly(query, uniform=True, subtitle_path=subtitle_path)
        # self.qstype = "uniform"

        if distribution is None:
            # distribution = self.distributor.get_distribution_directly(query, uniform=False, subtitle_path=subtitle_path)
            # self.qstype = "direct"

            if distribution_mode == 0:
                distribution = self.distributor.get_distribution_directly(query, uniform=True, subtitle_path=subtitle_path)
                self.qstype = "uniform"       
            elif distribution_mode == 1:
                distribution = self.distributor.get_distribution_directly(query, uniform=False, subtitle_path=subtitle_path)
                self.qstype = "direct"
            elif distribution_mode == 2:
                # # use question type to get distribution
                self.qstype, distribution = self.distributor.get_distribution_typely(query, subtitle_path=subtitle_path)
                self.type_acc[self.qstype.lower().replace(" ", "")][1] += 1  # total count
            else:
                raise ValueError(f"Unknown distribution_mode: {distribution_mode}")
            logger.info(f"Question type detected: {self.qstype}")
            logger.info(f"Distribution: {distribution}")
            self.dis_cache[cache_key] = {"qstype": self.qstype, "distribution": distribution}
            # self.dis_cache[self.id] = {"qstype": self.qstype, "distribution": distribution}
        else:
            self.type_acc[self.qstype.lower().replace(" ", "")][1] += 1  # total count

        if (self.qstype == "uniform" or self.qstype == "direct") and not ("estimated_time_range" in distribution):
            logger.info("Using uniform distribution for sampling...")
            #[0.13689571 0.12934579 0.09362994 0.09701249 0.08310744 0.09500698 0.0839084  0.09678854 0.08989483 0.09440989]
            # distribution = [0.13689571, 0.12934579, 0.09362994, 0.09701249, 0.08310744, 0.09500698, 0.0839084, 0.09678854, 0.08989483, 0.09440989]
            sorted_frame_idx = self.sample_frames(video_path, None, distribution, num_frames=Nf)
            # logger.info(f"Sampled frames (non-uniform): {sorted_frame_idx}")
        else:
            logger.info(f"Using {self.qstype} distribution for sampling...")
            sorted_frame_idx = self.sample_frames(video_path, distribution["estimated_time_range"], distribution["distribution"], num_frames=Nf)
        
        if subtitle_mode==2:
            # print('error here in subtitle mode!')
            # raise ValueError("Error distribution mode")
            # use subtitle as auxiliary information for answering
            return super().get_answer(video_path, query, sorted_frame_idx, subtitle_path=subtitle_path)
        else:
            return super().get_answer(video_path, query, sorted_frame_idx) #这里加入字幕作为辅助信息，和其他方法对比时可以不加入，仅辅助采帧

def evaluate_dataset(agent, json_path, video_root=None, bench_name='lvb', subtitle_root=None, num_samples=None, subtitle_mode=0, distribution_mode=0):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if num_samples:
        data = data[:num_samples]
        
    correct_count = 0
    total_count = 0
    
    for item in tqdm(data, desc="Evaluating"):
        if bench_name=="LongVideoBench":
            video_name = item['video_path']
            video_path = os.path.join(video_root, video_name)
            question = item['question']
            candidates = item['candidates']
            correct_choice_idx = item['correct_choice']
            
            subtitle_rel_path = item.get('subtitle_path')

        elif bench_name.lower()=="lvb":
            video_name = f'{item["video_id"]}.mp4'
            video_path = os.path.join(video_root, video_name)
            question = item['question']
            candidates = item['candidates']
            correct_choice_idx = item['correct_choice']
            
            subtitle_rel_path = f'{item["video_id"]}.json'
        
        elif bench_name=="Video-MME":
            video_id = item['videoID']
            video_name = f"{video_id}.mp4"
            video_path = os.path.join(video_root, video_name)
            question = item['question']
            candidates = item['options']
            correct_choice_idx = ord(item['answer'])-ord('A')  # Assuming answer is like "A", "B", etc.
            
            subtitle_rel_path = f"{video_id}.srt"

        subtitle_path = None
        if subtitle_rel_path:
            print("Subtitle found for this video.")
            subtitle_path = os.path.join(subtitle_root, subtitle_rel_path)
        if subtitle_mode == 0:
            print("Subtitle usage disabled.")
            subtitle_path = None

        # Format question with options
        options_str = ""
        for i, candidate in enumerate(candidates):
            option_pre = f'{chr(65+i)}. ' if candidate[0]!=chr(65+i) or candidate[1] != '.' else ""
            options_str += f"{option_pre}{candidate}\n"
        
        full_query = f"{question}\nOptions:\n{options_str}Answer with the option letter only."
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Question: {question}")
        logger.info(f"Options:\n{options_str}")
        
        if os.path.exists(video_path):
            agent.id += 1
            try:
                answer = agent.get_answer(video_path, full_query, subtitle_path=subtitle_path, subtitle_mode=subtitle_mode, distribution_mode=distribution_mode)
                # answer = agent.get_answer(video_path, full_query, subtitle_path=None)
                # Handle list output if get_answer returns a list
                if isinstance(answer, list):
                    answer = answer[0]
                
                logger.info(f"Model Answer: {answer}")
                
                # Simple parsing logic
                pred_idx = -1
                answer_clean = answer.strip().upper()
                
                # Check for A, B, C, D, E...
                found_options = []
                for i in range(len(candidates)):
                    option_char = chr(65+i)
                    # Check if the answer starts with the option, or is just the option
                    if answer_clean == option_char or answer_clean.startswith(option_char + ".") or answer_clean.startswith(option_char + ")") or answer_clean.startswith("OPTION " + option_char):
                         pred_idx = i
                         break
                
                if pred_idx == -1:
                     # Try to find the option letter in the text if it's short
                     if len(answer_clean) < 10:
                        for i in range(len(candidates)):
                            if chr(65+i) in answer_clean:
                                pred_idx = i
                                break
                
                if pred_idx == correct_choice_idx:
                    logger.info("Result: Correct")
                    correct_count += 1
                    if agent.qstype:
                        agent.type_acc[agent.qstype.lower().replace(" ", "")][0] += 1  # correct count
                else:
                    logger.info(f"Result: Incorrect (Expected {chr(65+correct_choice_idx) if type(correct_choice_idx)==int else correct_choice_idx})")
                
                total_count += 1
                logger.info(f"Current Accuracy: {correct_count}/{total_count} ({correct_count/total_count:.2%})")
                
            except Exception as e:
                logger.error(f"Error processing video: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.info(f"Video file not found at {video_path}. Please check the path.")
        print("-" * 50)

    if agent.qstype:
        logger.info("Per-type accuracy:")
        for qtype, (correct, total) in agent.type_acc.items():
            if total > 0:
                logger.info(f"  {qtype}: {correct}/{total} ({correct/total:.2%})")
    if total_count > 0:
        logger.info(f"Accuracy: {correct_count}/{total_count} ({correct_count/total_count:.2%})")

if __name__ == "__main__":
    # augument for command line arguments if needed
    num_samples = None  # Set to an integer for testing a subset
    distribution_mode = 2 # 0: uniform sampling, 1: distributional sampling with q, 2: distributional sampling with subtitles
    subtitle_mode = 1 # 0:no subtitles, 1:distribution with subtitles, 2:answering with subtitles
    # cache_path = None
    cache_path = f"./{log_tag}_cache.json"
    # cache_path = f"./LVBCaption_typely_cache.json"

    # Example usage
    # agent = SelfAgent('Qwen/Qwen2.5-VL-7B-Instruct')
    agent = SelfAgent('Qwen/Qwen2.5-VL-7B-Instruct')
    agent.load_cache(cache_path)

    video_root = "/data1/yangyan/benchmark/LongVideoBench/videos"
    subtitle_root = "/data1/yangyan/benchmark/LongVideoBench/subtitles"
    # json_path = "./lvbench_subset_50.json"
    json_path = "/data1/yangyan/benchmark/LongVideoBench/lvb_val.json"
    bench_name = 'LongVideoBench'

    # video_root = "/data1/yangyan/benchmark/Video-MME/data"
    # subtitle_root = "/data1/yangyan/benchmark/Video-MME/subtitle"
    # # json_path = "./lvbench_subset_50.json"
    # json_path = "/data1/yangyan/workdir/vmme_filtered_data.json"
    # bench_name = 'Video-MME'

    # video_root = "/data1/yangyan/benchmark/LVBCaption/LVBench_data/all_videos"
    # subtitle_root = "/data1/yangyan/benchmark/LVBCaption/LVBench_data/subtitles"
    # # json_path = "./lvbench_subset_50.json"
    # json_path = "/data1/yangyan/benchmark/LVBCaption/subtitle_set.json"
    # bench_name = 'lvb'
    
    # tag log info
    log_info = f"Log for frame-{Nf} with gemini on {bench_name}, subtitle_mode={subtitle_mode}, distribution_mode={distribution_mode}, cache_path={cache_path}"
    logger.info(log_info)
    agent.bench_name = bench_name

    evaluate_dataset(agent, json_path, video_root, bench_name=bench_name, subtitle_root=subtitle_root, num_samples=num_samples, subtitle_mode=subtitle_mode, distribution_mode=distribution_mode)
    agent.save_cache(cache_path)
