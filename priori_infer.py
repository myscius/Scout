
import os
import random
import json
from all_model_agent import  InternVL8B,InternVL78B, InternVL26B, Llava72B, Llava7B, Qwen2_5_Agent
from all_model_util import *
from tqdm import tqdm
import copy
import time
import pandas as pd
import re
from decord import VideoReader, cpu
import threading 
from PIL import Image
import torchvision.transforms as T

# Name: flash_attn
# Version: 2.7.4.post1

import logging
import sys
# 配置日志系统
log_file = 'lvbench_processing.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),  # 输出到文件
        logging.StreamHandler(sys.stdout)  # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)

# 加载clip模型
import torch
import torch.nn as nn
import numpy as np
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.modeling import CLIP4Clip

SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                 "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

model_path = '/data1/yangyan/checkpoint/mViTT_new/pytorch_model_0.0011.bin.25'
longvideobench_path = '/data1/yangyan/benchmark/LongVideoBench'
video_dir = os.path.join(longvideobench_path, 'videos')



def init_model(model_path, args):
    model_state_dict = torch.load(model_path, map_location='cpu')
    model = CLIP4Clip.from_pretrained("cross-base", cache_dir="", state_dict=model_state_dict, task_config=args)
    model.to('cuda')
    return model

def _get_text(tokenizer, video_id, sentence):
    choice_video_ids = [video_id]
    n_caption = len(choice_video_ids)
    k = n_caption
    pairs_text = np.zeros((k, 77), dtype=np.int64)
    pairs_mask = np.zeros((k, 77), dtype=np.int64)
    pairs_segment = np.zeros((k, 77), dtype=np.int64)

    words = tokenizer.tokenize(sentence)

    words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
    total_length_with_CLS = 76
    if len(words) > total_length_with_CLS:
        words = words[:total_length_with_CLS]
    words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]

    input_ids = tokenizer.convert_tokens_to_ids(words)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    while len(input_ids) < 77:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    pairs_text[0] = np.array(input_ids)
    pairs_mask[0] = np.array(input_mask)
    pairs_segment[0] = np.array(segment_ids)
    pairs_text = torch.Tensor(pairs_text).cuda()
    pairs_mask = torch.Tensor(pairs_mask).cuda()
    pairs_segment = torch.Tensor(pairs_segment).cuda()
    return pairs_text.long(), pairs_mask.long(), pairs_segment.long(), choice_video_ids

def extract_and_resize_frames(video_path, frame_indices):
    vr = VideoReader(video_path, ctx=cpu(0))
    frames = vr.get_batch(frame_indices).asnumpy()
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ])
    resized_frames = []
    for frame in frames:
        resized_frame = transform(frame)
        resized_frames.append(resized_frame)
    resized_frames = torch.stack(resized_frames)
    return resized_frames

def run_clip4clip(model, video_path, text, sample_idx):
    # 定义 args 字典
    args = {
        'video_dim': 1024,
        'max_words': 60,
        'max_frames': 16,
        'feature_framerate': 1,
        'margin': 0.1,
        'hard_negative_rate': 0.5,
        'negative_weighting': 1,
        'n_pair': 1,
        'text_num_hidden_layers': 16,
        'visual_num_hidden_layers': 16,
        'cross_num_hidden_layers': 4,
        'linear_patch': "2d",
        'sim_header': "seqTransf"
    }
    tokenizer = ClipTokenizer()
    input_ids, input_mask, segment_ids, choice_video_ids = _get_text(tokenizer, video_path, text)
    video = extract_and_resize_frames(video_path, sample_idx).cuda()
    video_mask = torch.Tensor([[1] * 16]).cuda()
    token_type_ids = torch.Tensor([[0] * 16]).cuda()
    visual_output = model.get_visual_output(video, video_mask=video_mask, shaped=True, video_frame=16)
    text_feat = model.get_sequence_output(input_ids, segment_ids, input_mask, shaped=True)
    b1b2_logits, *_tmp = model.get_similarity_logits(text_feat, visual_output, input_mask, video_mask,
                                                     loose_type=True, eval='myeval')
    
    del video, visual_output, text_feat, input_ids, input_mask, segment_ids
    torch.cuda.empty_cache()
    
    return b1b2_logits


    # 定义 args 字典
args = {
    'video_dim': 1024,
    'max_words': 60,
    'max_frames': 16,
    'feature_framerate': 1,
    'margin': 0.1,
    'hard_negative_rate': 0.5,
    'negative_weighting': 1,
    'n_pair': 1,
    'text_num_hidden_layers': 16,
    'visual_num_hidden_layers': 16,
    'cross_num_hidden_layers': 4,
    'linear_patch': "2d",
    'sim_header': "seqTransf"
}
asp_clip = init_model(model_path, args)
# 示例调用


def get_max_frame_block(video_path, text_prompt, sample_frame=16, sample_dict = None):
    best_clip_score_idx = []
    best_clip_score = 0
    select_block = 0
    sample_all_frames = []


    for i in range(1, 7):
        if sample_dict:
            sorted_frame_idx = sample_dict['all_samp'][i - 1]
        else:
            sorted_frame_idx = get_frame_idx_path(video_path, i, sample_frame)
        clip_score = run_clip4clip(asp_clip, video_path, text_prompt, sorted_frame_idx)
        if clip_score > best_clip_score:
            best_clip_score = clip_score
            best_clip_score_idx = sorted_frame_idx
            select_block = i
        sample_all_frames.append(sorted_frame_idx)

    return best_clip_score_idx, select_block, sample_all_frames

def get_frame_idx_path(video_path, round = 0, sample_frame=16, judge_whole=False):
    # 计算每一帧和text的相似度，然后选出最相似的帧，返回这些帧的路径
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    if len(vr) <= 96 or judge_whole:
        sorted_frame_idx = random.sample(range(len(vr)), sample_frame)
        sorted_frame_idx = sorted(sorted_frame_idx)
        return sorted_frame_idx
    inter = len(vr) // 6
    if 1<= round <= 6:
        sp = round * inter
        try:
            sorted_frame_idx = random.sample(range(sp-inter, sp - 1), sample_frame)
        except:
            sorted_frame_idx = random.sample(range(len(vr)), sample_frame)
    else:
        sorted_frame_idx = random.sample(range(len(vr)), sample_frame)

    sorted_frame_idx = sorted(sorted_frame_idx)
    return sorted_frame_idx
# 这个之后更改一下。

def get_result_first_round(agent_set, anno):
    answer_dict = {}
    sample_dict = {}
    decide_watch, info_prompt, get_mme_answer = get_lvbench_prompt(anno)
    video_path = os.path.join(video_dir, anno['video_path'])

    def process_agent(agent):
        agent_name = agent.get_model_name()
        # watch_ori = anno[agent_name]['watch'][0] if isinstance(anno[agent_name]['watch'], list) else anno[agent_name]['watch']
        if agent_name in anno:
            watch = agent.get_answer(video_path, decide_watch, anno[agent_name]["watch_samp"])
        else:
            watch = agent.get_answer(video_path, decide_watch, anno['intern_78b']["watch_samp"])
        watch = watch[0] if isinstance(watch, list) else watch
        anno[agent_name]['watch'] = watch
        logger.info(f"{agent_name} decide to watch: {watch}")
        if 'Yes' in watch:
            logger.info(f"{agent_name} is watching the video.")
            if 'sample_idx' in anno[agent_name].keys():
                sample_idx = anno[agent_name]['sample_idx']
            else:
                sample_idx = get_frame_idx_path(video_path, round=0, sample_frame=16)
            result = agent.get_answer(video_path, get_mme_answer, sample_idx)
            text_prompt = agent.get_answer(video_path, info_prompt, anno[agent_name]["watch_samp"])
            anno[agent_name]['info'] = text_prompt
        else:
            # text_prompt_ori = anno[agent_name]['info']
            text_prompt = agent.get_answer(video_path, info_prompt, anno[agent_name]["watch_samp"])
            anno[agent_name]['info'] = text_prompt

            if isinstance(text_prompt, list): text_prompt = text_prompt[0]
            if 'sample_dict' in anno[agent_name].keys():
                best_clip_score_idx, select_block, sample_all_frames = get_max_frame_block(video_path, text_prompt, 16, anno[agent_name]['sample_dict'])
            else:
                best_clip_score_idx, select_block, sample_all_frames = get_max_frame_block(video_path, text_prompt, sample_frame=16)

            result = agent.get_answer(video_path, get_mme_answer, sample_all_frames[select_block - 1])
        if isinstance(result, list):
            result = result[0]
        answer_dict[agent_name] = result.split('Answer: ')[-1][0]
        if 'Yes' not in watch and answer_dict[agent_name] == chr(ord('A') + anno['correct_choice']):
            sample_dict[agent_name] = {'all_samp': sample_all_frames, 'block': select_block}
            

    # threads = []
    # for agent in agent_set:
    #     thread = threading.Thread(target=process_agent, args=(agent,))
    #     threads.append(thread)
    #     thread.start()

    # for thread in threads:
    #     thread.join()
    for agent in agent_set:
        process_agent(agent)

    answer_set = set(answer_dict.values())
    logger.info(answer_dict)
    return answer_set, answer_dict, sample_dict
# 把这个函数改好

# first round进行一部分的更改。
def get_result_second_round(agent_set, anno, history_info=None):
    answer_dict = {}
    sample_dict = {}
    decide_watch, info_prompt, get_mme_answer = get_lvbench_prompt(anno)

    video_path = os.path.join(video_dir, anno['video_path'])

    def process_agent(agent):
        agent_name = agent.get_model_name()
        watch = anno[agent_name]['watch'][0] if isinstance(anno[agent_name]['watch'], list) else anno[agent_name]['watch']

        if 'Yes' in watch:

            sample_idx = get_frame_idx_path(video_path, round=0, sample_frame=16)
            result = agent.get_answer(video_path, get_mme_answer, sample_idx)
        else:
            text_prompt = history_info[agent_name]
            if isinstance(text_prompt, list): text_prompt = text_prompt[0]
            if agent_name == 'intern_78b':
                if 'sample_dict' in anno[agent_name].keys():
                    best_clip_score_idx, select_block, sample_all_frames = get_max_frame_block(video_path, text_prompt, 16, anno[agent_name]['sample_dict'])
                else:
                    best_clip_score_idx, select_block, sample_all_frames = get_max_frame_block(video_path, text_prompt, sample_frame=16)

                result = agent.get_answer(video_path, get_mme_answer, sample_all_frames[select_block - 1])
            # if 'sample_dict' in anno[agent_name].keys():
            #     best_clip_score_idx, select_block, sample_all_frames = get_max_frame_block(video_path, text_prompt, 16, anno[agent_name]['sample_dict'])
            else:
                best_clip_score_idx, select_block, sample_all_frames = get_max_frame_block(video_path, text_prompt, sample_frame=16)

            
            # if select_block == anno[agent_name]['block']:
            #     result = agent.get_answer(video_path, get_mme_answer, anno[agent_name]['sample_idx'])
            # else:
            result = agent.get_answer(video_path, get_mme_answer, sample_all_frames[select_block - 1])

        if isinstance(result, list):
            result = result[0]
        answer_dict[agent_name] = result.split('Answer: ')[-1][0]
        if 'Yes' not in watch and answer_dict[agent_name] == chr(ord('A') + anno['correct_choice']):
            sample_dict[agent_name] = {'all_samp': sample_all_frames, 'block': select_block}
  

    # threads = []
    # for agent in agent_set:
    #     thread = threading.Thread(target=process_agent, args=(agent,))
    #     threads.append(thread)
    #     thread.start()

    # for thread in threads:
    #     thread.join()
    for agent in agent_set:
        process_agent(agent)

    answer_set = set(answer_dict.values())
    return answer_set, answer_dict, sample_dict
# TODO: Add the history info here.
def reason_process(agent_set, anno, answer_dict, sample_idx=None, history_info=None):
    # 解释为什么要选择这个答案。
    video_path = os.path.join(video_dir, anno['video_path'])
    
    ans_dict = {}

    def process_agent(agent):
        agent_name = agent.get_model_name()
        reason_prompt = "Given the video frames you've seen, and the question along with your answer, deeply analyze the logical steps and evidence from the frames that led you to provide this particular answer. The Question is: {}\n, The predict answer is {}\n.".format(
            anno['question'], anno['candidates'][ord(answer_dict[agent_name]) - ord('A')])
        if 'sample_idx' not in anno[agent_name]:
            rand_block = random.randint(1, 6)
            local_sample_idx = get_frame_idx_path(video_path, round=rand_block, sample_frame=16, judge_whole=True)
        else:
            local_sample_idx = anno[agent_name]['sample_idx']
        try:
            result = agent.get_answer(video_path, reason_prompt, local_sample_idx)
        except:
            logger.info(anno['video_path'])
            rand_block = random.randint(1, 6)
            local_sample_idx = get_frame_idx_path(video_path, round=rand_block, sample_frame=16, judge_whole=True)
            result = agent.get_answer(video_path, reason_prompt, local_sample_idx)
        if isinstance(result, list):
            result = result[0]
        ans_dict[agent_name] = result

    threads = []
    for agent in agent_set:
        thread = threading.Thread(target=process_agent, args=(agent,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return ans_dict

def parse_json(text):
    if isinstance(text, list):
        text = text[0]
    text = re.sub(r"[\n\t]", "", text)
    text = text.replace('```json', '').replace('```', '')
    try:
        # First, try to directly parse the text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # If direct parsing fails, use regex to extract JSON
        json_pattern = r"\{.*?\}|\[.*?\]"  # Pattern for JSON objects and arrays

        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                match = match.replace("'", '"')
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # If no JSON structure is found
        logger.info("No valid JSON found in the text.")
        return None

def agent_back_process(agent_set, data):
    scores = {}

    for model in data.keys():
        score = 0
        for sub_dict in data.values():
            if model in sub_dict:
                score += int(sub_dict[model])
        scores[model] = score
    priority_order = ['intern_8b', 'llava_72b', 'intern_78b']
    min_score = min(scores.values())
    lowest_score_keys = [key for key, score in scores.items() if score == min_score]
    if len(lowest_score_keys) > 1:
        for priority_key in priority_order:
            if priority_key in lowest_score_keys:
                lowest_score_key = priority_key
                break
    else:
        lowest_score_key = lowest_score_keys[0]

    # 3. 排除掉分数最低的字典
    new_data = [key for key in data.keys() if key != lowest_score_key]

    return new_data, lowest_score_key, scores


def discuss_text_process(agent_set, anno, answer_dict, reason_dict):
    all_agent_name = [agent.get_model_name() for agent in agent_set]
    other_agent_name = copy.deepcopy(all_agent_name)
    sys_prompt = ""
    for key in reason_dict.keys():
        if isinstance(reason_dict[key], list):
            reason_dict[key] = reason_dict[key][0]
    discuss_dict = {}
    def process_agent(agent):
        agent_name = agent.get_model_name()
        local_other_agent_name = copy.deepcopy(other_agent_name)
        local_other_agent_name.remove(agent_name)
        if len(local_other_agent_name) == 1:
            answer_format = {agent_name: "1-10", local_other_agent_name[0]: "1-10"}
            discuss_prompt = f"""Given the answers and the reasoning for judgment from this model and two other models, please rate this model and the other two models. The score ranges from 1-10. Output in dictionary format.
            The question is: {anno['question']}, 
            The answer of this model is {answer_dict[agent_name]}, the reason is {reason_dict[agent_name]}.
            The answer of {local_other_agent_name[0]} model is {answer_dict[local_other_agent_name[0]]}, the reason is {reason_dict[local_other_agent_name[0]]}.
            You do not need to explain your answer, just give me scores as your answer following the answer_format.
            Please strictly follow the answer format! The answer_format is:
            {answer_format}
            """
        if len(local_other_agent_name) > 1:
            answer_format = {agent_name: "1-10", local_other_agent_name[0]: "1-10", local_other_agent_name[1]: "1-10"}
            discuss_prompt = f"""Given the answers and the reasoning for judgment from this model and two other models. 
            The question is: {anno['question']}
            The answer of this model is {answer_dict[agent_name]}, the reason is {reason_dict[agent_name]}.
            The answer of {local_other_agent_name[0]} model is {answer_dict[local_other_agent_name[0]]}, the reason is {reason_dict[local_other_agent_name[0]]}.
            The answer of {local_other_agent_name[1]} model is {answer_dict[local_other_agent_name[1]]}, the reason is {reason_dict[local_other_agent_name[1]]}.
            Please score the performance of this model an other two models base on their reasoning. The score ranges from 1-10. Output in dict format.
            You do not need to explain your answer, just give me scores as your answer following the answer_format.
            Please strictly follow the answer format! The answer_format is:
            {answer_format}
            """

        if agent_name == 'llava_72b':
            video_path = os.path.join(video_dir, anno['video_path'])
            temp = agent.get_answer(video_path, discuss_prompt, anno['llava_72b']['watch_samp'])
        else:
            temp = agent.get_text_answer(discuss_prompt)
        logger.info(temp)
        # 添加 None 检查
        temp = parse_json(temp)
        if temp is None:
            logger.warning(f"Failed to parse JSON from {agent_name}, using default scores")
            discuss_dict[agent_name] = {'intern_78b': 6, 'intern_8b': 6, 'llava_72b': 6}
            return

        is_valid = True
        for key, value in temp.items():
            try:
                temp[key] = int(value)
            except:
                temp[key] = 'no_val'

        for value in temp.values():
            if not isinstance(value, int) or value < 1 or value > 10:
                is_valid = False
                break
        if is_valid:
            discuss_dict[agent_name] = temp
        else:
            discuss_dict[agent_name] = {'intern_78b': 8, 'intern_8b': 6, 'llava_72b': 7}

    threads = []
    for agent in agent_set:
        thread = threading.Thread(target=process_agent, args=(agent,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return discuss_dict

# 整理历史信息，要注意一下。
def generate_history_info(agent_set, anno, new_data, lowest_score_key, scores, reason_dict, answer_dict):

    all_prompt = " Discussion History Summary:\n"
    for data in new_data:
        all_prompt += "{}'s answer: {}\n Reason: {}\n The final score is {}.\n".format(data, anno['candidates'][ord(answer_dict[data]) - ord('A')], reason_dict[data], scores[data])
    all_prompt += "Removed Answer ({})\n Answer: {}\n Reason {}\n However, this reason was deemed unconvincing, so this answer was removed from the discussion.".format(
        lowest_score_key, anno['candidates'][ord(answer_dict[lowest_score_key]) - ord('A')], reason_dict[lowest_score_key])
    # 请从这些history中提取出要回答这个问题需要什么关键信息。
    # 给之前的info, 问题，答案
    
    history_info = {}
    for agent in agent_set:
        agent_name = agent.get_model_name()
        
        history_generate_prompt = lvbench_info_history(anno, all_prompt, anno[agent_name]['info'])

        if agent_name == 'llava_72b':
            video_path = os.path.join(video_dir, anno['video_path'])
            history_info[agent_name] = agent.get_answer(video_path, history_generate_prompt, anno['llava_72b']['watch_samp'])
        else:
            history_info[agent_name] = agent.get_text_answer(history_generate_prompt)
        
        if isinstance(history_info[agent_name], list): history_info[agent_name] = history_info[agent_name][0] 
    return history_info

def agent_process(single_video_anno, agent_set_ori):
    count = 0
    total = 0
    for kkk, anno in tqdm(enumerate(single_video_anno), desc="processing items"):
        agent_set = agent_set_ori
        # all_num += 1
        logger.info(anno['video_path'])
        logger.info(anno['question'])
        logger.info(anno['candidates'])
        logger.info(chr(ord('A') + anno['correct_choice']))
        # if 'On the left side of the screen, there is an image with several pieces of paper' not in anno['question']:
        #     continue
        agent_first = agent_set[0]
        decide_watch, info_prompt, get_mme_answer = get_lvbench_prompt(anno)

        video_path = os.path.join(video_dir, anno['video_path'])
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame = len(vr)
        sample_idx = np.linspace(0, total_frame - 1, 10, dtype=int).tolist()

        # 尝试清空 Agent 的历史记录 (假设 agent 有 history 属性或 clear 方法)
        # 如果 InternVL8B 类中有维护 self.history，必须清空它，否则上下文会无限增长
        if hasattr(agent_first, 'history'):
            logging.info("Clearing agent history.")
            agent_first.history = []
        if hasattr(agent_first, 'messages'):
            logging.info("Clearing agent messages.")
            agent_first.messages = []
        if hasattr(agent_first, 'clear_history'):
            logging.info("Calling clear_history method.")
            agent_first.clear_history()

        with torch.no_grad():
            output = agent_first.get_answer(video_path, get_mme_answer,sample_idx)

        logger.info(f'{output},{type(output)}')
        if type(output) == list:
            total+=1
            count+=(output[0]==chr(ord('A') + anno['correct_choice']))
        elif type(output) == str:
            total+=1
            count+=(output==chr(ord('A') + anno['correct_choice']))
        
        if total > 0:
            logger.info(f"Current accuracy: {count}/{total} = {count/total:.2f}")
    logger.info(f"Final Accuracy: {count}/{total} = {count/total if total > 0 else 0:.2f}")   

def get_max_memory_torch():
    """使用PyTorch原生API获取显存信息"""
    max_memory = {}
    
    for i in range(torch.cuda.device_count()):
        # mem_get_info返回 (free, total) 单位是字节
        free, total = torch.cuda.mem_get_info(i)
        
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        used_gb = (total - free) / (1024**3)
        
        # 第一张卡保守一些
        safety_factor = 0.95 if i == 0 else 0.95
        usable_gb = free_gb * safety_factor
        
        max_memory[i] = f"{int(usable_gb)}GiB"
        
        print(f"GPU {i}:")
        print(f"  总显存: {total_gb:.2f} GB")
        print(f"  已使用: {used_gb:.2f} GB")
        print(f"  空闲: {free_gb:.2f} GB")
        print(f"  可分配: {usable_gb:.2f} GB")
        print()
    
    return max_memory

internvl8b = InternVL8B()
# # internvl78b = InternVL78B()
# internvl78b = InternVL26B()
# # llava72b = Llava72B()
# llava72b = Llava7B()
# max_memory = get_max_memory_torch()
max_memory = None
print("Max memory per GPU:", max_memory)
# 动态选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
qwen25_7b = Qwen2_5_Agent("Qwen/Qwen2.5-VL-3B-Instruct",device=device,max_memory=max_memory)

if __name__ == "__main__":
    # all_anno_path = '/fs-computility/video/shared/wangzikang/Qwen2-VL-main/VideoAgent-master/concat_result/lvbench_all_result_change.json'
    all_anno_path = './lvbench_subset_50.json'
    logger.info("Annotation file: {}".format(all_anno_path))

    result_anno = json.load(open(all_anno_path))
    agent_set_ori = [internvl8b]
    import copy
    ori_anno = copy.deepcopy(result_anno)

    # single test
    # target_video_id = "aX_HgA5SNLQ"
    # # # target_video_id = "ZsnfXfuGRrg"
    # # # target_video_id = "NHIT9vq6mJU"
    # single_video_anno = [item for item in result_anno if isinstance(item, dict) and item.get("video_id") == target_video_id]

    # if not single_video_anno:
    #     logger.warning("No item with video_id=%s found in result_anno", target_video_id)
    # agent_process(single_video_anno, agent_set_ori)

    # Full test
    agent_process(result_anno, agent_set_ori)

    