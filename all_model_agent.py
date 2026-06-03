# 把所有的以类作为初始化
# 方法：get_answer(video_path, idx) 这些的
# cal_score 给其他智能体评分
# 获取其他智能体评分
# 排除掉的智能体在其他地方写。
import copy
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor,Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModel
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
import torch
# from longvu.builder import load_pretrained_model_longvu
# from longvu.conversation import longvu_conv_templates, LongVUSeparatorStyle
# from longvu.mm_datautils import (
#     KeywordsStoppingCriteria,
#     longvu_process_images,
#     longvu_tokenizer_image_token,
# )

from decord import cpu, VideoReader  # @manual=fbsource//third-party/pypi/decord:decord
from torch import distributed as dist
from tqdm import tqdm
import json


from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from all_model_util import *
import torch
import os
import json
import random
import argparse
import time
import pandas as pd
import re
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T

from QuesGrouding.utils import get_subtitle

def get_anno(anno_path):
    # return sample_idx, anno [0]
    anno = json.load(open(anno_path, 'r'))
    return anno[0]

class BaseVisionModel:
    def __init__(self, model_path=None):
        # TODO: change path 
        self.model_path = model_path

        # path = '//data1/yangyan/checkpoint/InternVL2_5-8B'
        path = '/data1/yangyan/checkpoint/InternVL2_5-8B'
        # repo_id = 'OpenGVLab/InternVL2_5-8B'
        # device_map = self.split_model('InternVL2_5-8B')
        device_map = "auto"
        self.model = AutoModel.from_pretrained(
            path,
            # repo_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        self.generation_config = dict(max_new_tokens=64, do_sample=True)
    
    def get_model_name(self):
        return self.model_path
    def build_transform(self,input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform
    
    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=336, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def split_model(self, model_name):
        device_map = {}
        world_size = torch.cuda.device_count()
        num_layers = {
            'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
            'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
        device_map['language_model.model.rotary_emb'] = 0

        return device_map
    
    def load_video(self, video_path, frame_indices, input_size=448, max_num=1):

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        # 提取时间戳
        frame_times = []
        for i, idx in enumerate(frame_indices):
            time_seconds = idx / fps
            hours = int(time_seconds // 3600)
            minutes = int((time_seconds % 3600) // 60)
            seconds = time_seconds % 60
            frame_times.append(f"{hours:02d}:{minutes:02d}:{seconds:06.3f}")

        pixel_values_list, num_patches_list = [], []
        transform = self.build_transform(input_size=input_size)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img = self.dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        # torch.Size([48, 3, 448, 448])
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list, frame_times

    def get_answer(self, video_path, query, sorted_frame_idx):

        pixel_values, num_patches_list, frame_times = self.load_video(video_path, sorted_frame_idx)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + query
        # question = video_prefix + query.replace("<frame_times>", f"[{', '.join(frame_times)}]")
        # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
        response, history = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config,
                                    num_patches_list=num_patches_list, history=None, return_history=True)
        # print(f'User: {question}\nAssistant: {response}')
        return response
    
    def get_text_answer(self, query):

        # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
        response, history = self.model.chat(self.tokenizer, None, query, self.generation_config,
                                    num_patches_list=None, history=None, return_history=True)
        # print(f'User: {question}\nAssistant: {response}')
        return response

class BaseAgent:
    # TODO: change path
    def __init__(self, modeal_name='Qwen/Qwen2.5-VL-7B-Instruct',device='cuda',max_memory=None):
        model_path = modeal_name
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto", max_memory=max_memory)
        # self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, use_flash_attention_2=False, device_map="auto", max_memory=max_memory)
        self.device = device
        self.max_memory = max_memory

    def get_model_name(self):
        return 'qwen2_5_7b'

    def get_text_answer(self, text):
        messages = [[{"role": "user", "content": [{"type": "text", "text": text}]}]]
        images, videos = None, None
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        # print(inputs)
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

    def get_answer(self,video_path, text, sample_idx=None, multi_image_path= None, subtitle_path=None):
        subtitle_text = None
        if subtitle_path and os.path.exists(subtitle_path):
            try:
                # with open(subtitle_path, 'r', encoding='utf-8') as f:
                #     subtitle_text = f.read()
                subtitle_text = get_subtitle(subtitle_path)
                subtitle_prompt = f"""\nHere are the subtitles for the video.\n
                NOTE that you do not need to analyze all of the subtitles. Search according to your needs to avoid too much redundant information affecting your normal work.\n
                <Additional subtitles, available as needed>\n
                {subtitle_text}\n
                </Additional subtitles, available as needed>\n
                """
                text+=subtitle_prompt
                print("get answer with subtitles")
            except Exception as e:
                print(f"Error reading subtitle: {e}")

        if multi_image_path:
            messages = [
                {"role": "user", 
                "content": [{"type": "image", "image": video_path} for video_path in multi_image_path]}]
            messages[0]['content'].append({"type": "text", "text": text})
            images, videos = process_vision_info(messages)
        
    
        elif video_path == None:
            messages = [[{"role": "user", "content": [{"type": "text", "text": text}]}]]
            images, videos = None, None
            
        else:
            messages = [[{"role": "user", "content": [{"type": "video", "video": video_path}, {"type": "text", "text": text}]}]]
            images, videos = process_vision_info(messages, sample_idx)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt")
        del text,images,videos
        inputs = inputs.to(self.device)
        # # 打印最大和最小的 token id，以及模型支持的词表大小
        # print("Max Token ID:", inputs["input_ids"].max().item())
        # print("Min Token ID:", inputs["input_ids"].min().item())
        # print("Model Vocab Size:", self.model.config.vocab_size)
        # print("Input Shape:", inputs["input_ids"].shape)
        # # 1. 打印出来确认一下：你会发现它里面最大的数字很可能已经超过了模型当前的上限
        # if "position_ids" in inputs:
        #     print("Max Position ID in inputs:", inputs["position_ids"].max().item())

        # print("Current Model Max Position:", self.model.config.max_position_embeddings)
        # # 对于 Qwen-VL 等多模态模型，通常视觉特征是 pixel_values 或类似字段
        # for k, v in inputs.items():
        #     if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
        #         if torch.isnan(v).any() or torch.isinf(v).any():
        #             print(f"警告: 输入张量 {k} 中包含 NaN 或 Inf!")
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text


class InternVL8B:
    def __init__(self):
        # TODO: change path 

        # path = '/fs-computility/video/shared/wangzikang/internvl2.5/InternVL2_5-8B'
        path = '/data1/yangyan/checkpoint/InternVL2_5-8B'
        # repo_id = 'OpenGVLab/InternVL2_5-8B'
        # device_map = self.split_model('InternVL2_5-8B')
        device_map = "auto"
        self.model = AutoModel.from_pretrained(
            path,
            # repo_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        self.generation_config = dict(max_new_tokens=64, do_sample=True)
    
    def get_model_name(self):
        return 'intern_8b'
    def build_transform(self,input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform
    
    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=336, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def split_model(self, model_name):
        device_map = {}
        world_size = torch.cuda.device_count()
        num_layers = {
            'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
            'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
        device_map['language_model.model.rotary_emb'] = 0

        return device_map
    
    def load_video(self, video_path, frame_indices, input_size=448, max_num=1):

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        # 提取时间戳
        frame_times = []
        for i, idx in enumerate(frame_indices):
            time_seconds = idx / fps
            hours = int(time_seconds // 3600)
            minutes = int((time_seconds % 3600) // 60)
            seconds = time_seconds % 60
            frame_times.append(f"{hours:02d}:{minutes:02d}:{seconds:06.3f}")

        pixel_values_list, num_patches_list = [], []
        transform = self.build_transform(input_size=input_size)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img = self.dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        # torch.Size([48, 3, 448, 448])
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list, frame_times

    def get_answer(self, video_path, query, sorted_frame_idx):

        pixel_values, num_patches_list, frame_times = self.load_video(video_path, sorted_frame_idx)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + query
        # question = video_prefix + query.replace("<frame_times>", f"[{', '.join(frame_times)}]")
        # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
        response, history = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config,
                                    num_patches_list=num_patches_list, history=None, return_history=True)
        # print(f'User: {question}\nAssistant: {response}')
        return response
    
    def get_text_answer(self, query):

        # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
        response, history = self.model.chat(self.tokenizer, None, query, self.generation_config,
                                    num_patches_list=None, history=None, return_history=True)
        # print(f'User: {question}\nAssistant: {response}')
        return response


class InternVL78B:
    def __init__(self):
        # TODO: change path
        # path = '/fs-computility/video/shared/wangzikang/internvl2.5/checkpoint'
        # path = '/data1/yangyan/checkpoint/InternVL2_5-78B'
        path = '/data1/yangyan/checkpoint/InternVL2_5-26B'
        # device_map = self.split_model('InternVL2_5-78B')
        device_map = "auto"
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

        self.generation_config = dict(max_new_tokens=128, do_sample=True)
    
    def get_model_name(self):
        return 'intern_78b'

    def build_transform(self,input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform
    
    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=336, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def split_model(self, model_name):
        device_map = {}
        world_size = torch.cuda.device_count()
        num_layers = {
            'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
            'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
        device_map['language_model.model.rotary_emb'] = 0

        return device_map
    
    def load_video(self, video_path, frame_indices, input_size=448, max_num=1):

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        transform = self.build_transform(input_size=input_size)

        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img = self.dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        # torch.Size([48, 3, 448, 448])
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def get_text_answer(self, query):
        # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
        response, history = self.model.chat(self.tokenizer, None, query, self.generation_config,
                                    num_patches_list=None, history=None, return_history=True)
        # print(f'User: {question}\nAssistant: {response}')
        return response

    def get_answer(self, video_path, query, sorted_frame_idx):

        pixel_values, num_patches_list = self.load_video(video_path, sorted_frame_idx)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + query
        # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
        response, history = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config,
                                    num_patches_list=num_patches_list, history=None, return_history=True)
        # print(f'User: {question}\nAssistant: {response}')
        return response

class InternVL26B:
    def __init__(self):
        # TODO: change path
        # path = '/fs-computility/video/shared/wangzikang/internvl2.5/checkpoint'
        # path = '/data1/yangyan/checkpoint/InternVL2_5-78B'
        path = '/data1/yangyan/checkpoint/InternVL2_5-26B'
        # device_map = self.split_model('InternVL2_5-78B')
        device_map = "auto"
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

        self.generation_config = dict(max_new_tokens=128, do_sample=True)
    
    def get_model_name(self):
        return 'intern_78b'

    def build_transform(self,input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform
    
    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=336, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def split_model(self, model_name):
        device_map = {}
        world_size = torch.cuda.device_count()
        num_layers = {
            'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
            'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
        device_map['language_model.model.rotary_emb'] = 0

        return device_map
    
    def load_video(self, video_path, frame_indices, input_size=448, max_num=1):

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        # 提取时间戳
        frame_times = []
        for i, idx in enumerate(frame_indices):
            time_seconds = idx / fps
            hours = int(time_seconds // 3600)
            minutes = int((time_seconds % 3600) // 60)
            seconds = time_seconds % 60
            frame_times.append(f"{hours:02d}:{minutes:02d}:{seconds:06.3f}")

        pixel_values_list, num_patches_list = [], []
        transform = self.build_transform(input_size=input_size)

        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img = self.dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        # torch.Size([48, 3, 448, 448])
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list, frame_times

    def get_text_answer(self, query):
        # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
        response, history = self.model.chat(self.tokenizer, None, query, self.generation_config,
                                    num_patches_list=None, history=None, return_history=True)
        # print(f'User: {question}\nAssistant: {response}')
        return response

    def get_answer(self, video_path, query, sorted_frame_idx):

        pixel_values, num_patches_list, frame_times = self.load_video(video_path, sorted_frame_idx)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + query
        # replace Framei with timestamp
        # question = video_prefix + query.replace("<frame_times>", f"[{', '.join(frame_times)}]")
        # logging.info(f"Question with timestamps: {question}")
        # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
        response, history = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config,
                                    num_patches_list=num_patches_list, history=None, return_history=True)
        # print(f'User: {question}\nAssistant: {response}')
        return response

class Llava72B:
    def __init__(self):
        # TODO: change path
        # pretrained = "/fs-computility/video/shared/wangzikang/Qwen2-VL-main/llava_checkpoint"
        # pretrained = "/data1/yangyan/checkpoint/LLaVA-Video-72B-Qwen2"
        pretrained = "/data1/yangyan/checkpoint/LLaVA-Video-7B-Qwen2"
        model_name = "llava_qwen"
        device_map = "auto"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  
    
    def get_model_name(self):
        return 'llava_72b'

    def load_video(self, video_path, frame_idx):
        vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
        for i, idx in enumerate(frame_idx):
            if idx > len(vr) - 1:
                frame_idx[i] = len(vr) - (17-i)
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames

    def get_answer(self, video_path, question, sample_idx):
        video = self.load_video(video_path, sample_idx)
        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        video = [video]
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        
        question = DEFAULT_IMAGE_TOKEN  + question
        # prompt之后需要更改一下
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        cont = self.model.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=128,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        return text_outputs

class Llava7B:
    def __init__(self):
        # TODO: change path
        # pretrained = "/fs-computility/video/shared/wangzikang/Qwen2-VL-main/llava_checkpoint"
        # pretrained = "/data1/yangyan/checkpoint/LLaVA-Video-72B-Qwen2"
        pretrained = "/data1/yangyan/checkpoint/LLaVA-Video-7B-Qwen2"
        model_name = "llava_qwen"
        device_map = "auto"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  
    
    def get_model_name(self):
        return 'llava_72b'

    def load_video(self, video_path, frame_idx):
        vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
        for i, idx in enumerate(frame_idx):
            if idx > len(vr) - 1:
                frame_idx[i] = len(vr) - (17-i)
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames

    def get_answer(self, video_path, question, sample_idx):
        video = self.load_video(video_path, sample_idx)
        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        video = [video]
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        
        question = DEFAULT_IMAGE_TOKEN  + question
        # prompt之后需要更改一下
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        cont = self.model.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=128,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        return text_outputs


class Qwen72bAgent:
    def __init__(self):
        # TODO: change path
        # model_path = '/fs-computility/video/shared/wangzikang/Qwen2-VL-main/Qwen2-VL-main/Qwen2-VL-main/Qwen2-VL-main/qwen2_vl_checkpoint'
        model_path = '/data1/yangyan/checkpoint/Qwen2-VL-72B-Instruct'
        self.processor = Qwen2VLProcessor.from_pretrained(model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")

    def get_model_name(self):
        return 'qwen_72b'
    
    def get_text_answer(self, text):
        messages = [[{"role": "user", "content": [{"type": "text", "text": text}]}]]
        images, videos = None, None
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        # print(inputs)
        generated_ids = self.model.generate(**inputs, max_new_tokens=96)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
    def get_answer(self, video_path, question, sample_idx=None, multi_image_path= None):

        if multi_image_path:
            messages = [
                {"role": "user", 
                "content": [{"type": "image", "image": video_path} for video_path in multi_image_path]}]
            messages[0]['content'].append({"type": "text", "text": question})
            images, videos = process_vision_info(messages)
        
    
        elif video_path == None:
            messages = [[{"role": "user", "content": [{"type": "text", "text": question}]}]]
            images, videos = None, None
            
        else:
            messages = [[{"role": "user", "content": [{"type": "video", "video": video_path}, {"type": "text", "text": question}]}]]
            images, videos = process_vision_info(messages, sample_idx)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        # print(inputs)
        generated_ids = self.model.generate(**inputs, max_new_tokens=96)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text



class Qwen7bAgent:
    # TODO: change path
    def __init__(self):
        # model_path = '/fs-computility/video/shared/wangzikang/Qwen2-VL-main/Qwen2-VL-main/Qwen2-VL-7B-Instruct'
        model_path = '/data1/yangyan/checkpoint/Qwen2-VL-7B-Instruct'
        self.processor = Qwen2VLProcessor.from_pretrained(model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")

    def get_model_name(self):
        return 'qwen_7b'

    def get_text_answer(self, text):
        messages = [[{"role": "user", "content": [{"type": "text", "text": text}]}]]
        images, videos = None, None
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        # print(inputs)
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

    def get_answer(self,video_path, text, sample_idx=None, multi_image_path= None):

        if multi_image_path:
            messages = [
                {"role": "user", 
                "content": [{"type": "image", "image": video_path} for video_path in multi_image_path]}]
            messages[0]['content'].append({"type": "text", "text": text})
            images, videos = process_vision_info(messages)
        
    
        elif video_path == None:
            messages = [[{"role": "user", "content": [{"type": "text", "text": text}]}]]
            images, videos = None, None
            
        else:
            messages = [[{"role": "user", "content": [{"type": "video", "video": video_path}, {"type": "text", "text": text}]}]]
            images, videos = process_vision_info(messages, sample_idx)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        # print(inputs)
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text


class Qwen2_5_Agent:
    # TODO: change path
    def __init__(self, modeal_name='Qwen2.5-VL-7B-Instruct',device='cuda',max_memory=None):
        model_path = modeal_name
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto", max_memory=max_memory)
        # self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, use_flash_attention_2=False, device_map="auto", max_memory=max_memory)
        self.device = device
        self.max_memory = max_memory

    def get_model_name(self):
        return 'qwen2_5_7b'

    def get_text_answer(self, text):
        messages = [[{"role": "user", "content": [{"type": "text", "text": text}]}]]
        images, videos = None, None
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        # print(inputs)
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

    def get_answer(self,video_path, text, sample_idx=None, multi_image_path= None):

        if multi_image_path:
            messages = [
                {"role": "user", 
                "content": [{"type": "image", "image": video_path} for video_path in multi_image_path]}]
            messages[0]['content'].append({"type": "text", "text": text})
            images, videos = process_vision_info(messages)
        
    
        elif video_path == None:
            messages = [[{"role": "user", "content": [{"type": "text", "text": text}]}]]
            images, videos = None, None
            
        else:
            messages = [[{"role": "user", "content": [{"type": "video", "video": video_path}, {"type": "text", "text": text}]}]]
            images, videos = process_vision_info(messages, sample_idx)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt")
        del text,images,videos
        inputs = inputs.to(self.device)
        # print(inputs)
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

class Qwen3_8bAgent:
    # TODO: change path
    def __init__(self):
        model_name = "Qwen/Qwen3-VL-8B-Instruct"
        # model_path = '/data1/yangyan/checkpoint/Qwen2.5-VL-7B-Instruct'
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")

    def get_model_name(self):
        return 'qwen3_8b'

    def get_text_answer(self, text):
        messages = [[{"role": "user", "content": [{"type": "text", "text": text}]}]]
        images, videos = None, None
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        # print(inputs)
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

    def get_answer(self,video_path, text, sample_idx=None, multi_image_path= None):

        if multi_image_path:
            messages = [
                {"role": "user", 
                "content": [{"type": "image", "image": video_path} for video_path in multi_image_path]}]
            messages[0]['content'].append({"type": "text", "text": text})
            images, videos = process_vision_info(messages)
        
    
        elif video_path == None:
            messages = [[{"role": "user", "content": [{"type": "text", "text": text}]}]]
            images, videos = None, None
            
        else:
            messages = [[{"role": "user", "content": [{"type": "video", "video": video_path}, {"type": "text", "text": text}]}]]
            images, videos = process_vision_info(messages, sample_idx)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        # print(inputs)
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

    

