import json
import os
from tqdm import tqdm

class CacheChecker:
    def __init__(self):
        self.cache = {}
        self.replication = 0
        self.failures = 0

cache = CacheChecker()

class VideoCounter:
    def __init__(self):
        self.counts = {}

    def count_video(self, video_path):
        video_name = os.path.basename(video_path)
        self.counts[video_name] = self.counts.get(video_name, 0) + 1
    def video_num(self):
        return len(self.counts)
videocounter = VideoCounter()

def check_cache(video_path, full_query):
    cache_key = f'{os.path.basename(video_path)}#{full_query.split("\nOptions")[0]}'
    if cache_key in cache.cache:
        cache.cache[cache_key] += 1
        cache.replication +=1
        print(f"{'-'*10}Num.{cache.replication} : Duplicate found for video: {video_path}{'-'*10}")
        print(full_query)
    else:
        cache.cache[cache_key] = 1

def check_cache_key(json_path, video_root=None, bench_name='lvb', num_samples=None):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if num_samples:
        data = data[:num_samples]
            
    for item in tqdm(data, desc="Checking cache"):
        if bench_name.lower()=="lvb" or bench_name=="LongVideoBench":
            video_name = f'{item["video_id"]}.mp4'
            video_path = os.path.join(video_root, video_name)
            question = item['question']
            candidates = item['candidates']
        
        elif bench_name=="Video-MME":
            video_id = item['videoID']
            video_name = f"{video_id}.mp4"
            video_path = os.path.join(video_root, video_name)
            question = item['question']
            candidates = item['options']

        # Format question with options
        options_str = ""
        for i, candidate in enumerate(candidates):
            option_pre = f'{chr(65+i)}. ' if candidate[0]!=chr(65+i) or candidate[1] != '.' else ""
            options_str += f"{option_pre}{candidate}\n"
        
        full_query = f"{question}\nOptions:\n{options_str}Answer with the option letter only.\n"
        
        if os.path.exists(video_path):
            check_cache(video_path, full_query)
            videocounter.count_video(video_path)
        else:
            print(f"Video not found: {video_path}")
            cache.failures += 1
    
    print(f"Total cached: {cache.cache.__len__()}, Duplicated: {cache.replication}, Failures: {cache.failures}, Videos: {videocounter.video_num()}")


if __name__== "__main__":
    # Example usage
    num_samples = None
    video_root = "/data1/yangyan/benchmark/LVBCaption/LVBench_data/all_videos"
    subtitle_root = "/data1/yangyan/benchmark/LVBCaption/LVBench_data/subtitles"
    # json_path = "./lvbench_subset_50.json"
    json_path = "/data1/yangyan/benchmark/LVBCaption/subtitle_set.json"
    bench_name = 'lvb'
    
    check_cache_key(json_path, video_root=video_root, bench_name=bench_name, num_samples=num_samples)