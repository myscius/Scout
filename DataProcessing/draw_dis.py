import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def main():
    # LongVideoBench_typely_cache.json LVBCaption_typely_cache.json Video-MME_typely_gemini.json
    json_path = "Video-MME_typely_gemini.json"
    
    # 1. 读取 JSON 数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    type_dists = {}
    all_dists = []

    # 2. 遍历确定的字典结构并归一化
    for key, value in data.items():
        try:
            qstype = value["qstype"]
            # 判断 distribution 对应的是字典还是直接是列表
            dist_data = value["distribution"]
            if isinstance(dist_data, dict):
                dist_list = dist_data["distribution"]
            elif isinstance(dist_data, list):
                dist_list = dist_data
            else:
                continue # 数据格式不符合预期则跳过
        except (KeyError, TypeError):
            # 捕获缺失键或类型异常
            continue
            
        dist_array = np.array(dist_list, dtype=float)
        total = np.sum(dist_array)
        
        # 归一化处理（防止全0列表导致除以0报错）
        if total > 0:
            norm_dist = dist_array / total
        else:
            norm_dist = dist_array
            
        # 按 qstype 分类
        if qstype not in type_dists:
            type_dists[qstype] = []
        
        type_dists[qstype].append(norm_dist)
        all_dists.append(norm_dist)

    if not all_dists:
        print("未找到有效的数据用于统计！")
        return

    # 3. 分别计算每个 qstype 的平均分布
    mean_by_type = {}
    for q_type, dists in type_dists.items():
        # axis=0 保证对列表各对应位的元素求均值
        mean_by_type[q_type] = np.mean(dists, axis=0)
    
    # 4. 计算所有数据的整体平均分布
    global_mean = np.mean(all_dists, axis=0)
    print(f"整体平均分布: {global_mean}")

    # 获取 x 轴的基础刻度，并生成更密集的 x 刻度用于平滑插值
    x = np.arange(len(global_mean))
    x_smooth = np.linspace(x.min(), x.max(), 300)

    # 5. 可视化绘图
    plt.figure(figsize=(10, 6))

    # 遍历绘制各个 qstype 的平滑曲线
    for qstype, mean_dist in mean_by_type.items():
        # 使用三次样条插值得到平滑的 Y 值
        spline = make_interp_spline(x, mean_dist, k=3)
        y_smooth = spline(x_smooth)
        
        # 限制插值结果不小于0（防止样条拟合时出现物理上不合理的负值）
        y_smooth = np.clip(y_smooth, 0, None)
        
        plt.plot(x_smooth, y_smooth, alpha=0.3, label=f'{qstype} (n={len(type_dists[qstype])})')

    # 绘制所有数据的总平均平滑曲线
    spline_global = make_interp_spline(x, global_mean, k=3)
    y_smooth_global = np.clip(spline_global(x_smooth), 0, None)
    plt.plot(x_smooth, y_smooth_global, color='black', linewidth=3, alpha=1.0, label=f'Overall Average (n={len(all_dists)})')

    # 图表设置
    plt.title('Normalized Smooth Distribution Average by qstype')
    plt.xlabel('Distribution Index')
    plt.ylabel('Normalized Mean Value')
    
    # 因为使用插值，x 轴上的刻度设置为原本的维度点可能更直观
    plt.xticks(x) 
    
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 将图例放在图表内部
    plt.legend(loc='best')
    plt.tight_layout()  
    
    # 保存结果并显示
    pic_name = json_path.replace('.json', '_smooth_distribution.png')
    plt.savefig(pic_name, dpi=300)
    print(f"统计和平滑曲线绘图完成，结果已保存为 '{pic_name}'")

if __name__ == "__main__":
    main()