import site
import os

# 打印 site-packages 的路径
for path in site.getsitepackages():
    # 查找包含 nvidia_cufile 的文件夹
    if os.path.exists(path):
        print(f"检查目录: {path}")