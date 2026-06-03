import json

def remove_key_from_json(input_file, output_file, key_to_remove):
    """
    读取一个JSON文件，递归地移除所有指定的键，然后将结果写入新的JSON文件。

    Args:
        input_file (str): 输入的JSON文件路径。
        output_file (str): 输出的JSON文件路径。
        key_to_remove (str): 需要移除的键名。
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 递归函数来遍历和删除键
        def recursive_remove_key(obj):
            if isinstance(obj, dict):
                # 使用 list(obj.keys()) 来创建一个副本，以便在迭代时可以安全地删除键
                for key in list(obj.keys()):
                    if key == key_to_remove:
                        del obj[key]
                    else:
                        recursive_remove_key(obj[key])
            elif isinstance(obj, list):
                for item in obj:
                    recursive_remove_key(item)

        recursive_remove_key(data)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        
        print(f"成功处理文件：'{input_file}'")
        print(f"已移除所有 '{key_to_remove}' 键。")
        print(f"结果已保存到：'{output_file}'")

    except FileNotFoundError:
        print(f"错误：输入文件 '{input_file}' 未找到。")
    except json.JSONDecodeError:
        print(f"错误：无法解析 '{input_file}'。请检查文件是否为有效的JSON格式。")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    input_json_path = 'lvbench_subset_200.json'
    output_json_path = 'lvbench_subset_200_without_tmp.json'
    key_to_delete = 'sample_idx'
    
    remove_key_from_json(input_json_path, output_json_path, key_to_delete)