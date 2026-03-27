import json
import os
from pathlib import Path
from typing import Any


def remove_samples_recursive(obj: Any) -> Any:
    """
    递归删除字典中所有层级的 'samples' 字段。

    Parameters
    ----------
    obj : Any
        要处理的对象（字典、列表或其他类型）

    Returns
    -------
    Any
        处理后的对象
    """
    if isinstance(obj, dict):
        return {
            k: remove_samples_recursive(v) for k, v in obj.items() if k != "samples"
        }
    elif isinstance(obj, list):
        return [remove_samples_recursive(item) for item in obj]
    else:
        return obj


def clean_json_files(root_dir: str) -> None:
    """
    清理指定目录下所有 JSON 文件，递归删除所有层级的 'samples' 字段。
    entropy 目录使用紧凑格式保存，其他目录使用格式化保存。

    Parameters
    ----------
    root_dir : str
        要扫描的根目录路径
    """
    root_path = Path(root_dir)
    json_files = list(root_path.rglob("*.json"))

    cleaned_count = 0
    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            cleaned_data = remove_samples_recursive(data)

            is_entropy = "entropy" in json_path.parts
            with open(json_path, "w", encoding="utf-8") as f:
                if is_entropy:
                    json.dump(cleaned_data, f, separators=(",", ":"))
                else:
                    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

            print(f"已清理: {json_path}")
            cleaned_count += 1

        except Exception as e:
            print(f"处理失败: {json_path} - {e}")

    print(f"\n完成! 共清理 {cleaned_count} 个文件")


if __name__ == "__main__":
    clean_json_files(r"d:\workspace\PyCharm\MasterThesis\results")
