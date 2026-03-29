import json
import os
from pathlib import Path
from typing import Any


def remove_samples_recursive(obj: Any) -> Any:
    """Recursively remove all 'samples' fields from all levels in a dictionary.

    Args:
        obj: The object to process (dictionary, list, or other type).

    Returns:
        The processed object with all 'samples' fields removed.
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
    """Clean all JSON files in the specified directory by recursively removing all 'samples' fields.

    Files in the 'entropy' directory are saved in compact format, while files in other
    directories are saved with formatting.

    Args:
        root_dir: The root directory path to scan.
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

            print(f"Cleaned: {json_path}")
            cleaned_count += 1

        except Exception as e:
            print(f"Failed to process: {json_path} - {e}")

    print(f"\nCompleted! Total {cleaned_count} files cleaned")


if __name__ == "__main__":
    clean_json_files(r"d:\workspace\PyCharm\MasterThesis\results")
