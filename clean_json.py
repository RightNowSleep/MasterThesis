"""Utility script for cleaning JSON result files by removing sample data.

Recursively scans a directory tree for ``.json`` files and strips all keys
named ``"samples"`` from every level of nesting (dicts and lists). This is
useful for reducing file size when the raw sample token sequences are not
needed for downstream analysis.

Files residing in directories whose path contains ``"entropy"`` are saved in
compact JSON format (no whitespace) to further minimize size; all other files
are saved with standard 2-space indentation for readability.
"""

import json
import os
from pathlib import Path
from typing import Any


def remove_samples_recursive(obj: Any) -> Any:
    """Recursively remove all ``"samples"`` fields from a nested data structure.

    Walks through dictionaries and lists, filtering out any dictionary key
    exactly equal to ``"samples"`` while preserving all other data intact.

    Args:
        obj: The object to process — may be a dict, list, or scalar value.

    Returns:
        A new object of the same structure as *obj* but with every
        ``"samples"`` key removed from all nested dictionaries.
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
    """Clean all JSON files under *root_dir* by stripping ``"samples"`` fields.

    Discovers every ``.json`` file recursively via :func:`pathlib.Path.rglob`,
    loads each one, applies :func:`remove_samples_recursive`, and writes the
    cleaned data back to the same path.

    Args:
        root_dir: The root directory path to scan for JSON files.

    Returns:
        None

    Side Effects:
        - Overwrites each discovered JSON file in-place.
        - Prints progress and a summary count to stdout.
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
