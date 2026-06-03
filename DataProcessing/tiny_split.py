import json
import random
from pathlib import Path

INPUT = Path("lvbench_all_result_change.json")
OUTPUT = Path("lvbench_subset_50.json")


def main():
    with INPUT.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError("Expecting a JSON array.")

    subset = data if len(data) <= 200 else random.sample(data, 50)

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
