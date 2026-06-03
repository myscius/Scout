#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

VIDEO_RE = re.compile(r"Processing video:\s*(.+?)\s*$")
QUESTION_RE = re.compile(r"Question:\s*(.+?)\s*$")
RESULT_RE = re.compile(r"Result:\s*(.+?)\s*$")


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def normalize_relaxed(text: str) -> str:
    t = normalize_text(text).lower()
    return re.sub(r"[^a-z0-9]+", "", t)


def parse_result_is_correct(result_text: str) -> Optional[bool]:
    value = result_text.strip().lower()
    if value.startswith("correct"):
        return True
    if value.startswith("incorrect"):
        return False
    return None


def to_video_id(video_path: str) -> str:
    name = os.path.basename(video_path)
    return os.path.splitext(name)[0]


def load_level_index(json_path: str) -> Tuple[Dict[Tuple[str, str], str], Dict[str, List[Tuple[str, str]]]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    index: Dict[Tuple[str, str], str] = {}
    by_video: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for item in data:
        video_id = str(item.get("video_id", "")).strip()
        question = normalize_text(str(item.get("question", "")))
        level = item.get("level")
        level_key = "null" if level is None else str(level)
        if video_id and question:
            index[(video_id, question)] = level_key
            by_video[video_id].append((question, level_key))
    return index, by_video


def parse_log_records(log_path: str) -> List[dict]:
    records: List[dict] = []
    current_video_id: Optional[str] = None
    current_question: Optional[str] = None

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()

            mv = VIDEO_RE.search(line)
            if mv:
                current_video_id = to_video_id(mv.group(1).strip())
                continue

            mq = QUESTION_RE.search(line)
            if mq:
                current_question = normalize_text(mq.group(1))
                continue

            mr = RESULT_RE.search(line)
            if not mr:
                continue

            is_correct = parse_result_is_correct(mr.group(1))
            if is_correct is None:
                continue
            if not current_video_id or not current_question:
                continue

            records.append(
                {
                    "video_id": current_video_id,
                    "question": current_question,
                    "is_correct": is_correct,
                }
            )
            current_question = None
    return records


def find_level(
    video_id: str,
    question: str,
    level_index: Dict[Tuple[str, str], str],
    by_video: Dict[str, List[Tuple[str, str]]],
) -> Optional[str]:
    exact = level_index.get((video_id, question))
    if exact is not None:
        return exact

    q_relaxed = normalize_relaxed(question)
    relaxed_matches = []
    for cand_q, cand_level in by_video.get(video_id, []):
        c_relaxed = normalize_relaxed(cand_q)
        if q_relaxed == c_relaxed:
            relaxed_matches.append(cand_level)
    if len(relaxed_matches) == 1:
        return relaxed_matches[0]

    prefix_matches = []
    q_norm = normalize_text(question).lower()
    for cand_q, cand_level in by_video.get(video_id, []):
        c_norm = cand_q.lower()
        if c_norm.startswith(q_norm) or q_norm.startswith(c_norm):
            prefix_matches.append((cand_q, cand_level))
    if len(prefix_matches) == 1:
        return prefix_matches[0][1]

    if len(prefix_matches) > 1:
        prefix_matches.sort(key=lambda x: len(x[0]))
        if len(prefix_matches[0][0]) != len(prefix_matches[1][0]):
            return prefix_matches[0][1]
    return None


def compute_stats(
    records: List[dict],
    level_index: Dict[Tuple[str, str], str],
    by_video: Dict[str, List[Tuple[str, str]]],
) -> dict:
    by_level = defaultdict(lambda: {"total": 0, "correct": 0})
    unmatched: List[dict] = []

    for r in records:
        key = (r["video_id"], r["question"])
        level = find_level(r["video_id"], r["question"], level_index, by_video)
        if level is None:
            unmatched.append({"video_id": r["video_id"], "question": r["question"]})
            continue

        by_level[level]["total"] += 1
        if r["is_correct"]:
            by_level[level]["correct"] += 1

    result_levels = []
    for level in sorted(by_level.keys(), key=lambda x: (x == "null", x)):
        total = by_level[level]["total"]
        correct = by_level[level]["correct"]
        acc = (correct / total) if total else 0.0
        result_levels.append(
            {
                "level": level,
                "total": total,
                "correct": correct,
                "accuracy": acc,
            }
        )

    matched_total = sum(x["total"] for x in result_levels)
    matched_correct = sum(x["correct"] for x in result_levels)

    return {
        "summary": {
            "log_records": len(records),
            "matched_records": matched_total,
            "matched_correct": matched_correct,
            "matched_accuracy": (matched_correct / matched_total) if matched_total else 0.0,
            "unmatched_records": len(unmatched),
        },
        "by_level": result_levels,
        "unmatched_examples": unmatched[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="按 level 统计 VMME-v2 日志正确率")
    parser.add_argument("--log", required=True, help="日志文件路径")
    parser.add_argument("--json", required=True, help="题库 JSON 文件路径")
    parser.add_argument(
        "--output",
        default="vmme_level_accuracy_stats.json",
        help="输出统计结果 JSON（默认: vmme_level_accuracy_stats.json）",
    )
    args = parser.parse_args()

    level_index, by_video = load_level_index(args.json)
    log_records = parse_log_records(args.log)
    stats = compute_stats(log_records, level_index, by_video)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Level Accuracy:")
    for row in stats["by_level"]:
        print(
            f"  level={row['level']:>4} | total={row['total']:>5} | "
            f"correct={row['correct']:>5} | acc={row['accuracy']:.4f}"
        )
    s = stats["summary"]
    print(
        f"Summary: matched {s['matched_records']}/{s['log_records']}, "
        f"correct {s['matched_correct']}, acc={s['matched_accuracy']:.4f}, "
        f"unmatched={s['unmatched_records']}"
    )
    print(f"Saved: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
