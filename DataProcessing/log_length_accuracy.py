#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from decord import VideoReader, cpu
except Exception:  # 允许在未激活对应环境时先完成日志提取
    VideoReader = None
    cpu = None


VIDEO_RE = re.compile(r"Processing video:\s*(.+?)\s*$")
QUESTION_RE = re.compile(r"Question:\s*(.+?)\s*$")
RESULT_RE = re.compile(r"Result:\s*(.+?)\s*$")


@dataclass
class QARecord:
    log_file: str
    video_path: str
    question: str
    is_correct: bool
    result_raw: str
    duration_seconds: Optional[float]


class VideoPathResolver:
    def __init__(self, video_roots: Optional[Sequence[str]] = None, path_maps: Optional[Sequence[Tuple[str, str]]] = None):
        self.video_roots = [os.path.abspath(p) for p in (video_roots or [])]
        self.path_maps = path_maps or []

    def resolve(self, raw_path: str) -> str:
        candidates: List[str] = [raw_path]
        for old, new in self.path_maps:
            if raw_path.startswith(old):
                candidates.append(new + raw_path[len(old):])

        for c in list(candidates):
            if os.path.isabs(c):
                continue
            candidates.append(os.path.abspath(c))
            for root in self.video_roots:
                candidates.append(os.path.abspath(os.path.join(root, c)))

        for c in candidates:
            if os.path.exists(c):
                return c
        return raw_path


def parse_result_flag(result_text: str) -> Optional[bool]:
    t = result_text.strip().lower()
    if t.startswith("correct"):
        return True
    if t.startswith("incorrect"):
        return False
    return None


def parse_log_files(log_files: Sequence[str], resolver: VideoPathResolver, strict_result: bool = False) -> List[QARecord]:
    records: List[QARecord] = []
    length_cache: Dict[str, Optional[float]] = {}

    for log_file in log_files:
        current_video: Optional[str] = None
        current_question: Optional[str] = None

        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip()

                mv = VIDEO_RE.search(line)
                if mv:
                    current_video = mv.group(1).strip()
                    continue

                mq = QUESTION_RE.search(line)
                if mq:
                    current_question = mq.group(1).strip()
                    continue

                mr = RESULT_RE.search(line)
                if not mr:
                    continue

                result_raw = mr.group(1).strip()
                is_correct = parse_result_flag(result_raw)
                if is_correct is None:
                    if strict_result:
                        raise ValueError(f"无法识别的 Result 行: {line}")
                    current_question = None
                    continue

                if not current_video or not current_question:
                    if strict_result:
                        raise ValueError(f"Result 行缺少上下文: {line}")
                    continue

                resolved_video = resolver.resolve(current_video)
                if resolved_video not in length_cache:
                    length_cache[resolved_video] = get_video_duration_seconds(resolved_video)

                records.append(
                    QARecord(
                        log_file=os.path.abspath(log_file),
                        video_path=resolved_video,
                        question=current_question,
                        is_correct=is_correct,
                        result_raw=result_raw,
                        duration_seconds=length_cache[resolved_video],
                    )
                )
                current_question = None

    return records


def get_video_duration_seconds(video_path: str) -> Optional[float]:
    if not os.path.exists(video_path):
        return None
    if VideoReader is None or cpu is None:
        return None
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = float(vr.get_avg_fps())
        if fps <= 0:
            return None
        return len(vr) / fps
    except Exception:
        return None


def save_records(records: Sequence[QARecord], output_path: str) -> None:
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if output_path.endswith(".json"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=2)
        return

    if output_path.endswith(".jsonl"):
        with open(output_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        return

    if output_path.endswith(".csv"):
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["log_file", "video_path", "question", "is_correct", "result_raw", "duration_seconds"],
            )
            writer.writeheader()
            for r in records:
                writer.writerow(asdict(r))
        return

    raise ValueError("仅支持 .json / .jsonl / .csv 输出")


def load_records(input_path: str) -> List[QARecord]:
    input_path = os.path.abspath(input_path)
    if input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        return [QARecord(**x) for x in arr]

    if input_path.endswith(".jsonl"):
        out: List[QARecord] = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(QARecord(**json.loads(line)))
        return out

    if input_path.endswith(".csv"):
        out: List[QARecord] = []
        with open(input_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ds = row.get("duration_seconds")
                out.append(
                    QARecord(
                        log_file=row.get("log_file", ""),
                        video_path=row.get("video_path", ""),
                        question=row.get("question", ""),
                        is_correct=str(row.get("is_correct", "")).lower() in {"true", "1", "yes"},
                        result_raw=row.get("result_raw", ""),
                        duration_seconds=float(ds) if ds not in {"", None} else None,
                    )
                )
        return out

    raise ValueError("仅支持 .json / .jsonl / .csv 输入")


def parse_bins(bin_str: str) -> List[float]:
    vals = [float(x.strip()) for x in bin_str.split(",") if x.strip()]
    if len(vals) < 2:
        raise ValueError("bins 至少需要两个边界值，例如: 0,60,180")
    if any(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)):
        raise ValueError("bins 必须严格递增")
    return vals


def bin_label(left: float, right: float, is_last_open: bool = False) -> str:
    if is_last_open:
        return f"[{left:.0f}, +inf)"
    return f"[{left:.0f}, {right:.0f})"


def compute_bin_stats(records: Sequence[QARecord], bins: Sequence[float]) -> List[dict]:
    stats = []
    for i in range(len(bins) - 1):
        stats.append(
            {
                "bin": bin_label(bins[i], bins[i + 1]),
                "left": bins[i],
                "right": bins[i + 1],
                "total": 0,
                "correct": 0,
            }
        )
    stats.append(
        {
            "bin": bin_label(bins[-1], math.inf, is_last_open=True),
            "left": bins[-1],
            "right": math.inf,
            "total": 0,
            "correct": 0,
        }
    )

    unknown = 0
    for r in records:
        d = r.duration_seconds
        if d is None:
            unknown += 1
            continue
        for s in stats:
            if s["left"] <= d < s["right"]:
                s["total"] += 1
                s["correct"] += int(r.is_correct)
                break

    for s in stats:
        s["accuracy"] = (s["correct"] / s["total"]) if s["total"] > 0 else None
    return stats, unknown


def save_stats(stats: Sequence[dict], unknown_count: int, output_path: str) -> None:
    payload = {"stats": list(stats), "unknown_duration_count": unknown_count}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_stats(stats: Sequence[dict], unknown_count: int) -> None:
    print("\n长度分桶统计：")
    print(f"{'bin':<16} {'total':>8} {'correct':>8} {'accuracy':>10}")
    for s in stats:
        acc = "N/A" if s["accuracy"] is None else f"{s['accuracy']*100:.2f}%"
        print(f"{s['bin']:<16} {s['total']:>8} {s['correct']:>8} {acc:>10}")
    print(f"\n未知视频长度条目: {unknown_count}")


def parse_path_map(items: Optional[Sequence[str]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--path-map 需要 old=new 格式，收到: {item}")
        old, new = item.split("=", 1)
        old, new = old.strip(), new.strip()
        if not old or not new:
            raise ValueError(f"--path-map 非法: {item}")
        out.append((old, new))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="从日志提取 (视频, 问题, 正误, 时长) 并按时长分桶统计精度。"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common_extract_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--logs", nargs="+", required=True, help="手动指定一个或多个 log 文件")
        sp.add_argument(
            "--video-root",
            action="append",
            default=[],
            help="当日志里的视频路径不可直接访问时，尝试在该根目录下拼接查找，可重复传入",
        )
        sp.add_argument(
            "--path-map",
            action="append",
            default=[],
            help="路径前缀映射，格式 old_prefix=new_prefix，可重复传入",
        )
        sp.add_argument("--strict-result", action="store_true", help="遇到异常 Result/缺上下文时直接报错")

    p_extract = sub.add_parser("extract", help="仅提取日志为结构化结果")
    add_common_extract_args(p_extract)
    p_extract.add_argument("--output", required=True, help="提取结果输出文件 (.json/.jsonl/.csv)")

    p_stats = sub.add_parser("stats", help="按时长分桶统计精度")
    src = p_stats.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="extract 阶段产物文件 (.json/.jsonl/.csv)")
    src.add_argument("--logs", nargs="+", help="直接从 log 解析后统计")
    p_stats.add_argument("--video-root", action="append", default=[])
    p_stats.add_argument("--path-map", action="append", default=[])
    p_stats.add_argument("--strict-result", action="store_true")
    p_stats.add_argument(
        "--bins",
        default="0,60,180,300,600,1200",
        help="显式长度分桶边界（秒），逗号分隔，最后自动扩展到 +inf",
    )
    p_stats.add_argument("--stats-output", help="可选，保存统计结果 JSON")

    p_run = sub.add_parser("run", help="线性执行：先提取，再统计")
    add_common_extract_args(p_run)
    p_run.add_argument("--output", required=True, help="提取结果输出文件 (.json/.jsonl/.csv)")
    p_run.add_argument(
        "--bins",
        default="0,60,180,300,600,1200",
        help="显式长度分桶边界（秒），逗号分隔，最后自动扩展到 +inf",
    )
    p_run.add_argument("--stats-output", help="可选，保存统计结果 JSON")
    return p


def extract_records_from_args(args: argparse.Namespace) -> List[QARecord]:
    resolver = VideoPathResolver(
        video_roots=args.video_root,
        path_maps=parse_path_map(args.path_map),
    )
    return parse_log_files(
        log_files=args.logs,
        resolver=resolver,
        strict_result=args.strict_result,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "extract":
        records = extract_records_from_args(args)
        save_records(records, args.output)
        print(f"提取完成：{len(records)} 条，已保存到 {os.path.abspath(args.output)}")
        return

    if args.cmd == "stats":
        if args.input:
            records = load_records(args.input)
        else:
            records = extract_records_from_args(args)
        bins = parse_bins(args.bins)
        stats, unknown = compute_bin_stats(records, bins)
        print_stats(stats, unknown)
        if args.stats_output:
            save_stats(stats, unknown, args.stats_output)
            print(f"统计结果已保存到 {os.path.abspath(args.stats_output)}")
        return

    if args.cmd == "run":
        records = extract_records_from_args(args)
        save_records(records, args.output)
        print(f"提取完成：{len(records)} 条，已保存到 {os.path.abspath(args.output)}")
        bins = parse_bins(args.bins)
        stats, unknown = compute_bin_stats(records, bins)
        print_stats(stats, unknown)
        if args.stats_output:
            save_stats(stats, unknown, args.stats_output)
            print(f"统计结果已保存到 {os.path.abspath(args.stats_output)}")
        return

    raise RuntimeError(f"未知命令: {args.cmd}")


if __name__ == "__main__":
    main()
