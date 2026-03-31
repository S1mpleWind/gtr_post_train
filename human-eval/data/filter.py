#!/usr/bin/env python3
import argparse
import json
import random
from typing import Any, Dict, List, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_passed(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "passed", "pass", "1"}:
            return True
        if s in {"false", "failed", "fail", "0"}:
            return False
    if isinstance(x, (int, float)):
        return bool(x)
    return False


def build_groups(
    qwen_rows: List[Dict[str, Any]],
    spec_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    q_map = {r["task_id"]: r for r in qwen_rows if "task_id" in r}
    s_map = {r["task_id"]: r for r in spec_rows if "task_id" in r}
    common_ids = sorted(set(q_map.keys()) & set(s_map.keys()))

    both_pass: List[Dict[str, Any]] = []
    q_only_pass: List[Dict[str, Any]] = []
    s_only_pass: List[Dict[str, Any]] = []
    both_fail: List[Dict[str, Any]] = []

    for tid in common_ids:
        q = q_map[tid]
        s = s_map[tid]
        qp = parse_passed(q.get("passed", False))
        sp = parse_passed(s.get("passed", False))
        item = {
            "task_id": tid,
            "qwen_passed": qp,
            "spec_passed": sp,
            "qwen_result": q.get("result", ""),
            "spec_result": s.get("result", ""),
            "qwen_completion": q.get("completion", ""),
            "spec_completion": s.get("completion", ""),
        }
        if qp and sp:
            both_pass.append(item)
        elif qp and (not sp):
            q_only_pass.append(item)
        elif (not qp) and sp:
            s_only_pass.append(item)
        else:
            both_fail.append(item)

    return both_pass, q_only_pass, s_only_pass, both_fail


def choose_counts(
    n: int,
    both_pass_n: int,
    q_only_n: int,
    s_only_n: int,
    both_fail_n: int,
    diff_limit: float,
) -> Tuple[int, int, int, int]:
    diff_cap = int(round(diff_limit * n))

    best = None
    best_score = -10**18

    for kq in range(min(q_only_n, n) + 1):
        max_ks = min(s_only_n, n - kq)
        for ks in range(max_ks + 1):
            if abs(kq - ks) > diff_cap:
                continue

            rem = n - kq - ks
            kbp_min = max(0, rem - both_fail_n)
            kbp_max = min(both_pass_n, rem)
            if kbp_min > kbp_max:
                continue

            # 目标:
            # 1) 优先多选“分歧样本”（q_only + s_only）
            # 2) 其次优先让通过率更高一些（多取 both_pass）
            kbp = kbp_max
            kbf = rem - kbp
            score = 1000 * (kq + ks) + kbp

            if score > best_score:
                best_score = score
                best = (kbp, kq, ks, kbf)

    if best is None:
        raise ValueError(
            f"无法找到满足条件的 {n} 条样本。"
            f"请检查数据规模，或放宽差异阈值（当前阈值={diff_limit:.3f}）。"
        )
    return best


def summarize(selected: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(selected)
    q_pass = sum(1 for x in selected if x["qwen_passed"])
    s_pass = sum(1 for x in selected if x["spec_passed"])
    q_rate = q_pass / n if n else 0.0
    s_rate = s_pass / n if n else 0.0
    return {
        "n": n,
        "qwen_pass": q_pass,
        "spec_pass": s_pass,
        "qwen_pass_rate": q_rate,
        "spec_pass_rate": s_rate,
        "abs_gap": abs(q_rate - s_rate),
    }


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_task_ids(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r["task_id"] + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="筛选 N 条样本，使 qwen/spec 通过率差距不超过阈值。"
    )
    parser.add_argument(
        "--qwen-results",
        type=str,
        required=True,
        help="qwen 结果 jsonl，例如 qwen_samples.jsonl_results.jsonl",
    )
    parser.add_argument(
        "--spec-results",
        type=str,
        required=True,
        help="spec 结果 jsonl，例如 spec_qwen_samples.jsonl_results_bk.jsonl",
    )
    parser.add_argument("--output-jsonl", type=str, required=True, help="输出筛选后的详细 jsonl")
    parser.add_argument("--output-taskids", type=str, required=True, help="输出 task_id 列表")
    parser.add_argument("--n", type=int, default=108, help="筛选条目数，默认 100")
    parser.add_argument(
        "--max-gap",
        type=float,
        default=0.05,
        help="通过率最大差值"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    qwen_rows = load_jsonl(args.qwen_results)
    spec_rows = load_jsonl(args.spec_results)

    both_pass, q_only_pass, s_only_pass, both_fail = build_groups(qwen_rows, spec_rows)

    total_common = len(both_pass) + len(q_only_pass) + len(s_only_pass) + len(both_fail)
    if total_common < args.n:
        raise ValueError(f"可对齐 task_id 只有 {total_common} 条，小于目标 {args.n} 条。")

    kbp, kq, ks, kbf = choose_counts(
        n=args.n,
        both_pass_n=len(both_pass),
        q_only_n=len(q_only_pass),
        s_only_n=len(s_only_pass),
        both_fail_n=len(both_fail),
        diff_limit=args.max_gap,
    )

    selected = []
    selected += random.sample(both_pass, kbp) if kbp > 0 else []
    selected += random.sample(q_only_pass, kq) if kq > 0 else []
    selected += random.sample(s_only_pass, ks) if ks > 0 else []
    selected += random.sample(both_fail, kbf) if kbf > 0 else []

    random.shuffle(selected)

    stats = summarize(selected)
    if stats["abs_gap"] > args.max_gap + 1e-12:
        raise RuntimeError(
            f"筛选后仍不满足阈值: gap={stats['abs_gap']:.4f}, threshold={args.max_gap:.4f}"
        )

    write_jsonl(args.output_jsonl, selected)
    write_task_ids(args.output_taskids, selected)

    print("=== Filter Summary ===")
    print(f"common tasks     : {total_common}")
    print(f"target n         : {args.n}")
    print(f"selected counts  : both_pass={kbp}, q_only_pass={kq}, spec_only_pass={ks}, both_fail={kbf}")
    print(f"qwen pass rate   : {stats['qwen_pass_rate']:.4f} ({stats['qwen_pass']}/{stats['n']})")
    print(f"spec pass rate   : {stats['spec_pass_rate']:.4f} ({stats['spec_pass']}/{stats['n']})")
    print(f"absolute gap     : {stats['abs_gap']:.4f} (<= {args.max_gap:.4f})")
    print(f"output jsonl     : {args.output_jsonl}")
    print(f"output task_ids  : {args.output_taskids}")


if __name__ == "__main__":
    main()
