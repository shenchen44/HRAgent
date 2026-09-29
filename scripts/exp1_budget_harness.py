"""实验 1：预算阶梯 Harness 的收益量化。

三个条件，逐步加工程手段，隔离每个机制的贡献：
  A 朴素   : thinking 开 + 固定 max_tokens=2048（无阶梯）—— 大多数教程的默认写法
  B +阶梯  : thinking 开 + 预算阶梯 4096→8192→16384→32768
  C +关思考: thinking 关 + 预算阶梯

指标：空回复率、截断率、准确率、平均 tokens、P50/P95 延迟。
跑法: .venv/bin/python scripts/exp1_budget_harness.py
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hragent import config  # noqa: E402
from hragent.llm import LLM, Ledger  # noqa: E402

QUESTIONS = json.loads((config.DATA / "hr_numeric_qa.json").read_text())
WORKERS = 6

CONDITIONS = {
    "A_朴素(think开,固定2048)": dict(thinking=True, max_tokens=2048),
    "B_加阶梯(think开)": dict(thinking=True),
    "C_加关思考(阶梯)": dict(thinking=False),
}

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def extract_numbers(text: str) -> list[float]:
    return [float(m) for m in NUM.findall(text.replace(",", ""))]


def is_correct(text: str, expected: float) -> bool:
    """答案里出现过期望值即算对（容差 1% 或 0.01）。各条件同等宽松，不偏袒。"""
    tol = max(abs(expected) * 0.01, 0.01)
    return any(abs(n - expected) <= tol for n in extract_numbers(text))


def run_one(cfg: dict, item: dict) -> dict:
    llm = LLM(**cfg)
    t0 = time.time()
    try:
        reply = llm.ask(item["q"], tag=item["id"])
        err = None
    except Exception as e:
        reply, err = None, str(e)
    dt = time.time() - t0

    if reply is None:
        return {"id": item["id"], "ok": False, "empty": True, "truncated": True,
                "correct": False, "out_tokens": 0, "latency_s": dt, "error": err,
                "text": "", "ledger": llm.ledger}

    return {
        "id": item["id"],
        "ok": True,
        "empty": reply.empty,
        "truncated": reply.truncated,
        "correct": is_correct(reply.text, item["answer"]),
        "out_tokens": reply.usage.get("output_tokens", 0),
        "latency_s": dt,
        "attempts": reply.attempts,
        "max_tokens_used": reply.max_tokens_used,
        "text": reply.text.strip()[:120],
        "error": None,
        "ledger": llm.ledger,
    }


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> None:
    all_results = {}

    for name, cfg in CONDITIONS.items():
        print(f"\n{'=' * 78}\n### {name}\n{'=' * 78}", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            rows = list(ex.map(lambda it: run_one(cfg, it), QUESTIONS))
        wall = time.time() - t0

        merged = Ledger()
        for r in rows:
            led = r["ledger"]
            merged.calls += led.calls
            merged.input_tokens += led.input_tokens
            merged.output_tokens += led.output_tokens
            merged.cache_read_tokens += led.cache_read_tokens
            merged.truncated_retries += led.truncated_retries
            merged.total_latency_s += led.total_latency_s

        lat = sorted(r["latency_s"] for r in rows)
        n = len(rows)
        stats = {
            "n": n,
            "empty_rate": sum(r["empty"] for r in rows) / n,
            "truncated_rate": sum(r["truncated"] for r in rows) / n,
            "accuracy": sum(r["correct"] for r in rows) / n,
            "mean_out_tokens": statistics.mean(r["out_tokens"] for r in rows),
            "total_tokens": merged.input_tokens + merged.output_tokens,
            "p50_latency": lat[n // 2],
            "p95_latency": lat[min(int(n * 0.95), n - 1)],
            "wall_s": round(wall, 1),
            "truncated_retries": merged.truncated_retries,
        }
        all_results[name] = {"stats": stats, "rows": rows}

        print(f"  空回复率   {pct(stats['empty_rate'])}  ({sum(r['empty'] for r in rows)}/{n})")
        print(f"  截断率     {pct(stats['truncated_rate'])}")
        print(f"  准确率     {pct(stats['accuracy'])}")
        print(f"  平均输出   {stats['mean_out_tokens']:.0f} tok")
        print(f"  总 tokens  {stats['total_tokens']:,}")
        print(f"  延迟 P50/P95  {stats['p50_latency']:.1f}s / {stats['p95_latency']:.1f}s")
        print(f"  阶梯救回   {stats['truncated_retries']} 次")
        for r in rows:
            flag = "✅" if r["correct"] else ("⬜空" if r["empty"] else "❌")
            print(f"    {flag} {r['id']} tok={r['out_tokens']:5d} att={r.get('attempts', '-')} "
                  f"{r['latency_s']:5.1f}s {r['text'][:70]!r}")

    out = config.RESULTS / "exp1_budget_harness.json"
    out.write_text(json.dumps(
        {k: {"stats": v["stats"], "rows": [{kk: vv for kk, vv in r.items() if kk != "ledger"} for r in v["rows"]]}
         for k, v in all_results.items()},
        ensure_ascii=False, indent=2))

    print(f"\n{'=' * 78}\n### 汇总对比\n{'=' * 78}")
    hdr = f"{'条件':28s} {'空回复':>8s} {'截断':>8s} {'准确率':>8s} {'均tok':>8s} {'总tok':>10s} {'P95':>8s}"
    print(hdr)
    for name, v in all_results.items():
        s = v["stats"]
        print(f"{name:28s} {pct(s['empty_rate']):>8s} {pct(s['truncated_rate']):>8s} "
              f"{pct(s['accuracy']):>8s} {s['mean_out_tokens']:>8.0f} {s['total_tokens']:>10,} "
              f"{s['p95_latency']:>7.1f}s")
    print(f"\n明细已写入 {out}")


if __name__ == "__main__":
    main()
