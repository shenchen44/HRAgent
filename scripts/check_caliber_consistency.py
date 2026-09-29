"""一致性校验：口径注册表 vs 冻结的 gold SQL。

为什么必须查：
  `eval/text2sql_eval.jsonl` 的 gold SQL 是冻结的，运行时口径注册表是另一份实现。
  两份实现只要有一处口径漂移，"口径正确率"就是在跟不一致的参照物比 —— 数字全废。
  这类漂移不会报错，只会静默地让指标失真，所以必须显式对拍。

对拍范围：所有 metric 能映射到注册表的题。映射不上的（如 global 类）跳过并计数。

跑法: .venv/bin/python scripts/check_caliber_consistency.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hragent import config  # noqa: E402
from hragent.tools import caliber  # noqa: E402

# 评测集里的 metric 名 → 注册表指标名
NAME_MAP = {
    "headcount": "headcount",
    "attrition": "attrition_rate",
    "vacancy": "vacancy_rate",
    "avg_salary": "avg_salary",
    "headcount_by_level": "headcount_by_level",
    "attendance_rate": "attendance_rate",
    "avg_overtime": "avg_overtime",
    "perf_dist": "perf_dist",
    "perf_a_count": "perf_a_count",
    "recruit_completion": "recruit_completion",
    "open_hc": "open_hc",
    "avg_recruit_cycle": "avg_recruit_cycle",
    "channel_dist": "channel_dist",
    "offer_accept": "offer_accept",
    "training_completion": "training_completion",
    "social_insurance": "social_insurance",
}

# 期间标签 → (start, end)；与 gen_text2sql_eval.py 的 PERIODS 对齐
PERIOD_RANGE = {
    "2025H1": ("2025-01-01", "2025-06-30"),
    "2025H2": ("2025-07-01", "2025-12-31"),
    "2026H1": ("2026-01-01", "2026-06-30"),
}
PERIOD_TO_HP = {"2025H1": "2025H1", "2025H2": "2025H2", "2026H1": "2025H2"}


def norm(rows) -> list:
    """把结果集归一化再比：只比数值，容忍浮点尾差与行序。"""
    out = []
    for r in rows:
        out.append(tuple(round(v, 4) if isinstance(v, float) else v for v in r))
    return sorted(out, key=lambda t: tuple(str(x) for x in t))


def main() -> None:
    items = [json.loads(l) for l in (config.EVAL / "text2sql_eval.jsonl").read_text().splitlines()]
    conn = sqlite3.connect(config.DATA / "hr.db")

    checked = skipped = mismatch = 0
    fails = []
    for it in items:
        m = it["metric"]
        if m not in NAME_MAP:
            skipped += 1
            continue
        dept = it.get("dept")
        if not dept:
            skipped += 1
            continue

        # 从 gold SQL 里把参数反推出来（评测集只存了 dept 与 metric）
        kw: dict = {"dept": dept}
        gsql = it["gold_sql"]
        if m == "attrition":
            per = next((p for p in PERIOD_RANGE if PERIOD_RANGE[p][0] in gsql), None)
            if not per:
                skipped += 1
                continue
            kw["start"], kw["end"] = PERIOD_RANGE[per]
        elif m == "attendance_rate" or m == "social_insurance":
            mon = next((f"2026-{i:02d}" for i in range(1, 7) if f"'2026-{i:02d}'" in gsql), None)
            if not mon:
                skipped += 1
                continue
            kw["month"] = mon
        elif m == "avg_overtime":
            mon = next((f"2026-{i:02d}" for i in range(1, 7) if f"'2026-{i:02d}'" in gsql), None)
            if not mon:
                skipped += 1
                continue
            kw["month"] = mon
        elif m in ("perf_dist", "perf_a_count"):
            hp = next((p for p in ("2025H1", "2025H2") if f"'{p}'" in gsql), None)
            if not hp:
                skipped += 1
                continue
            kw["period"] = hp

        try:
            gold_rows = conn.execute(gsql).fetchall()
            reg = caliber.run(NAME_MAP[m], **kw)
        except Exception as e:
            fails.append((it["id"], f"执行失败 {type(e).__name__}: {e}"))
            mismatch += 1
            continue

        checked += 1
        if norm(gold_rows) != norm(reg["rows"]):
            mismatch += 1
            fails.append((it["id"], f"{it['metric']} {dept} {kw}\n"
                                     f"        gold: {norm(gold_rows)}\n"
                                     f"        注册表: {norm(reg['rows'])}"))
    conn.close()

    print(f"口径一致性对拍\n")
    print(f"   对拍 {checked} 题，跳过 {skipped} 题（global / 参数无法反推）")
    if fails:
        print(f"\n❌ {mismatch} 题不一致：")
        for i, msg in fails[:12]:
            print(f"   {i}: {msg}")
        if len(fails) > 12:
            print(f"   …另有 {len(fails) - 12} 条")
        raise SystemExit(1)
    print(f"\n✅ 全部 {checked} 题口径一致 —— 注册表与冻结 gold SQL 无漂移")


if __name__ == "__main__":
    main()
