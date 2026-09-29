"""冻结评测集：打 SHA256，并支持事后校验。

为什么冻结要可强制执行：
  只写一个 hash 文件是装饰品。真正要防的是 —— 开发过程中"顺手"改了数据或改了金标，
  然后拿旧分数交差。所以必须有 --verify，且在 P2/P3 跑实验前先过一遍。

用法:
  .venv/bin/python scripts/freeze_eval.py            # 冻结（已存在则拒绝）
  .venv/bin/python scripts/freeze_eval.py --verify   # 校验当前文件是否仍与冻结一致
  .venv/bin/python scripts/freeze_eval.py --force    # 重新冻结（作废既有结果，需自担后果）
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hragent import config  # noqa: E402

FROZEN = config.EVAL / "FROZEN.json"

# 冻结范围：评测集 + 金标所依赖的数据资产 + 生成它们的脚本
# 数据资产也在内 —— 数据一变，gold SQL 的结果就变，分数就不可比
TARGETS = [
    "eval/intent_set.jsonl",
    "eval/text2sql_eval.jsonl",
    "eval/rag_eval.jsonl",
    "eval/risk_eval.jsonl",
    "eval/skill_eval.jsonl",
    "eval/match_eval.jsonl",
    "data/hr.db",
    "data/policy_chunks.jsonl",
    "data/hr_data_dictionary.md",
    "data/policy_gaps.md",
]
GENERATORS = [
    "scripts/gen_hr_db.py",
    "scripts/build_policy_index.py",
    "scripts/gen_intent_set.py",
    "scripts/gen_text2sql_eval.py",
    "scripts/gen_rag_eval.py",
    "scripts/gen_risk_eval.py",
    "scripts/gen_match_eval.py",
    "src/hragent/metrics.py",
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def fingerprint(rel: str) -> dict:
    """只做 hash 与大小 —— 对任何字节序列都成立，不会因文件损坏而失败。

    校验路径必须先走这里：验证器的职责是发现坏状态，它自己不能在坏状态上挂掉。
    """
    p = config.ROOT / rel
    if not p.exists():
        raise SystemExit(f"❌ 冻结目标不存在: {rel}")
    return {"sha256": sha256(p), "bytes": p.stat().st_size}


def metadata(rel: str) -> dict:
    """人类可读的元信息（条数/分布/表行数）。尽力而为，解析失败就返回空。"""
    p = config.ROOT / rel
    d: dict = {}
    try:
        if p.suffix == ".jsonl":
            rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            d["records"] = len(rows)
            if rows and "split" in rows[0]:
                d["splits"] = dict(sorted(Counter(r["split"] for r in rows).items()))
            if rows and "kind" in rows[0]:
                d["kinds"] = dict(sorted(Counter(r["kind"] for r in rows).items()))
        elif p.suffix == ".db":
            conn = sqlite3.connect(p)
            tabs = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            d["tables"] = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                           for t in tabs}
            conn.close()
    except Exception as e:  # 文件损坏时不应让校验流程崩溃
        d["meta_error"] = f"{type(e).__name__}: {e}"
    return d


def describe(rel: str) -> dict:
    return {**fingerprint(rel), **metadata(rel)}


def build() -> dict:
    return {
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "note": "冻结后任何改动都会使既有实验结果不可比。改数据 = 重跑全部实验。",
        "files": {rel: describe(rel) for rel in TARGETS},
        "generators": {rel: sha256(config.ROOT / rel) for rel in GENERATORS
                       if (config.ROOT / rel).exists()},
    }


def verify() -> int:
    if not FROZEN.exists():
        print("❌ 尚未冻结。先跑 scripts/freeze_eval.py")
        return 1
    old = json.loads(FROZEN.read_text())
    print(f"对照冻结时间: {old['frozen_at']}\n")
    bad = 0
    for rel, rec in old["files"].items():
        try:
            cur_fp = fingerprint(rel)
        except SystemExit:
            print(f"   ❌ {rel:34s} 文件已丢失")
            bad += 1
            continue
        if cur_fp["sha256"] != rec["sha256"]:
            print(f"   ❌ {rel:34s} 内容已变")
            print(f"        冻结 {rec['sha256'][:16]}…  ({rec['bytes']} B)")
            print(f"        当前 {cur_fp['sha256'][:16]}…  ({cur_fp['bytes']} B)")
            cur_meta = metadata(rel)
            if "meta_error" in cur_meta:
                print(f"        ⚠️  文件已损坏，无法解析: {cur_meta['meta_error'][:70]}")
            elif "records" in rec and rec.get("records") != cur_meta.get("records"):
                print(f"        条数 {rec['records']} → {cur_meta.get('records')}")
            bad += 1
        else:
            extra = f"{rec.get('records', '')} 条" if "records" in rec else ""
            print(f"   ✅ {rel:34s} {extra}")
    for rel, h in old["generators"].items():
        p = config.ROOT / rel
        if not p.exists():
            print(f"   ⚠️  {rel:34s} 生成脚本已丢失")
        elif sha256(p) != h:
            print(f"   ⚠️  {rel:34s} 生成脚本已改（数据未变则无妨，但请确认不是手改数据）")
    print()
    if bad:
        print(f"❌ {bad} 个文件与冻结不一致 —— 既有实验结果已不可比，须重跑")
        return 1
    print("✅ 全部与冻结一致")
    return 0


def main() -> None:
    args = sys.argv[1:]
    if "--verify" in args:
        raise SystemExit(verify())

    history = []
    if FROZEN.exists():
        if "--force" not in args:
            print(f"❌ 已存在冻结文件: {FROZEN}")
            print("   冻结是单向操作。要校验请用 --verify；确实要重冻结请用 --force")
            print("   注意：重冻结会使所有既有实验结果失去可比性。")
            raise SystemExit(1)
        # 重冻结必须留痕：记录上一版每个文件的 hash 与改了什么，不能静默覆盖
        old = json.loads(FROZEN.read_text())
        history = old.get("history", [])
        reason = "（未说明理由）"
        if "--reason" in args:
            reason = args[args.index("--reason") + 1]
        new_data = build()
        changed = []
        for rel, rec in old["files"].items():
            cur = new_data["files"].get(rel)
            if cur is None:
                changed.append(f"{rel}: 移出冻结范围")
            elif cur["sha256"] != rec["sha256"]:
                changed.append(f"{rel}: {rec['sha256'][:12]}… → {cur['sha256'][:12]}…")
        for rel in new_data["files"]:
            if rel not in old["files"]:
                changed.append(f"{rel}: 新增")
        # **生成脚本的指纹也必须进 history。**
        # 它恰恰是最该留痕的一项：数据没变、`--verify` 的 files 检查全绿，
        # 但指标定义（`metrics.py`）或金标生成脚本改了 —— 这时"数据与冻结一致"
        # 是真的，"结果可比"却未必。只记 files 会让这一项在重冻结时静默消失。
        gen_changed = []
        for rel, h in old.get("generators", {}).items():
            cur = new_data["generators"].get(rel)
            if cur is None:
                gen_changed.append(f"{rel}: 移出生成脚本清单")
            elif cur != h:
                gen_changed.append(f"{rel}: {h[:12]}… → {cur[:12]}…")
        for rel in new_data["generators"]:
            if rel not in old.get("generators", {}):
                gen_changed.append(f"{rel}: 新增生成脚本")
        history.append({
            "replaced_at": old["frozen_at"],
            "replaced_by": new_data["frozen_at"],
            "reason": reason,
            "changes": changed,
            "generator_changes": gen_changed,
            "old_hashes": {rel: rec["sha256"] for rel, rec in old["files"].items()},
            "old_generator_hashes": dict(old.get("generators", {})),
        })
        print("⚠️  重冻结 —— 上一版记录已存入 history：")
        for c in changed:
            print(f"      {c}")
        for c in gen_changed:
            print(f"      [生成脚本] {c}")
        print(f"   理由: {reason}\n")

    data = build()
    if history:
        data["history"] = history
    FROZEN.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"✅ 已冻结 {len(data['files'])} 个文件 → {FROZEN}\n")
    print(f"{'文件':36s} {'大小':>10s}  {'条数/行数':>10s}  分布")
    for rel, d in data["files"].items():
        n = d.get("records", "")
        if not n and "tables" in d:
            n = sum(d["tables"].values())
        dist = ""
        if "splits" in d:
            dist = str(d["splits"])
        elif "tables" in d:
            dist = f"{len(d['tables'])} 张表"
        print(f"{rel:36s} {d['bytes']:>10,d}  {str(n):>10}  {dist}")
    print(f"\n   生成脚本指纹 {len(data['generators'])} 个")
    print(f"\n   冻结时间: {data['frozen_at']}")
    print("   P1 完成 —— 此后进入 P2 主链路开发，数据与金标不再改动。")


if __name__ == "__main__":
    main()
