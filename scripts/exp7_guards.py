"""实验 7：风控闸门逐道评测（拦截率必须与误报率成对报告）。

为什么不能只报一个"拦截率"：
  全部拦截 = 100% 拦截率。单独看拦截率，一道"永远返回 block"的闸门是完美的。
  风控集因此严格 1:1 配平（110 正常 : 110 注入），任何拦截率都必须与误报率同时出现。

三种召回率，必须分开报 —— 这是本项目最容易自欺的地方：
  · 指定闸门召回   故障声明的 expected_guard 是否真的拦住了它
                   （衡量"这道闸门本身有没有用"）
  · 任一闸门召回   是否有任何闸门拦住了它
                   （衡量"系统整体安不安全"）
  · 误报率         正常样本被误拦的比例
                   （衡量"代价有多大"）

两者会差很多：歧视样本的 evidence 为空，GroundingGuard 也会拦。
生产上无所谓（拦住就行），但做归因时必须说清楚，否则会把功劳记错闸门。

跑法:
  .venv/bin/python scripts/exp7_guards.py --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hragent import config  # noqa: E402
from hragent.risk.guards import (RiskAgent, candidate_from_payload,  # noqa: E402
                                 DEFAULT_GUARDS)

ALL_GUARDS = ("truncation", "grounding", "compliance")


def load(split: str, which: str) -> list[dict]:
    files = {"orig": ["risk_eval.jsonl"],
             "supplement": ["risk_eval_supplement.jsonl"],
             "both": ["risk_eval.jsonl", "risk_eval_supplement.jsonl"]}[which]
    rows = []
    for fn in files:
        p = config.EVAL / fn
        if not p.exists():
            raise SystemExit(f"❌ 缺少 {p}（补测集请先跑 scripts/gen_risk_supplement.py）")
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            r.setdefault("cell", "-")
            if split == "all" or r.get("split", "test") == split or "cell" in r:
                rows.append(r)
    return rows


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def zero_knowledge_baseline(rows: list[dict]) -> tuple[int, int, int]:
    """零知识基线：**只看证据是否为空**就拦。

    这条规则不需要任何领域知识、任何模型、任何代码逻辑：
        block  ⟺  len(evidence) == 0

    任何风控指标都必须与它对比才有意义。原评测集里这条规则能拿到
    66/110 召回、0/110 误报 —— 因为故障类型与「证据有无」完全共线。
    不报这个基线，110/110 会被误读成"闸门很强"。
    """
    tp = sum(1 for r in rows if r["expected_verdict"] == "block"
             and not (r["payload"].get("evidence") or []))
    fn = sum(1 for r in rows if r["expected_verdict"] == "block"
             and (r["payload"].get("evidence") or []))
    fp = sum(1 for r in rows if r["expected_verdict"] == "pass"
             and not (r["payload"].get("evidence") or []))
    return tp, fp, fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--set", dest="which", default="orig",
                    choices=["orig", "supplement", "both"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = load(args.split, args.which)
    faults = [r for r in rows if r["expected_verdict"] == "block"]
    normals = [r for r in rows if r["expected_verdict"] == "pass"]
    print(f"样本 {len(rows)} 条：注入 {len(faults)} · 正常 {len(normals)}"
          f"  （集：{args.which} / split={args.split}）")
    print(f"按故障类型: {dict(Counter(r['kind'] for r in faults))}")

    cands = {r["id"]: candidate_from_payload(r["payload"]) for r in rows}

    # ---------- 零知识基线 ----------
    ztp, zfp, zfn = zero_knowledge_baseline(rows)
    zp, zr, zf = prf(ztp, zfp, zfn)
    print(f"\n{'=' * 74}")
    print("零知识基线：block ⟺ evidence 为空（不需要任何检测能力）")
    print(f"{'=' * 74}")
    print(f"{'':<12}{ztp:>4}{zfp:>5}{zfn:>5}  {zp:>7.3f}{zr:>7.3f}{zf:>7.3f}   "
          f"{zfp/len(normals) if normals else 0:>8.3f}")
    print("  ⚠️  任何闸门成绩都必须与这一行对比。若与之持平，说明没测出检测能力。")

    # ---------- 单闸门 ----------
    print(f"\n{'=' * 74}")
    print("单闸门（每次只开一道）")
    print(f"{'=' * 74}")
    print(f"{'闸门':<12}{'TP':>4}{'FP':>5}{'FN':>5}  {'P':>7}{'R':>7}{'F1':>7}   {'误报率':>8}")
    single = {}
    for g in ALL_GUARDS:
        agent = RiskAgent(enabled=(g,))
        tp = fp = fn = 0
        for r in faults:
            if agent.blocked(agent.review(cands[r["id"]])):
                tp += 1
            else:
                fn += 1
        for r in normals:
            if agent.blocked(agent.review(cands[r["id"]])):
                fp += 1
        p, rc, f = prf(tp, fp, fn)
        fpr = fp / len(normals) if normals else 0.0
        single[g] = {"tp": tp, "fp": fp, "fn": fn, "precision": p,
                     "recall": rc, "f1": f, "fpr": fpr}
        print(f"{g:<12}{tp:>4}{fp:>5}{fn:>5}  {p:>7.3f}{rc:>7.3f}{f:>7.3f}   {fpr:>8.3f}")

    # ---------- 组合 ----------
    agent_all = RiskAgent(enabled=ALL_GUARDS)
    tp = fp = fn = 0
    for r in faults:
        if agent_all.blocked(agent_all.review(cands[r["id"]])):
            tp += 1
        else:
            fn += 1
    for r in normals:
        if agent_all.blocked(agent_all.review(cands[r["id"]])):
            fp += 1
    p, rc, f = prf(tp, fp, fn)
    print(f"\n{'组合（三道全开）':<12}{tp:>4}{fp:>5}{fn:>5}  "
          f"{p:>7.3f}{rc:>7.3f}{f:>7.3f}   {fp/len(normals):>8.3f}")

    # ---------- 归因：谁拦住了什么 ----------
    print(f"\n{'=' * 74}")
    print("归因（每类故障由哪道闸门拦住）")
    print(f"{'=' * 74}")
    print(f"{'故障类型':<18}{'n':>4}  {'指定闸门命中':>12}{'任一闸门命中':>13}   拦截者分布")
    attrib = {}
    for kind in sorted({r["kind"] for r in faults}):
        sub = [r for r in faults if r["kind"] == kind]
        desig = any_hit = 0
        who = Counter()
        for r in sub:
            c = cands[r["id"]]
            v = agent_all.review(c)
            blockers = [x.guard for x in v if x.severity == "block" and not x.passed]
            eg = r["expected_guard"]
            if eg and eg in blockers:
                desig += 1
            if blockers:
                any_hit += 1
            who["+".join(sorted(blockers)) or "漏放"] += 1
        n = len(sub)
        attrib[kind] = {"n": n, "designated_hit": desig, "any_hit": any_hit,
                        "blockers": dict(who)}
        print(f"{kind:<18}{n:>4}  {desig:>7}/{n:<4}{any_hit:>8}/{n:<4}   {dict(who)}")

    # ---------- 分格子（补测集专用）----------
    cells = sorted({r["cell"] for r in rows if r["cell"] != "-"})
    cell_stats = {}
    if cells:
        print(f"\n{'=' * 74}")
        print("分格子（补测集的核心产出：每个格子回答一个独立问题）")
        print(f"{'=' * 74}")
        for cell in cells:
            sub = [r for r in rows if r["cell"] == cell]
            f = [r for r in sub if r["expected_verdict"] == "block"]
            n = [r for r in sub if r["expected_verdict"] == "pass"]
            fh = sum(1 for r in f if agent_all.blocked(agent_all.review(cands[r["id"]])))
            nh = sum(1 for r in n if agent_all.blocked(agent_all.review(cands[r["id"]])))
            cell_stats[cell] = {"n_faults": len(f), "n_normals": len(n),
                                "faults_blocked": fh, "normals_blocked": nh}
            if f:
                print(f"  [{cell}] 注入 {fh}/{len(f)} 拦住"
                      f"   ← 该格子的检测能力")
            if n:
                print(f"  [{cell}] 正常 {nh}/{len(n)} 误拦"
                      f"   ← 该格子的误报代价")

    # ---------- 误报明细 ----------
    print(f"\n{'=' * 74}")
    print(f"误报明细（正常样本被误拦，共 {fp} 条）")
    print(f"{'=' * 74}")
    fp_detail = []
    for r in normals:
        v = agent_all.review(cands[r["id"]])
        bad = [x for x in v if x.severity == "block" and not x.passed]
        if bad:
            fp_detail.append({"id": r["id"], "kind": r["kind"],
                              "guards": [x.guard for x in bad],
                              "reasons": [x.reason for x in bad],
                              "answer": r["payload"].get("answer", "")[:90]})
            print(f"  [{r['id']}] {bad[0].guard}: {bad[0].reason[:100]}")
    if not fp_detail:
        print("  （无）")

    # ---------- 漏放明细 ----------
    print(f"\n{'=' * 74}")
    print(f"漏放明细（注入样本未被拦，共 {fn} 条）")
    print(f"{'=' * 74}")
    fn_detail = []
    for r in faults:
        v = agent_all.review(cands[r["id"]])
        if not agent_all.blocked(v):
            fn_detail.append({"id": r["id"], "kind": r["kind"],
                              "expected_guard": r["expected_guard"],
                              "subtlety": r.get("subtlety"),
                              "answer": r["payload"].get("answer", "")[:90]})
            print(f"  [{r['id']}] {r['kind']}/{r.get('subtlety')} "
                  f"期望 {r['expected_guard']}: {r['payload'].get('answer','')[:80]}")
    if not fn_detail:
        print("  （无）")

    out = Path(args.out) if args.out else \
        config.RESULTS / f"exp7_guards_{args.which}_{args.split}.json"
    out.write_text(json.dumps({
        "set": args.which, "split": args.split,
        "n_faults": len(faults), "n_normals": len(normals),
        "zero_knowledge_baseline": {"tp": ztp, "fp": zfp, "fn": zfn,
                                    "precision": zp, "recall": zr, "f1": zf,
                                    "fpr": zfp / len(normals) if normals else 0.0},
        "single": single,
        "combined": {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": rc,
                     "f1": f, "fpr": fp / len(normals) if normals else 0.0},
        "attribution": attrib, "cells": cell_stats,
        "false_positives": fp_detail, "false_negatives": fn_detail,
    }, ensure_ascii=False, indent=2))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
