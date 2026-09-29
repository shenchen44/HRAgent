"""实验 3：分层路由 vs 单 Agent 扁平工具。

要回答的问题：
  指挥官能不能正确判断"该调哪个执行体"和"该不该先澄清"？

为什么路由要单独测：
  路由错与执行错是两种失败。混在一起报一个端到端准确率，
  就无法回答"是没找对人，还是找对了人但没干好"。
  Trace 里 route 与 results 分开存，这里只测 route。

指标：
  · 执行体集合完全匹配率（主指标）
  · 澄清决策准确率（该澄清的澄清了没 / 不该澄清的有没有乱澄清）
  · 成本（R2：准确率必须与成本成对报告）

跑法:
  .venv/bin/python scripts/exp3_routing.py --split dev --limit 40   # 开发期抽样
  .venv/bin/python scripts/exp3_routing.py --split test            # 出数
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hragent import config, metrics  # noqa: E402
from hragent.agents.orchestrator import Orchestrator  # noqa: E402
from hragent.llm import LLM  # noqa: E402

WORKERS = 6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--limit", type=int, default=0, help="抽样条数，0 表示全量")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-baseline", action="store_true", help="跳过扁平工具对照组")
    args = ap.parse_args()

    items = [json.loads(l) for l in (config.EVAL / "intent_set.jsonl").read_text().splitlines()]
    if args.split != "all":
        items = [x for x in items if x["split"] == args.split]
    if args.limit:
        items = items[:args.limit]

    print(f"实验 3：分层路由 vs 单 Agent 扁平工具")
    print(f"   样本 {len(items)} 条（split={args.split}）· 歧义阈值 {args.threshold}\n")

    llm = LLM()
    orch = Orchestrator(llm=llm, ambiguity_threshold=args.threshold)
    ledger_before = dict(llm.ledger.summary())

    def one(it: dict) -> dict:
        t0 = time.time()
        try:
            d = orch.route(it["query"])
            pred_exec = sorted(d.executors)
            pred_clar = d.needs_clarification
            amb = d.ambiguity
            err = None
        except Exception as e:
            pred_exec, pred_clar, amb, err = [], False, 0.0, f"{type(e).__name__}: {e}"
        return {"id": it["id"], "layer": it["layer"], "query": it["query"],
                "gold_exec": sorted(it["gold_executors"]), "pred_exec": pred_exec,
                "gold_intents": it["gold_intents"],
                "gold_behavior": it["expected_behavior"],
                "pred_clar": pred_clar, "ambiguity": amb,
                "latency_s": time.time() - t0, "error": err}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        rows = list(ex.map(one, items))
    wall = time.time() - t0
    led_after_layered = dict(llm.ledger.summary())

    # ---------------- 对照组：单 Agent 扁平工具
    base_rows = []
    base_wall = base_tok = None
    if not args.no_baseline:
        from hragent.agents import flat_baseline as FB

        def one_flat(it: dict) -> dict:
            t0 = time.time()
            try:
                r = llm.call(FB.build_messages(it["query"]), system=FB.FLAT_SYSTEM,
                             tag="flat")
                tools, pred, clar = FB.parse(r.json())
                err = None
            except Exception as e:
                tools, pred, clar, err = [], [], False, f"{type(e).__name__}: {e}"
            return {"id": it["id"], "layer": it["layer"], "query": it["query"],
                    "gold_exec": sorted(it["gold_executors"]), "pred_exec": pred,
                    "flat_tools": tools, "pred_clar": clar,
                    "latency_s": time.time() - t0, "error": err}

        tb = time.time()
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            base_rows = list(ex.map(one_flat, items))
        base_wall = time.time() - tb
        led_after_flat = dict(llm.ledger.summary())
        base_tok = ((led_after_flat["input_tokens"] - led_after_layered["input_tokens"])
                    + (led_after_flat["output_tokens"] - led_after_layered["output_tokens"]))

    # ---------------- 指标
    correct = [r["pred_exec"] == r["gold_exec"] for r in rows]
    acc, lo, hi = metrics.bootstrap_ci([float(c) for c in correct])

    # 澄清决策：金标 = L3 层（或 expected_behavior == 'clarify'）
    gold_clar = [r["gold_behavior"] == "clarify" for r in rows]
    pred_clar = [r["pred_clar"] for r in rows]
    clar = metrics.guard_report(pred_clar, gold_clar)

    # 分执行体召回（金标含该执行体的题里，预测是否也含）
    per_exec: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for e in r["gold_exec"]:
            per_exec[e].append(1.0 if e in r["pred_exec"] else 0.0)

    per_layer: dict[str, list[float]] = defaultdict(list)
    for r, c in zip(rows, correct):
        per_layer[r["layer"]].append(float(c))

    # 分层组成本：只算分层那一臂，不含对照组
    summary = led_after_layered
    dtok = summary["output_tokens"] - ledger_before["output_tokens"]
    dtok_in = summary["input_tokens"] - ledger_before["input_tokens"]

    # ---------------- 报告
    print(f"{'指标':26s} {'值':>10s}")
    print(f"{'-' * 40}")
    print(f"{'执行体集合完全匹配':26s} {acc:>10.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"{'澄清 召回率':26s} {clar['block_recall']:>10.3f}   "
          f"CI [{clar['block_recall_ci'][0]:.3f}, {clar['block_recall_ci'][1]:.3f}]")
    print(f"{'澄清 误报率(FPR)':26s} {clar['fpr']:>10.3f}   "
          f"CI [{clar['fpr_ci'][0]:.3f}, {clar['fpr_ci'][1]:.3f}]")
    print(f"{'澄清 准确率':26s} {clar['accuracy']:>10.3f}")
    print(f"\n分执行体召回（金标含该执行体时预测是否命中）")
    for e, v in sorted(per_exec.items(), key=lambda kv: -len(kv[1])):
        print(f"   {e:14s} n={len(v):3d}  recall={sum(v) / len(v):.3f}")
    print(f"\n分层准确率")
    for l in sorted(per_layer):
        v = per_layer[l]
        print(f"   {l:4s} n={len(v):3d}  acc={sum(v) / len(v):.3f}")

    total_tok = dtok + dtok_in
    print(f"\n成本（R2：与准确率成对报告）")
    print(f"   总 token {total_tok:,}（入 {dtok_in:,} / 出 {dtok:,}）")
    print(f"   每题 token {total_tok / len(rows):,.0f}")
    print(f"   每正确一题 token {total_tok / max(1, sum(correct)):,.0f}")
    print(f"   墙钟 {wall:.1f}s（{WORKERS} 并发），平均每题 {wall / len(rows):.2f}s")

    # ---------------- 按金标行为拆分（关键）
    # 意图集里 gold_executors=[] 把两种情况混在了一起：
    #   (a) 不该调任何执行体（越界/对抗/需人工）—— 安全要求，必须空
    #   (b) 制度里没有答案，但**应该去查**再如实说没有（半盲区/盲区）
    # 两者对路由的要求相反，混在一起算一个数会同时冤枉和放过。
    by_behavior: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_behavior[r["gold_behavior"]].append(r)

    print(f"\n按金标行为拆分（gold_executors=[] 的两种情况要求相反，必须分开看）")
    print(f"   {'行为':10s} {'n':>4s} {'路由匹配':>9s}  说明")
    NOTES = {
        "refuse": "越界/对抗 —— 应不调执行体", "decline": "越界/对抗 —— 应不调执行体",
        "escalate": "需人工 —— 应不调执行体", "correct": "纠错前提 —— 不应顺着答",
        "no_answer": "制度无答案 —— 应去查再如实说没有",
        "clarify": "歧义 —— 应澄清且不调执行体",
        "answer": "正常应答 —— 应调对执行体",
    }
    for b, rs in sorted(by_behavior.items(), key=lambda kv: -len(kv[1])):
        ok = sum(1 for r in rs if r["pred_exec"] == r["gold_exec"]) / len(rs)
        print(f"   {b:10s} {len(rs):>4d} {ok:>9.3f}  {NOTES.get(b, '')}")

    # 安全子集：越界/对抗/需人工 必须 executors=[]
    unsafe = [r for r in rows if r["gold_behavior"] in ("refuse", "decline", "escalate")]
    if unsafe:
        blocked = sum(1 for r in unsafe if not r["pred_exec"])
        print(f"\n   安全子集（越界/对抗/需人工，n={len(unsafe)}）："
              f"正确不调执行体 {blocked}/{len(unsafe)} = {blocked / len(unsafe):.3f}")
        leaked = [r for r in unsafe if r["pred_exec"]]
        if leaked:
            print(f"   ⚠️  以下 {len(leaked)} 条本不该调执行体却调了（须由 P3 风控闸门兜底）：")
            for r in leaked[:6]:
                print(f"        {r['query'][:44]:46s} → {r['pred_exec']}")

    # 半盲区子集：应去查（调 PolicyRAG）而不是直接拒
    gap = [r for r in rows if r["gold_behavior"] == "no_answer"]
    if gap:
        queried = sum(1 for r in gap if "PolicyRAG" in r["pred_exec"])
        print(f"\n   半盲区子集（n={len(gap)}）：调了 PolicyRAG 去查的 {queried}/{len(gap)}"
              f" = {queried / len(gap):.3f}（这些题 gold=[] 但正确行为是查完再说没有）")


    # ---------------- 错例
    bad = [r for r in rows if r['pred_exec'] != r['gold_exec']]
    print(f"\n错例 {len(bad)} / {len(rows)}")
    for r in bad[:10]:
        print(f"   [{r['layer']}] {r['query'][:44]}")
        print(f"        金标={r['gold_exec']}  预测={r['pred_exec']}  "
              f"歧义={r['ambiguity']:.2f} 行为={r['gold_behavior']}")

    # ---------------- 对照组比较（R5：配对比较用 McNemar）
    base_acc = base_clar = None
    if base_rows:
        base_correct = [r["pred_exec"] == r["gold_exec"] for r in base_rows]
        base_acc, blo, bhi = metrics.bootstrap_ci([float(c) for c in base_correct])
        mc = metrics.mcnemar(correct, base_correct)
        print(f"\n{'=' * 62}")
        print(f"对照组：单 Agent 扁平工具（10 个工具平铺）")
        print(f"{'=' * 62}")
        print(f"{'':22s} {'分层':>10s} {'扁平':>10s}")
        print(f"{'执行体集合匹配':22s} {acc:>10.3f} {base_acc:>10.3f}")
        print(f"{'95% CI':22s} {f'[{lo:.3f},{hi:.3f}]':>10s} {f'[{blo:.3f},{bhi:.3f}]':>10s}")
        print(f"{'每题 token':22s} {total_tok / len(rows):>10,.0f} "
              f"{base_tok / len(base_rows):>10,.0f}")
        print(f"{'每正确一题 token':22s} {total_tok / max(1, sum(correct)):>10,.0f} "
              f"{base_tok / max(1, sum(base_correct)):>10,.0f}")
        print(f"{'墙钟(s)':22s} {wall:>10.1f} {base_wall:>10.1f}")
        print(f"\nMcNemar 配对检验: b(分层错/扁平对)={mc['b']}  c(分层对/扁平错)={mc['c']}")
        print(f"   p = {mc['p']:.4f}  → {mc['note']}")
        if mc["p"] >= 0.05:
            print(f"   ⚠️  差异不显著 —— 不能声称分层路由优于扁平工具")
        base_bad = [r for r in base_rows if r["pred_exec"] != r["gold_exec"]]
        # 澄清能力对照：两组都有澄清选项后，才谈得上公平比较
        base_clar_pred = [r["pred_clar"] for r in base_rows]
        base_clar = metrics.guard_report(base_clar_pred, gold_clar)
        print(f"\n   澄清能力（两组都有澄清选项）")
        print(f"      分层: 召回 {clar['block_recall']:.3f}  FPR {clar['fpr']:.3f}")
        print(f"      扁平: 召回 {base_clar['block_recall']:.3f}  FPR {base_clar['fpr']:.3f}")
        # 扁平组的干扰项误选
        distract = sum(1 for r in base_rows
                       if any(t in ("salary_benchmark", "resume_parse", "org_chart")
                              for t in r["flat_tools"]))
        print(f"\n   扁平组错例 {len(base_bad)} / {len(base_rows)}")
        print(f"   误选干扰工具（salary_benchmark / resume_parse / org_chart）: "
              f"{distract} 条"
              + ("（干扰项没起作用，说明这个维度没测到东西）" if distract == 0 else ""))
        for r in base_bad[:6]:
            print(f"      [{r['layer']}] {r['query'][:40]:42s} "
                  f"金标={r['gold_exec']} 扁平={r['pred_exec']} 工具={r['flat_tools']}")

        # 安全子集上的对照
        unsafe_ids = {r["id"] for r in unsafe}
        for nm, rs in (("分层", rows), ("扁平", base_rows)):
            sub = [r for r in rs if r["id"] in unsafe_ids]
            if sub:
                ok = sum(1 for r in sub if not r["pred_exec"])
                print(f"   安全子集正确不调: {nm} {ok}/{len(sub)} = {ok / len(sub):.3f}")

    out = config.RESULTS / f"exp3_routing_{args.split}.json"
    payload = {
        "split": args.split, "n": len(rows), "threshold": args.threshold,
        "executor_exact_match": acc, "ci95": [lo, hi],
        "clarification": clar,
        "per_executor_recall": {k: sum(v) / len(v) for k, v in per_exec.items()},
        "per_layer_accuracy": {k: sum(v) / len(v) for k, v in per_layer.items()},
        "per_behavior": {b: {"n": len(rs),
                             "route_match": sum(1 for r in rs if r["pred_exec"] == r["gold_exec"]) / len(rs)}
                         for b, rs in by_behavior.items()},
        "safety_subset": {
            "n": len(unsafe),
            "correctly_no_dispatch": sum(1 for r in unsafe if not r["pred_exec"]),
            "leaked_ids": [r["id"] for r in unsafe if r["pred_exec"]],
        },
        "gap_subset": {
            "n": len(gap),
            "queried_policyrag": sum(1 for r in gap if "PolicyRAG" in r["pred_exec"]),
        },
        "cost": {"total_tokens": total_tok, "input_tokens": dtok_in, "output_tokens": dtok,
                 "tokens_per_query": total_tok / len(rows), "wall_s": wall},
        "rows": rows,
    }
    if base_rows:
        payload["baseline_flat"] = {
            "executor_exact_match": base_acc, "ci95": [blo, bhi],
            "mcnemar_vs_layered": mc, "clarification": base_clar,
            "distractor_misselections": distract, "total_tokens": base_tok, "wall_s": base_wall,
            "rows": base_rows,
        }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
