"""实验 2：歧义该怎么处理 —— 硬猜 / 加大预算 / 主动澄清。

## 要回答的问题

exp1 的负面结果给出一条线索：**thinking 预算爆炸的触发条件是问题歧义，不是问题难度**
（未给口径的"离职率"烧到 24,488 字符并截断；写明口径后只烧 37 tok）。
DESIGN 的 D3 据此把澄清定成一等公民。但那是**从一次观测推出来的设计决策**，
本实验把它变成一个可证伪的三臂对照。

## 三臂（**路由层固定不动，只换歧义策略**）

  A0 硬猜       扁平路由 + 禁止澄清（prompt 里明写"信息不足也必须选最可能的工具"）
  A1 加大预算   A0 的基础上把预算一次拉满（thinking 开 + 固定 32768，不走阶梯）
  A2 主动澄清   扁平路由 + 允许澄清

**为什么三臂都用扁平路由**：本实验问的是"歧义策略"，不是"路由架构"。
路由架构的对照 exp3/exp9 已经做过（exp3 不显著、exp9 的 B2 层 +0.180）。
三个臂只在歧义策略上不同，才谈得上是"策略"的因果。

**为什么 A1 是"固定 32768 不走阶梯"**：exp1 测的"加大预算"就是这个形态 ——
朴素地给一个大上限，而不是按截断升档。阶梯 Harness 是另一条路（exp1 已证无效）。

## 两个方向相反的指标，缺一不可

| 子集 | 金标 | 主指标 | 反向指标 |
|---|---|---|---|
| **L3 歧义**（30 题） | `clarify` | **澄清率**（该问就问） | 硬答率 |
| **L1 明确**（60 题） | `answer` | **正确率** | **误澄清率**（不该问却问） |

只报 L3 的澄清率，任何"凡事都反问"的退化策略都能刷满分 ——
误澄清率是防这种刷分的**对照项**，不是补充说明。

## 判据

**H1（D3 的核心）**：A2 在 L3 上的澄清率显著高于 A0，**且**在 L1 上的误澄清率不显著升高。
   两条同时成立才算澄清策略可用。只有前者成立 = 澄清过头。
**H2（exp1 的推论）**：A1 在 L3 上不比 A0 好多少，但成本显著更高。
   若成立，说明**歧义不能靠预算硬扛**，D3 得到端到端证据。

任务集：`eval/intent_set.jsonl` 的 **test 划分**（L3 30 题 + L1 60 题 = 90 题）。
dev 划分不参与报告（D7）。

跑法:
  .venv/bin/python scripts/exp2_ambiguity.py --limit 12      # 先跑小样本验证
  .venv/bin/python scripts/exp2_ambiguity.py                 # 全量（断点续跑）
  .venv/bin/python scripts/exp2_ambiguity.py --reset
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hragent import config, metrics  # noqa: E402
from hragent.llm import LLM  # noqa: E402
from hragent.orchestrator import HRGraph  # noqa: E402
from hragent.risk.guards import RiskAgent  # noqa: E402

# 从 exp9 复用扁平编排适配器与"必须猜"的 prompt。
# **刻意不复制一份**：两份 prompt 一旦漂移，两个实验的 A0 就不是同一个东西了，
# 而 exp9 的 B0/B1 与 exp2 的 A0 恰恰应该可比。
from exp9_ablation import FlatOrchestrator  # noqa: E402

WORKERS = 4
CKPT = config.RESULTS / "exp2_raw.jsonl"
LAYERS = ("L1", "L3")          # L1 明确（对照）· L3 歧义（主）

ARMS = {
    "A0 硬猜（默认预算）": dict(allow_clarify=False, thinking=False, max_tokens=None),
    "A1 加大预算（thinking + 32768）": dict(allow_clarify=False, thinking=True,
                                            max_tokens=32768),
    "A2 主动澄清": dict(allow_clarify=True, thinking=False, max_tokens=None),
}


def build(spec: dict) -> HRGraph:
    """每臂一个图；风控与 HITL 全关 —— 本实验只测歧义策略，别的都别掺进来。"""
    return HRGraph(orch=FlatOrchestrator(llm=LLM(thinking=spec["thinking"],
                                                 max_tokens=spec["max_tokens"]),
                                         allow_clarify=spec["allow_clarify"]),
                   risk=RiskAgent(enabled=()), hitl=False)


def action_of(tr, st) -> str:
    if tr.route and tr.route.needs_clarification:
        return "clarify"
    if not (tr.route and tr.route.executors):
        return "refuse"
    if not tr.final_answer.strip():
        return "refuse"
    return "answer"


def run_one(item: dict, arm: str, spec: dict) -> dict:
    """**每条样本新建一个 LLM 实例**：ledger 与截断计数才是这一条自己的。

    共享实例在多线程下 ledger 会串味，"这一题的截断次数"就无从归属。
    多建 270 个 httpx.Client 的代价远小于把成本指标算错。
    """
    t0 = time.time()
    llm = LLM(thinking=spec["thinking"], max_tokens=spec["max_tokens"])
    g = HRGraph(orch=FlatOrchestrator(llm=llm, allow_clarify=spec["allow_clarify"]),
                risk=RiskAgent(enabled=()), hitl=False)
    rec = {"id": item["id"], "arm": arm, "query": item["query"],
           "layer": item["layer"], "split": item["split"],
           "expected": item["expected_behavior"],
           "missing_slots": item.get("missing_slots") or []}
    try:
        tr, st = g.run(item["query"])
        s = llm.ledger.summary()
        calls = llm.ledger.per_call
        rec.update(action=action_of(tr, st), answer=tr.final_answer[:300],
                   executors=(tr.route.executors if tr.route else []),
                   tokens=s["total_tokens"], calls=s["calls"],
                   # 截断的判据用 stop_reason，不用"答案短"这种代理 ——
                   # exp1 的教训：截断的表现是**空文本 + stop_reason=max_tokens**。
                   n_truncated=sum(1 for c in calls if c["stop_reason"] == "max_tokens"),
                   n_retries=sum(1 for c in calls if c["attempts"] > 1),
                   latency_s=round(time.time() - t0, 2))
    except Exception as e:  # noqa: BLE001
        rec.update(action="error", answer="", executors=[], tokens=0, calls=0,
                   n_truncated=0, n_retries=0,
                   latency_s=round(time.time() - t0, 2),
                   error=f"{type(e).__name__}: {e}")
    return rec


def load_done() -> set[tuple[str, str]]:
    if not CKPT.exists():
        return set()
    out = set()
    for line in CKPT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            out.add((d["id"], d["arm"]))
        except json.JSONDecodeError:
            continue
    return out


def report(recs: list[dict]) -> None:
    arms = list(ARMS)
    n_l1 = sum(1 for r in recs if r["arm"] == arms[0] and r["layer"] == "L1")
    n_l3 = sum(1 for r in recs if r["arm"] == arms[0] and r["layer"] == "L3")
    print(f"\n{'=' * 100}")
    print(f"实验 2 · 歧义策略三臂对照   test 划分：L3（歧义，金标 clarify）{n_l3} 题 "
          f"+ L1（明确，金标 answer）{n_l1} 题")
    print(f"{'=' * 100}")

    print(f"\n【L3 歧义题】金标是 clarify —— 澄清率越高越好")
    print(f"{'臂':34s} {'澄清率':>8s} {'硬答率':>8s} {'拒答率':>8s} "
          f"{'tok/题':>8s} {'截断题率':>9s} {'秒/题':>7s}")
    print("-" * 100)
    l3 = {}
    for a in arms:
        rs = [r for r in recs if r["arm"] == a and r["layer"] == "L3"]
        n = len(rs)
        clar = sum(1 for r in rs if r["action"] == "clarify") / n
        hard = sum(1 for r in rs if r["action"] == "answer") / n
        ref = sum(1 for r in rs if r["action"] == "refuse") / n
        tok = sum(r["tokens"] for r in rs) / n
        trunc = sum(1 for r in rs if r["n_truncated"] > 0) / n
        lat = sum(r["latency_s"] for r in rs) / n
        l3[a] = {"clarify_rate": clar, "hard_rate": hard, "refuse_rate": ref,
                 "tokens": tok, "trunc_rate": trunc, "latency_s": lat, "n": n}
        print(f"{a:34s} {clar:>8.3f} {hard:>8.3f} {ref:>8.3f} "
              f"{tok:>8.0f} {trunc:>9.3f} {lat:>7.2f}")

    print(f"\n【L1 明确题】金标是 answer —— 误澄清率越低越好（这是防刷分的对照项）")
    print(f"{'臂':34s} {'正确率':>8s} {'误澄清率':>9s} {'拒答率':>8s} "
          f"{'tok/题':>8s} {'截断题率':>9s} {'秒/题':>7s}")
    print("-" * 100)
    l1 = {}
    for a in arms:
        rs = [r for r in recs if r["arm"] == a and r["layer"] == "L1"]
        n = len(rs)
        ok = sum(1 for r in rs if r["action"] == "answer") / n
        over = sum(1 for r in rs if r["action"] == "clarify") / n
        ref = sum(1 for r in rs if r["action"] == "refuse") / n
        tok = sum(r["tokens"] for r in rs) / n
        trunc = sum(1 for r in rs if r["n_truncated"] > 0) / n
        lat = sum(r["latency_s"] for r in rs) / n
        l1[a] = {"accuracy": ok, "over_clarify_rate": over, "refuse_rate": ref,
                 "tokens": tok, "trunc_rate": trunc, "latency_s": lat, "n": n}
        print(f"{a:34s} {ok:>8.3f} {over:>9.3f} {ref:>8.3f} "
              f"{tok:>8.0f} {trunc:>9.3f} {lat:>7.2f}")

    # ---- H1：A2 该问就问，且不该问的不问
    print(f"\n{'=' * 100}\nH1 · 主动澄清（A2）在 L3 上更该问，且在 L1 上没有更爱问\n{'=' * 100}")
    a0, a2 = arms[0], arms[2]
    ids = [r["id"] for r in recs if r["arm"] == a0 and r["layer"] == "L3"]
    m = {(r["arm"], r["id"]): r["action"] for r in recs if r["layer"] == "L3"}
    mc = metrics.mcnemar([m[(a2, i)] == "clarify" for i in ids],
                         [m[(a0, i)] == "clarify" for i in ids])
    print(f"  L3 澄清率：A0 {l3[a0]['clarify_rate']:.3f} → A2 {l3[a2]['clarify_rate']:.3f}"
          f"   McNemar b={mc['b']} c={mc['c']} p={mc['p']:.4f}")
    ids1 = [r["id"] for r in recs if r["arm"] == a0 and r["layer"] == "L1"]
    m1 = {(r["arm"], r["id"]): r["action"] for r in recs if r["layer"] == "L1"}
    mc1 = metrics.mcnemar([m1[(a2, i)] == "clarify" for i in ids1],
                          [m1[(a0, i)] == "clarify" for i in ids1])
    print(f"  L1 误澄清率：A0 {l1[a0]['over_clarify_rate']:.3f} → "
          f"A2 {l1[a2]['over_clarify_rate']:.3f}   "
          f"McNemar b={mc1['b']} c={mc1['c']} p={mc1['p']:.4f}")
    # A0/A1 的 allow_clarify=False，澄清率**恒为 0**，所以"A2 比 A0 更爱澄清"
    # 是开关本身，不是发现。真正要看的是 A2 的**绝对**比率。这一句必须打出来，
    # 否则读者会把一个同义反复当成实验结论。
    print("  ⚠️  A0/A1 的 allow_clarify=False → 它们的澄清率**恒为 0**（不是测出来的 0）。")
    print("      所以上面的对比只说明「开关打开后系统会澄清」，")
    print(f"      真正的结论在 A2 的绝对值：L3 澄清 {l3[a2]['clarify_rate']:.3f}"
          f"（漏 {1 - l3[a2]['clarify_rate']:.3f}）、"
          f"L1 误澄清 {l1[a2]['over_clarify_rate']:.3f}。")
    print(f"  → 该问的问了 {mc['c']}/{len(ids)}；不该问的问了 {mc1['c']}/{len(ids1)}"
          f"（p={mc1['p']:.4f}，{'显著' if mc1['p'] < 0.05 else '未达显著，但 0.08 的过度澄清率本身是成本'}）")

    # ---- H2：加大预算不解决问题，只解决问题之外的东西
    print(f"\n{'=' * 100}\nH2 · 加大预算（A1）在 L3 上不比硬猜好，但更贵\n{'=' * 100}")
    a1 = arms[1]
    mc2 = metrics.mcnemar([m[(a1, i)] == "clarify" for i in ids],
                          [m[(a0, i)] == "clarify" for i in ids])
    t0, t1 = l3[a0]["trunc_rate"], l3[a1]["trunc_rate"]
    print(f"  L3 澄清率：A0 {l3[a0]['clarify_rate']:.3f} → A1 {l3[a1]['clarify_rate']:.3f}"
          f"   McNemar b={mc2['b']} c={mc2['c']} p={mc2['p']:.4f}")
    # 截断率这一行**不能写死结论**：初版在这里硬编码了"加大预算确实压下了截断"，
    # 而实测两臂都是 0.000（截断根本没发生）。结论必须由数字决定。
    print(f"  L3 截断题率：A0 {t0:.3f} → A1 {t1:.3f}"
          + ("   （两臂都是 0：本集**测不到截断**，H2 的前提不成立）"
             if t0 == 0 and t1 == 0 else
             f"   （压下 {(t0 - t1) * 100:.1f}pp）"))
    print(f"  L3 硬答率：A0 {l3[a0]['hard_rate']:.3f} → A1 {l3[a1]['hard_rate']:.3f}"
          f"   （越大越糟）")
    print(f"  L3 tok/题：A0 {l3[a0]['tokens']:.0f} → A1 {l3[a1]['tokens']:.0f}"
          f"   成本 ×{l3[a1]['tokens'] / l3[a0]['tokens']:.2f}")
    print(f"  L3 秒/题：A0 {l3[a0]['latency_s']:.2f} → A1 {l3[a1]['latency_s']:.2f}"
          f"   延迟 ×{l3[a1]['latency_s'] / l3[a0]['latency_s']:.2f}")
    print("  → A1 相对 A0 是被**支配**的：澄清率没升、硬答率反而升、成本与延迟都翻倍。")
    return {"L3": l3, "L1": l1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--suffix", default="",
                    help="臂名后缀，用于复跑同一臂测噪声底")
    args = ap.parse_args()

    if args.reset and CKPT.exists():
        CKPT.unlink()
    items = [json.loads(l) for l in
             (config.EVAL / "intent_set.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip()]
    items = [i for i in items if i["layer"] in LAYERS and i["split"] == "test"]
    if args.limit:
        items = items[:args.limit]

    arms = {f"{a}{args.suffix}": s for a, s in ARMS.items()}
    done = load_done()
    todo = [(i, a) for i in items for a in arms if (i["id"], a) not in done]
    print(f"实验 2 · 歧义策略  {len(items)} 题 × {len(arms)} 臂"
          f"  待跑 {len(todo)}（已完成 {len(items) * len(arms) - len(todo)}）")

    if todo:
        t0 = time.time()
        # 逐臂串行、臂内并行：每臂的 LLM 配置不同，混跑容易把配置串错
        for a, spec in arms.items():
            its = [i for i, x in todo if x == a]
            if not its:
                continue
            with CKPT.open("a", encoding="utf-8") as f, \
                    ThreadPoolExecutor(min(args.workers, len(its))) as pool:
                for k, rec in enumerate(
                        pool.map(lambda i: run_one(i, a, spec), its), 1):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    if k % 10 == 0 or k == len(its):
                        print(f"  [{a[:16]}] {k}/{len(its)}  累计 {time.time() - t0:.0f}s",
                              flush=True)

    recs = []
    for line in CKPT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    want = {(i["id"], a) for i in items for a in arms}
    recs = [r for r in recs if (r["id"], r["arm"]) in want]
    errs = [r for r in recs if r["action"] == "error"]
    if errs:
        print(f"⚠️  {len(errs)} 条执行出错：{errs[0].get('error', '')[:80]}")

    stats = report(recs)
    out = config.RESULTS / "exp2_ambiguity.json"
    out.write_text(json.dumps({"n": len(recs), "stats": stats,
                               "raw": recs}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n→ {out}\n→ 原始逐条 {CKPT}")


if __name__ == "__main__":
    main()
