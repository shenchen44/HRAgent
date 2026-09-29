"""实验 8：置信度门控的风险-覆盖曲线与 AURC。

要回答的问题：
  我们手上这些"弱信号"（自评置信度 / 检索分 / 证据条数），
  到底能不能把**该升级人工的样本**排到前面去？

为什么必须报曲线而不是单点：
  同一个信号卡在不同阈值上，会得到完全不同的"拦截率 / 误报率"组合。
  单点数字（"升级率 20%，准确率 92%"）可以靠调阈值任意打扮；整条曲线 + AURC 不能。
  这也是 DESIGN 把 AURC 定为风控主指标的原因。

**两个升级触发条件是两回事，必须分开测：**
  (一) 「制度没覆盖」→ 该升级而不是硬答。标签 = 盲区题（gold 为空）。
       这个标签**不与检索分共线** —— P1 实测盲区题分数落在正常题区间内（8/8 超过
       可答题最低分），所以"检索分能否识别盲区"是个真问题。
  (二) 「会答错」→ 该升级。标签 = 答案错了。见下方**标签效度披露**。

**标签效度披露（必须先读再看数字）：**
  `rag_eval` 只标了 `gold_chunk_ids`，没有金标答案文本，所以"答对"只能用引用判定。
  人工审计 7 条判为"错"的可答题后发现，**它们全部是检索失败**（金标片段不在 top-5），
  而 L1（检索命中）与 L2（引用命中）在这 86 条上**逐条完全一致** —— 说明
  "检索不到就引不到"，引用判据没有提供额外信息。这 7 条里：
    · **3 条模型仍靠其他片段答对了**（如"加班工资几倍"引 `attendance_policy#3.4`
      而非金标 `labor_law_excerpt#2.3`，两者都写了 150/200/300%）；
    · **4 条退化成了错误拒答**（说"制度中没有相关规定"，而金标存在）。
  → 所以「检索失败率 8.1%」是确定性、可审计的指标；**有效答错率是 4/86 = 4.7%**。
    本实验主指标用 L1。

四条对照臂（缺一不可）：
  S0 无门控     常数置信度 → AURC 恒等于基础率。**这是零知识基线。**
                任何信号的 AURC 若与它持平，说明该信号**没有排序能力**。
  Oracle        完美排序 → AURC 的理论下界。
  S1/S2/S3/S4   自评置信度 / 检索最高分 / 证据条数 / UncertaintyGate 融合分

跑法:
  .venv/bin/python scripts/exp8_risk_coverage.py                 # 全部 94 题
  .venv/bin/python scripts/exp8_risk_coverage.py --split test    # 只用冻结 test
  .venv/bin/python scripts/exp8_risk_coverage.py --reset         # 丢弃断点重跑
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hragent import config, metrics  # noqa: E402
from hragent.agents.executors import PolicyRAG  # noqa: E402
from hragent.risk.guards import Candidate, UncertaintyGate  # noqa: E402
from hragent.schema import Evidence  # noqa: E402

WORKERS = 6
CKPT = config.RESULTS / "exp8_raw.jsonl"

# 答案里出现这些**否定表述** = 系统在说"制度里没有"。
# 用正则而不是子串表：实测子串表漏掉 RAG0088（它写的是
# 「没有关于员工宿舍申请的规定」，关键词中间隔了字）。
_NOINFO_RE = re.compile(
    r"没有[^。；\n]{0,25}规定|未[^。；\n]{0,12}涉及|无[^。；\n]{0,12}规定"
    r"|均未[^。；\n]{0,15}|未[^。；\n]{0,10}检索到|没有[^。；\n]{0,15}相关")


# ---------------------------------------------------------------- 标签

def label_blind(rec: dict) -> bool:
    """(一) 该升级吗？盲区题（金标为空）= 制度没覆盖 = 该升级。

    **这是"风险事件"本身**，所以喂给曲线时要取反（见 `not_blind`）。
    """
    return not rec["gold_chunk_ids"]


def not_blind(rec: dict) -> bool:
    """(一) 的"正确"谓词：这道题制度覆盖了（不需要升级）。"""
    return bool(rec["gold_chunk_ids"])


def label_retrieval_hit(rec: dict) -> bool:
    """(二) L1 检索命中：金标片段是否出现在本次检索到的片段里。

    确定性、可复现、不依赖 LLM。**这是分析二的主标签。**
    """
    gold = set(rec["gold_chunk_ids"])
    return bool(gold & set(rec["retrieved"])) if gold else True


def label_citation_hit(rec: dict) -> bool:
    """(二) L2 引用命中：答案是否引用了金标片段。

    **本数据集上它与 L1 逐条完全一致（86/86）**，因为金标片段只要被检索到，
    模型基本都会引用它；反之检索不到就无从引用。所以 L2 不提供额外信息，
    保留它是为了把"这两个标签在本数据上等价"这件事**测出来并记录**，
    而不是假定它们不同。
    """
    gold = set(rec["gold_chunk_ids"])
    return bool(gold & set(rec["cited"])) if gold else True


def label_blind_refused(rec: dict) -> bool:
    """盲区题上"答对"的定义：明确说明制度里没有。

    **不能**要求"没有引用" —— 实测 8/8 条盲区题的答案都是
    「制度中没有相关规定 + 列出检索到的片段说明它们都不相关」，
    这是正确的**展示依据**行为，不是编造。第一版判据要求 `not cited`，
    把 8/8 条正确行为全判成了错误。
    """
    return bool(_NOINFO_RE.search(rec["answer"]))


def run_one(item: dict, ex: PolicyRAG) -> dict:
    t0 = time.time()
    try:
        r = ex.run(item["query"])
        return {"id": item["id"], "split": item["split"], "difficulty": item["difficulty"],
                "query": item["query"], "gold_chunk_ids": item["gold_chunk_ids"],
                "answer": r.answer, "confidence": r.confidence, "ok": r.ok,
                "top_scores": (r.artifacts or {}).get("top_scores") or [],
                "retrieved": (r.artifacts or {}).get("retrieved") or [],
                "cited": [e.ref for e in r.evidence],
                "n_evidence": len(r.evidence), "tokens": r.tokens,
                "latency_s": round(time.time() - t0, 2), "error": r.error}
    except Exception as e:  # noqa: BLE001
        return {"id": item["id"], "split": item["split"], "difficulty": item["difficulty"],
                "query": item["query"], "gold_chunk_ids": item["gold_chunk_ids"],
                "answer": "", "confidence": 0.0, "ok": False, "top_scores": [],
                "retrieved": [], "cited": [], "n_evidence": 0, "tokens": 0,
                "latency_s": round(time.time() - t0, 2),
                "error": f"{type(e).__name__}: {e}"}


def load_done() -> dict[str, dict]:
    """断点续跑（硬性约定 4）：已完成的样本直接跳过。"""
    if not CKPT.exists():
        return {}
    out: dict[str, dict] = {}
    for line in CKPT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            out[d["id"]] = d
        except json.JSONDecodeError:
            continue
    return out


# ---------------------------------------------------------------- 信号

def signals(rec: dict) -> dict[str, float]:
    """每个信号都是「越大越可信」。全部只用**推理时可得**的信息。"""
    top = rec["top_scores"][0] if rec["top_scores"] else 0.0
    c = Candidate(query=rec["query"], answer=rec["answer"],
                  evidence=[Evidence(kind="doc_span", ref=r, value="")
                            for r in rec["cited"]],
                  retrieved_refs=list(rec["cited"]),
                  confidence=rec["confidence"],
                  artifacts={"top_scores": rec["top_scores"]})
    return {
        "S1 自评置信度": float(rec["confidence"]),
        "S2 检索最高分": float(top),
        "S3 证据条数": min(rec["n_evidence"], 5) / 5.0,
        "S4 融合（UncertaintyGate）": 1.0 - UncertaintyGate().risk_score(c),
    }


def aurc_ci(conf: list[float], label: list[bool], n_boot: int = 2000,
            seed: int = 0) -> tuple[float, float]:
    """AURC 的 bootstrap 95% 区间。

    小样本（本实验错误数只有个位数）下 AURC 的点估计极不稳定，
    不报区间就等于把噪声当结论。
    """
    rng = random.Random(seed)
    n = len(conf)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        if len({label[i] for i in idx}) < 2:      # 全对或全错时 AURC 无意义
            continue
        vals.append(metrics.risk_coverage_curve([conf[i] for i in idx],
                                                [label[i] for i in idx])["aurc"])
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1])


def random_aurc(ok: list[bool], n_rep: int = 200, seed: int = 0) -> float:
    """无信息排序的 AURC 经验值：随机置信度重复 n_rep 次的均值。

    为什么不能直接用常数置信度当"无门控"臂：
      `risk_coverage_curve` 对同分样本用的是 Python 稳定排序 → 实际按**输入顺序**取前缀。
      本数据里 8 条盲区题恰好都在文件末尾（位置 86–93），于是常数臂的
      AURC 算出 0.004，而真实基础风险是 0.085 —— 基线被数据顺序伪装成了好成绩。
      随机排序的期望才等于基础风险，这里用 200 次采样把它测出来。
    """
    rng = random.Random(seed)
    n = len(ok)
    vals = [metrics.risk_coverage_curve([rng.random() for _ in range(n)], ok)["aurc"]
            for _ in range(n_rep)]
    return sum(vals) / len(vals)


def evaluate(recs: list[dict], ok_fn, title: str, positive_name: str) -> dict:
    """ok_fn(rec) → True 表示**这一条没问题**（不需要升级 / 命中 / 答对）。

    曲线上的"风险" = 1 - ok 的比例，所以所有信号都按"越大越可信"给分。
    """
    ok = [ok_fn(r) for r in recs]
    base = 1.0 - sum(ok) / len(ok) if ok else float("nan")
    print(f"\n{'=' * 84}\n{title}\n"
          f"n={len(recs)}  {positive_name} {sum(ok)} 条  基础风险={base:.3f}\n{'=' * 84}")

    per_sample = [signals(r) for r in recs]
    sig = {name: [s[name] for s in per_sample] for name in per_sample[0]}

    rand = random_aurc(ok)
    arms: list[tuple[str, list[float]]] = [("S0 无门控（零知识基线）", None)]
    arms += list(sig.items())
    arms.append(("Oracle 完美排序（下界）", [1.0 if h else 0.0 for h in ok]))

    print(f"{'信号':28s} {'AURC':>7s} {'95% CI':>16s} {'风险@80%':>9s} {'vs 基线':>9s}")
    print("-" * 84)
    out = {}
    for name, conf in arms:
        if conf is None:                      # S0 臂 = 基础风险本身（无信息排序的理论值）
            rc = {"aurc": base, "risk_at_80": base, "risk_at_100": base,
                  "coverage": [], "risk": [], "n": len(recs)}
            lo = hi = base
        else:
            rc = metrics.risk_coverage_curve(conf, ok)
            lo, hi = aurc_ci(conf, ok)
        d = rc["aurc"] - base
        tag = ("  ← 无排序能力" if name.startswith("S0") else
               (f"  {d:+.3f}" if d < -0.005 else
                ("  ≈基线" if d < 0.005 else f"  {d:+.3f} ⚠️更差")))
        print(f"{name:28s} {rc['aurc']:>7.3f} [{lo:>6.3f},{hi:>6.3f}] "
              f"{rc['risk_at_80']:>9.3f}{tag}")
        out[name] = {**rc, "ci": [lo, hi], "delta": d}
    print(f"{'（随机排序 200 次均值，校验基线）':28s} {rand:>7.3f}"
          f"   ← 应与基础风险 {base:.3f} 接近")
    return {"n": len(recs), "n_positive": sum(ok), "base_risk": base,
            "random_aurc": rand, "arms": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"])
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.reset and CKPT.exists():
        CKPT.unlink()
    items = [json.loads(l) for l in
             (config.EVAL / "rag_eval.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip()]
    if args.split != "all":
        items = [i for i in items if i["split"] == args.split]
    if args.limit:
        items = items[:args.limit]

    done = load_done()
    todo = [i for i in items if i["id"] not in done]
    print(f"实验 8 · 风险-覆盖曲线  split={args.split}  "
          f"共 {len(items)} 题（已完成 {len(items) - len(todo)}，待跑 {len(todo)}）")

    if todo:
        ex = PolicyRAG()
        t0 = time.time()
        with CKPT.open("a", encoding="utf-8") as f, \
                ThreadPoolExecutor(WORKERS) as pool:
            for k, rec in enumerate(pool.map(lambda i: run_one(i, ex), todo), 1):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()                       # 逐条落盘，中断不丢
                if k % 10 == 0 or k == len(todo):
                    el = time.time() - t0
                    print(f"  {k}/{len(todo)}  {el:.0f}s  "
                          f"预计剩余 {el / k * (len(todo) - k):.0f}s", flush=True)
        done = load_done()

    recs = [done[i["id"]] for i in items if i["id"] in done]
    errs = [r for r in recs if r.get("error")]
    if errs:
        print(f"⚠️  {len(errs)} 条执行出错：{errs[0]['error'][:80]}")

    report: dict = {"n": len(recs)}

    # ---- 信号方向诊断：先把"每个信号朝哪边"测出来，再看曲线
    blind = [r for r in recs if not r["gold_chunk_ids"]]
    ans0 = [r for r in recs if r["gold_chunk_ids"]]
    print(f"\n{'=' * 84}\n诊断 · 各信号在「盲区题 vs 可答题」上的方向\n{'=' * 84}")
    print(f"{'信号':28s} {'可答题均值':>11s} {'盲区题均值':>11s} {'方向':>10s}")
    for name in ("S1 自评置信度", "S2 检索最高分", "S3 证据条数",
                 "S4 融合（UncertaintyGate）"):
        a = sum(signals(r)[name] for r in ans0) / len(ans0)
        b = sum(signals(r)[name] for r in blind) / len(blind)
        arrow = "✅ 同向" if a > b else "❌ 反向（越该升级分越高）"
        print(f"{name:28s} {a:>11.3f} {b:>11.3f} {arrow:>10s}")
    n_zero_ev = sum(1 for r in recs if r["n_evidence"] == 0)
    print(f"\n  ⚠️  n_evidence == 0 的样本：{n_zero_ev}/{len(recs)}"
          f" —— UncertaintyGate 里权重最大的 +0.35「无证据」项"
          f"{'从不触发' if n_zero_ev == 0 else '极少触发'}")
    report["signal_direction"] = {
        "n_zero_evidence": n_zero_ev,
        "blind_conf_mean": sum(r["confidence"] for r in blind) / len(blind),
        "answerable_conf_mean": sum(r["confidence"] for r in ans0) / len(ans0),
        "blind_n_evidence_mean": sum(r["n_evidence"] for r in blind) / len(blind),
        "answerable_n_evidence_mean": sum(r["n_evidence"] for r in ans0) / len(ans0),
    }

    # ---- 分析一：能否识别「制度没覆盖」
    # 注意谓词方向：`not_blind` 才是"没问题"，风险 = 盲区率。
    report["escalate_blind"] = evaluate(
        recs, not_blind,
        "分析一 · 能否识别「制度没覆盖」的题（该升级而不是硬答）",
        "制度覆盖了（不需升级）")

    # ---- 分析二：能否识别「会答错」
    ans = [r for r in recs if r["gold_chunk_ids"]]
    report["answerable_L1"] = evaluate(
        ans, label_retrieval_hit,
        "分析二 · 能否识别「会答错」的题 —— L1 检索命中（确定性主标签）",
        "检索命中")
    report["answerable_L2"] = evaluate(
        ans, label_citation_hit,
        "分析二 · 能否识别「会答错」的题 —— L2 引用命中（预期与 L1 等价）",
        "引用命中")
    # 把"L1 与 L2 是否真的等价"测出来，而不是假定
    disagree = [r["id"] for r in ans
                if label_retrieval_hit(r) != label_citation_hit(r)]
    print(f"\n【标签等价性核验】L1（检索命中）与 L2（引用命中）判定不一致的样本："
          f"{len(disagree)}/{len(ans)}")
    report["L1_L2_disagreement"] = disagree

    # ---- 分析三：盲区题上的"正确拒答"率 + 阈值校准
    blind = [r for r in recs if not r["gold_chunk_ids"]]
    if blind:
        ref = [label_blind_refused(r) for r in blind]
        print(f"\n{'=' * 84}\n分析三 · 盲区题上「明确说明制度里没有」的比例\n{'=' * 84}")
        print(f"  {sum(ref)}/{len(blind)} = {sum(ref) / len(blind):.1%}"
              f"   （判据：正则匹配「没有…规定 / 未…涉及 / 均未…」等否定表述）")
        for r in blind:
            mark = "✅" if label_blind_refused(r) else "❌"
            print(f"    {mark} [{r['id']}] {r['query'][:34]}")
        report["blind_refusal_rate"] = sum(ref) / len(blind)

    print(f"\n{'=' * 84}\n分析四 · 阈值校准：融合分能不能直接当阈值用\n{'=' * 84}")
    hit = [label_blind(r) for r in recs]
    sig4 = [s["S4 融合（UncertaintyGate）"] for s in (signals(r) for r in recs)]
    print(f"{'阈值':>6s} {'自动处理率':>10s} {'升级率':>8s} {'升级精准度':>10s} "
          f"{'升级召回':>8s}")
    rows = []
    for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        h = metrics.hitl_report(sig4, [not x for x in hit], th)
        tp = sum(1 for i, c in enumerate(sig4) if c < th and hit[i])
        rec = tp / sum(hit) if sum(hit) else float("nan")
        print(f"{th:>6.1f} {h['coverage']:>10.3f} {h['escalation_rate']:>8.3f} "
              f"{h['escalation_precision']:>10.3f} {rec:>8.3f}")
        rows.append({"threshold": th, **{k: h[k] for k in
                                         ("coverage", "escalation_rate",
                                          "escalation_precision")}, "recall": rec})
    report["threshold_sweep"] = rows

    out = config.RESULTS / f"exp8_risk_coverage_{args.split}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {out}")
    print(f"→ 原始逐题结果 {CKPT}")


if __name__ == "__main__":
    main()
