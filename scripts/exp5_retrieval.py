"""实验 5：检索策略对比 —— BM25 / Dense / Hybrid(RRF) / Rerank。

要回答的问题：制度问答的召回环节，值不值得上向量与重排？
**注意这只测召回，不测最终答案** —— 组件级指标好看而端到端不动，
在本项目已经发生过（exp9：风控层硬答率 -0.155，行为正确率 -0.013）。

四臂（每步只加一样东西）：

  BM25   词频检索（`retrieval.PolicyIndex`，k1=1.5 b=0.75）
  Dense  双塔向量（`BAAI/bge-small-zh-v1.5`，查询侧加官方指令前缀）
  Hybrid BM25 + Dense 用 **RRF** 融合（`score = Σ 1/(60+rank)`）
  Rerank Hybrid 召回 20 条 → 交叉编码器（`BAAI/bge-reranker-base`）重排

**为什么 RRF 而不是加权求和**：BM25 分无界、余弦在 [0,1]，量纲不可比。
加权就得先归一化再调权重，而权重只能在 dev 上拟合 —— 多一个超参、
多一处过拟合。RRF 只用排名，天然免调参。本实验因此**没有任何可调超参**，
dev/test 划分只用于**验证稳定性**，不用于选型。

主指标：
  · Hit@1/3/5   金标片段是否出现在前 k（检索的底线），附 95% bootstrap 区间
  · MRR@5       第一个金标排在第几位
  · NDCG@5      考虑名次与多金标的排序质量
  · Recall@5    多金标题（17 题）的覆盖率
  · 盲区取舍     8 条全盲区题上，拿 top1 分数当拒答闸门的召回/误杀（P1 的复核）
  · 成本         预热后的单查询延迟 + 索引构建时间（硬性约定 5）

**为什么 Hit@5 不是主判据**：四臂的 Hit@5 都挤在 0.92~0.99（BM25 只差 6 题就满分），
区分度被天花板压没了。臂间真正的差距在 Hit@1（0.698→0.884）与 MRR 上。
所以配对检验对 Hit@1 与 Hit@5 **都**做，不能只报好看的那个。

基线：随机排序的**理论** Hit@k = 1 − C(N−g, k)/C(N, k)。
不用"随机跑一次"当基线 —— exp8 踩过：实现细节会把基线伪装成好成绩。

跑法:
  .venv/bin/python scripts/exp5_retrieval.py
  .venv/bin/python scripts/exp5_retrieval.py --split test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hragent import config, metrics  # noqa: E402
from hragent.tools import dense  # noqa: E402
from hragent.tools.retrieval import index as bm25_index  # noqa: E402

ARMS = ("BM25", "Dense", "Hybrid", "Rerank")
KS = (1, 3, 5)
K_MAX = 5
EVAL = config.ROOT / "eval" / "rag_eval.jsonl"
OUT = config.RESULTS / "exp5_retrieval.json"


def load_eval(split: str) -> list[dict]:
    rows = [json.loads(line) for line in EVAL.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if split != "all":
        rows = [r for r in rows if r.get("split") == split]
    return rows


def n_chunks() -> int:
    return len(bm25_index().chunks)


def random_hit_at_k(n: int, g: int, k: int) -> float:
    """随机排序下"前 k 条里至少有一个金标"的**理论**概率。

    = 1 − C(n−g, k) / C(n, k)，即"k 条全落在非金标里"的补。
    这是零知识基线：任何检索器的 Hit@k 必须显著高于它才有意义。
    """
    if g <= 0 or n <= 0:
        return 0.0
    if n - g < k:
        return 1.0
    miss = math.comb(n - g, k) / math.comb(n, k)
    return 1.0 - miss


def aggregate(per_item: list[dict]) -> dict:
    """把逐题明细汇总成指标。只统计有金标的题（盲区题没有可命中的目标）。"""
    scored = [p for p in per_item if p["n_gold"] > 0]
    if not scored:
        return {}
    agg = {f"hit@{k}": sum(p[f"hit@{k}"] for p in scored) / len(scored) for k in KS}
    agg["mrr@5"] = sum(p["mrr@5"] for p in scored) / len(scored)
    agg["ndcg@5"] = sum(p["ndcg@5"] for p in scored) / len(scored)
    multi = [p for p in scored if p["n_gold"] > 1]
    agg["recall@5"] = (sum(p["recall@5"] for p in multi) / len(multi)) if multi else float("nan")
    agg["n_scored"] = len(scored)
    # R4：比例型指标给 95% bootstrap 区间。94 题的样本量下，0.02 级别的差距
    # 本来就在噪声里，只报点估计会诱导过度解读。
    for k in ("hit@1", "hit@5"):
        agg[f"{k}_ci"] = metrics.bootstrap_ci([float(p[k]) for p in scored])[1:]
    return agg


def evaluate(idx, rows: list[dict]) -> dict:
    """逐题跑一遍，返回指标与逐题明细。"""
    per_item = []
    lat = []
    for r in rows:
        gold = set(r.get("gold_chunk_ids") or [])
        t0 = time.perf_counter()
        hits = idx.search(r["query"], K_MAX)
        lat.append(time.perf_counter() - t0)
        got = [h["chunk_id"] for h in hits]
        rel = [1 if c in gold else 0 for c in got]
        top1 = hits[0]["score"] if hits else 0.0
        per_item.append({
            "id": r["id"], "split": r.get("split"), "difficulty": r.get("difficulty"),
            "n_gold": len(gold), "got": got, "rel": rel, "top1_score": top1,
            "hit@1": int(bool(gold) and rel[0] == 1),
            "hit@3": int(bool(gold) and any(rel[:3])),
            "hit@5": int(bool(gold) and any(rel[:5])),
            "mrr@5": metrics.mrr(rel[:5]),
            "ndcg@5": metrics.ndcg_at_k(rel, K_MAX),
            "recall@5": metrics.recall_at_k(rel, K_MAX),
        })
    agg = aggregate(per_item)
    agg["latency_ms"] = 1000 * sum(lat) / len(lat)
    agg["per_item"] = per_item
    return agg


def blind_operating_point(per_item: list[dict]) -> dict:
    """检索侧的"风险-覆盖"取舍：用 top1 分数当闸门，能不能把盲区题挡掉？

    做法：扫阈值 t，把 `top1_score < t` 的题判为"检索不到 → 该拒答"。
      · 挡掉的盲区题 / 全部盲区题   = 这个闸门的**召回**（越高越好）
      · 连带挡掉的可答题 / 全部可答题 = 它的**误报**（越低越好）
    这两个数必须成对看（R3）—— 阈值调到 +∞ 时召回必然是 1.0，没有信息量。

    现实里最有用的一个点：**把 8 条盲区全挡掉所需的最低阈值，代价是多少可答题**。
    这就是 UncertaintyGate 里 `_retrieval_conf` 那一项能达到的上限。
    """
    blind = [p["top1_score"] for p in per_item if p["n_gold"] == 0]
    ans = [p["top1_score"] for p in per_item if p["n_gold"] > 0]
    if not blind or not ans:
        return {}
    t_all = max(blind)                      # 要挡掉全部盲区，阈值必须高于盲区最高分
    killed = sum(1 for s in ans if s < t_all)
    return {
        "n_blind": len(blind), "n_ans": len(ans),
        "blind_mean": sum(blind) / len(blind), "blind_max": max(blind),
        "ans_min": min(ans), "ans_mean": sum(ans) / len(ans),
        # AUC 把整条曲线压成一个数：正类 = 可答题。0.5 = 这个分数完全分不开两者
        "auc_ans_vs_blind": metrics.auc(ans + blind, [True] * len(ans) + [False] * len(blind)),
        # 该阈值下可答题的"误杀率"—— 这就是拿检索分当拒答闸门的代价
        "thr_for_all_blind": t_all,
        "ans_killed_at_thr": killed,
        "ans_killed_rate": killed / len(ans),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"])
    args = ap.parse_args()

    rows = load_eval(args.split)
    n = n_chunks()
    print("=" * 92)
    print(f"实验 5：检索策略对比   n={len(rows)} 题 · 语料 {n} 个片段 · split={args.split}")
    print("=" * 92)

    # 索引构建时间（含模型加载/文档向量），与单查询延迟分开报
    build_s, idxs = {}, {}
    for name in ARMS:
        t0 = time.perf_counter()
        idxs[name] = dense.build(name)
        build_s[name] = time.perf_counter() - t0

    # 预热：把模型加载与首查开销排除在单查询延迟之外
    for name in ARMS:
        idxs[name].search("预热查询", K_MAX)

    res = {name: evaluate(idxs[name], rows) for name in ARMS}

    # ---- 主表
    print(f"\n{'臂':8s} {'Hit@1':>8s} {'95%CI':>15s} {'Hit@3':>8s} {'Hit@5':>8s} "
          f"{'MRR@5':>8s} {'NDCG@5':>8s} {'Recall@5':>9s} {'延迟ms':>8s} {'建索引s':>8s}")
    print("-" * 108)
    for name in ARMS:
        a = res[name]
        lo, hi = a["hit@1_ci"]
        print(f"{name:8s} {a['hit@1']:>8.3f} [{lo:.3f},{hi:.3f}] {a['hit@3']:>8.3f} "
              f"{a['hit@5']:>8.3f} {a['mrr@5']:>8.3f} {a['ndcg@5']:>8.3f} "
              f"{a['recall@5']:>9.3f} {a['latency_ms']:>8.1f} {build_s[name]:>8.2f}")
    print(f"  注：延迟是**预热后**的单查询耗时；Rerank 的 {res['Rerank']['latency_ms']:.0f}ms "
          f"里绝大部分是 20 条候选的交叉编码器前向，不是检索本身。")

    # ---- 零知识基线
    scored_rows = [r for r in rows if r.get("gold_chunk_ids")]
    rnd = {k: sum(random_hit_at_k(n, len(r["gold_chunk_ids"]), k)
                  for r in scored_rows) / len(scored_rows) for k in KS}
    print(f"\n{'随机排序(理论)':8s} " + " ".join(f"{rnd[k]:>8.3f}" for k in KS)
          + "   ← 零知识基线（Hit@k）")
    for name in ARMS:
        lift = res[name]["hit@5"] - rnd[5]
        print(f"  {name:8s} Hit@5 相对基线 {lift:+.3f}"
              f"  （{'显著高于基线' if lift > 0.1 else '⚠️ 与基线差距小'}）")

    # ---- 按难度拆分
    print(f"\n按难度拆分（Hit@5）")
    diffs = sorted({p["difficulty"] for a in res.values() for p in a["per_item"]})
    print(f"{'臂':8s}" + "".join(f"{d:>12s}" for d in diffs))
    for name in ARMS:
        line = f"{name:8s}"
        for d in diffs:
            ps = [p for p in res[name]["per_item"] if p["difficulty"] == d and p["n_gold"] > 0]
            line += f"{(sum(p['hit@5'] for p in ps) / len(ps) if ps else float('nan')):>12.3f}"
        print(line)

    # ---- 盲区题行为（P1 复核）
    print(f"\n盲区题：拿 top1 分数当拒答闸门，能不能把 8 条盲区挡掉？")
    print(f"{'臂':8s} {'AUC':>7s} {'盲区均分':>9s} {'盲区最高':>9s} {'可答最低':>9s} "
          f"{'全挡所需阈值':>12s} {'连带误杀可答':>13s}")
    ops = {}
    for name in ARMS:
        op = blind_operating_point(res[name]["per_item"])
        ops[name] = op
        print(f"{name:8s} {op['auc_ans_vs_blind']:>7.3f} {op['blind_mean']:>9.3f} "
              f"{op['blind_max']:>9.3f} {op['ans_min']:>9.3f} "
              f"{op['thr_for_all_blind']:>12.3f} "
              f"{op['ans_killed_at_thr']:>6d}/{op['n_ans']:<6d}"
              f" = {op['ans_killed_rate']:.1%}")
    print("  注：AUC 的正类是**可答题**，所以 0.5 = top1 分数完全分不开盲区与可答，"
          "1.0 = 完美分开。\n"
          "      「连带误杀」是拿检索分当拒答闸门的真实代价 —— 全挡盲区必然要抬高阈值，"
          "抬到多高就误杀多少可答题。")

    # ---- 换检索器之后，风控层那一项还活着吗？
    # `UncertaintyGate` 的检索项是 `0.25 * (1 - _retrieval_conf(top1))`，
    # 而 `_retrieval_conf(top) = top / (top + _BM25_MID)`，`_BM25_MID = 27.0`
    # 是**按 BM25 的量纲**定的（94 题 BM25 最高分中位数）。
    # 换成余弦（~0.3~0.6）或 RRF 分（~0.03）之后，这一项会被压成常数 ——
    # 常数项对风险分没有区分度，**这正是本项目反复出现的"信号静默失效"**。
    # 所以换检索器必须同时重定 MID，否则风控会静默退化。下面就是把这件事量出来。
    print(f"\n换检索器后风控层的检索项是否还活着（沿用现行 `_BM25_MID=27.0`）")
    print(f"{'臂':8s} {'检索项 最小':>12s} {'最大':>8s} {'极差':>8s} "
          f"{'盲区均值':>9s} {'可答均值':>9s} {'AUC':>7s}")
    for name in ARMS:
        from hragent.risk.guards import _retrieval_conf as rc
        bl = [1.0 - rc(p["top1_score"]) for p in res[name]["per_item"] if p["n_gold"] == 0]
        an = [1.0 - rc(p["top1_score"]) for p in res[name]["per_item"] if p["n_gold"] > 0]
        allv = bl + an
        print(f"{name:8s} {min(allv):>12.4f} {max(allv):>8.4f} {max(allv) - min(allv):>8.4f} "
              f"{sum(bl)/len(bl):>9.4f} {sum(an)/len(an):>9.4f} "
              f"{metrics.auc(bl + an, [True] * len(bl) + [False] * len(an)):>7.3f}")
    print("  读法：正类 = 盲区题，所以 AUC 越大越好（1.0 = 这一项能把盲区完全挑出来）。\n"
          "        极差接近 0 = 这一项已退化成常数，加不加它对风险分没影响（信号静默失效）。\n"
          "        AUC 与上一张表数值相同是**必然的** —— `_retrieval_conf` 单调，\n"
          "        MID 只改变量纲、不改变次序，所以风险-覆盖曲线与 MID 无关。\n"
          "        真正被 MID 影响的是这一项的**动态范围**：范围塌了，权重 0.25 就白给。")

    # 修法验证：MID 不该是写死的常数，而该是**该检索器 top1 分的中位数**
    # （现行 27.0 就是这么来的：94 题 BM25 最高分中位数 27.46）。
    # 把它改成按检索器取，动态范围就恢复了 —— 说明这不是"换个模型就废"的硬伤，
    # 而是"常数没跟着换"的接线问题。
    print(f"\n按检索器重定 MID（= 该臂 top1 分的中位数）后，检索项的动态范围")
    mids = {}
    for name in ARMS:
        tops = sorted(p["top1_score"] for p in res[name]["per_item"] if p["n_gold"] > 0)
        mid = tops[len(tops) // 2] or 1.0
        mids[name] = mid
        vals = [1.0 - (p["top1_score"] / (p["top1_score"] + mid)) for p in res[name]["per_item"]]
        bl = [v for v, p in zip(vals, res[name]["per_item"]) if p["n_gold"] == 0]
        an = [v for v, p in zip(vals, res[name]["per_item"]) if p["n_gold"] > 0]
        print(f"  {name:8s} MID={mid:>8.3f}  极差={max(vals) - min(vals):.4f}  "
              f"盲区均值={sum(bl)/len(bl):.4f}  可答均值={sum(an)/len(an):.4f}")
    print("  → BM25 的 27.0 与这里的做法是同一个 recipe：**MID 是检索器的量纲属性**，\n"
          "    换检索器时必须跟着换，否则这一项退化成常数 —— 不报错、指标不变，\n"
          "    只在新检索器上悄悄失去风控。\n"
          "  → 但 Hybrid 换完 MID 极差也只有 0.025（盲区 0.505 vs 可答 0.502，几乎重合）。\n"
          "    这不是接线问题，是**结构问题**：RRF 分 = Σ 1/(60+rank)，只由名次决定，\n"
          "    任何查询的 top1 都约等于 2/61 —— 它设计上就是**查询内**融合用的，\n"
          "    分数本来就不可跨查询比较。所以 Hybrid 不能直接喂给风控层。")

    # ---- 配对检验（Hit@1 与 Hit@5）
    # Hit@5 在多数臂上已接近天花板（BM25 就有 0.919），区分度被压没了；
    # 真正的差距在 Hit@1 与 MRR 上。所以两个 k 都报，不能只报好看的那个。
    print(f"\n配对检验（McNemar，逐题配对）")
    pairs = (("BM25", "Dense"), ("BM25", "Hybrid"), ("Hybrid", "Rerank"), ("BM25", "Rerank"))
    for k in (1, 5):
        print(f"  -- Hit@{k}")
        for a, b in pairs:
            pa = {p["id"]: p[f"hit@{k}"] for p in res[a]["per_item"]}
            pb = {p["id"]: p[f"hit@{k}"] for p in res[b]["per_item"]}
            ids = [p["id"] for p in res[a]["per_item"] if p["n_gold"] > 0]
            m = metrics.mcnemar([bool(pa[i]) for i in ids], [bool(pb[i]) for i in ids])
            print(f"     {a:7s} → {b:7s} b={m['b']:>2d} c={m['c']:>2d} p={m['p']:.4f}  "
                  f"（n={len(ids)}，一致 {len(ids) - m['b'] - m['c']}）  {m['note']}")

    # ---- dev/test 稳定性
    # 本实验**没有任何在 dev 上拟合的超参**（RRF 只用排名），
    # 所以两半的差不是过拟合，而是**这批题的抽样噪声**。
    # 报出来的用处：给上面所有臂间差距提供一个"多大算大"的参照。
    print(f"\ndev/test 稳定性（各 47 题；无超参拟合，差异 = 抽样噪声的量级）")
    print(f"{'臂':8s} {'dev Hit@1':>10s} {'test Hit@1':>11s} {'Δ':>8s} "
          f"{'dev Hit@5':>10s} {'test Hit@5':>11s} {'Δ':>8s} {'dev MRR':>9s} {'test MRR':>9s}")
    stab = {}
    for name in ARMS:
        row = {}
        for sp in ("dev", "test"):
            row[sp] = aggregate([p for p in res[name]["per_item"] if p["split"] == sp])
        stab[name] = row
        print(f"{name:8s} {row['dev']['hit@1']:>10.3f} {row['test']['hit@1']:>11.3f} "
              f"{row['test']['hit@1'] - row['dev']['hit@1']:>+8.3f} "
              f"{row['dev']['hit@5']:>10.3f} {row['test']['hit@5']:>11.3f} "
              f"{row['test']['hit@5'] - row['dev']['hit@5']:>+8.3f} "
              f"{row['dev']['mrr@5']:>9.3f} {row['test']['mrr@5']:>9.3f}")
    spreads = [abs(stab[n]["test"]["hit@1"] - stab[n]["dev"]["hit@1"]) for n in ARMS]
    print(f"  Hit@1 的 dev/test 绝对差：最大 {max(spreads):.3f}，"
          f"均值 {sum(spreads)/len(spreads):.3f}  ← 小于这个量级的臂间差距不值得解释")

    OUT.write_text(json.dumps(
        {name: {k: v for k, v in res[name].items() if k != "per_item"} for name in ARMS}
        | {"_random_baseline": rnd, "_build_s": build_s, "_n_chunks": n,
           "_blind_op": ops, "_stability": stab},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
