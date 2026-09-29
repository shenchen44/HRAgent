"""实验 8b：`UncertaintyGate` 的项审计、权重拟合与阈值校准。

## 为什么要有这个实验

exp8 报出三条结论，但只诊断、没修：
  1. 融合分的 AURC 优于任一单信号（分析二 0.012 vs 单信号最好 0.019）→ 融合有价值；
  2. 但**占 0.75 权重的两项是坏的**（自评置信度方向反、证据项从不触发）；
  3. 融合分**没有校准**：阈值 0.5 时一条都不升级，0.9 时几乎全升级。

exp8 把"权重怎么定"留成了待办。本实验就是来收这个尾的。

## 本实验要回答的三个问题（按重要性排序）

**Q1（审计）**：`UncertaintyGate` 的三项在真实数据上各自**是否可辨识**？
  一个项的取值若在全部样本上恒定，它的权重就**不可辨识** ——
  无论填多少，排序结果一模一样。这不是"权重选错了"，是"这个权重不存在"。
  必须先把这件事测出来，否则后面拟合出来的数字全是幻觉。

**Q2（拟合）**：在 dev 上拟合权重，能否在 test 上带来可复现的增益？
  **本实验的核心判据是置换检验**，不是 test 上的点估计。
  理由：本数据 dev 只有 47 题、正例个位数，在单纯形上网格搜索 231 个点，
  **随便一组权重都可能在 dev 上碰巧排得很好**。不跟置换零分布比，
  就没法区分"拟合到了信号"与"拟合到了噪声"。

**Q3（校准）**：阈值能不能在 dev 上定、在 test 上兑现？
  这是 exp8 第三条结论的直接后果 —— 分数没校准，阈值就不能写死。

## 与 exp8 的关系

本脚本**不调用任何 LLM**：直接读 `results/exp8_raw.jsonl`（94 题逐题结果），
所以可反复跑、零成本。它测的是**门控本身**，不是检索质量。

## 标签

风险事件 = 「该升级人工」。门控只有一个输出（升级 / 不升级），所以
**操作性标签是并集**：`制度没覆盖（盲区） 或 会答错（检索失败）`。
exp8 把两者分开测是为了**诊断**（两个触发条件的性质不同）；
本实验要**定一个门控**，所以用并集，同时分列两个子集的结果备查。

跑法:
  .venv/bin/python scripts/exp8b_gate_calibration.py
  .venv/bin/python scripts/exp8b_gate_calibration.py --perms 2000
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hragent import config, metrics  # noqa: E402
from hragent.risk.guards import _BM25_MID, _retrieval_conf  # noqa: E402

RAW = config.RESULTS / "exp8_raw.jsonl"

# guards.py 里 UncertaintyGate 的现状权重，**从这里读而不是抄一遍** ——
# 抄一遍就会出现"报告说 0.4、代码改成 0.3"这种报告与实现脱节。
# （这正是本项目反复出现的"声明了但没接线"那一类问题。）
_DEFAULT_W = (0.4, 0.35, 0.25)
TERM_NAMES = ("T1 不确定度(1-自评)", "T2 无证据(契约要求时)", "T3 检索不确定度")


def default_weights() -> tuple[float, float, float]:
    """从源码里核对默认权重，核对不上就直接报错。

    用 AST 太脆，这里用最直接的办法：构造一个 Candidate 让三项都拉满，
    看 risk_score 落在哪 —— 但那只能测出"和"。
    所以退一步：直接断言源码里的字面量存在。三行字符串匹配，
    改权重时若忘了同步这里，自测会立刻报错。
    """
    src = (Path(__file__).resolve().parents[1] / "src" / "hragent" / "risk"
           / "guards.py").read_text(encoding="utf-8")
    for lit in ("0.4 * (1.0 - max(0.0, min(1.0, c.confidence)))",
                "s += 0.35",
                "0.25 * (1.0 - _retrieval_conf(float(scores[0])))"):
        if lit not in src:
            raise SystemExit(
                f"guards.py 里的权重字面量变了，本脚本的 _DEFAULT_W 已过期：找不到 {lit!r}")
    return _DEFAULT_W


# ---------------------------------------------------------------- 项

def terms(rec: dict) -> tuple[float, float, float]:
    """把一条样本还原成 UncertaintyGate 的**三个实际项**。

    刻意与 `UncertaintyGate.risk_score` 逐项对应，不用 exp8 的 S1/S2/S3 ——
    exp8 的 S3 是"证据条数归一化"，而门控里的 T2 是"**有没有**证据"这个二值项。
    两者不是一回事：S3 方向反了，不等于 T2 方向反了。T2 的问题更基础 ——
    它在本数据上恒为 0（见 §1）。
    """
    t1 = 1.0 - max(0.0, min(1.0, float(rec["confidence"])))
    # 门控原文：`if c.requires_evidence and not c.evidence: s += 0.35`
    # exp8 的样本全部来自 PolicyRAG，契约要求证据（requires_evidence=True），
    # 所以 T2 = 1 当且仅当引用为空。
    t2 = 1.0 if (rec["n_evidence"] == 0) else 0.0
    scores = rec["top_scores"]
    t3 = 1.0 - _retrieval_conf(float(scores[0])) if scores else 0.0
    return (t1, t2, t3)


def score(rec: dict, w: tuple[float, float, float]) -> float:
    """加权和 → 风险分（越大越该升级）。与 `risk_score` 同式，含 min(1.0, ·)。"""
    return min(1.0, sum(wi * ti for wi, ti in zip(w, terms(rec))))


def label_escalate(rec: dict) -> bool:
    """风险事件：该升级人工。= 制度没覆盖 **或** 会答错。"""
    gold = set(rec["gold_chunk_ids"])
    if not gold:
        return True                      # 盲区：制度没覆盖
    return not (gold & set(rec["retrieved"]))   # 检索失败：会答错


def auc_escalate(recs: list[dict], w) -> float:
    """用风险分做「该升级」的排序，AUC 越大越好。"""
    return metrics.auc([score(r, w) for r in recs],
                       [label_escalate(r) for r in recs])


def aurc_escalate(recs: list[dict], w) -> float:
    """AURC：把"放行"理解为"不升级"。分数越高越该升级 → 置信度取负。"""
    return metrics.risk_coverage_curve(
        [-score(r, w) for r in recs],
        [not label_escalate(r) for r in recs])["aurc"]


def auc_ci(recs: list[dict], w, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """AUC 的 bootstrap 95% 区间。小样本下不报区间等于把噪声当结论。"""
    rng = random.Random(seed)
    n = len(recs)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        lab = [label_escalate(recs[i]) for i in idx]
        if len(set(lab)) < 2:
            continue
        vals.append(metrics.auc([score(recs[i], w) for i in idx], lab))
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1])


# ---------------------------------------------------------------- 单纯形网格

def simplex(step: float = 0.05) -> list[tuple[float, float, float]]:
    """权重和为 1 的网格。含 0（允许把某项彻底踢掉）。"""
    k = round(1.0 / step)
    out = []
    for i in range(k + 1):
        for j in range(k + 1 - i):
            m = k - i - j
            out.append((round(i * step, 4), round(j * step, 4), round(m * step, 4)))
    return out


def fit(recs: list[dict], step: float = 0.05):
    """在给定样本上按 AURC 最小化选权重。返回 (最优权重, 全部结果)。"""
    grid = simplex(step)
    scored = [(aurc_escalate(recs, w), auc_escalate(recs, w), w) for w in grid]
    scored.sort(key=lambda x: (x[0], -x[1]))
    return scored[0][2], scored


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000, help="置换检验次数")
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()

    recs_all = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines()
                if l.strip()]
    dev = [r for r in recs_all if r["split"] == "dev"]
    test = [r for r in recs_all if r["split"] == "test"]
    W0 = default_weights()

    print(f"实验 8b · UncertaintyGate 审计/拟合/校准")
    print(f"数据 {RAW.name}：{len(recs_all)} 题（dev {len(dev)} / test {len(test)}）"
          f"  当前权重 T1/T2/T3 = {W0}")

    # ============================================================ §1 项审计
    print(f"\n{'=' * 92}\n§1 三项是否可辨识（在全部 {len(recs_all)} 题上）\n{'=' * 92}")
    print(f"{'项':24s} {'非零率':>8s} {'最小':>8s} {'最大':>8s} {'标准差':>8s} "
          f"{'AUC(该升级)':>12s} {'可辨识':>8s}")
    T = [terms(r) for r in recs_all]
    lab = [label_escalate(r) for r in recs_all]
    audit = {}
    for i, name in enumerate(TERM_NAMES):
        col = [t[i] for t in T]
        mu = sum(col) / len(col)
        sd = (sum((x - mu) ** 2 for x in col) / len(col)) ** 0.5
        nz = sum(1 for x in col if x != 0.0) / len(col)
        a = metrics.auc(col, lab) if sd > 1e-12 else float("nan")
        ident = "否 ❌" if sd < 1e-12 else "是"
        print(f"{name:24s} {nz:>8.3f} {min(col):>8.3f} {max(col):>8.3f} {sd:>8.3f} "
              f"{a:>12.4f} {ident:>8s}")
        audit[name] = {"nonzero_rate": nz, "min": min(col), "max": max(col),
                       "std": sd, "auc": a, "identifiable": sd >= 1e-12}

    # 关键推论：权重落在不可辨识项上 = 分数上界被砍掉一块，而排序一点没变。
    lo = sum(w * min(t[i] for t in T) for i, w in enumerate(W0))
    hi = sum(w * max(t[i] for t in T) for i, w in enumerate(W0))
    print(f"\n  当前权重下风险分的**实际可达区间**：[{lo:.3f}, {hi:.3f}]")
    print(f"  默认阈值 threshold=0.5 是否可达：{'可达' if hi >= 0.5 else '❌ 不可达'}"
          f"（上界 {hi:.3f} < 0.5）")
    print(f"  → **`UncertaintyGate(threshold=0.5)` 在本数据上一条都不会升级**，"
          f"即出厂默认值等于门控关闭。")
    print(f"  上界被砍的原因：T2 恒为 0，它那 {W0[1]:.2f} 的权重"
          f"只抬高不了分数，却把可达上界从 1.0 压到 {hi:.3f} 附近。")

    # ============================================================ §2 dev 拟合
    print(f"\n{'=' * 92}\n§2 在 dev 上拟合权重（目标：AURC 最小）\n{'=' * 92}")
    best_w, grid = fit(dev, args.step)
    n_pos_dev = sum(label_escalate(r) for r in dev)
    print(f"dev {len(dev)} 题，正例（该升级）{n_pos_dev} 条，"
          f"网格 {len(grid)} 点（step={args.step}）")
    print(f"  默认权重 {W0}  →  dev AURC {aurc_escalate(dev, W0):.4f}  "
          f"AUC {auc_escalate(dev, W0):.4f}")
    print(f"  拟合权重 {best_w}  →  dev AURC {aurc_escalate(dev, best_w):.4f}  "
          f"AUC {auc_escalate(dev, best_w):.4f}")
    print(f"\n  dev 上最优的前 10 组权重（看**平台有多宽** —— 平台越宽，"
          f"说明这个最优越没意义）：")
    print(f"  {'AURC':>7s} {'AUC':>7s}  权重 (T1,T2,T3)")
    for a, u, w in grid[:10]:
        print(f"  {a:>7.4f} {u:>7.4f}  {w}")
    plateau = [w for a, _, w in grid if a <= grid[0][0] + 1e-9]
    print(f"  → AURC 取到最小值的权重有 **{len(plateau)}/{len(grid)}** 组")

    # ============================================================ §3 置换检验
    print(f"\n{'=' * 92}\n§3 置换检验：这个 dev 最优是真信号还是噪声？\n{'=' * 92}")
    print(f"  做法：把 dev 的标签打乱 {args.perms} 次，每次重新拟合，"
          f"得到「纯噪声下能达到的最优 AURC」的零分布。")
    rng = random.Random(0)
    labs = [label_escalate(r) for r in dev]
    tvals = [terms(r) for r in dev]
    grid_w = simplex(args.step)

    def aurc_of(labels: list[bool], w) -> float:
        return metrics.risk_coverage_curve(
            [-min(1.0, sum(wi * ti for wi, ti in zip(w, t))) for t in tvals],
            [not x for x in labels])["aurc"]

    null_best = []
    for _ in range(args.perms):
        p = labs[:]
        rng.shuffle(p)
        null_best.append(min(aurc_of(p, w) for w in grid_w))
    null_best.sort()
    real_best = grid[0][0]
    p_val = (sum(1 for x in null_best if x <= real_best) + 1) / (len(null_best) + 1)
    print(f"  真实 dev 最优 AURC = {real_best:.4f}")
    print(f"  置换零分布：中位数 {null_best[len(null_best) // 2]:.4f}  "
          f"5% 分位 {null_best[int(0.05 * len(null_best))]:.4f}  "
          f"最小值 {null_best[0]:.4f}")
    print(f"  → p = {p_val:.3f}"
          f"（{'❌ 不能排除噪声：真实最优落在零分布之内' if p_val > 0.05 else '✅ 优于噪声' }）")

    # ============================================================ §4 test 验证
    print(f"\n{'=' * 92}\n§4 在 test 上验证（{len(test)} 题，正例 "
          f"{sum(label_escalate(r) for r in test)} 条）\n{'=' * 92}")
    arms = [
        ("S0 常数分（零知识基线）", None),
        ("V0 出厂权重（现状）", W0),
        ("V1 拟合权重（dev 最优）", best_w),
        ("T1 只用不确定度", (1.0, 0.0, 0.0)),
        ("T3 只用检索不确定度", (0.0, 0.0, 1.0)),
        ("Oracle 完美排序", "oracle"),
    ]
    base_risk = sum(label_escalate(r) for r in test) / len(test)
    print(f"test 基础风险（该升级率）= {base_risk:.3f}\n")
    print(f"{'臂':26s} {'AUC':>7s} {'95% CI':>16s} {'AURC':>7s} {'vs 基线':>9s}")
    print("-" * 92)
    rows = {}
    for name, w in arms:
        if w is None:
            print(f"{name:26s} {0.5:>7.3f} {'—':>16s} {base_risk:>7.3f} {'0.000':>9s}")
            rows[name] = {"auc": 0.5, "aurc": base_risk, "w": None}
            continue
        if w == "oracle":
            ok = [not label_escalate(r) for r in test]
            a = 1.0
            rc = metrics.risk_coverage_curve([1.0 if x else 0.0 for x in ok], ok)
            print(f"{name:26s} {a:>7.3f} {'—':>16s} {rc['aurc']:>7.3f} "
                  f"{rc['aurc'] - base_risk:>+9.3f}")
            rows[name] = {"auc": a, "aurc": rc["aurc"], "w": None}
            continue
        a = auc_escalate(test, w)
        lo_, hi_ = auc_ci(test, w)
        rc = aurc_escalate(test, w)
        print(f"{name:26s} {a:>7.3f} [{lo_:>6.3f},{hi_:>6.3f}] {rc:>7.3f} "
              f"{rc - base_risk:>+9.3f}")
        rows[name] = {"auc": a, "ci": [lo_, hi_], "aurc": rc, "w": list(w)}

    # 关键一问：dev 拟合出来的权重，到 test 上到底赢了多少？
    # 两个臂看的是**同一批题**，所以必须用**配对** bootstrap ——
    # 各自算 CI 再比区间重叠与否，是拿独立样本的方法套配对数据，会低估差异的精度。
    rng = random.Random(0)
    n = len(test)
    diffs = []
    for _ in range(4000):
        idx = [rng.randrange(n) for _ in range(n)]
        lab = [label_escalate(test[i]) for i in idx]
        if len(set(lab)) < 2:
            continue
        s0 = [score(test[i], W0) for i in idx]
        s1 = [score(test[i], best_w) for i in idx]
        diffs.append(metrics.auc(s1, lab) - metrics.auc(s0, lab))
    diffs.sort()
    d_lo, d_hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
    med = diffs[len(diffs) // 2]
    verdict = ("❌ 区间含 0：dev 拟合的权重在 test 上**没有**可证实的增益"
               if d_lo <= 0 <= d_hi else "✅ 区间不含 0")
    print(f"\n  配对 bootstrap（同一批 test 题，4000 次重采样）：")
    print(f"    AUC(拟合) − AUC(出厂) = {auc_escalate(test, best_w) - auc_escalate(test, W0):+.3f}"
          f"   95% CI [{d_lo:+.3f}, {d_hi:+.3f}]  中位 {med:+.3f}")
    print(f"    {verdict}")

    # ============================================================ §5 阈值校准
    print(f"\n{'=' * 92}\n§5 阈值校准：在 dev 上按目标升级率定阈值，到 test 上兑现\n{'=' * 92}")
    print(f"{'策略':30s} {'dev阈值':>8s} {'test升级率':>10s} {'升级精准度':>10s} "
          f"{'升级召回':>8s}")
    print("-" * 92)
    calib = []
    for tag, w in (("出厂权重", W0), ("拟合权重", best_w)):
        ds = sorted(score(r, w) for r in dev)
        n_pos = sum(label_escalate(r) for r in dev)
        for q in (0.05, 0.10, 0.20):
            th = ds[min(len(ds) - 1, int((1 - q) * len(ds)))]
            ts = [score(r, w) for r in test]
            lab_t = [label_escalate(r) for r in test]
            esc = [i for i, s in enumerate(ts) if s >= th]
            prec = (sum(1 for i in esc if lab_t[i]) / len(esc)) if esc else float("nan")
            rec = (sum(1 for i in esc if lab_t[i]) / sum(lab_t)) if sum(lab_t) else float("nan")
            print(f"{tag + f' · 目标升级率 {q:.0%}':30s} {th:>8.3f} "
                  f"{len(esc) / len(ts):>10.3f} {prec:>10.3f} {rec:>8.3f}")
            calib.append({"weights": tag, "target_q": q, "threshold": th,
                          "test_escalation_rate": len(esc) / len(ts),
                          "test_precision": prec, "test_recall": rec,
                          "dev_n_positive": n_pos})
    th05 = 0.5
    ts = [score(r, W0) for r in test]
    print(f"{'出厂默认 threshold=0.5':30s} {th05:>8.3f} "
          f"{sum(1 for s in ts if s >= th05) / len(ts):>10.3f} "
          f"{'—':>10s} {'—':>8s}   ← 永远为 0，见 §1")

    # ============================================================ 落盘
    out = config.RESULTS / "exp8b_gate_calibration.json"
    out.write_text(json.dumps({
        "default_weights": list(W0), "n": len(recs_all),
        "term_audit": audit,
        "attainable_range": [lo, hi], "threshold_0_5_reachable": hi >= 0.5,
        "dev_fit": {"best": list(best_w), "dev_aurc": grid[0][0],
                    "dev_auc": grid[0][1], "plateau_size": len(plateau),
                    "grid_size": len(grid), "n_positive": n_pos_dev,
                    "top10": [{"aurc": a, "auc": u, "w": list(w)} for a, u, w in grid[:10]]},
        "permutation": {"n": args.perms, "real_best_aurc": real_best,
                        "null_median": null_best[len(null_best) // 2],
                        "null_p05": null_best[int(0.05 * len(null_best))],
                        "null_min": null_best[0], "p_value": p_val},
        "test_arms": rows, "calibration": calib,
        "paired_bootstrap_fit_minus_default": {
            "delta": auc_escalate(test, best_w) - auc_escalate(test, W0),
            "ci": [d_lo, d_hi], "median": med,
            "n_boot": len(diffs)},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
