"""指标库自测：每个函数用手算值对。

为什么必须做：指标库算错是**静默**的 —— 数字照样出来，只是不对。
P2/P3 一跑就是几百个数字，到时候没人能逐个验算。所以在这里一次性钉死。

跑法: .venv/bin/python scripts/selftest_metrics.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hragent import metrics as M  # noqa: E402

FAILS: list[str] = []
CHECKS = 0


def close(a, b, tol=1e-4) -> bool:
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return abs(a - b) <= tol


def eq(name: str, got, want, tol=1e-4) -> None:
    global CHECKS
    CHECKS += 1
    ok = close(got, want, tol) if isinstance(want, (int, float)) and not isinstance(want, bool) \
        else got == want
    if not ok:
        FAILS.append(f"{name}: got={got!r} want={want!r}")
    print(f"   {'✅' if ok else '❌'} {name}")


def main() -> None:
    print("指标库自测\n")

    # ---------------- bootstrap / wilson
    print("[区间估计]")
    p, lo, hi = M.bootstrap_ci([1.0] * 50)
    eq("bootstrap 常量样本 → 区间退化到点", (p, lo, hi), (1.0, 1.0, 1.0))
    p, lo, hi = M.bootstrap_ci([0.0, 1.0] * 100)
    eq("bootstrap 点估计 = 0.5", p, 0.5)
    eq("bootstrap 下界 < 0.5", lo < 0.5, True)
    eq("bootstrap 上界 > 0.5", hi > 0.5, True)
    eq("bootstrap 空输入 → nan", math.isnan(M.bootstrap_ci([])[0]), True)

    lo, hi = M.wilson_interval(0, 10)
    eq("wilson k=0,n=10 下界 = 0", lo, 0.0, 1e-9)
    eq("wilson k=0,n=10 上界 ≈ 0.2776", hi, 0.2776, 1e-3)
    lo, hi = M.wilson_interval(10, 10)
    eq("wilson k=n 上界 = 1", hi, 1.0, 1e-9)
    eq("wilson k=n 下界 ≈ 0.7224", lo, 0.7224, 1e-3)

    # ---------------- mcnemar
    print("\n[配对检验 McNemar]")
    r = M.mcnemar([False] * 10, [True] * 10)
    eq("b=10,c=0 → p = 2/1024 ≈ 0.00195", r["p"], 0.001953125, 1e-9)
    eq("b=10,c=0 → 显著", r["p"] < 0.05, True)
    r = M.mcnemar([True, False] * 5, [False, True] * 5)
    eq("b=5,c=5 → p = 1.0", r["p"], 1.0)
    eq("b=5,c=5 → 不显著", r["p"] >= 0.05, True)
    r = M.mcnemar([True] * 5, [True] * 5)
    eq("完全一致 → p = 1.0", r["p"], 1.0)
    try:
        M.mcnemar([True], [True, False])
        eq("不等长应报错", False, True)
    except ValueError:
        eq("不等长应报错", True, True)

    # ---------------- prf1
    print("\n[抽取指标]")
    r = M.prf1(3, 1, 1)
    eq("prf1(3,1,1) precision = 0.75", r["precision"], 0.75)
    eq("prf1(3,1,1) recall = 0.75", r["recall"], 0.75)
    eq("prf1(3,1,1) f1 = 0.75", r["f1"], 0.75)
    r = M.prf1(0, 0, 0)
    eq("prf1 全零 → 0 而不是 nan", (r["precision"], r["recall"], r["f1"]), (0.0, 0.0, 0.0))

    r = M.set_prf1(["A", "B", "C"], ["B", "C", "D"])
    eq("set_prf1 交集 2/预测 3 → precision = 2/3", r["precision"], 2 / 3)
    eq("set_prf1 交集 2/金标 3 → recall = 2/3", r["recall"], 2 / 3)

    r = M.micro_macro_prf1([["A"], ["B", "C"]], [["A"], ["B"]])
    eq("micro: tp=2 fp=1 fn=0 → precision = 2/3", r["micro"]["precision"], 2 / 3)
    eq("micro: recall = 1.0", r["micro"]["recall"], 1.0)
    eq("macro: (1.0 + 0.5)/2 precision = 0.75", r["macro"]["precision"], 0.75)

    # ---------------- 排序
    print("\n[排序指标]")
    # rels=[2,1,0] 已是理想序 → NDCG=1
    eq("ndcg 理想排序 = 1.0", M.ndcg_at_k([2, 1, 0], 3), 1.0)
    # rels=[0,2,1]: DCG = 0/1 + 3/log2(3) + 1/log2(4) = 1.8927893 + 0.5 = 2.3927893
    #                IDCG = 3/1 + 1/log2(3) + 0 = 3.6309298  → 0.659001
    eq("ndcg [0,2,1] ≈ 0.6590", M.ndcg_at_k([0, 2, 1], 3), 0.6590, 1e-3)
    eq("ndcg 全 0 相关 → 0.0", M.ndcg_at_k([0, 0, 0], 3), 0.0)
    eq("ndcg k=1 只看首位，相关则 1.0", M.ndcg_at_k([2, 0, 0], 1), 1.0)
    eq("ndcg k=1 首位不相关 → 0.0", M.ndcg_at_k([0, 2, 0], 1), 0.0)

    eq("mrr 第 1 位命中 = 1.0", M.mrr([1, 0, 0]), 1.0)
    eq("mrr 第 3 位命中 = 1/3", M.mrr([0, 0, 1]), 1 / 3)
    eq("mrr 全不命中 = 0.0", M.mrr([0, 0, 0]), 0.0)
    eq("recall@2 命中 1/2 相关", M.recall_at_k([1, 0, 1], 2), 0.5)
    eq("recall@3 命中 2/2 相关", M.recall_at_k([1, 0, 1], 3), 1.0)

    # ---------------- AUC
    print("\n[AUC]")
    # 完全可分：负样本分数全低、正样本全高
    eq("完全可分 → AUC = 1.0", M.auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]), 1.0)
    # 完全反序
    eq("完全反序 → AUC = 0.0", M.auc([0.8, 0.9, 0.1, 0.2], [False, False, True, True]), 0.0)
    # 全部并列：秩全相同 → 恰好 0.5。这是"并列必须取平均秩"的直接后果，
    # 若按"严格大于"计数，全并列会算成 0.0（错得很难看）
    eq("全并列 → AUC = 0.5", M.auc([1.0] * 6, [True, False] * 3), 0.5)
    # 正样本两两并列在顶部：秩各为 3.5 → AUC = (7 − 2·3/2)/4 = 1.0
    eq("正样本并列居顶 → AUC = 1.0", M.auc([1.0, 1.0, 0.0, 0.0], [True, True, False, False]), 1.0)
    # 同分数上正负各一个 → 并列被摊平，只能是 0.5（若按"大于等于"计数会虚高成 1.0）
    eq("同分处正负各一 → AUC = 0.5", M.auc([1.0, 1.0, 0.0, 0.0], [True, False, True, False]), 0.5)
    # 手算：scores=[1,2,3,4] labels=[T,F,T,F] → 正样本秩 1,3，和 4
    #       AUC = (4 − 2·3/2) / (2·2) = 1/4
    eq("交错 → AUC = 0.25", M.auc([1, 2, 3, 4], [True, False, True, False]), 0.25)
    # 单一类别 → 无法定义，必须返回 nan 而不是 0.5（0.5 会被误读成"无区分力"）
    eq("只有正样本 → nan", math.isnan(M.auc([1, 2], [True, True])), True)
    eq("只有负样本 → nan", math.isnan(M.auc([1, 2], [False, False])), True)
    eq("空输入 → nan", math.isnan(M.auc([], [])), True)

    # 秩统计量的**关键性质**：只依赖相对次序。
    # BM25 分无界、余弦在 [0,1]，靠这条才能横向比较 —— 这正是本函数不按阈值扫描的理由。
    base = [0.1, 0.9, 0.4, 0.5, 0.2, 0.7]
    lab = [False, True, False, True, False, True]
    a0 = M.auc(base, lab)
    eq("单调变换不变：×100 + 7", M.auc([100 * x + 7 for x in base], lab), a0)
    eq("单调变换不变：log", M.auc([math.log(x) for x in base], lab), a0)
    eq("单调变换不变：倒数（反向）→ 1 − AUC", M.auc([1 / x for x in base], lab), 1 - a0, 1e-9)

    # ---------------- 公平性
    print("\n[公平性]")
    # top2 = 索引 3,2，都是 B → A 选中率 0，B 选中率 1
    r = M.disparate_impact_at_k([1, 2, 3, 4], ["A", "A", "B", "B"], 2)
    eq("DI: A 组全落选 → DI = 0.0", r["di"], 0.0)
    eq("DI: 最大差 = 1.0", r["max_gap"], 1.0)
    # top2 = 索引 3(B), 2(A) → 两组各 1/2
    r = M.disparate_impact_at_k([1, 2, 3, 4], ["A", "B", "A", "B"], 2)
    eq("DI: 两组选中率相同 → DI = 1.0", r["di"], 1.0)
    eq("DI: 最大差 = 0.0", r["max_gap"], 0.0)

    # ---------------- 风控
    print("\n[风控闸门]")
    r = M.guard_report([True, True, False, False], [True, True, False, False])
    eq("完美闸门 拦截率 = 1.0", r["block_recall"], 1.0)
    eq("完美闸门 误报率 = 0.0", r["fpr"], 0.0)
    eq("完美闸门 准确率 = 1.0", r["accuracy"], 1.0)
    # 全拦：拦截率 1.0，但误报率也 1.0 —— 这正是"只报拦截率"会掩盖的
    r = M.guard_report([True] * 4, [True, True, False, False])
    eq("全拦 拦截率 = 1.0", r["block_recall"], 1.0)
    eq("全拦 误报率 = 1.0（必须暴露）", r["fpr"], 1.0)
    # 全放
    r = M.guard_report([False] * 4, [True, True, False, False])
    eq("全放 拦截率 = 0.0", r["block_recall"], 0.0)
    eq("全放 误报率 = 0.0", r["fpr"], 0.0)
    eq("全放 准确率 = 0.5", r["accuracy"], 0.5)

    # ---------------- 风险-覆盖
    print("\n[风险-覆盖曲线]")
    # 全对时风险恒为 0，AURC 必须精确为 0
    r0 = M.risk_coverage_curve([0.9, 0.8, 0.7, 0.6], [True] * 4)
    eq("全对 → 每个覆盖点风险 0", set(r0["risk"]), {0.0})
    eq("全对 → AURC = 0", r0["aurc"], 0.0, 1e-9)

    n = 10
    correct = [True] * 5 + [False] * 5
    # 置信度与正确性完全一致：对的置信度高（oracle 排序）
    conf = [1.0 - i * 0.05 for i in range(5)] + [0.5 - i * 0.05 for i in range(5)]
    r = M.risk_coverage_curve(conf, correct)
    # oracle 排序下的理论下界（连续覆盖）：
    #   risk(c) = 0 (c<=0.5)，= (c-0.5)/c (c>0.5)
    #   ∫_0.5^1 (c-0.5)/c dc = [c - 0.5·ln c]_0.5^1 = 0.5 - 0.5·ln0.5 ≈ 0.15343
    # 离散化会有小偏差，故给 0.01 容差
    eq("oracle 排序 AURC ≈ 0.1534（理论下界）", r["aurc"], 0.15343, 0.01)
    eq("全放行风险 = 整体错误率 0.5", r["risk_at_100"], 0.5)
    eq("覆盖 80% 时风险 ≈ (0.8-0.5)/0.8 = 0.375", r["risk_at_80"], 0.375, 0.05)

    # 置信度完全反了：错的置信度最高
    conf_bad = [0.5 - i * 0.05 for i in range(5)] + [1.0 - i * 0.05 for i in range(5)]
    r2 = M.risk_coverage_curve(conf_bad, correct)
    eq("置信度完全反序 → AURC > 0", r2["aurc"] > 0, True)
    eq("反序时 AURC 大于 oracle 排序", r2["aurc"] > r["aurc"], True)
    eq("反序时首位就是错的 → 覆盖 5% 风险 = 1.0", r2["risk"][0], 1.0)

    # 性质检验：oracle 排序的 AURC 必须是所有排序里的最小值
    import random as _rnd
    rr = _rnd.Random(7)
    worse = 0
    for _ in range(300):
        perm = conf[:]
        rr.shuffle(perm)
        if M.risk_coverage_curve(perm, correct)["aurc"] < r["aurc"] - 1e-9:
            worse += 1
    eq("300 次随机排序中，无一优于 oracle 排序", worse, 0)

    # ---------------- HITL
    print("\n[HITL]")
    r = M.hitl_report([0.9, 0.9, 0.1, 0.1], [True, True, False, False], 0.5)
    eq("阈值 0.5 → 自动处理率 0.5", r["coverage"], 0.5)
    eq("自动部分准确率 = 1.0", r["auto_accuracy"], 1.0)
    eq("转人工率 = 0.5", r["escalation_rate"], 0.5)
    eq("转人工精准度 = 1.0（转的全是错题）", r["escalation_precision"], 1.0)

    # ---------------- 成本
    print("\n[成本]")
    r = M.acc_with_cost([True, False, True, True], [100, 200, 300, 400], [1, 2, 3, 4])
    eq("准确率 = 0.75", r["accuracy"], 0.75)
    eq("总 token = 1000", r["total_tokens"], 1000)
    eq("每题 token = 250", r["tokens_per_query"], 250)
    eq("每正确题 token = 1000/3", r["tokens_per_correct"], 1000 / 3)

    # ---------------- 混淆
    print("\n[混淆矩阵]")
    r = M.confusion(["a", "a", "b"], ["a", "b", "b"])
    eq("准确率 = 2/3", r["accuracy"], 2 / 3)
    eq("gold=a 预测分布", r["matrix"]["a"], {"a": 1})
    eq("gold=b 预测分布", r["matrix"]["b"], {"a": 1, "b": 1})

    print(f"\n{'=' * 56}")
    if FAILS:
        print(f"❌ {len(FAILS)}/{CHECKS} 项未通过：")
        for f in FAILS:
            print(f"   - {f}")
        raise SystemExit(1)
    print(f"✅ 全部 {CHECKS} 项自测通过")


if __name__ == "__main__":
    main()
