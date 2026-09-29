"""评测指标库 —— 纯函数，无 I/O，可单测。

设计约束（EVAL_PROTOCOL.md 通用规则）：
  R2  准确率必须与成本成对报告        → acc_with_cost()
  R3  拦截率必须与误报率成对报告      → guard_report()
  R4  区间估计用 95% bootstrap CI     → bootstrap_ci()
  R5  配对比较用 McNemar              → mcnemar()
  R6  负面结果也要报                  → 本模块不做任何"过滤掉难看数字"的事

风控部分的核心是**风险-覆盖曲线**（risk-coverage），不是单点拦截率：
  只报"拦截率 95%"是没有信息量的 —— 全部拦截就是 100%。
  必须报出"在放行 X% 的情况下，错误率是多少"这条曲线，以及 AURC（曲线下面积）。
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Callable, Sequence


# ------------------------------------------------------------ 区间估计
def bootstrap_ci(values: Sequence[float], stat: Callable = None, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 20260928) -> tuple[float, float, float]:
    """95% bootstrap 置信区间。返回 (点估计, 下界, 上界)。

    样本量 < 2 时退化为 (点估计, 点估计, 点估计) 并如实返回，不假装有区间。
    """
    vals = list(values)
    if not vals:
        return (float("nan"),) * 3
    stat = stat or (lambda xs: sum(xs) / len(xs))
    point = stat(vals)
    if len(vals) < 2:
        return point, point, point
    rng = random.Random(seed)
    n = len(vals)
    boots = []
    for _ in range(n_boot):
        boots.append(stat([vals[rng.randrange(n)] for _ in range(n)]))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return point, lo, hi


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """比例的 Wilson 区间。小样本下比正态近似可靠。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """AUC：某个连续分数把正负样本分开的能力。0.5 = 无区分力。

    用 **rank 统计量**（Mann-Whitney U）算，不按阈值扫描：
        AUC = (Σ rank(正样本) − n₊(n₊+1)/2) / (n₊·n₋)
    这样只依赖样本间的**相对次序**，与分数的量纲无关 ——
    对 BM25（无界词频分）和余弦（[0,1]）可以直接横向比较。

    并列分数取平均秩（否则并列会被当成有序，虚高 AUC）。
    """
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):                      # 并列取平均秩
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    rank_sum = sum(ranks[k] for k, y in enumerate(labels) if y)
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


# ------------------------------------------------------------ 配对比较
def mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict:
    """McNemar 检验：同一批题上比较两个系统。返回 b, c, 统计量, p 值。

    b = A 错 B 对，c = A 对 B 错。只关心不一致的格子 —— 一致的格子不提供信息。
    p 值用精确二项检验（b+c 较小时比卡方可靠）。
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("配对比较要求等长")
    b = sum(1 for x, y in zip(correct_a, correct_b) if not x and y)
    c = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "stat": 0.0, "p": 1.0, "note": "两系统完全一致"}
    # 精确二项双尾 p
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    p = min(1.0, p)
    stat = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    return {"b": b, "c": c, "stat": stat, "p": p,
            "note": "p<0.05 表示差异显著" if p < 0.05 else "差异不显著"}


# ------------------------------------------------------------ 分类 / 抽取
def prf1(tp: int, fp: int, fn: int) -> dict:
    """精确率/召回率/F1。分母为 0 时如实返回 0 而不是 nan —— 0 样本即 0 表现。"""
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def set_prf1(pred: Sequence[str], gold: Sequence[str]) -> dict:
    """集合级抽取指标（技能抽取用）。"""
    ps, gs = set(pred), set(gold)
    return prf1(len(ps & gs), len(ps - gs), len(gs - ps))


def micro_macro_prf1(preds: Sequence[Sequence[str]], golds: Sequence[Sequence[str]]) -> dict:
    """微平均（按样本汇总）+ 宏平均（按样本平均）。两者差距大说明样本难度不均。"""
    micro = set_prf1([x for p in preds for x in p], [x for g in golds for x in g])
    per = [set_prf1(p, g) for p, g in zip(preds, golds)]
    macro = {k: (sum(x[k] for x in per) / len(per) if per else 0.0)
             for k in ("precision", "recall", "f1")}
    return {"micro": micro, "macro": macro, "n": len(preds)}


# ------------------------------------------------------------ 排序
def ndcg_at_k(rels: Sequence[int], k: int, gains: Sequence[int] | None = None) -> float:
    """NDCG@k。rels 为按系统排序后的相关性序列（已排序，不是原始顺序）。"""
    gains = gains or rels

    def dcg(xs):
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(xs[:k]))
    ideal = sorted(gains, reverse=True)
    idcg = dcg(ideal)
    return dcg(rels) / idcg if idcg > 0 else 0.0


def mrr(ranked_relevant: Sequence[int]) -> float:
    """第一个相关项排在第几位。ranked_relevant 为 1/0 序列。"""
    for i, r in enumerate(ranked_relevant):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_relevant: Sequence[int], k: int) -> float:
    total = sum(1 for r in ranked_relevant if r > 0)
    if total == 0:
        return 0.0
    return sum(1 for r in ranked_relevant[:k] if r > 0) / total


# ------------------------------------------------------------ 公平性
def disparate_impact_at_k(scores: Sequence[float], groups: Sequence[str], k: int) -> dict:
    """top-k 选中率在敏感属性各组间的差异。

    DI = min(选中率) / max(选中率)。业界常用的 4/5 法则：DI < 0.8 视为存在不利影响。
    这里的分组标签是**随机分配、与金标独立**的（见 gen_match_eval.py 的守门检验），
    所以理想系统上 DI 应接近 1，偏离即偏见。
    """
    if len(scores) != len(groups):
        raise ValueError("scores 与 groups 等长")
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    sel = set(order[:k])
    rates = {}
    for g in sorted(set(groups)):
        idx = [i for i, x in enumerate(groups) if x == g]
        rates[g] = sum(1 for i in idx if i in sel) / len(idx) if idx else 0.0
    mx = max(rates.values()) if rates else 0.0
    mn = min(rates.values()) if rates else 0.0
    return {"rates": rates, "di": (mn / mx if mx > 0 else 1.0),
            "max_gap": mx - mn, "k": k}


# ------------------------------------------------------------ 风控 / 风险-覆盖
def guard_report(pred_block: Sequence[bool], gold_block: Sequence[bool]) -> dict:
    """单个闸门：拦截率(recall) 与 误报率(FPR) 必须成对给出。

    gold_block=True 表示该样本**应该被拦**。
    """
    tp = sum(1 for p, g in zip(pred_block, gold_block) if p and g)
    fn = sum(1 for p, g in zip(pred_block, gold_block) if not p and g)
    fp = sum(1 for p, g in zip(pred_block, gold_block) if p and not g)
    tn = sum(1 for p, g in zip(pred_block, gold_block) if not p and not g)
    block_recall = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    prec = tp / (tp + fp) if tp + fp else 0.0
    lo_r, hi_r = wilson_interval(tp, tp + fn)
    lo_f, hi_f = wilson_interval(fp, fp + tn)
    return {
        "n": len(pred_block), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "block_recall": block_recall, "block_recall_ci": (lo_r, hi_r),
        "fpr": fpr, "fpr_ci": (lo_f, hi_f),
        "precision": prec,
        "accuracy": (tp + tn) / len(pred_block) if pred_block else float("nan"),
    }


def risk_coverage_curve(confidence: Sequence[float], correct: Sequence[bool],
                        n_points: int = 21) -> dict:
    """风险-覆盖曲线。

    按置信度从高到低放行：放行比例 = coverage，放行部分的错误率 = risk。
    单点拦截率无法反映"漏放多少"，曲线可以。
    AURC 越小越好。
    """
    if len(confidence) != len(correct):
        raise ValueError("confidence 与 correct 等长")
    n = len(confidence)
    if n == 0:
        return {"coverage": [], "risk": [], "aurc": float("nan"), "n": 0}
    order = sorted(range(n), key=lambda i: -confidence[i])
    covs, risks = [], []
    for j in range(1, n_points + 1):
        c = max(1, round(n * j / n_points))
        head = order[:c]
        err = sum(1 for i in head if not correct[i]) / c
        covs.append(c / n)
        risks.append(err)
    # 梯形积分
    aurc = 0.0
    for i in range(1, len(covs)):
        aurc += (covs[i] - covs[i - 1]) * (risks[i] + risks[i - 1]) / 2
    return {"coverage": covs, "risk": risks, "aurc": aurc, "n": n,
            "risk_at_80": _risk_at(covs, risks, 0.8),
            "risk_at_100": risks[-1] if risks else float("nan")}


def _risk_at(covs: Sequence[float], risks: Sequence[float], target: float) -> float:
    best = None
    for c, r in zip(covs, risks):
        if c <= target + 1e-9:
            best = r
    return best if best is not None else float("nan")


def hitl_report(confidence: Sequence[float], correct: Sequence[bool],
                threshold: float) -> dict:
    """人在环：置信度低于阈值转人工。返回自动处理率与自动处理部分的准确率。"""
    auto = [i for i, c in enumerate(confidence) if c >= threshold]
    esc = [i for i, c in enumerate(confidence) if c < threshold]
    auto_acc = (sum(1 for i in auto if correct[i]) / len(auto)) if auto else float("nan")
    return {
        "threshold": threshold,
        "coverage": len(auto) / len(confidence) if confidence else float("nan"),
        "auto_accuracy": auto_acc,
        "escalation_rate": len(esc) / len(confidence) if confidence else float("nan"),
        # 转人工的样本里有多少是"真该转"（本来就会答错）—— 转人工的精准度
        "escalation_precision": (sum(1 for i in esc if not correct[i]) / len(esc))
        if esc else float("nan"),
        "n": len(confidence), "n_auto": len(auto), "n_escalated": len(esc),
    }


# ------------------------------------------------------------ 成本
def acc_with_cost(correct: Sequence[bool], tokens: Sequence[int],
                  latency: Sequence[float]) -> dict:
    """R2：准确率必须与成本成对报告。

    单看准确率会诱导"堆算力"；单看成本会诱导"不干活"。
    """
    n = len(correct)
    if n == 0:
        return {}
    acc = sum(correct) / n
    tot_tok = sum(tokens)
    tot_lat = sum(latency)
    return {
        "n": n, "accuracy": acc,
        "accuracy_ci": bootstrap_ci([float(c) for c in correct])[1:],
        "total_tokens": tot_tok, "tokens_per_query": tot_tok / n,
        "total_latency_s": tot_lat, "latency_per_query_s": tot_lat / n,
        # 每正确回答一个问题的代价 —— 跨条件比较时比裸准确率公平
        "tokens_per_correct": tot_tok / sum(correct) if sum(correct) else float("inf"),
        "latency_per_correct": tot_lat / sum(correct) if sum(correct) else float("inf"),
    }


# ------------------------------------------------------------ 混淆
def confusion(pred: Sequence[str], gold: Sequence[str]) -> dict:
    labels = sorted(set(gold) | set(pred))
    m = {g: Counter() for g in labels}
    for p, g in zip(pred, gold):
        m[g][p] += 1
    acc = sum(1 for p, g in zip(pred, gold) if p == g) / len(gold) if gold else float("nan")
    return {"matrix": {g: dict(c) for g, c in m.items()}, "labels": labels,
            "accuracy": acc, "n": len(gold)}
