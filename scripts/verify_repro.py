"""复现校验：把报告里已发表的数字，从**落盘的逐条记录**独立重算一遍。

为什么需要它（DESIGN §9 · P4 的验收标准是"全部数字可复现"）：
  报告里的数字是实验脚本运行时算出来、写进 JSON 的。只读 JSON，验证的是
  "文件没被手改"，不是"数字算对了"。真正要防的是**聚合代码本身写错**，
  而逐条记录是对的 —— 本项目已经出过三回：`_top_scores` 读错键、
  检索项恒为常数、exp6 的 τ 选在空信号上。共同特征都是"不报错、指标看着合理"。

  所以校验必须**从逐条记录独立重算，再与报告里发表的数字对账**。
  期望值写在下面的 `PUBLISHED` 里，**是从 `results/*.md` 抄下来的**，
  不是从代码里取 —— 抄错、或报告与数据不一致，这里就会红。这正是要防的。

**边界（必须说清，否则"可复现"是句空话）：**
  · 能校验：**聚合是否正确**。逐条记录 → 指标 → 与报告发表的数字比。
  · 不能校验：**LLM 的生成内容**。重跑得到的是不同文本（非确定性），
    所以逐条记录是**证据的存档**，不是可重放的中间产物。
    对这类实验，"可复现"的含义只能是：
    **给定同一批逐条记录，所有发表数字都能重算出来。**
  · exp5 例外：全链路无 LLM（BM25/稠密/混合/交叉编码器），
    所以**整体重跑并逐格比对** —— 这是最强的复现证据。
  · exp1 是早期探针实验（单一 JSON、无逐条记录），**不在机器校验范围内**，如实标注。

用法:
  .venv/bin/python scripts/verify_repro.py
  .venv/bin/python scripts/verify_repro.py --no-rerun    # 跳过 exp5 重跑
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hragent import config, metrics  # noqa: E402

R = config.RESULTS
TOL = 1e-3

# ---------------------------------------------------------------------------
# 期望值：**从 results/*.md 抄写**，出处见各行注释。改报告就必须改这里。
# ---------------------------------------------------------------------------
PUBLISHED = {
    # results/exp9_ablation.md L127-132 · 六臂主结果表
    # 列：行为正确率 / 硬答率 / 过度拒答率 / tok/题 / 秒/题
    # （不是"任务成功率/澄清率" —— 曾把第三列读错，见 verify_exp9 注释）
    "exp9": {
        "B0": (0.720, 0.655, 0.000, 5409, 5.2),
        "B1": (0.707, 0.500, 0.043, 4833, 5.8),
        "B2": (0.887, 0.190, 0.033, 5729, 5.2),
        "B3": (0.873, 0.155, 0.087, 5142, 7.6),
        "B4": (0.873, 0.103, 0.109, 3570, 6.4),
        "B5": (0.873, 0.138, 0.120, 3492, 6.4),
    },
    # results/exp6_skill_match.md L106-107 · A 段（micro F1 / trap 误抽率）
    "exp6_a": {"S0 词表匹配": (0.778, 0.790), "S1 LLM 抽取": (0.777, 0.029)},
    # results/exp6_skill_match.md L177-181 · B 段
    # 列：NDCG@k / 全序 NDCG / 缺must 分位 / DI(性别)
    "exp6_b": {
        "m0_embed": (0.6841, 0.7141, 0.356, 0.935),
        "m0n_deident": (0.7144, 0.7389, 0.413, 0.873),
        "m0x_deident_name": (0.6783, 0.7277, 0.388, 0.935),
        "m1_struct": (0.9909, 0.9909, 0.505, 0.708),
        "m2_llm": (0.9940, 0.9940, 0.566, 0.760),
    },
}

_results: list[tuple[str, bool, str]] = []


def check(name: str, got, want, tol: float = TOL) -> None:
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        ok = abs(float(got) - float(want)) <= tol
    else:
        ok = got == want
    _results.append((name, ok, f"重算 {got!r} ≠ 报告 {want!r}"))


def load(name: str):
    p = R / name
    if not p.exists():
        raise SystemExit(f"❌ 缺少结果文件: {p}")
    return json.loads(p.read_text())


def jsonl(name: str) -> list[dict]:
    out = []
    for line in (R / name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_script(rel: str, mod: str):
    """按路径导入实验脚本（scripts/ 不是包，只能这样拿函数）。"""
    spec = importlib.util.spec_from_file_location(mod, config.ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- 各实验

def verify_exp2() -> None:
    """歧义策略三臂：L3（金标 clarify）与 L1（金标 answer）两组率值。"""
    d = load("exp2_ambiguity.json")
    raw = d["raw"]
    for layer in ("L3", "L1"):
        for arm, s in d["stats"][layer].items():
            rs = [r for r in raw if r["arm"] == arm and r["layer"] == layer]
            if not rs:
                check(f"exp2/{layer}/{arm}/有逐条记录", False, True)
                continue
            n = len(rs)
            got = {
                "tokens": st.mean([r["tokens"] for r in rs]),
                "latency_s": st.mean([r["latency_s"] for r in rs]),
                "trunc_rate": sum(1 for r in rs if r["n_truncated"] > 0) / n,
                "n": n,
            }
            if layer == "L3":
                got |= {"clarify_rate": sum(1 for r in rs if r["action"] == "clarify") / n,
                        "hard_rate": sum(1 for r in rs if r["action"] == "answer") / n,
                        "refuse_rate": sum(1 for r in rs if r["action"] == "refuse") / n}
            else:
                got |= {"accuracy": sum(1 for r in rs if r["action"] == "answer") / n,
                        "over_clarify_rate": sum(1 for r in rs if r["action"] == "clarify") / n,
                        "refuse_rate": sum(1 for r in rs if r["action"] == "refuse") / n}
            for k, v in got.items():
                check(f"exp2/{layer}/{arm}/{k}", round(v, 9), round(s[k], 9))


def verify_exp3() -> None:
    d = load("exp3_routing_test.json")
    rows = d["rows"]
    check("exp3/n", len(rows), d["n"])
    correct = [r["pred_exec"] == r["gold_exec"] for r in rows]
    check("exp3/executor_exact_match",
          round(sum(correct) / len(rows), 9), round(d["executor_exact_match"], 9))
    # 澄清决策：金标 = gold_behavior == 'clarify'（与 exp3 脚本同定义）
    clar = metrics.guard_report([r["pred_clar"] for r in rows],
                                [r["gold_behavior"] == "clarify" for r in rows])
    for k in ("tp", "fp", "fn", "tn", "block_recall", "fpr", "precision", "accuracy"):
        check(f"exp3/clarification/{k}", clar[k], d["clarification"][k])
    per: dict[str, list[float]] = collections.defaultdict(list)
    for r in rows:
        for e in r["gold_exec"]:
            per[e].append(1.0 if e in r["pred_exec"] else 0.0)
    for e, v in per.items():
        check(f"exp3/per_executor_recall/{e}", round(sum(v) / len(v), 9),
              round(d["per_executor_recall"][e], 9))
    check("exp3/cost/tokens_per_query",
          round(d["cost"]["total_tokens"] / len(rows), 9),
          round(d["cost"]["tokens_per_query"], 9))


def verify_exp4() -> None:
    d = load("exp4_sql_caliber_test.json")
    rows = d["rows"]
    # `n` 是**每臂**的题数，`rows` 是三臂拼起来的 —— 不是一回事
    check("exp4/每臂题数", len(rows) // len(d["arms"]), d["n"])
    for arm, agg in d["per_arm"].items():
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            check(f"exp4/{arm}/有逐条记录", False, True)
            continue
        for field, want_key in (("valid", "valid_rate"), ("ex_norm", "ex_norm"),
                                ("truncated", "truncated_rate")):
            check(f"exp4/{arm}/{want_key}",
                  round(sum(1 for r in rs if r[field]) / len(rs), 9),
                  round(agg[want_key], 9))
        check(f"exp4/{arm}/tokens_per_q",
              round(st.mean([r["tokens"] for r in rs]), 9),
              round(agg["tokens_per_q"], 9))


def verify_exp5() -> None:
    """exp5 全链路无 LLM → 整体重跑，逐格比对（最强的复现证据）。

    **唯一排除项：`latency_ms`（墙钟时间）。** 它依赖机器负载，天然不可复现 ——
    实测同一份代码两次运行差 0.2ms~74ms（BM25 0.20→0.37，Rerank 688→614）。
    把它一起比会永远红，把它悄悄跳过又等于隐瞒，所以**显式排除并打印实测值**，
    让人看到差异有多大、以及为什么它不算"不可复现"。

    **重跑会写 `results/exp5_retrieval.json`（脚本路径写死），所以比对完必须把
    原始字节还原回去。** 校验工具不该改动它校验的对象 —— 否则跑一次校验就
    把待校验的证据换掉了，第二次跑校验的已经是第一次的输出。
    """
    out = R / "exp5_retrieval.json"
    original = out.read_bytes()
    try:
        r = subprocess.run([sys.executable, "scripts/exp5_retrieval.py"],
                           cwd=config.ROOT, capture_output=True, text=True,
                           timeout=3600)
        if r.returncode != 0:
            check("exp5/重跑成功", False, f"退出码 {r.returncode}")
            print(r.stdout[-1500:], r.stderr[-1500:])
            return
        old, new = json.loads(original), json.loads(out.read_text())
        timing = []
        for arm, v in old.items():
            if not isinstance(v, dict) or arm not in new:
                continue
            for m, val in v.items():
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    continue
                if m == "latency_ms":
                    timing.append((arm, val, new[arm][m]))
                    continue
                check(f"exp5/{arm}/{m}", round(new[arm][m], 9), round(val, 9))
        print("  · 以下为墙钟耗时，**不参与比对**（机器负载相关，非确定性）：")
        for arm, o, n in timing:
            print(f"      {arm:8s} 上次 {o:8.2f}ms → 本次 {n:8.2f}ms")
    finally:
        out.write_bytes(original)     # 还原，见 docstring


def verify_exp6() -> None:
    d = load("exp6_skill_match.json")
    test = d["test"]
    check("exp6/test 记录数", len(test), 120)

    # ---- A 段：重算自 exp6_raw.jsonl（S0 的词表由 dev 划分重建，不泄题）
    m6 = load_script("scripts/exp6_skill_match.py", "exp6mod")
    import hashlib

    h = lambda t: hashlib.sha1(t.encode()).hexdigest()[:16]  # noqa: E731
    skill_rows = [json.loads(l) for l in
                  (config.EVAL / "skill_eval.jsonl").read_text(encoding="utf-8").splitlines()
                  if l.strip()]
    lex = m6.build_lexicon(skill_rows)
    rows = [r for r in skill_rows if r["split"] == "test"]
    check("exp6/A 段简历数", len(rows), 126)
    llm_recs = {r["id"]: (r.get("skills") or [])
                for r in jsonl("exp6_raw.jsonl") if r.get("kind") == "extract"}
    for name, get in (("S0 词表匹配", lambda t: m6.lexicon_extract(t, lex)),
                      ("S1 LLM 抽取", lambda t: llm_recs.get(h(t), []))):
        tp = fp = fn = 0
        traps = []
        for r in rows:
            pred = {m6._norm(x) for x in get(r["text"])}
            gold = {m6._norm(x) for x in r["gold_skills"]}
            trap = {m6._norm(x) for x in (r.get("trap_skills") or [])}
            tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
            if trap:
                traps.append(len(pred & trap) / len(trap))
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        check(f"exp6/A/{name}/micro F1", round(f1, 6), PUBLISHED["exp6_a"][name][0])
        check(f"exp6/A/{name}/trap 误抽率", round(st.mean(traps), 6),
              PUBLISHED["exp6_a"][name][1])

    # ---- B 段：重算自 exp6_skill_match.json 的 test 记录
    by: dict[str, list[dict]] = collections.defaultdict(list)
    for r in test:
        by[r["job_id"]].append(r)
    genders = [(r["sensitive"] or {}).get("gender") or "未知" for r in test]
    for key, (nd_w, full_w, mm_w, di_w) in PUBLISHED["exp6_b"].items():
        ndcgs, fulls, mm = [], [], []
        for _, rs in sorted(by.items()):
            k = min(10, len(rs))
            rels = [r["relevance"] for r in rs]
            order = sorted(range(len(rs)), key=lambda i: -rs[i][key])
            ndcgs.append(metrics.ndcg_at_k([rels[i] for i in order], k))
            fulls.append(metrics.ndcg_at_k([rels[i] for i in order], len(rs)))
            mm += m6.rank_of_missing_must([r[key] for r in rs], rels,
                                          [r["kind"] for r in rs])
        check(f"exp6/B/{key}/NDCG@k", round(st.mean(ndcgs), 6), nd_w, tol=5e-4)
        check(f"exp6/B/{key}/全序 NDCG", round(st.mean(fulls), 6), full_w, tol=5e-4)
        check(f"exp6/B/{key}/缺must 分位", round(st.mean(mm), 6), mm_w, tol=5e-4)
        check(f"exp6/B/{key}/DI(性别)",
              round(metrics.disparate_impact_at_k(
                  [r[key] for r in test], genders,
                  k=max(1, len(test) // 2))["di"], 6), di_w, tol=5e-4)

    # τ 必须落在网格**内部** —— 落在边界上说明网格选错了（这个 bug 真出过）
    grid = sorted(float(t) for t in d["dev_ndcg_by_tau"])
    check("exp6/τ 不在网格边界",
          d["tau_selected_on_dev"] not in (grid[0], grid[-1]), True)


def verify_exp7() -> None:
    d = load("exp7_guards_orig_test.json")
    check("exp7/配平（注入数 = 正常数）", d["n_faults"] == d["n_normals"], True)
    for name, s in ([("zero_knowledge_baseline", d["zero_knowledge_baseline"]),
                     ("combined", d["combined"])] + list(d["single"].items())):
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        check(f"exp7/{name}/precision", round(p, 9), round(s["precision"], 9))
        check(f"exp7/{name}/recall", round(rc, 9), round(s["recall"], 9))
        check(f"exp7/{name}/f1",
              round(2 * p * rc / (p + rc), 9) if p + rc else 0.0, round(s["f1"], 9))


def verify_exp8() -> None:
    """AURC：用 exp8 自己的信号定义与标签定义，从 exp8_raw.jsonl 重算。

    **三个分析各有一套标签，不能混用**（踩过：把分析二的 0.012/0.081 当成
    分析一的数）：
      · escalate_blind  标签 `not_blind`          → 风险 = 盲区率
      · answerable_L1   标签 `label_retrieval_hit`（仅 86 条有金标的题）
      · answerable_L2   标签 `label_citation_hit`
    """
    m8 = load_script("scripts/exp8_risk_coverage.py", "exp8mod")
    recs = jsonl("exp8_raw.jsonl")
    stored = load("exp8_risk_coverage_all.json")
    check("exp8/记录数", len(recs), stored["n"])

    ans = [r for r in recs if r["gold_chunk_ids"]]
    for key, subset, ok_fn in (
            ("escalate_blind", recs, m8.not_blind),
            ("answerable_L1", ans, m8.label_retrieval_hit),
            ("answerable_L2", ans, m8.label_citation_hit)):
        arms = stored[key]["arms"]
        ok = [ok_fn(r) for r in subset]
        base = 1.0 - sum(ok) / len(ok)
        check(f"exp8/{key}/样本数", len(subset), stored[key]["n"])
        check(f"exp8/{key}/基础风险", round(base, 6),
              round(stored[key]["base_risk"], 6))
        check(f"exp8/{key}/S0 臂 = 基础风险",
              round(arms["S0 无门控（零知识基线）"]["aurc"], 6), round(base, 6))
        per = [m8.signals(r) for r in subset]
        for name in per[0]:
            got = metrics.risk_coverage_curve([s[name] for s in per], ok)["aurc"]
            check(f"exp8/{key}/{name}/AURC", round(got, 6),
                  round(arms[name]["aurc"], 6))
    # L1 与 L2 是否真的等价 —— 报告声称等价，这里核验它
    check("exp8/L1-L2 不一致样本数",
          len([r for r in ans
               if m8.label_retrieval_hit(r) != m8.label_citation_hit(r)]),
          len(stored["L1_L2_disagreement"]))
    # 信号方向表：盲区题 vs 可答题的证据条数
    blind = [r for r in recs if m8.label_blind(r)]
    ans0 = [r for r in recs if not m8.label_blind(r)]
    sd = stored["signal_direction"]
    check("exp8/盲区题 n_evidence 均值",
          round(st.mean([r["n_evidence"] for r in blind]), 6),
          round(sd["blind_n_evidence_mean"], 6))
    check("exp8/可答题 n_evidence 均值",
          round(st.mean([r["n_evidence"] for r in ans0]), 6),
          round(sd["answerable_n_evidence_mean"], 6))
    check("exp8/盲区题自评置信度均值",
          round(st.mean([r["confidence"] for r in blind]), 6),
          round(sd["blind_conf_mean"], 6))
    check("exp8/n_evidence==0 的样本数",
          sum(1 for r in recs if r["n_evidence"] == 0), sd["n_zero_evidence"])


def verify_exp9() -> None:
    """六臂主结果表：从 exp9_raw.jsonl 的「·终版」臂重算。

    **两个易错点，都踩过：**
    1. `exp9_ablation_test.json` 只有 **B5 一个臂**（同臂复跑，用于噪声底），
       六臂表在 `exp9_raw.jsonl` 里。别拿前者当全表。
    2. 列名是 行为正确率 / **硬答率** / **过度拒答率**。硬答率的分母是
       "不该答的题"，过度拒答率的分母是"该答的题" —— 都不是 150。
    """
    # 与 scripts/exp9_ablation.py 的金标映射一致（**转抄而非引用**：
    # 引用脚本里的常量会让"报告 vs 脚本"的一致性无法被检验）
    expected_action = {"answer": {"answer"}, "correct": {"answer"},
                       "clarify": {"clarify"}, "refuse": {"refuse"},
                       "no_answer": {"refuse"}, "decline": {"refuse"},
                       "escalate": {"escalate"}}
    must_not_answer = {"refuse", "no_answer", "decline", "escalate", "clarify"}
    should_answer = {"answer", "correct"}

    arms: dict[str, list[dict]] = collections.defaultdict(list)
    for r in jsonl("exp9_raw.jsonl"):
        arms[r["arm"]].append(r)
    for tag, (acc_w, hard_w, over_w, tok_w, lat_w) in PUBLISHED["exp9"].items():
        cand = [a for a in arms if a.startswith(tag + " ") and a.endswith("·终版")]
        if len(cand) != 1:
            check(f"exp9/{tag}/唯一定位到「·终版」臂", False, cand)
            continue
        rs = arms[cand[0]]
        n = len(rs)
        acc = sum(1 for r in rs if r["action"] in expected_action[r["expected"]]) / n
        mn = [r for r in rs if r["expected"] in must_not_answer]
        hard = sum(1 for r in mn if r["action"] == "answer") / len(mn)
        sa = [r for r in rs if r["expected"] in should_answer]
        over = sum(1 for r in sa if r["action"] != "answer") / len(sa)
        check(f"exp9/{tag}/行为正确率", round(acc, 6), acc_w)
        check(f"exp9/{tag}/硬答率", round(hard, 6), hard_w)
        check(f"exp9/{tag}/过度拒答率", round(over, 6), over_w)
        check(f"exp9/{tag}/tok/题", round(st.mean([r["tokens"] for r in rs]), 3),
              tok_w, tol=0.5)
        check(f"exp9/{tag}/秒/题", round(st.mean([r["latency_s"] for r in rs]), 3),
              lat_w, tol=0.05)


# ---------------------------------------------------------------- 闸门

def run_gates() -> None:
    for name, cmd in (("selftest_metrics", ["scripts/selftest_metrics.py"]),
                      ("selftest_guards", ["scripts/selftest_guards.py"]),
                      ("smoke_graph", ["scripts/smoke_graph.py"]),
                      ("freeze_eval --verify",
                       ["scripts/freeze_eval.py", "--verify"])):
        r = subprocess.run([sys.executable] + cmd, cwd=config.ROOT,
                           capture_output=True, text=True, timeout=1800)
        check(f"gate/{name}", r.returncode == 0, True)


def _dump(since: int) -> None:
    for name, ok, msg in _results[since:]:
        print(f"  {'✅' if ok else '❌'} {name}")
        if not ok:
            print(f"       {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rerun", action="store_true",
                    help="跳过 exp5 的整体重跑（它要几分钟）")
    args = ap.parse_args()

    print("=" * 78)
    print("复现校验：从逐条记录独立重算，与 results/*.md 里发表的数字对账")
    print("=" * 78)

    print("\n【确定性闸门】")
    before = len(_results)
    run_gates()
    _dump(before)

    for label, fn in (("exp2 歧义策略", verify_exp2),
                      ("exp3 分层路由", verify_exp3),
                      ("exp4 SQL 口径", verify_exp4),
                      ("exp6 技能抽取与匹配", verify_exp6),
                      ("exp7 风控闸门", verify_exp7),
                      ("exp8 风险-覆盖曲线", verify_exp8),
                      ("exp9 端到端消融", verify_exp9)):
        print(f"\n【{label}】")
        before = len(_results)
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            _results.append((f"{label}/校验本身报错", False, f"{type(e).__name__}: {e}"))
        _dump(before)

    if not args.no_rerun:
        print("\n【exp5 整体重跑（全链路无 LLM，可逐格复现）】")
        before = len(_results)
        verify_exp5()
        _dump(before)

    bad = [x for x in _results if not x[1]]
    print("\n" + "=" * 78)
    print(f"共 {len(_results)} 项，通过 {len(_results) - len(bad)}，失败 {len(bad)}")
    if bad:
        print("\n❌ 以下数字无法复现：")
        for name, _, msg in bad:
            print(f"   {name}: {msg}")
        raise SystemExit(1)
    print("✅ 全部数字可从逐条记录复现")
    print("\n范围声明（这是「可复现」的边界，不是免责）：")
    print("  · 本校验证明**聚合正确**，不证明**LLM 生成可重放** —— 后者本质不可复现，")
    print("    逐条记录是证据存档。详见本文件顶部说明。")
    print("  · exp1 是早期探针实验（单一 JSON、无逐条记录），不在机器校验范围内。")
    print("  · bootstrap/置换检验的**区间与 p 值**含随机数，本校验只对点估计；")
    print("    区间与 p 的确定性由 selftest_metrics / selftest_guards 覆盖。")


if __name__ == "__main__":
    main()
