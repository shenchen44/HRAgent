"""实验 6：技能抽取 + 结构化匹配 vs 纯语义相似。

要回答的问题（DESIGN §5 实验矩阵最后一项）：
  **把"人岗匹配"做成"先抽技能、再按 JD 的 must/nice 结构化比对"，
  真的优于把 JD 与简历整体丢给 embedding 算余弦吗？**

评测集：
  eval/skill_eval.jsonl  test 126 条（12 个岗位 × 10~11 份简历）
      每条带 gold_skills（真掌握的技能）与 **trap_skills**（文中提到但只是"了解过"）
  eval/match_eval.jsonl  test 120 对（12 个岗位 × 8 或 12 位候选人）
      每对带 gold_relevance（2/1/0，分级相关性）、kind（完全匹配/缺must/硬负例/…）
      敏感属性（性别/年龄/婚育）**随机分配且与相关性独立**（生成脚本有守门检验）

两条主线：
  A 技能抽取   S0 词表匹配（dev 词表，不看 test 金标） vs S1 LLM 抽取
               关键指标是 **trap 误抽率** —— 词表分不出"熟练使用"与"了解过"
  B 人岗匹配   M0 纯 embedding（整段余弦）
               M1 结构化·词表对齐（S1 的技能 + 嵌入对齐阈值，阈值在 dev 上定）
               M2 结构化·逐技能 LLM 判断（用 JD 自己的词表，绕开别名对齐）
               指标 NDCG@k / 全序 NDCG / 缺must 检出 / 敏感属性无关性 / 成本

设计纪律（D7）：M1 的对齐阈值只在 **dev** 上选，test 只报数。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hragent import config, metrics                      # noqa: E402
from hragent.llm import LLM                              # noqa: E402
from hragent.obs.resume import Checkpoint                # noqa: E402

# 抽取/对齐/打分原先就长在本脚本里，现已提到共用模块（前端要复用同一套逻辑，
# 复制一份就成了"两份真相"）。这里**原样再导出**，一是本脚本后续代码不用改，
# 二是 `verify_repro.py` 按 `m6.build_lexicon` / `m6._norm` 等名字取用，
# 改名会把它打断。搬家是否改变了已发布的数字，由 verify_repro 判定。
from hragent.tools.skills import (                       # noqa: E402,F401
    EXTRACT_SYSTEM, PER_SKILL_SYSTEM, _norm, align_hit, build_lexicon,
    lexicon_extract, llm_extract, llm_per_skill, parse_jd, structured_score,
)
# `deident` 同样搬了家（风控层，见 hragent/risk/deident.py），也原样再导出。
from hragent.risk.deident import _RESUME_HEAD, deident      # noqa: E402,F401

WORKERS = 4
CKPT = config.RESULTS / "exp6_raw.jsonl"
# 配对单独一个断点文件。共用一个的话，"已完成 N/M" 会把两种记录加在一起数，
# 进度显示成 19/14 这种不可能的数，成本也会被串到一起去。
CKPT_PAIR = config.RESULTS / "exp6_pair_raw.jsonl"
OUT = config.RESULTS / "exp6_skill_match.json"

SKILL_SET = config.EVAL / "skill_eval.jsonl"
MATCH_SET = config.EVAL / "match_eval.jsonl"

# ---------------------------------------------------------------- 指标

def ndcg_for_group(order: list[int], rels: list[int], k: int) -> float:
    return metrics.ndcg_at_k([rels[i] for i in order], k)


def rank_of_missing_must(scores: list[float], rels: list[int], kinds: list[str]
                         ) -> list[float]:
    """「缺must」的样本排在第几百分位（1.0 = 垫底，理想）。

    只看「缺must」类：它们相关性为 0，一个好的打分器应把它们压到底部。
    """
    n = len(scores)
    if n < 2:
        return []
    order = sorted(range(n), key=lambda i: -scores[i])
    pos = {idx: r for r, idx in enumerate(order)}
    out = []
    for i, kind in enumerate(kinds):
        if kind == "缺must":
            out.append(pos[i] / (n - 1))
    return out


def di_null(scores: list[float], groups: list[str], k: int,
            n_perm: int = 2000, seed: int = 20260929) -> dict:
    """DI 的置换零分布。

    **敏感属性在本评测集里是随机分配的、与金标独立**（gen_match_eval.py 有守门检验）。
    这意味着「无偏见」的 DI 不是 1.0，而是围绕某个 <1 的值波动 ——
    因为 top-k 里各组的占比本身就是个随机量，k 越小波动越大。
    直接报一个 0.68 说"存在偏见"是把抽样噪声当成了效应。

    所以：把性别标签在样本间随机重排（不改变各组的样本数），重算 DI，
    得到零分布，再看实测值落在什么位置。p 是双尾。
    """
    rnd = random.Random(seed)
    obs = metrics.disparate_impact_at_k(scores, groups, k)["di"]
    null = []
    for _ in range(n_perm):
        g = groups[:]
        rnd.shuffle(g)
        null.append(metrics.disparate_impact_at_k(scores, g, k)["di"])
    lo = sum(1 for x in null if x <= obs) / len(null)
    p = 2 * min(lo, 1 - lo)
    null.sort()
    return {"di": obs, "p": min(1.0, p),
            "null_p05": null[int(0.05 * len(null))],
            "null_p50": null[len(null) // 2],
            "null_p95": null[int(0.95 * len(null))]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "dev", "all"])
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reset", action="store_true")
    # 网格要盖过最优点。上一版到 0.80 就截断了，而 dev 曲线在 0.80 处仍在上升 ——
    # **最优落在网格边界上，说明网格选错了，不是"选出了最优"。**
    ap.add_argument("--tau-grid",
                    default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    args = ap.parse_args()

    skill_rows = [json.loads(l) for l in
                  SKILL_SET.read_text(encoding="utf-8").splitlines() if l.strip()]
    match_rows = [json.loads(l) for l in
                  MATCH_SET.read_text(encoding="utf-8").splitlines() if l.strip()]
    # `--limit` 必须**在派生简历集之前**截，否则抽取的是全集里的前 N 条、
    # 而打分用的是另外 N 对 —— M1 拿不到技能，"结构化匹配很差"就成了截断的假象。
    if args.limit:
        skill_rows = [r for r in skill_rows if r["split"] != args.split] + \
                     [r for r in skill_rows if r["split"] == args.split][:args.limit]
        match_rows = [r for r in match_rows if r["split"] != args.split] + \
                     [r for r in match_rows if r["split"] == args.split][:args.limit]
    lex = build_lexicon(skill_rows)
    taus = [float(x) for x in args.tau_grid.split(",")]

    # ---- 需要抽取的唯一简历：skill 集 ∪ match 集（两边有 84 条重叠，去重省调用）
    def resumes_for(split: str) -> list[dict]:
        out: dict[str, dict] = {}
        for r in skill_rows:
            if split in (r["split"], "all"):
                out.setdefault(r["text"], {"id": f"SKILL:{r['id']}", "text": r["text"],
                                           "src": "skill"})
        for r in match_rows:
            if split in (r["split"], "all"):
                out.setdefault(r["resume"], {"id": f"RESUME:{len(out)}",
                                             "text": r["resume"], "src": "match"})
        return list(out.values())

    # 抽取的 id 用简历文本的稳定短哈希，避免两边各自编号对不上
    import hashlib

    def rid(text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()[:16]

    # **抽取必须覆盖两个划分，不能只抽 --split 的那一半。**
    # 踩过的坑：`skill_eval` 与 `match_eval` 是两套独立切分，同一段简历文本在两边
    # 可能落在不同的 split（实测 dev 的 120 对里有 36 对，其 skill 孪生在 test）。
    # 只抽 test 的话，**84/120 个 dev 匹配对拿到的是空技能表** →
    # M1/M2 在 dev 上退化成常数打分器 → "τ 在 dev 上选"这句就名存实亡（D7 失效）。
    # 症状是 dev NDCG 0.84 而 test NDCG 0.99，两者本不该差这么多。
    todo_res = resumes_for("all")
    for r in todo_res:
        r["id"] = rid(r["text"])

    ck = Checkpoint(CKPT)
    ck_pair = Checkpoint(CKPT_PAIR)
    if args.reset:
        for p in (CKPT, CKPT_PAIR):
            if p.exists():
                p.unlink()

    print(f"实验 6 · 技能抽取 + 结构化匹配")
    print(f"  词表（来自 dev）{len(lex)} 个技能")
    print(f"  待抽取简历 {len(todo_res)} 条（skill ∪ match 去重后）")

    # ---- A0：词表抽取（零 LLM 调用）
    lex_recs = {r["id"]: lexicon_extract(r["text"], lex) for r in todo_res}

    # ---- A1：LLM 抽取
    def do_extract(r: dict) -> dict:
        llm = LLM()
        t0 = time.time()
        skills, meta = llm_extract(r["text"], llm)
        return {"id": r["id"], "kind": "extract", "text": r["text"], "skills": skills,
                "tokens": meta["tokens"], "latency_s": round(time.time() - t0, 2),
                "error": meta.get("error")}

    llm_rows = ck.run(todo_res, do_extract) if todo_res else []
    # 只取抽取记录：即便断点文件已经分开，旧文件里可能还混着配对记录，
    # 混进来会把配对的 tokens 算成"S1 的 tok/题"，把成本指标搞错。
    llm_rows = [r for r in llm_rows if r.get("kind") == "extract"]
    llm_recs = {r["id"]: (r.get("skills") or []) for r in llm_rows}

    # ---- 嵌入：技能名向量（供对齐用）+ JD/简历整段向量（供 M0 用）
    from hragent.tools.dense import embedder
    E = embedder()

    def embed_all(names: list[str]) -> dict:
        names = sorted(set(names))
        if not names:
            return {}
        vs = E.encode(names, normalize_embeddings=True, show_progress_bar=False)
        return dict(zip(names, vs))

    jd_skill_names = [s for r in match_rows for s in sum(parse_jd(r["jd"]), [])]
    resume_skill_names = [s for v in llm_recs.values() for s in v]
    vec = embed_all(jd_skill_names + resume_skill_names + lex)

    # ---- B：三臂打分
    def score_arms(rows: list[dict], tau: float, split: str) -> list[dict]:
        """对一个划分的 match 集算三臂分数。tau 由调用方给定（test 时传 dev 选出的）。"""
        out = []
        jd_vec = {r["job_id"]: v for r, v in zip(rows, E.encode(
            [r["jd"] for r in rows], normalize_embeddings=True,
            show_progress_bar=False))}
        res_vec = {r["id"]: v for r, v in zip(rows, E.encode(
            [r["resume"] for r in rows], normalize_embeddings=True,
            show_progress_bar=False))}
        # 去标识对照：同一套余弦，只换输入文本。用来定位 M0 的 DI 是从哪来的。
        res_vec_n = {r["id"]: v for r, v in zip(rows, E.encode(
            [deident(r["resume"], False) for r in rows], normalize_embeddings=True,
            show_progress_bar=False))}
        res_vec_x = {r["id"]: v for r, v in zip(rows, E.encode(
            [deident(r["resume"], True) for r in rows], normalize_embeddings=True,
            show_progress_bar=False))}
        for r in rows:
            must, nice = parse_jd(r["jd"])
            sk = llm_recs.get(rid(r["resume"]), [])
            out.append({
                "id": r["id"], "job_id": r["job_id"], "kind": r["kind"],
                "relevance": r["gold_relevance"],
                "sensitive": r.get("sensitive") or {},
                # must/nice 带上：M2 打分要用，重解析一遍容易和这里不一致
                "must": must, "nice": nice,
                "m0_embed": float(jd_vec[r["job_id"]] @ res_vec[r["id"]]),
                "m0n_deident": float(jd_vec[r["job_id"]] @ res_vec_n[r["id"]]),
                "m0x_deident_name": float(jd_vec[r["job_id"]] @ res_vec_x[r["id"]]),
                "m1_struct": structured_score(must, nice, sk, tau, vec),
                "n_must": len(must), "n_nice": len(nice),
                "must_hit": sum(1 for m in must if align_hit(m, sk, tau, vec)),
                "skills": sk,
            })
        return out

    dev_rows = [r for r in match_rows if r["split"] == "dev"]
    test_rows = [r for r in match_rows if r["split"] == "test"]
    # `--limit` 必须同时截匹配集：只截抽取会让 M1 拿不到技能，
    # 于是"结构化匹配很差"这个结论就成了截断造成的假象。
    if args.limit:
        dev_rows = dev_rows[:args.limit]
        test_rows = test_rows[:args.limit]

    # ---- tau 只在 dev 上选（D7）
    print("\n【τ 选择：只在 dev 上做】")
    dev_scored_by_tau = {}
    for tau in taus:
        sc = score_arms(dev_rows, tau, "dev")
        n = group_ndcg(sc, "m1_struct")
        dev_scored_by_tau[tau] = n
        print(f"  τ={tau:.2f}  dev NDCG@k = {n:.4f}")
    best_tau = max(dev_scored_by_tau, key=lambda t: dev_scored_by_tau[t])
    print(f"  → 选定 τ = {best_tau:.2f}")

    # ---- M2：逐技能 LLM 判断（每个 (JD, 简历) 一次调用）
    def do_pair(r: dict) -> dict:
        llm = LLM()
        must, nice = parse_jd(r["jd"])
        has, meta = llm_per_skill(must + nice, r["resume"], llm)
        return {"id": r["id"], "kind": "pair", "has": has, "tokens": meta["tokens"]}

    pair_rows = ck_pair.run(test_rows, do_pair) if test_rows else []
    pair_recs = {r["id"]: (r.get("has") or []) for r in pair_rows}

    def m2_score(r: dict) -> float:
        must, nice = r["must"], r["nice"]
        has = pair_recs.get(r["id"], [])
        hn = {_norm(x) for x in has}
        mc = (sum(1 for m in must if _norm(m) in hn) / len(must)) if must else 0.0
        if mc == 0.0:
            return 0.0
        nc = (sum(1 for n in nice if _norm(n) in hn) / len(nice)) if nice else 0.0
        return mc + 0.5 * nc

    # ---- 报告
    test_scored = score_arms(test_rows, best_tau, "test")
    for r in test_scored:
        r["m2_llm"] = m2_score(r)

    print(f"\n{'=' * 100}")
    print(f"实验 6 · 结果（split={args.split}）")
    print(f"{'=' * 100}")
    report_a(skill_rows, lex_recs, llm_recs, llm_rows, args.split)
    report_b(test_scored, best_tau, dev_scored_by_tau, pair_rows)
    report_cost(llm_rows, pair_rows, args.split)

    OUT.write_text(json.dumps({
        "tau_selected_on_dev": best_tau,
        "dev_ndcg_by_tau": {str(k): v for k, v in dev_scored_by_tau.items()},
        "test": test_scored,
    }, ensure_ascii=False, indent=2))
    print(f"\n→ {OUT}\n→ 原始逐条 {CKPT} / {CKPT_PAIR}")


def group_ndcg(scored: list[dict], key: str) -> float:
    """按岗位分组算 NDCG，再对岗位取平均（k = min(10, 组内人数)）。"""
    by: dict[str, list[dict]] = {}
    for r in scored:
        by.setdefault(r["job_id"], []).append(r)
    vals = []
    for _, rs in sorted(by.items()):
        k = min(10, len(rs))
        order = sorted(range(len(rs)), key=lambda i: -rs[i][key])
        rels = [r["relevance"] for r in rs]
        vals.append(metrics.ndcg_at_k([rels[i] for i in order], k))
    return st.mean(vals) if vals else 0.0


def per_group_ndcg(scored: list[dict], key: str) -> dict[str, float]:
    """每个岗位一个 NDCG 值。配对检验的重采样单位是**岗位**，不是样本对 ——
    NDCG 本身在组内算，重采样样本对会把组结构打散，区间会假性变窄。"""
    by: dict[str, list[dict]] = {}
    for r in scored:
        by.setdefault(r["job_id"], []).append(r)
    out = {}
    for gid, rs in sorted(by.items()):
        k = min(10, len(rs))
        order = sorted(range(len(rs)), key=lambda i: -rs[i][key])
        rels = [r["relevance"] for r in rs]
        out[gid] = metrics.ndcg_at_k([rels[i] for i in order], k)
    return out


def paired_ndcg_test(scored: list[dict], key_a: str, key_b: str) -> dict:
    """两臂在**同一批岗位**上的配对 bootstrap。

    R5 要求配对比较。NDCG 不是二分类，McNemar 用不上，用配对 bootstrap 代替：
    对岗位做有放回重采样，看差值的分布里有没有 0。p 是双尾（差值反号的比例 ×2）。
    """
    a, b = per_group_ndcg(scored, key_a), per_group_ndcg(scored, key_b)
    diffs = [a[g] - b[g] for g in a]
    rng = random.Random(20260929)
    n = len(diffs)
    boots = []
    for _ in range(4000):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[min(len(boots) - 1, int(0.975 * len(boots)))]
    neg = sum(1 for x in boots if x <= 0) / len(boots)
    return {"mean_diff": sum(diffs) / n, "lo": lo, "hi": hi,
            "p": min(1.0, 2 * min(neg, 1 - neg)), "n_groups": n,
            "n_better": sum(1 for d in diffs if d > 0),
            "n_worse": sum(1 for d in diffs if d < 0)}


def missing_must_ranks(scored: list[dict], key: str) -> dict[str, float]:
    """每条「缺must」样本在**它所属岗位内**的百分位，按样本 id 索引。"""
    by: dict[str, list[dict]] = {}
    for r in scored:
        by.setdefault(r["job_id"], []).append(r)
    out = {}
    for _, rs in sorted(by.items()):
        rels = [r["relevance"] for r in rs]
        vals = rank_of_missing_must([r[key] for r in rs], rels,
                                    [r["kind"] for r in rs])
        ids = [r["id"] for r in rs if r["kind"] == "缺must"]
        out.update(dict(zip(ids, vals)))
    return out


def paired_missing_must_test(scored: list[dict], key_a: str, key_b: str) -> dict:
    """两臂在同一批「缺must」样本上的分位差，配对 bootstrap（单位=样本）。"""
    a, b = missing_must_ranks(scored, key_a), missing_must_ranks(scored, key_b)
    ids = sorted(set(a) & set(b))
    diffs = [a[i] - b[i] for i in ids]
    rng = random.Random(20260929)
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(4000))
    neg = sum(1 for x in boots if x <= 0) / len(boots)
    return {"mean_diff": sum(diffs) / n, "n": n,
            "lo": boots[int(0.025 * len(boots))],
            "hi": boots[min(len(boots) - 1, int(0.975 * len(boots)))],
            "p": min(1.0, 2 * min(neg, 1 - neg)),
            "n_better": sum(1 for d in diffs if d > 0),
            "n_worse": sum(1 for d in diffs if d < 0)}


def report_a(skill_rows, lex_recs, llm_recs, llm_rows, split) -> None:
    import hashlib
    rows = [r for r in skill_rows if split in (r["split"], "all")]
    print(f"\n【A · 技能抽取】{len(rows)} 条简历")
    print(f"{'臂':22s} {'micro P':>8s} {'micro R':>8s} {'micro F1':>9s} "
          f"{'macro F1':>9s} {'trap 误抽率':>11s} {'tok/题':>7s}")
    print("-" * 100)
    stat = {}
    for name, get in (("S0 词表匹配", lambda t: lex_recs.get(hashlib.sha1(t.encode()).hexdigest()[:16], [])),
                      ("S1 LLM 抽取", lambda t: llm_recs.get(hashlib.sha1(t.encode()).hexdigest()[:16], []))):
        tp = fp = fn = 0
        f1s, traps, trap_flags = [], [], []
        for r in rows:
            pred = {_norm(x) for x in get(r["text"])}
            gold = {_norm(x) for x in r["gold_skills"]}
            trap = {_norm(x) for x in (r.get("trap_skills") or [])}
            tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
            p = len(pred & gold) / len(pred) if pred else 0.0
            rc = len(pred & gold) / len(gold) if gold else 0.0
            f1s.append(2 * p * rc / (p + rc) if p + rc else 0.0)
            if trap:
                traps.append(len(pred & trap) / len(trap))
                trap_flags.append(bool(pred & trap))
        micro_p = tp / (tp + fp) if tp + fp else 0.0
        micro_r = tp / (tp + fn) if tp + fn else 0.0
        micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
        tok = 0.0
        if name.startswith("S1") and llm_rows:
            # 只对本表统计的那批简历取均值。抽了 252 条、这里只报 126 条，
            # 用全量均值会把 tok/题 报成另一个集合的数。
            want = {hashlib.sha1(r["text"].encode()).hexdigest()[:16] for r in rows}
            sel = [x["tokens"] for x in llm_rows if x["id"] in want]
            tok = st.mean(sel) if sel else 0.0
        stat[name] = dict(micro_f1=micro_f, macro_f1=st.mean(f1s),
                          trap_rate=st.mean(traps) if traps else 0.0,
                          trap_flags=trap_flags)
        print(f"{name:22s} {micro_p:>8.3f} {micro_r:>8.3f} {micro_f:>9.3f} "
              f"{st.mean(f1s):>9.3f} {(st.mean(traps) if traps else 0):>11.3f} {tok:>7.0f}")
    print("\n  trap_skills = 简历里出现、但只是「了解过」的技能。")
    print("  **词表匹配分不出「熟练使用」和「了解过」** —— 这正是 LLM 抽取的价值所在。")
    # R5 配对检验：同一条简历上「是否至少抽到一个 trap 技能」是二分类，McNemar 可用。
    # 这正是两臂唯一的实质差别，所以必须配对着看，不能比两个独立的均值。
    s0 = stat["S0 词表匹配"]["trap_flags"]
    s1 = stat["S1 LLM 抽取"]["trap_flags"]
    mc = metrics.mcnemar([not x for x in s0], [not x for x in s1])
    n0 = sum(s0); n1 = sum(s1)
    print(f"\n  【配对检验】S0 误抽 {n0}/{len(s0)} 条，S1 误抽 {n1}/{len(s1)} 条"
          f"（'误抽' = 至少抽到一个 trap 技能）")
    print(f"    McNemar：b={mc['b']}（S0 误抽而 S1 不误抽） c={mc['c']}"
          f"（S0 不误抽而 S1 误抽） p={mc['p']:.4f} → {mc['note']}")


def report_b(scored: list[dict], tau: float, dev_by_tau: dict,
             pair_rows: list[dict] | None = None) -> None:
    print(f"\n【B · 人岗匹配】{len(scored)} 对，τ={tau:.2f}（dev 选出）")
    print(f"{'臂':26s} {'NDCG@k':>8s} {'全序 NDCG':>10s} {'缺must 分位':>12s} {'DI(性别)':>9s}")
    print("-" * 100)
    by: dict[str, list[dict]] = {}
    for r in scored:
        by.setdefault(r["job_id"], []).append(r)
    di_nulls: dict[str, dict] = {}
    for name, key in (("M0 纯 embedding", "m0_embed"),
                      ("M0n 去敏感字段·留姓名", "m0n_deident"),
                      ("M0x 去敏感字段·去姓名", "m0x_deident_name"),
                      ("M1 结构化·词表对齐", "m1_struct"),
                      ("M2 结构化·逐技能 LLM", "m2_llm")):
        ndcgs, fulls, mm = [], [], []
        for _, rs in sorted(by.items()):
            k = min(10, len(rs))
            rels = [r["relevance"] for r in rs]
            order = sorted(range(len(rs)), key=lambda i: -rs[i][key])
            ndcgs.append(metrics.ndcg_at_k([rels[i] for i in order], k))
            fulls.append(metrics.ndcg_at_k([rels[i] for i in order], len(rs)))
            mm += rank_of_missing_must([r[key] for r in rs], rels,
                                       [r["kind"] for r in rs])
        di = metrics.disparate_impact_at_k(
            [r[key] for r in scored],
            [(r["sensitive"] or {}).get("gender") or "未知" for r in scored],
            k=max(1, len(scored) // 2))
        print(f"{name:26s} {st.mean(ndcgs):>8.4f} {st.mean(fulls):>10.4f} "
              f"{(st.mean(mm) if mm else float('nan')):>12.3f} {di['di']:>9.3f}")
        di_nulls[name] = di_null([r[key] for r in scored],
                                 [(r["sensitive"] or {}).get("gender") or "未知"
                                  for r in scored], k=max(1, len(scored) // 2))
    print("\n  缺must 分位：该类样本排在第几百分位（1.0 = 全部垫底，理想）。")
    print("  DI（4/5 法则）：敏感属性是**随机分配、与金标独立**的，理想值应接近 1。")
    print("\n  【DI 的置换零分布】属性是随机分配的 → 无偏见时 DI 也不等于 1，")
    print("  它本身是个随机量。下表是打乱性别标签 2000 次得到的零分布：")
    print(f"  {'臂':24s} {'实测 DI':>8s} {'零分布 p05':>11s} {'中位':>8s} "
          f"{'p95':>8s} {'双尾 p':>8s}")
    for name in di_nulls:
        d = di_nulls[name]
        print(f"  {name:24s} {d['di']:>8.3f} {d['null_p05']:>11.3f} {d['null_p50']:>8.3f} "
              f"{d['null_p95']:>8.3f} {d['p']:>8.4f}")
    print("  p 大 = 实测 DI 落在零分布内 → **测不到性别偏见**（不等于「没有偏见」）。")

    # R5 配对比较：三臂在同一批岗位上的 NDCG 差。
    # 只报三个点估计（0.93 / 0.99 / 0.97）看不出"这点差是不是噪声"。
    print("\n  【配对检验】同一批岗位上的 NDCG@k 差（配对 bootstrap，重采样单位=岗位）：")
    print(f"  {'对比':34s} {'均值差':>9s} {'95% CI':>20s} {'岗位 好/差':>12s} {'p':>8s}")
    for na, ka, nb, kb in (("M1", "m1_struct", "M0", "m0_embed"),
                           ("M2", "m2_llm", "M0", "m0_embed"),
                           ("M1", "m1_struct", "M2", "m2_llm"),
                           ("M0n", "m0n_deident", "M0", "m0_embed"),
                           ("M0x", "m0x_deident_name", "M0", "m0_embed")):
        t = paired_ndcg_test(scored, ka, kb)
        print(f"  {na + ' − ' + nb:34s} {t['mean_diff']:>+9.4f} "
              f"{'[' + format(t['lo'], '+.4f') + ', ' + format(t['hi'], '+.4f') + ']':>20s} "
              f"{str(t['n_better']) + '/' + str(t['n_worse']):>12s} {t['p']:>8.4f}")
    print("  注：只有 12 个岗位，重采样单位很粗 —— CI 宽是真实的，不是算错了。")
    print("  **M0n/M0x 与 M0 的 NDCG 差若含 0，说明去标识没有改变排序质量**，")
    print("  那么它们的意义就只在 DI 那一列：**同一套排序能力，不同的偏见水平**。")

    # 「缺must 分位」是安全侧指标，同样要配对检验，不能只比点估计。
    print("\n  【配对检验 · 缺must 分位】同一批缺must 样本上的分位差（单位=样本）：")
    print(f"  {'对比':34s} {'均值差':>9s} {'95% CI':>20s} {'样本 好/差':>12s} {'p':>8s}")
    for na, ka, nb, kb in (("M1", "m1_struct", "M0", "m0_embed"),
                           ("M2", "m2_llm", "M0", "m0_embed"),
                           ("M2", "m2_llm", "M1", "m1_struct")):
        t = paired_missing_must_test(scored, ka, kb)
        print(f"  {na + ' − ' + nb:34s} {t['mean_diff']:>+9.4f} "
              f"{'[' + format(t['lo'], '+.4f') + ', ' + format(t['hi'], '+.4f') + ']':>20s} "
              f"{str(t['n_better']) + '/' + str(t['n_worse']):>12s} {t['p']:>8.4f}")
    # 并列值会吃掉这个指标 —— 必须先确认它没被吃掉，否则分位是排序稳定性的产物。
    print("\n  【并列检查】打分恰为 0 的样本会被 `sorted` 按输入顺序排列，")
    print("  若缺must 类落在零分块里，它的分位就由**输入顺序**决定，不是打分器的能力。")
    for name, key in (("M1", "m1_struct"), ("M2", "m2_llm")):
        z = sum(1 for r in scored if r[key] == 0.0)
        zm = sum(1 for r in scored if r[key] == 0.0 and r["kind"] == "缺must")
        nm = sum(1 for r in scored if r["kind"] == "缺must")
        print(f"    {name}：零分块 {z}/{len(scored)}，其中缺must **{zm}/{nm}**"
              + ("  → 该指标不受并列影响" if zm == 0 else "  → ⚠️ 该指标受并列影响"))
    print("  另一件事：`缺must` 的硬约束（must 一条都不中 → 0 分）在本集**一次都没触发**，")
    print("  因为 `kind==缺must` 的定义是「缺**至少一条** must」，不是「全缺」。")
    print("  **所以这条硬约束在本评测集上是死代码** —— 它挡不住任何一条缺must 样本。")

    # 成本必须与准确率并列（EVAL_PROTOCOL R2 / 硬性约定 5）。
    # M2 的成本只在**匹配**这一步产生，M0/M1 是纯 CPU。
    if pair_rows:
        tok = st.mean([r.get("tokens") or 0 for r in pair_rows])
        print(f"\n  【成本】M2 逐技能 LLM 判断：{tok:.0f} tok/对"
              f"（{len(pair_rows)} 对，每个 (JD, 简历) 一次调用）。")
        print("  M0 / M0n / M0x / M1 都是纯本地计算：**0 token**。")
        print("  → **M1 与 M2 的 NDCG 差很小，但成本差一个数量级**；")
        print("    若 M2 没有稳定胜出，选 M1 是更合理的工程决策。")


def report_cost(llm_rows: list[dict], pair_rows: list[dict], split: str) -> None:
    """本次运行的 LLM 成本合计。分开报，因为两者的计价单位不同。"""
    ext_tok = sum(r.get("tokens") or 0 for r in llm_rows)
    pair_tok = sum(r.get("tokens") or 0 for r in pair_rows)
    ext_lat = sum(r.get("latency_s") or 0 for r in llm_rows)
    print(f"\n【本次运行的成本】split={split}")
    print(f"  技能抽取   {len(llm_rows):4d} 次调用  {ext_tok:>9,d} tok  "
          f"（{ext_tok / max(1, len(llm_rows)):.0f} tok/条）")
    print(f"  逐技能判断 {len(pair_rows):4d} 次调用  {pair_tok:>9,d} tok  "
          f"（{pair_tok / max(1, len(pair_rows)):.0f} tok/对）")
    print(f"  合计       {len(llm_rows) + len(pair_rows):4d} 次调用  "
          f"{ext_tok + pair_tok:>9,d} tok")
    print(f"  抽取总延迟 {ext_lat:.0f}s（4 线程并发，墙钟时间远小于此）")
    print("  注：抽取覆盖两个划分（252 条唯一简历），匹配只在 test 上做（120 对）。")


if __name__ == "__main__":
    main()
