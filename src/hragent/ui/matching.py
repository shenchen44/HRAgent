"""人岗匹配流水线：召回（M1）→ 可选精排（M2）。

**两级的取舍直接来自 exp6 的实测结论，不是拍脑袋的架构美感：**

  | 臂 | NDCG@k | tok/对 |
  |---|---|---|
  | M1 结构化·词表对齐 | 0.9909 | 0 |
  | M2 结构化·逐技能 LLM | 0.9940 | 88 |

  配对检验 M1 − M2：均值差 −0.0031，95% CI [−0.0143, +0.0040]，p = 0.6430。
  **打平。** M2 每条多花 88 token，没有买到排序质量。

所以默认只跑 M1（全库、零成本、毫秒级），M2 作为**显式的"深度复核"**
只对用户圈定的少数候选人跑。这正是推荐系统的召回/精排分层，
但这里的依据是本项目的实验数据，不是行业惯例。

**精排的输入是去标识后的简历**（`risk/deident.py`）。exp6 量过：
同一套余弦，读敏感字段 DI 0.356，去掉敏感字段与姓名 DI 0.935。
精排要读整段简历，所以它比召回更需要这一步。

**关于公平性，前端必须一起显示的话**：M1 的 DI = 0.708，按 4/5 法则会被判
"存在不利影响"，但本集敏感属性是随机分配的，DI 的零分布中位数是 0.875、
5% 分位恰好是 0.708，置换双尾 p = 0.120 —— **落在零分布内**。
本集规模下 4/5 法则约有 20% 假阳性。反过来也一样：真有一个小到 DI≈0.85 的
偏见，本集也发现不了。见 `FAIRNESS_NOTE`。
"""

from __future__ import annotations

import json

from ..llm import LLM
from ..risk.deident import deident
from ..tools.skills import SkillAligner, llm_extract, llm_per_skill
from .store import text_hash

# exp6 在 dev 上选出的对齐阈值（D7：只在 dev 上选，test 只报数）。
TAU = 0.75

# 界面上显示的匹配率与 exp6 报告里的 NDCG **不是同一个东西**，必须说清楚，
# 否则等于拿一个加了词典的产品指标去蹭论文里没加词典的数字。
ALIAS_NOTE = (
    "排序用的是 exp6 的结构化对齐方法（τ=0.75），但**多了一张人工同义词表**"
    "（7 组、19 个写法，如 Microservices ↔ 微服务架构、Kubernetes ↔ K8s）。"
    "原因是嵌入模型对跨语言的同名技术几乎无能为力（实测余弦 0.57~0.58，低于阈值），"
    "不收这张表，招聘官会看到每个候选人都标着「✗ Microservices」，"
    "而其中一半人简历里写着「微服务架构」。"
    "**exp6 报出的 NDCG 0.9909 不含这张表** —— 那是纯嵌入对齐的成绩。"
    "在评测集上两者排序几乎一致：所有候选人都缺同一条要求，名次不受影响。"
)

# 前端展示的公平性声明 —— 措辞与 exp6 报告一致，不弱化也不夸大。
FAIRNESS_NOTE = {
    "metric": "DI（4/5 法则）· 性别",
    "observed": 0.708,
    "null_p05": 0.708,
    "null_median": 0.875,
    "p_value": 0.120,
    "rule_would_flag": True,
    "in_null": True,
    "text": (
        "M1 的 DI = 0.708，按 4/5 法则（<0.8）会被判「存在不利影响」。"
        "但本评测集的敏感属性是随机分配的，DI 本身是随机量："
        "零分布中位数 0.875、5% 分位 0.708，置换双尾 p = 0.120，落在零分布内。"
        "本集规模下 4/5 法则约有 20% 的假阳性 —— 这个数字不足以判定偏见。"
    ),
    "caveat": (
        "这不是「没有偏见」的证明。本集 n=120、k=60 的检出力有限，"
        "真存在一个 DI≈0.85 的小偏见，本集也发现不了。"
        "排序器只读技能证据、不读性别年龄婚育，这是设计上的约束，"
        "不是本指标的结论。"
    ),
}


def _verdict(must_hit: list[str], must_miss: list[str]) -> str:
    """把分数翻译成 HR 能据以行动的判断。

    分数是连续量，但 HR 的决策是分档的：能不能进面。所以除了 score 之外
    必须给一个**离散结论**，且依据要能指回具体哪条要求没达标。

    分档依据来自评测集本身：`缺must` 类样本的金标相关性为 0 ——
    must 缺一条就已经不该进面，缺全部则连"待定"都不该给。
    """
    if not must_miss:
        return "推荐"
    if not must_hit:
        return "不达标"
    return "待定"


def recall(job: dict, candidates: list[dict], aligner: SkillAligner,
           tau: float = TAU) -> list[dict]:
    """M1：对全库候选人打结构化分。零 LLM 调用。

    返回按分降序排列的结果，每条都带 must/nice 的**逐条命中明细** ——
    HR 需要知道"差在哪一条"。只给一个分数是不可复核的，也就无法申诉。
    """
    must, nice = job.get("must") or [], job.get("nice") or []
    out = []
    for c in candidates:
        skills = c.get("skills") or []
        must_hit, must_miss = aligner.coverage(must, skills, tau)
        nice_hit, _ = aligner.coverage(nice, skills, tau)
        mc = len(must_hit) / len(must) if must else 0.0
        nc = len(nice_hit) / len(nice) if nice else 0.0
        # 与 exp6 的 structured_score 同构：must 为主、nice 为辅，
        # must 全缺时不给 nice 分（缺 must 直接出局）。
        score = 0.0 if mc == 0.0 else mc + 0.5 * nc
        out.append({
            "candidate_id": c["candidate_id"],
            "name": c.get("name", ""),
            "source": c.get("source", ""),
            "skills": skills,
            "score": round(score, 4),
            # 归一化到 0~100 只为展示。**排序永远用原始 score**，
            # 用归一化值排序会在 must/nice 条数不同的岗位间产生错觉。
            "match_pct": round(100 * score / 1.5),
            "must_pct": round(100 * mc),
            "nice_pct": round(100 * nc),
            "must_hit": must_hit, "must_miss": must_miss, "nice_hit": nice_hit,
            "verdict": _verdict(must_hit, must_miss),
            "mode": "召回",
        })
    out.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    return out


def rerank(job: dict, rows: list[dict], candidates: dict[str, dict],
           llm: LLM, tau: float = TAU) -> list[dict]:
    """M2：对给定结果逐条做 LLM 逐技能复核，就地更新分数与明细。

    只该对**用户圈定的少数人**调用：每条 88 token，全库跑一遍成本是召回的无穷倍
    （召回是 0），而 exp6 说这点钱买不到排序质量。

    复核用的是**去标识后**的简历。这会让 `has` 的判定与召回阶段不完全一致 ——
    这是刻意的：不一致的地方恰好是"模型原本靠敏感属性推断出的技能"。
    """
    must, nice = job.get("must") or [], job.get("nice") or []
    reqs = must + nice
    for r in rows:
        c = candidates.get(r["candidate_id"])
        if not c:
            continue
        clean = deident(c["resume_text"], True)
        has, meta = llm_per_skill(reqs, clean, llm)
        hit = {h for h in has}
        r["must_hit"] = [m for m in must if m in hit]
        r["must_miss"] = [m for m in must if m not in hit]
        r["nice_hit"] = [n for n in nice if n in hit]
        mc = len(r["must_hit"]) / len(must) if must else 0.0
        nc = len(r["nice_hit"]) / len(nice) if nice else 0.0
        r["score"] = round(0.0 if mc == 0.0 else mc + 0.5 * nc, 4)
        r["match_pct"] = round(100 * r["score"] / 1.5)
        r["must_pct"] = round(100 * mc)
        r["nice_pct"] = round(100 * nc)
        r["verdict"] = _verdict(r["must_hit"], r["must_miss"])
        r["mode"] = "精排"
        r["rerank_tokens"] = meta.get("tokens", 0)
    rows.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    return rows


def decorate(rows: list[dict], job: dict) -> list[dict]:
    """把库里存的命中明细还原成展示所需的派生字段（百分比、结论、分数）。

    **派生量不落库，每次读出来重算。** 匹配结果只存事实：
    命中了哪几条要求、缺了哪几条。理由是派生量会随岗位要求的编辑而失效 ——
    岗位的 must 改掉一条，历史匹配记录里的百分比就全是错的，
    而库里那份数字不会有任何提示，界面上照常显示一个陈旧的匹配度。
    重算的代价是几十次除法，比"显示一个没人知道已经过期的数字"便宜得多。

    也正因为原先没做这件事，岗位详情页整列显示 `undefined%` ——
    库里只有 `must_hit`/`must_miss`，没有 `must_pct`。
    """
    must, nice = job.get("must") or [], job.get("nice") or []
    for r in rows:
        mc = len(r["must_hit"]) / len(must) if must else 0.0
        nc = len(r["nice_hit"]) / len(nice) if nice else 0.0
        r["score"] = round(0.0 if mc == 0.0 else mc + 0.5 * nc, 4)
        r["match_pct"] = round(100 * r["score"] / 1.5)
        r["must_pct"] = round(100 * mc)
        r["nice_pct"] = round(100 * nc)
        r["verdict"] = _verdict(r["must_hit"], r["must_miss"])
    rows.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    return rows


def extract_job_skills(job: dict, llm: LLM) -> dict:
    """给岗位抽 must/nice —— 录入手写 JD 时的辅助。"""
    from ..tools.skills import extract_requirements
    return extract_requirements(job["jd_text"], llm)[0]


def extract_resume_skills(text: str, llm: LLM) -> tuple[list[str], int]:
    """给简历抽技能（exp6 的 S1 抽取器）。返回 (技能, token 数)。"""
    skills, meta = llm_extract(text, llm)
    return skills, meta.get("tokens", 0)


def candidates_from_match_eval(path, limit_jobs: int = 0) -> tuple[list[dict], list[dict]]:
    """把冻结评测集 `eval/match_eval.jsonl` 读成 (岗位, 候选人) 供演示装载。

    **只读**。它不写回评测集，也不碰 `data/hr.db`。
    这样演示一打开就有真实规模的数据（12 个岗位 / 上百份简历），
    而不必让评审先手工录 100 份简历。
    """
    from ..tools.skills import parse_jd
    jobs: dict[str, dict] = {}
    cands: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        jid = r["job_id"]
        if jid not in jobs:
            if limit_jobs and len(jobs) >= limit_jobs:
                continue
            must, nice = parse_jd(r["jd"])
            jobs[jid] = {"job_id": jid, "title": r.get("job_title") or jid,
                         "dept": "", "jd_text": r["jd"], "must": must, "nice": nice,
                         "headcount": 1, "status": "招聘中"}
        # 候选人 id 用**简历文本的稳定哈希**，不能用内置 `hash()` ——
        # 后者对 str 是按进程加盐的（PYTHONHASHSEED），重启一次 id 全变，
        # 已存的匹配结果会全部指空。这种"数据看着还在、关联悄悄断了"的
        # 故障不报错，只表现为匹配列表变空。
        cid = "C" + text_hash(r["resume"])
        cands.setdefault(cid, {"candidate_id": cid, "name": _resume_name(r["resume"]),
                               "resume_text": r["resume"], "skills": [],
                               "source": "评测集导入", "applied_job": jid})
    return list(jobs.values()), list(cands.values())


def _resume_name(text: str) -> str:
    head = text.splitlines()[0] if text else ""
    return head.split("|")[0].split("　")[0].strip() or "（未署名）"
