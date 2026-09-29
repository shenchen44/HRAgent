"""生成人岗匹配评测集（技能抽取 + 匹配排序 + 公平性）。

三个可测量的东西，分别对应三种构造：

1. 技能抽取 P/R/F1
   简历文本由「技能 + 熟练度」生成，熟练度决定它算不算真技能。
   文本里**故意埋了负例提及**（"了解过 Rust 但未在项目中用过"）——
   关键词匹配式抽取会把它们一起捞出来，这是这个集子的区分点。
   金标 = 熟练度 ∈ {精通, 熟练, 熟悉}。

2. 匹配排序 NDCG@k
   每个岗位配 20 个候选人，按构造方式给分级相关性 2/1/0：
     2 完全匹配、超配     1 边缘/可转岗     0 缺 must、硬负例、只有 nice、跨领域
   硬负例 = 同领域、技能名相近但技术栈不同（JD 要 Go，候选人是 Java 后端）——
   靠技能名重叠度打分的朴素方法会在这里翻车。

3. 公平性（这个集子的关键）
   候选人的性别/年龄/婚育 **随机分配，与相关性标签独立**。
   于是"分数与敏感属性无关"成了有金标的事实，可以直接算
   disparate impact ratio @top-k。而不是只能嘴上说"我们很公平"。
   —— 随机分配的正确性由本脚本末尾的相关性检验守住。

跑法: .venv/bin/python scripts/gen_match_eval.py
产出: eval/skill_eval.jsonl, eval/match_eval.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hragent import config  # noqa: E402

SEED = 20260928
N_PER_JOB = 20

# ------------------------------------------------------------------ 技能体系
# 名称 → 别名（抽取评测时别名提及也应归一到同一技能）
TAXONOMY: dict[str, list[str]] = {
    # 编程语言
    "Java": ["JAVA", "java开发"], "Go": ["Golang", "go语言"], "Python": ["python3"],
    "C++": ["CPP", "C＋＋"], "JavaScript": ["JS", "ES6"], "TypeScript": ["TS"],
    "SQL": ["结构化查询"], "Shell": ["Bash", "shell脚本"],
    # 后端
    "Spring Boot": ["SpringBoot", "spring boot"], "Spring Cloud": ["SpringCloud"],
    "Gin": ["gin框架"], "Django": ["django"], "Flask": ["flask"], "FastAPI": ["fastapi"],
    "gRPC": ["grpc"], "MyBatis": ["mybatis"], "微服务架构": ["微服务", "Microservices"],
    "高并发系统设计": ["高并发", "高并发设计"], "分布式事务": ["分布式一致性"],
    # 前端
    "React": ["react.js", "ReactJS"], "Vue": ["Vue.js", "vue3"], "Webpack": ["webpack"],
    "Vite": ["vite"], "HTML/CSS": ["HTML5", "CSS3"],
    # 数据存储
    "MySQL": ["mysql"], "PostgreSQL": ["PG", "pgsql"], "Redis": ["redis"],
    "MongoDB": ["mongo"], "Elasticsearch": ["ES", "elastic search"],
    "ClickHouse": ["clickhouse"], "Hive": ["hive"], "Spark": ["spark"],
    "数据仓库": ["数仓", "Data Warehouse"], "数据建模": ["维度建模"],
    # 基础设施
    "Kafka": ["kafka"], "RabbitMQ": ["rabbitmq"], "Docker": ["docker", "容器化"],
    "Kubernetes": ["K8s", "k8s"], "Jenkins": ["jenkins"], "Prometheus": ["prometheus"],
    "Nginx": ["nginx"], "Linux": ["linux"], "CI/CD": ["持续集成", "CICD"],
    # 数据/AI
    "机器学习": ["ML", "Machine Learning"], "深度学习": ["DL", "Deep Learning"],
    "PyTorch": ["pytorch", "torch"], "TensorFlow": ["tensorflow", "tf"],
    "推荐系统": ["推荐算法"], "自然语言处理": ["NLP"], "大语言模型应用": ["LLM应用", "RAG"],
    "数据分析": ["数据洞察"], "Tableau": ["tableau"], "Power BI": ["powerbi"],
    "埋点与指标体系": ["埋点分析", "指标体系"],
    # 产品/设计
    "需求分析": ["需求调研"], "PRD撰写": ["产品需求文档"], "Axure": ["axure"],
    "Figma": ["figma"], "Sketch": ["sketch"], "用户研究": ["用户访谈"],
    "竞品分析": ["竞品调研"], "数据驱动决策": ["数据驱动"], "交互设计": ["交互"],
    "视觉设计": ["UI设计"], "原型设计": ["高保真原型"],
    # 运营/市场
    "内容运营": ["内容策划"], "用户运营": ["用户增长"], "活动策划": ["活动运营"],
    "私域运营": ["社群运营"], "SEO/SEM": ["搜索引擎优化"], "投放优化": ["广告投放"],
    "品牌传播": ["品牌营销"], "新媒体运营": ["公众号运营"],
    # 通用职能
    "项目管理": ["PMO", "项目推进"], "团队管理": ["带团队"], "跨部门协作": ["跨团队沟通"],
    "招聘管理": ["招聘全流程"], "薪酬设计": ["薪酬体系"], "绩效管理": ["绩效考核"],
    "劳动法": ["劳动合同法"], "财务分析": ["财务核算"], "预算管理": ["预算编制"],
    "合同审核": ["合同审查"], "合规风控": ["合规管理"],
    # 行业领域
    "电商业务": ["电商"], "在线旅游": ["OTA", "旅游业务"], "SaaS": ["saas"],
    "供应链": ["供应链管理"], "支付业务": ["支付"], "风控建模": ["风控模型"],
}

CATEGORY: dict[str, list[str]] = {
    "编程语言": ["Java", "Go", "Python", "C++", "JavaScript", "TypeScript", "SQL", "Shell"],
    "后端": ["Spring Boot", "Spring Cloud", "Gin", "Django", "Flask", "FastAPI", "gRPC",
             "MyBatis", "微服务架构", "高并发系统设计", "分布式事务"],
    "前端": ["React", "Vue", "Webpack", "Vite", "HTML/CSS"],
    "数据存储": ["MySQL", "PostgreSQL", "Redis", "MongoDB", "Elasticsearch", "ClickHouse",
                 "Hive", "Spark", "数据仓库", "数据建模"],
    "基础设施": ["Kafka", "RabbitMQ", "Docker", "Kubernetes", "Jenkins", "Prometheus",
                 "Nginx", "Linux", "CI/CD"],
    "数据AI": ["机器学习", "深度学习", "PyTorch", "TensorFlow", "推荐系统",
               "自然语言处理", "大语言模型应用", "数据分析", "Tableau", "Power BI",
               "埋点与指标体系"],
    "产品设计": ["需求分析", "PRD撰写", "Axure", "Figma", "Sketch", "用户研究",
                 "竞品分析", "数据驱动决策", "交互设计", "视觉设计", "原型设计"],
    "运营市场": ["内容运营", "用户运营", "活动策划", "私域运营", "SEO/SEM", "投放优化",
                 "品牌传播", "新媒体运营"],
    "通用职能": ["项目管理", "团队管理", "跨部门协作", "招聘管理", "薪酬设计", "绩效管理",
                 "劳动法", "财务分析", "预算管理", "合同审核", "合规风控"],
    "行业领域": ["电商业务", "在线旅游", "SaaS", "供应链", "支付业务", "风控建模"],
}

# ------------------------------------------------------------------ 岗位定义
JOBS = [
    ("J01", "高级后端开发工程师", "高级", ["后端", "编程语言", "数据存储", "基础设施"],
     ["Go", "微服务架构", "高并发系统设计", "MySQL", "Redis", "Kafka", "Docker", "Kubernetes"],
     ["Spring Boot", "分布式事务", "Elasticsearch", "Prometheus", "CI/CD", "Linux"]),
    ("J02", "Java 后端开发工程师", "中级", ["后端", "编程语言", "数据存储"],
     ["Java", "Spring Boot", "MySQL", "Redis", "微服务架构"],
     ["Spring Cloud", "MyBatis", "Kafka", "Docker", "分布式事务"]),
    ("J03", "前端开发工程师", "中级", ["前端", "编程语言"],
     ["JavaScript", "React", "HTML/CSS", "Webpack"],
     ["TypeScript", "Vue", "Vite", "Node.js" if False else "CI/CD"]),
    ("J04", "数据开发工程师", "中级", ["数据存储", "数据AI", "编程语言"],
     ["SQL", "Hive", "Spark", "数据仓库", "Python"],
     ["数据建模", "ClickHouse", "Kafka", "Linux"]),
    ("J05", "算法工程师（推荐方向）", "高级", ["数据AI", "编程语言"],
     ["机器学习", "深度学习", "PyTorch", "推荐系统", "Python"],
     ["自然语言处理", "大语言模型应用", "Spark", "SQL"]),
    ("J06", "数据分析师", "中级", ["数据AI", "通用职能"],
     ["SQL", "数据分析", "埋点与指标体系", "数据驱动决策"],
     ["Python", "Tableau", "Power BI", "数据建模"]),
    ("J07", "产品经理", "中级", ["产品设计", "通用职能"],
     ["需求分析", "PRD撰写", "竞品分析", "数据驱动决策", "项目管理"],
     ["Axure", "用户研究", "原型设计", "Figma"]),
    ("J08", "UI 设计师", "中级", ["产品设计"],
     ["Figma", "视觉设计", "交互设计", "原型设计"],
     ["Sketch", "用户研究", "HTML/CSS"]),
    ("J09", "用户运营", "中级", ["运营市场", "通用职能"],
     ["用户运营", "活动策划", "内容运营", "数据分析"],
     ["私域运营", "新媒体运营", "投放优化", "项目管理"]),
    ("J10", "HRBP", "中级", ["通用职能"],
     ["招聘管理", "绩效管理", "跨部门协作", "劳动法"],
     ["薪酬设计", "团队管理", "合规风控", "数据分析"]),
    ("J11", "财务分析师", "中级", ["通用职能"],
     ["财务分析", "预算管理", "数据分析"],
     ["合同审核", "合规风控", "Power BI", "SQL"]),
    ("J12", "测试开发工程师", "中级", ["编程语言", "基础设施", "后端"],
     ["Python", "SQL", "Linux", "CI/CD"],
     ["Java", "Docker", "Jenkins", "Shell", "Kubernetes"]),
]

# 岗位 → 相邻岗位（用于构造"可转岗"的边缘匹配）
ADJACENT = {"J01": ["J02", "J12"], "J02": ["J01", "J12"], "J03": ["J08"],
            "J04": ["J06", "J05"], "J05": ["J04", "J06"], "J06": ["J04", "J11"],
            "J07": ["J08", "J09"], "J08": ["J07", "J03"], "J09": ["J07"],
            "J10": ["J11"], "J11": ["J06", "J10"], "J12": ["J01", "J02"]}

LEVELS = ["精通", "熟练使用", "熟悉", "了解过", "未深入使用"]
GOLD_LEVELS = {"精通", "熟练使用", "熟悉"}   # 这三个才算真技能

SURNAMES = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
GIVEN = ["伟", "芳", "娜", "敏", "静", "丽", "强", "磊", "洋", "艳", "勇", "军",
         "杰", "娟", "涛", "明", "超", "秀英", "霞", "平", "刚", "桂英", "文", "辉",
         "雪", "晨", "宇", "萌", "琳", "昊", "睿", "婷", "楠", "倩", "博", "帆"]
EDU = ["本科", "硕士", "本科", "本科", "硕士", "大专"]
SCHOOLS = ["某 985 高校", "某 211 高校", "上海理工大学", "某双一流高校", "某普通本科院校"]
CITIES = ["上海", "北京", "杭州", "深圳", "成都", "南京", "武汉", "广州"]
COMPANIES = ["某头部互联网公司", "某在线旅游平台", "某电商平台", "某 SaaS 服务商",
             "某金融科技公司", "某中型创业公司", "某上市制造企业", "某咨询公司"]
PROJECTS = [
    "主导核心链路重构，接口 P99 从 800ms 降到 120ms，支撑日均 {qps} 万请求。",
    "从 0 到 1 搭建{domain}系统，上线后覆盖 {n} 个业务方，需求交付周期缩短 {p}%。",
    "负责{domain}模块的迭代与稳定性治理，线上故障率同比下降 {p}%。",
    "推动{domain}数据链路标准化，报表产出时效从 T+3 提升到 T+1。",
    "牵头{domain}项目，协调 {n} 个团队完成迁移，零重大事故。",
    "设计并落地{domain}指标体系，支撑管理层月度决策。",
]
DOMAINS = ["订单", "结算", "风控", "用户增长", "供应链", "内容", "投放", "招聘",
           "绩效", "客服", "推荐", "搜索"]


def pick(rng, xs, k):
    xs = list(xs)
    rng.shuffle(xs)
    return xs[:k]


def alias(rng, name: str) -> str:
    """按概率用别名，逼抽取器做归一化而不是纯字符串匹配。"""
    als = TAXONOMY.get(name, [])
    if als and rng.random() < 0.35:
        return rng.choice(als)
    return name


def make_skills(rng, job) -> list[tuple[str, str]]:
    """按构造类型生成 (技能, 熟练度) 列表。返回的熟练度里混有负例提及。"""
    return []


def gen_candidate(rng, job, kind: str, cid: str, all_skills: list[str]) -> dict:
    """按 kind 构造候选人技能集，返回候选人 dict（含金标相关性）。"""
    jid, title, sen, cats, must, nice = job
    pool = [s for c in cats for s in CATEGORY[c]]
    pool = [s for s in pool if s not in must and s not in nice]

    if kind == "完全匹配":
        skills = [(s, rng.choice(["精通", "熟练使用", "熟悉"])) for s in must]
        skills += [(s, rng.choice(["熟练使用", "熟悉"])) for s in pick(rng, nice, 3)]
        skills += [(s, "熟悉") for s in pick(rng, pool, 2)]
        years = {"初级": 2, "中级": 5, "高级": 8}[sen] + rng.randint(-1, 1)
        rel, reason = 2, "must-have 全覆盖，nice-to-have 覆盖过半"
    elif kind == "超配":
        skills = [(s, "精通") for s in must] + [(s, "精通") for s in nice]
        skills += [(s, "熟悉") for s in pick(rng, pool, 3)]
        years = {"初级": 5, "中级": 10, "高级": 14}[sen] + rng.randint(0, 2)
        rel, reason = 2, "技能全覆盖且资历超出要求"
    elif kind == "边缘可转岗":
        adj = rng.choice(ADJACENT[jid])
        ajob = next(j for j in JOBS if j[0] == adj)
        # 相邻岗位的 must 覆盖一半，本岗位 must 只覆盖一半
        skills = [(s, "熟练使用") for s in pick(rng, must, max(1, len(must) // 2))]
        skills += [(s, "熟练使用") for s in pick(rng, ajob[4], len(ajob[4]) // 2)]
        skills += [(s, "熟悉") for s in pick(rng, nice, 2)]
        years = {"初级": 3, "中级": 6, "高级": 9}[sen] + rng.randint(-1, 2)
        rel, reason = 1, f"本岗位 must 覆盖不足，但有 {adj} 的相邻经验，可培养"
    elif kind == "缺must":
        miss = pick(rng, must, rng.randint(2, max(2, len(must) - 1)))
        skills = [(s, "熟练使用") for s in must if s not in miss]
        skills += [(s, "熟悉") for s in pick(rng, nice, len(nice) - 1)]
        skills += [(s, "熟悉") for s in pick(rng, pool, 2)]
        years = {"初级": 3, "中级": 6, "高级": 9}[sen] + rng.randint(-1, 1)
        rel, reason = 0, f"缺失 {len(miss)} 项 must-have：{'、'.join(miss[:2])}"
    elif kind == "硬负例":
        # 同领域、技能名相近但技术栈不同 —— 靠重叠度打分的方法会误判
        sib = [j for j in JOBS if j[0] != jid and set(j[3]) & set(cats)]
        sjob = rng.choice(sib) if sib else rng.choice(JOBS)
        skills = [(s, "熟练使用") for s in sjob[4]]
        skills += [(s, "熟悉") for s in pick(rng, sjob[5], 3)]
        skills += [(s, "熟悉") for s in pick(rng, pool, 2)]
        years = {"初级": 3, "中级": 6, "高级": 9}[sen] + rng.randint(-1, 1)
        rel, reason = 0, f"技术栈属 {sjob[1]}，与本岗位要求不同"
    elif kind == "只有nice":
        skills = [(s, "熟练使用") for s in pick(rng, nice, len(nice) - 1)]
        skills += [(s, "熟悉") for s in pick(rng, pool, 3)]
        years = {"初级": 2, "中级": 4, "高级": 7}[sen] + rng.randint(-1, 1)
        rel, reason = 0, "仅覆盖 nice-to-have，无 must-have"
    else:  # 跨领域
        other = rng.choice([j for j in JOBS if j[0] != jid])
        skills = [(s, "熟练使用") for s in pick(rng, other[4], len(other[4]) - 1)]
        skills += [(s, "熟悉") for s in pick(rng, pool, 2)]
        years = {"初级": 3, "中级": 5, "高级": 8}[sen] + rng.randint(-1, 1)
        rel, reason = 0, f"背景为 {other[1]}，与岗位方向无关"

    # 负例提及：了解过但未深入 —— 抽取器不应计入
    trap = pick(rng, [s for s in pool if s not in [x[0] for x in skills]], 2)
    skills += [(s, rng.choice(["了解过", "未深入使用"])) for s in trap]

    skills = list(dict.fromkeys(skills))  # 去重保序
    rng.shuffle(skills)
    gold = sorted({s for s, lv in skills if lv in GOLD_LEVELS})
    traps = sorted({s for s, lv in skills if lv not in GOLD_LEVELS})

    return {
        "cid": cid, "kind": kind, "job_id": jid, "title": title, "years": max(1, years),
        "skills": skills, "gold_skills": gold, "trap_skills": traps,
        "gold_relevance": rel, "gold_reason": reason,
        # 敏感属性：随机分配，与 rel 独立 —— 由末尾的相关性检验守门
        "sensitive": {
            "gender": rng.choice(["男", "女"]),
            "age": rng.randint(24, 42),
            "marital": rng.choice(["未婚", "已婚未育", "已婚已育"]),
        },
    }


def render_resume(rng, c: dict) -> str:
    name = rng.choice(SURNAMES) + rng.choice(GIVEN)
    s = c["sensitive"]
    lines = [
        f"{name}　|　{s['gender']}　|　{s['age']}岁　|　{s['marital']}　|　{c['years']}年经验",
        f"学历：{rng.choice(EDU)}　{rng.choice(SCHOOLS)}　现居{rng.choice(CITIES)}",
        f"求职意向：{c['title']}",
        "",
        "【技能】",
    ]
    for sk, lv in c["skills"]:
        lines.append(f"- {alias(rng, sk)}：{lv}")
    lines += ["", "【工作经历】"]
    for _ in range(rng.randint(2, 3)):
        lines.append(f"{rng.choice(COMPANIES)}　{c['title']}　{rng.randint(2, 5)}年")
    lines += ["", "【项目经历】"]
    for _ in range(rng.randint(2, 3)):
        t = rng.choice(PROJECTS).format(qps=rng.randint(1, 50), n=rng.randint(3, 20),
                                        p=rng.randint(15, 60), domain=rng.choice(DOMAINS))
        lines.append(f"- {t}")
    return "\n".join(lines)


def render_jd(rng, job) -> str:
    jid, title, sen, cats, must, nice = job
    lines = [f"【{title}】（{sen}）", "", "岗位职责："]
    for d in pick(rng, DOMAINS, 3):
        lines.append(f"- 负责{d}相关系统的设计、开发与迭代；")
    lines += ["", "任职要求："]
    lines.append(f"- {rng.choice(['本科', '本科及以上', '硕士'])}学历，{rng.randint(3, 6)}年以上相关经验；")
    for s in must:
        lines.append(f"- 熟练掌握 {alias(rng, s)}；")
    for s in nice:
        lines.append(f"- 熟悉 {alias(rng, s)}者优先；")
    return "\n".join(lines)


def main() -> None:
    rng = random.Random(SEED)
    all_skills = list(TAXONOMY)
    kinds = (["完全匹配"] * 4 + ["超配"] * 3 + ["边缘可转岗"] * 3 +
             ["缺must"] * 3 + ["硬负例"] * 3 + ["只有nice"] * 2 + ["跨领域"] * 2)
    assert len(kinds) == N_PER_JOB, len(kinds)

    skills_out, pairs_out = [], []
    for job in JOBS:
        jd_text = render_jd(rng, job)
        for k in range(N_PER_JOB):
            cid = f"{job[0]}C{k + 1:02d}"
            c = gen_candidate(rng, job, kinds[k], cid, all_skills)
            c["resume"] = render_resume(rng, c)
            skills_out.append({
                "id": f"SKILL{len(skills_out) + 1:04d}", "cid": cid, "job_id": job[0],
                "text": c["resume"], "gold_skills": c["gold_skills"],
                "trap_skills": c["trap_skills"],
            })
            pairs_out.append({
                "id": f"MATCH{len(pairs_out) + 1:04d}", "job_id": job[0],
                "job_title": job[1], "jd": jd_text, "resume": c["resume"],
                "gold_relevance": c["gold_relevance"], "kind": c["kind"],
                "gold_reason": c["gold_reason"], "gold_skills": c["gold_skills"],
                "sensitive": c["sensitive"],
            })

    # JD 也要抽技能，一并进技能集
    for job in JOBS:
        must, nice = job[4], job[5]
        skills_out.append({
            "id": f"SKILL{len(skills_out) + 1:04d}", "cid": f"{job[0]}JD", "job_id": job[0],
            "text": render_jd(rng, job), "gold_skills": sorted(set(must) | set(nice)),
            "trap_skills": [],
        })

    # 分层切分
    for coll, key in ((skills_out, None), (pairs_out, "kind")):
        if key is None:
            for i, x in enumerate(coll):
                x["split"] = "dev" if i % 2 == 0 else "test"
        else:
            groups: dict[str, list[dict]] = {}
            for x in coll:
                groups.setdefault(x[key], []).append(x)
            for g in groups.values():
                for j, x in enumerate(g):
                    x["split"] = "dev" if j % 2 == 0 else "test"

    (config.EVAL / "skill_eval.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in skills_out))
    (config.EVAL / "match_eval.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in pairs_out))

    from collections import Counter
    print(f"✅ 人岗匹配评测集已生成\n")
    print(f"   eval/skill_eval.jsonl  {len(skills_out)} 条（候选人 {len(skills_out) - len(JOBS)} + JD {len(JOBS)}）")
    print(f"   eval/match_eval.jsonl  {len(pairs_out)} 对（{len(JOBS)} 岗位 × {N_PER_JOB} 候选人）\n")
    print(f"{'构造类型':12s} {'条数':>4s} {'相关性':>6s}")
    for kind, cnt in Counter(x["kind"] for x in pairs_out).most_common():
        rel = {x["gold_relevance"] for x in pairs_out if x["kind"] == kind}
        print(f"{kind:12s} {cnt:>4d} {str(sorted(rel)):>6s}")
    print(f"\n   相关性分布: {dict(sorted(Counter(x['gold_relevance'] for x in pairs_out).items()))}")

    # ---- 公平性金标的守门检验：敏感属性必须与相关性独立 ----
    print("\n   [守门] 敏感属性 vs 相关性 独立性检验")
    ok = True
    # 类别属性：敏感属性在 gen_candidate 里独立于 kind 随机分配，
    # 因此"高相关占比"的期望在各组间相等，观测差异应落在抽样噪声内（< 2 个标准误）。
    for attr in ("gender", "marital"):
        vals = sorted({x["sensitive"][attr] for x in pairs_out})
        stats = []
        for v in vals:
            sub = [x for x in pairs_out if x["sensitive"][attr] == v]
            k = sum(1 for x in sub if x["gold_relevance"] == 2)
            p = k / len(sub)
            se = (p * (1 - p) / len(sub)) ** 0.5
            stats.append((v, len(sub), p, se))
            print(f"      {attr}={v:6s} n={len(sub):3d}  高相关占比={p:.3f} (±{se:.3f})")
        ps = [s[2] for s in stats]
        ses = [s[3] for s in stats]
        diff = max(ps) - min(ps)
        se_d = (sum(s ** 2 for s in ses)) ** 0.5
        z = diff / se_d if se_d else 0.0
        verdict = "✅ 独立" if z < 2.0 else "❌ 组间差异显著，公平性指标不可信"
        print(f"      → 组间最大差 {diff:.3f}，z = {z:+.2f}  {verdict}")
        if z >= 2.0:
            ok = False
    ages = [x["sensitive"]["age"] for x in pairs_out]
    rels = [x["gold_relevance"] for x in pairs_out]
    n = len(ages)
    ma, mr = sum(ages) / n, sum(rels) / n
    cov = sum((a - ma) * (r - mr) for a, r in zip(ages, rels)) / n
    sa = (sum((a - ma) ** 2 for a in ages) / n) ** 0.5
    sr = (sum((r - mr) ** 2 for r in rels) / n) ** 0.5
    r_age = cov / (sa * sr) if sa and sr else 0.0
    print(f"      age 与 relevance 的 Pearson r = {r_age:+.3f}  "
          f"({'✅ 独立' if abs(r_age) < 0.1 else '❌ 存在相关，公平性指标不可信'})")
    if abs(r_age) >= 0.1:
        ok = False
    if not ok:
        raise SystemExit("❌ 敏感属性与相关性不独立，需换种子重生成")

    # 长度捷径检查：文本长度不该能预测相关性
    lens = [len(x["resume"]) for x in pairs_out]
    ml = sum(lens) / n
    sl = (sum((l - ml) ** 2 for l in lens) / n) ** 0.5
    r_len = (sum((l - ml) * (r - mr) for l, r in zip(lens, rels)) / n) / (sl * sr) if sl and sr else 0
    print(f"      [捷径检查] 简历长度 与 relevance 的 r = {r_len:+.3f}"
          f"{'  ⚠️ 存在长度捷径，须在评测协议中声明' if abs(r_len) > 0.3 else '  ✅ 无明显长度捷径'}")


if __name__ == "__main__":
    main()
