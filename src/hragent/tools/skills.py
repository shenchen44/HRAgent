"""技能抽取与结构化人岗匹配 —— exp6 与 UI **共用的唯一实现**。

为什么要有这个模块：这套逻辑原先只长在 `scripts/exp6_skill_match.py` 里。
前端要复用时就只有两条路 —— 让 UI 去 import 一个实验脚本（依赖方向是反的），
或者复制一份（**两份真相**，正是本项目反复踩的那类 bug 的温床）。
所以把它提到这里，exp6 改成 import，用 `verify_repro.py` 证明这次搬家
**没有改变任何一个已发布的数字**。

搬家的同时修掉一个潜伏的坑：原先 `llm_extract` 报的是
`llm.ledger.summary()["total_tokens"]` —— **账本的累计值**，不是本次调用的增量。
它在 exp6 里是对的，只因为 exp6 每次调用都 `LLM()` 新建一个账本
（见 `do_extract` / `do_pair`，每次循环内新建）。一旦有人把那个构造提到循环外
（很自然的"优化"），成本指标就会静默变成累计和，且不报错。
现在改成**显式取增量**：对每次新建账本的调用方结果完全不变，
对共用账本的调用方（UI）才是对的。
"""

from __future__ import annotations

import re

from ..llm import LLM

# ---------------------------------------------------------------- 抽取

# S0 的**词表**只能来自 dev 划分：用 test 的 gold_skills 建词表就是泄题。
# 词表 = dev 的 gold_skills ∪ trap_skills —— trap 也在里面，
# 因为"词表匹配"分不出 trap，这正是 exp6 要量出来的东西。
def build_lexicon(skill_rows: list[dict]) -> list[str]:
    lex: set[str] = set()
    for r in skill_rows:
        if r["split"] != "dev":
            continue
        lex |= set(r["gold_skills"]) | set(r.get("trap_skills") or [])
    return sorted(lex)


def lexicon_extract(text: str, lex: list[str]) -> list[str]:
    """子串命中即算抽到。大小写不敏感 —— 给词表匹配**最有利**的实现，
    否则会低估它、把结论做偏。"""
    low = text.lower()
    return [s for s in lex if s.lower() in low]


EXTRACT_SYSTEM = """你是简历技能抽取器。从简历中抽取候选人**真正掌握**的技能。

判定标准（关键）：
1. 简历里技能后面常带熟练度标注：「精通」「熟练使用」「熟悉」→ 算掌握。
2. **「了解过」「听说过」「初步接触」→ 不算掌握，必须排除。**
3. 工作经历/项目经历里体现出的技能算掌握（即使技能栏没列）。
4. 技能名用**通用规范写法**：`k8s` → `Kubernetes`，`持续集成` → `CI/CD`，
   `数仓` → `数据仓库`，`微服务` → `微服务架构`。保留英文原名的大小写规范形式。

输出 JSON：{"skills": ["...", "..."]}
只输出 JSON，不要解释。"""


def llm_extract(text: str, llm: LLM) -> tuple[list[str], dict]:
    """抽技能。返回 (技能列表, 本次调用成本)。

    成本取**账本增量**而非累计值，理由见模块 docstring。
    """
    before = llm.ledger.summary()["total_tokens"]
    r = llm.call([{"role": "user", "content": text}], system=EXTRACT_SYSTEM,
                 tag="skill_extract")
    after = llm.ledger.summary()["total_tokens"]
    try:
        d = r.json()
        skills = [str(s).strip() for s in (d.get("skills") or []) if str(s).strip()]
    except Exception as e:  # noqa: BLE001
        return [], {"error": f"{type(e).__name__}: {e}", "tokens": after - before,
                    "latency_s": r.latency_s}
    return skills, {"tokens": after - before, "latency_s": r.latency_s}


# ---------------------------------------------------------------- JD 需求

def parse_jd(jd: str) -> tuple[list[str], list[str]]:
    """把 JD 拆成 must（熟练掌握 X）与 nice（熟悉 X者优先）。

    这是"结构化"的**全部**含义：不靠模型判断哪些是硬要求，
    而是 JD 本来就写明了 —— 抽取它，而不是猜它。

    **只认评测集那套固定句式**。前端录入手写的 JD 时句式不受控，
    要用 `extract_requirements`。两者并存不是重复：这条是零成本的确定路，
    评测集 12 个岗位全部命中；抽不到时会由调用方回落到 LLM 路。
    """
    must = re.findall(r"熟练掌握\s*(.+?)[；;]", jd)
    nice = re.findall(r"熟悉\s*(.+?)者优先[；;]", jd)
    return [m.strip() for m in must], [n.strip() for n in nice]


REQUIRE_SYSTEM = """你是岗位需求结构化器。把 JD 拆成两类要求：

- `must` 硬性要求：不具备就不能胜任（学历、年限、必须掌握的技能/工具/证书）
- `nice` 加分项：JD 里写「优先」「加分」「更佳」的那些

规则：
1. 每条要求是一个**短语**（不超过 20 字），不要整句照抄。
2. 技能名用通用规范写法（`k8s` → `Kubernetes`，`数仓` → `数据仓库`）。
3. **不要**把性别、年龄、婚育、户籍、院校层次等写进要求 —— 那是歧视性条件，
   不是岗位要求；遇到这类表述直接忽略，由系统的合规判据单独处理。
4. JD 里没写清楚的，不要替它编。

输出 JSON：{"must": ["..."], "nice": ["..."]}
只输出 JSON。"""


def extract_requirements(jd: str, llm: LLM) -> tuple[dict, dict]:
    """任意 JD 文本 → {"must": [...], "nice": [...]}。

    先走 `parse_jd` 的确定路（评测集句式），抽不到再用 LLM。
    这样评测集那 12 个岗位零成本、零方差，手写 JD 也有兜底。
    """
    must, nice = parse_jd(jd)
    if must or nice:
        return {"must": must, "nice": nice}, {"tokens": 0, "latency_s": 0.0,
                                              "source": "regex"}
    before = llm.ledger.summary()["total_tokens"]
    r = llm.call([{"role": "user", "content": jd}], system=REQUIRE_SYSTEM,
                 tag="jd_require")
    after = llm.ledger.summary()["total_tokens"]
    meta = {"tokens": after - before, "latency_s": r.latency_s, "source": "llm"}
    try:
        d = r.json()
    except Exception as e:  # noqa: BLE001
        return {"must": [], "nice": []}, {**meta, "error": f"{type(e).__name__}: {e}"}
    clean = lambda xs: [str(s).strip() for s in (xs or []) if str(s).strip()]  # noqa: E731
    return {"must": clean(d.get("must")), "nice": clean(d.get("nice"))}, meta


# ---------------------------------------------------------------- 对齐与打分

def _norm(s: str) -> str:
    return re.sub(r"[\s\-_/]+", "", s).lower()


# ---------------------------------------------------------------- 别名表

# **产品路径专用的人工同义词表。exp6 的数字不包含它。**
#
# 为什么需要：嵌入模型对**跨语言的同名技术**几乎无能为力。实测余弦
# （τ=0.75）：`Microservices ↔ 微服务架构` 0.570、`Kubernetes ↔ K8s` 0.582。
# 抽取器按提示词把简历里的 `微服务` 规范成 `微服务架构`，而 JD 里写的是
# `Microservices` —— 两边指同一件事，对齐却判不中。
#
# 在评测集上这**不影响排序**：所有候选人都缺同一条，名次不变，
# 所以 exp6 的 NDCG 0.9909 是真的。但在产品里，招聘官会看到
# 每一个候选人都标着「✗ Microservices」，而其中一半人简历里写着「微服务架构」——
# 这是可见的假阴性，会直接摧毁对这个功能的信任。
#
# 收录标准：**严格同义**。`Kafka` 与 `消息队列` 是产品与品类的关系，
# 不是同义，不收 —— 收了会把"用过 Kafka"错算成"懂消息队列"。
ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("微服务架构", "微服务", "Microservices", "microservice"),
    ("Kubernetes", "K8s", "k8s"),
    ("CI/CD", "持续集成", "持续交付", "持续集成与交付"),
    ("数据仓库", "数仓"),
    ("分布式一致性", "分布式事务"),
    ("高并发系统设计", "高并发"),
    ("数据可视化", "可视化"),
)

_ALIAS: dict[str, str] = {
    _norm(name): _norm(group[0]) for group in ALIAS_GROUPS for name in group
}


def canon(s: str) -> str:
    """把技能名映射到规范写法。不在表里就返回 `_norm` 的结果。"""
    n = _norm(s)
    return _ALIAS.get(n, n)


def align_hit(jd_skill: str, resume_skills: list[str], tau: float, vec: dict,
              use_alias: bool = False) -> bool:
    """JD 技能名是否被简历技能命中。

    三条路，按成本从低到高：
      1. **规范化后完全相等**（免费）—— `docker` vs `Docker` 这类只差大小写的，
         不该占用嵌入预算，也不该被阈值误杀。
      2. **别名表命中**（免费，`use_alias=True` 时启用）—— 见 `ALIAS_GROUPS`。
      3. **嵌入余弦 ≥ τ** —— 管剩下那些没被人工收录的别名。

    `use_alias` 默认 **False**：exp6 量的是"纯嵌入对齐"这个方法本身，
    它报出的 NDCG 必须对应代码里的默认行为，否则报告与实现就对不上了。
    产品路径（`SkillAligner`）显式传 True，并在界面上声明这一点。
    """
    j = _norm(jd_skill)
    cj = canon(jd_skill)
    for s in resume_skills:
        if _norm(s) == j:
            return True
        if use_alias and canon(s) == cj:
            return True
    if not vec or jd_skill not in vec:
        return False
    return any(float(vec[jd_skill] @ vec[s]) >= tau for s in resume_skills if s in vec)


def structured_score(must: list[str], nice: list[str], resume_skills: list[str],
                     tau: float, vec: dict, use_alias: bool = False) -> float:
    """must 覆盖为主，nice 覆盖为辅。

    `must` 一条都没覆盖时**不给任何 nice 分** —— 这是"缺 must 直接出局"的硬约束，
    对应评测集里 relevance=0 的「缺must」类。
    """
    if not must:
        mc = 0.0
    else:
        mc = sum(1 for m in must
                 if align_hit(m, resume_skills, tau, vec, use_alias)) / len(must)
    if mc == 0.0:
        return 0.0
    nc = (sum(1 for n in nice
              if align_hit(n, resume_skills, tau, vec, use_alias)) / len(nice)
          if nice else 0.0)
    return mc + 0.5 * nc


PER_SKILL_SYSTEM = """你在做岗位要求逐条比对。用户会给你一份简历和一组岗位要求技能。

对**每一条**要求，判断这位候选人是否真的具备：
- 简历里写了「精通/熟练使用/熟悉」→ 具备
- 简历里写「了解过/听说过」或完全没提 → 不具备
- 工作/项目经历里能看出用过 → 具备

输出 JSON：{"has": ["具备的技能1", "具备的技能2"]}
`has` 里只放**确实具备**的，原样照抄要求里的写法。只输出 JSON。"""


def llm_per_skill(jd_skills: list[str], resume: str, llm: LLM
                  ) -> tuple[list[str], dict]:
    q = ("【岗位要求】\n" + "\n".join(f"- {s}" for s in jd_skills)
         + f"\n\n【候选人简历】\n{resume}")
    before = llm.ledger.summary()["total_tokens"]
    r = llm.call([{"role": "user", "content": q}], system=PER_SKILL_SYSTEM,
                 tag="per_skill")
    after = llm.ledger.summary()["total_tokens"]
    try:
        has = [str(s).strip() for s in (r.json().get("has") or [])]
    except Exception:  # noqa: BLE001
        has = []
    return has, {"tokens": after - before, "latency_s": r.latency_s}


class SkillAligner:
    """带缓存的技能对齐器 —— **产品路径**。

    与 exp6 的差别只有一处：启用别名表（`use_alias=True`）。嵌入阈值 τ 与
    判定结构完全一致，所以它不是"另一套算法"，而是同一套算法多了一张词典。
    前端会把这件事写出来，不能让界面上的匹配率看起来等于论文里的 NDCG。

    为什么需要缓存：`align_hit` 每次都要对一对技能名算余弦，
    一个岗位 × 一千份简历 × 十几条要求 = 上万次查表。逐个 `encode` 会慢到不可用，
    而技能名的集合其实很小（几百个），一次编码反复查即可。
    """

    def __init__(self, embed=None, use_alias: bool = True):
        self._embed = embed
        self.use_alias = use_alias
        self._v: dict[str, object] = {}

    def _encoder(self):
        if self._embed is None:
            from .dense import embedder
            self._embed = embedder()
        return self._embed

    def vec(self, names) -> dict:
        """补齐缺失的技能名向量，返回全量缓存。"""
        todo = sorted({n for n in names if n and n not in self._v})
        if todo:
            vs = self._encoder().encode(todo, normalize_embeddings=True,
                                        show_progress_bar=False)
            self._v.update({n: v for n, v in zip(todo, vs)})
        return self._v

    def hit(self, jd_skill: str, resume_skills: list[str], tau: float) -> bool:
        return align_hit(jd_skill, resume_skills, tau,
                         self.vec([jd_skill, *resume_skills]), self.use_alias)

    def coverage(self, reqs: list[str], resume_skills: list[str], tau: float
                 ) -> tuple[list[str], list[str]]:
        """返回 (命中的要求, 未命中的要求)。用于把匹配结果**拆开给人看** ——
        HR 需要知道"差在哪一条"，只给一个分数是不可复核的。"""
        v = self.vec([*reqs, *resume_skills])
        hit = [r for r in reqs
               if align_hit(r, resume_skills, tau, v, self.use_alias)]
        miss = [r for r in reqs if r not in hit]
        return hit, miss

    def score(self, must: list[str], nice: list[str], resume_skills: list[str],
              tau: float) -> float:
        return structured_score(must, nice, resume_skills, tau,
                                self.vec([*must, *nice, *resume_skills]),
                                self.use_alias)
