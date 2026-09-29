"""模块间数据契约（DESIGN.md §3）。

所有跨模块通信只走这里定义的结构，避免后期接口漂移。
改这里 = 改契约，必须同步更新 DESIGN.md。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Severity = Literal["pass", "warn", "block"]
EvidenceKind = Literal["sql_result", "doc_span", "rule"]


@dataclass
class Evidence:
    """一条可追溯的证据。防幻觉的唯一自动化抓手。"""

    kind: EvidenceKind
    ref: str                        # "employees.status" | "annual_leave.md#L12-L18"
    value: str
    source_query: str | None = None  # SQL 或检索 query，供追溯

    def __str__(self) -> str:
        return f"[{self.kind}] {self.ref} = {self.value}"


@dataclass
class ExecResult:
    """执行体统一返回结构。"""

    executor: str
    ok: bool
    answer: str
    evidence: list[Evidence] = field(default_factory=list)
    confidence: float = 0.0          # 自评置信度（弱信号，实测判别力有限）
    artifacts: dict = field(default_factory=dict)  # sql / retrieved_spans / match_scores
    latency_s: float = 0.0
    tokens: int = 0
    error: str | None = None
    # 契约要求的输入没拿到（如 RecruitMatch 缺简历）。**由执行体自己声明**，
    # 不让编排层去猜错误文本 —— 猜文本正是"读错键"那类静默失效的温床。
    # 编排层据此把"我做不了"改判成"请你补材料"：前者要占用人工，后者几乎免费。
    missing_inputs: list[str] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """有实质答案就必须有证据。"""
        return bool(self.evidence) or not self.answer.strip()


@dataclass
class GuardVerdict:
    """单道闸门的裁决。"""

    guard: str                       # truncation | grounding | compliance | uncertainty
    passed: bool
    severity: Severity = "pass"
    reason: str = ""
    evidence_refs: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


@dataclass
class RouteDecision:
    """指挥官的决策。歧义时 clarification 非空，且不应调用任何执行体。"""

    intents: list[str] = field(default_factory=list)
    executors: list[str] = field(default_factory=list)
    slots: dict = field(default_factory=dict)
    ambiguity: float = 0.0
    clarification: str | None = None

    @property
    def needs_clarification(self) -> bool:
        return bool(self.clarification)


@dataclass
class Trace:
    """单轮请求的完整轨迹，用于 badcase 归因与断点续跑。"""

    query: str
    route: RouteDecision | None = None
    results: list[ExecResult] = field(default_factory=list)
    verdicts: list[GuardVerdict] = field(default_factory=list)
    final_answer: str = ""
    escalated: bool = False
    escalation_reason: str | None = None
    # 「答案可能不可信，建议人工**复核**」—— 与 `escalated`（转人工**处置**）是两回事。
    # 分开的理由是 exp8b 实测：不确定性门控即便校准到最优工作点，
    # 升级精准度也只有 0.43，叫来的人一多半没事干。弱信号该提示，不该叫人。
    # 合成一个字段会让"这道题为什么升级"无法归因 —— 而这两类的修法完全不同。
    needs_review: bool = False
    # 请求级合规判据的裁决（risk/screen.py）。放这里是为了 badcase 归因：
    # 只看 action 无法区分"路由判成越界"与"判据判成越界"，而这两者的修法完全不同。
    screen: dict | None = None
    # 成本必须与准确率并列报（DESIGN.md 硬性约定 5）。
    # 单看准确率会掩盖成本失控 —— exp3 的教训：分层路由准确率略高，代价是 2 倍 token。
    latency_s: float = 0.0
    tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
