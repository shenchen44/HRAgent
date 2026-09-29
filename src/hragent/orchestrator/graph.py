"""L2 编排层 —— 用 LangGraph 把指挥官、执行体、风控串成可中断、可续跑的状态机。

为什么要有这一层，而不是在 `agents/orchestrator.py` 里继续加 if-else：

1. **`agents/orchestrator.py` 不能改。** exp3 测的就是 `Orchestrator.run()` 的行为，
   改它等于让 exp3 的结果失效。所以这一层是**叠加**，不是替换 ——
   `Orchestrator` 负责"想什么"（路由、歧义、汇总），本层负责"按什么顺序走、在哪停"。

2. **风控必须嵌进控制流，而不是事后调用。** 原 `run()` 里 dispatch 与 synthesize 之间
   没有任何调用点，`Trace.verdicts` 永远是空的 —— L3 建好了也接不上。
   图结构让"每步后置校验"成为拓扑的一部分，而不是靠调用者记得去调。

3. **人工介入需要真正的暂停/恢复。** `interrupt()` + checkpointer 能做到
   "拦下来 → 人看了 → 从断点继续"，而不是"拦下来 → 整轮重跑"。
   这是 HITL 与"打日志然后重试"的本质区别。

4. **断点续跑。** 一次 dispatch 可能跑几个执行体、几十秒。checkpointer 让
   中断后可以从上一个完成的节点继续，不必重跑已完成的步骤。

图的拓扑：

    START → screen ─┬─ 请求级判据拒答 ─────────────────────→ refuse → END
                    ├─ 请求级判据升级 ─────────────────────→ handoff → hitl ⇢ END
                    └─ 放行 → route ─┬─ 需澄清 ───────────→ clarify → END
                                     ├─ 越界/盲区 → refuse → END
                                     └─ dispatch → review ─┬─ block → block → END
                                                           └─ synthesize → final_review
                                                                 ├─ block → END
                                                                 ├─ 不确定 → hitl
                                                                 └─ 通过 → END

`screen` 放在**最前面**而不是 route 之后：route 有两条短路（澄清、拒答），
判据若在其后会**被这两条短路绕过** —— 路由一旦把违规请求误判成"歧义"，
判据就再也看不到它。放在最前面，它面对的是未经任何组件加工的原始请求。
（详见 risk/screen.py 的模块注释。）

`hitl` 节点用 `interrupt()` 暂停；恢复时传 `Command(resume={"decision": ...})`。
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..agents.executors import REGISTRY, input_label, requires_evidence
from ..agents.orchestrator import Orchestrator
from ..risk.guards import Candidate, RiskAgent
from ..risk.screen import QueryScreen
from ..schema import Evidence, ExecResult, GuardVerdict, RouteDecision, Trace

# 「哪些执行体的输出按契约必须带证据」原先在这里硬编码成 `EVIDENCE_REQUIRED`。
# 已挪到生产端（各执行体类的 `requires_evidence` 属性，见 agents/executors.py）。
# 原因：这张名单有两个消费者（G1/不确定性门控、编排层的升级判据），
# 而它描述的是**执行体自己的契约**。放在消费端时，新增执行体没人会想起来改它，
# 缺省又恰好是"不要求证据"= 放行 —— 漏配会静默降级为无风控，不报错。

REFUSAL = "这个问题超出了我能处理的 HR 范围，建议转人工或咨询对应部门。"
BLOCKED = "该回答未通过风控校验，已拦截，请转人工复核。"
HANDOFF = "这个请求涉及需要人工判断的事项（可能是合规风险或需要审批的操作），已转交 HR 人工处理。"

# 请求级判据拒答（`screen`）的**用户可见文案按类别分开写**，不与 `REFUSAL` 共用。
#
# 两者是**不同的边界**：`REFUSAL` 说的是"这个问题我答不了"（**能力**边界），
# 这里说的是"这个请求我不做"（**政策**边界）。共用一句"超出了我能处理的 HR 范围"
# 会把政策拒答说成能力不足 —— 用户收到的信号是"换个说法再试"，
# 而正确的信号是"换说法也一样，别再试了"。对受保护属性筛选这类请求，
# 前者是有害的误导。
#
# 分类取自 `risk/screen.py` 的 category 取值；未识别的类别回落到通用政策文案。
REFUSAL_POLICY = "这个请求我不能执行，建议改为正常的 HR 业务问题，或转人工处理。"
REFUSAL_BY_CATEGORY = {
    "attribute": "这个请求我不能执行：它要求按受保护属性（性别、年龄、婚育、户籍等）"
                 "对候选人做筛选或评价。请改为按岗位要求筛选，或转人工处理。",
    "privacy":   "这个请求我不能执行：它索取特定个人的隐私信息或越权数据。"
                 "如确因工作需要，请走正式审批流程或转人工。",
    "injection": "这个请求我不能执行：它试图让我忽略既有规则或越权操作。"
                 "请提出正常的 HR 业务问题。",
    "off_topic": "这个问题不属于 HR 业务范围，我不能代为处理。",
}


class HRState(TypedDict, total=False):
    """图状态。全部用可序列化类型，checkpointer 才能落盘。"""

    query: str
    ctx: dict
    screen: dict
    route: dict
    results: list[dict]
    verdicts: list[dict]
    final_answer: str
    escalated: bool
    escalation_reason: str | None
    needs_review: bool
    blocked: bool
    block_reason: str | None
    human: dict | None
    latency_s: float
    n_exec_verdicts: int   # 执行体级裁决的条数，用于把两级风控分开呈现


# ---------------------------------------------------------------- 状态互转

def _cand(executor: str, query: str, r: ExecResult) -> Candidate:
    c = Candidate.from_exec_result(query, r,
                                   requires_evidence=requires_evidence(executor))
    c.artifacts["top_scores"] = _top_scores([r])
    return c


def _answer_confidence(results: list[ExecResult]) -> float:
    """答案级置信度：取**各执行体自评置信度的最小值**（短板原则）。

    为什么必须显式算（真实 bug）：`n_final_review` 原先手工构造 Candidate 时
    漏传 `confidence`，于是取 dataclass 默认值 0.0。后果是 `UncertaintyGate` 的
    置信度项 `0.4*(1-conf)` 对**全部 150 题恒等于 0.400** —— 风控分被钉在 0.4 以上，
    唯一还剩区分度的是检索项（权重仅 0.25）。实测后果：HITL 阈值 0.5 触发 26% 升级、
    精度 5.1%（基础率 2.7%），端到端行为正确率反掉 18pp。
    这与 `_top_scores()` 读错键是同一类毛病 —— **信号静默失效，不报错**，
    只能靠对着生产端核字段发现。

    为什么取 min 而不是 mean：汇总答案是**多个执行体断言的合取**，其可信度受
    最弱一环约束。取均值会让一个高置信执行体掩盖另一个低置信执行体 ——
    而那恰好是风控要抓的情况。

    只统计 `ok` 的执行体：失败结果不产出可引用的断言，其占位文本不进入答案。
    """
    ok = [r.confidence for r in results if r.ok]
    return min(ok) if ok else 0.0


def _top_scores(results: list[ExecResult]) -> list[float]:
    """从检索产物里取回分数，供 UncertaintyGate 使用。

    检索分是**弱信号**（P1 实测：盲区题与正常题分数分布严重重叠），
    但融合进风险分后仍有边际价值 —— 单独用不可靠，不等于没有信息。

    键名必须是 `top_scores`（executors.py 里 PolicyRAG 写的就是这个）。
    这里原先读的是 `retrieved_spans`，取不到、恒为空 ——
    结果是 UncertaintyGate 里 1/4 权重的检索信号**从来没生效过，且不报错**。
    这类"读错键导致信号静默失效"只能靠对着生产端核键名发现。
    """
    out: list[float] = []
    for r in results:
        for s in (r.artifacts or {}).get("top_scores") or []:
            try:
                out.append(float(s))
            except (TypeError, ValueError):
                continue
    return sorted(out, reverse=True)[:5]


def _allowed_refs(results: list[ExecResult]) -> list[str]:
    """答案级 G3 的白名单：汇总答案里允许出现的引用全集。

    **必须是「各执行体的检索集 ∪ 各执行体自己的证据 ref」**，不能只取检索集。
    实测教训（两条真实误报）：

      1. `HRDataSQL` 的证据 ref 形如 `hr.db#avg_salary` —— 它**不做检索**，
         压根没有"检索集"这回事。只取检索集会让「SQL + 制度」的混合问答
         被判成引用错配。
      2. `OnboardFlow` 曾经只写 `retrieved` 不写 `retrieved_refs`（已修），
         于是它合法引用的制度片段进不了白名单。

    为什么把"证据 ref"也算作白名单是**安全**的：证据是执行体/工具产出的，
    不是汇总模型写出来的 —— 汇总模型无法凭空往里塞一个 ref。
    （注：G3 校验的是**证据列表**里的 ref，不解析答案正文里的引用标记；
    正文引用标记是另一个待测的判据，见 results/exp7_guards.md「已知残余风险」。）
    """
    out: list[str] = []
    for r in results:
        out += list((r.artifacts or {}).get("retrieved_refs") or [])
        out += [e.ref for e in r.evidence]
    return list(dict.fromkeys(out))


def _verdicts_to_dicts(vs: list[GuardVerdict]) -> list[dict]:
    return [asdict(v) for v in vs]


def _results_to_dicts(rs: list[ExecResult]) -> list[dict]:
    return [asdict(r) for r in rs]


def _results_from_dicts(ds: list[dict]) -> list[ExecResult]:
    out = []
    for d in ds or []:
        d = dict(d)
        d["evidence"] = [Evidence(**e) for e in (d.get("evidence") or [])]
        out.append(ExecResult(**d))
    return out


# ---------------------------------------------------------------- 建图

def build_graph(orch: Orchestrator | None = None, risk: RiskAgent | None = None,
                checkpointer: Any | None = None, hitl: bool = True,
                screen: QueryScreen | None = None, ask_inputs: bool = False):
    """编译编排图。

    hitl=False 时不确定性只记录不暂停 —— exp9 做消融要能关掉人工环节，
    否则无法量化"人工介入"本身的收益。

    `screen` 与 `ask_inputs` 缺省都是**关闭**：这样 exp9 里 B0~B3 的行为与开销
    完全不变，历史结果仍然可比，新层作为独立的一臂（B4/B5）叠加即可。
    """
    orch = orch or Orchestrator()
    risk = risk or RiskAgent()
    ckpt = checkpointer if checkpointer is not None else MemorySaver()
    scr = screen or QueryScreen(enabled=False)
    # **判据必须复用编排层的 LLM 实例**，否则它自己 new 一个 LLM，
    # 消耗记在另一个账本上 —— 成本会被静默少报（硬性约定 5）。
    # 与"信号静默失效"是同一类毛病：不报错，只是数字不对。
    # 用 getattr：冒烟测试里的 StubOrch 没有 llm 属性，判据此时是关闭的。
    if scr.llm is None:
        scr.llm = getattr(orch, "llm", None)

    # ---- 节点
    def n_screen(state: HRState) -> dict:
        """请求级判据。只看原始请求，不看任何回答。"""
        t0 = time.time()
        v = scr.check(state["query"])
        return {"screen": {"action": v.action, "category": v.category,
                           "reason": v.reason, "tokens": v.tokens},
                "latency_s": time.time() - t0}

    def n_route(state: HRState) -> dict:
        t0 = time.time()
        d = orch.route(state["query"])
        return {"route": asdict(d),
                "latency_s": state.get("latency_s", 0.0) + time.time() - t0}

    def n_refuse(state: HRState) -> dict:
        s = state.get("screen") or {}
        if s.get("action") == "refuse":
            # 请求级判据拒答：理由要写清是哪一类，否则事后归因时分不清
            # "路由判成越界"与"判据判成越界"—— 这两者的修法完全不同。
            # 给用户的文案也按类别分开：政策拒答不能说成能力不足。
            cat = s.get("category")
            return {"final_answer": REFUSAL_BY_CATEGORY.get(cat, REFUSAL_POLICY),
                    "escalated": True,
                    "escalation_reason":
                        f"请求级判据拒答（{cat}）：{s.get('reason')}"}
        return {"final_answer": REFUSAL, "escalated": True,
                "escalation_reason": "无匹配执行体（越界或制度盲区）"}

    def n_handoff(state: HRState) -> dict:
        """请求级判据判定需人工 —— 直接进 HITL，不派任何执行体。

        为什么是"转人工"而不是"拒答"：这一档（代理变量筛选、越权处置）
        **不是一眼违法**，该由人判断；一律拒答会把"灰区"和"明显违规"混成一档。
        """
        s = state.get("screen") or {}
        return {"final_answer": HANDOFF, "escalated": True,
                "escalation_reason":
                    f"请求级判据转人工（{s.get('category')}）：{s.get('reason')}"}

    def n_clarify(state: HRState) -> dict:
        r = state.get("route") or {}
        return {"final_answer": r.get("clarification") or "能否补充更多信息？",
                "escalated": False}

    def n_dispatch(state: HRState) -> dict:
        route = RouteDecision(**state["route"])
        t0 = time.time()
        results = orch.dispatch(state["query"], route, state.get("ctx"))
        return {"results": _results_to_dicts(results),
                "latency_s": state.get("latency_s", 0.0) + time.time() - t0}

    def n_ask_inputs(state: HRState) -> dict:
        """缺输入 → 问用户要，而不是升级人工。

        实测场景（exp9 的 4 例）：用户说"筛一下简历""对比这两份简历"，
        却没附任何材料。原实现走到 `synthesize` → 全部执行体失败 → 升级人工。
        但**人工手里同样没有这两样东西** —— 这次升级是纯浪费
        （实测 HITL 的秒/题比反问高 46%）。

        澄清问题按缺什么拼，不写死："请提供候选人简历" 比 "能否补充信息" 有用。

        为什么要回写 `route.clarification`：本层的 `action_of` 判据是
        "route.needs_clarification → clarify"。系统**事实上**澄清了，
        Trace 就该如实反映这个结果 —— 否则同一次澄清会被记成 answer，
        指标就假了。（`clarification` 在 DESIGN §3 的定义本就是"需要澄清时的问题文本"。）
        """
        rs = _results_from_dicts(state.get("results") or [])
        miss: list[str] = []
        for r in rs:
            for k in r.missing_inputs:
                if k not in miss:
                    miss.append(k)
        text = "为了完成这个请求，还需要您提供：" + "、".join(input_label(k) for k in miss) + "。"
        route = dict(state.get("route") or {})
        route["clarification"] = text
        route["ambiguity"] = max(float(route.get("ambiguity") or 0.0), 0.6)
        return {"route": route, "final_answer": text, "escalated": False}

    def n_review(state: HRState) -> dict:
        """执行体级风控：逐个结果过闸门。

        **这里只判硬违规（block），不判不确定性。**
        实测教训：最初让 UncertaintyGate 也在这里触发，结果是 hitl 在 synthesize
        **之前**暂停 —— 人工被问了一个还没生成的答案，resume 回来 final_answer 是空的。
        不确定性是关于"最终答案可不可信"的，只能在答案生成之后判。
        """
        rs = _results_from_dicts(state.get("results") or [])
        vs: list[GuardVerdict] = []
        for r in rs:
            if not r.ok:
                continue
            vs += risk.review(_cand(r.executor, state["query"], r))
        blocked = risk.blocked(vs)
        first = risk.first_block(vs)
        return {"verdicts": _verdicts_to_dicts(vs), "blocked": blocked,
                "block_reason": first.reason if first else None,
                "n_exec_verdicts": len(vs)}

    def n_synthesize(state: HRState) -> dict:
        rs = _results_from_dicts(state.get("results") or [])
        answer, esc, why = orch.synthesize(state["query"], rs)
        return {"final_answer": answer, "escalated": esc, "escalation_reason": why}

    def n_final_review(state: HRState) -> dict:
        """答案级风控：汇总后的文本是**新生成的**，可能引入新的违规。

        证据与引用**必须从执行体结果继承下来**，否则汇总答案里的数字与引用
        完全不受校验 —— 那等于风控在最后一步失效。
        只有生成型输出（无任何执行体按契约要求证据）才允许证据为空。
        """
        rs = _results_from_dicts(state.get("results") or [])
        evidence = [e for r in rs for e in r.evidence]
        refs = _allowed_refs(rs)
        needs = any(requires_evidence(r.executor) for r in rs)

        c = Candidate(query=state["query"], answer=state.get("final_answer", ""),
                      evidence=evidence, retrieved_refs=refs,
                      requires_evidence=needs,
                      confidence=_answer_confidence(rs),
                      artifacts={"top_scores": _top_scores(rs)})
        vs = risk.review(c)
        blocked = risk.blocked(vs)
        first = risk.first_block(vs)
        warn = next((v for v in vs if v.severity == "warn" and not v.passed), None)
        # 注意：必须写 `warn is not None`，不能写 `bool(warn)` 或 `if warn`。
        # GuardVerdict.__bool__ 返回的是 self.passed，所以 `bool(warn)` 的含义是
        # "这条 warn 通过了" —— 恒为 False，会让升级逻辑静默失效。
        has_warn = warn is not None
        return {"verdicts": (state.get("verdicts") or []) + _verdicts_to_dicts(vs),
                "blocked": blocked or state.get("blocked", False),
                "block_reason": (first.reason if first else state.get("block_reason")),
                # 不确定性门控的 warn **只置 needs_review，不置 escalated**。
                # 早先的写法是 `escalated or has_warn`，那会让一道精准度只有 0.43 的
                # 弱信号把请求送进 HITL —— 叫来的人一多半没事干，且
                # "答案可能不可信"与"请求需人工处置"被混成同一个字段，事后无法归因。
                # 送 HITL 的强规则仍然在：`Orchestrator.synthesize` 的 confidence < 0.3。
                "needs_review": has_warn,
                "escalation_reason": (state.get("escalation_reason")
                                      or (warn.reason if has_warn else None))}

    def n_block(state: HRState) -> dict:
        return {"final_answer": BLOCKED, "escalated": True}

    def n_hitl(state: HRState) -> dict:
        """人工介入。`interrupt()` 在这里暂停整个图，等人给决定。"""
        if not hitl:
            return {"human": {"decision": "auto_escalate", "note": "HITL 已关闭"}}
        decision = interrupt({
            "query": state.get("query"),
            "answer": state.get("final_answer", ""),
            "reason": state.get("escalation_reason") or state.get("block_reason"),
            "verdicts": state.get("verdicts") or [],
        })
        return {"human": decision}

    # ---- 条件边
    def after_screen(state: HRState) -> str:
        a = (state.get("screen") or {}).get("action") or "ok"
        return a if a in ("refuse", "escalate") else "ok"

    def after_route(state: HRState) -> str:
        r = state.get("route") or {}
        if r.get("clarification"):
            return "clarify"
        if not r.get("executors"):
            return "refuse"
        return "dispatch"

    def after_dispatch(state: HRState) -> str:
        """全部执行体都因**缺输入**失败 → 问用户，不升级人工。

        条件刻意收得很紧：`全部失败` 且 `每个失败都是缺输入`。
        · 有一个成功 → 正常汇总（缺的那个丢掉即可，synthesize 已处理）。
        · 有失败但不是缺输入 → 那是执行体自己的问题，交给 review/synthesize。
        放宽任何一条都会把"执行体报错"也变成反问用户，那是拿澄清掩盖故障。
        """
        if not ask_inputs:
            return "review"
        rs = state.get("results") or []
        if not rs or any(r.get("ok") for r in rs):
            return "review"
        if not all(r.get("missing_inputs") for r in rs):
            return "review"
        return "ask_inputs"

    def after_review(state: HRState) -> str:
        # 执行体级只有硬违规才中断；不确定性留给答案生成之后判
        return "block" if state.get("blocked") else "synthesize"

    def after_final(state: HRState) -> str:
        if state.get("blocked"):
            return "block"
        if state.get("escalated"):
            return "hitl"
        return END

    g = StateGraph(HRState)
    for name, fn in (("screen", n_screen), ("route", n_route), ("refuse", n_refuse),
                     ("clarify", n_clarify), ("handoff", n_handoff),
                     ("dispatch", n_dispatch), ("ask_inputs", n_ask_inputs),
                     ("review", n_review),
                     ("synthesize", n_synthesize), ("final_review", n_final_review),
                     ("block", n_block), ("hitl", n_hitl)):
        g.add_node(name, fn)

    g.add_edge(START, "screen")
    g.add_conditional_edges("screen", after_screen,
                            {"ok": "route", "refuse": "refuse", "escalate": "handoff"})
    g.add_conditional_edges("route", after_route,
                            {"clarify": "clarify", "refuse": "refuse",
                             "dispatch": "dispatch"})
    g.add_conditional_edges("dispatch", after_dispatch,
                            {"ask_inputs": "ask_inputs", "review": "review"})
    g.add_conditional_edges("review", after_review,
                            {"block": "block", "synthesize": "synthesize"})
    g.add_edge("synthesize", "final_review")
    g.add_conditional_edges("final_review", after_final,
                            {"block": "block", "hitl": "hitl", END: END})
    g.add_edge("clarify", END)
    g.add_edge("ask_inputs", END)
    g.add_edge("refuse", END)
    g.add_edge("block", END)
    g.add_edge("handoff", "hitl")
    g.add_edge("hitl", END)

    return g.compile(checkpointer=ckpt)


# ---------------------------------------------------------------- 便捷入口

class HRGraph:
    """图的薄封装：对外只暴露 `run()`，返回与旧编排层一致的 Trace。"""

    def __init__(self, orch: Orchestrator | None = None, risk: RiskAgent | None = None,
                 hitl: bool = True, checkpointer: Any | None = None,
                 store: Any | None = None, screen: QueryScreen | None = None,
                 ask_inputs: bool = False):
        self.orch = orch or Orchestrator()
        self.graph = build_graph(self.orch, risk, checkpointer, hitl=hitl,
                                 screen=screen, ask_inputs=ask_inputs)
        self.store = store          # TraceStore | None

    def _cost(self, before: tuple[int, int]) -> dict:
        """取本轮 LLM 消耗。成本必须与准确率并列（硬性约定 5）。"""
        led = getattr(self.orch, "llm", None)
        led = getattr(led, "ledger", None)
        if led is None:
            return {"tokens": 0}
        return {"tokens": (led.input_tokens + led.output_tokens) - before[0],
                "calls": len(led.per_call) - before[1]}

    def _ledger_snapshot(self) -> tuple[int, int]:
        led = getattr(getattr(self.orch, "llm", None), "ledger", None)
        if led is None:
            return (0, 0)
        return (led.input_tokens + led.output_tokens, len(led.per_call))

    def run(self, query: str, ctx: dict | None = None, thread_id: str | None = None,
            **meta) -> tuple[Trace, dict]:
        """跑一轮。返回 (Trace, 图状态)。

        若图在 hitl 处暂停，Trace.escalated=True 且状态里带 `__interrupt__`，
        用 `resume(thread_id, decision)` 从断点继续。
        """
        before = self._ledger_snapshot()
        cfg = {"configurable": {"thread_id": thread_id or f"t{int(time.time() * 1000)}"}}
        st = self.graph.invoke({"query": query, "ctx": ctx or {}}, cfg)
        tr = self._to_trace(query, st)
        tr.tokens = self._cost(before)["tokens"]
        if self.store is not None:
            self.store.append(tr, thread_id=cfg["configurable"]["thread_id"],
                              **self._cost(before), **meta)
        return tr, st

    def resume(self, thread_id: str, decision: dict) -> tuple[Trace, dict]:
        before = self._ledger_snapshot()
        cfg = {"configurable": {"thread_id": thread_id}}
        st = self.graph.invoke(Command(resume=decision), cfg)
        tr = self._to_trace(st.get("query", ""), st)
        tr.tokens = self._cost(before)["tokens"]
        if self.store is not None:
            self.store.append(tr, thread_id=thread_id, resumed=True, **self._cost(before))
        return tr, st

    @staticmethod
    def _to_trace(query: str, st: dict) -> Trace:
        tr = Trace(query=query)
        if st.get("route"):
            tr.route = RouteDecision(**st["route"])
        tr.screen = st.get("screen")
        tr.results = _results_from_dicts(st.get("results") or [])
        tr.verdicts = [GuardVerdict(**v) for v in (st.get("verdicts") or [])]
        tr.final_answer = st.get("final_answer", "")
        tr.escalated = bool(st.get("escalated"))
        tr.escalation_reason = st.get("escalation_reason")
        tr.needs_review = bool(st.get("needs_review"))
        tr.latency_s = float(st.get("latency_s") or 0.0)
        tr.tokens = 0
        return tr


if __name__ == "__main__":
    import sys
    g = HRGraph()
    for q in sys.argv[1:] or ["研发中心现在有多少在职员工？"]:
        tr, st = g.run(q)
        print(f"\n问题: {q}")
        print(f"  执行体={tr.route.executors if tr.route else []} "
              f"歧义={tr.route.ambiguity if tr.route else 0:.2f}")
        for v in tr.verdicts:
            mark = "✅" if v.passed else ("⛔" if v.severity == "block" else "⚠️")
            print(f"  {mark} [{v.guard}] {v.reason[:80]}")
        print(f"  升级={tr.escalated} 理由={tr.escalation_reason}")
        print(f"  最终: {tr.final_answer[:200]}")
