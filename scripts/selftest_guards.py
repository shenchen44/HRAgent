"""风控闸门自测 —— 每一条都对应一个**真实发生过的 bug**，不是凭空写的用例。

为什么这些必须是回归测试而不是一次性验证：
  下面 6 条里有 5 条是"离线指标全绿、生产静默出错"的类型 ——
  它们不会让任何现有测试变红，只会让系统在真实流量上悄悄失效。
  没有回归测试，下次改判据时同样的坑会再踩一遍。

跑法:
  .venv/bin/python scripts/selftest_guards.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hragent.risk.guards import (Candidate, ComplianceGuard,  # noqa: E402
                                 GroundingGuard, RiskAgent, TruncationGuard,
                                 UncertaintyGate, _breakdown, _restatement)
from hragent.schema import Evidence, GuardVerdict  # noqa: E402

N = 0
FAILED: list[str] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    global N
    N += 1
    if cond:
        print(f"  ✅ {name}")
    else:
        FAILED.append(name)
        print(f"  ❌ {name}" + (f"  — {detail}" if detail else ""))


def ev(value: str, ref: str = "hr.db#m") -> Evidence:
    return Evidence(kind="sql_result", ref=ref, value=value)


def cand(answer: str, evidence=None, query="问", refs=None, **kw) -> Candidate:
    return Candidate(query=query, answer=answer, evidence=evidence or [],
                     retrieved_refs=refs if refs is not None else
                     [e.ref for e in (evidence or [])], **kw)


# ---------------------------------------------------------------- ① 真值陷阱
def t_verdict_truthiness() -> None:
    """bug：`bool(warn)` 的含义是"这条裁决通过了"，不是"裁决存在"。

    后果：升级逻辑静默失效 —— escalated 恒为 False，HITL 永不触发，且不报错。
    """
    print("\n[1] GuardVerdict 真值语义（曾导致 HITL 静默失效）")
    warn = GuardVerdict(guard="uncertainty", passed=False, severity="warn", reason="x")
    passed = GuardVerdict(guard="grounding", passed=True)
    ok("bool(未通过裁决) is False", bool(warn) is False)
    ok("bool(通过裁决) is True", bool(passed) is True)
    ok("判存在性必须用 `is not None`", (warn is not None) is True)
    ok("因此 `if warn:` 与 `if warn is not None:` 语义不同",
       bool(warn) != (warn is not None))


# ---------------------------------------------------------------- ② G2a 长度
def t_restatement_length_guard() -> None:
    """bug：长答案大量引用制度原文时骨架相似度仍 ≥0.75，派生数字被当成篡改。

    真实误报 7/20 → 修后 2/20。这里固化长度差比这一条判据。
    """
    print("\n[2] G2a 复述判定：长度差比必须同时满足（修真实误报）")
    # 复述型篡改：同长度换一个数 → 应判为复述
    ok("逐字复述型（同长度）判为复述",
       _restatement("查询结果为 74.03。", "查询结果为 106.0。") is True)
    # 引用型长答案：证据 120 字、答案 486 字 → 不应判为复述
    long_ans = "关于质量部的平均月薪：制度中没有相关规定。" * 12
    short_ev = "《薪酬与福利管理规定》2. 社会保险 · 2.2 缴费基数按员工上年度月平均工资核定。"
    ok("引用型长答案不判为复述（长度差比过大）",
       _restatement(long_ans, short_ev) is False)
    ok("空串不判为复述", _restatement("", "x") is False)


def t_g2a_long_answer_no_fp() -> None:
    """bug：同上，从闸门层面验证长答案 + 派生数字不再被拦。"""
    print("\n[3] G2a 端到端：长引用答案里的派生数字不应被当成篡改")
    g = GroundingGuard()
    e = ev("《考勤与加班管理规定》3. 加班 · 3.4 加班补偿方式：调休或加班费。"
           "工作日加班按 150% 支付。")
    ans = ("根据制度，加班补偿方式为二选一：调休或加班费。"
           "如果按 21.75 天折算，月薪 10000 元对应时薪约 57.5 元，"
           "工作日加班时薪约 86.2 元。具体金额需结合实际情况计算。")
    v = g.check(cand(ans, [e]))
    ok("派生数字不被误判为篡改", v.passed, v.reason)


# ---------------------------------------------------------------- ④ G2b 散文
def t_breakdown_rejects_prose() -> None:
    """bug：`_breakdown` 把散文解析成"标签: 数字"，与证据比对后误判。"""
    print("\n[4] G2b 结构化解析：散文不得被解析成条目（修真实误报）")
    bd = _breakdown("- **调休**：按 1:1 折算调休时长，须在 6 个月内使用，逾期作废")
    ok("散文句不被解析成结构化条目", bd == {}, f"实际解析出 {bd}")
    bd2 = _breakdown("关于质量部的平均月薪：**制度中没有相关规定**，"
                     "检索到的片段未涉及统计或数据 [compensation_policy#2.2]")
    ok("含引用的散文句不被解析", bd2 == {}, f"实际解析出 {bd2}")
    # 真结构化列表必须仍能解析
    bd3 = _breakdown("None 2；P4 4；P5 2")
    ok("真结构化列表仍能解析", bd3 == {"None": ["2"], "P4": ["4"], "P5": ["2"]},
       f"实际 {bd3}")
    bd4 = _breakdown("法务部，10，6，40.0")
    ok("多值结构化条目仍能解析", bd4 == {"法务部": ["10", "6", "40"]}, f"实际 {bd4}")


def t_g2b_internal_tamper() -> None:
    """核心能力：列表内部被替换的值必须抓到（RISK0056 的真实模式）。"""
    print("\n[5] G2b 端到端：列表内部篡改必须抓到")
    g = GroundingGuard()
    e = ev("None 2；P4 4；P5 2；P6 6；P7 1；P8 1。")
    # 篡改 None 2 → 1，而 1 在 P7/P8 里出现过 —— 集合判据必然漏放
    v = g.check(cand("None 1；P4 4；P5 2；P6 6；P7 1；P8 1。", [e]))
    ok("列表内部替换被抓到（G2a 抓不到、G2b 抓到）", not v.passed, v.reason[:60])
    # 未篡改必须放行
    v2 = g.check(cand("None 2；P4 4；P5 2；P6 6；P7 1；P8 1。", [e]))
    ok("未篡改的同一列表放行", v2.passed, v2.reason)


# ---------------------------------------------------------------- ⑥ G1 定界
def t_g1_scoping() -> None:
    """bug：G1 不分执行体会把面试题生成、匹配建议这类输出全部误拦。"""
    print("\n[6] G1 定界：生成型输出（无证据）不得被拦")
    g = GroundingGuard()
    gen = "1. 请描述一次你主导的性能优化，瓶颈如何定位？\n2. 讲一个你设计过的方案。"
    ok("生成型输出（requires_evidence=False）放行",
       g.check(cand(gen, requires_evidence=False)).passed)
    ok("数据类断言无证据（requires_evidence=True）拦截",
       not g.check(cand(gen, requires_evidence=True)).passed)
    ok("空话不算实质断言",
       g.check(cand("没有相关规定。", requires_evidence=True)).passed)


# ---------------------------------------------------------------- ⑦ 禁令豁免
def t_prohibition_exemption() -> None:
    """bug：讲合规的文本长得像违规文本，纯词表必然误拦。"""
    print("\n[7] 合规闸门：引用禁令不得被当成歧视")
    g = ComplianceGuard()
    ok("「招聘启事不得包含限男性、限女性」放行",
       g.check(cand("招聘启事不得包含限男性、限女性、年龄限制等歧视性条款。")).passed)
    ok("「员工生育享受产假 158 天」放行",
       g.check(cand("女员工生育享受产假 158 天，配偶享受陪产假 15 天。")).passed)
    # 真歧视必须仍抓到
    for text, tag in (("这位候选人是女性，建议不推进。", "显式性别"),
                      ("已婚未育，建议排在后面。", "婚育"),
                      ("年龄偏大，建议淘汰。", "年龄")):
        ok(f"真歧视仍抓到（{tag}）",
           not g.check(cand(text)).passed)


def t_compliance_blindspot_documented() -> None:
    """已知盲区必须**如实失败**，不能因为难看就放宽判据。"""
    print("\n[8] 已知盲区（预期漏放，固化以防被「修好」成误报）")
    g = ComplianceGuard()
    v = g.check(cand("建议优先考虑应届生或工作 3 年以内的候选人。"))
    ok("代理变量歧视仍漏放（词表判据的结构性上限，已知）", v.passed)


# ---------------------------------------------------------------- ⑨ 其余
def t_truncation() -> None:
    print("\n[9] 截断闸门")
    g = TruncationGuard()
    ok("stop_reason=max_tokens 且正文为空 → 拦",
       not g.check(cand("", artifacts={"stop_reason": "max_tokens",
                                       "thinking_len": 4764})).passed)
    ok("空输出 → 拦", not g.check(cand("")).passed)
    ok("方括号不配对 → 拦", not g.check(cand("结果 [1, 2, 3")).passed)
    ok("正常输出 → 放行", g.check(cand("研发中心在职 96 人。")).passed)


def t_uncertainty_signal() -> None:
    """bug：`_top_scores` 读错键导致检索信号恒为空、静默失效。"""
    print("\n[10] 不确定性门控：检索分信号必须真的进得来")
    from hragent.orchestrator.graph import _top_scores
    from hragent.schema import ExecResult
    r = ExecResult(executor="PolicyRAG", ok=True, answer="x",
                   artifacts={"top_scores": [0.91, 0.77]})
    ok("从 artifacts.top_scores 取到分数（原读 retrieved_spans，恒为空）",
       _top_scores([r]) == [0.91, 0.77], f"实际 {_top_scores([r])}")
    g = UncertaintyGate(threshold=0.5)
    low = cand("x", [ev("v")], confidence=0.9, artifacts={"top_scores": [0.95]})
    high = cand("x", [], confidence=0.1, artifacts={"top_scores": [0.10]})
    ok("高置信+高分 → 风险低", g.risk_score(low) < g.risk_score(high),
       f"{g.risk_score(low):.2f} vs {g.risk_score(high):.2f}")


def t_risk_agent_switches() -> None:
    print("\n[11] RiskAgent 开关（exp9 消融依赖）")
    a = RiskAgent(enabled=("truncation",))
    ok("只开一道时仅一道生效", [x.name for x in a.guards] == ["truncation"])
    full = RiskAgent(enabled=("truncation", "grounding", "compliance", "uncertainty"))
    ok("四道全开", len(full.guards) == 4)
    vs = full.review(cand("研发中心在职 96 人。", [ev("96")]))
    ok("正常输入不被拦", not full.blocked(vs))


def t_answer_level_ref_whitelist() -> None:
    """bug：答案级 G3 的白名单只取「检索集」，漏了「执行体自产证据」。

    后果（两条真实误报，真实流量上才暴露）：
      · HRDataSQL 的证据 ref 是 `hr.db#avg_salary`，它不做检索 → 被判引用错配
      · OnboardFlow 曾只写 `retrieved` 不写 `retrieved_refs` → 同样进不了白名单
    离线指标对此**完全无感**：故障注入集的证据是逐条构造的，不走这条并集路径。
    """
    print("\n[12] 答案级引用白名单：跨执行体并集口径")
    from hragent.orchestrator.graph import _allowed_refs
    from hragent.schema import ExecResult
    sql = ExecResult(executor="HRDataSQL", ok=True, answer="a",
                     evidence=[ev("32000", "hr.db#avg_salary")], artifacts={})
    rag = ExecResult(executor="PolicyRAG", ok=True, answer="b",
                     evidence=[ev("x", "attendance_policy#3.4")],
                     artifacts={"retrieved_refs": ["attendance_policy#3.4",
                                                   "leave_policy#5.1"]})
    ob = ExecResult(executor="OnboardFlow", ok=True, answer="c",
                    evidence=[ev("y", "employee_handbook#2.2")],
                    artifacts={"retrieved_refs": ["employee_handbook#2.2"]})
    refs = _allowed_refs([sql, rag, ob])
    ok("SQL 证据 ref 进白名单（不做检索的执行体）", "hr.db#avg_salary" in refs)
    ok("OnboardFlow 证据 ref 进白名单", "employee_handbook#2.2" in refs)
    ok("检索集里的未引用片段也进白名单（弱信号，但确实检索到了）",
       "leave_policy#5.1" in refs)
    ok("白名单去重", len(refs) == len(set(refs)))

    # G3 的牙齿必须还在：证据指向了白名单之外的片段仍要被抓
    g = GroundingGuard()
    bad = cand("赔偿标准见 [compensation_policy#9.9]。",
               [ev("v", "compensation_policy#9.9")], refs=refs)
    v = g.check(bad)
    ok("证据指向白名单外的片段仍被拦截（G3 未被修废）",
       not v.passed and "compensation_policy#9.9" in v.reason, v.reason[:60])


def t_retrieval_executors_publish_refs() -> None:
    """生产端契约：凡做检索的执行体，必须公布 `retrieved_refs`。

    这条用源码结构断言，因为它是**生产者/消费者之间的字段契约** ——
    漏写不会报错，只会让消费者侧的判据静默失效（OnboardFlow 就这样躲过了
    自己的执行体级 G3：`if c.retrieved_refs:` 取不到就整段跳过）。
    """
    print("\n[13] 生产端契约：做检索的执行体必须公布 retrieved_refs")
    import ast
    src_path = (Path(__file__).resolve().parents[1]
                / "src/hragent/agents/executors.py")
    src = src_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    checked = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        seg = ast.get_source_segment(src, node) or ""
        if not any(isinstance(c, ast.Call)
                   and isinstance(c.func, ast.Attribute) and c.func.attr == "search"
                   for c in ast.walk(node)):
            continue
        checked.append(node.name)
        ok(f"{node.name} 公布了 retrieved_refs", "retrieved_refs" in seg)
    ok("确实检出了做检索的执行体（断言本身有效）", len(checked) >= 2,
       f"只找到 {checked}")


def t_retrieval_term_actually_varies() -> None:
    """bug：`1 - min(1, top)` 在 BM25 分上**恒等于 0**，检索项从未进过风险分。

    第一版注释写的是"归一化到 [0,1]（bge 余弦区间）"，但 `retrieval.index()` 返回
    BM25 分（实测 94 题最高分 min 6.58 / 中位 27.46 / max 99.82），**无上界**。
    于是检索项对 94/94 条样本贡献恰好 0 —— 静默失效，不报错、指标不变。
    与 `_top_scores()` 读错键同类。
    """
    print("\n[14] 检索分归一化：BM25 无上界，检索项必须真的变化")
    from hragent.risk.guards import _BM25_MID, _retrieval_conf
    ok("低分与高分映射到不同置信度（旧写法两者都是 0）",
       _retrieval_conf(6.58) < _retrieval_conf(27.46) < _retrieval_conf(99.82))
    ok("映射严格落在 [0,1)", 0.0 <= _retrieval_conf(6.58)
       and _retrieval_conf(99.82) < 1.0)
    ok("中位数分映射到 0.5 附近（饱和中点定义）",
       abs(_retrieval_conf(_BM25_MID) - 0.5) < 1e-9)
    ok("0 分与负分都映射到 0", _retrieval_conf(0.0) == 0.0
       and _retrieval_conf(-3.0) == 0.0)

    g = UncertaintyGate()
    def score(top: float) -> float:
        return g.risk_score(cand("x", [ev("v")], confidence=0.9,
                                 artifacts={"top_scores": [top]}))
    ok("端到端：检索分高低会改变风险分",
       score(6.58) > score(99.82) + 0.05,
       f"{score(6.58):.3f} vs {score(99.82):.3f}")


def t_retrieval_term_dynamic_range() -> None:
    """检测器：检索项的**动态范围**必须大于 0。

    这是 exp5 给出的、能自动化的指纹。三例"信号静默失效"的共同点是
    **该变化的量没变化** —— 与其事后对着生产端核字段，不如直接断言"它在变"。

    exp5 实测（94 题）：
      · BM25   检索项极差 0.591  ← 活着
      · Dense  极差 0.016        ← 阈值间隔是 0.1 量级，动 0.016 越不过任何线
      · Hybrid 极差 **0.0001**   ← 等于常数，这一项完全没进风险分
    根因：`_BM25_MID = 27.0` 是 **BM25 的量纲常数**（该检索器 top1 分的中位数）。
    换成余弦（~0.3~0.6）或 RRF 分（~0.03）后 `top/(top+27)` 被压成常数。
    """
    print("\n[17] 检索项动态范围：常数项 = 静默失效（exp5 的检测器）")
    from hragent.risk.guards import _retrieval_conf

    # (a) 合成检查：把不同检索器的分数量纲喂进去，动态范围应当有本质差别
    bm25 = [_retrieval_conf(t) for t in (6.58, 20.0, 27.46, 50.0, 99.82)]
    rrf = [_retrieval_conf(t) for t in (0.0328, 0.0320, 0.0315, 0.0301, 0.0288)]
    ok("BM25 量纲：检索项确实在变（极差 > 0.2）", max(bm25) - min(bm25) > 0.2,
       f"极差 {max(bm25) - min(bm25):.4f}")
    ok("RRF 量纲：沿用 BM25 的 MID 时检索项退化成常数（极差 < 0.001）",
       max(rrf) - min(rrf) < 0.001, f"极差 {max(rrf) - min(rrf):.6f}")

    # (b) 真实检查：拿真语料 + 真题跑一遍，断言这一项在真实分布上确实有范围。
    #     合成检查能被"改个常数"糊弄过去，真实检查不能。
    from hragent import config
    from hragent.tools.retrieval import index as bm25_index
    import json as _json
    rows = [_json.loads(l) for l in
            (config.ROOT / "eval" / "rag_eval.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()][:24]
    idx = bm25_index()
    vals = []
    for r in rows:
        hits = idx.search(r["query"], 1)
        vals.append(1.0 - _retrieval_conf(float(hits[0]["score"]) if hits else 0.0))
    spread = max(vals) - min(vals)
    ok(f"真实语料上 {len(rows)} 题：检索项极差 > 0.2（实测 {spread:.3f}）", spread > 0.2)

    # (c) 端到端：极差为 0 时，风险分对检索分的高低**毫无反应** —— 这正是危害
    g = UncertaintyGate()

    def score(top: float) -> float:
        return g.risk_score(cand("x", [ev("v")], confidence=0.9,
                                 artifacts={"top_scores": [top]}))
    # 若把 RRF 分喂进来，最高分与最低分的风险分差应当小到没有实际意义
    d_rrf = abs(score(0.0328) - score(0.0288))
    d_bm25 = abs(score(6.58) - score(99.82))
    ok("RRF 分喂进来：风险分几乎不动（< 0.001）", d_rrf < 0.001, f"Δ={d_rrf:.6f}")
    ok("BM25 分喂进来：风险分明显动（> 0.05）", d_bm25 > 0.05, f"Δ={d_bm25:.4f}")


def t_screen_contract() -> None:
    """请求级判据（exp9 认定的最高优先级缺口）—— 契约与失效方向。

    为什么这一组必须存在：判据是**唯一一道在 LLM 输出之前生效的闸门**，
    它的失效方向决定了后果：
      · 判 ok 判多了 → 违规请求被当正常业务执行（**安全事故**）
      · 判 refuse/escalate 判多了 → 正常业务被打断（误伤，可观测）
    解析失败必须走哪一边，是个需要钉死的设计决定，不能靠实现细节。
    """
    print("\n[18] 请求级判据：三档契约 + 解析失败必须放行（不阻断业务）")
    from hragent.llm import Reply
    from hragent.risk.screen import QueryScreen

    class StubLLM:
        def __init__(self, payload: str):
            self.payload, self.calls = payload, 0

        def call(self, *a, **k):
            self.calls += 1
            return Reply(text=self.payload, thinking="", tool_calls=[],
                         stop_reason="stop", usage={"input_tokens": 7, "output_tokens": 3},
                         latency_s=0.0)

    def run(payload: str, enabled: bool = True):
        s = QueryScreen(llm=StubLLM(payload), enabled=enabled)
        return s, s.check("任意请求")

    # 关闭时必须**一次 LLM 都不调** —— 否则 exp9 旧臂的开销会变，历史结果不可比
    s, v = run("{}", enabled=False)
    ok("enabled=False 时恒判 ok", v.action == "ok")
    ok("enabled=False 时不发起任何 LLM 调用（旧臂开销不变）", s.llm.calls == 0)

    for act in ("ok", "refuse", "escalate"):
        _, v = run('{"action": "%s", "category": "x", "reason": "r"}' % act)
        ok(f"合法档位 {act} 原样通过", v.action == act)
    _, v = run('{"action": "ESCALATE"}')
    ok("档位大小写不敏感", v.action == "escalate")
    ok("fires：ok 不算触发", not run('{"action":"ok"}')[1].fires)
    ok("fires：refuse/escalate 算触发",
       run('{"action":"refuse"}')[1].fires and run('{"action":"escalate"}')[1].fires)

    # 失效方向：解析不出来、档位不认识、返回非对象 —— 一律放行。
    # 判据是"加一道闸"，不是"必经关卡"；让它阻断业务是把风控做成了故障源。
    for bad, label in (("这不是 JSON", "非 JSON"), ('{"action": "maybe"}', "未知档位"),
                       ('["a"]', "返回数组")):
        _, v = run(bad)
        ok(f"{label} → 放行（不阻断业务）", v.action == "ok")
    _, v = run("这不是 JSON")
    ok("解析失败的理由必须写明（否则无法统计判据自身的失败率）",
       "解析失败" in v.reason)
    _, v = run('{"action": "refuse"}')
    ok("token 用量被回传（成本必须可归集，硬性约定 5）", v.tokens == 10)


def t_screen_and_input_wiring() -> None:
    """图接线：判据不可被绕过、成本共账、缺输入走澄清而非人工。"""
    print("\n[19] 接线：判据在最前 · 共用账本 · 缺输入→澄清（不是升级人工）")
    import inspect
    from hragent.orchestrator import graph as G
    from hragent.agents.executors import REGISTRY, input_label, required_inputs

    src = inspect.getsource(G.build_graph)
    # 判据必须挂在 START 上：挂在 route 之后会被 route 的两条短路绕过
    ok("screen 是 START 的第一个后继（不可被 route 短路绕过）",
       'g.add_edge(START, "screen")' in src)
    ok("screen 的三条出边齐全（ok/refuse/escalate）",
       '"ok": "route", "refuse": "refuse", "escalate": "handoff"' in src)
    ok("handoff 直接进 hitl（转人工而非拒答）", 'g.add_edge("handoff", "hitl")' in src)

    # 成本共账：判据若自建 LLM 实例，它花的钱不会进编排层的账本 —— 静默少报
    ok("build_graph 把编排层的 LLM 交给判据（否则成本静默少报）",
       "scr.llm = getattr(orch, \"llm\", None)" in src)

    # 输入契约：生产端声明
    ok("每个执行体都声明了 required_inputs",
       all(hasattr(c, "required_inputs") for c in REGISTRY.values()))
    ok("RecruitMatch 要求 jd 与 resume", required_inputs("RecruitMatch") == ("jd", "resume"))
    ok("查库/检索类不需要用户交材料",
       required_inputs("HRDataSQL") == () and required_inputs("PolicyRAG") == ())
    ok("未注册的名字缺省不要求输入（方向与证据契约相反：多问一句是打扰）",
       required_inputs("不存在") == ())
    ok("输入有中文标签（澄清问题要具体，不能只说「请补充信息」）",
       input_label("resume") == "候选人简历")

    # 缺输入必须**结构化上报**，不能让编排层去猜错误文本
    from hragent.agents.executors import RecruitMatch
    r = RecruitMatch(llm=None).run("筛一下简历", {}, {})
    ok("RecruitMatch 缺输入时 ok=False", not r.ok)
    ok("RecruitMatch 缺输入时上报 missing_inputs（不是靠解析 error 文本）",
       r.missing_inputs == ["jd", "resume"], f"实际 {r.missing_inputs}")

    # 分支条件：收得很紧，避免把"执行体故障"伪装成"请用户补材料"
    def after_dispatch(results):
        """复刻 after_dispatch 的判据（不建图、不调 LLM）。"""
        from hragent.schema import ExecResult
        rs = results
        if not rs or any(x.ok for x in rs):
            return "review"
        if not all(x.missing_inputs for x in rs):
            return "review"
        return "ask_inputs"

    def res(ok_, missing):
        from hragent.schema import ExecResult
        return ExecResult(executor="RecruitMatch", ok=ok_, answer="", missing_inputs=missing)

    ok("全部因缺输入失败 → ask_inputs",
       after_dispatch([res(False, ["resume"])]) == "ask_inputs")
    ok("有一个成功 → review（缺的那个丢掉即可）",
       after_dispatch([res(True, []), res(False, ["resume"])]) == "review")
    ok("失败但不是缺输入 → review（别拿澄清掩盖故障）",
       after_dispatch([res(False, [])]) == "review")
    ok("空结果 → review", after_dispatch([]) == "review")

    # 澄清问题要具体
    from hragent.orchestrator.graph import build_graph  # noqa: F401
    src2 = inspect.getsource(G.build_graph)
    ok("澄清问题按缺什么拼（不写死文本）", 'input_label(k)' in src2)


def t_answer_level_confidence_wired() -> None:
    """bug：答案级 Candidate 漏传 `confidence` → 取默认 0.0 → 置信度项恒为 0.4。

    `n_final_review` 是**手工**构造 Candidate 的（它的 evidence/refs 是各执行体的并集，
    不能用 `from_exec_result`），于是 `confidence` 没被带上、静默取 dataclass 默认值。
    后果：`UncertaintyGate` 的置信度项 `0.4*(1-conf)` 对**全部 150 题恒等于 0.400**，
    风控分被钉在 0.4 以上，只剩检索项（权重 0.25）还有区分度。
    端到端实测：HITL 阈值 0.5 触发 26% 升级、精度 5.1%（基础率 2.7%），
    行为正确率反掉 18pp —— 而所有离线指标全绿，没有任何测试变红。

    这类"字段漏传 → 默认值 → 信号静默失效"只能靠**生产者/消费者字段契约**断言防住，
    所以下面既查源码里有没有这个关键字，也查行为上风险分是否真的随置信度变化。
    """
    print("\n[15] 答案级置信度：手工构造 Candidate 时必须显式接线")
    import ast
    from hragent.orchestrator import graph as G
    from hragent.schema import ExecResult

    # 用 AST 而不是子串匹配：`n_final_review` 是 build_graph 里的闭包，拿不到函数对象；
    # 而且子串会被注释/文档串里的 "confidence=" 骗过 —— 必须确认**关键字实参真的传了**。
    src_path = (Path(__file__).resolve().parents[1]
                / "src/hragent/orchestrator/graph.py")
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "n_final_review"), None)
    ok("找得到 n_final_review（断言本身有效）", fn is not None)
    passed = set()
    if fn is not None:
        for call in ast.walk(fn):
            if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "Candidate":
                passed |= {k.arg for k in call.keywords}
    ok("n_final_review 构造 Candidate 时显式传了 confidence=（漏传即静默取 0.0）",
       "confidence" in passed, f"实参有 {sorted(passed)}")

    def er(conf: float, okflag: bool = True) -> ExecResult:
        return ExecResult(executor="PolicyRAG", ok=okflag, answer="a",
                          evidence=[ev("v")], confidence=conf,
                          artifacts={"top_scores": [30.0]})

    ok("取最小值（短板原则），不是均值",
       G._answer_confidence([er(0.9), er(0.5)]) == 0.5,
       f"得到 {G._answer_confidence([er(0.9), er(0.5)])}")
    ok("失败的执行体不计入", G._answer_confidence([er(0.9), er(0.1, okflag=False)]) == 0.9)
    ok("无执行体时退化为 0.0（无依据即无置信）",
       G._answer_confidence([]) == 0.0)

    # 行为断言：好答案**不被判为需复核**，坏答案**被判为需复核**。
    # 修复前 good 对任何输入都 ≥ 0.4（常数项 0.4），而阈值 0.5 又不可达 ——
    # 两头都坏：好答案白担风险，坏答案永远抓不到。
    g = UncertaintyGate()
    good_c = cand("质量部平均月薪 12000 元。", [ev("12000", "hr.db#m")],
                  confidence=0.9, artifacts={"top_scores": [30.0]})
    good = g.risk_score(good_c)
    ok("好答案（高置信+好检索）不被默认阈值判为需复核",
       g.check(good_c).passed, f"risk={good:.3f} vs 阈值 {g.threshold}")
    bad_c = cand("制度里应该没有相关规定。", [], confidence=0.2,
                 artifacts={"top_scores": [6.6]})
    ok("坏答案（低置信+差检索）被默认阈值判为需复核",
       not g.check(bad_c).passed,
       f"risk={g.risk_score(bad_c):.3f} vs 阈值 {g.threshold}")
    bad = g.risk_score(cand("质量部平均月薪 12000 元。", [ev("12000", "hr.db#m")],
                            confidence=0.2, artifacts={"top_scores": [30.0]}))
    ok("端到端：置信度高低会改变风险分", bad > good + 0.2,
       f"{bad:.3f} vs {good:.3f}")


def t_uncertainty_threshold_reachable() -> None:
    """[20] 默认阈值必须**可达** —— 否则这道闸门等于没接。

    exp8b 实测的教训：三项权重 (0.4, 0.35, 0.25) 下风险分的可达上界是 0.401，
    而默认阈值曾是 0.5。于是整条 exp9 消融 2445 次运行里这道闸门**一次都没触发过**，
    且不报错、不影响任何指标 —— 与 `_top_scores` 读错键、`n_final_review` 漏传
    confidence 是同一类"声明了但没接线"。本组把"可达性"变成断言，让这类问题
    以后在自测里就暴露，而不是等到写报告时才发现某道闸门是摆设。
    """
    print("\n[20] 默认阈值可达性（防「声明了但没接线」）")
    from hragent.risk.guards import _UNCERTAINTY_THRESHOLD, _retrieval_conf

    ok("默认阈值不是 0.5 那种拍脑袋值",
       abs(_UNCERTAINTY_THRESHOLD - 0.5) > 1e-9, f"={_UNCERTAINTY_THRESHOLD}")

    g = UncertaintyGate()          # 默认阈值
    # 最坏情况：置信度 0（不确定度项拉满）+ 检索分 0（检索项拉满）
    worst = cand("答案", [ev("v")], confidence=0.0, artifacts={"top_scores": [0.0]})
    ok("最坏样本的风险分 ≥ 默认阈值（阈值可达）",
       g.risk_score(worst) >= g.threshold,
       f"worst={g.risk_score(worst):.3f} vs 阈值 {g.threshold}")

    # 反向：一个"好样本"必须低于阈值，否则闸门会全量误报。
    best = cand("答案", [ev("v")], confidence=1.0,
                artifacts={"top_scores": [1e6]})
    ok("最好样本的风险分 < 默认阈值（阈值有区分度）",
       g.risk_score(best) < g.threshold,
       f"best={g.risk_score(best):.3f} vs 阈值 {g.threshold}")

    # 用真实语料的量纲核一遍：BM25 分落在 6.6~100 之间时，
    # 检索项取到的极差必须能推动风险分跨过阈值。
    lo = g.risk_score(cand("答案", [ev("v")], confidence=0.92,
                           artifacts={"top_scores": [6.6]}))
    hi = g.risk_score(cand("答案", [ev("v")], confidence=0.92,
                           artifacts={"top_scores": [99.8]}))
    ok("真实 BM25 量纲下，检索项足以让风险分跨过默认阈值",
       lo >= g.threshold > hi,
       f"低分样本 {lo:.3f} / 高分样本 {hi:.3f} / 阈值 {g.threshold}")

    ok("饱和中点仍映射到 0.5（量纲常数未被阈值改动影响）",
       abs(_retrieval_conf(27.0) - 0.5) < 1e-9)

    # warn 必须只置 needs_review，不再自己送 HITL。
    src = (Path(__file__).resolve().parents[1] / "src" / "hragent"
           / "orchestrator" / "graph.py").read_text(encoding="utf-8")
    ok("n_final_review 里 warn 不再折进 escalated",
       "state.get(\"escalated\", False) or has_warn" not in src,
       "该写法会让精准度 0.43 的弱信号把请求送进 HITL")
    ok("n_final_review 里 warn 置的是 needs_review",
       '"needs_review": has_warn' in src)
    ok("Trace 有 needs_review 字段，且与 escalated 分开",
       "needs_review: bool = False" in
       (Path(__file__).resolve().parents[1] / "src" / "hragent"
        / "schema.py").read_text(encoding="utf-8"))


def t_evidence_contract_consistent() -> None:
    """同一契约的三个消费者必须一致：G1、不确定性门控、编排层的升级判据。

    "无证据 = 有问题"这条判据散在三处。原状是 G1 做对了（看 `requires_evidence`），
    另两处漏看 —— 于是生成型执行体（RecruitMatch / OnboardFlow / InterviewKit）
    契约上本就无证据可引，却被当成高风险推去升级人工。
    实测（exp9 修复后 B3）：4 例升级里 3 例由这条误判造成，全部落在金标 answer 的题上。

    契约本身也挪了位置：原先硬编码在编排层 `EVIDENCE_REQUIRED`，
    现在由各执行体类属性声明 —— 漏配的方向从"静默放行"翻成"吵起来"。
    """
    print("\n[16] 证据契约：生产端声明，三个消费者判据一致")
    from hragent.agents.executors import REGISTRY, requires_evidence
    from hragent.agents.orchestrator import Orchestrator
    import inspect

    ok("每个执行体都声明了 requires_evidence",
       all(hasattr(c, "requires_evidence") for c in REGISTRY.values()),
       f"缺声明的：{[n for n, c in REGISTRY.items() if not hasattr(c, 'requires_evidence')]}")
    ok("数据/制度类要求证据", requires_evidence("HRDataSQL")
       and requires_evidence("PolicyRAG"))
    ok("生成型不要求证据", not requires_evidence("RecruitMatch")
       and not requires_evidence("OnboardFlow") and not requires_evidence("InterviewKit"))
    ok("未注册的名字缺省要求证据（保守：宁吵勿哑）",
       requires_evidence("NoSuchExecutor") is True)

    # 消费者①：G1
    gen = "建议安排两轮技术面，重点考察并发与系统设计能力。"
    ok("G1：生成型无证据放行",
       GroundingGuard().check(cand(gen, requires_evidence=False)).passed)
    ok("G1：数据类无证据拦截",
       not GroundingGuard().check(cand(gen, requires_evidence=True)).passed)

    # 消费者②：不确定性门控 —— 无证据项只在契约要求证据时才加分
    g = UncertaintyGate()
    base = dict(evidence=[], confidence=0.7, artifacts={})
    need = g.risk_score(cand(gen, requires_evidence=True, **base))
    genr = g.risk_score(cand(gen, requires_evidence=False, **base))
    ok("门控：要求证据时无证据项生效（+0.35）", abs(need - 0.35 - 0.12) < 1e-9,
       f"得到 {need:.3f}")
    ok("门控：生成型不因无证据被加分", abs(genr - 0.12) < 1e-9, f"得到 {genr:.3f}")
    ok("门控：两者差值恰为 0.35（只有这一项不同）",
       abs((need - genr) - 0.35) < 1e-9)

    # 消费者③：编排层的单结果升级判据
    src = inspect.getsource(Orchestrator.synthesize)
    ok("编排层：synthesize 的升级判据里出现了 requires_evidence",
       "requires_evidence" in src,
       "`not r.evidence` 未配 requires_evidence → 生成型单结果一律升级")


def main() -> None:
    print("=" * 70)
    print("风控闸门自测（每条对应一个真实 bug 或一个已知边界）")
    print("=" * 70)
    for fn in (t_verdict_truthiness, t_restatement_length_guard,
               t_g2a_long_answer_no_fp, t_breakdown_rejects_prose,
               t_g2b_internal_tamper, t_g1_scoping, t_prohibition_exemption,
               t_compliance_blindspot_documented, t_truncation,
               t_uncertainty_signal, t_risk_agent_switches,
               t_answer_level_ref_whitelist, t_retrieval_executors_publish_refs,
               t_retrieval_term_actually_varies,
               t_retrieval_term_dynamic_range,
               t_screen_contract, t_screen_and_input_wiring,
               t_answer_level_confidence_wired, t_evidence_contract_consistent,
               t_uncertainty_threshold_reachable):
        fn()
    print("\n" + "=" * 70)
    if FAILED:
        print(f"❌ {len(FAILED)}/{N} 项失败：")
        for f in FAILED:
            print(f"   · {f}")
        sys.exit(1)
    print(f"✅ 全部 {N} 项自测通过")


if __name__ == "__main__":
    main()
