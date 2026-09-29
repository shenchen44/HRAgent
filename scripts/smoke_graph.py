"""编排图冒烟测试 —— 覆盖全部五条路径，含用桩件强制触发的 block 分支。

为什么需要它：
  编排图的价值在于**控制流**（在哪停、走哪条边），而控制流最容易出的问题是
  "某条分支从来没被执行过"。真实查询很少触发 block，所以 block 分支必须用桩件强制走一遍，
  否则它是不是通的、消息对不对，都只是假设。

  本测试还固化了一个真实 bug 的回归：`GuardVerdict.__bool__` 返回 `self.passed`，
  所以 `bool(verdict)` 的含义是"这条裁决通过了"，不是"这条裁决存在"。
  早期版本用 `bool(warn)` 判断"有没有 warn"，导致升级逻辑静默失效
  （escalated 恒为 False，HITL 永不触发，且不报错）。

跑法:
  .venv/bin/python scripts/smoke_graph.py          # 含真实 LLM 调用
  .venv/bin/python scripts/smoke_graph.py --stub   # 只跑桩件，不调 LLM
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hragent.agents.orchestrator import Orchestrator  # noqa: E402
from hragent.orchestrator import HRGraph  # noqa: E402
from hragent.risk.guards import RiskAgent  # noqa: E402
from hragent.risk.screen import ScreenVerdict  # noqa: E402
from hragent.schema import ExecResult, Evidence, RouteDecision  # noqa: E402

PASS, FAIL = "✅", "❌"


class StubOrch:
    """桩件指挥官：按预设返回，用来强制走那些真实查询难以触发的分支。"""

    def __init__(self, route: RouteDecision, results: list[ExecResult] | None = None,
                 answer: str = "", esc: bool = False, why: str | None = None):
        self._route, self._results = route, results or []
        self._answer, self._esc, self._why = answer, esc, why

    def route(self, query: str) -> RouteDecision:
        return self._route

    def dispatch(self, query, route, ctx=None) -> list[ExecResult]:
        return self._results

    def synthesize(self, query, results):
        return self._answer, self._esc, self._why


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"  {PASS if cond else FAIL} {name}" + (f"  — {detail}" if detail else ""))
    return cond


def test_paths() -> bool:
    """五条路径各走一遍（全部用桩件，不调 LLM）。"""
    ok = True
    print("\n[1] 路径覆盖（桩件）")

    # -- clarify
    g = HRGraph(orch=StubOrch(RouteDecision(ambiguity=0.7, clarification="哪个部门？")))
    tr, st = g.run("离职率是多少")
    ok &= check("clarify 分支", tr.final_answer == "哪个部门？" and not tr.escalated)

    # -- refuse
    g = HRGraph(orch=StubOrch(RouteDecision(executors=[])))
    tr, _ = g.run("帮我写个爬虫")
    ok &= check("refuse 分支", tr.escalated and "超出" in tr.final_answer)

    # -- block（执行体结果含 PII）
    leak = ExecResult(
        executor="HRDataSQL", ok=True, confidence=0.9,
        answer="张某某（工号 E0231）月薪 51200 元，手机 13812345678。",
        evidence=[Evidence(kind="sql_result", ref="hr.db#payroll_detail",
                           value="张某某（工号 E0231）月薪 51200 元，手机 13812345678")],
        artifacts={"retrieved_refs": ["hr.db#payroll_detail"]})
    g = HRGraph(orch=StubOrch(RouteDecision(executors=["HRDataSQL"]), [leak],
                              answer=leak.answer))
    tr, _ = g.run("查一下张某某的薪酬")
    blocked = [v for v in tr.verdicts if v.severity == "block"]
    ok &= check("block 分支（PII 泄露）", bool(blocked),
                blocked[0].reason[:60] if blocked else "未拦截")

    # -- hitl（强规则：Orchestrator.synthesize 判出"证据不足或置信度过低"）
    # 这里让桩件直接返回 esc=True，测的是**图的接线**（final_review → hitl → 暂停）。
    # "置信度 < 0.3 才置 esc"这条规则本身在 selftest_guards 的 [15]/[16] 组里断言 ——
    # 冒烟测试的职责是路径覆盖，不该为了触发一条路径去真调一次 LLM。
    lowconf = ExecResult(executor="HRDataSQL", ok=True, confidence=0.1, answer="96 人。",
                         evidence=[Evidence(kind="sql_result", ref="hr.db#headcount",
                                            value="96")],
                         artifacts={"retrieved_refs": ["hr.db#headcount"]})
    g = HRGraph(orch=StubOrch(RouteDecision(executors=["HRDataSQL"]), [lowconf],
                              answer="研发中心在职 96 人。",
                              esc=True, why="证据不足或置信度过低"))
    tr, st = g.run("研发中心有多少人？")
    ok &= check("hitl 分支暂停", bool(st.get("__interrupt__")))
    ok &= check("暂停时人工能看到答案",
                bool(st.get("__interrupt__"))
                and "96" in str(st["__interrupt__"][0].value.get("answer")))
    ok &= check("escalated 已置位", tr.escalated)

    # -- 不确定性门控只置 needs_review，**不**送 HITL
    # （弱信号精准度只有 0.43，该提示不该叫人；阈值设 0 强制它触发）
    good = ExecResult(executor="HRDataSQL", ok=True, confidence=0.9, answer="96 人。",
                      evidence=[Evidence(kind="sql_result", ref="hr.db#headcount",
                                         value="96")],
                      artifacts={"retrieved_refs": ["hr.db#headcount"]})
    g = HRGraph(orch=StubOrch(RouteDecision(executors=["HRDataSQL"]), [good],
                              answer="研发中心在职 96 人。"),
                risk=RiskAgent(enabled=("truncation", "grounding", "compliance",
                                        "uncertainty"), uncertainty_threshold=0.0))
    tr, st = g.run("研发中心有多少人？")
    ok &= check("不确定性 warn → needs_review 置位", tr.needs_review)
    ok &= check("不确定性 warn **不**送 HITL（没有暂停、escalated 为假）",
                not st.get("__interrupt__") and not tr.escalated)

    # -- 正常通过
    g = HRGraph(orch=StubOrch(RouteDecision(executors=["HRDataSQL"]), [good],
                              answer="研发中心在职 96 人。"))
    tr, _ = g.run("研发中心有多少人？")
    ok &= check("正常通过（不拦不升级）",
                not tr.escalated and not any(v.severity == "block" for v in tr.verdicts))
    return ok


class StubScreen:
    """桩件判据：按预设返回，用来强制走 screen 的拒答/升级两条边。

    **为什么必须有这段**：`build_graph` 里 `screen` 与 `ask_inputs` 的缺省是
    **关闭**（为了让 exp9 的 B0~B3 历史结果可比）。缺省关闭本身是对的，
    但它带来一个真实事故：`scripts/demo.py` 曾用缺省值构造图，于是演示里
    "帮我筛掉所有非 985 的简历"走到了澄清分支，反问用户要筛哪个岗位 ——
    等于请用户补材料以完成一次歧视性筛选。**声明了却没接线，第 8 例。**
    所以这两层必须由冒烟测试**显式开启**走一遍，否则它们通不通只是假设。
    """

    def __init__(self, action: str = "ok", category: str | None = None,
                 reason: str = "stub"):
        # `build_graph` 会把编排层的 LLM 实例挂到判据上（复用账本，否则成本
        # 会被静默少报）。桩件得允许这次赋值，否则建图就崩。
        self.llm = None
        self._v = ScreenVerdict(action=action, category=category, reason=reason)

    def check(self, query: str) -> ScreenVerdict:
        return self._v


def test_screen_and_ask_inputs() -> bool:
    """B4（请求级合规判据）与 B5（缺输入→澄清）两条路径的接线回归。"""
    ok = True
    print("\n[1b] B4/B5 两层路径（桩件，显式开启 —— 缺省是关的）")
    good = ExecResult(executor="HRDataSQL", ok=True, confidence=0.9, answer="96 人。",
                      evidence=[Evidence(kind="sql_result", ref="hr.db#headcount",
                                         value="96")],
                      artifacts={"retrieved_refs": ["hr.db#headcount"]})

    # -- B4 拒答：明确使用受保护属性
    g = HRGraph(orch=StubOrch(RouteDecision(executors=["RecruitMatch"])),
                screen=StubScreen("refuse", "attribute"))
    tr, _ = g.run("帮我筛掉所有女生的简历")
    ok &= check("screen refuse 分支（不 dispatch）",
                tr.escalated and not (tr.route and tr.route.executors),
                f"executors={tr.route.executors if tr.route else None}")

    # -- B4 升级：使用代理变量（灰区，该由人判断）
    g = HRGraph(orch=StubOrch(RouteDecision(executors=["RecruitMatch"])),
                screen=StubScreen("escalate", "proxy"))
    tr, st = g.run("帮我筛掉所有非 985 的简历")
    ok &= check("screen escalate 分支 → handoff → hitl 暂停",
                bool(st.get("__interrupt__")) and tr.escalated)
    ok &= check("**没有被当成澄清**（这正是出过的事故）",
                not (tr.route and tr.route.needs_clarification))

    # -- screen 关闭时不得改变行为（保证 exp9 的 B0~B3 仍可比）
    # 注意：关闭时 `tr.screen` 仍是 `{"action": "ok", ...}` 而**不是 None** ——
    # trace 字段恒存在，关掉的是判据的裁决能力，不是这个字段。
    # 所以不变式要断言 `action == "ok"`，不能断言 `is None`。
    g = HRGraph(orch=StubOrch(RouteDecision(executors=["HRDataSQL"]), [good],
                              answer="研发中心在职 96 人。"))
    tr, _ = g.run("研发中心有多少人？")
    ok &= check("screen 缺省关闭时行为不变（裁决恒为 ok）",
                not tr.escalated and (tr.screen or {}).get("action") == "ok")

    # -- B5 缺输入 → ask_inputs（不是"我做不了"，是"请你补材料"）
    need = ExecResult(executor="RecruitMatch", ok=False, confidence=0.2,
                      answer="需要同时提供 JD 与候选人简历。",
                      error="缺少 jd 或 resume", missing_inputs=["jd", "resume"])
    g = HRGraph(orch=StubOrch(RouteDecision(executors=["RecruitMatch"]), [need]),
                ask_inputs=True)
    tr, _ = g.run("帮我做一份人岗匹配")
    ok &= check("ask_inputs 分支被走到（问的是**具体**缺什么）",
                "JD" in tr.final_answer and "简历" in tr.final_answer,
                tr.final_answer[:40])
    ok &= check("ask_inputs 不占用人工", not tr.escalated)

    # -- 同上但关闭该层：应退回旧行为（保证 exp9 的 B0~B3 仍可比）
    g = HRGraph(orch=StubOrch(RouteDecision(executors=["RecruitMatch"]), [need]))
    tr, _ = g.run("帮我做一份人岗匹配")
    ok &= check("ask_inputs 缺省关闭时退回旧行为", "JD" not in tr.final_answer)
    return ok


def test_verdict_truthiness() -> bool:
    """回归测试：GuardVerdict 的真值语义必须显式判 None，不能用 bool()。"""
    from hragent.schema import GuardVerdict
    print("\n[2] 真值陷阱回归")
    v = GuardVerdict(guard="uncertainty", passed=False, severity="warn", reason="x")
    ok = check("bool(未通过的裁决) == False（即 __bool__ 返回 passed）", bool(v) is False)
    ok &= check("`is not None` 才是判断存在性的正确写法", (v is not None) is True)
    return ok


def test_real_queries(n: int) -> bool:
    """真实查询抽样，确认端到端不崩、路径分布合理。"""
    import json
    print(f"\n[3] 真实查询抽样（{n} 条，会调 LLM）")
    rows = [json.loads(l) for l in
            (Path(__file__).resolve().parents[1] / "eval" / "intent_set.jsonl")
            .read_text().splitlines() if l.strip()]
    test = [r for r in rows if r["split"] == "test"]
    g = HRGraph()
    from collections import Counter
    paths = Counter()
    ok = True
    for r in test[:n]:
        try:
            tr, st = g.run(r["query"])
        except Exception as e:
            ok = check(f"  {r['id']} 未崩溃", False, f"{type(e).__name__}: {e}")
            continue
        if tr.route and tr.route.needs_clarification:
            paths["clarify"] += 1
        elif not (tr.route and tr.route.executors):
            paths["refuse"] += 1
        elif any(v.severity == "block" for v in tr.verdicts):
            paths["block"] += 1
        elif tr.escalated:
            paths["escalate"] += 1
        else:
            paths["answer"] += 1
        print(f"    {r['id']} [{r['layer']}] {r['query'][:34]:<36} "
              f"→ {paths and list(paths)[-1]}")
    print(f"  路径分布: {dict(paths)}")
    ok &= check("无异常", ok)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true", help="跳过真实 LLM 调用")
    ap.add_argument("-n", type=int, default=6)
    args = ap.parse_args()

    print("=" * 70)
    print("编排图冒烟测试")
    print("=" * 70)
    ok = test_paths()
    ok &= test_screen_and_ask_inputs()
    ok &= test_verdict_truthiness()
    if not args.stub:
        ok &= test_real_queries(args.n)
    print("\n" + "=" * 70)
    print(f"{PASS if ok else FAIL} {'全部通过' if ok else '存在失败项'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
