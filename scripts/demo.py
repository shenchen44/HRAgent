"""端到端演示 —— 一条命令看完整链路。

展示的是**控制流**，不是"能答对几道题"：
  路由 → 执行体 → 证据 → 四道闸门 → （拦截 | 升级人工 | 放行）
每一步的裁决都打出来，因为本项目的卖点就是"每一步都可追溯、可拦截"。

跑法:
  .venv/bin/python scripts/demo.py "研发中心现在有多少在职员工？"
  .venv/bin/python scripts/demo.py --repl          # 交互模式
  .venv/bin/python scripts/demo.py --hitl "..."    # 强制打开人工介入（阈值压到 0）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hragent import config  # noqa: E402
from hragent.obs import TraceStore  # noqa: E402
from hragent.orchestrator import HRGraph  # noqa: E402
from hragent.risk.guards import RiskAgent  # noqa: E402
from hragent.risk.screen import QueryScreen  # noqa: E402

MARK = {"pass": "✅", "warn": "⚠️ ", "block": "⛔"}
W = 74


def show(tr, st, query: str) -> None:
    print(f"\n{'─' * W}")
    print(f"问题：{query}")
    print(f"{'─' * W}")

    if tr.route:
        print(f"  路由  ·  歧义 {tr.route.ambiguity:.2f}  ·  执行体 "
              f"{tr.route.executors or '（空）'}")
        if tr.route.needs_clarification:
            print(f"\n  💬 澄清：{tr.final_answer}")
            return

    for r in tr.results:
        ev = "；".join(f"{e.ref}" for e in r.evidence[:3])
        print(f"  [{r.executor}] ok={r.ok} conf={r.confidence:.2f} "
              f"证据 {len(r.evidence)} 条  {r.latency_s:.1f}s")
        if ev:
            print(f"      出处：{ev}")
        if not r.ok:
            print(f"      ⚠️  {r.error}")

    if tr.verdicts:
        # 风控是两级：执行体级（每个结果各查一遍）与答案级（汇总文本再查一遍）。
        # 分开呈现，否则同一个闸门会连着出现两次，看起来像重复执行。
        n = int(st.get("n_exec_verdicts") or 0)
        for stage, vs in (("执行体级", tr.verdicts[:n]), ("答案级", tr.verdicts[n:])):
            if not vs:
                continue
            print(f"  风控 · {stage}：")
            for v in vs:
                mark = MARK.get(v.severity, "  ")
                detail = f"  — {v.reason[:78]}" if v.reason else ""
                print(f"    {mark} {v.guard:<11}{detail}")

    if st.get("__interrupt__"):
        payload = st["__interrupt__"][0].value
        print(f"\n  ⏸  已暂停，等待人工裁决")
        print(f"     理由：{payload.get('reason')}")
        ans = str(payload.get("answer") or "")
        print(f"     待审答案：{ans[:110]}{'…' if len(ans) > 110 else ''}")
        print(f"     （恢复：HRGraph.resume(thread_id, {{'decision': 'approve'}})）")
        return

    print(f"\n  回答：{tr.final_answer[:300]}")
    print(f"\n  成本：{tr.tokens} tok  ·  {tr.latency_s:.1f}s"
          f"  ·  升级={tr.escalated}"
          + (f"（{tr.escalation_reason}）" if tr.escalation_reason else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="*")
    ap.add_argument("--repl", action="store_true")
    ap.add_argument("--hitl", action="store_true",
                    help="把不确定性阈值压到 0，强制走人工介入分支")
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--no-screen", action="store_true",
                    help="关掉请求级合规判据（B4 层），用于对照")
    ap.add_argument("--no-ask-inputs", action="store_true",
                    help="关掉缺输入→澄清（B5 层），用于对照")
    args = ap.parse_args()

    risk = (RiskAgent(enabled=("truncation", "grounding", "compliance", "uncertainty"),
                      uncertainty_threshold=0.0) if args.hitl else None)
    store = None if args.no_trace else TraceStore(config.RESULTS / "demo_trace.jsonl")
    # **B4/B5 两层默认打开。** 曾经这里是 `HRGraph(risk=risk, store=store)` ——
    # 两个参数都取缺省值，而缺省是**关闭**（`build_graph` 为了让 exp9 的
    # B0~B3 历史结果可比才这么定的）。后果：演示里"帮我筛掉所有非 985 的简历"
    # 会走到澄清分支，反问用户"要筛哪个岗位" —— 等于请用户补材料以完成一次
    # 歧视性筛选，正是 `risk/screen.py` 存在的理由。**声明了却没接线，第 8 例。**
    # 缺省值服务的是消融实验，不该顺延成演示的配置。
    g = HRGraph(risk=risk, store=store,
                screen=QueryScreen(enabled=not args.no_screen),
                ask_inputs=not args.no_ask_inputs)

    if args.repl:
        print("交互模式（空行退出）")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            tr, st = g.run(q)
            show(tr, st, q)
        return

    for q in args.queries or ["研发中心现在有多少在职员工？"]:
        tr, st = g.run(q)
        show(tr, st, q)

    if store is not None:
        print(f"\n{'─' * W}")
        store.print_report()
        print(f"  trace → {store.path}")


if __name__ == "__main__":
    main()
