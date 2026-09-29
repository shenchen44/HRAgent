"""Trace 记录与 badcase 归因（L4 观测层）。

为什么归因要单独做一层：
  "端到端准确率 78%" 这个数字本身没有行动价值 —— 它不告诉你下一步该改哪里。
  失败可能发生在**四个完全不同的环节**，改法完全不同：

    路由错     → 改指挥官 prompt / 意图体系
    执行错     → 改执行体 / 工具 / 口径
    风控错拦   → 改闸门判据（且这是误报，代价与漏放不同）
    无证据断言 → 改证据链设计
    升级人工   → 看是门控太紧还是任务本身超纲

  所以每条 trace 必须能回答"它死在哪一步"，而不只是"它错了"。
  这也是 exp9 端到端消融的归因基础。

存储用 JSONL 追加写：一行一条 trace，写完即 flush。
好处是中断最多丢一条，且可以直接 `grep`/`jq` 查，不需要额外的读工具。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from ..schema import Trace

# 失败环节的判定顺序**重要**：先发生的先判。
# 路由错会导致后面全错，但归因必须归到根因，不能归到下游症状。
STAGES = ("route_error", "exec_error", "guard_block", "ungrounded", "escalated", "ok")


def attribute(tr: Trace, gold_executors: list[str] | None = None) -> str:
    """判定这条 trace 死在哪一步。返回 STAGES 之一。"""
    if gold_executors is not None and tr.route is not None:
        if set(tr.route.executors) != set(gold_executors):
            return "route_error"
    if any(not r.ok for r in tr.results):
        return "exec_error"
    if not tr.results and tr.route is not None and tr.route.executors:
        return "exec_error"
    if any(v.severity == "block" and not v.passed for v in tr.verdicts):
        return "guard_block"
    if any(r.ok and r.answer.strip() and not r.evidence for r in tr.results):
        return "ungrounded"
    if tr.escalated:
        return "escalated"
    return "ok"


class TraceStore:
    """trace 的追加落盘与查询。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, tr: Trace, **meta) -> dict:
        """写一条 trace。写完立即 flush —— 中断最多丢当前这一条。"""
        rec = {"ts": time.time(), **meta, **asdict(tr)}
        rec["stage"] = attribute(tr, meta.get("gold_executors"))
        with self.path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
        return rec

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def badcases(self, stages: tuple[str, ...] = ("route_error", "exec_error",
                                                  "guard_block", "ungrounded")) -> list[dict]:
        """只取真失败的（不含 escalated —— 升级人工是设计内行为，不是失败）。"""
        return [r for r in self.load() if r.get("stage") in stages]

    def report(self) -> dict:
        """按环节统计。这是"下一步该改哪里"的直接依据。"""
        rows = self.load()
        n = len(rows)
        counts = {s: 0 for s in STAGES}
        for r in rows:
            counts[r.get("stage", "ok")] = counts.get(r.get("stage", "ok"), 0) + 1
        tok = sum(int((r.get("tokens") or 0)) for r in rows)
        lat = sum(float((r.get("latency_s") or 0.0)) for r in rows)
        return {
            "n": n,
            "stages": counts,
            "stage_rate": {k: (v / n if n else 0.0) for k, v in counts.items()},
            # 成本必须与准确率并列报（DESIGN.md 硬性约定 5）
            "total_tokens": tok, "mean_tokens": (tok / n if n else 0.0),
            "mean_latency_s": (lat / n if n else 0.0),
        }

    def print_report(self) -> None:
        r = self.report()
        print(f"trace 总数 {r['n']}  ·  平均 {r['mean_tokens']:.0f} tok  "
              f"·  平均 {r['mean_latency_s']:.1f}s")
        label = {"route_error": "路由错", "exec_error": "执行错",
                 "guard_block": "风控拦截", "ungrounded": "无证据断言",
                 "escalated": "升级人工", "ok": "通过"}
        for s in STAGES:
            c = r["stages"][s]
            if c:
                print(f"  {label[s]:<8} {c:>4}  {r['stage_rate'][s]:>6.1%}")
