"""断点续跑（L4 观测层 · DESIGN.md 硬性约定 4）。

为什么这是硬性约定而不是锦上添花：
  本项目的每个实验都要跑几十到几百次 LLM 调用。exp4 一次 5 轮重复就是 300+ 次调用，
  几分钟到十几分钟。**中途任何一次网络抖动、限流、Ctrl-C 都会让整轮白跑。**
  没有断点续跑，实验成本会高到让人不敢跑重复实验 —— 而"不敢跑重复"正是
  exp3 那条教训（单次 LLM 评测不可靠）的成因。

做法：按样本 id 落盘，一行一条，写完 flush。
  重跑时先读已完成 id 集合，跳过它们。
  中断最多丢当前正在跑的那一条，重跑时补上。

**结果文件必须自包含。** 每条记录里存全部原始字段，而不是只存一个分数 ——
否则事后想换个指标重算就得全部重跑。这是 exp4 的教训：
第一版只存了 EX，后来发现单位约定要归一化，只能重跑全部 5 轮。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable


class Checkpoint:
    """按样本 id 断点续跑的 JSONL 落盘器。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def done_ids(self) -> set[str]:
        if not self.path.exists():
            return set()
        ids = set()
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # 中断可能写坏最后一行 —— 跳过它，让该样本重跑，而不是整轮崩掉
                continue
            if "id" in rec:
                ids.add(str(rec["id"]))
        return ids

    def append(self, rec: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()

    def run(self, items: Iterable[dict], fn: Callable[[dict], dict],
            id_key: str = "id") -> list[dict]:
        """对 items 逐个跑 fn，跳过已完成的。

        fn 必须返回一个**自包含**的 dict（含全部原始字段），会被原样落盘。
        返回全部记录（含本轮之前已完成的）。
        """
        done = self.done_ids()
        items = list(items)
        todo = [it for it in items if str(it.get(id_key)) not in done]
        if done:
            print(f"  ↻ 断点续跑：已完成 {len(done)}/{len(items)}，"
                  f"本轮跑剩余 {len(todo)} 条")
        for i, it in enumerate(todo, 1):
            rec = fn(it)
            rec.setdefault(id_key, it.get(id_key))
            self.append(rec)
            if i % 10 == 0 or i == len(todo):
                print(f"    {i}/{len(todo)}", flush=True)
        return self.load()

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def reset(self) -> None:
        """清空。**只在明确要重跑全部时调用**，且应先归档旧文件。"""
        if self.path.exists():
            self.path.unlink()
