"""制度检索 —— BM25 + 中文字符二元分词。

为什么不用向量检索：
  本项目要测的是"引用是否真的支撑断言"，需要**可复现、可解释**的召回结果。
  向量检索引入 embedding 模型依赖，且召回不可解释，会让"检索错了"和"生成错了"
  混在一起分不开。BM25 的召回可以逐词追溯。
  代价是语义泛化弱 —— 这个代价会被如实报出来，而不是藏起来。

分词：中文按字符二元组（bigram），ASCII 按小写词。无需外部分词器。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from .. import config

ASCII = re.compile(r"[a-zA-Z][a-zA-Z0-9_+#./-]*")
CJK = re.compile(r"[一-鿿]")


def tokenize(text: str) -> list[str]:
    """中文 bigram + ASCII 词。中文单字也保留（短查询时 bigram 会不足）。"""
    toks: list[str] = []
    for m in ASCII.finditer(text):
        toks.append(m.group().lower())
    for m in CJK.finditer(text):
        toks.append(m.group())
    # 中文 bigram：只对连续中文段做
    for seg in re.findall(r"[一-鿿]+", text):
        for i in range(len(seg) - 1):
            toks.append(seg[i:i + 2])
    return toks


class PolicyIndex:
    """BM25 索引。k1/b 用文献常用默认值。"""

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(c["text"]) for c in chunks]
        self.tf = [Counter(d) for d in self.docs]
        self.len = [len(d) for d in self.docs]
        self.avg_len = sum(self.len) / len(self.len) if self.len else 0.0
        self.df: Counter = Counter()
        for d in self.docs:
            self.df.update(set(d))
        self.n = len(self.docs)
        self.postings: dict[str, list[int]] = {}
        for i, d in enumerate(self.docs):
            for t in set(d):
                self.postings.setdefault(t, []).append(i)

    def _idf(self, t: str) -> float:
        df = self.df.get(t, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 5) -> list[dict]:
        """返回 top-k 片段，附带 score 与命中的词（便于诊断）。"""
        q = tokenize(query)
        if not q:
            return []
        scores: dict[int, float] = {}
        hits: dict[int, list[str]] = {}
        for t in set(q):
            if t not in self.postings:
                continue
            idf = self._idf(t)
            for i in self.postings[t]:
                tf = self.tf[i][t]
                denom = tf + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg_len)
                scores[i] = scores.get(i, 0.0) + idf * tf * (self.k1 + 1) / denom
                hits.setdefault(i, []).append(t)
        order = sorted(scores, key=lambda i: -scores[i])[:k]
        return [{**self.chunks[i], "score": scores[i], "matched": sorted(hits[i])}
                for i in order]

    def retrieve_ids(self, query: str, k: int = 5) -> list[str]:
        return [r["chunk_id"] for r in self.search(query, k)]


_CACHE: PolicyIndex | None = None


def index(path: Path | None = None) -> PolicyIndex:
    """加载并缓存索引。"""
    global _CACHE
    if _CACHE is None:
        p = path or (config.DATA / "policy_chunks.jsonl")
        chunks = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        _CACHE = PolicyIndex(chunks)
    return _CACHE


if __name__ == "__main__":
    import sys
    idx = index()
    for q in sys.argv[1:] or ["年休假有多少天", "加班费怎么算"]:
        print(f"\n查询: {q}")
        for r in idx.search(q, 3):
            print(f"  {r['score']:6.2f}  {r['chunk_id']:28s} {r['heading']}")
