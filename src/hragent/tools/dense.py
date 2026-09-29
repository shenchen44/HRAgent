"""稠密检索 / 混合检索 / 重排 —— exp5 的三个实验臂。

为什么单独建文件而不塞进 retrieval.py：
  BM25（稀疏）与稠密是**两种独立方法**，混在一个类里就没法单独开关、
  也就没法做消融。exp5 要分别测它们，所以各自独立。

三个实现：
  DenseIndex    bge-small-zh-v1.5 向量检索（余弦）
  HybridIndex   BM25 + Dense，用 RRF 融合 —— 不需要调分数权重，比加权求和稳健
  RerankedIndex 任意索引召回后，用 bge-reranker-base 交叉编码器重排

模型加载慢（首次 ~40s + 下载），全部做进程级缓存。
依赖 sentence-transformers，未安装时给出明确报错而不是静默降级。
"""

from __future__ import annotations

import functools

import numpy as np

from .retrieval import index as bm25_index

# bge 系列官方要求：**查询侧**加指令前缀，文档侧不加。
# 漏掉会让检索质量明显下降，且不会报错 —— 属于典型的静默错误。
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
RRF_K = 60


def _require_st() -> None:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "缺少 sentence-transformers。安装：\n"
            "  uv pip install --python .venv/bin/python sentence-transformers"
        ) from e


@functools.lru_cache(maxsize=2)
def embedder(model_name: str = EMBED_MODEL):
    _require_st()
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


@functools.lru_cache(maxsize=2)
def reranker(model_name: str = RERANK_MODEL):
    _require_st()
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def _chunks() -> list[dict]:
    return bm25_index().chunks


@functools.lru_cache(maxsize=1)
def _doc_embeddings() -> np.ndarray:
    """把全部制度片段编码一次并缓存（归一化后内积即余弦）。"""
    texts = [f"{c['doc_title']} {c['heading']}\n{c['text']}" for c in _chunks()]
    return embedder().encode(texts, normalize_embeddings=True,
                             batch_size=32, show_progress_bar=False)


class DenseIndex:
    """向量检索：bge-small-zh-v1.5 + 余弦相似度。"""

    name = "Dense"

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = embedder().encode([BGE_QUERY_PREFIX + query],
                              normalize_embeddings=True,
                              show_progress_bar=False)[0]
        sims = _doc_embeddings() @ q
        order = np.argsort(-sims)[:k]
        cs = _chunks()
        return [{"chunk_id": cs[i]["chunk_id"], "doc_title": cs[i]["doc_title"],
                 "heading": cs[i]["heading"], "text": cs[i]["text"],
                 "score": float(sims[i]), "retriever": self.name} for i in order]


class HybridIndex:
    """BM25 + 向量，用 RRF（Reciprocal Rank Fusion）融合。

    为什么用 RRF 而不是加权求和：
      BM25 分数是无界的词频量，余弦相似度在 [0,1]，两者量纲不可比。
      要加权就得先归一化调权重，而权重只能在 dev 上拟合 —— 又多一个超参、
      又多一处过拟合。RRF 只用**排名**，天然免调参：
          score(d) = Σ_r 1 / (RRF_K + rank_r(d))
    """

    name = "Hybrid"

    def __init__(self, pool: int = 20):
        self.pool = pool
        self.bm25 = bm25_index()
        self.dense = DenseIndex()

    def search(self, query: str, k: int = 5) -> list[dict]:
        lists = [self.bm25.search(query, self.pool), self.dense.search(query, self.pool)]
        fused: dict[str, float] = {}
        meta: dict[str, dict] = {}
        for hits in lists:
            for rank, h in enumerate(hits, start=1):
                cid = h["chunk_id"]
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
                meta.setdefault(cid, h)
        order = sorted(fused, key=lambda c: -fused[c])[:k]
        out = []
        for cid in order:
            h = dict(meta[cid])
            h["score"] = fused[cid]
            h["retriever"] = self.name
            out.append(h)
        return out


class RerankedIndex:
    """先用 base 索引召回一个候选池，再用交叉编码器重排。

    交叉编码器把 (query, doc) 拼在一起过一遍模型，比双塔的余弦准得多，
    但**不能预先算文档向量**，所以只能对候选池重排，不能全库检索。
    这就是"召回-重排"两阶段架构存在的原因。
    """

    name = "Rerank"

    def __init__(self, base=None, pool: int = 20):
        self.base = base or HybridIndex(pool=pool)
        self.pool = pool

    def search(self, query: str, k: int = 5) -> list[dict]:
        cands = self.base.search(query, self.pool)
        if not cands:
            return []
        pairs = [(query, f"{c['doc_title']} {c['heading']}\n{c['text']}") for c in cands]
        scores = reranker().predict(pairs, show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))[:k]
        out = []
        for i in order:
            h = dict(cands[i])
            h["score"] = float(scores[i])
            h["retriever"] = self.name
            out.append(h)
        return out


def build(name: str):
    return {"BM25": lambda: bm25_index(), "Dense": DenseIndex,
            "Hybrid": HybridIndex, "Rerank": RerankedIndex}[name]()
