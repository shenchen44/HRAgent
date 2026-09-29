"""SQL 执行工具 —— 只读、有界、可审计（D4）。

设计约束：
  · LLM **永远不持有写权限**。这里在连接层就设 query_only，不靠 prompt 约束。
  · 只允许单条 SELECT。DDL/DML/PRAGMA/ATTACH 一律拒绝。
  · 必须带 LIMIT 上限，防止全表拉取（对应风控的 full_scan 故障类型）。
  · 每次执行返回可追溯证据（SQL 原文 + 结果），供证据校验闸门复算。

注意：本模块的拒绝是**硬拒绝**（抛异常），不是"劝模型别这么做"。
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field

from .. import config

FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|"
    r"PRAGMA|VACUUM|REINDEX|GRANT|REVOKE)\b", re.I)

DEFAULT_LIMIT = 500


class SQLRejected(Exception):
    """SQL 被安全策略拒绝。不是运行时错误，是策略决定。"""


@dataclass
class SQLResult:
    sql: str
    rows: list = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    error: str | None = None
    rejected: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.rejected is None

    def scalar(self):
        if len(self.rows) == 1 and len(self.rows[0]) == 1:
            return self.rows[0][0]
        return None


def check(sql: str) -> str | None:
    """静态检查。返回拒绝理由，或 None 表示通过。"""
    s = sql.strip().rstrip(";").strip()
    if not s:
        return "空 SQL"
    if ";" in s:
        return "不允许多条语句"
    if FORBIDDEN.search(s):
        return f"检测到非只读关键字: {FORBIDDEN.search(s).group()}"
    if not re.match(r"^(SELECT|WITH)\b", s, re.I):
        return "只允许 SELECT / WITH 查询"
    return None


def _connect(db) -> sqlite3.Connection:
    conn = sqlite3.connect(db or (config.DATA / "hr.db"))
    # 连接层只读 —— 即使 SQL 检查被绕过，也写不进去
    conn.execute("PRAGMA query_only = ON")
    return conn


def run_sql(sql: str, db=None, limit: int = DEFAULT_LIMIT) -> SQLResult:
    """执行只读查询。超限自动加 LIMIT；被拒则返回 rejected 而不抛异常。"""
    reason = check(sql)
    if reason:
        return SQLResult(sql=sql, rejected=reason)

    s = sql.strip().rstrip(";").strip()
    if not re.search(r"\bLIMIT\b", s, re.I):
        s = f"{s} LIMIT {limit}"

    t0 = time.time()
    conn = _connect(db)
    try:
        cur = conn.execute(s)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return SQLResult(sql=s, rows=rows, columns=cols, latency_s=time.time() - t0)
    except Exception as e:
        return SQLResult(sql=s, error=f"{type(e).__name__}: {e}", latency_s=time.time() - t0)
    finally:
        conn.close()


def is_full_scan(sql: str) -> bool:
    """启发式：无 WHERE 且无 GROUP BY 的查询视为全表扫描。

    这是给风控闸门用的判据，不是安全边界 —— 安全边界是 query_only 与静态检查。
    """
    s = sql.lower()
    if re.search(r"\bwhere\b", s) or re.search(r"\bgroup\s+by\b", s):
        return False
    return bool(re.search(r"\bfrom\s+\w+", s))


def touches_sensitive(sql: str) -> list[str]:
    """返回 SQL 触及的敏感表/字段。用于越权检测。"""
    hits = []
    s = sql.lower()
    if re.search(r"\bpayroll\b", s):
        hits.append("payroll(薪酬)")
    if re.search(r"\bsocial_insurance\b|\bhousing_fund\b|\bnet_pay\b|\bgross_pay\b", s):
        hits.append("薪酬明细字段")
    if re.search(r"\bphone\b|\bid_card\b|\bbank_card\b", s):
        hits.append("个人敏感字段")
    return hits


if __name__ == "__main__":
    for q in [
        "SELECT COUNT(*) FROM employees WHERE status='在职'",
        "SELECT * FROM payroll",
        "DROP TABLE employees",
        "SELECT 1; SELECT 2",
        "PRAGMA table_info(employees)",
    ]:
        r = run_sql(q)
        tag = "拒绝" if r.rejected else ("错误" if r.error else f"{len(r.rows)} 行")
        print(f"  [{tag:8s}] {q[:52]:54s} {r.rejected or r.error or ''}")
