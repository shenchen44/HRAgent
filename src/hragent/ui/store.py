"""前端系统的持久化 —— **独立的 `data/ui.db`**，与 `data/hr.db` 无关。

为什么必须分开：`data/hr.db` 是评测基准库，其 SHA256 记在 `eval/FROZEN.json` 里，
所有实验数字都建立在"这份数据没动过"之上。HR 系统要能录岗位、录简历、
记匹配结果 —— 这些都是**写操作**。写进 `hr.db` 就等于亲手把冻结基准改了，
而且是静默的：实验照样跑，数字照样出，只是不再可比。

所以：`hr.db` 只读，`ui.db` 读写。本模块不 import 任何会碰 `hr.db` 的东西。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

DB = Path(__file__).resolve().parents[3] / "data" / "ui.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    dept        TEXT DEFAULT '',
    jd_text     TEXT NOT NULL,
    must_json   TEXT DEFAULT '[]',
    nice_json   TEXT DEFAULT '[]',
    headcount   INTEGER DEFAULT 1,
    status      TEXT DEFAULT '招聘中',
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    resume_text  TEXT NOT NULL,
    skills_json  TEXT DEFAULT '[]',
    source       TEXT DEFAULT '',
    applied_job  TEXT DEFAULT '',
    created_at   REAL NOT NULL
);
-- 简历文本 → 技能。同一段简历重复录入时不再花一次 LLM 调用。
-- 键是文本哈希而不是候选人 id：批量导入常常同一个人投多个岗位。
CREATE TABLE IF NOT EXISTS skill_cache (
    text_hash   TEXT PRIMARY KEY,
    skills_json TEXT NOT NULL,
    tokens      INTEGER DEFAULT 0,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS matches (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id       TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    score        REAL NOT NULL,
    must_json    TEXT DEFAULT '[]',
    must_miss    TEXT DEFAULT '[]',
    nice_json    TEXT DEFAULT '[]',
    mode         TEXT DEFAULT '召回',
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_matches_job ON matches(job_id, score DESC);
-- 决策留痕：谁在什么时候让系统做了什么、系统当时的裁决是什么。
-- HR 场景的合规要求，也是本项目 L4 观测层在前端的出口。
CREATE TABLE IF NOT EXISTS audit (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       REAL NOT NULL,
    kind     TEXT NOT NULL,
    summary  TEXT NOT NULL,
    payload  TEXT DEFAULT '{}'
);
"""


def _conn() -> sqlite3.Connection:
    """连一个连接。

    **变量一律叫 `db`，不叫 `c`** —— 本模块到处有叫 `c` 的形参
    （candidate），`with _conn() as db:` 会把它悄悄盖掉，
    症状是 `'sqlite3.Connection' object is not subscriptable`，
    出现在函数体中间而不是 `with` 那一行，很难一眼看出是命名冲突。
    """
    DB.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")   # 服务是多线程的，默认回滚日志会互相锁
    return db


def init() -> None:
    with _conn() as db:
        db.executescript(SCHEMA)


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def now() -> float:
    return time.time()


# ---------------------------------------------------------------- 审计

def audit(kind: str, summary: str, payload: dict | None = None) -> None:
    with _conn() as db:
        db.execute("INSERT INTO audit(ts,kind,summary,payload) VALUES(?,?,?,?)",
                  (now(), kind, summary, json.dumps(payload or {}, ensure_ascii=False)))


def list_audit(limit: int = 100) -> list[dict]:
    with _conn() as db:
        rows = db.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?",
                         (limit,)).fetchall()
    return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]


# ---------------------------------------------------------------- 岗位

def _job_row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["must"] = json.loads(d.pop("must_json"))
    d["nice"] = json.loads(d.pop("nice_json"))
    return d


def upsert_job(job: dict) -> str:
    """按 job_id 覆盖写入。批量录入时重复提交同一个岗位不该报错 ——
    HR 用 Excel 反复导出导入是常态。"""
    with _conn() as db:
        db.execute("""INSERT INTO jobs(job_id,title,dept,jd_text,must_json,nice_json,
                                     headcount,status,created_at)
                     VALUES(?,?,?,?,?,?,?,?,?)
                     ON CONFLICT(job_id) DO UPDATE SET
                       title=excluded.title, dept=excluded.dept,
                       jd_text=excluded.jd_text, must_json=excluded.must_json,
                       nice_json=excluded.nice_json, headcount=excluded.headcount,
                       status=excluded.status""",
                  (job["job_id"], job["title"], job.get("dept", ""), job["jd_text"],
                   json.dumps(job.get("must", []), ensure_ascii=False),
                   json.dumps(job.get("nice", []), ensure_ascii=False),
                   int(job.get("headcount", 1)), job.get("status", "招聘中"), now()))
    return job["job_id"]


def list_jobs() -> list[dict]:
    with _conn() as db:
        rows = db.execute("""SELECT j.*, (SELECT COUNT(*) FROM matches m
                                          WHERE m.job_id=j.job_id) AS n_matched
                            FROM jobs j ORDER BY j.created_at DESC""").fetchall()
    return [_job_row(r) for r in rows]


def get_job(job_id: str) -> dict | None:
    with _conn() as db:
        r = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    return _job_row(r) if r else None


# ---------------------------------------------------------------- 候选人

def _cand_row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["skills"] = json.loads(d.pop("skills_json"))
    return d


def upsert_candidate(c: dict) -> str:
    with _conn() as db:
        db.execute("""INSERT INTO candidates(candidate_id,name,resume_text,skills_json,
                                            source,applied_job,created_at)
                     VALUES(?,?,?,?,?,?,?)
                     ON CONFLICT(candidate_id) DO UPDATE SET
                       name=excluded.name, resume_text=excluded.resume_text,
                       skills_json=excluded.skills_json, source=excluded.source,
                       applied_job=excluded.applied_job""",
                  (c["candidate_id"], c["name"], c["resume_text"],
                   json.dumps(c.get("skills", []), ensure_ascii=False),
                   c.get("source", ""), c.get("applied_job", ""), now()))
    return c["candidate_id"]


def list_candidates() -> list[dict]:
    with _conn() as db:
        rows = db.execute("""SELECT * FROM candidates ORDER BY created_at DESC""").fetchall()
    return [_cand_row(r) for r in rows]


def get_candidate(cid: str) -> dict | None:
    with _conn() as db:
        r = db.execute("SELECT * FROM candidates WHERE candidate_id=?", (cid,)).fetchone()
    return _cand_row(r) if r else None


# ---------------------------------------------------------------- 技能缓存

def cached_skills(text: str) -> list[str] | None:
    with _conn() as db:
        r = db.execute("SELECT skills_json FROM skill_cache WHERE text_hash=?",
                      (text_hash(text),)).fetchone()
    return json.loads(r["skills_json"]) if r else None


def cache_skills(text: str, skills: list[str], tokens: int = 0) -> None:
    with _conn() as db:
        db.execute("""INSERT OR REPLACE INTO skill_cache(text_hash,skills_json,tokens,
                                                         created_at) VALUES(?,?,?,?)""",
                  (text_hash(text), json.dumps(skills, ensure_ascii=False),
                   int(tokens), now()))


# ---------------------------------------------------------------- 匹配结果

def save_matches(job_id: str, rows: list[dict], mode: str) -> None:
    """一次匹配的结果整体替换。留旧结果会让人分不清看到的是哪一次跑的。"""
    with _conn() as db:
        db.execute("DELETE FROM matches WHERE job_id=?", (job_id,))
        db.executemany("""INSERT INTO matches(job_id,candidate_id,score,must_json,
                                             must_miss,nice_json,mode,created_at)
                         VALUES(?,?,?,?,?,?,?,?)""",
                      [(job_id, r["candidate_id"], r["score"],
                        json.dumps(r.get("must_hit", []), ensure_ascii=False),
                        json.dumps(r.get("must_miss", []), ensure_ascii=False),
                        json.dumps(r.get("nice_hit", []), ensure_ascii=False),
                        mode, now()) for r in rows])


def get_matches(job_id: str) -> list[dict]:
    with _conn() as db:
        rows = db.execute("""SELECT m.*, c.name, c.skills_json, c.source
                            FROM matches m JOIN candidates c
                              ON c.candidate_id = m.candidate_id
                            WHERE m.job_id=? ORDER BY m.score DESC""", (job_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["must_hit"] = json.loads(d.pop("must_json"))
        d["must_miss"] = json.loads(d.pop("must_miss"))
        d["nice_hit"] = json.loads(d.pop("nice_json"))
        d["skills"] = json.loads(d.pop("skills_json"))
        out.append(d)
    return out


# ---------------------------------------------------------------- 概览

def stats() -> dict:
    with _conn() as db:
        g = lambda q: db.execute(q).fetchone()[0]  # noqa: E731
        by_stage = db.execute("""SELECT status, COUNT(*) n FROM jobs
                                GROUP BY status""").fetchall()
        top = db.execute("""SELECT j.title, COUNT(m.id) n FROM jobs j
                           LEFT JOIN matches m ON m.job_id=j.job_id
                           GROUP BY j.job_id HAVING n>0
                           ORDER BY n DESC LIMIT 5""").fetchall()
    return {
        "jobs": g("SELECT COUNT(*) FROM jobs"),
        "candidates": g("SELECT COUNT(*) FROM candidates"),
        "matches": g("SELECT COUNT(*) FROM matches"),
        "matched_jobs": g("SELECT COUNT(DISTINCT job_id) FROM matches"),
        "skills_known": g("SELECT COUNT(*) FROM skill_cache"),
        "audit_events": g("SELECT COUNT(*) FROM audit"),
        "job_status": {r["status"]: r["n"] for r in by_stage},
        "top_jobs": [{"title": r["title"], "n": r["n"]} for r in top],
    }
