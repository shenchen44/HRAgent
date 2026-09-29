"""HR 系统后端 —— 只依赖标准库，不引入 Web 框架。

为什么不用 FastAPI/Flask：本项目已经装了 torch 这一层重依赖，
再加一个框架只会让评审多一步安装。而这里的接口一共十来个、
没有鉴权、没有 ORM、没有 WebSocket —— `ThreadingHTTPServer` 足够，
且**零新增依赖**意味着 `git clone` 后 `.venv/bin/python scripts/serve_ui.py`
就能跑起来。

**接线纪律（本项目踩过的第 8 例 bug）**：`build_graph` 里 `screen`（B4 请求级
合规判据）与 `ask_inputs`（B5 缺输入→澄清）的**缺省是关闭**的 ——
那是为了让 exp9 的 B0~B3 历史结果可比。`scripts/demo.py` 曾经顺延了这个缺省，
于是演示里"帮我筛掉所有非 985 的简历"被反问"要筛哪个岗位"，
等于请用户补材料以完成一次歧视性筛选。
**所以本文件显式传参打开这两层，并且 `smoke_graph.py` 有对应的回归项。**
缺省值服务的是消融实验，不是产品。

跑法:
  .venv/bin/python scripts/serve_ui.py            # http://127.0.0.1:8848
  .venv/bin/python scripts/serve_ui.py --port 9000
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hragent import config                                  # noqa: E402
from hragent.agents.orchestrator import Orchestrator        # noqa: E402
from hragent.llm import LLM                                 # noqa: E402
from hragent.orchestrator import HRGraph                    # noqa: E402
from hragent.risk.guards import RiskAgent                   # noqa: E402
from hragent.risk.screen import QueryScreen                 # noqa: E402
from hragent.tools.skills import SkillAligner               # noqa: E402
from hragent.ui import matching, store                      # noqa: E402

WEB = ROOT / "web"
MATCH_EVAL = config.EVAL / "match_eval.jsonl"

# ---------------------------------------------------------------- 全局状态

# 技能对齐器持有嵌入模型，加载一次约数秒 —— 每个请求新建会把接口拖垮。
_aligner: SkillAligner | None = None
_aligner_lock = threading.Lock()

# 后台任务表（批量导入用）。导入 100+ 份简历要逐条调 LLM，同步做会让
# 浏览器等到超时且全程无反馈；改成任务 + 轮询。
_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=2)

# LangGraph 的记忆检查点 + 人工介入后的续跑表
_graph: HRGraph | None = None
_graph_lock = threading.Lock()


def aligner() -> SkillAligner:
    global _aligner
    if _aligner is None:
        with _aligner_lock:
            if _aligner is None:
                _aligner = SkillAligner()
    return _aligner


def graph() -> HRGraph:
    """构建编排图。**B4/B5 两层显式打开**，理由见模块 docstring。"""
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                _graph = HRGraph(
                    orch=Orchestrator(llm=LLM()),
                    risk=RiskAgent(enabled=("truncation", "grounding", "compliance",
                                            "uncertainty")),
                    screen=QueryScreen(enabled=True),
                    ask_inputs=True,
                )
    return _graph


def task_new(kind: str, total: int) -> str:
    tid = uuid.uuid4().hex[:12]
    with _tasks_lock:
        _tasks[tid] = {"id": tid, "kind": kind, "state": "running", "done": 0,
                       "total": total, "message": "", "result": None, "error": None}
    return tid


def task_update(tid: str, **kw) -> None:
    with _tasks_lock:
        _tasks[tid].update(kw)


def task_get(tid: str) -> dict | None:
    with _tasks_lock:
        return dict(_tasks[tid]) if tid in _tasks else None


# ---------------------------------------------------------------- 业务逻辑

def import_resumes(rows: list[dict], tid: str, use_llm: bool = True) -> None:
    """批量录入简历：逐条抽技能并落库。

    技能抽取结果按**简历文本哈希**缓存 —— 同一个人投多个岗位、
    或者评审重跑一次导入，都不会重复花 LLM 调用。
    """
    store.init()
    llm = LLM()
    done = 0
    for r in rows:
        text = r["resume_text"]
        skills = store.cached_skills(text)
        tokens = 0
        if skills is None and use_llm:
            try:
                skills, tokens = matching.extract_resume_skills(text, llm)
            except Exception as e:  # noqa: BLE001
                skills = []
                task_update(tid, message=f"{r['name']} 抽取失败：{e}")
            # **只有真的跑过抽取才写缓存。** 关掉抽取时写 `[]` 会把
            # "没抽过" 记成 "抽过了，结果是空" —— 之后再开着抽取重跑，
            # 缓存命中，这批人永远拿不到技能，而且不报错。
            store.cache_skills(text, skills, tokens)
        elif skills is None:
            skills = []
        store.upsert_candidate({**r, "skills": skills})
        done += 1
        task_update(tid, done=done, message=f"已录入 {r['name']}（{len(skills)} 项技能）")
    s = llm.ledger.summary()
    store.audit("candidate.import", f"批量录入简历 {done} 份",
                {"tokens": s["total_tokens"], "calls": s["calls"]})
    task_update(tid, state="done", result={"imported": done,
                                           "tokens": s["total_tokens"]})


def do_match(job_id: str, top_k: int, rerank: bool) -> dict:
    job = store.get_job(job_id)
    if not job:
        raise KeyError(f"岗位不存在：{job_id}")
    cands = store.list_candidates()
    if not cands:
        raise ValueError("候选人库为空，请先录入简历")
    rows = matching.recall(job, cands, aligner())
    rerank_tokens = 0
    if rerank:
        # 精排只对**圈定的前 top_k** 跑：exp6 说它买不到排序质量（p=0.6430），
        # 所以它不能是默认路径，只能是被明确要求的一次复核。
        by_id = {c["candidate_id"]: c for c in cands}
        llm = LLM()
        rows = matching.rerank(job, rows[:top_k], by_id, llm)
        rerank_tokens = llm.ledger.summary()["total_tokens"]
    store.save_matches(job_id, rows, "精排" if rerank else "召回")
    store.audit("match.run",
                f"{job['title']}：{'精排' if rerank else '召回'} "
                f"{len(rows)} 位候选人，取前 {top_k}",
                {"job_id": job_id, "top_k": top_k, "rerank": rerank,
                 "rerank_tokens": rerank_tokens})
    return {"job": job, "top": rows[:top_k], "total": len(rows),
            "rerank_tokens": rerank_tokens,
            "fairness": matching.FAIRNESS_NOTE,
            "alias_note": matching.ALIAS_NOTE,
            "tau": matching.TAU}


def _trace_payload(tr, st: dict, thread_id: str) -> dict:
    interrupted = bool(st.get("__interrupt__"))
    # 裁决分两级：`n_review` 判**执行体结果**，`n_final_review` 判**汇总后的答案**
    # （答案是新生成的文本，可能引入执行体结果里没有的违规）。
    # 两级都追加进同一个 `verdicts` 列表，所以同名的闸门会出现两次 ——
    # 那不是重复执行，是两处不同的检查。图上必须分开标，
    # 否则评审看到四个闸门各出现两遍，只会读成"这里有 bug"。
    n_exec = st.get("n_exec_verdicts")
    verdicts = [{"guard": v.guard, "passed": v.passed, "severity": v.severity,
                 "reason": v.reason} for v in tr.verdicts]
    for i, v in enumerate(verdicts):
        v["stage"] = ("执行体" if n_exec is None or i < n_exec else "答案")
    out = {
        "answer": tr.final_answer,
        "escalated": bool(tr.escalated),
        "escalation_reason": tr.escalation_reason,
        "needs_review": bool(tr.needs_review),
        "interrupted": interrupted,
        "thread_id": thread_id,
        "route": {
            "executors": (tr.route.executors if tr.route else []),
            "ambiguity": (tr.route.ambiguity if tr.route else None),
            "needs_clarification": bool(tr.route and tr.route.needs_clarification),
        },
        "screen": tr.screen or {},
        "verdicts": verdicts,
        # 人工处置结果。`n_hitl` 只把它记进 state，**不改写答案** ——
        # 这个分离是对的：答案归系统，处置归人。但界面上必须把它显示出来，
        # 否则用户点完"拦截"看到的还是那句"已转交人工处理"，
        # 会以为自己的操作没生效。
        "human": st.get("human"),
        "evidence": [{"kind": e.kind, "ref": e.ref, "value": str(e.value)[:200]}
                     for r in tr.results for e in (r.evidence or [])],
        "cost": {"tokens": tr.tokens, "latency_s": tr.latency_s},
    }
    if interrupted:
        out["interrupt"] = st["__interrupt__"][0].value
    return out


def do_assistant(query: str, thread_id: str | None = None) -> dict:
    """走完整的编排链路，并把风控/观测信息一并返回。

    返回 trace 是刻意的：HR 场景里"系统为什么拒答""为什么转人工"
    必须能被解释，只给一个答案是不能上线的。
    """
    tid = thread_id or f"ui{int(__import__('time').time() * 1000)}"
    tr, st = graph().run(query, thread_id=tid)
    out = _trace_payload(tr, st, tid)
    # `thread_id` 必须一起落盘。人工处置那条审计（`assistant.resume`）只带
    # thread_id，查询这条原先不带 —— 于是"处置"能指回线程、"请求"却不能，
    # 事后归因时只能靠时间戳猜是哪一问被处置了。
    # 审计日志的全部意义就是事后能把"谁、对哪一问、做了什么"串起来，
    # 单向的链接等于没有链接。
    store.audit("assistant.query", query[:80],
                {"answer": tr.final_answer[:200], "escalated": bool(tr.escalated),
                 "executors": out["route"]["executors"],
                 "screen": out["screen"].get("action"),
                 "thread_id": tid})
    return out


def do_resume(thread_id: str, decision: dict) -> dict:
    """人工介入的续跑：把人的处置结果送回暂停点。

    没有这个接口，`escalated` 就只是一个**显示用的标记** ——
    系统说"已转人工"，但人工永远处置不了。这正是本项目反复出现的
    "声明了却没接线"那一类问题，所以它必须真的接上。
    """
    tr, st = graph().resume(thread_id, decision)
    store.audit("assistant.resume", f"人工处置：{decision.get('action', '?')}",
                {"thread_id": thread_id, "decision": decision})
    return _trace_payload(tr, st, thread_id)



# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "hragent-ui"

    def log_message(self, fmt, *args):  # noqa: A002
        # 默认会把每个请求打到 stderr，静态资源一多就淹没真正的日志
        if "/api/" in (self.path or ""):
            sys.stderr.write(f"  {self.command} {self.path} → {args[1]}\n")

    # -- 工具

    def _json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code: int, msg: str) -> None:
        self._json({"error": msg}, code)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode())

    def _need(self, b: dict, *keys) -> bool:
        """必填字段校验。缺字段是**客户端**错误，该回 400 而不是 500 ——
        500 会让调用方以为是服务端崩了，去查日志、重启进程，全是在查一个不存在的问题。
        """
        missing = [k for k in keys if not b.get(k)]
        if missing:
            self._err(400, "缺少必填字段：" + "、".join(missing))
            return False
        return True

    def _static(self, rel: str) -> None:
        p = (WEB / rel.lstrip("/")).resolve()
        # 目录穿越：`..` 解析后必须仍在 web/ 内。
        # 用 `is_relative_to` 而不是 `str(p).startswith(str(WEB))` ——
        # 后者对 `/path/web-evil/x` 也会通过，因为它是字符串前缀而不是路径前缀。
        if not p.is_relative_to(WEB.resolve()) or not p.is_file():
            self._err(404, "not found")
            return
        body = p.read_bytes()
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or p.suffix in (".js", ".css"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # -- 路由

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path == "/api/overview":
                return self._json({**store.stats(),
                                   "fairness": matching.FAIRNESS_NOTE,
                                   "tau": matching.TAU})
            if path == "/api/jobs":
                return self._json(store.list_jobs())
            if path == "/api/candidates":
                return self._json(store.list_candidates())
            if path == "/api/audit":
                return self._json(store.list_audit(200))
            if path.startswith("/api/jobs/"):
                jid = path[len("/api/jobs/"):]
                if jid.endswith("/matches"):
                    jid = jid[:-len("/matches")]
                    j = store.get_job(jid)
                    if not j:
                        return self._err(404, "岗位不存在")
                    # 派生量现算，不读库里存的 —— 见 matching.decorate 的说明
                    return self._json(matching.decorate(store.get_matches(jid), j))
                j = store.get_job(jid)
                return self._json(j) if j else self._err(404, "岗位不存在")
            if path.startswith("/api/candidates/"):
                c = store.get_candidate(path[len("/api/candidates/"):])
                return self._json(c) if c else self._err(404, "候选人不存在")
            if path.startswith("/api/tasks/"):
                t = task_get(path[len("/api/tasks/"):])
                return self._json(t) if t else self._err(404, "任务不存在")
            return self._err(404, "not found")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._err(500, f"{type(e).__name__}: {e}")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            b = self._body()

            if path == "/api/jobs":
                store.init()
                jobs = b.get("jobs") or []
                if not jobs:
                    return self._err(400, "jobs 不能为空")
                for j in jobs:
                    store.upsert_job(j)
                store.audit("job.import", f"录入岗位 {len(jobs)} 个",
                            {"ids": [j["job_id"] for j in jobs]})
                return self._json({"imported": len(jobs)})

            if path == "/api/jobs/extract":
                # 手写 JD → must/nice。评测集句式走零成本正则路，其余走 LLM。
                llm = LLM()
                reqs = matching.extract_job_skills({"jd_text": b.get("jd_text", "")}, llm)
                return self._json({**reqs,
                                   "tokens": llm.ledger.summary()["total_tokens"]})

            if path == "/api/jobs/screen":
                # 合规检测：把 JD 文本当查询过一遍请求级判据。
                # 这是产品侧的用法 —— 判据原本拦的是用户输入，
                # 而"岗位要求里写了性别年龄"是同一类风险的另一种来源。
                v = QueryScreen(enabled=True).check(b.get("jd_text", ""))
                store.audit("job.screen",
                            f"岗位合规检测：{v.action}",
                            {"category": v.category, "reason": v.reason})
                return self._json({"action": v.action, "category": v.category,
                                   "reason": v.reason})

            if path == "/api/candidates":
                store.init()
                rows = b.get("candidates") or []
                tid = task_new("import_candidates", len(rows))
                _pool.submit(_guard, import_resumes, rows, tid, b.get("use_llm", True))
                return self._json({"task_id": tid, "total": len(rows)})

            if path == "/api/match":
                if not self._need(b, "job_id"):
                    return
                return self._json(do_match(b["job_id"], int(b.get("top_k", 10)),
                                           bool(b.get("rerank"))))

            if path == "/api/assistant":
                if not self._need(b, "query"):
                    return
                return self._json(do_assistant(b["query"]))

            if path == "/api/assistant/resume":
                if not self._need(b, "thread_id"):
                    return
                return self._json(do_resume(b["thread_id"], b.get("decision") or {}))

            if path == "/api/seed":
                jobs, cands = matching.candidates_from_match_eval(
                    MATCH_EVAL, int(b.get("limit_jobs", 0)))
                for j in jobs:
                    store.upsert_job(j)
                tid = task_new("seed", len(cands))
                _pool.submit(_guard, import_resumes, cands, tid,
                             b.get("use_llm", True))
                return self._json({"task_id": tid, "jobs": len(jobs),
                                   "candidates": len(cands)})

            return self._err(404, "not found")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._err(500, f"{type(e).__name__}: {e}")


def _guard(fn, *a) -> None:
    """后台任务里未捕获的异常必须落到任务状态上。

    否则线程静默死掉，前端永远停在"进行中" —— 一个不会报错的挂起，
    比直接失败更难查。
    """
    tid = a[1] if len(a) > 1 else None
    try:
        fn(*a)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        if isinstance(tid, str):
            task_update(tid, state="error", error=f"{type(e).__name__}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8848)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    store.init()
    print("=" * 68)
    print("HR Agent 系统 · 前端")
    print("=" * 68)
    print(f"  http://{args.host}:{args.port}")
    print(f"  数据库 {store.DB}（与冻结的 data/hr.db 分开）")
    print(f"  对齐阈值 τ={matching.TAU}（exp6 在 dev 上选出）")
    print("  B4 请求级合规判据：开    B5 缺输入→澄清：开")
    print("  Ctrl-C 停止")
    print("=" * 68)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
