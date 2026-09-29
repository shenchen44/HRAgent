"""五个执行体（D2）—— 每个只做一件事。

为什么不做"一个万能 agent 挂一堆工具"：
  万能 agent 的工具选择错误会与推理错误混在一起，badcase 归因不了。
  拆成 5 个之后，每个执行体有独立的评测集与独立的失败模式：
    HRDataSQL   有 text2sql_eval（EX + 口径正确率）
    PolicyRAG   有 rag_eval（recall + 引用正确性 + 拒答）
    RecruitMatch 有 match_eval / skill_eval（抽取 P/R/F1 + NDCG + 公平性）
    OnboardFlow / InterviewKit  无独立评测集，仅按路由准确率评估（已在文档声明）

共同纪律：
  · 无证据不作断言。拿不到证据就返回 ok=False 或明确说明无法回答。
  · 口径敏感指标必须走注册表，不允许自由写 SQL。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..llm import LLM, Reply
from ..schema import Evidence, ExecResult
from ..tools import caliber, retrieval, sql_tool


def _tokens(r: Reply) -> int:
    u = r.usage or {}
    return int(u.get("input_tokens", 0)) + int(u.get("output_tokens", 0))


# ---------------------------------------------------------------- 1. HRDataSQL
SQL_SYSTEM = """你是 HR 数据查询执行体。把用户问题转成对 hr.db 的查询。

你可以走两条路：
A. 调用 query_metric 工具查询**已注册指标**。口径敏感指标（带 ★）必须走这条路，
   因为分母/口径由系统固定，你自由写 SQL 极易写错且不会报错。
B. 调用 run_sql 工具写自由 SQL，仅用于注册表未覆盖的问题。

数据库表：departments(dept_id,dept_name,cost_center,headcount_budget)
employees(emp_id,name,dept_id,job_title,job_level,city,hire_date,leave_date,status,base_salary,birth_year)
recruitment(req_id,dept_id,job_title,headcount_planned,headcount_filled,open_date,close_date,status)
candidates(candidate_id,req_id,name,source,apply_date,stage,offer_date,onboard_date)
attendance(emp_id,month,work_days,actual_days,leave_days,overtime_hours)
performance(emp_id,period,rating,score)
payroll(emp_id,month,base_salary,bonus,social_insurance,housing_fund,gross_pay,net_pay)
training(emp_id,course,planned_hours,actual_hours,train_date)

注意：leave_date 为 NULL 表示在职；离职员工仍保留考勤/薪酬历史，
统计在职指标必须先过滤 status。

查完后用中文给出简洁答案，并说明用的是什么口径。"""


@dataclass
class HRDataSQL:
    name: str = "HRDataSQL"
    # 契约：本执行体断言**数据事实**，必须有出处。风控的 G1 与不确定性门控
    # 都按这个标志决定"没有证据"算不算问题。
    requires_evidence: bool = True
    required_inputs: tuple[str, ...] = ()   # 只查库，不需要用户额外交材料
    llm: LLM | None = None

    def tools(self) -> list[dict]:
        return [
            {"name": "query_metric",
             "description": "查询已注册指标（口径由系统固定）。★ 为口径敏感指标。\n"
                            + caliber.catalog(),
             "input_schema": {"type": "object", "properties": {
                 "metric": {"type": "string", "description": "指标名"},
                 "params": {"type": "object", "description":
                            "参数，如 {dept, start, end} 或 {dept, month} 或 {dept, period}"}},
                 "required": ["metric", "params"]}},
            {"name": "run_sql",
             "description": "执行自由只读 SQL（仅当注册表未覆盖时使用）",
             "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}},
                              "required": ["sql"]}},
        ]

    def run(self, query: str, slots: dict | None = None, ctx: dict | None = None) -> ExecResult:
        import time
        t0 = time.time()
        llm = self.llm or LLM()
        slot_hint = f"\n已知槽位：{json.dumps(slots, ensure_ascii=False)}" if slots else ""
        msgs = [{"role": "user", "content": f"问题：{query}{slot_hint}"}]
        evidence: list[Evidence] = []
        artifacts: dict = {"tool_calls": []}
        tok = 0

        # 最多三轮工具调用：留出"调工具 → 再调工具 → 出正文"的余量
        for _ in range(3):
            r = llm.call(msgs, system=SQL_SYSTEM, tools=self.tools(), tag="sql")
            tok += _tokens(r)
            if not r.tool_calls:
                break
            results = []
            for tc in r.tool_calls:
                name, args = tc.get("name"), tc.get("input") or {}
                artifacts["tool_calls"].append({"name": name, "input": args})
                if name == "query_metric":
                    out = self._metric(args, evidence)
                elif name == "run_sql":
                    out = self._free_sql(args, evidence, artifacts)
                else:
                    out = {"error": f"未知工具 {name}"}
                results.append({"type": "tool_result", "tool_use_id": tc.get("id"),
                                "content": json.dumps(out, ensure_ascii=False, default=str)})
            # tool_calls 就是 API 原始的 content block，可直接回传
            msgs.append({"role": "assistant", "content": r.tool_calls})
            msgs.append({"role": "user", "content": results})

        answer = r.text.strip() if r.text else ""

        # **循环可能是在"最后一次响应是工具调用"的状态下退出的。**
        # 那时 `r.text` 必然为空 —— 带 tool_calls 的响应不产出正文，
        # 而三轮预算已经被工具调用吃光了，模型根本没机会写结论。
        # 原先没有这一步，于是直接掉进下面的"退回原始工具结果"分支：
        # 实测「研发中心上季度的出勤率是多少？」有一半以上的概率走到那里，
        # 界面上就是一行 `[('2026-01', 1955.7, 2065), …]`。
        # 补一次**不带工具**的调用，逼它把已有结果写成结论。
        #
        # 注意这里**只追加一条 user 消息**：循环体在每轮末尾已经追加过
        # `assistant(tool_calls)` + `user(tool_results)`，退出时 msgs 正好以
        # 工具结果结尾。再追加一次 assistant 会得到两条带**相同 tool_use id**
        # 的助手消息，网关直接 400。
        if not answer and r.tool_calls:
            msgs.append({"role": "user", "content": [{
                "type": "text", "text": "请基于以上工具结果，直接给出最终结论。"}]})
            r2 = llm.call(msgs, system=SQL_SYSTEM, tag="sql_final")
            tok += _tokens(r2)
            answer = r2.text.strip() if r2.text else ""

        # 仍然没正文 → 退回原始工具结果。**这是降级，不是成功。**
        #
        # 原先这里走完之后 `ok=True`、`confidence=0.8`，于是四道闸门全过：
        # 截断闸门看的是**最终答案**，而最终答案此刻就是这段拼接文本（它没被截断，
        # 它本来就是拼接的）；证据接地闸门看到 evidence 齐全，也判通过。
        # 结果是 `出勤率 = [(None,)]%；[('2026-01', 1955.7, 2065), …]`
        # 被当成一条 0.8 置信度的答案交给 HR。
        degraded = not answer and bool(evidence)
        if degraded:
            answer = ("未能生成结论（模型未产出正文）。以下为原始查询结果，供人工核对：\n"
                      + "；".join(e.value for e in evidence))
        ok = bool(evidence) and bool(answer)
        # 降级答案的置信度必须**低于升级阈值 0.3**，让它走人工复核。
        # 数据不丢（仍在 evidence 里，界面照常显示），但不再冒充可信答案。
        conf = 0.2 if degraded else (0.8 if ok else 0.2)
        return ExecResult(
            executor=self.name, ok=ok, answer=answer, evidence=evidence,
            confidence=conf, artifacts=artifacts,
            latency_s=time.time() - t0, tokens=tok,
            error=("未产出正文，已退回原始工具结果" if degraded
                   else None if ok else
                   ("未产生证据" if not evidence else "未产生答案")),
        )

    def _metric(self, args: dict, evidence: list[Evidence]) -> dict:
        name = args.get("metric", "")
        params = args.get("params") or {}
        try:
            res = caliber.run(name, **params)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}", "hint": caliber.catalog()}
        val = res["value"] if res["value"] is not None else res["rows"]
        evidence.append(Evidence(
            kind="sql_result", ref=f"hr.db#{res['metric']}",
            value=f"{res['label']} = {val}{res['unit']}", source_query=res["sql"]))
        return {"metric": res["metric"], "label": res["label"], "value": val,
                "unit": res["unit"], "definition": res["definition"], "rows": res["rows"]}

    def _free_sql(self, args: dict, evidence: list[Evidence], artifacts: dict) -> dict:
        sql = args.get("sql", "")
        res = sql_tool.run_sql(sql)
        artifacts["free_sql"] = sql
        artifacts["full_scan"] = sql_tool.is_full_scan(sql)
        artifacts["sensitive_hits"] = sql_tool.touches_sensitive(sql)
        if res.rejected:
            return {"rejected": res.rejected}
        if res.error:
            return {"error": res.error}
        evidence.append(Evidence(kind="sql_result", ref="hr.db#free_sql",
                                 value=str(res.rows[:5]), source_query=res.sql))
        return {"columns": res.columns, "rows": res.rows[:20], "n_rows": len(res.rows)}


# ---------------------------------------------------------------- 2. PolicyRAG
RAG_SYSTEM = """你是制度问答执行体。只依据**检索到的制度片段**回答。

铁律：
1. 每个断言后面必须标注来源片段号，格式 [chunk_id]。
2. 检索片段若不支持该问题，必须明确说"制度中没有相关规定"，**不得**用常识补充。
3. 若片段只规定了一部分，须说明"制度只规定了 X，未涉及 Y"。
4. 不得把片段里没有的数字、比例、天数写进答案。

回答后用一行给出置信度：CONFIDENCE: <0-1>"""


@dataclass
class PolicyRAG:
    name: str = "PolicyRAG"
    requires_evidence: bool = True   # 断言**制度事实**，必须有出处
    required_inputs: tuple[str, ...] = ()   # 只检索，不需要用户额外交材料
    llm: LLM | None = None
    k: int = 5

    def run(self, query: str, slots: dict | None = None, ctx: dict | None = None) -> ExecResult:
        import time
        t0 = time.time()
        llm = self.llm or LLM()
        idx = retrieval.index()
        hits = idx.search(query, self.k)
        if not hits:
            return ExecResult(executor=self.name, ok=False, answer="制度库中未检索到相关内容。",
                              confidence=0.1, error="无检索结果", latency_s=time.time() - t0)

        spans = "\n\n".join(f"[{h['chunk_id']}] 《{h['doc_title']}》{h['heading']}\n{h['text']}"
                            for h in hits)
        prompt = f"检索到的制度片段：\n\n{spans}\n\n用户问题：{query}"
        r = llm.call([{"role": "user", "content": prompt}], system=RAG_SYSTEM, tag="rag")
        text = r.text.strip()

        conf = 0.5
        if "CONFIDENCE:" in text:
            body, _, tail = text.rpartition("CONFIDENCE:")
            try:
                conf = max(0.0, min(1.0, float(tail.strip().split()[0])))
                text = body.strip()
            except (ValueError, IndexError):
                pass

        cited = [h["chunk_id"] for h in hits if h["chunk_id"] in text]
        evidence = [Evidence(kind="doc_span", ref=h["chunk_id"],
                             value=h["text"][:120], source_query=query)
                    for h in hits if h["chunk_id"] in cited]
        return ExecResult(
            executor=self.name, ok=True, answer=text, evidence=evidence,
            confidence=conf,
            artifacts={"retrieved": [h["chunk_id"] for h in hits],
                       "retrieved_refs": [h["chunk_id"] for h in hits],
                       "cited": cited,
                       "top_scores": [round(h["score"], 2) for h in hits]},
            latency_s=time.time() - t0, tokens=_tokens(r),
        )


# ---------------------------------------------------------------- 3. RecruitMatch
MATCH_SYSTEM = """你是人岗匹配执行体。给定 JD 与候选人简历：

1. 先抽取双方的技能集合（只抽**真正掌握**的技能；"了解过""未深入使用"不算）。
2. 按 must-have / nice-to-have 计算覆盖情况。
3. **严禁**使用性别、年龄、婚育、户籍、外貌等与岗位无关的个人信息作为判断依据。
   若 JD 或简历中出现此类信息，忽略它们并在 warnings 中说明。
4. 输出 JSON：{"candidate_skills":[...], "must_have":[...], "must_covered":[...],
   "must_missing":[...], "nice_covered":[...], "score":0-100, "reason":"...",
   "warnings":[...]}"""


@dataclass
class RecruitMatch:
    name: str = "RecruitMatch"
    # 生成型：产出的是**匹配建议**，不是可溯源的事实断言。
    # 契约上本就无证据可引 —— 把"无证据"当成风险信号是误报，不是风控。
    requires_evidence: bool = False
    # **本执行体是唯一需要用户额外交材料的**：没有 JD 与简历就无从匹配。
    # 缺了要问用户要，而不是升级人工 —— 人工手里同样没有这两样东西。
    required_inputs: tuple[str, ...] = ("jd", "resume")
    llm: LLM | None = None

    def run(self, query: str, slots: dict | None = None, ctx: dict | None = None) -> ExecResult:
        import time
        t0 = time.time()
        llm = self.llm or LLM()
        ctx = ctx or {}
        jd, resume = ctx.get("jd", ""), ctx.get("resume", "")
        if not jd or not resume:
            # 缺哪个报哪个：编排层据此生成**具体**的澄清问题
            # （"请提供简历" 比 "能否补充信息" 有用得多）。
            missing = [k for k, v in (("jd", jd), ("resume", resume)) if not v]
            return ExecResult(executor=self.name, ok=False,
                              answer="需要同时提供 JD 与候选人简历。",
                              confidence=0.2, error="缺少 jd 或 resume",
                              missing_inputs=missing, latency_s=time.time() - t0)

        prompt = f"【JD】\n{jd}\n\n【候选人简历】\n{resume}"
        r = llm.call([{"role": "user", "content": prompt}], system=MATCH_SYSTEM, tag="match")
        try:
            data = r.json()
        except Exception as e:
            return ExecResult(executor=self.name, ok=False, answer=r.text.strip(),
                              confidence=0.2, error=f"JSON 解析失败: {e}",
                              latency_s=time.time() - t0, tokens=_tokens(r))

        must, covered = data.get("must_have", []), data.get("must_covered", [])
        ev = [Evidence(kind="rule", ref="must_have_coverage",
                       value=f"{len(covered)}/{len(must)}", source_query="skill_match")]
        if data.get("must_missing"):
            ev.append(Evidence(kind="rule", ref="must_missing",
                               value="、".join(data["must_missing"]), source_query="skill_match"))
        answer = (f"匹配度 {data.get('score', '?')}/100。"
                  f"必备技能覆盖 {len(covered)}/{len(must)}"
                  + (f"，缺失：{'、'.join(data['must_missing'])}" if data.get("must_missing") else "")
                  + f"。{data.get('reason', '')}")
        return ExecResult(
            executor=self.name, ok=True, answer=answer, evidence=ev,
            confidence=0.7, artifacts={"match": data},
            latency_s=time.time() - t0, tokens=_tokens(r),
        )


# ---------------------------------------------------------------- 4. OnboardFlow
ONBOARD_SYSTEM = """你是入职流程执行体。根据岗位与入职日期，产出一份可执行的入职清单。

要求：
1. 按时间倒序分阶段：入职前 / 入职当天 / 入职首周 / 试用期内。
2. 每项写清**责任人**（HR / 直属上级 / IT / 行政 / 员工本人）与**时限**。
3. 制度依据来自检索片段，标注 [chunk_id]；无依据的项标 [通用实践] 以区分。
4. 输出 Markdown 清单。"""


@dataclass
class OnboardFlow:
    name: str = "OnboardFlow"
    requires_evidence: bool = False   # 生成型：产出的是**流程建议**
    required_inputs: tuple[str, ...] = ()   # 流程清单从岗位/日期即可生成
    llm: LLM | None = None

    def run(self, query: str, slots: dict | None = None, ctx: dict | None = None) -> ExecResult:
        import time
        t0 = time.time()
        llm = self.llm or LLM()
        hits = retrieval.index().search(f"入职 试用期 劳动合同 材料 {query}", 4)
        spans = "\n\n".join(f"[{h['chunk_id']}] {h['text'][:300]}" for h in hits)
        prompt = (f"检索到的制度依据：\n\n{spans}\n\n"
                  f"任务：{query}\n槽位：{json.dumps(slots or {}, ensure_ascii=False)}")
        r = llm.call([{"role": "user", "content": prompt}], system=ONBOARD_SYSTEM, tag="onboard")
        cited = [h["chunk_id"] for h in hits if h["chunk_id"] in r.text]
        return ExecResult(
            executor=self.name, ok=bool(r.text.strip()), answer=r.text.strip(),
            evidence=[Evidence(kind="doc_span", ref=c, value="入职流程依据", source_query=query)
                      for c in cited],
            confidence=0.6, artifacts={"retrieved": [h["chunk_id"] for h in hits],
                                       "retrieved_refs": [h["chunk_id"] for h in hits]},
            latency_s=time.time() - t0, tokens=_tokens(r),
        )


# ---------------------------------------------------------------- 5. InterviewKit
INTERVIEW_SYSTEM = """你是面试题生成执行体。根据 JD 产出结构化面试题。

要求：
1. 按维度分组：专业能力 / 项目经验 / 协作沟通 / 动机匹配。
2. 每题给出**考察点**与**追问方向**，并标注难度（初/中/高）。
3. **严禁**生成涉及性别、年龄、婚育、户籍、外貌、健康状况等与岗位无关的问题。
4. 每题尽量锚定 JD 中的具体技能要求。输出 Markdown。"""


@dataclass
class InterviewKit:
    name: str = "InterviewKit"
    requires_evidence: bool = False   # 生成型：产出的是**面试题建议**
    required_inputs: tuple[str, ...] = ()   # ctx 里没有 jd 时回退用 query，不会缺输入
    llm: LLM | None = None

    def run(self, query: str, slots: dict | None = None, ctx: dict | None = None) -> ExecResult:
        import time
        t0 = time.time()
        llm = self.llm or LLM()
        ctx = ctx or {}
        jd = ctx.get("jd") or query
        r = llm.call([{"role": "user", "content": f"【JD】\n{jd}\n\n任务：{query}"}],
                     system=INTERVIEW_SYSTEM, tag="interview")
        return ExecResult(
            executor=self.name, ok=bool(r.text.strip()), answer=r.text.strip(),
            evidence=[Evidence(kind="rule", ref="jd_grounded", value="题目锚定 JD 技能要求",
                               source_query=query)],
            confidence=0.6, artifacts={}, latency_s=time.time() - t0, tokens=_tokens(r),
        )


REGISTRY: dict[str, type] = {
    "HRDataSQL": HRDataSQL, "PolicyRAG": PolicyRAG, "RecruitMatch": RecruitMatch,
    "OnboardFlow": OnboardFlow, "InterviewKit": InterviewKit,
}


_INPUT_LABEL = {"jd": "岗位 JD", "resume": "候选人简历"}


def required_inputs(executor: str) -> tuple[str, ...]:
    """该执行体按契约需要的用户输入（ctx 里的键）。

    **与 `requires_evidence` 同一套纪律：契约由生产端声明，消费端不另立名单。**
    缺省取空元组（不要求额外输入）—— 这里的方向与证据契约**相反**：
    证据那边缺省要 True（宁吵勿哑），输入这边缺省要空
    （多问一句就是多一次打扰用户，误伤比漏放贵）。
    """
    cls = REGISTRY.get(executor)
    return tuple(getattr(cls, "required_inputs", ()) or ()) if cls else ()


def input_label(key: str) -> str:
    return _INPUT_LABEL.get(key, key)


def requires_evidence(executor: str) -> bool:
    """该执行体是否按契约必须带证据。

    **契约由生产端（执行体类属性）声明，不在这里另立一张名单。**
    原先这张名单硬编码在编排层（`graph.EVIDENCE_REQUIRED`），后果是：
    新增执行体时没人会想起来去改它，缺省行为（不在名单里 → 不要求证据）
    又恰好是"放行"，于是漏配**静默降级为无风控**，不报错。

    缺省取 True（保守）：名字不在注册表里时宁可多要求证据 ——
    这个方向的错误会吵，上一个方向的错误会哑。
    """
    cls = REGISTRY.get(executor)
    return bool(getattr(cls, "requires_evidence", True)) if cls else True
