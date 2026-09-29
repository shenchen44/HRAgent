"""探测 ocean-code 网关的 Agent 关键能力，决定主链路能否按 JD 的"动态工具调用"来做。

跑法: .venv/bin/python scripts/probe_api.py
"""

import json
import os
import time

import httpx

BASE_URL = os.environ["ANTHROPIC_BASE_URL"]
TOKEN = os.environ["ANTHROPIC_AUTH_TOKEN"]
MODEL = os.environ.get("ANTHROPIC_MODEL", "deepseek-v4.1-flash")

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}


def call(payload: dict, timeout: float = 120.0) -> dict:
    payload.setdefault("model", MODEL)
    t0 = time.time()
    r = httpx.post(f"{BASE_URL}/v1/messages", headers=HEADERS, json=payload, timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    body = r.json()
    body["_latency_s"] = round(dt, 2)
    return body


def show(title: str, body: dict) -> None:
    print(f"\n{'=' * 70}\n### {title}")
    print(f"stop_reason={body.get('stop_reason')}  latency={body['_latency_s']}s  usage={body.get('usage')}")
    for blk in body.get("content", []):
        t = blk.get("type")
        if t == "thinking":
            txt = blk.get("thinking", "")
            print(f"  [thinking] {len(txt)} chars :: {txt[:100]!r}")
        elif t == "text":
            print(f"  [text] {blk['text'][:400]!r}")
        elif t == "tool_use":
            print(f"  [tool_use] name={blk['name']} input={json.dumps(blk['input'], ensure_ascii=False)}")
        else:
            print(f"  [{t}] {json.dumps(blk, ensure_ascii=False)[:200]}")


# ---------------------------------------------------------------- 1. 工具调用
TOOLS = [
    {
        "name": "query_hr_database",
        "description": "查询 HR 数据库。当用户询问编制、HC、考勤、绩效、离职率、薪酬等需要真实数据的 HR 问题时使用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "要执行的 SQLite 查询语句"},
                "reason": {"type": "string", "description": "为什么需要这条查询"},
            },
            "required": ["sql", "reason"],
        },
    },
    {
        "name": "search_policy_docs",
        "description": "检索公司制度文档（员工手册、假期规则、社保公积金、竞业限制等）。当用户询问政策规定时使用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词"},
                "top_k": {"type": "integer", "description": "返回条数"},
            },
            "required": ["query"],
        },
    },
]


def probe_tool_use():
    body = call({
        "max_tokens": 2048,
        "tools": TOOLS,
        "tool_choice": {"type": "any"},
        "messages": [{"role": "user", "content": "帮我查一下研发部这个季度还有几个 HC，另外年假没休完能不能顺延到明年？"}],
    })
    show("1. 工具调用（多工具选择）", body)
    calls = [b for b in body["content"] if b["type"] == "tool_use"]
    print(f"  >>> 调用数={len(calls)}  名称={[c['name'] for c in calls]}")
    return body, calls


# ------------------------------------------------------- 2. tool_result 回灌
def probe_tool_roundtrip(prev_body: dict, calls: list):
    if not calls:
        print("\n### 2. 跳过（上一步没产生 tool_use）")
        return None
    results = []
    for c in calls:
        fake = '{"hc_remaining": 3, "dept": "研发部"}' if c["name"] == "query_hr_database" else "年假可顺延至次年3月31日，逾期作废。"
        results.append({"type": "tool_result", "tool_use_id": c["id"], "content": fake})
    body = call({
        "max_tokens": 2048,
        "tools": TOOLS,
        "messages": [
            {"role": "user", "content": "帮我查一下研发部这个季度还有几个 HC，另外年假没休完能不能顺延到明年？"},
            {"role": "assistant", "content": prev_body["content"]},
            {"role": "user", "content": results},
        ],
    })
    show("2. tool_result 回灌 + 汇总输出", body)
    return body


# --------------------------------------------------------- 3. 结构化 JSON
def probe_json():
    body = call({
        "max_tokens": 2048,
        "messages": [{
            "role": "user",
            "content": (
                "把这句话解析成 JSON，只输出 JSON，不要任何解释、不要 markdown 代码块：\n"
                '"招高级后端工程师，5年以上经验，精通 Go 和 Kubernetes，base 上海，薪资 40-60K。"\n'
                '字段: title(str), years_min(int), skills(list[str]), city(str), salary_min_k(int), salary_max_k(int)'
            ),
        }],
    })
    show("3. 结构化 JSON 输出", body)
    text = "".join(b.get("text", "") for b in body["content"] if b["type"] == "text")
    try:
        parsed = json.loads(text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        print(f"  >>> JSON 解析成功: {parsed}")
    except Exception as e:
        print(f"  >>> JSON 解析失败: {e}\n      原文: {text[:300]!r}")


# --------------------------------------------------- 4. thinking 与 max_tokens
def probe_thinking_budget():
    for mt in (256, 1024):
        body = call({
            "max_tokens": mt,
            "messages": [{"role": "user", "content": "一个部门原有 24 人，离职 3 人，又入职 5 人，现在离职率是多少？给出百分比。"}],
        })
        n_think = sum(len(b.get("thinking", "")) for b in body["content"] if b["type"] == "thinking")
        n_text = sum(len(b.get("text", "")) for b in body["content"] if b["type"] == "text")
        print(f"\n### 4. max_tokens={mt}: stop_reason={body.get('stop_reason')} "
              f"thinking={n_think}chars text={n_text}chars out_tokens={body['usage']['output_tokens']}")
        if body.get("stop_reason") == "max_tokens":
            print("  >>> ⚠️ 截断：thinking 吃掉了预算，正文没出完")


if __name__ == "__main__":
    print(f"model={MODEL}  base={BASE_URL}")
    body, calls = probe_tool_use()
    probe_tool_roundtrip(body, calls)
    probe_json()
    probe_thinking_budget()
