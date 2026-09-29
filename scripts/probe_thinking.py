"""探测 thinking 预算控制能力：能否关闭/限制 thinking，以及完成任务所需的真实预算。

这决定了 Harness 的预算策略，是"隔离不确定性"的第一道工程手段。
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

QUESTION = "一个部门原有 24 人，离职 3 人，又入职 5 人，现在离职率是多少？只回答一个百分比数字。"


def call(payload: dict, timeout: float = 180.0):
    payload.setdefault("model", MODEL)
    t0 = time.time()
    r = httpx.post(f"{BASE_URL}/v1/messages", headers=HEADERS, json=payload, timeout=timeout)
    dt = time.time() - t0
    if r.status_code != 200:
        return {"_error": f"HTTP {r.status_code}: {r.text[:300]}", "_latency_s": round(dt, 2)}
    b = r.json()
    b["_latency_s"] = round(dt, 2)
    return b


def summarize(tag: str, b: dict) -> None:
    if "_error" in b:
        print(f"{tag:38s} ❌ {b['_error']}")
        return
    think = sum(len(x.get("thinking", "")) for x in b.get("content", []) if x["type"] == "thinking")
    text = "".join(x.get("text", "") for x in b.get("content", []) if x["type"] == "text")
    u = b["usage"]
    print(f"{tag:38s} stop={b.get('stop_reason'):12s} think={think:5d}c text={len(text):4d}c "
          f"out={u['output_tokens']:5d}tok {b['_latency_s']:5.1f}s | {text[:60]!r}")


print(f"model={MODEL}\n\n--- A. 是否支持 thinking 参数控制 ---")
for tag, extra in [
    ("thinking disabled", {"thinking": {"type": "disabled"}}),
    ("thinking budget=512", {"thinking": {"type": "enabled", "budget_tokens": 512}}),
    ("thinking budget=2048", {"thinking": {"type": "enabled", "budget_tokens": 2048}}),
]:
    summarize(tag, call({"max_tokens": 4096, "messages": [{"role": "user", "content": QUESTION}], **extra}))

print("\n--- B. 不控制 thinking 时，完成任务需要多少预算 ---")
for mt in (2048, 4096, 8192):
    summarize(f"max_tokens={mt}", call({"max_tokens": mt, "messages": [{"role": "user", "content": QUESTION}]}))

print("\n--- C. 提示词层面抑制 thinking（不依赖 API 参数）---")
SUPPRESS = QUESTION + "\n\n直接给出最终数字，不要展开推理过程。"
summarize("max_tokens=512 + 抑制提示", call({"max_tokens": 512, "messages": [{"role": "user", "content": SUPPRESS}]}))
summarize("max_tokens=2048 + 抑制提示", call({"max_tokens": 2048, "messages": [{"role": "user", "content": SUPPRESS}]}))

print("\n--- D. 同题重复 3 次，看 thinking 长度方差 ---")
for i in range(3):
    summarize(f"重复 #{i + 1} (max_tokens=8192)", call({"max_tokens": 8192, "messages": [{"role": "user", "content": QUESTION}]}))
