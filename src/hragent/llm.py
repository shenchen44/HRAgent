"""LLM 客户端 —— 封装 ocean-code 网关，内置预算阶梯 Harness。

网关约束（实测，见 results/phase0_api_probe.md）:
  1. thinking 只能整体开关，budget_tokens 被忽略 → 预算控制只能在 harness 层做
  2. 截断时 stop_reason == "max_tokens" 且 text 为空 → 可零成本检测
  3. 同任务 thinking 开销有 10x 方差（800~8154 tok）→ 固定 max_tokens 必然翻车
  4. "预算刚好够"时会产出格式完整但内容错误的答案 → 截断检测抓不到，需上层证据校验

本模块只负责 1~3：给调用方一个"要么拿到完整输出、要么明确告诉你拿不到"的接口。
第 4 类失败（干净答错）由风控层负责，见 risk/ 子模块。
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field

import httpx

from . import config

# 默认预算阶梯：截断后逐档上调。实测 4096 成功而 8192 截断，故阶梯不保证收敛，
# 到顶仍截断则如实返回 truncated=True，由调用方决定是否升级人工。
DEFAULT_LADDER = (4096, 8192, 16384, 32768)


@dataclass
class Reply:
    text: str
    thinking: str
    tool_calls: list[dict]
    stop_reason: str
    usage: dict
    latency_s: float
    attempts: int = 1
    max_tokens_used: int = 0

    @property
    def truncated(self) -> bool:
        """截断 = 模型没说完就断了。注意与"说完了但说错了"是两回事。"""
        return self.stop_reason == "max_tokens"

    @property
    def empty(self) -> bool:
        return not self.text.strip() and not self.tool_calls

    def json(self) -> dict:
        """把 text 解析成 JSON。容忍 markdown 围栏与前后废话。"""
        return parse_json(self.text)


@dataclass
class Ledger:
    """全局成本账本 —— 成本方差是本项目的评测指标之一，必须逐调用记账。"""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    truncated_retries: int = 0
    total_latency_s: float = 0.0
    per_call: list[dict] = field(default_factory=list)

    def record(self, reply: Reply, tag: str) -> None:
        self.calls += 1
        self.input_tokens += reply.usage.get("input_tokens", 0)
        self.output_tokens += reply.usage.get("output_tokens", 0)
        self.cache_read_tokens += reply.usage.get("cache_read_input_tokens", 0)
        self.total_latency_s += reply.latency_s
        self.per_call.append({
            "tag": tag,
            "attempts": reply.attempts,
            "stop_reason": reply.stop_reason,
            "max_tokens": reply.max_tokens_used,
            "input_tokens": reply.usage.get("input_tokens", 0),
            "output_tokens": reply.usage.get("output_tokens", 0),
            "latency_s": reply.latency_s,
        })

    def summary(self) -> dict:
        n = max(self.calls, 1)
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "truncated_retries": self.truncated_retries,
            "mean_latency_s": round(self.total_latency_s / n, 2),
            "total_latency_s": round(self.total_latency_s, 2),
        }


def parse_json(text: str) -> dict:
    """从模型输出里抠出 JSON 对象。"""
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    if start == -1:
        raise ValueError(f"输出中找不到 JSON: {text[:200]!r}")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(s[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start:i + 1])
    raise ValueError(f"JSON 括号不闭合: {text[:200]!r}")


class LLM:
    """一次实例 = 一个配置档（是否开 thinking、预算阶梯、重试策略）。

    用法:
        llm = LLM(thinking=False)          # 路由/抽取/SQL 生成，快且省
        llm = LLM(thinking=True)           # 复杂分析/合规判断
        r = llm.call([{"role": "user", "content": "..."}], tag="route")
        print(r.text, llm.ledger.summary())
    """

    def __init__(
        self,
        model: str | None = None,
        thinking: bool = False,
        max_tokens: int | None = None,
        ladder: tuple[int, ...] = DEFAULT_LADDER,
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        self.model = model or config.MODEL
        self.thinking = thinking
        # 传了 max_tokens 就退化为固定预算（无阶梯），用于跑朴素基线做对照
        self.fixed_max_tokens = max_tokens
        self.ladder = ladder
        self.timeout = timeout
        self.max_retries = max_retries
        self.ledger = Ledger()
        self._client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------ 内部
    def _post(self, payload: dict) -> dict:
        r = self._client.post(f"{config.BASE_URL}/v1/messages", headers=config.HEADERS, json=payload)
        r.raise_for_status()
        return r.json()

    def _build(self, messages, system, tools, tool_choice, max_tokens) -> dict:
        payload: dict = {"model": self.model, "max_tokens": max_tokens, "messages": messages}
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        # budget_tokens 实测被网关忽略，这里只发开关
        payload["thinking"] = {"type": "enabled" if self.thinking else "disabled"}
        return payload

    # ------------------------------------------------------------------ 公开
    def call(
        self,
        messages: list[dict],
        system: str | None = None,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        max_tokens: int | None = None,
        tag: str = "",
    ) -> Reply:
        """带预算阶梯的调用。

        截断且无任何输出（text 与 tool_calls 皆空）时，按阶梯升档重试；
        一旦拿到非空输出就返回——即使它可能内容有错，那属于上层风控的职责。
        """
        if max_tokens:
            ladder = [max_tokens]
        elif self.fixed_max_tokens:
            ladder = [self.fixed_max_tokens]
        else:
            ladder = list(self.ladder)
        last: Reply | None = None

        for attempt, mt in enumerate(ladder, start=1):
            payload = self._build(messages, system, tools, tool_choice, mt)
            body, latency, err = None, 0.0, None
            for retry in range(self.max_retries):
                t0 = time.time()
                try:
                    body = self._post(payload)
                    latency = time.time() - t0
                    break
                except Exception as e:  # 网络/限流/5xx
                    err = e
                    time.sleep(min(2 ** retry + random.random(), 30))
            if body is None:
                raise RuntimeError(f"调用失败（已重试 {self.max_retries} 次）: {err}")

            content = body.get("content", [])
            reply = Reply(
                text="".join(b.get("text", "") for b in content if b.get("type") == "text"),
                thinking="".join(b.get("thinking", "") for b in content if b.get("type") == "thinking"),
                tool_calls=[b for b in content if b.get("type") == "tool_use"],
                stop_reason=body.get("stop_reason", ""),
                usage=body.get("usage", {}),
                latency_s=round(latency, 2),
                attempts=attempt,
                max_tokens_used=mt,
            )
            self.ledger.record(reply, tag)
            last = reply

            if not (reply.truncated and reply.empty):
                return reply
            if attempt < len(ladder):
                self.ledger.truncated_retries += 1

        return last  # 阶梯到顶仍截断，如实返回

    def ask(self, prompt: str, **kw) -> Reply:
        return self.call([{"role": "user", "content": prompt}], **kw)

    def ask_json(self, prompt: str, **kw) -> dict:
        return self.ask(prompt, **kw).json()

    def close(self) -> None:
        self._client.close()
