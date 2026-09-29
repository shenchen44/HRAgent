"""路径与环境配置。

网关凭据优先读环境变量（Claude Code 会把 settings.json 的 env 注入子进程）；
若在 Claude Code 之外运行，回退到 ~/.claude/settings.json。
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
EVAL = ROOT / "eval"

for _d in (DATA, RESULTS, EVAL):
    _d.mkdir(parents=True, exist_ok=True)

_SETTINGS = Path.home() / ".claude" / "settings.json"


def _fallback(key: str) -> str | None:
    """环境变量缺失时，从 ~/.claude/settings.json 的 env 段取。"""
    try:
        return json.loads(_SETTINGS.read_text()).get("env", {}).get(key)
    except Exception:
        return None


def _get(key: str, default: str | None = None) -> str:
    val = os.environ.get(key) or _fallback(key) or default
    if val is None:
        raise RuntimeError(f"缺少配置 {key}：请设置环境变量或在 ~/.claude/settings.json 的 env 中提供")
    return val


BASE_URL = _get("ANTHROPIC_BASE_URL")
AUTH_TOKEN = _get("ANTHROPIC_AUTH_TOKEN")
MODEL = _get("ANTHROPIC_MODEL", "deepseek-v4.1-flash")

HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}
