# Phase 0 · 网关能力探测报告

日期: 2026-09-28
端点: `https://ocean-code-cn.tuya-inc.com:7799` (Anthropic Messages API 兼容)
模型: `deepseek-v4.1-flash` (实际解析为 `deepseek-v4-1-flash-260910`)
脚本: `scripts/probe_api.py`, `scripts/probe_thinking.py`

---

## 一、主链路能力（决定 JD「意图识别 → 动态工具调用 → 完整输出」能否落地）

| 能力 | 结论 | 证据 |
|---|---|---|
| 工具调用 (tool_use) | ✅ 可用 | 一次提问并行选中 2 个工具，thinking 内正确分解双意图 |
| 多轮 tool_result 回灌 | ✅ 可用 | 正确汇总两路结果并生成结构化回答，带追问 |
| `tool_choice: any` | ✅ 可用 | 强制调用；模型会自发 `SELECT name FROM sqlite_master` 探索 schema |
| 结构化 JSON 输出 | ✅ 可用 | 裸 JSON，无 markdown 包裹，`json.loads` 一次通过 |
| prompt caching | ✅ 有信号 | 第二轮 `cache_read_input_tokens=384` |

**结论：主链路无阻塞，可按 JD 描述的形态实现。**

## 二、thinking 控制能力（关键约束）

| 尝试 | 结果 |
|---|---|
| `thinking: {"type": "disabled"}` | ✅ **生效**：think=0c，out=4 tok，1.5s |
| `thinking: {"type": "enabled", "budget_tokens": 512}` | ❌ **被忽略**：实际 think=4861c / out=3099 tok |
| `thinking: {"type": "enabled", "budget_tokens": 2048}` | ❌ **被忽略**：实际 out=1131 tok |
| 提示词写「不要展开推理过程」 | ❌ **无效**：仍 think=816c 并截断 |

**结论：thinking 只能整体开关，无法设定预算上限。** 预算控制必须在 harness 层做。

## 三、预算不确定性（本项目第一个真实工程靶子）

同一道题（24 人，离职 3，入职 5，求离职率），重复 3 次：

| 轮次 | thinking | 输出 tok | 延迟 | 答案 |
|---|---|---|---|---|
| #1 | 1,292 c | 800 | 6.2 s | 12.5% ✅ |
| #2 | 13,137 c | 8,154 | 58.2 s | 12.5% ✅ |
| #3 | 4,055 c | 2,567 | 18.4 s | 12% ✅ |

**thinking 开销 10× 方差，延迟 9× 方差。**

### 两种失败模式

| 模式 | 触发 | 表现 | 危害 |
|---|---|---|---|
| **截断** | `max_tokens` 偏小 | `stop_reason=max_tokens`，**`text` 为空** | 用户拿到空回复，可被检测 |
| **干净答错** | 预算"刚好够" | `stop_reason=end_turn`，格式完整 | **`10.34%`**（正确 12.5%，模型把分母算成 24+5=29）**不可被朴素检测发现** |

预算扫描：

| max_tokens | stop_reason | 结果 |
|---|---|---|
| 2048 | `max_tokens` | 空 |
| 4096 | `end_turn` | `10.34%` ❌ 干净答错 |
| 8192 | `max_tokens` | 空（think 24,488 c） |

> **注意：4096 这一档"成功了"但答案是错的，而 8192 反而截断。**
> 固定预算策略下，成功/失败/正确/错误四者完全解耦 —— 这是 harness 必须处理的。

## 四、对设计的直接输入

1. **默认关闭 thinking** 用于：意图路由、槽位抽取、SQL 生成、分类 —— 快 10×、省 100×
2. **按需开启 thinking** 用于：复杂分析、合规判断、冲突消解
3. **必须实现预算阶梯**：检测 `stop_reason == max_tokens` → 升档重试（零成本触发信号）
4. **必须实现干净答错检测**：截断可检测，干净答错不能 —— 需要**证据校验/交叉验证**，这正是「风控 Agent」的核心职责
5. **成本与延迟必须入评测指标**：同一任务的 tokens 波动 10×，单报准确率会掩盖成本失控

## 五、复现

```bash
.venv/bin/python scripts/probe_api.py       # 工具调用 / JSON / 截断
.venv/bin/python scripts/probe_thinking.py  # thinking 控制 / 预算阶梯
```
