# HRAgent

HRAgent 是一个面向 HR COE 场景的多智能体实验系统。项目使用 LangGraph 编排指挥官、业务执行体和风控模块，覆盖 HR 数据查询、制度问答、人岗匹配、面试题生成及入离职流程，并通过冻结评测集、对照实验和端到端消融验证各组件的实际收益。

本项目关注两个问题：

1. 如何让 Agent 在信息不足时主动澄清，而不是继续生成不可靠答案。
2. 如何通过确定性执行、证据校验、合规检查和人工介入降低 HR 场景中的错误与风险。

> 项目使用合成 HR 数据和离线评测集，结果用于验证系统设计，不代表生产环境业务指标。

## 系统架构

```text
用户请求
   │
   ▼
Query Screen
合规预检 · PII 检测 · 高风险请求识别
   │
   ▼
Orchestrator
意图识别 · 槽位补全 · 执行体路由
   │
   ├── HRDataSQL      Text-to-SQL / 编制与考勤查询
   ├── PolicyRAG      制度与劳动法规问答
   ├── RecruitMatch   技能抽取与人岗匹配
   ├── InterviewKit   面试题与追问生成
   └── OnboardFlow    入离职流程（写操作需审批）
   │
   ▼
RiskAgent
截断检测 · 证据校验 · 合规检查 · 不确定性门控
   │
   ├── pass       返回结果
   ├── refuse     拒绝无依据或不合规请求
   └── escalate   LangGraph interrupt / Human-in-the-loop
```

系统按四层组织：

- L1 模型与 Harness：统一模型调用、预算控制、截断检测和成本记录。
- L2 编排：基于 LangGraph 的状态图、Checkpoint、子图和人工中断恢复。
- L3 风控：查询前置筛查与执行后置校验，覆盖证据、合规和不确定性。
- L4 观测：Trace、失败样本分类、断点续跑和实验结果落盘。

## 核心能力

| 模块 | 实现 |
|---|---|
| Text-to-SQL | 数据字典与业务口径注入、SQL 执行、结果值校验 |
| Policy RAG | BM25、稠密检索、RRF、Rerank、引用校验与知识盲区拒答 |
| 人岗匹配 | 技能抽取、结构化要求对齐、简历去标识与公平性检查 |
| Agent 编排 | 分层路由、并行执行、主动澄清、Checkpoint 与 Human-in-the-loop |
| 风控 | 截断、证据、合规、不确定性四类闸门；写操作 Dry-run 与白名单 |
| 评测 | 冻结数据集、重复运行、配对检验、风险-覆盖曲线和端到端消融 |
| Web UI | 岗位、候选人、智能匹配、AI 助手和审计日志六个视图 |

## 数据与评测

评测资产在实验前冻结，并在 [`eval/FROZEN.json`](eval/FROZEN.json) 中记录 SHA256。主要数据包括：

| 数据集 | 规模 | 用途 |
|---|---:|---|
| HR 数据库 | 413 名员工、8 张表 | Text-to-SQL 与业务口径验证 |
| 意图集 | 300 条 | 路由与越界请求识别 |
| Text-to-SQL | 127 题 | SQL 有效性、执行结果与口径正确性 |
| 制度检索 | 94 题 | 检索、引用和知识盲区拒答 |
| 人岗匹配 | 240 对 | 结构化匹配与公平性分析 |
| 技能抽取 | 252 条 | 技能及掌握程度识别 |
| 风控集 | 220 条 | 故障注入、误报率与召回率 |

指标定义、数据泄漏控制和统计方法见 [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md)。

## 实验结果

项目共完成 10 组实验。下表仅保留影响系统设计的主要结论；完整报告及逐条结果位于 [`results/`](results/) 目录。

| 实验 | 主要结果 | 结论 |
|---|---|---|
| Thinking 预算 | 准确率 1.00 → 0.95，成本增加 39% | 增加推理预算未带来收益 |
| 歧义处理 | L3 澄清成功数 0/30 → 19/30，Token/题 923 → 734 | 主动澄清优于硬猜和增加预算 |
| 分层路由 | 平均提升 4.8 个百分点，但 5 次重复仅 1 次显著；成本 2.00× | 方向一致，但证据不足以支持显著优于扁平路由 |
| Text-to-SQL | 口径正确率平均提升 12.7 个百分点，5 次重复中 4 次显著 | 业务口径注入有效，语法有效率不是充分指标 |
| 制度检索 | BM25 → Rerank 的 Hit@1 提升 18.6 个百分点，p=0.0025 | 仅该组差异达到统计显著 |
| 技能抽取 | 技能 F1 基本不变；干扰技能误抽率 0.790 → 0.029 | 需要单独评估掌握程度和误抽 |
| 风控闸门 | 平衡故障集上组合闸门 Precision/Recall 均为 1.00 | 组件结果良好，但不能替代端到端评测 |
| 端到端消融 | 行为正确率 +15.3 个百分点，硬答率 -51.7 个百分点，成本 -35% | 澄清层贡献最大，单独风控主要将错误回答转为拒答 |

三点实验结论：

- 组件指标提升不等于端到端收益。风控闸门能降低硬答率，但单独加入时未提高行为正确率。
- SQL 可执行不等于结果正确。主要错误来自聚合方式、过滤条件和业务口径，而不是语法。
- 单次 LLM 评测波动较大。涉及模型判断的实验采用重复运行或同臂复跑估计噪声。

详细报告：

- [`results/exp2_ambiguity.md`](results/exp2_ambiguity.md)：歧义处理策略
- [`results/exp3_routing.md`](results/exp3_routing.md)：分层路由与扁平路由
- [`results/exp4_sql_caliber.md`](results/exp4_sql_caliber.md)：Text-to-SQL 口径注入
- [`results/exp5_retrieval.md`](results/exp5_retrieval.md)：检索与重排
- [`results/exp6_skill_match.md`](results/exp6_skill_match.md)：技能抽取与人岗匹配
- [`results/exp7_guards.md`](results/exp7_guards.md)：风控闸门
- [`results/exp8_risk_coverage.md`](results/exp8_risk_coverage.md)：风险-覆盖分析
- [`results/exp8b_gate_calibration.md`](results/exp8b_gate_calibration.md)：门控校准
- [`results/exp9_ablation.md`](results/exp9_ablation.md)：端到端消融

## 快速开始

### 环境

- Python 3.12
- `uv`
- 一个兼容 Anthropic Messages API 的模型服务

```bash
git clone https://github.com/shenchen44/HRAgent.git
cd HRAgent

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  httpx sentence-transformers langgraph
```

设置模型服务：

```bash
export ANTHROPIC_BASE_URL="https://your-endpoint.example.com"
export ANTHROPIC_AUTH_TOKEN="your-token"
export ANTHROPIC_MODEL="your-model"
```

运行端到端示例：

```bash
.venv/bin/python scripts/demo.py "研发中心现在有多少在职员工？"
```

### Web UI

仓库包含可直接使用的演示数据库。启动服务后访问 `http://127.0.0.1:8765`：

```bash
.venv/bin/python scripts/serve_ui.py
```

前端使用原生 HTML、CSS 和 JavaScript，后端基于 Python `ThreadingHTTPServer`，无需额外构建步骤。

## 复现实验

验证冻结数据与已落盘结果：

```bash
.venv/bin/python scripts/freeze_eval.py --verify
.venv/bin/python scripts/selftest_metrics.py
.venv/bin/python scripts/selftest_guards.py
.venv/bin/python scripts/smoke_graph.py
.venv/bin/python scripts/verify_repro.py
```

运行主要实验：

```bash
.venv/bin/python scripts/exp4_sql_caliber.py
.venv/bin/python scripts/exp5_retrieval.py
.venv/bin/python scripts/exp7_guards.py --set orig --split all
.venv/bin/python scripts/exp8_risk_coverage.py
.venv/bin/python scripts/exp9_ablation.py --split test
```

部分实验需要调用模型服务，运行时间与费用取决于模型、并发设置和网络环境。为避免测试集漂移，重新生成数据前请先阅读 [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md)。

## 项目结构

```text
hragent/
├── src/hragent/
│   ├── agents/          # 指挥官与五类执行体
│   ├── orchestrator/    # LangGraph 状态图与 HITL
│   ├── tools/           # SQL、检索和技能匹配
│   ├── risk/            # 前置筛查与后置风控
│   ├── obs/             # Trace、Checkpoint 与恢复
│   └── ui/              # Web UI 后端
├── data/                # 合成 HR 数据、制度语料和演示数据库
├── eval/                # 冻结评测集与校验清单
├── results/             # 实验报告和原始结果
├── scripts/             # 数据生成、实验、回归和启动脚本
├── web/                 # 原生前端
└── docs/                # 系统设计与评测协议
```

## 已知限制

- 数据库、制度语料和评测问题主要为合成或离线构造，结论尚未经过真实企业流量验证。
- 分层路由的平均结果优于扁平路由，但统计证据不足且 Token 成本更高。
- 当前合规筛查对隐性代理歧视的识别仍依赖模型判断，规则词表不能覆盖全部风险。
- 检索器分数量纲不同，更换检索器时必须重新校准门控阈值。
- Human-in-the-loop 的触发条件仍需结合真实审核反馈继续校准。

## 文档

- [`docs/DESIGN.md`](docs/DESIGN.md)：系统设计、接口契约和关键决策
- [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md)：评测集、指标和统计协议
