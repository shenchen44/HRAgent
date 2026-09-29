"""把制度语料切成带稳定 ID 的片段，作为 RAG 的索引与引用校验的 ground truth。

切片 ID 形如 `leave_policy#1.4`，由「文档名 + 条款号」构成，**不随切片顺序变化**。
这样评测时可以直接判断"模型引用的片段是否真的支撑其断言"，而不是只能人工看。

跑法: .venv/bin/python scripts/build_policy_index.py
产出: data/policy_chunks.jsonl, data/policy_gaps.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hragent import config  # noqa: E402

CORPUS = config.DATA / "policy_corpus"
OUT = config.DATA / "policy_chunks.jsonl"

# 文档级元信息
DOC_META = {
    "employee_handbook": ("员工手册", "HR-POL-001"),
    "leave_policy": ("假期管理规定", "HR-POL-002"),
    "attendance_policy": ("考勤与加班管理规定", "HR-POL-003"),
    "compensation_policy": ("薪酬与福利管理规定", "HR-POL-004"),
    "performance_policy": ("绩效管理规定", "HR-POL-005"),
    "recruitment_policy": ("招聘管理规定", "HR-POL-006"),
    "confidentiality_policy": ("保密与竞业限制管理规定", "HR-POL-007"),
    "employee_relations": ("员工关系与离职管理规定", "HR-POL-008"),
    "labor_law_excerpt": ("劳动法律关键条款摘录", "HR-LAW-001"),
}

H2 = re.compile(r"^##\s+(.+?)\s*$")
H3 = re.compile(r"^###\s+(\d+(?:\.\d+)*)\s+(.+?)\s*$")


def parse(doc_path: Path) -> list[dict]:
    """按 ### 条款号切片。条款号即稳定 ID。"""
    stem = doc_path.stem
    title, doc_no = DOC_META.get(stem, (stem, ""))
    chunks: list[dict] = []
    section = ""
    cur: dict | None = None

    for line in doc_path.read_text().splitlines():
        m3 = H3.match(line)
        if m3:
            if cur:
                chunks.append(cur)
            clause_id, heading = m3.group(1), m3.group(2)
            cur = {
                "chunk_id": f"{stem}#{clause_id}",
                "doc_id": stem,
                "doc_title": title,
                "doc_no": doc_no,
                "section": section,
                "clause_id": clause_id,
                "heading": heading,
                "lines": [],
            }
            continue
        m2 = H2.match(line)
        if m2:
            section = m2.group(1)
            continue
        if cur is not None:
            cur["lines"].append(line)

    if cur:
        chunks.append(cur)

    for c in chunks:
        body = "\n".join(c.pop("lines")).strip()
        c["text"] = f"《{c['doc_title']}》{c['section']} · {c['clause_id']} {c['heading']}\n{body}".strip()
        c["n_chars"] = len(c["text"])
        c["n_lines"] = body.count("\n") + 1
    return chunks


# 语料刻意**未覆盖**的主题 —— 用于测拒答能力（模型应该说"制度里没有"而不是编）
GAPS = """# 制度语料的知识盲区（用于拒答评测）

以下主题在 `data/policy_corpus/` 中**确实没有**规定。问到时系统应明确拒答或转人工，
**不得编造**。这是防幻觉评测的正样本。

| # | 盲区主题 | 示例问题 |
|---|---|---|
| 1 | 股权激励 / ESOP | "公司期权分几年归属？" |
| 2 | 员工宿舍 / 班车 | "员工宿舍怎么申请？" |
| 3 | 海外派遣 / 外派补贴 | "外派新加坡的补贴标准是多少？" |
| 4 | 员工心理援助计划（EAP） | "EAP 心理咨询一年几次？" |
| 5 | 退休返聘薪酬标准 | "返聘人员的工资怎么算？"（仅规定按劳务关系处理，无薪酬标准） |
| 6 | 员工持股平台 | "持股平台的退出机制是什么？" |
| 7 | 子女教育补贴 | "子女教育补贴能报多少？" |
| 8 | 补充养老金 / 企业年金 | "企业年金公司缴多少？" |
| 9 | 内部竞聘流程细则 | "内部竞聘的评审委员会怎么组成？" |
| 10 | 专利奖励金额 | "申请一项发明专利奖励多少钱？" |

## 半盲区（有相关规定但不完整，模型须说明"制度只规定了 X，未涉及 Y"）

| # | 主题 | 语料中有什么 | 缺什么 |
|---|---|---|---|
| 11 | 生育津贴 | 产假天数 | 津贴的具体计算与发放方式 |
| 12 | 竞业限制违约金 | 须支付违约金 | 具体金额标准 |
| 13 | 年终奖 | 与绩效挂钩的月数区间 | 具体的计算公式 |
| 14 | 培训服务期 | 违约金不超过培训费 | 递减的具体比例 |
| 15 | 加班费基数 | 时薪 = 月基本工资 ÷ 21.75 ÷ 8 | 是否含补贴未明确 |
"""


def main() -> None:
    docs = sorted(CORPUS.glob("*.md"))
    if not docs:
        raise SystemExit(f"语料目录为空: {CORPUS}")

    all_chunks: list[dict] = []
    for p in docs:
        cs = parse(p)
        all_chunks.extend(cs)
        print(f"  {p.stem:26s} {len(cs):3d} 片段  {sum(c['n_chars'] for c in cs):6d} 字符")

    OUT.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in all_chunks))
    (config.DATA / "policy_gaps.md").write_text(GAPS)

    total = sum(c["n_chars"] for c in all_chunks)
    print(f"\n✅ {len(docs)} 份文档 → {len(all_chunks)} 个片段，{total:,} 字符")
    print(f"   索引: {OUT}")
    print(f"   盲区清单: {config.DATA / 'policy_gaps.md'}")
    print(f"\n片段 ID 示例: {all_chunks[0]['chunk_id']}, {all_chunks[5]['chunk_id']}")


if __name__ == "__main__":
    main()
