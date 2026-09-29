"""生成制度检索（RAG）评测集：问题 + gold chunk id。

为什么手写而不是机械生成：
  从片段正文生成问题，问题里就会带上答案的词 —— BM25 必然命中，测不出真检索能力。
  员工真实提问用的是口语和场景（"我下个月结婚能休几天"），不是条款标题。
  所以问题手写，gold id 对着片段清单标，脚本只负责校验 id 存在与统计难度。

四类难度：
  单条款   一个片段即可回答
  跨条款   需要综合 2–3 个片段（如公司制度 + 劳动法）
  半盲区   制度只规定了一部分，系统应说明"只规定了 X，未涉及 Y"
  全盲区   制度里确实没有，系统应拒答（gold 为空）

跑法: .venv/bin/python scripts/gen_rag_eval.py
产出: eval/rag_eval.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hragent import config  # noqa: E402
from hragent.tools import retrieval  # noqa: E402

# (问题, [gold_chunk_ids], 难度, 备注)
QUESTIONS: list[tuple[str, list[str], str, str]] = [
    # ---------------- 考勤
    ("我昨天忘打卡了，还能补吗？", ["attendance_policy#1.2"], "单条款", "补卡时限与次数"),
    ("让同事帮我打个卡行不行？", ["attendance_policy#1.3"], "单条款", "代打卡后果"),
    ("迟到了半小时会怎么样？", ["attendance_policy#2.1", "attendance_policy#2.2"], "跨条款", "30 分钟边界两侧"),
    ("我这个月迟到好几次了，有什么影响？", ["attendance_policy#2.4"], "单条款", "累计迟到"),
    ("周末在家自己加班到很晚，算加班吗？", ["attendance_policy#3.1"], "单条款", "须事前审批"),
    ("一个月最多能加多少班？", ["attendance_policy#3.3"], "单条款", "加班上限"),
    ("加班费是怎么算出来的？", ["attendance_policy#3.4", "attendance_policy#3.5"], "跨条款", "补偿方式 + 时薪公式"),
    ("去北京出差一天补多少钱？", ["attendance_policy#4.2"], "单条款", "一线城市标准"),
    ("出差赶上周末，算加班吗？", ["attendance_policy#4.3"], "单条款", "休息日不计加班"),
    ("什么情况会被算旷工？", ["attendance_policy#5.1", "attendance_policy#5.2"], "跨条款", "认定 + 处理"),
    ("上个月考勤什么时候出来？我有异议怎么办？", ["attendance_policy#6.2"], "单条款", "异议时限"),

    # ---------------- 薪酬
    ("我的年度总现金收入包括哪些部分？", ["compensation_policy#1.1"], "单条款", "Total Cash 构成"),
    ("社保是按什么基数交的？", ["compensation_policy#2.2"], "单条款", "缴费基数"),
    ("公积金公司交多少比例？", ["compensation_policy#3.1"], "单条款", "12%"),
    ("公积金基数什么时候调整？", ["compensation_policy#3.2"], "单条款", "每年 7 月"),
    ("公司给上商业保险吗？", ["compensation_policy#4.1"], "单条款", "补充商业保险"),
    ("一年有几次调薪机会？", ["compensation_policy#5.1"], "单条款", "调薪窗口"),
    ("可以打听同事的工资吗？", ["compensation_policy#6.1"], "单条款", "薪酬保密"),
    ("如果我给公司造成损失，最多能扣我多少工资？", ["compensation_policy#7.2"], "单条款", "扣除上限 20%"),

    # ---------------- 保密
    ("我离职之后还需要继续保密吗？", ["confidentiality_policy#2.3", "confidentiality_policy#1.2"], "跨条款", "离职后无期限"),
    ("竞业限制最长能限制多久？", ["confidentiality_policy#3.2", "labor_law_excerpt#1.6"], "跨条款", "公司规定 + 法定上限"),
    ("竞业限制期间公司给钱吗？", ["confidentiality_policy#3.4"], "单条款", "经济补偿"),
    ("我在职期间做出来的东西，专利归谁？", ["confidentiality_policy#4.1"], "单条款", "职务成果"),
    ("能把公司数据传到自己的网盘吗？", ["confidentiality_policy#5.3"], "单条款", "禁止上传"),
    ("公司数据分几个级别？", ["confidentiality_policy#5.1"], "单条款", "四级分类"),
    ("薪酬明细属于什么级别的数据？", ["confidentiality_policy#5.2"], "单条款", "机密级"),
    ("发现有人泄露公司机密，去哪举报？", ["confidentiality_policy#6.1"], "单条款", "举报渠道"),

    # ---------------- 员工手册
    ("入职第一天要带什么材料？", ["employee_handbook#2.1"], "单条款", "入职材料"),
    ("试用期最长能有多久？", ["employee_handbook#2.2", "labor_law_excerpt#1.1"], "跨条款", "公司规定 + 法定"),
    ("试用期工资会打折吗？", ["employee_handbook#2.3", "labor_law_excerpt#1.2"], "跨条款", "不低于 80%"),
    ("公司是弹性工作制吗？", ["employee_handbook#3.2"], "单条款", "弹性工作"),
    ("公司出钱培训后我想辞职，要赔钱吗？", ["employee_handbook#5.2"], "单条款", "培训服务期"),
    ("对考核结果不服，可以去哪申诉？", ["employee_handbook#6.1", "performance_policy#4.4"], "跨条款", "申诉渠道"),

    # ---------------- 员工关系
    ("入职之后多久必须签劳动合同？", ["employee_relations#1.1"], "单条款", "1 个月内"),
    ("签几次固定期限合同之后可以签无固定期限？", ["employee_relations#1.2", "labor_law_excerpt#1.3"], "跨条款", "两次后续订"),
    ("我想辞职，要提前多久说？", ["employee_relations#2.2"], "单条款", "员工单方解除"),
    ("被辞退能拿到多少补偿？", ["employee_relations#3.1", "employee_relations#3.2"], "跨条款", "N 的计算与基数"),
    ("N+1 是什么意思？", ["employee_relations#3.3"], "单条款", "代通知金"),
    ("什么情况下辞退不用给补偿？", ["employee_relations#3.5", "employee_relations#5.2"], "跨条款", "免补偿情形"),
    ("离职证明什么时候能拿到？", ["employee_relations#4.2"], "单条款", "离职证明"),
    ("离职当月的工资什么时候结清？", ["employee_relations#4.4"], "单条款", "离职结算"),
    ("公司处分一共有几档？", ["employee_relations#5.1"], "单条款", "处分等级"),
    ("退休返聘是按劳动关系还是劳务关系？", ["employee_relations#6.2"], "单条款", "劳务关系"),

    # ---------------- 劳动法
    ("法定产假是多少天？", ["labor_law_excerpt#2.8", "leave_policy#5.1"], "跨条款", "法定 98 天 + 公司规定"),
    ("国家规定的年休假是几天？", ["labor_law_excerpt#2.5", "leave_policy#1.1"], "跨条款", "法定 + 公司"),
    ("没休完的年假怎么补偿？", ["labor_law_excerpt#2.6", "leave_policy#1.5"], "跨条款", "300% 补偿"),
    ("病假最长能休多久？", ["labor_law_excerpt#2.7", "leave_policy#2.3"], "跨条款", "医疗期"),
    ("申请劳动仲裁有时间限制吗？", ["labor_law_excerpt#4.1"], "单条款", "一年时效"),
    ("加班工资是平时的几倍？", ["labor_law_excerpt#2.3"], "单条款", "150/200/300%"),
    ("法定节假日都有哪些？", ["labor_law_excerpt#2.4"], "单条款", "法定节假日"),

    # ---------------- 假期
    ("我工作满 3 年了，有几天年假？", ["leave_policy#1.1"], "单条款", "年假天数分档"),
    ("我是年中入职的，年假怎么折算？", ["leave_policy#1.2"], "单条款", "年假折算"),
    ("去年的年假没休完，能留到今年吗？", ["leave_policy#1.4"], "单条款", "顺延与作废"),
    ("病假期间工资怎么发？", ["leave_policy#2.2"], "单条款", "病假工资"),
    ("请事假会扣钱吗？", ["leave_policy#3.1"], "单条款", "无薪假"),
    ("一年最多能请多少天事假？", ["leave_policy#3.2"], "单条款", "15 天上限"),
    ("我下个月结婚，能休几天假？", ["leave_policy#4.1"], "单条款", "婚假 10 天"),
    ("婚假有有效期吗？", ["leave_policy#4.2"], "单条款", "12 个月内"),
    ("再婚还能享受婚假吗？", ["leave_policy#4.3"], "单条款", "再婚同享"),
    ("陪产假有多少天？", ["leave_policy#5.2"], "单条款", "陪产假"),
    ("育儿假是怎么规定的？", ["leave_policy#5.3"], "单条款", "育儿假"),
    ("我爷爷去世了，有丧假吗？", ["leave_policy#6.3"], "单条款", "祖父母 1 天"),
    ("去外地奔丧，路上的时间算假吗？", ["leave_policy#6.4"], "单条款", "路程假"),
    ("工伤养伤期间工资怎么发？", ["leave_policy#7.1"], "单条款", "停工留薪期"),
    ("临时有急事来不及走请假流程怎么办？", ["leave_policy#8.3"], "单条款", "24 小时内补办"),

    # ---------------- 绩效
    ("公司多久做一次绩效考核？", ["performance_policy#1.1"], "单条款", "半年度"),
    ("绩效等级是怎么分布的？", ["performance_policy#2.1"], "单条款", "强制分布"),
    ("我们部门人比较少，强制分布怎么算？", ["performance_policy#2.2"], "单条款", "少于 10 人合并"),
    ("绩效 C 有比例下限吗？", ["performance_policy#2.3"], "单条款", "C 不低于 10%"),
    ("绩效考核结果会影响什么？", ["performance_policy#5.1"], "单条款", "结果应用"),
    ("连续拿 C 会怎么样？", ["performance_policy#5.2"], "单条款", "调岗或解除"),
    ("PIP 是什么？一般要多久？", ["performance_policy#6.1", "performance_policy#6.2"], "跨条款", "PIP 定义与周期"),
    ("绩效考核的流程有哪几步？", ["performance_policy#4.1"], "单条款", "考核流程"),

    # ---------------- 招聘
    ("招聘 JD 里可以写“限男性”吗？", ["recruitment_policy#1.2"], "单条款", "禁止歧视性条件"),
    ("超编招聘需要谁审批？", ["recruitment_policy#2.2"], "单条款", "分管高管"),
    ("面试最多安排几轮？", ["recruitment_policy#3.3"], "单条款", "不超过 3 轮"),
    ("面试结束后多久要提交评价？", ["recruitment_policy#4.2"], "单条款", "24 小时"),
    ("背景调查需要候选人同意吗？", ["recruitment_policy#5.1"], "单条款", "书面授权"),
    ("背景调查能查候选人的婚育情况吗？", ["recruitment_policy#5.3"], "单条款", "禁止调查"),
    ("内推奖金什么时候发？", ["recruitment_policy#7.2", "recruitment_policy#7.3"], "跨条款", "发放时点与例外"),
    ("Offer 的有效期是多久？", ["recruitment_policy#6.2"], "单条款", "7 个工作日"),
    ("校园招聘一般什么时候进行？", ["recruitment_policy#9.1"], "单条款", "9–11 月"),

    # ---------------- 半盲区：制度只规定了一部分
    ("生育津贴怎么算、怎么发？", ["leave_policy#5.1"], "半盲区", "只有产假天数，无津贴计算"),
    ("竞业限制的违约金具体是多少钱？", ["confidentiality_policy#3.5"], "半盲区", "只说须支付，无金额"),
    ("年终奖的具体计算公式是什么？", ["compensation_policy#1.4"], "半盲区", "只挂钩绩效，无公式"),
    ("培训服务期违约金按什么比例递减？", ["employee_handbook#5.2"], "半盲区", "只有上限，无递减比例"),

    # ---------------- 全盲区：制度里确实没有 → 应拒答
    ("公司的期权分几年归属？", [], "全盲区", "无股权激励规定"),
    ("员工宿舍怎么申请？", [], "全盲区", "无宿舍规定"),
    ("外派新加坡的补贴标准是多少？", [], "全盲区", "无外派规定"),
    ("EAP 心理咨询一年可以约几次？", [], "全盲区", "无 EAP 规定"),
    ("企业年金公司缴多少？", [], "全盲区", "无年金规定"),
    ("子女教育补贴能报销多少？", [], "全盲区", "无教育补贴规定"),
    ("申请一项发明专利奖励多少钱？", [], "全盲区", "无专利奖励规定"),
    ("员工持股平台的退出机制是什么？", [], "全盲区", "无持股平台规定"),
]


def main() -> None:
    chunks = {json.loads(l)["chunk_id"]: json.loads(l)
              for l in (config.DATA / "policy_chunks.jsonl").read_text().splitlines()}
    gaps = {json.loads(l)["chunk_id"] for l in
            (config.DATA / "policy_chunks.jsonl").read_text().splitlines()}

    # 校验 gold id 全部存在
    bad = [(q, g) for q, golds, _, _ in QUESTIONS for g in golds if g not in chunks]
    if bad:
        for q, g in bad:
            print(f"❌ gold id 不存在: {g}  （问题: {q}）")
        raise SystemExit(f"{len(bad)} 个 gold id 无效")

    # 去重
    seen, items = set(), []
    for q, golds, diff, note in QUESTIONS:
        if q in seen:
            raise SystemExit(f"❌ 重复问题: {q}")
        seen.add(q)
        items.append({"query": q, "gold_chunk_ids": golds, "difficulty": diff, "note": note})
    for i, it in enumerate(items):
        it["id"] = f"RAG{i + 1:04d}"
        it["split"] = "dev" if i % 2 == 0 else "test"

    order = ["id", "split", "difficulty", "query", "gold_chunk_ids", "note"]
    (config.EVAL / "rag_eval.jsonl").write_text(
        "\n".join(json.dumps({k: it[k] for k in order}, ensure_ascii=False) for it in items))

    from collections import Counter
    print(f"✅ 制度检索评测集已生成: eval/rag_eval.jsonl\n")
    print(f"   共 {len(items)} 题（dev {sum(1 for x in items if x['split'] == 'dev')} / "
          f"test {sum(1 for x in items if x['split'] == 'test')}）")
    for d, c in Counter(x["difficulty"] for x in items).most_common():
        print(f"   {d:6s} {c:3d} 条")
    n_gold = sum(len(x["gold_chunk_ids"]) for x in items)
    print(f"   平均 gold 片段数 {n_gold / len(items):.2f}，覆盖 "
          f"{len({g for x in items for g in x['gold_chunk_ids']})} / {len(chunks)} 个片段")

    # ---- BM25 基线：难度自检 ----
    # 如果基线就接近满分，说明题太容易，检索指标没有区分度 —— 这必须当场暴露
    idx = retrieval.index()
    print(f"\n   [难度自检] BM25 基线（这版检索器的真实水平，不是天花板）")
    for k in (1, 3, 5, 10):
        per_diff: dict[str, list[float]] = {}
        for it in items:
            if not it["gold_chunk_ids"]:
                continue
            got = set(idx.retrieve_ids(it["query"], k))
            per_diff.setdefault(it["difficulty"], []).append(
                len(got & set(it["gold_chunk_ids"])) / len(it["gold_chunk_ids"]))
        allv = [v for vs in per_diff.values() for v in vs]
        parts = "  ".join(f"{d} {sum(v) / len(v):.2f}" for d, v in sorted(per_diff.items()))
        print(f"      recall@{k:<2d} 总体 {sum(allv) / len(allv):.2f}   {parts}")

    # 全盲区：检索器应当"什么都找不到" —— 用最高分衡量
    blind = [it for it in items if it["difficulty"] == "全盲区"]
    tops = [(it, idx.search(it["query"], 1)) for it in blind]
    top_scores = [t[0]["score"] if t else 0.0 for _, t in tops]
    ranked = sorted(tops, key=lambda p: -(p[1][0]["score"] if p[1] else 0))
    print(f"\n      全盲区 {len(blind)} 题的最高检索分（应显著低于正常题，否则无法靠分数拒答）")
    print(f"        最高 {max(top_scores):.1f} / 最低 {min(top_scores):.1f} / "
          f"均值 {sum(top_scores) / len(top_scores):.1f}")
    it0, t0 = ranked[0]
    print(f"        分数最高的盲区题: 「{it0['query']}」"
          f" → {t0[0]['chunk_id']} ({t0[0]['score']:.1f})")
    normal_tops = [idx.search(it["query"], 1)[0]["score"]
                   for it in items if it["gold_chunk_ids"]]
    print(f"      有 gold 的题最高分: 均值 {sum(normal_tops) / len(normal_tops):.1f} / "
          f"最低 {min(normal_tops):.1f}")
    overlap = sum(1 for s in top_scores if s > min(normal_tops))
    print(f"      → 盲区题中有 {overlap}/{len(blind)} 条最高分超过正常题最低分，"
          f"说明单纯靠分数阈值拒答会有重叠")


if __name__ == "__main__":
    main()
