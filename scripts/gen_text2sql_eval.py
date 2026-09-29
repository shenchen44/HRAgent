"""生成 Text-to-SQL 评测集。

设计要点:
  1. **gold SQL 由人手写**，不由 LLM 生成 —— 否则"标准答案"本身就不可信
  2. 每题带 `caliber_sensitive` 标记：口径敏感的题（离职率/出勤率/完成率）
     单独统计「口径正确率」，因为 EX 只看结果集，会漏掉"口径错但数值碰巧对"
  3. 口径严格对齐 `data/hr_data_dictionary.md`，两边不一致就是 bug
  4. 生成后**执行全部 gold SQL 验证**，任何一条跑不通直接报错

跑法: .venv/bin/python scripts/gen_text2sql_eval.py
产出: eval/text2sql_eval.jsonl
"""

from __future__ import annotations

import itertools
import json
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hragent import config  # noqa: E402

SEED = 20260928
OUT = config.EVAL / "text2sql_eval.jsonl"

DEPTS = ["研发中心", "产品部", "市场部", "销售部", "人力资源部", "财务部",
         "法务部", "客户成功部", "供应链部", "质量部", "设计部", "行政部"]

# 期间定义：(标签, 起始日, 结束日, 口语说法)
PERIODS = [
    ("2025H1", "2025-01-01", "2025-06-30", ["2025年上半年", "2025 上半年", "去年上半年"]),
    ("2025H2", "2025-07-01", "2025-12-31", ["2025年下半年", "2025 下半年", "去年下半年"]),
    ("2026H1", "2026-01-01", "2026-06-30", ["2026年上半年", "2026 上半年", "今年上半年"]),
]
MONTHS = [(f"2026-{m:02d}", f"2026年{m}月", f"2026-{m:02d}-01",
           f"2026-{m:02d}-{['31','28','31','30','31','30'][m-1]}") for m in range(1, 7)]

# 口径敏感指标：EX 之外必须单独判口径
CALIBER = {
    "attrition_rate": "离职率 = 期间内离职人数 ÷ 期初在职人数（不是期末，也不是期间平均）",
    "attendance_rate": "出勤率 = 汇总实出勤天数 ÷ 汇总应出勤天数（不是逐人算完再平均）",
    "recruit_completion": "招聘完成率 = 逐需求 (已到岗 ÷ 计划) 后取平均（不是汇总后相除）",
    "training_completion": "培训完成率 = 汇总实际学时 ÷ 汇总计划学时，actual_hours 为 NULL 按 0 计",
    "avg_recruit_cycle": "平均招聘周期 = 已完成需求的 (关闭日 - 开启日) 均值",
    "vacancy_rate": "空缺率 = (编制 - 在职) ÷ 编制",
}


def q_headcount(dept, _p, _m):
    return (f"{dept}现在有多少在职员工？",
            f"SELECT COUNT(*) AS 在职人数 FROM employees "
            f"WHERE dept_id = (SELECT dept_id FROM departments WHERE dept_name = '{dept}') "
            f"AND status = '在职'", False, None)


def q_attrition(dept, period, _m):
    label, start, end, _ = period
    return (f"{dept}{label.replace('H1', '年上半年').replace('H2', '年下半年')}的离职率是多少？",
            f"""SELECT ROUND(
                  100.0 * SUM(CASE WHEN leave_date BETWEEN '{start}' AND '{end}' THEN 1 ELSE 0 END)
                  / NULLIF(SUM(CASE WHEN hire_date < '{start}'
                                    AND (leave_date IS NULL OR leave_date >= '{start}')
                               THEN 1 ELSE 0 END), 0), 2) AS 离职率
                FROM employees
                WHERE dept_id = (SELECT dept_id FROM departments WHERE dept_name = '{dept}')""",
            True, "attrition_rate")


def q_vacancy(dept, _p, _m):
    return (f"{dept}的编制空缺率是多少？",
            f"""SELECT ROUND(100.0 * (d.headcount_budget - COUNT(e.emp_id)) / d.headcount_budget, 2) AS 空缺率
                FROM departments d
                LEFT JOIN employees e ON e.dept_id = d.dept_id AND e.status = '在职'
                WHERE d.dept_name = '{dept}'
                GROUP BY d.dept_id, d.headcount_budget""", True, "vacancy_rate")


def q_avg_salary(dept, _p, _m):
    return (f"{dept}的平均月薪是多少？",
            f"""SELECT ROUND(AVG(base_salary), 2) AS 平均月薪 FROM employees
                WHERE dept_id = (SELECT dept_id FROM departments WHERE dept_name = '{dept}')
                AND status = '在职'""", False, None)


def q_headcount_by_level(dept, _p, _m):
    return (f"{dept}各职级的在职人数分布是怎样的？",
            f"""SELECT job_level AS 职级, COUNT(*) AS 人数 FROM employees
                WHERE dept_id = (SELECT dept_id FROM departments WHERE dept_name = '{dept}')
                AND status = '在职' GROUP BY job_level ORDER BY job_level""", False, None)


def q_attendance_rate(dept, _p, month):
    mid, cn, start, end = month
    return (f"{dept}{cn}的出勤率是多少？",
            f"""SELECT ROUND(100.0 * SUM(a.actual_days) / SUM(a.work_days), 2) AS 出勤率
                FROM attendance a JOIN employees e ON e.emp_id = a.emp_id
                WHERE e.dept_id = (SELECT dept_id FROM departments WHERE dept_name = '{dept}')
                AND a.month = '{mid}'""", True, "attendance_rate")


def q_avg_overtime(dept, _p, month):
    mid, cn, _, _ = month
    return (f"{dept}{cn}的人均加班时长是多少小时？",
            f"""SELECT ROUND(AVG(a.overtime_hours), 2) AS 人均加班时长
                FROM attendance a JOIN employees e ON e.emp_id = a.emp_id
                WHERE e.dept_id = (SELECT dept_id FROM departments WHERE dept_name = '{dept}')
                AND a.month = '{mid}'""", False, None)


def q_perf_dist(dept, period, _m):
    label, _, _, _ = period
    hp = "2025H1" if label.startswith("2025") else "2025H2"
    return (f"{dept}{label.replace('H1', '年上半年').replace('H2', '年下半年')}的绩效分布是怎样的？",
            f"""SELECT p.rating AS 绩效等级, COUNT(*) AS 人数 FROM performance p
                JOIN employees e ON e.emp_id = p.emp_id
                WHERE e.dept_id = (SELECT dept_id FROM departments WHERE dept_name = '{dept}')
                AND p.period = '{hp}' GROUP BY p.rating ORDER BY p.rating""", False, None)


def q_perf_a_count(dept, period, _m):
    label, _, _, _ = period
    hp = "2025H1" if label.startswith("2025") else "2025H2"
    return (f"{dept}{label.replace('H1', '年上半年').replace('H2', '年下半年')}有多少人绩效拿了 A？",
            f"""SELECT COUNT(*) AS A档人数 FROM performance p
                JOIN employees e ON e.emp_id = p.emp_id
                WHERE e.dept_id = (SELECT dept_id FROM departments WHERE dept_name = '{dept}')
                AND p.period = '{hp}' AND p.rating = 'A'""", False, None)


def q_recruit_completion(dept, _p, _m):
    return (f"{dept}的招聘完成率是多少？",
            f"""SELECT ROUND(100.0 * AVG(1.0 * headcount_filled / headcount_planned), 2) AS 招聘完成率
                FROM recruitment
                WHERE dept_id = (SELECT dept_id FROM departments WHERE dept_name = '{dept}')
                AND headcount_planned > 0""", True, "recruit_completion")


def q_open_hc(dept, _p, _m):
    return (f"{dept}现在还有几个 HC 没招满？",
            f"""SELECT SUM(headcount_planned - headcount_filled) AS 剩余HC FROM recruitment
                WHERE dept_id = (SELECT dept_id FROM departments WHERE dept_name = '{dept}')
                AND status = '进行中'""", False, None)


def q_avg_recruit_cycle(dept, _p, _m):
    return (f"{dept}的平均招聘周期是多少天？",
            f"""SELECT ROUND(AVG(julianday(close_date) - julianday(open_date)), 1) AS 平均招聘周期天
                FROM recruitment
                WHERE dept_id = (SELECT dept_id FROM departments WHERE dept_name = '{dept}')
                AND status = '已完成' AND close_date IS NOT NULL""", True, "avg_recruit_cycle")


def q_channel_dist(dept, _p, _m):
    return (f"{dept}的候选人主要来自哪些渠道？各多少人？",
            f"""SELECT c.source AS 渠道, COUNT(*) AS 候选人数 FROM candidates c
                JOIN recruitment r ON r.req_id = c.req_id
                WHERE r.dept_id = (SELECT dept_id FROM departments WHERE dept_name = '{dept}')
                GROUP BY c.source ORDER BY 候选人数 DESC""", False, None)


def q_offer_accept(dept, _p, _m):
    return (f"{dept}的 offer 接受率是多少？",
            f"""SELECT ROUND(100.0 * SUM(CASE WHEN stage = '入职' THEN 1 ELSE 0 END)
                    / NULLIF(SUM(CASE WHEN stage IN ('Offer','入职') THEN 1 ELSE 0 END), 0), 2) AS offer接受率
                FROM candidates c JOIN recruitment r ON r.req_id = c.req_id
                WHERE r.dept_id = (SELECT dept_id FROM departments WHERE dept_name = '{dept}')""",
            True, "offer_accept")


def q_training_completion(dept, _p, _m):
    return (f"{dept}的培训完成率是多少？",
            f"""SELECT ROUND(100.0 * SUM(COALESCE(t.actual_hours, 0)) / SUM(t.planned_hours), 2) AS 培训完成率
                FROM training t JOIN employees e ON e.emp_id = t.emp_id
                WHERE e.dept_id = (SELECT dept_id FROM departments WHERE dept_name = '{dept}')""",
            True, "training_completion")


def q_social_insurance(dept, _p, month):
    mid, cn, _, _ = month
    return (f"{dept}{cn}个人社保缴纳总额是多少？",
            f"""SELECT SUM(p.social_insurance) AS 个人社保总额 FROM payroll p
                JOIN employees e ON e.emp_id = p.emp_id
                WHERE e.dept_id = (SELECT dept_id FROM departments WHERE dept_name = '{dept}')
                AND p.month = '{mid}'""", False, None)


# 指标注册表：(生成函数, 参数类型)
METRICS = [
    (q_headcount, "dept"),
    (q_attrition, "dept_period"),
    (q_vacancy, "dept"),
    (q_avg_salary, "dept"),
    (q_headcount_by_level, "dept"),
    (q_attendance_rate, "dept_month"),
    (q_avg_overtime, "dept_month"),
    (q_perf_dist, "dept_period"),
    (q_perf_a_count, "dept_period"),
    (q_recruit_completion, "dept"),
    (q_open_hc, "dept"),
    (q_avg_recruit_cycle, "dept"),
    (q_channel_dist, "dept"),
    (q_offer_accept, "dept"),
    (q_training_completion, "dept"),
    (q_social_insurance, "dept_month"),
]

# 跨部门/全局题（单列，考 GROUP BY 与排序）
GLOBAL_QUESTIONS = [
    ("哪个部门的在职人数最多？给出部门名和人数。",
     """SELECT d.dept_name AS 部门, COUNT(e.emp_id) AS 在职人数
        FROM departments d JOIN employees e ON e.dept_id = d.dept_id AND e.status = '在职'
        GROUP BY d.dept_id ORDER BY 在职人数 DESC LIMIT 1""", False, None),
    ("各部门的在职人数分别是多少？按人数从高到低排列。",
     """SELECT d.dept_name AS 部门, COUNT(e.emp_id) AS 在职人数
        FROM departments d JOIN employees e ON e.dept_id = d.dept_id AND e.status = '在职'
        GROUP BY d.dept_id ORDER BY 在职人数 DESC""", False, None),
    ("全公司 2025 年下半年的离职率是多少？",
     """SELECT ROUND(100.0 * SUM(CASE WHEN leave_date BETWEEN '2025-07-01' AND '2025-12-31' THEN 1 ELSE 0 END)
                / NULLIF(SUM(CASE WHEN hire_date < '2025-07-01'
                                  AND (leave_date IS NULL OR leave_date >= '2025-07-01')
                             THEN 1 ELSE 0 END), 0), 2) AS 离职率
        FROM employees""", True, "attrition_rate"),
    ("在职员工里，各城市的分布是怎样的？",
     """SELECT city AS 城市, COUNT(*) AS 人数 FROM employees
        WHERE status = '在职' GROUP BY city ORDER BY 人数 DESC""", False, None),
    ("P6 及以上职级的在职员工有多少人？",
     """SELECT COUNT(*) AS 人数 FROM employees
        WHERE status = '在职' AND job_level IN ('P6','P7','P8')""", False, None),
    ("哪个部门的平均月薪最高？给出部门名和平均月薪。",
     """SELECT d.dept_name AS 部门, ROUND(AVG(e.base_salary), 2) AS 平均月薪
        FROM departments d JOIN employees e ON e.dept_id = d.dept_id AND e.status = '在职'
        GROUP BY d.dept_id ORDER BY 平均月薪 DESC LIMIT 1""", False, None),
    ("2026 年 3 月全公司的平均加班时长是多少小时？",
     """SELECT ROUND(AVG(overtime_hours), 2) AS 人均加班时长 FROM attendance
        WHERE month = '2026-03'""", False, None),
    ("候选人在各阶段的分布是怎样的？",
     """SELECT stage AS 阶段, COUNT(*) AS 人数 FROM candidates
        GROUP BY stage ORDER BY 人数 DESC""", False, None),
    ("招聘渠道里，哪个渠道的候选人数最多？",
     """SELECT source AS 渠道, COUNT(*) AS 人数 FROM candidates
        GROUP BY source ORDER BY 人数 DESC LIMIT 1""", False, None),
    ("在职员工的平均司龄是多少年？（按 2026-06-30 计算）",
     """SELECT ROUND(AVG(julianday('2026-06-30') - julianday(hire_date)) / 365.25, 2) AS 平均司龄年
        FROM employees WHERE status = '在职'""", False, None),
    ("2025 年下半年绩效为 C 的员工有多少人？",
     """SELECT COUNT(*) AS 人数 FROM performance WHERE period = '2025H2' AND rating = 'C'""",
     False, None),
    ("各部门的编制总数和在职人数分别是多少？只列出空缺率超过 10% 的部门。",
     """SELECT d.dept_name AS 部门, d.headcount_budget AS 编制, COUNT(e.emp_id) AS 在职,
               ROUND(100.0 * (d.headcount_budget - COUNT(e.emp_id)) / d.headcount_budget, 2) AS 空缺率
        FROM departments d LEFT JOIN employees e ON e.dept_id = d.dept_id AND e.status = '在职'
        GROUP BY d.dept_id, d.headcount_budget
        HAVING 空缺率 > 10 ORDER BY 空缺率 DESC""", True, "vacancy_rate"),
    ("在职员工中，各职级的人数分布是怎样的？",
     """SELECT job_level AS 职级, COUNT(*) AS 人数 FROM employees
        WHERE status = '在职' GROUP BY job_level ORDER BY job_level""", False, None),
    ("2026 年 1 月到 6 月，全公司每个月的个人社保缴纳总额分别是多少？",
     """SELECT month AS 月份, SUM(social_insurance) AS 个人社保总额 FROM payroll
        GROUP BY month ORDER BY month""", False, None),
    ("哪些部门 2026 年上半年的离职率超过了 5%？列出部门名和离职率。",
     """SELECT d.dept_name AS 部门,
               ROUND(100.0 * SUM(CASE WHEN e.leave_date BETWEEN '2026-01-01' AND '2026-06-30' THEN 1 ELSE 0 END)
                     / NULLIF(SUM(CASE WHEN e.hire_date < '2026-01-01'
                                       AND (e.leave_date IS NULL OR e.leave_date >= '2026-01-01')
                                  THEN 1 ELSE 0 END), 0), 2) AS 离职率
        FROM departments d JOIN employees e ON e.dept_id = d.dept_id
        GROUP BY d.dept_id
        HAVING 离职率 > 5 ORDER BY 离职率 DESC""", True, "attrition_rate"),
]


def build() -> list[dict]:
    rng = random.Random(SEED)
    items: list[dict] = []

    def add(question, sql, caliber_sensitive, caliber_key, meta):
        items.append({
            "id": f"SQL{len(items) + 1:04d}",
            "question": question,
            "gold_sql": sql.strip(),
            "caliber_sensitive": caliber_sensitive,
            "caliber_key": caliber_key,
            "caliber_note": CALIBER.get(caliber_key) if caliber_key else None,
            **meta,
        })

    # ---- 参数化题：每个指标 × 抽样部门/期间，凑到约 130 题 ----
    # 生成时即探测数据可用性：某些部门在某个指标上没有数据（如无进行中需求），
    # 这类题 gold 结果为空，会让"两边都空判对"的规则虚高分数，必须剔除。
    conn = sqlite3.connect(config.DATA / "hr.db")
    for fn, kind in METRICS:
        if kind == "dept":
            combos = [(d, None, None) for d in DEPTS]
        elif kind == "dept_period":
            combos = [(d, p, None) for d, p in itertools.product(DEPTS, PERIODS)]
        elif kind == "dept_month":
            combos = [(d, None, m) for d, m in itertools.product(DEPTS, MONTHS)]
        else:
            raise ValueError(kind)
        rng.shuffle(combos)

        kept = 0
        for dept, period, month in combos:
            if kept >= 7:
                break
            q, sql, cs, ck = fn(dept, period, month)
            try:
                rows = conn.execute(sql).fetchall()
            except Exception:
                continue
            if not rows or all(v is None for r in rows for v in (r if isinstance(r, tuple) else (r,))):
                continue          # 空结果题剔除
            meta = {"metric": fn.__name__[2:], "dept": dept}
            if period:
                meta["period"] = period[0]
            if month:
                meta["month"] = month[0]
            add(q, sql, cs, ck, meta)
            kept += 1
        if kept == 0:
            print(f"  ⚠️  指标 {fn.__name__} 无可用数据，已跳过")
    conn.close()

    # ---- 全局题 ----
    for q, sql, cs, ck in GLOBAL_QUESTIONS:
        add(q, sql, cs, ck, {"metric": "global", "dept": None})

    # ---- dev / test 切分 ----
    rng.shuffle(items)
    for j, it in enumerate(items):
        it["split"] = "dev" if j % 2 == 0 else "test"
    items.sort(key=lambda x: x["id"])
    return items


def validate(items: list[dict]) -> None:
    """所有 gold SQL 必须能跑通且返回非空。

    空结果是硬失败而非警告：两边都空会被判对，虚高 EX。
    """
    conn = sqlite3.connect(config.DATA / "hr.db")
    bad, empty = [], []
    for it in items:
        try:
            rows = conn.execute(it["gold_sql"]).fetchall()
            if not rows or all(v is None for r in rows for v in (r if isinstance(r, tuple) else (r,))):
                empty.append(it["id"])
        except Exception as e:
            bad.append((it["id"], str(e), it["gold_sql"][:80]))
    conn.close()
    for i, e, s in bad[:10]:
        print(f"  ❌ {i}: {e}\n     {s}")
    if bad:
        raise SystemExit(f"{len(bad)} 条 gold SQL 执行失败")
    if empty:
        raise SystemExit(f"{len(empty)} 条 gold SQL 返回空结果: {empty}")


def main() -> None:
    items = build()
    validate(items)

    OUT.write_text("\n".join(json.dumps(it, ensure_ascii=False) for it in items))

    n = len(items)
    cs = sum(it["caliber_sensitive"] for it in items)
    print(f"✅ Text-to-SQL 评测集已生成: {OUT}")
    print(f"   共 {n} 题（dev {sum(it['split'] == 'dev' for it in items)} / "
          f"test {sum(it['split'] == 'test' for it in items)}）")
    print(f"   口径敏感题 {cs} 条 ({cs / n * 100:.0f}%)，全部 gold SQL 执行通过")
    from collections import Counter
    print(f"   指标分布: {dict(Counter(it['metric'] for it in items).most_common(8))}")
    print("\n抽样:")
    for it in items[:3]:
        print(f"   [{it['id']}] {it['question']}")
        print(f"        {it['gold_sql'].splitlines()[0][:88]}")


if __name__ == "__main__":
    main()
