"""口径即代码（D5）—— 指标注册表。

要解决的问题：
  "离职率"这种指标，分母用期初、期末还是期间平均，三种都"看起来对"，
  但数值能差出一倍。靠 prompt 里写一句"注意用期初人数"是拦不住的 ——
  模型会在长上下文里丢掉这条约束，而且丢了以后输出格式完全正常，**看不出来**。

做法：
  口径敏感指标的 SQL 由**代码固定**，模型只能选指标名和参数，不能自己写分母。
  于是"口径正确"是构造出来的，不是祈祷出来的。
  模型若绕过注册表自己写 SQL，其结果会与注册表结果比对 —— 口径正确率因此可测。

traps 字段记录该指标最常见的错法，用于 P3 的口径检测闸门与错因分析。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Callable

from .. import config

DEPT_SUB = "(SELECT dept_id FROM departments WHERE dept_name = ?)"


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    params: tuple[str, ...]
    definition: str
    traps: tuple[str, ...]
    build: Callable[..., str] = field(repr=False)
    unit: str = ""

    def sql(self, **kw) -> tuple[str, list]:
        """返回 (SQL, 参数)。参数用占位符，不做字符串拼接。"""
        missing = [p for p in self.params if p not in kw]
        if missing:
            raise ValueError(f"指标 {self.name} 缺少参数: {missing}")
        return self.build(**{p: kw[p] for p in self.params})


def _headcount(dept):
    return (f"SELECT COUNT(*) FROM employees WHERE dept_id = {DEPT_SUB} AND status = '在职'",
            [dept])


def _attrition(dept, start, end):
    # 分子：期间内离职；分母：期初在职（入职早于期初，且期初时尚未离职）
    return (f"""SELECT ROUND(100.0 * SUM(CASE WHEN leave_date BETWEEN ? AND ? THEN 1 ELSE 0 END)
                / NULLIF(SUM(CASE WHEN hire_date < ?
                                  AND (leave_date IS NULL OR leave_date >= ?)
                             THEN 1 ELSE 0 END), 0), 2)
            FROM employees WHERE dept_id = {DEPT_SUB}""",
            [start, end, start, start, dept])


def _vacancy(dept):
    return (f"""SELECT ROUND(100.0 * (d.headcount_budget - COUNT(e.emp_id)) / d.headcount_budget, 2)
            FROM departments d
            LEFT JOIN employees e ON e.dept_id = d.dept_id AND e.status = '在职'
            WHERE d.dept_name = ? GROUP BY d.dept_id, d.headcount_budget""", [dept])


def _avg_salary(dept):
    return (f"SELECT ROUND(AVG(base_salary), 2) FROM employees "
            f"WHERE dept_id = {DEPT_SUB} AND status = '在职'", [dept])


def _headcount_by_level(dept):
    return (f"SELECT job_level, COUNT(*) FROM employees WHERE dept_id = {DEPT_SUB} "
            f"AND status = '在职' GROUP BY job_level ORDER BY job_level", [dept])


def _attendance_rate(dept, month):
    # 汇总后相除 —— 不是逐人算完再平均
    return (f"""SELECT ROUND(100.0 * SUM(a.actual_days) / SUM(a.work_days), 2)
            FROM attendance a JOIN employees e ON e.emp_id = a.emp_id
            WHERE e.dept_id = {DEPT_SUB} AND a.month = ?""", [dept, month])


def _avg_overtime(dept, month):
    return (f"""SELECT ROUND(AVG(a.overtime_hours), 2)
            FROM attendance a JOIN employees e ON e.emp_id = a.emp_id
            WHERE e.dept_id = {DEPT_SUB} AND a.month = ?""", [dept, month])


def _perf_dist(dept, period):
    return (f"""SELECT p.rating, COUNT(*) FROM performance p JOIN employees e ON e.emp_id = p.emp_id
            WHERE e.dept_id = {DEPT_SUB} AND p.period = ?
            GROUP BY p.rating ORDER BY p.rating""", [dept, period])


def _perf_a_count(dept, period):
    return (f"""SELECT COUNT(*) FROM performance p JOIN employees e ON e.emp_id = p.emp_id
            WHERE e.dept_id = {DEPT_SUB} AND p.period = ? AND p.rating = 'A'""", [dept, period])


def _recruit_completion(dept):
    # 逐需求算完再平均 —— 不是汇总后相除
    return (f"""SELECT ROUND(100.0 * AVG(1.0 * headcount_filled / headcount_planned), 2)
            FROM recruitment WHERE dept_id = {DEPT_SUB} AND headcount_planned > 0""", [dept])


def _open_hc(dept):
    return (f"SELECT SUM(headcount_planned - headcount_filled) FROM recruitment "
            f"WHERE dept_id = {DEPT_SUB} AND status = '进行中'", [dept])


def _avg_recruit_cycle(dept):
    return (f"""SELECT ROUND(AVG(julianday(close_date) - julianday(open_date)), 1)
            FROM recruitment WHERE dept_id = {DEPT_SUB}
            AND status = '已完成' AND close_date IS NOT NULL""", [dept])


def _channel_dist(dept):
    return (f"""SELECT c.source, COUNT(*) FROM candidates c JOIN recruitment r ON r.req_id = c.req_id
            WHERE r.dept_id = {DEPT_SUB} GROUP BY c.source ORDER BY COUNT(*) DESC""", [dept])


def _offer_accept(dept):
    return (f"""SELECT ROUND(100.0 * SUM(CASE WHEN stage = '入职' THEN 1 ELSE 0 END)
                / NULLIF(SUM(CASE WHEN stage IN ('Offer','入职') THEN 1 ELSE 0 END), 0), 2)
            FROM candidates c JOIN recruitment r ON r.req_id = c.req_id
            WHERE r.dept_id = {DEPT_SUB}""", [dept])


def _training_completion(dept):
    # actual_hours 为 NULL 按 0 计 —— COALESCE 不能省
    return (f"""SELECT ROUND(100.0 * SUM(COALESCE(t.actual_hours, 0)) / SUM(t.planned_hours), 2)
            FROM training t JOIN employees e ON e.emp_id = t.emp_id
            WHERE e.dept_id = {DEPT_SUB}""", [dept])


def _social_insurance(dept, month):
    return (f"""SELECT SUM(p.social_insurance) FROM payroll p JOIN employees e ON e.emp_id = p.emp_id
            WHERE e.dept_id = {DEPT_SUB} AND p.month = ?""", [dept, month])


def _build_metrics() -> dict[str, Metric]:
    ms = [
        Metric("headcount", "在职人数", ("dept",), "COUNT(status='在职')", (), _headcount, "人"),
        Metric("attrition_rate", "离职率", ("dept", "start", "end"),
               "离职人数 ÷ 期初在职人数",
               ("用期末人数做分母", "用期间平均人数做分母", "把期间入职的人算进分母",
                "未按 dept 过滤", "把已离职员工排除在分母外"), _attrition, "%"),
        Metric("vacancy_rate", "编制空缺率", ("dept",),
               "(编制 - 在职) ÷ 编制", ("用离职人数代替缺口", "分母用了在职人数"),
               _vacancy, "%"),
        Metric("avg_salary", "平均月薪", ("dept",),
               "在职员工 base_salary 均值", ("未过滤 status 导致含离职员工", "用了 gross_pay"),
               _avg_salary, "元"),
        Metric("headcount_by_level", "职级分布", ("dept",),
               "在职员工按 job_level 分组计数",
               ("未处理 job_level 为 NULL（约 8%）", "未过滤 status"), _headcount_by_level, "人"),
        Metric("attendance_rate", "出勤率", ("dept", "month"),
               "汇总实出勤 ÷ 汇总应出勤",
               ("逐人算出勤率再平均", "actual_days 为 NULL 时按 0 计（应视为缺失）",
                "未过滤离职员工的考勤历史"), _attendance_rate, "%"),
        Metric("avg_overtime", "人均加班时长", ("dept", "month"),
               "加班小时均值", ("未过滤离职员工历史",), _avg_overtime, "小时"),
        Metric("perf_dist", "绩效分布", ("dept", "period"),
               "按 rating 分组计数", ("未按 period 过滤",), _perf_dist, "人"),
        Metric("perf_a_count", "A 档人数", ("dept", "period"),
               "rating='A' 计数", ("未按 period 过滤",), _perf_a_count, "人"),
        Metric("recruit_completion", "招聘完成率", ("dept",),
               "逐需求 (已到岗 ÷ 计划) 后取平均",
               ("汇总后相除（SUM(filled)/SUM(planned)）", "把 headcount_planned=0 的需求算进去",
                "只统计已完成需求"), _recruit_completion, "%"),
        Metric("open_hc", "剩余 HC", ("dept",),
               "进行中需求的 (计划 - 已到岗) 之和", ("未过滤 status='进行中'",), _open_hc, "人"),
        Metric("avg_recruit_cycle", "平均招聘周期", ("dept",),
               "已完成需求 (关闭日 - 开启日) 均值",
               ("未过滤 status='已完成'", "close_date 为 NULL 时未排除"), _avg_recruit_cycle, "天"),
        Metric("channel_dist", "渠道分布", ("dept",),
               "候选人按 source 分组计数", ("未按部门过滤",), _channel_dist, "人"),
        Metric("offer_accept", "Offer 接受率", ("dept",),
               "入职人数 ÷ (Offer + 入职)",
               ("分母漏掉仅到 Offer 未入职的人",), _offer_accept, "%"),
        Metric("training_completion", "培训完成率", ("dept",),
               "汇总实际学时 ÷ 汇总计划学时，actual_hours 为 NULL 按 0 计",
               ("未用 COALESCE 处理 NULL（约 12%）", "对完成率取平均而非汇总后相除"),
               _training_completion, "%"),
        Metric("social_insurance", "个人社保总额", ("dept", "month"),
               "social_insurance 求和", ("未按月份过滤",), _social_insurance, "元"),
    ]
    return {m.name: m for m in ms}


METRICS: dict[str, Metric] = _build_metrics()

# 口径敏感指标：这些必须走注册表，自由发挥极易出错
CALIBER_SENSITIVE = frozenset({
    "attrition_rate", "vacancy_rate", "attendance_rate", "recruit_completion",
    "training_completion", "avg_recruit_cycle", "offer_accept", "headcount_by_level",
})


def build_sql(name: str, **params) -> tuple[str, list]:
    if name not in METRICS:
        raise KeyError(f"未注册的指标: {name}（可用: {sorted(METRICS)}）")
    return METRICS[name].sql(**params)


def run(name: str, db: str | None = None, **params) -> dict:
    """执行注册指标，返回结果与可追溯的证据。"""
    sql, args = build_sql(name, **params)
    conn = sqlite3.connect(db or (config.DATA / "hr.db"))
    try:
        rows = conn.execute(sql, args).fetchall()
    finally:
        conn.close()
    m = METRICS[name]
    return {
        "metric": name, "label": m.label, "unit": m.unit, "definition": m.definition,
        "params": params, "sql": sql.strip(), "sql_args": args, "rows": rows,
        "value": _scalar(rows), "n_rows": len(rows),
    }


def _scalar(rows):
    if not rows:
        return None
    if len(rows) == 1 and len(rows[0]) == 1:
        return rows[0][0]
    return None


def catalog() -> str:
    """给编排器看的指标目录（不含 SQL 细节，只给口径说明与参数）。"""
    lines = []
    for m in METRICS.values():
        flag = "★" if m.name in CALIBER_SENSITIVE else " "
        lines.append(f"{flag} {m.name}({', '.join(m.params)}) — {m.label}：{m.definition}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(catalog())
    print("\n示例：")
    for name, kw in [("attrition_rate", {"dept": "研发中心", "start": "2026-01-01", "end": "2026-06-30"}),
                     ("attendance_rate", {"dept": "销售部", "month": "2026-03"}),
                     ("recruit_completion", {"dept": "市场部"})]:
        r = run(name, **kw)
        print(f"  {r['label']}({kw}) = {r['value']} {r['unit']}")
