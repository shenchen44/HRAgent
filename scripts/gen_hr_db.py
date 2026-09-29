"""生成合成 HR 数据库（SQLite）+ 数据字典。

设计原则:
  1. 确定性 —— 固定随机种子，任何人重跑得到同一份库，评测结果可复现
  2. 贴近真实 —— 刻意注入脏数据（缺失值、格式不一致、离职员工仍有历史记录），
     因为真实 Text-to-SQL 的难点从来不在 schema 干净的时候
  3. 覆盖 HR COE 高频域 —— 编制/招聘/考勤/绩效/薪酬/培训/离职

跑法: .venv/bin/python scripts/gen_hr_db.py
产出: data/hr.db, data/hr_data_dictionary.md
"""

from __future__ import annotations

import random
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hragent import config  # noqa: E402

SEED = 20260928
DB_PATH = config.DATA / "hr.db"
DICT_PATH = config.DATA / "hr_data_dictionary.md"

DEPARTMENTS = [
    ("D01", "研发中心", "CC1001", 120), ("D02", "产品部", "CC1002", 30),
    ("D03", "市场部", "CC2001", 40), ("D04", "销售部", "CC2002", 80),
    ("D05", "人力资源部", "CC3001", 20), ("D06", "财务部", "CC3002", 18),
    ("D07", "法务部", "CC3003", 10), ("D08", "客户成功部", "CC4001", 45),
    ("D09", "供应链部", "CC4002", 35), ("D10", "质量部", "CC4003", 22),
    ("D11", "设计部", "CC5001", 25), ("D12", "行政部", "CC5002", 15),
]

TITLES = {
    "D01": ["初级开发工程师", "开发工程师", "高级开发工程师", "技术专家", "架构师"],
    "D02": ["产品助理", "产品经理", "高级产品经理", "产品总监"],
    "D03": ["市场专员", "市场经理", "品牌经理", "市场总监"],
    "D04": ["销售代表", "客户经理", "高级客户经理", "销售总监"],
    "D05": ["HR 专员", "HRBP", "招聘经理", "HRD"],
    "D06": ["会计", "高级会计", "财务经理", "财务总监"],
    "D07": ["法务专员", "法务经理", "法务总监"],
    "D08": ["客户成功专员", "客户成功经理", "客户成功总监"],
    "D09": ["采购专员", "供应链经理", "供应链总监"],
    "D10": ["质量工程师", "高级质量工程师", "质量经理"],
    "D11": ["UI 设计师", "交互设计师", "设计总监"],
    "D12": ["行政专员", "行政经理"],
}

LEVELS = ["P4", "P5", "P6", "P7", "P8"]
CITIES = ["上海", "杭州", "北京", "深圳", "成都", "苏州"]
SOURCES = ["BOSS直聘", "猎聘", "内推", "官网", "校招", "智联招聘"]
RATINGS = ["A", "B", "C"]
SURNAMES = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
GIVEN = ["伟", "芳", "娜", "敏", "静", "磊", "强", "军", "洋", "勇", "艳", "杰",
         "娟", "涛", "明", "超", "秀英", "霞", "平", "刚", "桂英", "文", "辉", "婷"]


def cn_name(rng: random.Random) -> str:
    return rng.choice(SURNAMES) + rng.choice(GIVEN)


def d(dt: date) -> str:
    return dt.isoformat()


def build(conn: sqlite3.Connection, rng: random.Random) -> dict[str, int]:
    c = conn.cursor()
    c.executescript("""
    DROP TABLE IF EXISTS departments;
    DROP TABLE IF EXISTS employees;
    DROP TABLE IF EXISTS recruitment;
    DROP TABLE IF EXISTS candidates;
    DROP TABLE IF EXISTS attendance;
    DROP TABLE IF EXISTS performance;
    DROP TABLE IF EXISTS payroll;
    DROP TABLE IF EXISTS training;

    CREATE TABLE departments (
        dept_id        TEXT PRIMARY KEY,
        dept_name      TEXT NOT NULL,
        cost_center    TEXT NOT NULL,
        headcount_budget INTEGER NOT NULL   -- 编制人数
    );

    CREATE TABLE employees (
        emp_id      TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        dept_id     TEXT NOT NULL REFERENCES departments(dept_id),
        job_title   TEXT NOT NULL,
        job_level   TEXT,                   -- 脏数据：部分历史员工未定级
        city        TEXT,
        hire_date   TEXT NOT NULL,
        leave_date  TEXT,                   -- 非空表示已离职
        status      TEXT NOT NULL,          -- 在职 / 离职
        base_salary INTEGER,                -- 月基本工资（元）
        birth_year  INTEGER
    );

    CREATE TABLE recruitment (
        req_id          TEXT PRIMARY KEY,
        dept_id         TEXT NOT NULL REFERENCES departments(dept_id),
        job_title       TEXT NOT NULL,
        headcount_planned INTEGER NOT NULL,
        headcount_filled  INTEGER NOT NULL DEFAULT 0,
        open_date       TEXT NOT NULL,
        close_date      TEXT,
        status          TEXT NOT NULL       -- 进行中 / 已完成 / 已关闭
    );

    CREATE TABLE candidates (
        candidate_id TEXT PRIMARY KEY,
        req_id       TEXT NOT NULL REFERENCES recruitment(req_id),
        name         TEXT NOT NULL,
        source       TEXT NOT NULL,         -- 渠道
        apply_date   TEXT NOT NULL,
        stage        TEXT NOT NULL,         -- 简历初筛/一面/二面/HR面/Offer/入职/淘汰
        offer_date   TEXT,
        onboard_date TEXT
    );

    CREATE TABLE attendance (
        emp_id          TEXT NOT NULL REFERENCES employees(emp_id),
        month           TEXT NOT NULL,      -- YYYY-MM
        work_days       INTEGER NOT NULL,   -- 应出勤
        actual_days     REAL,               -- 脏数据：部分月份缺失
        leave_days      REAL NOT NULL DEFAULT 0,
        overtime_hours  REAL NOT NULL DEFAULT 0,
        PRIMARY KEY (emp_id, month)
    );

    CREATE TABLE performance (
        emp_id  TEXT NOT NULL REFERENCES employees(emp_id),
        period  TEXT NOT NULL,              -- 2025H1 / 2025H2
        rating  TEXT NOT NULL,              -- A / B / C
        score   REAL,
        PRIMARY KEY (emp_id, period)
    );

    CREATE TABLE payroll (
        emp_id         TEXT NOT NULL REFERENCES employees(emp_id),
        month          TEXT NOT NULL,
        base_salary    INTEGER NOT NULL,
        bonus          INTEGER NOT NULL DEFAULT 0,
        social_insurance INTEGER NOT NULL,  -- 个人社保
        housing_fund   INTEGER NOT NULL,    -- 个人公积金
        gross_pay      INTEGER NOT NULL,
        net_pay        INTEGER NOT NULL,
        PRIMARY KEY (emp_id, month)
    );

    CREATE TABLE training (
        emp_id         TEXT NOT NULL REFERENCES employees(emp_id),
        course         TEXT NOT NULL,
        planned_hours  REAL NOT NULL,
        actual_hours   REAL,                -- 脏数据：未完成时为空
        train_date     TEXT NOT NULL
    );
    """)

    c.executemany("INSERT INTO departments VALUES (?,?,?,?)", DEPARTMENTS)

    # ---- 员工：约 480 人，含 2024-2026 的入职与离职 ----
    employees, emp_ids = [], []
    eid = 0
    for dept_id, _, _, budget in DEPARTMENTS:
        n = int(budget * rng.uniform(0.82, 0.98))
        for _ in range(n):
            eid += 1
            emp_id = f"E{eid:04d}"
            emp_ids.append(emp_id)
            hire = date(2019, 1, 1) + timedelta(days=rng.randint(0, 2500))
            # 12% 概率已离职，离职日期落在 2024-2026
            left = None
            if rng.random() < 0.12:
                left = max(hire + timedelta(days=180),
                           date(2024, 1, 1) + timedelta(days=rng.randint(0, 950)))
                if left > date(2026, 9, 1):
                    left = None
            status = "离职" if left else "在职"
            level = rng.choice(LEVELS) if rng.random() > 0.08 else None
            employees.append((
                emp_id, cn_name(rng), dept_id, rng.choice(TITLES[dept_id]), level,
                rng.choice(CITIES), d(hire), d(left) if left else None, status,
                rng.choice([9000, 12000, 15000, 18000, 22000, 28000, 35000, 45000]),
                rng.randint(1978, 2002),
            ))
    c.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?,?,?,?,?,?)", employees)

    active = [e for e in employees if e[8] == "在职"]

    # ---- 招聘需求 ----
    reqs, rid = [], 0
    for dept_id, _, _, _ in DEPARTMENTS:
        for _ in range(rng.randint(3, 7)):
            rid += 1
            req_id = f"REQ{rid:04d}"
            planned = rng.randint(1, 6)
            filled = rng.randint(0, planned)
            open_d = date(2025, 1, 1) + timedelta(days=rng.randint(0, 600))
            status = rng.choice(["进行中", "已完成", "已关闭"])
            close_d = None if status == "进行中" else open_d + timedelta(days=rng.randint(20, 150))
            reqs.append((req_id, dept_id, rng.choice(TITLES[dept_id]), planned, filled,
                         d(open_d), d(close_d) if close_d else None, status))
    c.executemany("INSERT INTO recruitment VALUES (?,?,?,?,?,?,?,?)", reqs)

    # ---- 候选人 ----
    cands, cid = [], 0
    STAGES = ["简历初筛", "一面", "二面", "HR面", "Offer", "入职", "淘汰"]
    for req in reqs:
        for _ in range(rng.randint(8, 35)):
            cid += 1
            apply_d = date.fromisoformat(req[5]) + timedelta(days=rng.randint(0, 90))
            stage = rng.choices(STAGES, weights=[30, 18, 12, 8, 6, 5, 21])[0]
            offer_d = apply_d + timedelta(days=rng.randint(10, 45)) if stage in ("Offer", "入职") else None
            onboard_d = (offer_d + timedelta(days=rng.randint(7, 30))) if stage == "入职" and offer_d else None
            cands.append((f"C{cid:05d}", req[0], cn_name(rng), rng.choice(SOURCES),
                          d(apply_d), stage, d(offer_d) if offer_d else None,
                          d(onboard_d) if onboard_d else None))
    c.executemany("INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?)", cands)

    # ---- 考勤 / 绩效 / 薪酬 / 培训：仅覆盖 2026-01 ~ 2026-06 ----
    months = [f"2026-{m:02d}" for m in range(1, 7)]
    att, perf, pay, train = [], [], [], []
    for e in active:
        emp_id, _, dept_id, _, level, _, hire, _, _, salary, _ = e
        for mo in months:
            work = rng.choice([20, 21, 22, 23])
            actual = None if rng.random() < 0.03 else round(work - rng.randint(0, 2) + rng.random(), 1)
            att.append((emp_id, mo, work, actual, round(rng.uniform(0, 3), 1),
                        round(rng.choice([0, 0, 0, 2, 5, 8, 12, 20, 36]) + rng.random(), 1)))
            si = int(salary * 0.105)
            hf = int(salary * 0.12)
            bonus = rng.choice([0, 0, 0, 1000, 3000, 8000])
            gross = salary + bonus
            pay.append((emp_id, mo, salary, bonus, si, hf, gross, gross - si - hf))
        for period in ("2025H1", "2025H2"):
            rating = rng.choices(RATINGS, weights=[10, 75, 15])[0]
            perf.append((emp_id, period, rating, round(rng.uniform(60, 100), 1)))
        for course in rng.sample(["信息安全意识", "管理者领导力", "新员工入职培训",
                                  "商务沟通", "项目管理实战", "合规与反舞弊"],
                                 rng.randint(1, 3)):
            planned = float(rng.choice([2, 4, 8, 16]))
            actual = None if rng.random() < 0.12 else round(planned * rng.uniform(0.6, 1.0), 1)
            train.append((emp_id, course, planned, actual,
                          d(date(2026, rng.randint(1, 6), rng.randint(1, 28)))))

    c.executemany("INSERT INTO attendance VALUES (?,?,?,?,?,?)", att)
    c.executemany("INSERT INTO performance VALUES (?,?,?,?)", perf)
    c.executemany("INSERT INTO payroll VALUES (?,?,?,?,?,?,?,?)", pay)
    c.executemany("INSERT INTO training VALUES (?,?,?,?,?)", train)

    conn.commit()
    return {
        "departments": len(DEPARTMENTS), "employees": len(employees),
        "在职": len(active), "离职": len(employees) - len(active),
        "recruitment": len(reqs), "candidates": len(cands),
        "attendance": len(att), "performance": len(perf),
        "payroll": len(pay), "training": len(train),
    }


DATA_DICT = """# HR 数据字典

> 合成数据，用于 HRAgent 的 Text-to-SQL 评测。固定种子 {seed}，重跑结果一致。
> 数据库: `data/hr.db` (SQLite)

## 口径约定（Text-to-SQL 的关键，模型必须知道）

| 指标 | 口径 | 说明 |
|---|---|---|
| 在职 | `employees.status = '在职'` | `leave_date IS NULL` 等价 |
| 离职率 | 离职人数 / 期初人数 | **不是**除以期末人数，也**不是**除以往期平均 |
| 空缺率 | (编制 - 在职) / 编制 | 编制取 `departments.headcount_budget` |
| 招聘完成率 | `headcount_filled / headcount_planned` | 按 `recruitment` 表逐需求计算，**不是**汇总后相除 |
| 平均招聘周期 | `close_date - open_date` 的均值 | 仅统计 `status = '已完成'` 的需求 |
| 人均产出 | 营收 / 期末在职人数 | 本项目暂无营收表，营收由外部给定 |
| 出勤率 | `actual_days / work_days` | 按月汇总后相除，**不是**逐人算完再平均 |
| 培训完成率 | `actual_hours / planned_hours` | `actual_hours IS NULL` 视为未完成，按 0 计 |
| 社保个人比例 | 养老 8% + 医疗 2% + 失业 0.5% ≈ 10.5% | `payroll.social_insurance` 已按此计算 |
| 公积金比例 | 个人 12%，单位 12% | `payroll.housing_fund` 仅含个人部分 |

## 表结构

### departments 部门
| 字段 | 类型 | 说明 |
|---|---|---|
| dept_id | TEXT | 主键，D01~D12 |
| dept_name | TEXT | 部门名 |
| cost_center | TEXT | 成本中心 |
| headcount_budget | INTEGER | **编制人数**（HC 上限） |

### employees 员工
| 字段 | 类型 | 说明 |
|---|---|---|
| emp_id | TEXT | 主键 |
| name | TEXT | 姓名 |
| dept_id | TEXT | 外键 → departments |
| job_title | TEXT | 职位 |
| job_level | TEXT | 职级 P4~P8，**可能为 NULL**（历史员工未定级） |
| city | TEXT | 工作城市 |
| hire_date | TEXT | 入职日期 YYYY-MM-DD |
| leave_date | TEXT | 离职日期，**NULL 表示在职** |
| status | TEXT | `在职` / `离职` |
| base_salary | INTEGER | 月基本工资（元） |
| birth_year | INTEGER | 出生年份 |

### recruitment 招聘需求
| 字段 | 类型 | 说明 |
|---|---|---|
| req_id | TEXT | 主键 |
| dept_id | TEXT | 外键 → departments |
| job_title | TEXT | 招聘职位 |
| headcount_planned | INTEGER | 计划招聘人数 |
| headcount_filled | INTEGER | 已到岗人数 |
| open_date | TEXT | 需求开启日期 |
| close_date | TEXT | 关闭日期，**进行中为 NULL** |
| status | TEXT | `进行中` / `已完成` / `已关闭` |

### candidates 候选人
| 字段 | 类型 | 说明 |
|---|---|---|
| candidate_id | TEXT | 主键 |
| req_id | TEXT | 外键 → recruitment |
| name | TEXT | 姓名 |
| source | TEXT | 渠道：BOSS直聘/猎聘/内推/官网/校招/智联招聘 |
| apply_date | TEXT | 投递日期 |
| stage | TEXT | 简历初筛/一面/二面/HR面/Offer/入职/淘汰 |
| offer_date | TEXT | Offer 日期，可为 NULL |
| onboard_date | TEXT | 入职日期，可为 NULL |

### attendance 考勤（2026-01 ~ 2026-06）
| 字段 | 类型 | 说明 |
|---|---|---|
| emp_id | TEXT | 外键 → employees |
| month | TEXT | YYYY-MM |
| work_days | INTEGER | 应出勤天数 |
| actual_days | REAL | 实出勤天数，**可能为 NULL**（数据缺失） |
| leave_days | REAL | 请假天数 |
| overtime_hours | REAL | 加班小时数 |

### performance 绩效（2025H1 / 2025H2）
| 字段 | 类型 | 说明 |
|---|---|---|
| emp_id | TEXT | 外键 → employees |
| period | TEXT | `2025H1` / `2025H2` |
| rating | TEXT | `A` / `B` / `C` |
| score | REAL | 绩效分 |

### payroll 薪酬（2026-01 ~ 2026-06）
| 字段 | 类型 | 说明 |
|---|---|---|
| emp_id | TEXT | 外键 → employees |
| month | TEXT | YYYY-MM |
| base_salary | INTEGER | 基本工资 |
| bonus | INTEGER | 奖金 |
| social_insurance | INTEGER | 个人社保（约 10.5%） |
| housing_fund | INTEGER | 个人公积金（12%） |
| gross_pay | INTEGER | 应发合计 |
| net_pay | INTEGER | 实发合计 |

### training 培训
| 字段 | 类型 | 说明 |
|---|---|---|
| emp_id | TEXT | 外键 → employees |
| course | TEXT | 课程名 |
| planned_hours | REAL | 计划学时 |
| actual_hours | REAL | 实际学时，**NULL 表示未完成** |
| train_date | TEXT | 培训日期 |

## 已知脏数据（刻意注入）

1. `employees.job_level` 约 8% 为 NULL
2. `attendance.actual_days` 约 3% 为 NULL
3. `training.actual_hours` 约 12% 为 NULL（未完成）
4. 离职员工仍保留考勤/薪酬历史记录 → **统计在职指标时必须先过滤 status**
5. `recruitment` 中 `headcount_filled` 可能小于 `headcount_planned` 但状态已为「已完成」（提前关闭）
"""


def main() -> None:
    rng = random.Random(SEED)
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    counts = build(conn, rng)
    conn.close()

    DICT_PATH.write_text(DATA_DICT.format(seed=SEED))

    print(f"✅ 数据库已生成: {DB_PATH}")
    for k, v in counts.items():
        print(f"   {k:14s} {v:>6,}")
    print(f"✅ 数据字典已生成: {DICT_PATH}")
    print(f"   大小: {DB_PATH.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
