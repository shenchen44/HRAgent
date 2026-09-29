# HR 数据字典

> 合成数据，用于 HRAgent 的 Text-to-SQL 评测。固定种子 20260928，重跑结果一致。
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
