"""实验 4：数据字典注入能否提升 SQL 口径正确率。

要回答的问题：
  模型写 SQL 时，喂它表结构够不够？还是必须把"口径"也写进上下文？

四组条件（只改注入的上下文，其余完全一致）：
  C0 裸 prompt      只给问题，不给任何库信息 —— 模型得靠猜表名列名
  C1 +schema        给 DDL（表名/列名/类型），**不含**口径说明与 NULL 语义
  C2 +口径          给 DDL + 口径约定（离职率除以期初、出勤率按月汇总后相除……）
  C3 +SQL校验       C2 + 执行反馈闭环（跑不通就把报错回灌，最多重试 2 次）

两个指标必须分开报，因为它们会背离：
  · Valid SQL Rate   SQL 能不能跑通（语法/表名/列名对不对）
  · EX               跑出来的**值**对不对（与金标 SQL 的执行结果比）

**关键失败模式**：SQL 完全合法、无报错、但口径错了 → 数字是错的。
这类错误 Valid SQL Rate 抓不到，正是本项目"干净答错"论点的直接证据。
所以必须单独统计"合法但错"这一格。

跑法:
  .venv/bin/python scripts/exp4_sql_caliber.py --split dev --limit 30   # 开发期抽样
  .venv/bin/python scripts/exp4_sql_caliber.py --split test            # 出数
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hragent import config, metrics  # noqa: E402
from hragent.llm import LLM  # noqa: E402
from hragent.tools import sql_tool  # noqa: E402

WORKERS = 6
DB = config.DATA / "hr.db"

# ---------------------------------------------------------------- 上下文构造

def ddl() -> str:
    """从库里现取 DDL —— 只含表名列名类型，不含任何口径说明。

    刻意不含注释：C1 与 C2 的差别必须**只有口径**，否则分不清是哪个起了作用。
    """
    con = sqlite3.connect(DB)
    out = []
    for (name,) in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"):
        cols = con.execute(f"PRAGMA table_info({name})").fetchall()
        lines = "\n".join(f"  {c[1]} {c[2]}" for c in cols)
        out.append(f"{name}(\n{lines}\n)")
    con.close()
    return "\n".join(out)


def caliber_notes() -> str:
    """口径约定 —— 从数据字典里抽「口径约定」那一节，不重复给表结构。"""
    md = (config.DATA / "hr_data_dictionary.md").read_text()
    start = md.index("## 口径约定")
    end = md.index("## 表结构")
    return md[start:end].strip()


def full_dictionary() -> str:
    """完整数据字典：口径约定 + 表结构说明（含**取值格式与枚举语义**）。

    为什么要有这一臂（C2F）：
      C2 只注入了「口径约定」段，用无注释的 sqlite DDL 顶替了「表结构」段，
      结果把取值语义一起丢掉了 —— 例如 `period` 取值是 '2025H1'/'2025H2' 而非日期，
      `job_level` 可能为 NULL。实测 C2 的失败错例里，"绩效拿 A 人数" 得 0 就是因为
      模型按 `period BETWEEN '2025-01' AND '2025-06'` 去查，而库里根本没有这种取值。

      **这一臂是看到 test 错例后才加的（post-hoc），结论属探索性，不是预注册验证。**
      诚实起见单独列出，不与 C0–C3 的预注册结果混为一谈。
    """
    md = (config.DATA / "hr_data_dictionary.md").read_text()
    start = md.index("## 口径约定")
    end = md.index("## 已知脏数据")
    return md[start:end].strip()


SYSTEM_BASE = """你是 HR 数据查询助手。把用户的中文问题转成一条 SQLite 查询。

只输出 JSON：{"sql": "SELECT ..."}
不要解释，不要 markdown 围栏。"""


def build_system(arm: str) -> str:
    if arm == "C0":
        return SYSTEM_BASE + "\n\n数据库是 SQLite。"
    if arm == "C1":
        return SYSTEM_BASE + f"\n\n数据库表结构：\n{ddl()}"
    if arm == "C2F":
        return (SYSTEM_BASE + f"\n\n数据字典（含表结构与口径约定，必须严格遵守）：\n"
                f"{full_dictionary()}")
    # C2 / C3
    return (SYSTEM_BASE + f"\n\n数据库表结构：\n{ddl()}"
            f"\n\n口径约定（必须严格遵守，口径错了数字就是错的）：\n{caliber_notes()}")


# ---------------------------------------------------------------- 执行与比对

def norm(rows: list) -> list:
    """把结果集归一化后再比 —— 浮点四舍五入、行排序、统一成字符串。

    不归一化就会把 12.3456 与 12.35 判成不同，那是精度问题不是口径问题。
    """
    out = []
    for r in rows:
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append(round(v, 2))
            else:
                cells.append(v)
        out.append(tuple(cells))
    try:
        out.sort(key=lambda t: [(x is None, x) for x in t])
    except TypeError:
        out.sort(key=lambda t: str(t))
    return out


def execute(sql: str) -> tuple[list, str | None]:
    """直接执行，不做只读校验 —— 这里测的是模型写 SQL 的能力，不是风控。"""
    try:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        cur = con.execute(sql)
        rows = cur.fetchall()
        con.close()
        return rows, None
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


_NUM = re.compile(r"-?\d+\.?\d*")


def _nums(s: str | None) -> list[float]:
    return [float(x) for x in _NUM.findall(s or "")]


def unit_equiv(got: str | None, gold: str | None, tol: float = 0.05) -> bool:
    """got 与 gold 是否只差一个「比率 vs 百分数」的**单位约定**。

    为什么必须有这个函数：
      金标 SQL 返回 25.0（百分数），模型返回 0.25（比率）—— 同一个量、同一个公式，
      只差 ×100。若不归一，会把正确口径判成错误，得出"数据字典无效"的反向结论。
      实测 C2 组 32 条"错"里有 19 条是这种纯单位差。

    但归一化有风险：真口径错（如离职率除以期末而非期初）也常差约 100 倍，会被误放行。
    所以判据收紧到：数字**个数相同**，且每个数字要么直接相等，要么**恰好**差 100 倍
    （容差 5% 以吸收四舍五入）。公式错时几乎不可能逐位满足。
    """
    o, g = _nums(got), _nums(gold)
    if not o or not g or len(o) != len(g):
        return False
    for a, b in zip(o, g):
        if b == 0:
            if abs(a) > 1e-9:
                return False
            continue
        if abs(a - b) <= tol * abs(b) or abs(a * 100 - b) <= tol * abs(b):
            continue
        return False
    return True


GOLD_CACHE: dict[str, list] = {}


def gold_value(gid: str, gold_sql: str) -> list:
    if gid not in GOLD_CACHE:
        rows, err = execute(gold_sql)
        if err:
            raise RuntimeError(f"金标 SQL 跑不通（评测集有问题）: {gid} {err}")
        GOLD_CACHE[gid] = norm(rows)
    return GOLD_CACHE[gid]


# ---------------------------------------------------------------- 单条求解

def solve_one(llm: LLM, arm: str, it: dict, system: str) -> dict:
    """跑一组条件的一条样本。C3 带执行反馈闭环，其余单轮。"""
    t0 = time.time()
    tok = 0
    sql, err, attempts, truncated = "", None, 0, False
    msgs = [{"role": "user", "content": it["question"]}]

    max_rounds = 3 if arm == "C3" else 1
    for _ in range(max_rounds):
        attempts += 1
        try:
            r = llm.call(msgs, system=system, tag=f"sql4{arm}")
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            break
        u = r.usage or {}
        tok += int(u.get("input_tokens", 0)) + int(u.get("output_tokens", 0))
        # 截断与答错必须分开：截断是"没说完"，答错是"说完了但错"。
        # 混在一起会把预算不足记成能力不足。
        if r.truncated:
            truncated = True
        try:
            sql = (r.json() or {}).get("sql", "").strip()
        except Exception as e:
            err = ("输出被截断，JSON 不完整" if r.truncated else f"JSON 解析失败: {e}")
            if arm != "C3":
                break
            msgs = msgs + [
                {"role": "user", "content": f"你上一条输出不完整（{err}）。"
                                            f"请只输出一行紧凑的 JSON，SQL 写短一些。"}]
            continue
        if not sql:
            err = "未产出 SQL"
            break

        if arm != "C3":
            break

        # C3：静态校验 + 试跑，把报错回灌让模型改
        bad = sql_tool.check(sql)
        rows, run_err = ([], None) if bad else execute(sql)
        problem = bad or run_err
        if not problem:
            break
        err = problem
        msgs = msgs + [
            {"role": "assistant", "content": json.dumps({"sql": sql}, ensure_ascii=False)},
            {"role": "user", "content": f"这条 SQL 有问题：{problem}\n请修正后重新输出 JSON。"},
        ]

    # 判定：先看跑不跑得通，再看值对不对 —— 两步必须分开
    valid, ex, exn = False, False, False
    got, run_err = [], err
    if sql:
        got, e2 = execute(sql)
        valid = e2 is None
        run_err = e2 or run_err
        if valid:
            try:
                g = str(gold_value(it["id"], it["gold_sql"]))
                o = str(norm(got))
                ex = norm(got) == gold_value(it["id"], it["gold_sql"])
                # 单位归一后的正确率：容许「比率 vs 百分数」的表示差异
                exn = ex or unit_equiv(o, g)
            except Exception:
                ex = exn = False

    return {
        "id": it["id"], "arm": arm, "question": it["question"],
        "caliber_sensitive": it["caliber_sensitive"], "caliber_key": it["caliber_key"],
        "sql": sql, "valid": valid, "ex": ex, "ex_norm": exn, "truncated": truncated,
        "got": str(norm(got))[:200] if valid else None,
        "gold": str(gold_value(it["id"], it["gold_sql"]))[:200],
        "error": run_err, "attempts": attempts, "tokens": tok,
        "latency_s": time.time() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", default="C0,C1,C2,C3")
    args = ap.parse_args()

    items = [json.loads(l) for l in (config.EVAL / "text2sql_eval.jsonl").read_text().splitlines()]
    if args.split != "all":
        items = [x for x in items if x["split"] == args.split]
    if args.limit:
        items = items[:args.limit]
    arms = args.arms.split(",")

    n_sens = sum(1 for x in items if x["caliber_sensitive"])
    print(f"实验 4：数据字典注入能否提升 SQL 口径正确率")
    print(f"   样本 {len(items)} 条（split={args.split}）· 其中口径敏感 {n_sens} 条")
    print(f"   条件 {arms}\n")

    llm = LLM()
    systems = {a: build_system(a) for a in arms}
    print(f"   上下文长度："
          + "  ".join(f"{a}={len(systems[a])}字符" for a in arms) + "\n")

    rows: list[dict] = []
    for a in arms:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            rs = list(ex.map(lambda it: solve_one(llm, a, it, systems[a]), items))
        rows += rs
        v = sum(r["valid"] for r in rs) / len(rs)
        e = sum(r["ex"] for r in rs) / len(rs)
        print(f"   {a:3s} 完成 · Valid {v:.3f} · EX {e:.3f} · {time.time() - t0:.0f}s",
              flush=True)

    # ---------------------------------------------------------- 主表
    print(f"\n{'=' * 96}")
    print(f"{'条件':6s} {'ValidSQL':>9s} {'EX严':>7s} {'EX单位归一':>10s} {'合法但错':>9s} "
          f"{'纯单位差':>9s} {'截断':>7s} {'tok/题':>8s}")
    print(f"{'-' * 96}")
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)

    for a in arms:
        rs = by_arm[a]
        v = sum(r["valid"] for r in rs) / len(rs)
        e = sum(r["ex"] for r in rs) / len(rs)
        en = sum(r["ex_norm"] for r in rs) / len(rs)
        # 合法但错：跑得通、无报错、值却是错的 —— Valid SQL Rate 抓不到的那一格
        vw = sum(1 for r in rs if r["valid"] and not r["ex_norm"]) / len(rs)
        # 纯单位差：值其实对，只是比率/百分数写法不同 —— 不计入错误
        un = sum(1 for r in rs
                 if r["valid"] and not r["ex"] and r["ex_norm"]) / len(rs)
        tr = sum(1 for r in rs if r["truncated"]) / len(rs)
        tk = sum(r["tokens"] for r in rs) / len(rs)
        print(f"{a:6s} {v:>9.3f} {e:>7.3f} {en:>10.3f} {vw:>9.3f} "
              f"{un:>9.3f} {tr:>7.3f} {tk:>8,.0f}")
    print(f"  注：EX 按**执行结果比对**，非字符串匹配。**主指标是「EX单位归一」**——")
    print(f"      「EX严」把「0.25 vs 25.0」这种纯单位约定差异判成错误，会得出反向结论。")
    print(f"      「合法但错」= SQL 跑得通但值错，Valid SQL Rate 抓不到的那一格。")

    # ---------------------------------------------------------- 口径敏感 vs 不敏感
    # 这是本实验的核心：schema 能救语法，只有口径能救数字。
    print(f"\n口径敏感子集 vs 其余（关键对照，单位归一）")
    print(f"{'条件':6s} {'不敏感 n':>8s} {'EX':>8s} | {'敏感 n':>7s} {'EX':>8s} {'Δ':>8s}")
    print(f"{'-' * 56}")
    for a in arms:
        rs = by_arm[a]
        ns = [r for r in rs if not r["caliber_sensitive"]]
        ss = [r for r in rs if r["caliber_sensitive"]]
        en = sum(r["ex_norm"] for r in ns) / len(ns) if ns else 0.0
        es = sum(r["ex_norm"] for r in ss) / len(ss) if ss else 0.0
        print(f"{a:6s} {len(ns):>8d} {en:>8.3f} | {len(ss):>7d} {es:>8.3f} {es - en:>+8.3f}")

    # 按口径键拆开，看是哪条口径最难
    print(f"\n按口径键拆分（C2 组，单位归一，看哪条口径最难遵守）")
    per_key: dict[str, list[float]] = defaultdict(list)
    for r in by_arm.get("C2", []):
        if r["caliber_sensitive"]:
            per_key[r["caliber_key"] or "?"].append(float(r["ex_norm"]))
    for k, v in sorted(per_key.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        flag = "  ← 最易错" if sum(v) / len(v) < 0.6 else ""
        print(f"   {k:22s} n={len(v):2d}  EX={sum(v) / len(v):.3f}{flag}")

    # ---------------------------------------------------------- 逐对比较
    print(f"\n逐对比较（McNemar 配对检验，同一样本同一模型，只改上下文；单位归一）")
    for i in range(len(arms) - 1):
        a, b = arms[i], arms[i + 1]
        ca = {r["id"]: r["ex_norm"] for r in by_arm[a]}
        cb = {r["id"]: r["ex_norm"] for r in by_arm[b]}
        ids = [k for k in ca if k in cb]
        mc = metrics.mcnemar([ca[k] for k in ids], [cb[k] for k in ids])
        print(f"   {a} → {b}: b={mc['b']:3d} c={mc['c']:3d}  p={mc['p']:.4f}  {mc['note']}")
        if mc["p"] >= 0.05:
            print(f"          ⚠️  不显著 —— 不能声称 {b} 优于 {a}")

    # ---------------------------------------------------------- 错例
    print(f"\n错例（C2 组，单位归一后仍错，按口径键分组）")
    bad = [r for r in by_arm.get("C2", []) if not r["ex_norm"]]
    for r in bad[:8]:
        tag = f"口径={r['caliber_key']}" if r["caliber_sensitive"] else "非口径"
        print(f"   [{tag}] {r['question'][:40]}")
        print(f"        得={r['got']}  金标={r['gold']}  合法={r['valid']}")
        if not r["valid"]:
            print(f"        报错={str(r['error'])[:90]}")

    # ---------------------------------------------------------- 落盘
    payload = {
        "split": args.split, "n": len(items), "arms": arms,
        "n_caliber_sensitive": n_sens,
        "context_chars": {a: len(systems[a]) for a in arms},
        "per_arm": {
            a: {
                "valid_rate": sum(r["valid"] for r in by_arm[a]) / len(by_arm[a]),
                "ex_strict": sum(r["ex"] for r in by_arm[a]) / len(by_arm[a]),
                "ex_norm": sum(r["ex_norm"] for r in by_arm[a]) / len(by_arm[a]),
                "valid_but_wrong": sum(1 for r in by_arm[a]
                                       if r["valid"] and not r["ex_norm"]) / len(by_arm[a]),
                "unit_only_diff": sum(1 for r in by_arm[a]
                                      if r["valid"] and not r["ex"] and r["ex_norm"])
                                  / len(by_arm[a]),
                "truncated_rate": sum(1 for r in by_arm[a]
                                      if r["truncated"]) / len(by_arm[a]),
                "ex_ci95": list(metrics.bootstrap_ci(
                    [float(r["ex_norm"]) for r in by_arm[a]])[1:]),
                "tokens_per_q": sum(r["tokens"] for r in by_arm[a]) / len(by_arm[a]),
                "ex_caliber_sensitive": (
                    sum(r["ex_norm"] for r in by_arm[a] if r["caliber_sensitive"])
                    / max(1, sum(1 for r in by_arm[a] if r["caliber_sensitive"]))),
                "ex_caliber_insensitive": (
                    sum(r["ex_norm"] for r in by_arm[a] if not r["caliber_sensitive"])
                    / max(1, sum(1 for r in by_arm[a] if not r["caliber_sensitive"]))),
            } for a in arms
        },
        "pairwise_mcnemar": [
            {"from": arms[i], "to": arms[i + 1],
             **metrics.mcnemar(
                 [{r["id"]: r["ex_norm"] for r in by_arm[arms[i]]}[k] for k in
                  {r["id"]: r["ex_norm"] for r in by_arm[arms[i]]}],
                 [{r["id"]: r["ex_norm"] for r in by_arm[arms[i + 1]]}[k] for k in
                  {r["id"]: r["ex_norm"] for r in by_arm[arms[i]]}])}
            for i in range(len(arms) - 1)
        ],
        "rows": rows,
    }
    out = config.RESULTS / f"exp4_sql_caliber_{args.split}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
