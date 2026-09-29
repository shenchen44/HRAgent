"""重复运行工具 —— 量化 LLM 评测的运行间方差。

为什么必须做：
  同一份代码、同一份数据，只因为采样随机性，路由准确率在 0.833~0.880 之间漂移，
  配对的 McNemar p 值在 0.043~0.136 之间跳。**单次运行得到的"显著"可能是噪声。**
  只跑一次就宣称 A 优于 B，是在报告随机数。

做法：把实验脚本跑 N 次，每次收走结果 JSON，汇总关键指标的分布，
并统计"p<0.05 出现了几次"。若结论只在部分运行里成立，就必须如实标注为不稳健。

跑法:
  .venv/bin/python scripts/repeat.py --script scripts/exp3_routing.py \
      --args "--split test" --n 5 --metric executor_exact_match
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hragent import config  # noqa: E402


def dig(d, path: str):
    """按点分路径取值，如 baseline_flat.executor_exact_match 或 pairwise_mcnemar.0.p。

    支持列表下标：纯数字的段按 int 处理，否则按 dict 键。
    """
    cur = d
    for k in path.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(k)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            if k not in cur:
                return None
            cur = cur[k]
        else:
            return None
    return cur


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--args", default="")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--metric", default="executor_exact_match")
    ap.add_argument("--baseline-metric", default="")
    ap.add_argument("--p-path", default="",
                    help="p 值在结果 JSON 里的点分路径；留空则不统计显著性")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    script = Path(args.script)
    # 从脚本里推断它会写出哪个结果文件
    stem = script.stem
    split = "test"
    if "--split" in args.args:
        toks = args.args.split()
        split = toks[toks.index("--split") + 1]
    result_file = config.RESULTS / f"{stem}_{split}.json"
    if not result_file.exists():
        raise SystemExit(f"❌ 未找到结果文件 {result_file}，请先单独跑一次该实验")

    runs = []
    for i in range(args.n):
        print(f"  第 {i + 1}/{args.n} 次 …", flush=True)
        p = subprocess.run([sys.executable, str(script)] + args.args.split(),
                           capture_output=True, text=True, cwd=config.ROOT)
        if p.returncode != 0:
            print(p.stdout[-1500:])
            print(p.stderr[-1500:])
            raise SystemExit(f"❌ 第 {i + 1} 次运行失败")
        d = json.loads(result_file.read_text())
        runs.append({
            "main": dig(d, args.metric),
            "baseline": dig(d, args.baseline_metric) if args.baseline_metric else None,
            "p": dig(d, args.p_path) if args.p_path else None,
            "snapshot": d,
        })

    mains = [r["main"] for r in runs if r["main"] is not None]
    print(f"\n{'=' * 64}")
    print(f"重复运行汇总：{script.name} {args.args}")
    print(f"{'=' * 64}")
    print(f"\n主指标 {args.metric}（n={len(mains)} 次运行）")
    print(f"   均值 {statistics.mean(mains):.4f}   "
          f"标准差 {statistics.stdev(mains) if len(mains) > 1 else 0:.4f}")
    print(f"   最小 {min(mains):.4f}   最大 {max(mains):.4f}   "
          f"极差 {max(mains) - min(mains):.4f}")
    print(f"   各次: {[round(x, 4) for x in mains]}")

    if args.baseline_metric:
        bases = [r["baseline"] for r in runs if r["baseline"] is not None]
        ps = [r["p"] for r in runs if r["p"] is not None]
        diffs = [m - b for m, b in zip(mains, bases)]
        print(f"\n对照指标 {args.baseline_metric}")
        print(f"   均值 {statistics.mean(bases):.4f}   "
              f"各次: {[round(x, 4) for x in bases]}")
        print(f"\n差值（主 - 对照）")
        print(f"   均值 {statistics.mean(diffs):+.4f}   "
              f"最小 {min(diffs):+.4f}   最大 {max(diffs):+.4f}")
        print(f"   方向一致性: {sum(1 for d in diffs if d > 0)}/{len(diffs)} 次为正")
        if not ps:
            print(f"\n（未提供 --p-path，跳过显著性统计）")
        else:
            print(f"\nMcNemar p 值")
            print(f"   各次: {[round(x, 4) for x in ps]}")
            sig = sum(1 for x in ps if x < 0.05)
            # 只看"几次显著"太粗：4/5 和 0/5 的差别是天上地下，但 4/5 和 5/5 差不多。
            # 正确做法是算「零假设下出现至少这么多次显著」的概率。
            # 每次检验独立、α=0.05，故 sig ~ Binomial(n, 0.05)。
            n = len(ps)
            p_under_null = sum(math.comb(n, k) * 0.05 ** k * 0.95 ** (n - k)
                               for k in range(sig, n + 1))
            print(f"   p<0.05 出现 {sig}/{n} 次")
            print(f"   零假设下出现 ≥{sig}/{n} 次显著的概率 = {p_under_null:.2e}")
            if p_under_null >= 0.05:
                print(f"   ⚠️  与「没有效应」相容 —— 不能声称 A 优于 B。")
                print(f"      方向一致只说明可能有个小效应，样本量不足以确立它。")
            elif p_under_null >= 1e-3:
                print(f"   🟡 弱证据：方向一致且显著次数超出随机预期，但仍不稳健。")
                print(f"      建议写「方向一致，多数运行显著」，不写「稳健显著」。")
            else:
                print(f"   ✅ 强证据：零假设下几乎不可能出现这个显著次数。")
                print(f"      可写「效应稳健」，但需注明个别运行未达显著。")

        # 效应量与方差的对比 —— 判断是否只是样本不够
        if len(diffs) > 1:
            sd = statistics.stdev(diffs)
            print(f"\n   效应量 {statistics.mean(diffs):+.4f} vs 运行间标准差 {sd:.4f}"
                  f"  → 信噪比 {abs(statistics.mean(diffs)) / sd if sd else float('inf'):.2f}")

    out = Path(args.out) if args.out else config.RESULTS / f"repeat_{stem}_{split}.json"
    out.write_text(json.dumps({
        "script": str(script), "args": args.args, "n": args.n,
        "metric": args.metric, "baseline_metric": args.baseline_metric,
        "main_values": mains,
        "baseline_values": [r["baseline"] for r in runs],
        "p_values": [r["p"] for r in runs],
        "summary": {
            "main_mean": statistics.mean(mains),
            "main_stdev": statistics.stdev(mains) if len(mains) > 1 else 0.0,
            "main_min": min(mains), "main_max": max(mains),
        },
    }, ensure_ascii=False, indent=2))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
