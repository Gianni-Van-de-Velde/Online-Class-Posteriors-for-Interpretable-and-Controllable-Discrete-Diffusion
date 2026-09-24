"""
Print aggregate metrics for all runs that have a predictions.jsonl.

Usage:
    python stats.py                     # all datapoints, ./runs
    python stats.py --first 30          # first 30 by id
    python stats.py --runs_dir ./runs   # explicit dir
    python stats.py --mean              # report mean PPL instead of median
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "plots"))
from _breakdown_common import classify, LAYER_SPECS, RED_OUTCOMES


def load_records(pred_file, first=None):
    try:
        records = [json.loads(l) for l in Path(pred_file).read_text().splitlines() if l.strip()]
    except json.JSONDecodeError:
        return []
    if first is not None:
        records.sort(key=lambda r: r.get("id", 0))
        records = records[:first]
    return records


def compute_metrics(records, use_mean_ppl=False, ppl_threshold=300.0):
    n        = len(records)
    on_topic = np.mean([r["on_topic"] for r in records])
    violated = np.mean([r["violated"] for r in records])
    sr1      = np.mean([r["seq-rep-1"] for r in records])
    cb       = np.mean([r.get("copying_bleu_score", r.get("copying_bleu", 0)) for r in records])
    exact_cp = np.mean([r["exact_copy"] for r in records])
    ppls     = [r["ppl"] for r in records if r.get("ppl", 0) > 0]
    if ppls:
        ppl = float(np.mean(ppls) if use_mean_ppl else np.median(ppls))
    else:
        ppl = 0.0
    fmt_vals = [r["fmt_adh"] for r in records if "fmt_adh" in r]
    fmt_adh  = float(np.mean(fmt_vals)) if fmt_vals else float("nan")
    len_vals = [r["gen_length"] for r in records if "gen_length" in r]
    len_p50  = float(np.median(len_vals)) if len_vals else float("nan")
    me_vals  = [r["mask_entropy"] for r in records if r.get("mask_entropy") is not None]
    mask_entropy = float(np.mean(me_vals)) if me_vals else float("nan")

    # Cascade breakdown: mutually exclusive categories summing to 1
    keys   = [key for key, *_ in LAYER_SPECS]
    counts = {key: 0 for key in keys}
    for r in records:
        counts[classify(r, ppl_threshold)] += 1
    breakdown = {key: counts[key] / n for key in keys}
    # red/orange/green aggregate the same breakdown -- see classify() in _breakdown_common.py
    red    = sum(breakdown[k] for k in RED_OUTCOMES)
    orange = breakdown["on_violated"]
    green  = breakdown["ic"] + breakdown["off_safe_llm"]

    result = dict(n=n, on_topic=on_topic, violated=violated,
                  red=red, orange=orange, green=green,
                  sr1=sr1, cb=cb, exact_cp=exact_cp,
                  ppl=ppl, fmt_adh=fmt_adh, len_p50=len_p50, mask_entropy=mask_entropy)
    result.update(breakdown)
    return result


def find_runs(runs_dir):
    runs_dir = Path(runs_dir)
    runs = []
    for pred_file in sorted(runs_dir.rglob("predictions.jsonl")):
        run_name = pred_file.parts[len(runs_dir.parts)]
        runs.append((run_name, pred_file))
    return runs


def print_table(rows, ppl_label):
    header = f"{'run':<48}  {'n':>3}  {'on_topic':>8}  {'violated':>8}  {'red':>6}  {'orange':>6}  {'green':>6}  {'IC':>6}  {'sr1':>6}  {'cb':>6}  {'exact_cp':>8}  {ppl_label:>7}  {'fmt_adh':>7}  {'len(p50)':>8}  {'collapse':>8}  {'sh_degen':>8}  {'mask_ent':>8}"
    sep    = "-" * len(header)
    print(header)
    print(sep)
    for run_name, m in rows:
        fmt_str = f"{m['fmt_adh']:>7.3f}" if not np.isnan(m['fmt_adh']) else f"{'n/a':>7}"
        len_str = f"{m['len_p50']:>8.1f}" if not np.isnan(m['len_p50']) else f"{'n/a':>8}"
        me_str  = f"{m['mask_entropy']:>8.3f}" if not np.isnan(m['mask_entropy']) else f"{'n/a':>8}"
        print(
            f"{run_name:<48}  {m['n']:>3}  {m['on_topic']:>8.3f}  {m['violated']:>8.3f}"
            f"  {m['red']:>6.3f}  {m['orange']:>6.3f}  {m['green']:>6.3f}"
            f"  {m['ic']:>6.3f}  {m['sr1']:>6.3f}  {m['cb']:>6.3f}  {m['exact_cp']:>8.3f}"
            f"  {m['ppl']:>7.2f}  {fmt_str}  {len_str}  {m['collapse']:>8.3f}  {m['sh_degen']:>8.3f}"
            f"  {me_str}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="./runs")
    parser.add_argument("--first", type=int, default=None, metavar="N",
                        help="Only use the first N datapoints (by id)")
    parser.add_argument("--mean", action="store_true",
                        help="Report mean PPL instead of median")
    args = parser.parse_args()

    runs = find_runs(args.runs_dir)
    if not runs:
        print(f"No predictions.jsonl found under {args.runs_dir}")
        return

    rows = []
    for run_name, pred_file in runs:
        records = load_records(pred_file, first=args.first)
        if not records:
            continue
        m = compute_metrics(records, use_mean_ppl=args.mean)
        rows.append((run_name, m))

    n_label = f"first {args.first}" if args.first else "all"
    ppl_label = "ppl(mean)" if args.mean else "ppl(med)"
    print(f"runs_dir={args.runs_dir}  datapoints={n_label}\n")
    print_table(rows, ppl_label)


if __name__ == "__main__":
    main()
