"""Regenerate package test fixtures from Paper A's archived KSS extract."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MATCH_SEED = 1
WAGE_SEED = 20260802
NAMES = ("worker", "firm", "year", "log_wage", "x1", "x2",
         "d1", "d2", "d3", "d4")


def dense_panel():
    rng = np.random.default_rng(42)
    rows = []
    firm = 0
    for country, count in enumerate((7, 6, 6)):
        for _ in range(count):
            base = rng.integers(3, 8)
            for year in range(15):
                base = int(np.clip(base + rng.choice((-1, 0, 0, 1)), 0, 9))
                rows.append((firm, f"c{country}y{year}", float(base)))
            firm += 1
    frame = pd.DataFrame(rows, columns=("unit", "time", "x"))
    return frame.sample(n=217, random_state=7).reset_index(drop=True)


def worker_summary(raw):
    ordered = raw.sort_values(["worker", "year"], kind="stable").reset_index(drop=True)
    x1 = ordered.x1.to_numpy(float)
    x2 = ordered.x2.to_numpy(float)
    ratio = np.divide(x2, x1, out=np.zeros_like(x1), where=np.abs(x1) > 1e-14)
    ordered["age"] = np.rint(40 + 40 * ratio).astype(int)
    first = ordered.groupby("worker", sort=False).first().reset_index()
    last = ordered.groupby("worker", sort=False).last().reset_index()
    summary = first[["worker", "firm", "age"]].rename(
        columns={"firm": "firm_1999", "age": "age_1999"})
    summary["firm_2001"] = last.firm.to_numpy()
    summary["stayer"] = summary.firm_1999 == summary.firm_2001
    return ordered, summary


def write_match(raw, out):
    match_id = raw.groupby(["worker", "firm"]).ngroup().to_numpy()
    rng = np.random.default_rng(MATCH_SEED)
    x = rng.standard_normal(match_id.max() + 1)[match_id]
    frame = pd.DataFrame({"unit": raw.worker, "time": raw.firm, "x": x})
    frame.to_csv(out / "kss_match.csv", index=False, float_format="%.17g")


def write_wage(raw, out):
    ordered, summary = worker_summary(raw)
    keep = summary.age_1999.between(20, 29) | summary.age_1999.between(50, 59)
    sample_workers = summary.loc[keep, "worker"]
    summary = summary.loc[keep].reset_index(drop=True)
    summary["late"] = summary.age_1999.between(50, 59)

    panel = ordered[ordered.worker.isin(sample_workers)].copy()
    panel = panel.sort_values(["worker", "year"], kind="stable").reset_index(drop=True)
    late = dict(zip(summary.worker, summary.late))
    panel["x"] = [int(bool(late[w]) and year == 2001)
                  for w, year in zip(panel.worker, panel.year)]
    panel[["worker", "firm", "year", "x"]].to_csv(
        out / "kss_wage.csv", index=False)

    early = np.flatnonzero(summary.stayer.to_numpy(bool) & ~summary.late.to_numpy(bool))
    late_idx = np.flatnonzero(summary.stayer.to_numpy(bool) & summary.late.to_numpy(bool))
    rng = np.random.default_rng(WAGE_SEED)
    chosen_early = rng.choice(early, size=len(late_idx), replace=False)
    chosen_early = rng.permutation(chosen_early)
    chosen_late = rng.permutation(late_idx)
    row = {(int(w), int(year)): i + 1
           for i, (w, year) in enumerate(zip(panel.worker, panel.year))}
    supports = []
    for early_i, late_i in zip(chosen_early, chosen_late):
        early_worker = int(summary.worker.iloc[early_i])
        late_worker = int(summary.worker.iloc[late_i])
        supports.append((row[(early_worker, 1999)], row[(early_worker, 2001)],
                         row[(late_worker, 1999)], row[(late_worker, 2001)]))
    with (out / "kss_wage_supports.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(("early_1999", "early_2001", "late_1999", "late_2001"))
        writer.writerows(supports)

    checksum = int(summary.worker.iloc[chosen_early].sum() % 2147483647)
    if checksum != 1718865421:
        raise AssertionError(f"paired-stayer checksum changed: {checksum}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.source, header=None, names=NAMES)
    write_match(raw, args.output)
    write_wage(raw, args.output)
    dense_panel().to_csv(args.output / "calibrated_dense.csv", index=False)
    metadata = {
        "source": "Kline-Saggio-Solvsten public LeaveOutTwoWay test.csv",
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "match_seed": MATCH_SEED,
        "wage_seed": WAGE_SEED,
        "dense_numpy_seed": 42,
        "dense_pandas_seed": 7,
    }
    (args.output / "provenance.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
