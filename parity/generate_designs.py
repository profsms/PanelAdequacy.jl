"""Regenerate the committed cross-language parity designs."""

import csv
import math
import random
from pathlib import Path


SEED = 20260802
OUT = Path(__file__).resolve().parent / "designs"
FIELDS = ("unit", "time", "x", "y")


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", newline="", encoding="ascii") as handle:
        fields = FIELDS + (("control",) if rows and "control" in rows[0] else ())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dense(rng):
    rows = []
    for unit in range(1, 13):
        unit_effect = 0.07 * unit
        for time in range(1, 10):
            time_effect = -0.04 * time
            treatment = (
                0.6 * (unit % 3 == 0)
                + 0.35 * (time >= 6)
                + 0.12 * math.sin(unit * time)
                + rng.uniform(-0.08, 0.08)
            )
            outcome = (
                unit_effect
                + time_effect
                + 0.45 * treatment
                + rng.gauss(0.0, 0.18 + 0.01 * (unit % 4))
            )
            rows.append(
                {
                    "unit": f"u{unit:02d}",
                    "time": f"t{time:02d}",
                    "x": f"{treatment:.15f}",
                    "y": f"{outcome:.15f}",
                }
            )
    return rows


def sparse(rng):
    rows = []
    seen = set()
    for unit in range(1, 25):
        # Overlapping deterministic neighborhoods create a connected mobility
        # graph with many cycles while remaining substantially incomplete.
        times = {
            1 + (unit - 1) % 13,
            1 + unit % 13,
            1 + (2 * unit + 3) % 13,
            1 + (5 * unit + 1) % 13,
        }
        if unit % 4 == 0:
            times.add(1 + (7 * unit + 2) % 13)
        for time in sorted(times):
            key = (unit, time)
            if key in seen:
                continue
            seen.add(key)
            treatment = (
                0.3 * (unit % 2)
                + 0.22 * (time % 3)
                + 0.09 * math.cos(unit + 2 * time)
                + rng.uniform(-0.12, 0.12)
            )
            outcome = (
                0.03 * unit
                - 0.025 * time
                + 0.55 * treatment
                + rng.gauss(0.0, 0.22)
            )
            rows.append(
                {
                    "unit": f"w{unit:02d}",
                    "time": f"f{time:02d}",
                    "x": f"{treatment:.15f}",
                    "y": f"{outcome:.15f}",
                }
            )
    return rows


def controlled(rng):
    rows = []
    for unit in range(1, 7):
        for time in range(1, 7):
            control = 0.18 * unit - 0.11 * time + 0.09 * math.cos(unit * time)
            treatment = (
                0.7 * control
                + 0.16 * math.sin(unit + 2 * time)
                + (1.25 if (unit, time) == (6, 6) else 0.0)
                + rng.uniform(-0.05, 0.05)
            )
            outcome = (
                0.04 * unit - 0.03 * time + 0.5 * treatment
                + 0.3 * control + rng.gauss(0.0, 0.15)
            )
            rows.append({
                "unit": f"c{unit:02d}", "time": f"t{time:02d}",
                "x": f"{treatment:.15f}", "y": f"{outcome:.15f}",
                "control": f"{control:.15f}",
            })
    return rows


def main():
    rng = random.Random(SEED)
    dense_rows = dense(rng)
    sparse_rows = sparse(rng)
    controlled_rows = controlled(rng)
    write("dense.csv", dense_rows)
    write("sparse.csv", sparse_rows)
    write("controlled.csv", controlled_rows)

    permuted = list(dense_rows)
    random.Random(SEED + 1).shuffle(permuted)
    write("dense_permuted.csv", permuted)


if __name__ == "__main__":
    main()
