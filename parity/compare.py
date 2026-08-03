"""Compare PanelAdequacy.jl and panelcert parity outputs."""

import json
import math
import sys
from pathlib import Path


FIELDS = (
    "n", "N", "T", "d_K", "rho", "Vn", "lambda_n", "n_eff",
    "kappa_greedy", "kappa_designed", "C", "effective_C", "verdict",
)
FLOATS = {"rho", "Vn", "lambda_n", "n_eff", "kappa_greedy", "kappa_designed"}
INTEGERS = {"n", "N", "T", "d_K", "C", "effective_C"}


def load(path):
    with Path(path).open(encoding="ascii") as handle:
        return json.load(handle)


def main(argv):
    if len(argv) not in (3, 5):
        raise SystemExit(
            "usage: compare.py JULIA.json R.json [JULIA_PACKING.json R_PACKING.json]"
        )
    julia, r = load(argv[1]), load(argv[2])
    if set(julia) != set(r):
        raise AssertionError(f"design sets differ: {set(julia) ^ set(r)}")
    for design in sorted(julia):
        left, right = julia[design], r[design]
        if tuple(left) != FIELDS or tuple(right) != FIELDS:
            raise AssertionError(f"{design}: output schema/order differs from {FIELDS}")
        for field in FIELDS:
            a, b = left[field], right[field]
            if field in FLOATS:
                if not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12):
                    raise AssertionError(f"{design}.{field}: Julia={a!r}, R={b!r}")
            elif field in INTEGERS:
                if type(a) is not int or type(b) is not int or a != b:
                    raise AssertionError(f"{design}.{field}: Julia={a!r}, R={b!r}")
            elif a != b:
                raise AssertionError(f"{design}.{field}: Julia={a!r}, R={b!r}")
    if len(argv) == 5:
        julia_packing, r_packing = load(argv[3]), load(argv[4])
        if julia_packing != r_packing:
            raise AssertionError("packed supports/signs differ between languages")
    print(f"parity passed for {len(julia)} designs at absolute tolerance 1e-12")


if __name__ == "__main__":
    main(sys.argv)
