"""Stepped-wedge parity: compare R, Julia and the Python reference (absolute tolerance 1e-10).

    python parity/sw_compare.py REFERENCE.csv R.csv JULIA.csv
"""
import csv
import sys


def load(path, key=("case", "quantity")):
    with open(path, newline="") as f:
        return {(r["case"], r["quantity"]): float(r["value"]) for r in csv.DictReader(f)}


ref = load(sys.argv[1])
rr = load(sys.argv[2])
jl = load(sys.argv[3])
keys = sorted(set(rr) & set(jl))
assert keys and set(rr) == set(jl), "R and Julia emitted different case sets"
worst = {"R-Julia": 0.0, "R-Python": 0.0, "Julia-Python": 0.0}
for k in keys:
    worst["R-Julia"] = max(worst["R-Julia"], abs(rr[k] - jl[k]))
    worst["R-Python"] = max(worst["R-Python"], abs(rr[k] - ref[k]))
    worst["Julia-Python"] = max(worst["Julia-Python"], abs(jl[k] - ref[k]))
print(f"{len(keys)} values compared")
for name, w in worst.items():
    print(f"  max |{name}| = {w:.3e}")
ok = all(w < 1e-10 for w in worst.values())
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
