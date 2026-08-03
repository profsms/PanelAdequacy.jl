# Cross-language parity

`generate_designs.py` deterministically creates the committed CSV fixtures with seed `20260802`. The Julia and R emitters produce the same record schema:

`n, N, T, d_K, rho, Vn, lambda_n, n_eff, kappa_greedy, kappa_designed, C, effective_C, verdict`.

`compare.py` requires absolute agreement to `1e-12` for floating-point fields and exact agreement for integer fields, verdicts, and canonical packed-support/sign signatures.

From the Julia repository, with `panelcert` installed:

```powershell
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. parity/julia_emit.jl parity/julia.json parity/julia_packing.json
Rscript parity/r_emit.R parity/r.json parity/r_packing.json
python parity/compare.py parity/julia.json parity/r.json parity/julia_packing.json parity/r_packing.json
```
