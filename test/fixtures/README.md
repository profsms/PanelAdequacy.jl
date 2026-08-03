# Published-result fixtures

These files are narrow, package-local derivatives of Paper A's archived public application data. `generate_published_fixtures.py` rebuilds them from the Kline-Saggio-Solvsten `LeaveOutTwoWay` extract and records the source SHA-256 and fixed seeds in `provenance.json`.

- `kss_match.csv`: worker, firm, and seed-1 match-level treatment.
- `kss_wage.csv`: the early/late-career analysis design.
- `kss_wage_supports.csv`: the 2,883 design-selected paired-stayer supports.
- `calibrated_dense.csv`: Paper A's seed-42 dense calibration, with the seed-7 row sample.

The fixtures contain only identifiers and design variables needed to regression-lock capture. They contain no additional outcomes or personal attributes.
