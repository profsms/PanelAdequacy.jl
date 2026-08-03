# Bundled Data Sources

The package bundles the small analysis panels used by the papers so examples and
published-number regression tests run offline. These files are data, not part of
the package's MIT-licensed source code; their original terms and citations apply.

## Grunfeld

`data/grunfeld_panel.csv` is the canonical 11-firm, 1935--1954 investment panel
distributed by statsmodels, whose dataset metadata declares it public domain.
It was retrieved from statsmodels commit
`57169f2cfc7089141513ed6103f79fa56ab213ac` on 2026-08-02. SHA-256:
`6f6ca138e645eeee6ff3e54fe5b9b498f7ddb5c484237d2a8489c524b3c94098`.

The source identifies the data as the original 11-firm sample from Grunfeld's
thesis, reconstructed by Achim Zeileis and Christian Kleiber. Cite Grunfeld
(1958) and Kleiber and Zeileis (2010), *The Grunfeld Data at 50*. That paper
documents the 10-firm and erroneous five-firm versions in circulation; the
package deliberately uses the complete corrected 11-firm version.

## Other Panels

The remaining files are the exact analysis extracts used by the papers:

| Use | Julia file(s) | R object | Source |
|---|---|---|---|
| Paper A diffuse check | `f_score_panel.csv` | `fscore` | Author-assembled Warsaw, Budapest, and Prague exchange filings, 2010--2024 |
| Paper B V-Dem application | `eiv_vdem_panel.csv`, `vdem_gate1.csv` | `vdem` | V-Dem measurement-model output and Maddison Project Database 2020 |
| Paper B mechanism illustration | `psid_wages_panel.csv` | `psid` | Cornwell--Rupert PSID extract distributed by `plm` |
| Paper C certified design | `castle_panel.csv` | `castle` | Cheng--Hoekstra castle-doctrine replication panel |
| Paper C flagged design | `divorce_panel.csv` | `divorce` | Stevenson--Wolfers data distributed by `bacondecomp` |

Julia keeps the V-Dem gate-1 subset as a separate CSV because it is also a
published-number fixture; R's `vdem` object contains the columns needed to
recreate that subset. Dataset help, the papers, and their replication archives
record variable definitions, transformations, citations, and public-extract
qualifications.

The Kline--Saggio--Sølvsten worker--firm test extract is intentionally not
bundled: its public upstream repository does not state a redistribution license.
The Paper A replication archive instead provides a commit-pinned downloader and
verifies the source file's SHA-256 digest
`93e57a413a8cfccdcb043c5d793105a67b2dc9ebd27d5d3a4f1800abf89a2241`.
