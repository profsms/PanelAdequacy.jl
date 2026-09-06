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
| FE--EIV V-Dem application | `eiv_vdem_panel.csv`, `vdem_gate1.csv` | `vdem` | V-Dem measurement-model output and Maddison Project Database 2020 |
| FE--EIV repeated-report application | `twins.csv` | `twins` | `RbyExample::twins` 0.0.100; Ashenfelter--Krueger design |
| Legacy measurement-error example | `psid_wages_panel.csv` | `psid` | Cornwell--Rupert PSID extract distributed by `plm` |
| TWFE certified design | `castle_panel.csv` | `castle` | Cheng--Hoekstra castle-doctrine replication panel |
| TWFE flagged design | `divorce_panel.csv` | `divorce` | Stevenson--Wolfers data distributed by `bacondecomp` |
| TWFE headline design | `minimum_wage_panel.csv` | `minimum_wage` | Callaway--Sant'Anna public minimum-wage replication panel |

The FE--EIV panel uses the direct posterior-standard-deviation variables from
the V-Dem release at commit `f4dd26922e658442524dfd954bf14f7ebe622d5d`.
The downloaded `vdem.RData` artifact has SHA-256
`39b412d39a061c18f20c98e4ad4d6355b05a0441df31be4ee9aec420dc3d95ea`;
`data/eiv_vdem_SOURCE.txt` records the pinned URL and digest. Credible-interval
half-widths are not used as substitutes for posterior SDs.

Julia keeps the V-Dem gate-1 subset as a separate CSV because it is also a
published-number fixture; R's `vdem` object contains the columns needed to
recreate that subset. Dataset help, the papers, and their replication archives
record variable definitions, transformations, citations, and public-extract
qualifications.

`minimum_wage_panel.csv` is the four-column analysis extract from the exact
2,284-county, 2001--2007 panel used by Callaway and Sant'Anna (2021). The source
replication RDS has SHA-256
`ba4497c4b41fdc447e89c4421bb35a5ab579246cd92064adc4f21f69fd6f61a7`;
this shared Julia/R CSV has SHA-256
`ac5a6e96ed4e9eead62e8c1d6c36c40a96e872b82d143477a31c15b49001a87b`.

`data/twins.csv` is an unmodified export of `RbyExample::twins` version
0.0.100 (GPL >= 2), whose documentation attributes the study to Ashenfelter
and Krueger (1994), *American Economic Review* 84(5), 1157--1173. It contains
183 rows and 16 variables; the article uses the 147 complete observations on
`DLHRWAGE`, `DEDUC1`, `DEDUC2`, `DTEN`, `DMARRIED`, and `DUNCOV`. SHA-256:
`8565aa0a3d1f0b8e99d091f872905bdeb0ec8fc1b4aec46158bd1ebb421ffa08`.
The data retain their upstream terms and are not relicensed under MIT.

The Kline--Saggio--Sølvsten worker--firm test extract is intentionally not
bundled: its public upstream repository does not state a redistribution license.
The Paper A replication archive instead provides a commit-pinned downloader and
verifies the source file's SHA-256 digest
`93e57a413a8cfccdcb043c5d793105a67b2dc9ebd27d5d3a4f1800abf89a2241`.
