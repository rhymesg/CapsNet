# Santander traffic speed data

This guide describes the two preprocessed CSV inputs to [CapsNet](../README.md). The [traffic paper](../README.md#citation), Section III and Figure 6, attributes the measurements to induction loop detectors in Santander, Spain, during 2016 through the SETA project.

## Files and contract

| Input | Rows × sensor columns | Observed range |
|---|---|---|
| [Case 1](Santander_ST_speed_pp_case1.csv) | 33,504 × 50 | 1–257 |
| [Case 2](Santander_ST_speed_pp_case2.csv) | 33,504 × 50 | 1–328 |

- Files have no header, index, timestamps, or sensor IDs; entries are finite numeric speeds interpreted as km/h from the paper.
- Rows follow the supplied temporal order; the nominal measurement interval is 15 minutes. Removed days mean continuous calendar spacing cannot be inferred from row index alone.
- Columns preserve the supplied road-segment order. The loader selects leading columns when `num_sensor_set` is positive; the paper depicts two different sets of road segments, but the CSVs do not independently identify them.
- Preserve column order between training and evaluation. No adjacency matrix is used.
- The observed maxima are reported without physical-range validation; the code does not clip outliers or apply a missing-value mask.

The paper describes removing days with failed sensors and replacing isolated missing measurements by same-time averages from other days. The repository supplies only the processed arrays; that procedure cannot be reconstructed from a checked-in script.

## Loader behavior

`load_data` in [main_capsnet.py](../main_capsnet.py) reads the selected file, selects leading rows/columns, and uses [MinMaxScaler.py](../MinMaxScaler.py) to fit a single global range across all selected entries. It then splits rows into training and evaluation portions and creates windows within each split.

For the `[0, 1]` range, scaling is `(speed - minimum)/(maximum - minimum)` and inverse scaling is `scaled*(maximum - minimum) + minimum`. Constant or nonfinite inputs are unsupported; the scaler does not reject them explicitly.

The default 75% split is by row count, not calendar month. Fitting the scaler before splitting includes evaluation data; see [implementation details](../docs/implementation-notes.md#paper-and-code-differences).

## Provenance and checksums

These are the files at source revision `b1967d2e48b68ef031732f120342203c84314185`, inspected on 2026-09-27. The paper reports 33,054 rows, which differs from the bundled files; the reason is not established.

| File | SHA-256 |
|---|---|
| `Santander_ST_speed_pp_case1.csv` | `eab670674403aba8e7a255c6af111866c278e90f4162650893e162c1dc5a6412` |
| `Santander_ST_speed_pp_case2.csv` | `34c1624d3acbba965afe25d5161ecb716dc17e9ab5d4fade3bab4bc32e4214d9` |

The [software DOI](https://doi.org/10.15131/shef.data.7424507.v1) links back to GitHub; it is not an independent raw-data archive. Dataset redistribution terms and the upstream data retrieval path remain unresolved; see [license and provenance](../docs/implementation-notes.md#provenance-and-licensing).
