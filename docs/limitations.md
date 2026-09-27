# Provenance and limitations

This reference describes the implementation at [b1967d2e48b68ef031732f120342203c84314185](https://github.com/rhymesg/CapsNet/tree/b1967d2e48b68ef031732f120342203c84314185), with documentation and example additions. The algorithm, training behavior, and datasets are preserved; these additions are not a new validated release.

## Provenance and licensing

- The [University of Sheffield software record](https://doi.org/10.15131/shef.data.7424507.v1), also accessible through its [public metadata](https://api.figshare.com/v2/articles/7424507), names the four paper authors and links directly to this repository; this supports identifying it as author code.
- That record is labeled MIT, but stores only a GitHub link rather than a source snapshot. The checkout has no `LICENSE` file or associated copyright notice; the owner should supply the applicable text and confirm its scope before redistribution.
- `routing` explicitly credits Huadong Liao. [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow) supplies the corresponding [capsLayer.py](https://github.com/naturomics/CapsNet-Tensorflow/blob/master/capsLayer.py) under Apache-2.0; the exact borrowed revision and required retained notices are not recorded here.
- The traffic architecture adapts the dynamic-routing method and CNN baseline cited in the [README](../README.md#related-methods); it is not an official implementation by those earlier papers' authors.
- Dataset collection is attributed to the SETA project in the traffic paper. A separate dataset license, sensor mapping, raw source, and preprocessing script are absent; the software record's label alone does not resolve those gaps.
- Citation requests do not replace license conditions. No new license or third-party copyright attribution has been invented in this update.

## Paper and code differences

The [traffic paper](https://arxiv.org/pdf/1807.10603v2) describes the architecture and experiments; the table below records what [main_capsnet.py](../main_capsnet.py) actually does.

| Topic | Evidence and implication |
|---|---|
| Routing squash | Section II-C describes squashed weighted sums; `routing` overwrites `squash(s_J)` with `s_J` in both branches. TrafficCaps lengths are unbounded. |
| Loss | Section III states MSE; `train` minimizes its square root (RMSE), changing gradients. |
| Regularization | Slim registers L2 penalties, but the minimized `total_loss` does not add the regularization collection. |
| Scaling leakage | `load_data` fits one global minimum/maximum before the chronological split, including evaluation values. |
| Reported RMSE | `scale_inverse` is applied to normalized RMSE as if it were a speed, adding the data minimum. Correct unit conversion for errors uses only the range factor; both supplied files have a minimum of 1. |
| Aggregate error | “Average Loss” averages transformed per-batch RMSE values; it is not the pooled RMSE defined in Eq. (5). No direct MAE/MRE calculation is supplied. |
| Window coverage | Image construction includes every valid window, but train/evaluation loop bounds omit final windows; see [running](running.md#training-and-evaluation). |
| Data and split | Paper Section III reports 33,054 rows and January–September training; bundled files have 33,504 rows without timestamps. The default run uses only 5,000 rows and a 75% row split. |
| Validation | `assert num_data, num_sensor == dataset.shape` asserts only the truthiness of `num_data`; it does not validate the matrix shape. |

These observations do not establish why the implementation differs from the paper. Changing routing, splitting, or metrics requires an explicit research decision and a new evaluation.

## Reproduction boundaries

- The paper's tasks use `(M,N,L) = (10,20,1), (10,20,2), (14,50,1), (14,50,2)` in both cases; none is the unchanged default configuration.
- The paper reports an average 13.1% RMSE improvement over its CNN baseline. This is a published result, not a measurement reproduced from this checkout.
- No trained checkpoints, original dependency lock, experiment manifest, or paper-result verification harness is supplied.
- TensorFlow weight initialization has no fixed seed in the training script. Optional shuffling uses NumPy seed `1234567890` and also shuffles evaluation windows.
- Temporal continuity and exact calendar splits cannot be checked without timestamps; road-segment IDs and preprocessing provenance are also missing from the CSVs.
- The dense routing tensor's cost grows quadratically with sensor count. The repository does not establish production readiness or generalization to other traffic networks.

## Verification status

Checked on 2026-09-27 on macOS arm64, Python 3.12.14, NumPy 2.3.5, and pandas 2.2.3:

- Both bundled CSVs parse as finite numeric matrices with the [recorded shapes and hashes](../data/README.md).
- `python example_synthetic.py --scaler-only` passes the scaler range and inverse round-trip checks.
- Original `load_data`, `get_dataset_image`, and `get_batch` function definitions, executed in isolation without TensorFlow, pass checks of default split sizes, window counts, shapes, ordering, and inverse scaling for both CSVs.
- All Python files parse with Python 3.5 syntax; original executable ASTs are unchanged, local Markdown links resolve, and `CITATION.cff` validates against the official CFF 1.2.0 schema.
- The full synthetic command stops at `ModuleNotFoundError: No module named 'tensorflow'`; graph execution, training, checkpoint restoration, and evaluation are **unverified**. The candidate Python 3.7 environment is also unverified.

The modern NumPy/pandas versions above were used for data inspection only; they do not establish compatibility of the TensorFlow training script.

## Evidence still needed

The independent assessment identified three remaining requirements for broader reuse:

1. Supply the owner-approved license text and copyright notice, trace the credited routing implementation to its borrowed revision, retain applicable upstream notices, and establish dataset terms.
2. Run the full synthetic command in the proposed legacy environment, record resolved dependencies and output, then verify training, checkpoint restoration, and evaluation separately; a passing smoke check would still not reproduce the paper.
3. Apply the [suggested repository metadata](repository-metadata.md) and repeat the public retrieval check after publication; search visibility remains unverified for the updated documentation.
