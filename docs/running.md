# Running CapsNet

This guide describes [main_capsnet.py](../main_capsnet.py) and the synthetic example. Read the [limitations](limitations.md) before interpreting evaluation output.

## Environment

The historical README records Python 3.5 but no dependency versions. The code requires TensorFlow 1.x APIs, including `tf.contrib.slim`, sessions, placeholders, and `tf.app.run`; [TensorFlow's migration guide](https://www.tensorflow.org/guide/migrate) explains the removed APIs.

The following is a **candidate compatibility environment**, not a recovered or tested experiment environment. Use an existing CPython 3.7 installation on a platform with a [TensorFlow 1.15 wheel](https://pypi.org/project/tensorflow/1.15.0/#files), such as Linux x86-64; this is not a native Apple Silicon setup.

Create an isolated environment:

```bash
python3.7 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install candidate versions compatible with that Python generation:

```bash
python -m pip install 'tensorflow==1.15.0' 'numpy==1.18.5' 'pandas==1.1.5' 'matplotlib==3.3.4' 'protobuf==3.20.3'
```

These direct pins do not lock transitive dependencies. pandas must retain `DataFrame.append`, which was [removed in pandas 2.0](https://pandas.pydata.org/docs/whatsnew/v2.0.0.html#removal-of-prior-version-deprecations-changes); installing `tf-slim` alone does not replace the script's `tensorflow.contrib.slim` import.

## Synthetic example

Run only scaling/inverse scaling with NumPy:

```bash
python example_synthetic.py --scaler-only
```

Expected final message: `Scaler check passed: range [0, 1], inverse round trip.` It exercises [MinMaxScaler.py](../MinMaxScaler.py) on nonconstant synthetic speeds, not the neural network or dataset loader.

Run the full example in the legacy environment:

```bash
python example_synthetic.py
```

It uses one batch, ten history steps, one forecast step, and eight sensors; the CNN needs at least eight rows and columns for its three VALID 2×2 pooling layers. Expected final messages are `CapsNet check passed: predictions (1, 8), finite loss.` and `CNN_Ma check passed: predictions (1, 8), finite loss.`, with intermediate activation prints.

Both graphs run on CPU, use fixed example seeds, and perform one Adam update. Numeric prediction values are intentionally unspecified; this is an execution check, not evidence of convergence, reproducible trained weights, or paper accuracy.

## Training and evaluation

From the repository root, create the checkpoint directory:

```bash
mkdir -p log
```

After reviewing the settings and resource requirements below, start training and evaluation:

```bash
python main_capsnet.py
```

`main` always calls `load_data`, `train`, and `evaluate` in that order; evaluation restores the checkpoint just written by training.

| Setting | Meaning |
|---|---|
| `whichNet` | `0` selects `CNN_Ma`; other values select CapsNet |
| `case`, `data_path`, `eval_dir` | Dataset selection and paths; change `case` before dependent paths are assigned |
| `time_step_in`, `time_step_out` | Historical and future steps; one step denotes 15 minutes in the source dataset |
| `num_sensor_set`, `num_data_set` | Leading columns/rows to use; `-1` selects all |
| `num_data_train_set` | Training row count; `-1` selects `ceil(0.75*num_data)` |
| `batch_size`, `shuffle_batch` | Fixed graph batch size and window shuffling |
| `CAP_*`, `CNN_*`, `repeatConv` | Model, routing, initialization, and optimizer settings |
| `checkpoint_dir`, `checkpoint_path` | Shared model output location; use distinct locations for experiments |
| `drawPlot`, `printEvery` | Interactive plots and logging cadence |

The default run uses the first 5,000 rows and all 50 sensors from case 1, split into 3,750 training and 1,250 evaluation rows. Its routing weight tensor alone contains 51.2 million float32 entries (about 195 MiB); optimizer state and tiled activations require additional memory.

The loop calls each update an “epoch”, but processes a new batch each iteration rather than repeating passes over the dataset. With defaults, it performs 3,738 training updates and 1,238 evaluation batches, omitting two available windows from each split.

Outputs use headerless CSVs in `eval/<case>/`:

- `result_prediction_CAP.csv` or `result_prediction_CNN.csv`: inverse-scaled predictions, one row per sample, `L*N` columns in time-major order.
- `result_error_CAP.csv` or `result_error_CNN.csv`: truth minus prediction in the same layout.
- `result_RMSE_CAP.csv` or `result_RMSE_CNN.csv`: one transformed batch loss per row; **these values are not correctly inverse-scaled RMSE**, as explained in [limitations](limitations.md#paper-and-code-differences).

Training writes `log/checkpoint.ckpt*` and TensorFlow checkpoint metadata. Repeated runs and switching models reuse these paths unless edited.

## Input constraints and failures

- Each split needs more than `time_step_in + time_step_out + batch_size` rows for positive loop counts; small splits can produce allocation, undefined-index, or division errors.
- Use finite numeric data with nonzero global range; constant data divides by zero in the scaler.
- Keep `num_channel=1`, dimensions consistent with globals, and convolution channels divisible by capsule vector length.
- The script does not validate these contracts comprehensively; file errors, shape errors, and out-of-memory errors surface from dependencies.
