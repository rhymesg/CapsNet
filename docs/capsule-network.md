# Capsule network reference

This reference maps traffic-speed forecasting concepts to [rhymesg/CapsNet](https://github.com/rhymesg/CapsNet). Read the [paper and related-method citations](../README.md#citation) and [current-code limitations](limitations.md) together with [main_capsnet.py](../main_capsnet.py).

## Input and output

- Let `B`, `M`, `N`, and `L` denote batch size, history length, sensor count, and prediction horizon.
- `get_dataset_image` creates overlapping windows independently within training and evaluation splits; each window has shape `[M+L, N, 1]`.
- `get_batch` returns inputs `[B, M, N, 1]` and targets `[B, L*N]`, flattened in time-major, sensor-minor order; these implement traffic-paper Eqs. (1)–(2).
- Speeds use one channel and global min–max scaling; see the [data contract](../data/README.md).
- Sensor-column order is fixed. There is no adjacency matrix or road graph; neighboring columns need not represent neighboring roads.
- The models require statically known dimensions consistent with module globals; they are research functions rather than a configurable public model API.

## Architecture and equations

| Operation | Source | Publication mapping |
|---|---|---|
| Two 3×3 convolutions with ReLU; optional second convolution controlled by `repeatConv` | `CapsNet` | Traffic paper Section II-C, Table II |
| PrimaryCaps convolution, reshape to 8-component vectors, squash | `CapsNet`, `squash` | Traffic paper Figure 4; Sabour Eq. (1) |
| Learned transforms, output-capsule softmax, agreement updates | `routing` | Sabour Eqs. (2)–(3), Procedure 1, with departures below |
| Length of each 16-component TrafficCaps vector, then flatten | `CapsNet` | Traffic paper Section II-C |
| Three convolution/pooling pairs, flatten, linear dense output | `CNN_Ma` | Traffic paper Section II-B, Table I, based on Ma et al. |

The implementation's numerically stabilized squash is

$$
\operatorname{squash}(s)=\frac{\|s\|^2}{1+\|s\|^2}\frac{s}{\sqrt{\|s\|^2+\epsilon}},\qquad \epsilon=10^{-9}.
$$

For PrimaryCaps vector $u_i$, routing forms $\hat u_{j|i}=W_{ij}u_i$ and starts with zero logits $b_{ij}$. Each iteration computes

$$
c_{ij}=\operatorname{softmax}_{j}(b_{ij}),\qquad
s_j=\sum_i c_{ij}\hat u_{j|i}+\beta_j.
$$

Here $\beta_j$ is a learned bias added by this implementation. Before the final iteration, logits are updated by $b_{ij}\leftarrow b_{ij}+\hat u_{j|i}^{T}v_j$.

**Current code sets $v_j=s_j$:** both routing branches compute `squash(s_J)` and immediately overwrite it with `s_J`. PrimaryCaps remains squashed, but TrafficCaps lengths are not bounded by one; this differs from the squashed routing described in the traffic paper and Sabour's Procedure 1.

The output is $\sqrt{\|v_j\|^2+\epsilon}$ for each horizon/sensor pair. It is nonnegative, but not a calibrated probability or a guaranteed normalized speed in `[0, 1]`.

## Implementation choices

- `CAP_caps1_num_filter` is the number of scalar convolution channels, grouped into vectors; it is not the number of vector-capsule channels.
- With SAME padding and stride one, the primary capsule count is `M*N*CAP_caps1_num_filter/CAP_caps1_vec_len` and the output capsule count is `L*N`.
- `routing` returns `[B, 1, L*N, CAP_caps2_vec_len, 1]`; `CapsNet` removes the singleton axis before computing lengths.
- `tf.stop_gradient` blocks gradient flow through the intermediate routing predictions; the final weighted sum uses the live predictions.
- Routing tiles inputs for every output capsule component. Weight count is `M*L*N*N*CAP_caps1_num_filter*CAP_caps2_vec_len`; memory grows quadratically with sensor count, before gradients, optimizer state, and activations.
- `train` minimizes normalized RMSE with Adam and a per-step exponential learning-rate schedule; [limitations](limitations.md#paper-and-code-differences) explain the loss and metric departures.

## Worked execution path

[example_synthetic.py](../example_synthetic.py) scales a synthetic speed matrix, passes synthetic sequences through the original window/batch functions, then calls each model in its own graph. It checks output shape and finite loss after one optimizer step; see [commands and verification scope](running.md#synthetic-example).
