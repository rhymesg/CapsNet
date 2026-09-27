# CapsNet

## Overview

TensorFlow capsule network (CapsNet) and convolutional neural network (CNN) research code for short-term traffic speed prediction from spatio-temporal images of road-sensor time series.

This is the authors' implementation accompanying **“A Capsule Network for Traffic Speed Prediction in Complex Road Networks”** (SDF 2018), identified by its [software record](https://doi.org/10.15131/shef.data.7424507.v1). The [canonical repository](https://github.com/rhymesg/CapsNet) is useful for studying traffic forecasting with dynamic routing; see the [algorithm reference](docs/capsule-network.md) and [paper citation](#citation).

For adaptation in another language or neural-network framework, start with the [equations, tensor layouts, and routing procedure](docs/capsule-network.md).

## Method

Encode road-sensor histories as spatio-temporal images, extract local features, and use dynamic routing between capsules to form a traffic-speed forecast. The CNN baseline provides a second architecture for the same input representation.

### Algorithms and source

| Concept | Paper location | Implementation or guide |
|---|---|---|
| Traffic histories as images; flattened future speeds | Eqs. (1)–(2), Section II-A | `get_dataset_image`, `get_batch` in [main_capsnet.py](main_capsnet.py); [data](data/README.md) |
| Convolutions, PrimaryCaps, TrafficCaps | Section II-C, Figure 4, Table II | `CapsNet`, `squash`, `routing`; [algorithm reference](docs/capsule-network.md) |
| CNN baseline | Section II-B, Figure 3, Table I | `CNN_Ma`; [related work](#related-methods) |
| Scaling, optimization, evaluation | Section III, Eqs. (3)–(5) | `load_data`, `train`, `evaluate`, [MinMaxScaler.py](MinMaxScaler.py); [implementation notes](docs/implementation-notes.md) |

## Examples

Clone the repository:

```bash
git clone https://github.com/rhymesg/CapsNet.git
```

Enter its root:

```bash
cd CapsNet
```

The Python source uses TensorFlow 1.x with `tf.contrib.slim`, NumPy, and pandas; plotting also requires Matplotlib.

For the NumPy-only example, install NumPy in your Python environment:

```bash
python -m pip install numpy
```

For neural-network execution, follow the [TensorFlow 1.x environment setup](docs/running.md#environment).

Check scaling and inverse scaling on synthetic speeds without TensorFlow or Santander data:

```bash
python example_synthetic.py --scaler-only
```

With the legacy dependencies installed, exercise the actual sequence batching, CapsNet, and CNN functions on synthetic data:

```bash
python example_synthetic.py
```

Expected checks and their scope are in the [running guide](docs/running.md#synthetic-example). The example checks scaling, tensor shapes, and a single optimization step using synthetic data.

## Implementation scope

The repository contains CapsNet and CNN model functions, sequence batching, scaling, training, and evaluation. The [implementation notes](docs/implementation-notes.md#paper-and-code-differences) describe routing, optimization, and output-metric conventions for adaptation.

### Checks

Use the synthetic commands above to check scaling and, with the legacy environment, network execution. See the [running guide](docs/running.md) for the training workflow and output contract.

## Citation

For academic attribution, please acknowledge this repository when adapting its code or examples.

Please cite the traffic-prediction paper when using its method:

> Youngjoo Kim, Peng Wang, Yifei Zhu, and Lyudmila Mihaylova. “A Capsule Network for Traffic Speed Prediction in Complex Road Networks.” *2018 Sensor Data Fusion: Trends, Solutions, Applications (SDF)*, 2018. [doi:10.1109/SDF.2018.8547068](https://doi.org/10.1109/SDF.2018.8547068). [Full text, arXiv v2](https://arxiv.org/pdf/1807.10603v2); [institutional publication record](https://eprints.whiterose.ac.uk/id/eprint/134023/); [ResearchGate](https://www.researchgate.net/publication/327798102_A_Capsule_Network_for_Traffic_Speed_Prediction_in_Complex_Road_Networks).

[CITATION.cff](CITATION.cff) identifies the software and provides the paper as the preferred citation. The citation request is separate from license conditions.

### Related methods

- Dynamic routing: Sara Sabour, Nicholas Frosst, and Geoffrey E. Hinton. “Dynamic Routing Between Capsules.” *NeurIPS*, 2017. [arXiv:1710.09829v2](https://arxiv.org/abs/1710.09829v2).
- CNN baseline: Xiaolei Ma, Zhuang Dai, Zhengbing He, Jihui Ma, Yong Wang, and Yunpeng Wang. “Learning Traffic as Images: A Deep Convolutional Neural Network for Large-Scale Transportation Network Speed Prediction.” *Sensors*, 17(4), 818, 2017. [doi:10.3390/s17040818](https://doi.org/10.3390/s17040818).

## License and provenance

The software record labels the project **MIT**, but the checkout has no license file. See [provenance and licensing](docs/implementation-notes.md#provenance-and-licensing) for source attribution and reuse details.
