# Implementation and provenance

Source reference for the traffic CapsNet and CNN in [main_capsnet.py](../main_capsnet.py). For research attribution, use the [paper citation](../README.md#citation).

## Provenance and licensing

- The [University of Sheffield software record](https://doi.org/10.15131/shef.data.7424507.v1) names the four paper authors and links to this repository.
- The record labels the project MIT. Obtain the applicable license and copyright text from the rights holder before redistribution; the checkout supplies no license file.
- `routing` credits Huadong Liao; the corresponding [upstream capsLayer.py](https://github.com/naturomics/CapsNet-Tensorflow/blob/master/capsLayer.py) is distributed under Apache-2.0. Preserve applicable upstream notices when reusing that implementation.
- The paper attributes traffic-data collection to SETA; consult the [dataset guide](../data/README.md) for supplied files and provenance. Software and dataset permissions are separate.

## Paper and code differences

The [paper](https://arxiv.org/pdf/1807.10603v2) gives the research architecture. These source conventions matter when adapting it:

| Component | Source behavior |
|---|---|
| Routing | Both branches use the unsquashed `s_J` after computing `squash(s_J)`; the paper describes squashed sums. |
| Optimization | Adam minimizes normalized RMSE; the paper states MSE. Slim registers L2 terms separately from the minimized loss. |
| Scaling | One global minimum and maximum are fitted before the chronological split, including evaluation values. |
| Error units | `scale_error_inverse` multiplies normalized RMSE by the speed range divided by the feature range. Speed predictions use `scale_inverse`, which also restores the offset. |
| Summary | “Average Loss” is the mean of batch RMSE values; pooled RMSE requires pooling squared errors before taking the root. |
| Windows | The loop processes a subset of constructed windows; exact counts are in the [running guide](running.md#training-and-evaluation). |
| Data | Bundled files contain 33,504 rows. Defaults select 5,000 rows and a 75% training split; paper experiments use their own configurations. |

## Checks

The [synthetic example](../example_synthetic.py) checks scaling and provides a TensorFlow network exercise. Run the error-unit regression checks with `python -m unittest discover -s tests -v`; the fixtures use known speed differences and zero error.
