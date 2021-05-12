# Towards hybrid modeling of the global hydrological cycle

## Description

This repository contains the code for a paper to be published as:

*Kraft B., Jung M., Körner M., Koirala S., and Reichstein M. (2021, under review). Towards hybrid modeling of the global hydrological cycle. Hydrology and Earth System Sciences (HESS)*

The datasets used cannot be shared but all datasets are referenced in the paper. The simulated variables are available and linked in the paper.

We share the code for transparency and to demonstrate the concept of hybrid modeling by backpropagation. However, the code is tweaked to our environment and data infrastructure, it cannot be run without adaptions.

* The required python packages can be found in the dockerfile (`/docker/Dockerfile`). Note that the dockerfile cannot be run without adaptions as some information was removed (login tokens etc.).
* All data paths need to be adapted to your data infrastructure.

**We are happy to answer your questions, discuss, and collaborate!**

Contact: bkraft@bgc-jena.mpg.de

## Structure

* Data preprocessing of the individual datasets: `src/datasets/`
* Data is compiled into a sigle `zarr` file using `src/dataset.py`, which also defines the `torch.Dataset`.
* The core model can be found here: `src/models/hybridmodel_loop.py`
* The entire beast is run using `src/tune.py --tune true` for HP tuning and `src/tune.py --tune false` for cross-validation runs (using best config from HP tuning). This is not supposed to work out-of-the-box.

## Citation

```tex
@article{kraft2021towards,
  title={Towards hybrid modeling of the global hydrological cycle},
  author={Kraft, Basil and Jung, Martin and K{\"o}rner, Marco and Koirala, Sujan and Reichstein, Markus},
  journal={Hydrology and Earth System Sciences (HESS)},
  year={2021, under revision}
}
```
