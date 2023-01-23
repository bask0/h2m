# Towards hybrid modeling of the global hydrological cycle

## Description

This repository contains the code for a paper to be published as:

*[Kraft B., Jung M., Körner M., Koirala S., and Reichstein M. (2022). Towards hybrid modeling of the global hydrological cycle. Hydrology and Earth System Sciences (HESS)](https://doi.org/10.5194/hess-26-1579-2022)*

The datasets used cannot be shared but all datasets are referenced in the paper. The simulated variables are available and linked in the paper.

We share the code for transparency and to demonstrate the concept of hybrid modeling by backpropagation. However, the code is tweaked to our environment and data infrastructure, it cannot be run without adaptions.

* The required python packages can be found in the dockerfile (`/docker/Dockerfile`). Note that the dockerfile cannot be run without adaptions as some information was removed (login tokens etc.).
* All data paths need to be adapted to your data infrastructure.

**We are happy to answer your questions, discuss, and collaborate!**

## Structure

* Data preprocessing of the individual datasets: `src/datasets/`
* Data is compiled into a sigle `zarr` file using `src/dataset.py`, which also defines the `torch.Dataset`.
* The core model can be found here: `src/models/hybridmodel_loop.py`
* The entire beast is run using `src/tune.py --tune true` for HP tuning and `src/tune.py --tune false` for cross-validation runs (using best config from HP tuning). This is not supposed to work out-of-the-box.

## Citation

```tex
@Article{hess-26-1579-2022,
  AUTHOR = {Kraft, B. and Jung, M. and K\"orner, M. and Koirala, S. and Reichstein, M.},
  TITLE = {Towards hybrid modeling of the global hydrological cycle},
  JOURNAL = {Hydrology and Earth System Sciences},
  VOLUME = {26},
  YEAR = {2022},
  NUMBER = {6},
  PAGES = {1579--1614},
  URL = {https://hess.copernicus.org/articles/26/1579/2022/},
  DOI = {10.5194/hess-26-1579-2022}
}
```
