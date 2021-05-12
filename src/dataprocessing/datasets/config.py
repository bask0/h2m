"""
Config for the individual datasets.
"""

import numpy as np
from typing import Iterable, Tuple

# Time subset.
years_targets: Iterable[int] = np.arange(2002, 2014 + 1)
years_features: Iterable[int] = np.arange(2001, 2014 + 1)

dir_source: str = '/workspace/BGI/data/DataStructureMDI/DATA/grid/Global/'
dir_target: str = '/scratch/hydrodl/data/'
dir_bgi: str = '/workspace/BGI/'
dir_ppl: str = '/workspace/bkraft/'

nlat: int = 180
nlon: int = 360
nlat_static: int = 180 * 30
nlon_static: int = 360 * 30

# Used to normalize TWS.
tws_norm_period: Tuple[str, str] = ('2002-01-01', '2008-12-31')

overwrite: bool = True
