"""TerraMind 1.0 NYC LoRA adapters.

>>> from terramind_nyc_adapters import load_terramind_adapter
>>> bld_model, preprocess, _ = load_terramind_adapter({
...     "adapter_dir": "buildings_nyc", "num_classes": 2,
... })
>>> lulc_model, lulc_pre, _ = load_terramind_adapter({
...     "adapter_dir": "lulc_nyc", "num_classes": 5,
... })
"""

from .data import (
    NYC_AOIS,
    iter_holdout_tiles,
    iter_lulc_holdout_tiles,
    load_buildings_adapter,
    load_terramind_adapter,
)

__version__ = "0.1.0"
__all__ = [
    "NYC_AOIS",
    "iter_holdout_tiles",
    "iter_lulc_holdout_tiles",
    "load_buildings_adapter",
    "load_terramind_adapter",
]
