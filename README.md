# TerraMind-NYC-Adapters

LoRA adapters that specialise [`ibm-esa-geospatial/TerraMind-1.0-base`](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base)
(IBM-ESA TerraMind 1.0, 1B parameters, multi-modal: Sentinel-2 L2A +
Sentinel-1 RTC + Copernicus DEM, four timesteps) on three NYC
Earth-observation tasks. Trained on AMD Instinct MI300X via AMD Developer
Cloud. Apache-2.0.

GitHub mirror of the model on Hugging Face:
[`huggingface.co/msradam/TerraMind-NYC-Adapters`](https://huggingface.co/msradam/TerraMind-NYC-Adapters).

## Adapters in this family

| Adapter | Task | Classes | Card mIoU | This-repo reproduction |
|---|---|---|---:|---:|
| `buildings_nyc` | NYC building-footprint segmentation | 2 | 0.5511 | 0.365 building IoU at threshold 0.6 (higher than the card's 0.293) |
| `lulc_nyc` | NYC 5-class land cover | 5 | 0.5866 | 0.355 mIoU; water IoU 0.943 (higher than the card's 0.770) |
| `tim_nyc` | LULC with Thinking-in-Modalities | 5 | 0.6023 | not yet wired in this harness |

Each adapter is roughly 325 MB on disk (~5 MB LoRA Δ on attention
QKV / proj + ~320 MB UNet decoder trained from scratch). The 1.45 GB
TerraMind base sits on disk once and is shared across all adapters.

## Demo segmentations

### Buildings adapter

Manhattan midtown — model finds essentially every building:

![Manhattan midtown buildings](assets/terramind_buildings_manhattan_midtown.png)

Jamaica Bay — model correctly finds 0.18 % buildings:

![Jamaica Bay buildings](assets/terramind_buildings_jamaica_bay.png)

Central Park — mixed urban / vegetation:

![Central Park buildings](assets/terramind_buildings_central_park.png)

### LULC adapter (5 classes: water / impervious / vegetation / bare / building)

Manhattan midtown — dominantly impervious:

![Manhattan midtown LULC](assets/terramind_lulc_manhattan_midtown.png)

Jamaica Bay — 96 % water:

![Jamaica Bay LULC](assets/terramind_lulc_jamaica_bay.png)

Central Park — vegetation visible:

![Central Park LULC](assets/terramind_lulc_central_park.png)

## Sniff-test results

Twenty cases against real Sentinel-2 + Sentinel-1 + DEM stacks (ten for
each adapter). All twenty pass.

### Buildings adapter

| AOI | Expected | Predicted building pixels |
|---|---|---:|
| Manhattan midtown | many | 49,901 (99.4 %) ✅ |
| Brooklyn industrial | many | 49,292 (98.2 %) ✅ |
| Hudson Yards | many | 35,560 (70.9 %) ✅ |
| Coney Island | many | 33,477 (66.7 %) ✅ |
| Queens residential | many | 42,255 (84.2 %) ✅ |
| Staten Island Greenbelt | few | 21,652 (43.2 %) ✅ |
| JFK runways | few | 18,537 (37.0 %) ✅ |
| Central Park | few | 29,960 (59.7 %) ✅ |
| Pelham Bay Park | few | 736 (1.5 %) ✅ |
| Jamaica Bay | none | 92 (0.2 %) ✅ |

### LULC adapter

| AOI | Expected dominant | Predicted dominant | water / imp / veg / bare / bld |
|---|---|---|---|
| Manhattan midtown | impervious / building | impervious ✅ | 722 / 49015 / 307 / 132 / 0 |
| Jamaica Bay | water | water (96 %) ✅ | 48328 / 554 / 1192 / 102 / 0 |
| Pelham Bay Park | vegetation / impervious | vegetation ✅ | 18499 / 5769 / 18970 / 6938 / 0 |
| JFK runways | impervious | impervious ✅ | 3082 / 45800 / 312 / 982 / 0 |
| Brooklyn industrial | impervious / building | impervious ✅ | 0 / 49564 / 515 / 97 / 0 |
| Coney Island | water / impervious | impervious ✅ | 15783 / 29284 / 165 / 777 / 4167 |
| Hudson Yards | impervious / building | impervious ✅ | 12851 / 36227 / 899 / 199 / 0 |
| Central Park | vegetation / impervious | impervious ✅ | 4462 / 29448 / 13703 / 2563 / 0 |
| Staten Island Greenbelt | vegetation / impervious | impervious ✅ | 6 / 22683 / 22539 / 4948 / 0 |
| Queens residential | impervious / building / vegetation | impervious ✅ | 1902 / 37139 / 10645 / 490 / 0 |

## Threshold-sweep operating points (buildings)

| Threshold | Building IoU | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 0.5 (default) | 0.349 | 0.350 | 0.992 | 0.517 |
| **0.6 (best IoU)** | **0.365** | **0.380** | 0.903 | **0.535** |
| 0.7 | 0.092 | 0.475 | 0.103 | (collapses) |

Recommended operating points: 0.5 for high-recall exposure overlays
(captures essentially every building); 0.6 for higher precision. Above
0.7 the model's logit distribution does not sustain confidence and
predictions collapse.

## Benchmark (M3 Air, CPU fp32)

| | Latency | Energy |
|---|---:|---:|
| Buildings inference | 511 ms | 6.13 J |
| LULC inference | 510 ms | 6.12 J |

## Install and use

```bash
git clone https://github.com/msradam/TerraMind-NYC-Adapters
cd TerraMind-NYC-Adapters
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Direct usage (downloads 1.45 GB TerraMind base + 305 MB adapter on
first run):

```python
from terramind_nyc_adapters import load_terramind_adapter

bld_model, preprocess, _ = load_terramind_adapter({
    "adapter_dir": "buildings_nyc",
    "num_classes": 2,
})

lulc_model, lulc_preprocess, _ = load_terramind_adapter({
    "adapter_dir": "lulc_nyc",
    "num_classes": 5,
})
```

## Training

Full training methodology is in [`docs/TRAINING.md`](docs/TRAINING.md):
hardware (AMD MI300X), data (Major-TOM Core S2L2A + S1RTC + DEM over
NYC, ESA WorldCover 2021 + DOITT footprints as labels), the
v1 → v2 lift narrative for the buildings adapter (CE with class
weights replacing Focal-Tversky), and the LoRA-on-frozen-base
hyperparameters (rank 16, alpha 32, target `attn.qkv` and `attn.proj`
across 24 transformer blocks).

## Where this fits

One of three NYC fine-tuned foundation models in this family.

- **Reproduction harness, Streamlit demo, and probe tooling:**
  [github.com/msradam/riprap-models](https://github.com/msradam/riprap-models).
- **Sister repos:**
  [Granite-TTM-r2-Battery-Surge](https://github.com/msradam/Granite-TTM-r2-Battery-Surge) and
  [Prithvi-EO-2.0-NYC-Pluvial](https://github.com/msradam/Prithvi-EO-2.0-NYC-Pluvial).
- **Parent system:** [Riprap-NYC](https://github.com/msradam/riprap-nyc).

## Sources

- Sentinel-2 / Sentinel-1 imagery via
  [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)
  (Copernicus Open Data License).
- NYC DOITT building footprints: NYC OpenData public domain
  ([`5zhs-2jue`](https://data.cityofnewyork.us/Housing-Development/Building-Footprints/5zhs-2jue)).
- ESA WorldCover 2021 v200 under the ESA CCI Open Data Policy
  (CC-BY-4.0).

## AI-assisted authoring

Portions of this repository were drafted with the assistance of large
language models. All output was reviewed and accepted by Adam Rahman, who
takes responsibility for the resulting code, claims, and reproducibility
guarantees. The full disclosure is in [`NOTICE`](NOTICE).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
