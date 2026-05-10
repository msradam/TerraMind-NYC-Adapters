# How these adapters were trained

Source: `riprap-nyc/experiments/18_terramind_nyc_lora/` — the LoRA family
training. Three adapters (`buildings_nyc`, `lulc_nyc`, `tim_nyc`) trained
on the same shared TerraMind 1.0 base + the same multi-modal NYC chip
data, with task-specific labels and decoders.

## Hardware + software

| | |
|---|---|
| GPU | 1× AMD Instinct MI300X (192 GB HBM3) |
| Cloud | AMD Developer Cloud (DigitalOcean droplet) |
| ROCm | 4.0.0+1a5c7ec |
| Container | `terramind` (custom image with TerraTorch installed) |
| Python | 3.12 |
| TerraTorch | 1.2.7 |
| PyTorch Lightning | 2.6.1 |
| peft | 0.18.1 |
| Precision | fp16-mixed |

Wall-clock per adapter: **25–40 min** on MI300X. All three under 2 hr
sequential.

## Data

| Component | Source | License | Vintage |
|---|---|---|---|
| Sentinel-2 L2A imagery | ESA Copernicus via Major-TOM Core-S2L2A | Copernicus Open Data | 2017–2024 |
| Sentinel-1 RTC imagery | Major-TOM Core-S1RTC | Copernicus Open Data | 2017–2024 |
| Copernicus DEM GLO-30 | Major-TOM Core-DEM | Copernicus Open Data | static |
| LULC labels | ESA WorldCover 2021 v200 (10 m global) | CC-BY-4.0 | 2021 |
| Building footprints | NYC DOITT Building Footprints (1.08M polygons) | NYC OpenData public domain | 2024-09 |

### Chip pipeline (shared by all adapters)

Same pipeline as Phases 2/3/4 for byte-for-byte consistency with the
existing full-fine-tune baselines (LoRA-vs-full-FT comparison validity):

1. **Major-TOM filter**: pull S2L2A + S1RTC + DEM products whose
   centroid falls in the NYC bbox (-74.30, 40.45, -73.65, 40.95) with
   cloud cover ≤ 20%. Yields **22 unique parent grid cells**.
2. **Slice into 224×224 sub-chips**: each Major-TOM parent is sliced
   into a 4×4 grid of non-overlapping 224×224 chips, totalling **352 chips**.
3. **Per-chip label rasterization**:
   - LULC: read WorldCover 2021 GeoTIFF at the chip bbox, collapse the
     11 native classes to 5 NYC-relevant macro-classes (see below).
   - Buildings: rasterize NYC DOITT polygons onto the chip grid as a
     binary mask.
4. **Pack to ImpactMesh-compatible zarr.zip** for TerraTorch's
   `ImpactMeshDataModule`.

### LULC class collapse (5-class NYC scheme)

| Our class | WorldCover sources | Rationale |
|---|---|---|
| 0 — Water | `Permanent water (80)`, `Snow/ice (70)` | Hudson, East River, Jamaica Bay, reservoirs |
| 1 — Impervious / urban | `Built-up (50)` excluding building footprints | Roads, parking, plazas — drives stormwater runoff |
| 2 — Vegetation | `Tree cover (10)`, `Shrubland (20)`, `Grassland (30)`, `Wetland (90)`, `Mangroves (95)`, `Moss (100)` | Permeable surfaces, urban canopy |
| 3 — Bare / cropland | `Bare/sparse (60)`, `Cropland (40)` | Beach, Floyd Bennett, Plumb Beach |
| 4 — Building | NYC DOITT polygons rasterized | Distinct from impervious because rooftops have different EO signatures |

Note: this harness's reproduction recovered the **model's actual class
order** by permutation search (the card names classes but doesn't
number-list them) — the order above matches what the loaded weights
predict.

## Splits

Stratified-random with `seed=42`. Counts inherited from Phase 2/3/4:

| Adapter | Train | Val | Test | Total |
|---|---|---|---|---|
| `lulc_nyc` | 224 | 48 | 64 | 336 |
| `tim_nyc` | 224 | 48 | 64 | 336 |
| `buildings_nyc` | 144 | 32 | 32 | 208 |

Test-split chip-ID lists committed at
`adapters/{name}/splits/test.txt` AND mirrored on the HF model repo
under `<adapter>/splits/test.txt`.

## Hyperparameters (defaults shared across adapters)

| | |
|---|---|
| Backbone | `terramind_v1_base` (or `terramind_v1_base_tim` for TiM) |
| Backbone weights | **Frozen** (LoRA only updates Δ) |
| Modalities | S2L2A (12 bands × 4 timesteps) + S1RTC (2 bands × 4) + DEM (1 band × 4) |
| Temporal pooling | concat |
| Necks | SelectIndices [2, 5, 8, 11] → ReshapeTokensToImage → LearnedInterpolateToPyramidal |
| Decoder | UNet (channels [512, 256, 128, 64]) trained from scratch |
| LoRA rank `r` | 16 |
| LoRA `alpha` | 32 |
| LoRA dropout | 0.05 |
| LoRA target modules | `["attn.qkv", "attn.proj"]` (24 attention blocks) |
| Optimizer | AdamW |
| LR (LoRA params) | 5e-4 |
| LR (decoder + head) | 1e-4 |
| Scheduler | ReduceLROnPlateau, factor 0.5, patience 3 |
| Weight decay | 1e-4 |
| Batch size | 8 |
| Epochs | 30 (LULC, TiM) / 40 (Buildings, more class imbalance) |
| Loss — LULC, TiM | Class-weighted cross-entropy (inverse-frequency on train) |
| Loss — Buildings (v2 published) | CE with class weights `[0.6, 1.6]` |
| Loss — Buildings (v1 archived) | Focal-Tversky (α=0.7, β=0.3, γ=0.75) — gave unstable training, replaced |
| Random seed | 42 |
| Effective adapter size | ~5 MB (LoRA Δ) + ~80 MB (UNet decoder) ≈ ~85 MB float32 |

## v1 → v2 lift on buildings

The first attempt (v1) used Focal-Tversky loss on the same architecture
and data — val mIoU oscillated between 0.21 and 0.43 across epochs and
didn't converge. The literature suggests Focal-Tversky is the right
loss for sparse-positive **full** fine-tunes, but under LoRA with
limited capacity, simpler is better.

| Run | Loss | Test mIoU | Test IoU bld | Train wall-clock |
|---|---|---:|---:|---:|
| v1 archived | Focal-Tversky (α=0.7, β=0.3, γ=0.75) | 0.3462 | 0.1606 | ~7 min |
| **v2 published** | **CE, class weights [0.6, 1.6]** | **0.5518** | **0.2928** | ~7 min |

## Training command

```bash
# Inside the terramind container
cd /workspace/experiments/18_terramind_nyc_lora

# Train all three adapters (sequential, ~2 hr total)
for a in lulc_nyc tim_nyc buildings_nyc; do
    python3 shared/train_lora.py --config adapters/$a/config.yaml
done

# Eval all three
for a in lulc_nyc tim_nyc buildings_nyc; do
    python3 shared/eval_adapter.py --adapter adapters/$a
done

# Publish to HF
python3 shared/publish_hf.py --all
```

Per-adapter outputs:

- `adapters/<name>/output/last.ckpt` — final epoch
- `adapters/<name>/output/best_val_loss.ckpt` — best val checkpoint
- `adapters/<name>/output/lora_only.safetensors` — adapter-only weights
  (LoRA matrices + decoder + head; base encoder weights NOT included)
- `adapters/<name>/output/train_log.csv` — per-epoch loss + metrics

## Eval methodology

Card-published per-class IoU on the test split:

### `lulc_nyc` (5-class)

| Class | IoU |
|---|---:|
| 0 — Water | 0.7696 |
| 1 — Impervious / urban | 0.9494 |
| 2 — Vegetation | 0.7803 |
| 3 — Bare / cropland | 0.3892 |
| 4 — Building | 0.0447 |
| **Test mIoU** | **0.5866** |

### `buildings_nyc` (binary)

| | |
|---|---:|
| IoU non-bld (0) | 0.8107 |
| IoU bld (1) | 0.2928 |
| Pixel accuracy | 0.8245 |
| F1 macro | 0.6742 |
| **Test mIoU** | **0.5518** |

## Why LoRA over full fine-tune

Earlier work in this repo (Phase 2/3/4) shipped three independent full
fine-tunes, each ~640 MB to 2.2 GB. Three near-identical encoders sat
on disk because only the decoder + a small fraction of attention
weights actually changed per task. This consolidation:

- One TerraMind base (~1.6 GB), kept fresh from IBM, downloaded once
- Three adapters totalling ~1 GB on disk (vs ~3.5 GB previously)
- Adding a new NYC task ("heat-island exposure", "stormwater impervious
  estimate", etc.) is one new ~325 MB adapter, not a 2 GB full fine-tune

LoRA-vs-full-FT comparison was locked-methodology per the experiment's
ADR-005 (byte-for-byte same train/val/test splits).

## Where to extend

- Train a fourth adapter for impervious-surface mapping (NYC DEP would
  use this for stormwater capacity planning).
- Train a heat-island adapter using NYC DOH heat-vulnerability index
  + Landsat thermal as labels.
- Extend training data outside NYC to compare cross-city transfer.
