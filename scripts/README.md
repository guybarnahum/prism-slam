# PRISM-SLAM · `scripts/` toolkit

Utilities for generating synthetic datasets, simulating multi-level model outputs, validating structure, and visualizing GT vs predictions with [Rerun](https://www.rerun.io/).

> Install tips: `./setup.sh --with-viz` adds `opencv-python`, `tqdm`, and `rerun`.  
> All CLIs use **hyphenated flags** (argparse maps them to `args.some_flag`).

---

## Contents

- `simulate_sim_output.py` — **Synthetic dataset generator** (Isaac-/Cosmos-style, no engine needed)
- `validate_dataset.py` — **Dataset structure/schema checks**
- `simulate_model_output.py` — **Model output simulator** (L0/L1/L2) from a dataset
- `viz_rerun.py` — **Side-by-side viewer** for inputs/GT vs model outputs

---

## Coordinate conventions

- **World**: right-handed, **+Z up**, ground plane is `z=0` (plane normal `[0,0,1]`).
- **Camera**: **+Z forward**, **+X right**, **+Y down** (matches image coordinates).

---

## 1) Synthetic dataset · `simulate_sim_output.py`

Generates a small indoor scene with a floor plane and randomly placed cuboids, a camera trajectory, and optional sparse scene-coords.

**Outputs (under `<out_dir>/`):**
```

images/{fid}.png
meta/{fid}.json
primitives/{fid}.json
scene\_coords/{fid}.npz          # optional (--write-scene-coords)
masks/objects/{fid}.png         # union of all cuboids (255=object)
masks/dynamic/{fid}.png         # subset tagged as dynamic

````

**Key flags**
- `--frames N` · number of frames
- `--motion {circle,line,fixed}` + `--radius`, `--track-len`, `--height`
- `--size WxH` + `--fov-deg` (or `--fx`)
- `--objects N` · number of cuboids
- `--dyn-ratio r` · fraction of cuboids marked dynamic (for masks)
- `--write-scene-coords` · save sparse `(uv,XYZ)` samples
- `--overwrite` `--purge` · clobber outputs safely
- `-v/--verbose`

**Example**
```bash
python scripts/simulate_sim_output.py examples/sim_circle_60 \
  --frames 60 --motion circle --radius 12 --height 3 \
  --objects 8 --fov-deg 60 --size 640x480 --seed 0 \
  --write-scene-coords --overwrite --purge
````

---

## 2) Validate dataset · `validate_dataset.py`

Lightweight checks for directory layout and JSON schema conformance.

**Example**

```bash
python scripts/validate_dataset.py examples/sim_circle_60
```

---

## 3) Simulate model outputs · `simulate_model_output.py`

Creates toy **L0/L1/L2** predictions over a dataset to exercise the pipeline and the viewer.

**Outputs (under `--out`):**

```
l0/occupancy/{fid}.png           # coarse free-space/obstacle likelihood (uint8)
l0/dynamic_mask/{fid}.png        # movers only (uint8, synthetic)
l1/scene_coords/{fid}.npz        # sampled uv/XYZ (from GT npz if present)
l1/primitives_pred/{fid}.json    # jittered/dropped GT + spurious
l1/pose/{fid}.json               # noisy T_cw (good-model defaults baked in)
l2/refinements/{fid}.json        # placeholder
l2/attributes/{fid}.json         # placeholder
```

**Core flags (already implemented)**

* `--out PATH` · output root (default `outputs/simulated`)
* `--stride-l0 S` · downsample stride for L0 occupancy/mask
* `--scene-stride S` · downsample stride for L1 scene-coord UVs
* `--num-scene-pts N` · max sampled scene-coord points per frame
* `--scene-xyz-noise σ` · Gaussian noise (meters) added to scene XYZ samples
* `--l1-drop-prob p` · probability to drop each GT primitive before jitter (misses)
* `--l1-spurious-max k` · up to `k` extra hallucinated primitives per frame
* `--limit N`, `--seed SEED`, `--overwrite`, `--purge`, `-v`

> Note: The viewer shows the **same RGB image** under GT and Pred cameras (we’re visualizing model overlays, not a re-render). Differences appear in 3D pose, primitives, heatmaps, and points.

**Example (baseline good-model-ish)**

```bash
python scripts/simulate_model_output.py examples/sim_circle_60 \
  --out outputs/sim_circle_60_model \
  --num-scene-pts 8000 --l1-drop-prob 0.2 --l1-spurious-max 2 \
  --overwrite --purge --seed 0
```

### Noise & degradation controls (today)

Use these **existing** flags to make outputs look worse/better:

| What you want to degrade   | Knob(s) to turn                                                | Effect                                       |
| -------------------------- | -------------------------------------------------------------- | -------------------------------------------- |
| Fewer/messier scene points | `--num-scene-pts ↓`, `--scene-stride ↑`, `--scene-xyz-noise ↑` | Sparser, noisier PnP/DSAC input              |
| Missed detections          | `--l1-drop-prob ↑`                                             | GT primitives randomly removed before jitter |
| Hallucinated objects       | `--l1-spurious-max ↑`                                          | Extra (low-confidence) cuboids per frame     |
| Coarser L0                 | `--stride-l0 ↑`                                                | L0 occupancy/mask at lower resolution        |

**Ready-to-run “error profiles” (no code changes):**

* **Messy perception (misses + hallucinations + noisy scene coords)**

```bash
python scripts/simulate_model_output.py examples/sim_circle_60 \
  --out outputs/bad_perception --overwrite --purge \
  --l1-drop-prob 0.35 --l1-spurious-max 3 \
  --scene-xyz-noise 0.03 --scene-stride 16 --num-scene-pts 1500 \
  --stride-l0 32 --seed 13
```

* **Sparse geometry (OK detections, poor PnP support)**

```bash
python scripts/simulate_model_output.py examples/sim_circle_60 \
  --out outputs/sparse_geom --overwrite --purge \
  --num-scene-pts 800 --scene-stride 32 --scene-xyz-noise 0.02 \
  --l1-drop-prob 0.1 --l1-spurious-max 1 --stride-l0 32
```

* **Hallucination-heavy (detector gone wild)**

```bash
python scripts/simulate_model_output.py examples/sim_circle_60 \
  --out outputs/hallu --overwrite --purge \
  --l1-drop-prob 0.15 --l1-spurious-max 5 \
  --num-scene-pts 4000 --scene-xyz-noise 0.01
```

> **Pose noise:** The script currently uses baked-in “good model” pose jitter internally. If you want to tune pose drift/bias, see the *Optional knobs (WIP)* section below.

### Optional knobs (WIP / add later)

If you later add the small patch we discussed, you’ll gain:

* `--pose-rot-deg`, `--pose-trans-m` — tune per-frame pose jitter
* `--pose-bias-rot-deg`, `--pose-bias-trans-m` — constant bias (miscalib/drift)
* `--fail-rate` — occasional catastrophic failures (empty preds/pose score 0)
* `--mask-holes` — punch random holes in masks (sensor dropout)
* `--l0-occupancy-noise`, `--l0-esdf-sigma`, `--l0-dyn-fp`, `--l0-dyn-fn` — shape the L0 heads

Those appear in the example “error profiles” we shared earlier; they’ll work once those flags are added.

---

## 4) Visualize · `viz_rerun.py`

Streams GT vs Pred into a Rerun viewer with one camera per frame in `/world/gt` and `/world/pred`.
The blueprint lays out **GT 3D**, **Pred 3D**, and synced **GT/Pred images**; scrubbing the *frame* timeline moves both sides together.

**Common flags**

* `--pred PATH` · model outputs root
* `--fps F` · playback speed
* `--limit N`, `--start-index I`
* `--show-gt-primitives`
* `--show-pred-primitives`
* `--show-scene-coords`

**Example**

```bash
python scripts/viz_rerun.py examples/sim_circle_60 \
  --pred outputs/sim_circle_60_model \
  --fps 10 \
  --show-gt-primitives --show-pred-primitives --show-scene-coords
```

> Notes: Uses the Rerun 0.23+ API (`rr.Pinhole(..., resolution=[W,H])`, `rr.set_time(..., duration=...)`). Both image panels display the dataset RGB; 3D overlays differ (pose, cuboids, points, L0 upsampled maps).

---

## Masks: dynamic vs all-objects

* `masks/objects/*`: **union of all cuboids** (static + dynamic), white on black.
* `masks/dynamic/*`: **subset** of moving objects (per `--dyn-ratio` in the generator).
  Use it to down-weight movers for pose/mapping.

(Need instance-ID masks? We can add a `--write-instance-masks` option that saves `uint16` labels.)

---

## Typical workflow

```bash
# 1) Generate a small synthetic sequence
python scripts/simulate_sim_output.py examples/sim_circle_60 --frames 60 --overwrite --purge

# 2) Validate structure
python scripts/validate_dataset.py examples/sim_circle_60

# 3) Simulate model outputs
python scripts/simulate_model_output.py examples/sim_circle_60 \
  --out outputs/sim_circle_60_model --overwrite --purge

# 4) Visualize
python scripts/viz_rerun.py examples/sim_circle_60 \
  --pred outputs/sim_circle_60_model \
  --show-gt-primitives --show-pred-primitives --show-scene-coords
```

---

## Troubleshooting

* **Viewer shows orange cuboids on both sides:** That’s expected if you enabled `--show-gt-primitives` and `--show-pred-primitives`; colors differ (GT = green, Pred = orange) in 3D panes. Both *image* panes show the same dataset RGB by design.
* **Nothing shows:** Ensure `--pred` points to the simulated outputs and the `fid`s match your dataset’s `meta/*`.
* **Overwrites:** Use `--overwrite --purge` to avoid mixing old/new outputs.


::contentReference[oaicite:0]{index=0}
```
