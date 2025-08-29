# PRISM-SLAM · `scripts/` toolkit

Utilities for generating synthetic datasets, simulating multi-level model outputs, validating structure, and visualizing GT vs predictions with [Rerun](https://www.rerun.io/).

> Install tips: `./setup.sh --with-viz` adds `opencv-python`, `tqdm`, and `rerun`.  
> All CLIs use **hyphenated flags** (argparse maps them to `args.some_flag`).

---

## Contents

- `simulate_sim_output.py` — **Synthetic dataset generator** (Isaac-/Cosmos-style, no engine needed)
- `validate_dataset.py` — **Dataset structure/schema checks**
- `simulate_model_outputs.py` — **Model output simulator** (L0/L1/L2) from a dataset
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
scene\_coords/{fid}.npz           # optional (--write-scene-coords)
masks/objects/{fid}.png          # union of all cuboids (255=object)
masks/dynamic/{fid}.png          # subset tagged as dynamic

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

## 3) Simulate model outputs · `simulate_model_outputs.py`

Creates toy **L0/L1/L2** predictions over a dataset to exercise the pipeline and the viewer.

**Outputs (under `--out`):**

```
l0/occupancy/{fid}.png           # coarse free-space/obstacle likelihood (uint8)
l0/dynamic_mask/{fid}.png        # movers only (uint8)
l1/scene_coords/{fid}.npz        # sampled uv/XYZ (from GT npz if present)
l1/primitives_pred/{fid}.json    # jittered/dropped GT + spurious
l1/pose/{fid}.json               # noisy T_cw
l2/refinements/{fid}.json        # placeholder
l2/attributes/{fid}.json         # placeholder
```

**Key flags**

* `--out PATH` · output root (default `outputs/simulated`)
* `--stride-l0 S`, `--scene-stride S`
* `--num-scene-pts N`, `--scene-xyz-noise σ`
* `--l1-drop-prob p`, `--l1-spurious-max k`
* `--limit N`, `--seed SEED`, `--overwrite`, `--purge`, `-v`

**Example**

```bash
python scripts/simulate_model_outputs.py examples/sim_circle_60 \
  --out outputs/sim_circle_60_model \
  --num-scene-pts 8000 --l1-drop-prob 0.2 --l1-spurious-max 2 \
  --overwrite --purge --seed 0
```

---

## 4) Visualize · `viz_rerun.py`

Streams GT vs Pred into a Rerun viewer with one camera per frame in `/world/gt` and `/world/pred`.

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
  --show-pred-primitives --show-scene-coords
```

> Notes: Uses the Rerun 0.23+ API (`rr.Pinhole(..., resolution=[W,H])`, `rr.set_time(..., duration=...)`).

---

## Typical workflow

```bash
# 1) Generate a small synthetic sequence
python scripts/simulate_sim_output.py examples/sim_circle_60 --frames 60 --overwrite --purge

# 2) Validate structure
python scripts/validate_dataset.py examples/sim_circle_60

# 3) Simulate model outputs
python scripts/simulate_model_outputs.py examples/sim_circle_60 \
  --out outputs/sim_circle_60_model --overwrite --purge

# 4) Visualize
python scripts/viz_rerun.py examples/sim_circle_60 \
  --pred outputs/sim_circle_60_model --show-pred-primitives --show-scene-coords
```

---

## Masks: dynamic vs all-objects

* `masks/objects/*`: **union of all cuboids** (static + dynamic), white on black.
* `masks/dynamic/*`: **subset** of moving objects (per `--dyn-ratio` in the generator).
  Use it to down-weight movers for pose/mapping.

(If you need instance-ID masks, we can add a `--write-instance-masks` option that saves `uint16` labels.)

---

## Troubleshooting

* **Rerun warnings**: If you see `from_parent` or `set_time_seconds` deprecations, you’re on ≥0.23; the scripts already use the new API.
* **Nothing shows in viewer**: Ensure `--pred` points to the simulated outputs and the `fid`s match your dataset.
* **Overwrites**: Use `--overwrite --purge` to avoid mixing old/new outputs.

---


Want me to commit this content into `scripts/README.md` and add a link to it from the top-level `README.md`?
::contentReference[oaicite:0]{index=0}
```

