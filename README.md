# PRISM-SLAM
**PRImitive Scene Map SLAM** — a hybrid visual SLAM that combines a **scene-coordinate head** (PnP/DSAC) for precise pose with a **primitive-shape head** (planes/cuboids/cylinders/superquadrics) for a **tiny, planner-ready map**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guybarnahum/prism-slam/blob/main/notebooks/PRISM_SLAM_Colab.ipynb)

> Compact, navigation-grade maps with loop closures and multi-session merging — without photorealistic reconstruction.

## Highlights
- **Hybrid**: Scene-coordinates → tight 6-DoF pose; Primitive set-prediction → compact map
- **Planner-friendly**: direct ESDF/costmap derivation from shapes
- **Dynamic-aware**: masks movers before both heads
- **Colab-first**: launch the paper/notebook above; edit & run in the cloud

## Repo Layout
```
prism-slam/
  notebooks/PRISM_SLAM_Colab.ipynb
  src/prismslam/...
  scripts/validate_dataset.py
  examples/minimal_scene/...
  LICENSE
  pyproject.toml
  requirements.txt
  .gitignore
  .gitattributes
```

## Quickstart
```bash
# Clone your new repo once you've pushed it to GitHub:
git clone git@github.com:guybarnahum/prism-slam.git
cd prism-slam

# (Optional) local env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Validate example dataset structure
python scripts/validate_dataset.py examples/minimal_scene
```

## Dataset (brief)
Per-frame JSON (`meta/{frame}.json`):
```json
{ "frame_id": 1, "rgb_path": "images/000001.png", "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
   "T_cw": [[...],[...],[...],[0,0,0,1]], "scene_id": "scene_000" }
```
Scene-coords: `scene_coords/{frame}.npz` with `uv (N,2)`, `XYZ (N,3)`.  
Primitives: `primitives/{frame}.json` (or per-scene) with items of `type` ∈ {plane,cuboid,cylinder,superquadric} and params.  
See **notebooks/PRISM_SLAM_Colab.ipynb** for full spec.

## Colab
- Click the badge above. The notebook reads like a short paper and includes dataset schemas and code stubs.
- To use GitHub Releases for datasets, upload zips to your repo’s Releases and `wget` them in Colab.

## License
MIT © 2025 PRISM-SLAM contributors
