#!/usr/bin/env python3
"""
Simulate PRISM-SLAM *model* outputs (L0/L1/L2) given a dataset folder
containing images/meta/primitives[/scene_coords]. Designed to be fast,
repeatable, and easy to eyeball with the viz tool.

Inputs (per frame):
  meta/{fid}.json                 # required
  images/{fid}.png                # optional (blank if missing)
  primitives/{fid}.json           # optional (empty if missing)
  scene_coords/{fid}.npz          # optional (for L1 scene-coords sampling)

Outputs (under --out):
  l0/occupancy/{fid}.png
  l0/dynamic_mask/{fid}.png
  l1/scene_coords/{fid}.npz       # (uv, XYZ) sampled from GT npz if present
  l1/primitives_pred/{fid}.json   # set-prediction w/ confidences
  l1/pose/{fid}.json              # noisy T_cw prediction
  l2/refinements/{fid}.json
  l2/attributes/{fid}.json
"""
from __future__ import annotations
import argparse, json, math, random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
import sys

# ---------- progress ----------
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs): return x  # graceful fallback

# ---------- small utils ----------

def vprint(enabled: bool, *args, **kwargs):
    if enabled:
        print(*args, **kwargs); sys.stdout.flush()

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def load_json(p: Path) -> dict:
    with open(p, "r") as f:
        return json.load(f)

def inv_SE3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]; t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def jitter_pose(Twc: np.ndarray, rot_deg: float = 1.5, trans_std: float = 0.02) -> np.ndarray:
    """Add small SO(3) & R^3 noise to a world->camera pose (Twc)."""
    rvec = np.deg2rad(np.random.randn(3).astype(np.float32) * rot_deg)
    Rj, _ = cv2.Rodrigues(rvec)
    Tn = np.eye(4, dtype=np.float32)
    Tn[:3, :3] = Rj @ Twc[:3, :3]
    Tn[:3, 3] = Twc[:3, 3] + np.random.randn(3).astype(np.float32) * trans_std
    return Tn

# ---------- primitives ----------

@dataclass
class Primitive:
    type: str
    params: dict
    instance_id: str | int | None = None
    confidence: float = 1.0

def load_gt_primitives(path: Path) -> List[Primitive]:
    if not path.exists():
        return []
    data = load_json(path)
    items = data.get("primitives", data)
    out: List[Primitive] = []
    for i, it in enumerate(items):
        out.append(Primitive(
            type=it["type"],
            params=it["params"],
            instance_id=it.get("id", i),
            confidence=1.0,
        ))
    return out

def jitter_primitive(p: Primitive, scale: float = 0.02) -> Primitive:
    # Deep copy via json to keep lists/dicts simple
    q = json.loads(json.dumps({"type": p.type, "params": p.params}))
    def n(v):
        if isinstance(v, (int, float)):
            return float(v + np.random.randn() * scale * max(1.0, abs(v)))
        if isinstance(v, list):  return [n(x) for x in v]
        if isinstance(v, dict):  return {k: n(vv) for k, vv in v.items()}
        return v
    q["params"] = n(q["params"])
    return Primitive(
        type=q["type"],
        params=q["params"],
        instance_id=p.instance_id,
        confidence=float(np.clip(np.random.normal(0.82, 0.08), 0.05, 0.99)),
    )

def make_spurious_cuboid() -> Primitive:
    # Use same schema keys as GT (c, s, q) to keep tooling happy
    yaw = float(np.random.uniform(-math.pi, math.pi))
    q = [math.cos(yaw/2), 0.0, 0.0, math.sin(yaw/2)]
    return Primitive(
        type="cuboid",
        params={
            "c": [float(np.random.randn()*0.5), float(np.random.uniform(0.5, 2.0)), float(np.random.randn()*0.5)],
            "s": [float(np.random.uniform(0.3, 0.8)), float(np.random.uniform(0.3, 0.8)), float(np.random.uniform(0.3, 1.2))],
            "q": q,
        },
        instance_id=None,
        confidence=float(np.random.uniform(0.2, 0.6)),
    )

# ---------- L0 heads ----------

def make_l0_occupancy(img_shape: Tuple[int,int,int], prims: List[Primitive], stride: int) -> np.ndarray:
    h, w = img_shape[:2]
    H, W = (h + stride - 1) // stride, (w + stride - 1) // stride
    occ = np.zeros((H, W), np.float32)
    # Toy raster: cuboids add rectangles, cylinders (if any) add disks
    for p in prims:
        if p.type == "cuboid":
            cx, cy = np.random.randint(0, W), np.random.randint(0, H)
            sx, sy = np.random.randint(1, 4), np.random.randint(1, 4)
            occ[max(0, cy-sy):min(H, cy+sy+1), max(0, cx-sx):min(W, cx+sx+1)] += 1.0
        elif p.type == "cylinder":
            cx, cy = np.random.randint(0, W), np.random.randint(0, H)
            r = np.random.randint(1, 3)
            cv2.circle(occ, (cx, cy), r, 1.0, -1)
    if occ.max() > 0:
        occ = cv2.GaussianBlur(occ, (5,5), 0)
        occ = np.clip(occ / (occ.max() + 1e-6), 0, 1)
    return (occ * 255).astype(np.uint8)

def make_l0_dynamic_mask(img_shape: Tuple[int,int,int], stride: int) -> np.ndarray:
    # Produce a downsampled (stride) mask with a few blobby movers
    h, w = img_shape[:2]
    H, W = (h + stride - 1) // stride, (w + stride - 1) // stride
    mask = np.zeros((H, W), np.uint8)
    n_blobs = max(1, int((h*w)/(256*256) * 2))
    for _ in range(n_blobs):
        cx, cy = np.random.randint(0, W), np.random.randint(0, H)
        r = np.random.randint(2, 6)
        cv2.circle(mask, (cx, cy), r, 255, -1)
    mask = cv2.medianBlur(mask, 5)
    return mask

# ---------- scene-coords sampling (from GT npz if present) ----------

def sample_scene_coords(npz_path: Path, num: int, xyz_noise: float, stride: int):
    if not npz_path.exists():
        return np.zeros((0,2), np.float32), np.zeros((0,3), np.float32)
    data = np.load(npz_path)
    uv  = data["uv"].astype(np.float32)
    xyz = data["XYZ"].astype(np.float32)
    N = len(uv)
    if N == 0:
        return uv, xyz
    idx = np.random.choice(N, size=min(num, N), replace=False)
    uv_s  = uv[idx] / max(1, stride)
    xyz_s = xyz[idx] + (np.random.randn(len(idx), 3).astype(np.float32) * xyz_noise)
    return uv_s, xyz_s

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Simulate multi-level model outputs for a dataset.")
    ap.add_argument("dataset_dir", type=Path, help="Dataset root (expects images/meta/primitives/...)")
    ap.add_argument("--out", type=Path, default=Path("outputs/simulated"), help="Output root dir")
    ap.add_argument("--stride-l0", dest="stride_l0", type=int, default=16, help="L0 occupancy/mask stride")
    ap.add_argument("--scene-stride", type=int, default=8, help="Stride used to downsample uv in L1 scene-coords")
    ap.add_argument("--num-scene-pts", type=int, default=5000, help="Max sampled scene-coord points per frame")
    ap.add_argument("--scene-xyz-noise", type=float, default=0.01, help="Gaussian noise (m) added to XYZ samples")
    ap.add_argument("--l1-drop-prob", type=float, default=0.15, help="Probability to drop a GT primitive before jitter")
    ap.add_argument("--l1-spurious-max", type=int, default=1, help="Max # of spurious primitives to add")
    ap.add_argument("--seed", type=int, default=0, help="PRNG seed")
    ap.add_argument("--limit", type=int, default=None, help="Process at most first N frames")
    ap.add_argument("--overwrite", action="store_true", help="Allow writing over existing outputs")
    ap.add_argument("--purge", action="store_true", help="Delete existing output subdirs before writing")
    ap.add_argument("--verbose", "-v", action="store_true", help="Print per-frame details")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    ds = args.dataset_dir
    frames = sorted((ds / "meta").glob("*.json"))
    if args.limit: frames = frames[:args.limit]
    if not frames:
        raise SystemExit(f"No meta/*.json found under {ds}")

    # Prepare outputs
    subdirs = [
        args.out/"l0"/"occupancy",
        args.out/"l0"/"dynamic_mask",
        args.out/"l1"/"scene_coords",
        args.out/"l1"/"primitives_pred",
        args.out/"l1"/"pose",
        args.out/"l2"/"refinements",
        args.out/"l2"/"attributes",
    ]
    for p in subdirs:
        ensure_dir(p)

    if any(p.iterdir() for p in [args.out/"l0", args.out/"l1", args.out/"l2"]) and not args.overwrite:
        print(f"⚠️  {args.out} has existing content. Use --overwrite (and optionally --purge)."); return
    if args.purge and args.overwrite:
        # wipe leaf folders to avoid mixing old & new
        for p in subdirs:
            for child in p.iterdir():
                child.unlink() if child.is_file() else None

    print("Simulating model outputs:")
    print(f"  dataset : {ds}")
    print(f"  frames  : {len(frames)}")
    print(f"  out     : {args.out}")
    print(f"  seed    : {args.seed}")
    print(f"  strides : L0={args.stride_l0}, scene={args.scene_stride}")
    if args.verbose: print("  mode    : verbose\n")

    # Stats
    total_gt_prims = 0
    total_pred_prims = 0
    total_spurious = 0
    total_sc_pts = 0

    pbar = tqdm(frames, desc="Frames", unit="frame")
    for mpath in pbar:
        meta = load_json(mpath)
        fid = f"{int(meta['frame_id']):06d}"

        # frame-local seeding for determinism per fid (but different across frames)
        base_seed = (args.seed * 1_000_003 + int(meta['frame_id'])) & 0xFFFFFFFF
        rng = np.random.RandomState(base_seed)  # only used where deterministic behavior helps

        # Image (tolerate missing)
        img_path = ds/"images"/f"{fid}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((480, 640, 3), np.uint8)
            vprint(args.verbose, f"[{fid}] WARNING: image missing ({img_path}); using blank frame.")
        H, W = img.shape[:2]

        # GT
        prim_gt = load_gt_primitives(ds/"primitives"/f"{fid}.json")
        sc_npz = ds/"scene_coords"/f"{fid}.npz"  # optional
        T_cw = np.array(meta["T_cw"], dtype=np.float32)

        # ---------- L0
        # (use global np.random to keep outputs different each run unless seed fixed)
        occ = make_l0_occupancy(img.shape, prim_gt, args.stride_l0)
        dyn = make_l0_dynamic_mask(img.shape, args.stride_l0)
        cv2.imwrite(str(args.out/"l0"/"occupancy"/f"{fid}.png"), occ)
        cv2.imwrite(str(args.out/"l0"/"dynamic_mask"/f"{fid}.png"), dyn)

        # ---------- L1 scene-coords
        uv, XYZ = sample_scene_coords(sc_npz, num=args.num_scene_pts,
                                      xyz_noise=args.scene_xyz_noise, stride=args.scene_stride)
        np.savez_compressed(args.out/"l1"/"scene_coords"/f"{fid}.npz", uv=uv, XYZ=XYZ)

        # ---------- L1 primitives (drop + jitter + spurious)
        keep: List[Primitive] = []
        drop_prob = float(np.clip(args.l1_drop_prob, 0.0, 0.95))
        for p in prim_gt:
            if rng.rand() >= drop_prob:
                keep.append(jitter_primitive(p, scale=0.02))
        n_spur = int(rng.randint(0, max(1, args.l1_spurious_max + 1)))
        for _ in range(n_spur):
            keep.append(make_spurious_cuboid())
        with open(args.out/"l1"/"primitives_pred"/f"{fid}.json", "w") as f:
            json.dump({"primitives": [p.__dict__ for p in keep]}, f, indent=2)

        # ---------- L1 pose (noisy T_cw)
        Twc = inv_SE3(T_cw)
        # temporarily swap the global RNG with our per-frame RNG for pose noise
        np_rand_backup = np.random.get_state()
        np.random.seed(base_seed ^ 0x9E3779B1)
        Twc_noisy = jitter_pose(Twc, rot_deg=1.5, trans_std=0.02)
        np.random.set_state(np_rand_backup)

        Tcw_pred = inv_SE3(Twc_noisy)
        with open(args.out/"l1"/"pose"/f"{fid}.json", "w") as f:
            json.dump({"T_cw": Tcw_pred.tolist(), "score": float(np.random.uniform(0.7, 0.95))}, f, indent=2)

        # ---------- L2 (toy placeholders)
        refinements = [{"instance_id": p.instance_id, "delta": 0.5} for p in keep]
        attributes  = [{"instance_id": p.instance_id,
                        "label": ("cabinet" if p.type == "cuboid" else "plane")} for p in keep]
        json.dump(refinements, open(args.out/"l2"/"refinements"/f"{fid}.json", "w"), indent=2)
        json.dump(attributes,  open(args.out/"l2"/"attributes"/f"{fid}.json",  "w"), indent=2)

        # Stats
        n_gt = len(prim_gt)
        n_pred = len(keep)
        n_sc = int(len(XYZ))
        n_spurious_frame = sum(1 for p in keep if p.instance_id is None)
        total_gt_prims   += n_gt
        total_pred_prims += n_pred
        total_spurious   += n_spurious_frame
        total_sc_pts     += n_sc

        # Progress bar info / verbose line
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(frame=fid, sc_pts=n_sc, pred=n_pred, spur=n_spurious_frame, refresh=False)
        vprint(args.verbose,
               f"[{fid}] L0: occ+dyn | L1: scene_pts={n_sc}, prim_gt={n_gt} -> pred={n_pred} (+{n_spurious_frame} spurious) | pose ✓")

    # Final summary
    print("\nDone.")
    print(f"  Frames processed     : {len(frames)}")
    print(f"  Scene-coord points   : {total_sc_pts}")
    print(f"  Primitives (GT total): {total_gt_prims}")
    print(f"  Primitives (Pred)    : {total_pred_prims}  (spurious added: {total_spurious})")
    print(f"  Output root          : {args.out}")

if __name__ == "__main__":
    main()
