#!/usr/bin/env python3
# scripts/viz_rerun.py
import argparse, json
from pathlib import Path
import numpy as np
import cv2
import rerun as rr

# progress (tqdm if available → auto picks nice renderer)
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kw): return x  # graceful fallback

# ---------- IO helpers ----------

def load_json(p: Path):
    if not p.exists(): return None
    with open(p, "r") as f: return json.load(f)

def read_image_or_blank(p: Path, fallback_size=(480, 640)):
    img = cv2.imread(str(p))
    if img is None:
        h, w = fallback_size
        img = np.zeros((h, w, 3), np.uint8)
    return img

def to_np4(Tlist):  # 4x4
    return np.array(Tlist, dtype=np.float32)

# ---------- Logging helpers ----------

def log_camera(ns: str, K: np.ndarray, img: np.ndarray, T_cw: np.ndarray):
    """Log an RGB pinhole camera and its pose (world->camera)."""
    H, W = img.shape[:2]
    # Images in Rerun are RGB; OpenCV is BGR:
    rr.log(f"{ns}/rgb", rr.Image(img[..., ::-1]))  # BGR->RGB
    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    # Rerun 0.23+: use width/height or resolution=[W, H] (image_size was removed)
    rr.log(
        f"{ns}",
        rr.Pinhole(
            focal_length=(fx, fy),
            principal_point=(cx, cy),
            resolution=[W, H],
        ),
    )

    # world <- cam: we store /cam as child of /world using transform from world to cam’s parent
    rr.log(
        ns,
        rr.Transform3D(
            mat3x3=T_cw[:3, :3],
            translation=T_cw[:3, 3],
            relation=rr.TransformRelation.ChildFromParent,
        ),
    )
    
def log_l0(ns: str, occ: np.ndarray | None, dyn: np.ndarray | None, up_to_hw: tuple[int,int]):
    H, W = up_to_hw
    if occ is not None:
        occ_up = cv2.resize(occ, (W, H), interpolation=cv2.INTER_NEAREST)
        rr.log(f"{ns}/l0/occupancy", rr.Image(occ_up))
    if dyn is not None:
        dyn_up = cv2.resize(dyn, (W, H), interpolation=cv2.INTER_NEAREST)
        rr.log(f"{ns}/l0/dynamic_mask", rr.SegmentationImage(dyn_up))

def _extract_c_s_q(prim: dict):
    """Support both {center,size,yaw?} AND {c,s,q} schemas."""
    p = prim.get("params", {})
    if "c" in p and "s" in p:  # preferred schema from generator
        c = np.array(p["c"], np.float32)
        s = np.array(p["s"], np.float32)
    else:
        c = np.array(p.get("center", [0,0,0]), np.float32)
        s = np.array(p.get("size",   [0.5,0.5,0.5]), np.float32)
    q = np.array(p.get("q", [1,0,0,0]), np.float32)
    return c, s, q

def log_primitives(ns_root: str, prims_path: Path):
    data = load_json(prims_path)
    if not data: return
    centers, sizes = [], []
    # (Rotation is optional; Boxes3D renders fine without it.
    #  If you want orientation, you can add 'quaternions=' below after verifying API.)
    for p in data.get("primitives", []):
        if p.get("type") == "cuboid":
            c, s, _q = _extract_c_s_q(p)
            centers.append(c); sizes.append(s)
    if centers:
        rr.log(f"{ns_root}/primitives/cuboids",
               rr.Boxes3D(centers=np.vstack(centers), sizes=np.vstack(sizes)))

def log_scene_points(ns_root: str, sc_path: Path):
    if not sc_path.exists(): return
    npz = np.load(sc_path)
    XYZ = np.asarray(npz.get("XYZ", []), np.float32)
    if len(XYZ):
        rr.log(f"{ns_root}/scene_coords", rr.Points3D(XYZ, radii=0.01))

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Side-by-side GT vs Pred visualization in Rerun.")
    ap.add_argument("dataset_dir", type=Path, help="Dataset root (images/meta/primitives/scene_coords)")
    ap.add_argument("--pred", type=Path, default=Path("outputs/simulated"), help="Model output root")
    ap.add_argument("--fps", type=float, default=10.0, help="Playback FPS")
    ap.add_argument("--limit", type=int, default=None, help="Show at most N frames")
    ap.add_argument("--start-index", type=int, default=0, help="Start from this frame index (after sorting meta/*)")
    ap.add_argument("--show-gt-primitives", action="store_true", help="Also render GT cuboids (if present)")
    ap.add_argument("--show-pred-primitives", action="store_true", help="Render predicted cuboids")
    ap.add_argument("--show-scene-coords", action="store_true", help="Render predicted scene-coord XYZ points")
    args = ap.parse_args()

    rr.init("PRISM-SLAM Viz", spawn=True)
    rr.set_time("frame", duration=0.0)  # 0.23 API
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    frames = sorted((args.dataset_dir / "meta").glob("*.json"))
    if args.start_index:
        frames = frames[args.start_index:]
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        raise SystemExit("No meta/*.json found under dataset_dir")

    pbar = tqdm(frames, desc="Viz frames", unit="frame")
    for i, mpath in enumerate(pbar):

        tsec = i / max(1e-6, args.fps)
        rr.set_time("frame", duration=float(tsec))  # 0.23 API

        meta = load_json(mpath)
        fid = f"{int(meta['frame_id']):06d}"

        K = np.array(meta["K"], np.float32)
        T_cw = np.array(meta["T_cw"], np.float32)

        img_path = args.dataset_dir / "images" / f"{fid}.png"
        img = read_image_or_blank(img_path)
        H, W = img.shape[:2]

        # --- Ground-truth namespace (left)
        ns_gt = f"world/gt/cam/{fid}"
        log_camera(ns_gt, K, img, T_cw)
        if args.show_gt_primitives:
            log_primitives("world/gt", args.dataset_dir / "primitives" / f"{fid}.json")

        # --- Prediction namespace (right)
        ns_pred = f"world/pred/cam/{fid}"
        pose_pred_path = args.pred / "l1" / "pose" / f"{fid}.json"
        T_pred = T_cw
        if pose_pred_path.exists():
            try:
                T_pred = to_np4(load_json(pose_pred_path)["T_cw"])
            except Exception:
                pass
        log_camera(ns_pred, K, img, T_pred)

        # L0 overlays (upsampled to image res for easy viewing)
        occ_path = args.pred / "l0" / "occupancy" / f"{fid}.png"
        dyn_path = args.pred / "l0" / "dynamic_mask" / f"{fid}.png"
        occ = cv2.imread(str(occ_path), cv2.IMREAD_GRAYSCALE) if occ_path.exists() else None
        dyn = cv2.imread(str(dyn_path), cv2.IMREAD_GRAYSCALE) if dyn_path.exists() else None
        log_l0(ns_pred, occ, dyn, (H, W))

        # L1 primitives + scene coords
        if args.show_pred_primitives:
            log_primitives("world/pred", args.pred / "l1" / "primitives_pred" / f"{fid}.json")
        if args.show_scene_coords:
            log_scene_points("world/pred", args.pred / "l1" / "scene_coords" / f"{fid}.npz")

        if hasattr(pbar, "set_postfix"):
            has_pose = "✓" if pose_pred_path.exists() else "–"
            pbar.set_postfix(frame=fid, pose=has_pose, refresh=False)

    print("Streaming to Rerun… close the window to exit.")

if __name__ == "__main__":
    main()
