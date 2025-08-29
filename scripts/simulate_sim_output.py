#!/usr/bin/env python3
# scripts/simulate_sim_output.py
from __future__ import annotations
import argparse, json, math, random, sys, shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List
import numpy as np
import cv2

# progress (tqdm if available)
try:
    from tqdm import trange
    def progress_iter(n, **kw): return trange(1, n+1, **kw)
    USING_TQDM = True
except Exception:  # pragma: no cover
    USING_TQDM = False
    def progress_iter(n, **kw): return range(1, n+1)

def iprint(*a, **k): print(*a, **k); sys.stdout.flush()

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def clear_dir(p: Path):
    if p.exists():
        for child in p.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)

def q_to_R(q):  # [w,x,y,z] -> 3x3
    w,x,y,z = q
    n = math.sqrt(w*w+x*x+y*y+z*z) + 1e-9
    w,x,y,z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=np.float32)

def look_at_T_cw(cam_pos, target=(0,0,0), up=(0,0,1)):
    """
    Build T_cw (world -> camera) with camera axes:
      +Z forward (towards target), +X right, +Y down (matches image pixel y-down).
    """
    cam_pos = np.array(cam_pos, np.float32)
    target  = np.array(target,  np.float32)
    up      = np.array(up,      np.float32)

    f = target - cam_pos
    f = f / (np.linalg.norm(f) + 1e-9)       # forward (+Z_cam)
    s = np.cross(f, up); s /= (np.linalg.norm(s) + 1e-9)   # right   (+X_cam)
    u = np.cross(s, f)                        # world-up aligned camera up

    x_cam = s
    y_cam = -u                                # make camera +Y point *down* in image
    z_cam = f

    R_wc = np.stack([x_cam, y_cam, z_cam], axis=1)  # columns are camera axes in world
    R_cw = R_wc.T
    t_cw = -R_cw @ cam_pos
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_cw
    T[:3,  3] = t_cw
    return T


def world_to_cam(T_cw, Xw):
    Xw1 = np.concatenate([Xw, np.ones((len(Xw),1), np.float32)], axis=1)
    Xc1 = (T_cw @ Xw1.T).T
    return Xc1[:,:3]

def project(K, Xc):
    z = Xc[:,2:3]
    ok = z.squeeze() > 1e-6
    x = (K[0,0]*Xc[:,0]/z[:,0] + K[0,2])
    y = (K[1,1]*Xc[:,1]/z[:,0] + K[1,2])
    return np.stack([x,y],1), ok

def make_K_from_fov(width:int, height:int, fov_deg:float) -> np.ndarray:
    fov = math.radians(fov_deg)
    fx = 0.5*width / math.tan(0.5*fov)
    fy = fx
    cx, cy = width/2.0, height/2.0
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)

@dataclass
class Plane:
    n: np.ndarray; d: float; sx: float; sy: float

@dataclass
class Cuboid:
    c: np.ndarray; s: np.ndarray; q: np.ndarray

def random_cuboid(xy_lim=8.0, z_min=0.75, z_max=2.5, size_min=(0.4,0.4,0.6), size_max=(2.0,2.0,3.0)):
    c = np.array([random.uniform(-xy_lim, xy_lim),
                  random.uniform(-xy_lim, xy_lim),
                  random.uniform(z_min, z_max)], dtype=np.float32)
    s = np.array([random.uniform(size_min[0], size_max[0]),
                  random.uniform(size_min[1], size_max[1]),
                  random.uniform(size_min[2], size_max[2])], dtype=np.float32)
    yaw = random.uniform(-math.pi, math.pi)
    q = np.array([math.cos(yaw/2), 0, 0, math.sin(yaw/2)], dtype=np.float32)
    return Cuboid(c=c, s=s, q=q)

def draw_polyline(img, pts, color, thickness=1):
    pts_i = pts.astype(np.int32).reshape(-1,1,2)
    if len(pts_i) >= 2:
        cv2.polylines(img, [pts_i], False, color, thickness, cv2.LINE_AA)

def draw_edges(img, uv, idx_pairs, color, W, H):
    for a,b in idx_pairs:
        pa, pb = uv[a], uv[b]
        if np.any(np.isnan(pa)) or np.any(np.isnan(pb)): continue
        if (0<=pa[0]<W and 0<=pa[1]<H) or (0<=pb[0]<W and 0<=pb[1]<H):
            cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)), color, 2, cv2.LINE_AA)

def render_plane(img, K, T_cw, plane: Plane, W, H, step=1.0, color=(225,225,225)):
    n = plane.n / (np.linalg.norm(plane.n)+1e-9)
    d = float(plane.d); sx=float(plane.sx); sy=float(plane.sy)
    tmp = np.array([1,0,0], np.float32) if abs(n[0])<0.9 else np.array([0,1,0], np.float32)
    e1 = np.cross(n, tmp); e1 /= (np.linalg.norm(e1)+1e-9)
    e2 = np.cross(n, e1)
    X0 = -d * n
    us = np.arange(-sx/2, sx/2+1e-6, step)
    vs = np.arange(-sy/2, sy/2+1e-6, step)
    for u in us:
        line = np.stack([X0 + u*e1 + v*e2 for v in vs], axis=0)
        uv, ok = project(K, world_to_cam(T_cw, line))
        uv[~ok] = np.nan
        draw_polyline(img, uv[~np.isnan(uv).any(1)], color, 1)
    for v in vs:
        line = np.stack([X0 + u*e1 + v*e2 for u in us], axis=0)
        uv, ok = project(K, world_to_cam(T_cw, line))
        uv[~ok] = np.nan
        draw_polyline(img, uv[~np.isnan(uv).any(1)], color, 1)

def cuboid_corners_world(cb: Cuboid):
    he = 0.5*cb.s
    corners_local = np.array([[+he[0],+he[1],+he[2]],
                              [+he[0],+he[1],-he[2]],
                              [+he[0],-he[1],+he[2]],
                              [+he[0],-he[1],-he[2]],
                              [-he[0],+he[1],+he[2]],
                              [-he[0],+he[1],-he[2]],
                              [-he[0],-he[1],+he[2]],
                              [-he[0],-he[1],-he[2]]], dtype=np.float32)
    R = q_to_R(cb.q)
    return (R @ corners_local.T).T + cb.c

def render_cuboid(img, K, T_cw, cb: Cuboid, W, H, color=(40,120,255)):
    corners_world = cuboid_corners_world(cb)
    uv, ok = project(K, world_to_cam(T_cw, corners_world))
    uv[~ok] = np.nan
    E = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    draw_edges(img, uv, E, color, W, H)
    return uv, ok

def fill_cuboid_masks(obj_mask, dyn_mask, uv, ok):
    """
    Fill projected faces of a cuboid into the given masks.
    - obj_mask: union of ALL cuboids (uint8)
    - dyn_mask: only dynamic cuboids (uint8) or None
    Faces are quads; we require all 4 face corners to be in front of the camera.
    """
    # Corner indexing must match cuboid_corners_world()
    faces = [
        (0,1,3,2),  # +X
        (4,5,7,6),  # -X
        (0,1,5,4),  # +Y
        (2,3,7,6),  # -Y
        (0,2,6,4),  # +Z (top)
        (1,3,7,5),  # -Z (bottom)
    ]
    for fidx in faces:
        ids = np.array(fidx, dtype=np.int32)
        if np.all(ok[ids]):  # all 4 face corners are in front of camera
            pts = uv[ids].astype(np.int32)
            cv2.fillConvexPoly(obj_mask, pts, 255)
            if dyn_mask is not None:
                cv2.fillConvexPoly(dyn_mask, pts, 255)


def draw_dyn_mask(mask, uv, ok):
    valid = uv[ok]
    if len(valid) >= 3:
        hull = cv2.convexHull(valid.astype(np.float32))
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)

def sample_plane_scene_coords(plane: Plane, num=500):
    n = plane.n / (np.linalg.norm(plane.n)+1e-9); d=plane.d
    e1 = np.array([1,0,0], np.float32)
    if abs(np.dot(e1, n))>0.9: e1 = np.array([0,1,0], np.float32)
    e1 = np.cross(n, e1); e1 /= (np.linalg.norm(e1)+1e-9)
    e2 = np.cross(n, e1)
    X0 = -d * n
    u = (np.random.rand(num)-0.5)*plane.sx
    v = (np.random.rand(num)-0.5)*plane.sy
    XYZ = X0 + u[:,None]*e1 + v[:,None]*e2
    return XYZ

def sample_cuboid_scene_coords(cb: Cuboid, num=500):
    XYZ = []
    he = 0.5*cb.s
    R = q_to_R(cb.q); c = cb.c
    faces = [
        (np.array([+he[0],0,0]), np.array([0,+he[1],0]), np.array([0,0,+he[2]])),
        (np.array([-he[0],0,0]), np.array([0,+he[1],0]), np.array([0,0,+he[2]])),
        (np.array([0,+he[1],0]), np.array([+he[0],0,0]), np.array([0,0,+he[2]])),
        (np.array([0,-he[1],0]), np.array([+he[0],0,0]), np.array([0,0,+he[2]])),
        (np.array([0,0,+he[2]]), np.array([+he[0],0,0]), np.array([0,+he[1],0])),
        (np.array([0,0,-he[2]]), np.array([+he[0],0,0]), np.array([0,+he[1],0])),
    ]
    per_face = max(1, num//6)
    for origin, uvec, vvec in faces:
        u = (np.random.rand(per_face)*2-1)
        v = (np.random.rand(per_face)*2-1)
        pts_local = origin[None,:] + u[:,None]*uvec[None,:] + v[:,None]*vvec[None,:]
        pts_world = (R @ pts_local.T).T + c
        XYZ.append(pts_world)
    return np.concatenate(XYZ, axis=0)

def compute_scene_coords_sparse(K, T_cw, W, H, plane: Plane, cuboids: List[Cuboid],
                                n_plane=2000, n_cuboid=1000, stride=1):
    XYZ_world = []; uv_list = []
    XYZp = sample_plane_scene_coords(plane, num=n_plane)
    uvp, okp = project(K, world_to_cam(T_cw, XYZp))
    keep = okp & (uvp[:,0]>=0) & (uvp[:,0]<W) & (uvp[:,1]>=0) & (uvp[:,1]<H)
    XYZ_world.append(XYZp[keep]); uv_list.append(uvp[keep])
    for cb in cuboids:
        XYZc = sample_cuboid_scene_coords(cb, num=n_cuboid)
        uvc, okc = project(K, world_to_cam(T_cw, XYZc))
        keepc = okc & (uvc[:,0]>=0) & (uvc[:,0]<W) & (uvc[:,1]>=0) & (uvc[:,1]<H)
        XYZ_world.append(XYZc[keepc]); uv_list.append(uvc[keepc])
    if not uv_list:
        return np.zeros((0,2), np.float32), np.zeros((0,3), np.float32)
    UV = np.vstack(uv_list).astype(np.float32) / max(1, stride)
    XYZ = np.vstack(XYZ_world).astype(np.float32)
    return UV, XYZ

def main():
    ap = argparse.ArgumentParser(description="Simulate Isaac/Cosmos-like multi-frame dataset.")
    ap.add_argument("out_dir", type=Path, help="Output dataset directory")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--motion", choices=["circle","line","fixed"], default="circle")
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--track-len", type=float, default=16.0)
    ap.add_argument("--height", type=float, default=2.0)
    ap.add_argument("--fov-deg", type=float, default=60.0)
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--size", type=str, default="640x480")     # WxH
    ap.add_argument("--objects", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dyn-ratio", type=float, default=0.25)
    ap.add_argument("--plane-size", type=float, default=40.0)
    ap.add_argument("--write-scene-coords", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--purge", action="store_true", help="Delete existing files in subdirs when --overwrite")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    # camera intrinsics
    W,H = [int(x) for x in args.size.lower().split("x")]
    if args.fx is not None:
        K = np.array([[args.fx,0,W/2],[0,args.fx,H/2],[0,0,1]], dtype=np.float32)
    else:
        K = make_K_from_fov(W, H, args.fov_deg)

    # prepare dirs
    subdirs = ["images","meta","primitives","scene_coords","masks/dynamic","masks/objects"]

    for sub in subdirs: ensure_dir(args.out_dir/sub)

    # overwrite handling
    already = any((args.out_dir/s).iterdir() for s in ["images","meta","primitives"])
    if already and not args.overwrite:
        iprint(f"⚠️  {args.out_dir} not empty. Use --overwrite to replace existing files.")
        return
    if already and args.overwrite and args.purge:
        for s in subdirs:
            clear_dir(args.out_dir/s)
        iprint("🧹 Purged existing files in output subdirectories.")

    # scene
    plane = Plane(n=np.array([0,0,1], dtype=np.float32), d=0.0, sx=args.plane_size, sy=args.plane_size)
    cuboids = [random_cuboid(xy_lim=max(6.0, 0.35*args.plane_size)) for _ in range(args.objects)]

    # camera path
    def cam_pose(i):
        if args.motion == "circle":
            theta = 2*math.pi * (i/(args.frames))
            pos = (args.radius*math.cos(theta), args.radius*math.sin(theta), args.height)
            return look_at_T_cw(pos, target=(0,0,0))
        elif args.motion == "line":
            t = (i/(max(1,args.frames-1))) - 0.5
            pos = (t*args.track_len, -args.radius, args.height)  # ✅ track_len
            return look_at_T_cw(pos, target=(0,0,0))
        else:
            return look_at_T_cw((-args.radius, args.radius*0.5, args.height), target=(0,0,0))

    iprint(f"Generating dataset at: {args.out_dir}  (frames={args.frames}, objects={args.objects})")

    try:
        from tqdm.auto import tqdm  # ← auto is nicer across envs
        pbar = tqdm(range(1, args.frames + 1), desc="Frames", unit="frame")
    except Exception:
        pbar = range(1, args.frames + 1)

    using_tqdm = hasattr(pbar, "set_postfix")  # ← runtime check

    for i in pbar:
        fid = f"{i:06d}"
        T_cw = cam_pose(i-1)

        # RGB + masks
        img = np.full((H, W, 3), 255, np.uint8)
        render_plane(img, K, T_cw, plane, W, H, step=1.0)

        obj_mask = np.zeros((H, W), np.uint8)   # union of all cuboids
        dyn_mask = np.zeros((H, W), np.uint8)   # only selected "dynamic" cuboids

        # primitives
        n_dyn = int(round(args.dyn_ratio * len(cuboids))) or 0
        dynamic_idx = set(np.random.choice(len(cuboids), size=n_dyn, replace=False)) if len(cuboids) else set()

        for j, cb in enumerate(cuboids):
            uv, ok = render_cuboid(img, K, T_cw, cb, W, H)  # outlines for visualization
            # fill masks by faces (solid)
            fill_cuboid_masks(obj_mask, dyn_mask if j in dynamic_idx else None, uv, ok)

        # axes (debug)
        axes = np.array([[0,0,0],[2,0,0],[0,2,0],[0,0,2]], dtype=np.float32)
        uv_axes, ok_axes = project(K, world_to_cam(T_cw, axes))
        if ok_axes[0]:
            for k, col in zip([1,2,3], [(0,0,255),(0,255,0),(255,0,0)]):
                if ok_axes[k]:
                    cv2.line(img, tuple(uv_axes[0].astype(int)), tuple(uv_axes[k].astype(int)), col, 2, cv2.LINE_AA)

        # write files ...
        cv2.imwrite(str(args.out_dir/"images"/f"{fid}.png"), img)
        cv2.imwrite(str(args.out_dir/"masks"/"objects"/f"{fid}.png"), obj_mask)
        cv2.imwrite(str(args.out_dir/"masks"/"dynamic"/f"{fid}.png"), dyn_mask)
        # (primitives/meta/scene_coords same as you have)

        if using_tqdm:
            pbar.set_postfix(frame=fid, dyn=len(dynamic_idx),
                            sc="✓" if args.write_scene_coords else "–",
                            refresh=False)
        elif (i == 1) or (i % 10 == 0) or (i == args.frames) or args.verbose:
            iprint(f"  frame {i}/{args.frames}: img+meta+prims{' +sc' if args.write_scene_coords else ''}")

    iprint("✅ Done.")
    iprint(f"  images:        {args.out_dir/'images'}")
    iprint(f"  meta:          {args.out_dir/'meta'}")
    iprint(f"  primitives:    {args.out_dir/'primitives'}")
    iprint(f"  scene_coords:  {args.out_dir/'scene_coords'} (present only if --write-scene-coords)")
    iprint(f"  dyn masks:     {args.out_dir/'masks'/'dynamic'}")
    iprint("Tip: validate with:  python scripts/validate_dataset.py <out_dir>")
    iprint("     simulate outputs: python scripts/simulate_outputs.py <out_dir> --out outputs/simulated")

if __name__ == "__main__":
    main()
