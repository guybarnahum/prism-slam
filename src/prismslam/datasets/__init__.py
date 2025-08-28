"""Dataset utilities for PRISM-SLAM."""
from __future__ import annotations
import json, os, numpy as np

def load_frame_meta(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def load_scene_coords(npz_path: str):
    return np.load(npz_path)
