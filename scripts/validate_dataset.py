#!/usr/bin/env python3
import os, json, sys
from pathlib import Path

def main(root):
    root = Path(root)
    metas = sorted(root.glob("meta/*.json"))
    prims = sorted(root.glob("primitives/*.json"))
    print(f"Found {len(metas)} meta files, {len(prims)} primitive files")
    bad = False
    for m in metas:
        try:
            meta = json.loads(m.read_text())
            assert "frame_id" in meta and "rgb_path" in meta and "K" in meta and "T_cw" in meta
        except Exception as e:
            print("BAD META:", m, e); bad=True
    for p in prims:
        try:
            data = json.loads(p.read_text())
            assert "primitives" in data
        except Exception as e:
            print("BAD PRIMS:", p, e); bad=True
    print("OK" if not bad else "FAIL")
    return 0 if not bad else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv)>1 else "."))
