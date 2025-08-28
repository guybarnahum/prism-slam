import numpy as np

def _norm(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v) + eps
    return (v / n).astype(np.float32)

def plane_from_params(n, d):
    n = _norm(n); d = float(d); return n, d

def cuboid_from_params(c, s, q):
    c = np.asarray(c, np.float32)
    s = np.maximum(np.asarray(s,np.float32), 1e-3)
    q = _norm(q)
    return c, s, q

def cylinder_from_params(p, v, r, h):
    p = np.asarray(p, np.float32)
    v = _norm(v)
    r = float(max(r, 1e-5)); h = float(max(h, 1e-5))
    return p, v, r, h

def superquadric_from_params(c, q, a, e):
    c = np.asarray(c, np.float32)
    q = _norm(q)
    a = np.maximum(np.asarray(a, np.float32), 1e-4)
    e = np.maximum(np.asarray(e, np.float32), 0.1)
    return c, q, a, e
