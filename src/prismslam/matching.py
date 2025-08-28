import numpy as np

def greedy_match(cost):
    M, N = cost.shape
    pairs, used_r, used_c = [], set(), set()
    big = cost.max() + 1e6 if cost.size > 0 else 1e6
    for _ in range(min(M, N)):
        masked = cost.copy()
        if used_r: masked[list(used_r), :] = big
        if used_c: masked[:, list(used_c)] = big
        i, j = np.unravel_index(np.argmin(masked), (M, N))
        if i in used_r or j in used_c: break
        pairs.append((i, j))
        used_r.add(i); used_c.add(j)
    return pairs

def hungarian_match(cost):
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    except Exception:
        return greedy_match(cost)
