"""Alpha-Rank validation fixtures."""
from itertools import product

import numpy as np

from rq2.psro_lite import alpha_rank


def _run(tensors, m, alpha):
    shapes = tensors[0].shape
    profiles = list(product(*[range(n) for n in shapes]))
    U = np.array([[tensor[p] for tensor in tensors] for p in profiles], float)
    return alpha_rank(profiles, U, shapes, alpha, m=m, return_transition=True)


def _check(label, value, limit):
    ok = value < limit
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {value:.3e}")
    if not ok:
        raise AssertionError(label)


def main():
    rps = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], float)
    pi, C = _run([rps, -rps], m=50, alpha=1.0)
    _check("RPS uniformity", np.abs(pi - 1 / 9).max(), 1e-8)

    dominance = np.array([[0.0, 1.0], [2.0, 3.0]])
    pi, C = _run([dominance, dominance.T], m=50, alpha=20.0)
    _check("dominance residual", 1.0 - pi.reshape(2, 2)[1, 1], 1e-8)

    rng = np.random.default_rng(0)
    tensors = []
    for player in range(5):
        tensor = rng.normal(size=(2,) * 5)
        index = [slice(None)] * 5
        index[player] = 1
        tensor[tuple(index)] += 5.0
        tensors.append(tensor)
    pi, C = _run(tensors, m=50, alpha=20.0)
    _check("five-population dominance residual",
           1.0 - pi.reshape((2,) * 5)[1, 1, 1, 1, 1], 1e-8)

    _check("transition rows", np.abs(C.sum(1) - 1.0).max(), 1e-12)
    _check("stationary residual", np.abs(pi @ C - pi).max(), 1e-8)
    _check("probability mass", abs(pi.sum() - 1.0) + max(0.0, -pi.min()), 1e-12)


if __name__ == "__main__":
    main()
