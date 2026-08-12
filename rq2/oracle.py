"""Shared best-response optimiser."""
import numpy as np
import cma

DEFAULT_SEED = 42
DEFAULT_SIGMA0 = 12.0
DEFAULT_RESTARTS = 3

# Fixed portfolio — don't reshuffle if you want the Table B.1 numbers back.
WARM_STARTS = [np.array(v, float) for v in
               [(0, 0), (-40, 0), (-15, -15), (-10, -20), (+20, 0), (+15, +15),
                (-25, -5), (+25, -10), (-5, +20)]]

# Extra exit-shrink corners for obstruction only (EU shrink at (-40,-45) is
# otherwise seed-dependent).
EXIT_SHRINK_STARTS = [np.array(v, float) for v in
                      [(-40, -45), (-30, -35), (-35, -40)]]


def cma_minimize(obj, budget, sigma0=DEFAULT_SIGMA0, seed=DEFAULT_SEED,
                 n_restarts=DEFAULT_RESTARTS, x0=None, warm_starts=None):
    """Minimise obj(x) over R^2; restarted CMA + global best-ever tracking.

    The unbounded domain is part of the submitted experiment implementation.
    It differs from the [-60, 60]^2 report box stated in the dissertation; see
    the README's report-domain qualification before interpreting results.
    """
    x0 = np.zeros(2, float) if x0 is None else np.asarray(x0, float)
    best_f, best_x = float(obj(x0)), x0.copy()
    for xs in warm_starts or ():
        xs = np.asarray(xs, float)
        fv = float(obj(xs))
        if fv < best_f:
            best_f, best_x = fv, xs.copy()
    per = max(1, budget // n_restarts)
    for k in range(n_restarts):
        es = cma.CMAEvolutionStrategy(
            list(x0), sigma0 * (1.5 ** k),
            {"seed": int(seed + k), "verbose": -9, "maxfevals": per},
        )
        while not es.stop():
            xs = es.ask()
            fs = [float(obj(x)) for x in xs]
            es.tell(xs, fs)
            j = int(np.argmin(fs))
            if fs[j] < best_f:
                best_f, best_x = fs[j], np.asarray(xs[j], float)
    return best_f, best_x
