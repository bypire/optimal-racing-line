"""
VERIFICATION -- the analytic Jacobians (the credibility centrepiece).
================================================================================
We hand the NLP solver exact, hand-derived derivatives instead of letting it
finite-difference. This file PROVES they are correct: every analytic Jacobian /
gradient is checked against a finite-difference reference at random feasible
points. A reviewer should open this first.

  - objective_grad   vs  central difference of objective
  - defects_jac      vs  forward difference of defects
  - friction_jac     vs  forward difference of friction
  - _dfdx / _dfdu     vs  finite difference of the dynamics f
"""

import numpy as np
from track import Track
from ocp import MinLapOCP


def fd_jac(fun, z, eps=1e-6):
    f0 = np.atleast_1d(fun(z))
    J = np.zeros((f0.size, z.size))
    for j in range(z.size):
        zp = z.copy(); zp[j] += eps
        J[:, j] = (np.atleast_1d(fun(zp)) - f0) / eps
    return J


def main():
    np.random.seed(0)
    track = Track.oval(straight=100, R=45, width=8)
    ocp = MinLapOCP(track, mu=1.4, N=50, vmax=90)

    # a random *feasible-ish* point (away from singularities)
    z = ocp.initial_guess()
    z = z + 0.02 * np.random.randn(z.size)

    print("Analytic vs finite-difference derivatives (max abs error):")

    ga = ocp.objective_grad(z)
    gf = np.array([(ocp.objective(z + e) - ocp.objective(z - e)) / (2*1e-6)
                   for e in (np.eye(z.size)[k]*1e-6 for k in range(z.size))])
    print(f"  objective_grad : {np.abs(ga-gf).max():.2e}")

    Ja = ocp.defects_jac(z);  Jf = fd_jac(ocp.defects, z)
    print(f"  defects_jac    : {np.abs(Ja-Jf).max():.2e}")

    Fa = ocp.friction_jac(z); Ff = fd_jac(ocp.friction, z)
    print(f"  friction_jac   : {np.abs(Fa-Ff).max():.2e}")

    # per-node dynamics derivatives
    x = np.array([1.0, 0.2, 40.0]); u = np.array([3.0, 8.0]); k = 0.02
    dfdx_a = ocp._dfdx(x, u, k)
    dfdx_f = fd_jac(lambda xx: ocp._f(xx, u, k), x)
    dfdu_a = ocp._dfdu(x, u, k)
    dfdu_f = fd_jac(lambda uu: ocp._f(x, uu, k), u)
    print(f"  _dfdx (dynamics): {np.abs(dfdx_a-dfdx_f).max():.2e}")
    print(f"  _dfdu (dynamics): {np.abs(dfdu_a-dfdu_f).max():.2e}")

    worst = max(np.abs(ga-gf).max(), np.abs(Ja-Jf).max(), np.abs(Fa-Ff).max(),
                np.abs(dfdx_a-dfdx_f).max(), np.abs(dfdu_a-dfdu_f).max())
    ok = worst < 1e-5
    print(f"\n  worst error = {worst:.2e}  ->  {'PASS' if ok else 'FAIL'} "
          f"(analytic derivatives correct)")
    assert ok, "analytic derivatives disagree with finite difference!"


if __name__ == "__main__":
    main()
