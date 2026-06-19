"""
Checks for the optimal-control result (not just the forward car).

Tight defects only show the NLP satisfied its own constraints, so here the
output itself is checked:

  (1) Closed-form lap time on a skidpad (constant-radius circle): the pure
      cornering limit gives T = 2*pi*sqrt(R/(mu g)). The solver should hit it.
  (2) Mesh convergence: lap time converges as the collocation mesh is refined
      (trapezoidal -> error ~ ds^2).
  (3) Friction-circle saturation: the optimal solution rides the grip limit --
      the fraction of the lap at |a| = mu g should be high.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from track import Track
from ocp import MinLapOCP, grip_usage

G = 9.81


def skidpad_closed_form(mu=1.4, R=50.0):
    T_exact = 2 * np.pi * np.sqrt(R / (mu * G))
    v_exact = np.sqrt(mu * G * R)
    print(f"(1) SKIDPAD closed-form check  (R={R}, mu={mu})")
    print(f"    exact: v=sqrt(mu g R)={v_exact:.3f} m/s, "
          f"T=2*pi*sqrt(R/mu g)={T_exact:.4f} s")
    # thin track -> car is pinned to radius R, so the closed form applies
    track = Track.skidpad(R=R, width=0.3, ds=1.0)
    errs = []
    for N in (80, 160, 320):
        ocp = MinLapOCP(track, mu=mu, N=N, vmax=60)
        r = ocp.solve(maxiter=300, verbose=False)
        e = (r["T"] - T_exact) / T_exact * 100
        errs.append((N, r["T"], e))
        print(f"    N={N:3d}: T={r['T']:.4f} s  ({e:+.3f}%)  v={r['X'][:,2].mean():.3f}")
    return T_exact, errs


def mesh_convergence(mu=1.4):
    print("\n(2) MESH CONVERGENCE on the oval (lap time vs N)")
    track = Track.oval(straight=100, R=45, width=8)
    Ns = [60, 100, 160, 240]
    Ts = []
    res = None
    for N in Ns:
        ocp = MinLapOCP(track, mu=mu, N=N, vmax=90)
        z0 = None
        if res is not None:
            sn = ocp.sg; per = track.L
            Xn = np.column_stack([np.interp(sn, res["s"], res["X"][:, j], period=per)
                                  for j in range(3)])
            Un = np.column_stack([np.interp(sn, res["s"], res["U"][:, j], period=per)
                                  for j in range(2)])
            z0 = np.concatenate([Xn.reshape(-1), Un.reshape(-1)])
        res = ocp.solve(z0=z0, maxiter=400, verbose=False)
        Ts.append(res["T"])
        print(f"    N={N:3d}: T={res['T']:.4f} s")
    Ts = np.array(Ts)
    # Richardson-ish: differences should shrink ~ (ds)^2 = (L/N)^2
    print(f"    successive |dT|: {np.abs(np.diff(Ts))}")
    return Ns, Ts


def friction_saturation(mu=1.4):
    print("\n(3) FRICTION-CIRCLE SATURATION on the oval")
    track = Track.oval(straight=100, R=45, width=8)
    ocp = MinLapOCP(track, mu=mu, N=200, vmax=90)
    r = ocp.solve(maxiter=400, verbose=False)
    g = grip_usage(r["U"], ocp.a_max)
    print(f"    max grip usage = {g.max():.3f}  (1.0 = at the limit)")
    print(f"    fraction of lap at >=97% grip = {np.mean(g > 0.97)*100:.0f}%")
    return r, g


def main():
    T_exact, errs = skidpad_closed_form()
    r, g = friction_saturation()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    Ns = np.array([e[0] for e in errs]); aerr = np.abs([e[2] for e in errs])
    ax[0].loglog(Ns, aerr, "o-", lw=2, label="skidpad lap-time error")
    ax[0].loglog(Ns, aerr[0]*(Ns[0]/Ns)**2, "k--", label="$\\propto N^{-2}$ (trapezoidal)")
    ax[0].set_xlabel("collocation nodes N"); ax[0].set_ylabel("|lap-time error| [%]")
    ax[0].set_title("mesh convergence to the closed-form lap time")
    ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
    s = r["ocp"].sg
    ax[1].plot(s, g, lw=1.5, color="crimson")
    ax[1].axhline(1.0, color="0.5", ls="--", label="grip limit")
    ax[1].set_xlabel("distance s [m]"); ax[1].set_ylabel("|a| / (mu g)")
    ax[1].set_title("the optimum rides the grip limit"); ax[1].legend()
    ax[1].grid(alpha=0.3); ax[1].set_ylim(0, 1.1)
    fig.suptitle("Verification of the optimal-control result", fontweight="bold")
    fig.tight_layout()
    fig.savefig("output/verify_lap.png", dpi=130)
    print("\n  saved output/verify_lap.png")


if __name__ == "__main__":
    main()
