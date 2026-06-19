"""
DESIGN TRADE STUDIES -- the optimiser as an engineering tool, not a pretty line.
================================================================================
Two things a computational-optimisation professor actually cares about:

  (1) the AERO performance envelope: with downforce the grip limit grows with
      speed, a_grip(v) = mu(g + k_down v^2), so the g-g "circle" becomes a
      speed-dependent ENVELOPE -- high-speed corners pull more g than slow ones.
      This is the defining feature of a real race car, and it falls straight out
      of the model.

  (2) LAP-TIME SENSITIVITY: re-solve the lap while sweeping grip, power and
      downforce -> design trade curves (wet-vs-dry, power's diminishing returns,
      what downforce buys). Warm-started across parameters so the whole study is
      cheap.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from track import Track
from ocp import MinLapOCP, grip_usage

G = 9.81
# an F1-ish car (coefficients k = 0.5*rho*C*A/m, units 1/m; power = P/m)
BASE = dict(mu=1.4, drag=0.0011, downforce=0.0032, power=720.0, vmax=110.0)


def solve(track, N=150, z0=None, **kw):
    p = dict(BASE, **kw)
    ocp = MinLapOCP(track, N=N, **p)
    return ocp.solve(z0=z0, maxiter=400, verbose=False)


def aero_envelope(track):
    """g-g of the aero car: points ride a grip limit that grows with speed."""
    r = solve(track, N=180)
    ocp = r["ocp"]; U = r["U"]; v = r["X"][:, 2]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8))

    sc = ax[0].scatter(U[:, 1]/G, U[:, 0]/G, c=v*3.6, cmap="turbo", s=14)
    th = np.linspace(0, 2*np.pi, 100)
    for vv, col in [(40, "#888"), (80, "#d09000"), (110, "#c01000")]:
        ag = ocp.a_grip(vv) / G
        ax[0].plot(ag*np.cos(th), ag*np.sin(th), color=col, lw=1.2,
                   label=f"grip limit @ {vv*3.6:.0f} km/h")
    ax[0].set_aspect("equal"); ax[0].axhline(0, color="0.8", lw=.5); ax[0].axvline(0, color="0.8", lw=.5)
    ax[0].set_xlabel("lateral g"); ax[0].set_ylabel("long. g")
    ax[0].set_title("aero g-g: the envelope GROWS with speed\n(downforce)")
    ax[0].legend(fontsize=8, loc="upper right")
    cb = fig.colorbar(sc, ax=ax[0], shrink=0.8); cb.set_label("speed [km/h]")

    # max lateral g vs speed: the envelope
    vv = np.linspace(20, 120, 100)
    ax[1].plot(vv*3.6, ocp.a_grip(vv)/G, lw=2.5, color="navy",
               label="grip limit a_grip(v)/g")
    ax[1].axhline(BASE["mu"], color="0.5", ls="--",
                  label=f"no-downforce limit (mu={BASE['mu']})")
    ax[1].scatter(v*3.6, np.hypot(U[:, 0], U[:, 1])/G, s=8, c="crimson",
                  alpha=0.5, label="achieved |a|/g")
    ax[1].set_xlabel("speed [km/h]"); ax[1].set_ylabel("max cornering g")
    ax[1].set_title("more speed -> more grip (downforce)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle(f"Aerodynamic performance envelope  (lap {r['T']:.2f} s)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig("output/aero_envelope.png", dpi=130)
    print(f"  saved output/aero_envelope.png  (aero lap {r['T']:.2f} s)")
    return r


def sensitivity(track):
    """Lap time vs grip / power / downforce, warm-started across parameters."""
    base = solve(track, N=150)
    z0 = base["z"]
    sweeps = {
        "grip mu": ("mu", [1.0, 1.15, 1.3, 1.45, 1.6, 1.75]),
        "power P/m [m^2/s^3]": ("power", [350, 500, 650, 800, 950, 1100]),
        "downforce k_down [1/m]": ("downforce", [0.0, 0.0015, 0.003, 0.0045, 0.006, 0.0075]),
    }
    results = {}
    for label, (key, vals) in sweeps.items():
        Ts, z = [], z0
        for val in vals:
            r = solve(track, N=150, z0=z, **{key: val})
            Ts.append(r["T"]); z = r["z"] if r["success"] else z0
        results[label] = (vals, Ts)
        print(f"  {label}: " + " ".join(f"{t:.2f}" for t in Ts))

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, (label, (vals, Ts)) in zip(ax, results.items()):
        a.plot(vals, Ts, "o-", lw=2, ms=8, color="crimson")
        a.set_xlabel(label); a.set_ylabel("lap time [s]"); a.grid(alpha=0.3)
        a.set_title(f"lap time vs {label.split()[0]}")
    fig.suptitle("Lap-time sensitivity -- the optimiser as a design trade tool",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig("output/sensitivity.png", dpi=130)
    print("  saved output/sensitivity.png")
    return results


def main():
    track = Track.grand_circuit(width=9.0, ds=2.0)
    print("=== Aero performance envelope ===")
    aero_envelope(track)
    print("=== Lap-time sensitivity sweeps ===")
    sensitivity(track)


if __name__ == "__main__":
    main()
