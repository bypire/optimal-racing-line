"""
THE LAP -- solve the minimum-time racing line on a full closed circuit.
================================================================================
Headline deliverable: a lap time on a named circuit, the optimal racing line
(an OUTPUT of the optimiser), the g-g diagram proving the car is at the grip
limit, and the lap-time gain of the racing line over the geometric centreline.

Mesh continuation: solve coarse, then warm-start successively finer meshes -- a
standard trajectory-optimisation technique that also makes the solve robust.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from track import Track
from ocp import MinLapOCP, grip_usage


def _interp_solution(res, new_ocp):
    """Warm-start: interpolate a coarse solution onto a finer s-grid (periodic)."""
    s_old, X, U = res["s"], res["X"], res["U"]
    L = new_ocp.track.L
    per = L if new_ocp.closed else None
    sn = new_ocp.sg
    Xn = np.column_stack([np.interp(sn, s_old, X[:, j], period=per) for j in range(3)])
    Un = np.column_stack([np.interp(sn, s_old, U[:, j], period=per) for j in range(2)])
    return np.concatenate([Xn.reshape(-1), Un.reshape(-1)])


def solve_lap(track, mu=1.5, vmax=95.0, N_levels=(120, 200), n_cap=None,
              verbose=True):
    """Mesh-continuation solve. n_cap forces |n|<=n_cap (use a tiny value to
    pin the car to the centreline for the comparison baseline)."""
    res = None
    for N in N_levels:
        ocp = MinLapOCP(track, mu=mu, N=N, vmax=vmax)
        if n_cap is not None:                    # override width -> centreline
            ocp.wid = np.minimum(ocp.wid, n_cap)
            ocp.Sx[0] = max(n_cap, 0.5)
            ocp.S = np.concatenate([np.tile(ocp.Sx, N + 1), np.tile(ocp.Su, N + 1)])
        z0 = _interp_solution(res, ocp) if res is not None else None
        if verbose:
            tag = "centreline" if n_cap is not None else "racing line"
            print(f"  [{tag}] N={N} ...")
        res = ocp.solve(z0=z0, maxiter=500, verbose=verbose)
    return res


def lap_summary(res, mu):
    ocp = res["ocp"]
    g = grip_usage(res["U"], ocp.a_max)
    v = res["X"][:, 2]
    sat = np.mean(g > 0.97) * 100
    print(f"  lap time      = {res['T']:.3f} s")
    print(f"  top speed     = {v.max()*3.6:.1f} km/h   min speed = {v.min()*3.6:.1f} km/h")
    print(f"  grip at limit = {sat:.0f}% of the lap   (max grip {g.max():.3f})")
    print(f"  max defect    = {res['maxdef']:.1e}")


def plot_lap(res_rl, res_cl, track, mu, fname="output/the_lap.png"):
    ocp = res_rl["ocp"]
    s = res_rl["s"]; n = res_rl["X"][:, 0]; v = res_rl["X"][:, 2]
    x, y = track.to_xy(s, n)
    xc, yc = track.to_xy(res_cl["s"], res_cl["X"][:, 0])
    (xl, yl), (xr, yr) = track.edges()

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=0.3, wspace=0.3)

    # --- the racing line, speed-coloured (the money shot) ------------------
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(xl, yl, color="0.25", lw=1); ax.plot(xr, yr, color="0.25", lw=1)
    ax.plot(xc, yc, "--", color="0.6", lw=1, label="centreline")
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="turbo",
                        norm=plt.Normalize(v.min()*3.6, v.max()*3.6))
    lc.set_array(v[:-1]*3.6); lc.set_linewidth(3.5); ax.add_collection(lc)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{track.name} -- optimal racing line  (lap {res_rl['T']:.2f} s)",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    cb = fig.colorbar(lc, ax=ax, shrink=0.7, pad=0.01); cb.set_label("speed [km/h]")

    # --- g-g diagram -------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    U = res_rl["U"]
    th = np.linspace(0, 2*np.pi, 100)
    ax2.plot(ocp.a_max*np.cos(th)/9.81, ocp.a_max*np.sin(th)/9.81, "r-", lw=1.5,
             label="grip limit")
    sc = ax2.scatter(U[:, 1]/9.81, U[:, 0]/9.81, c=v*3.6, cmap="turbo", s=10)
    ax2.set_aspect("equal"); ax2.axhline(0, color="0.7", lw=.5); ax2.axvline(0, color="0.7", lw=.5)
    ax2.set_xlabel("lateral g"); ax2.set_ylabel("long. g (accel +, brake -)")
    ax2.set_title("g-g diagram"); ax2.legend(fontsize=8)

    # --- speed trace -------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(s, v*3.6, lw=1.8, color="navy")
    ax3.fill_between(s, 0, v*3.6, alpha=0.08, color="navy")
    ax3.set_xlabel("distance along lap  s [m]"); ax3.set_ylabel("speed [km/h]")
    ax3.set_xlim(0, s[-1]); ax3.grid(alpha=0.3)
    ax3.set_title("speed profile")

    gain = res_cl["T"] - res_rl["T"]
    fig.suptitle(f"Minimum-lap-time optimal control  |  racing line {res_rl['T']:.2f} s "
                 f"vs centreline {res_cl['T']:.2f} s  ->  {gain:.2f} s "
                 f"({gain/res_cl['T']*100:.1f}%) faster",
                 fontweight="bold", y=0.98)
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    print(f"  saved {fname}")


def export_web(res, track, fname="output/lap_data.js"):
    import json
    s = res["s"]; n = res["X"][:, 0]; v = res["X"][:, 2]
    x, y = track.to_xy(s, n)
    (xl, yl), (xr, yr) = track.edges()
    # cumulative time along the lap (for real-time animation)
    g = res["ocp"]._dtds_all(res["X"])
    t = np.concatenate([[0], np.cumsum(0.5*np.diff(s)*(g[:-1]+g[1:]))])
    U = res["U"]
    data = dict(
        name=track.name, T=round(float(res["T"]), 3),
        x=[round(float(a), 1) for a in x], y=[round(float(a), 1) for a in y],
        v=[round(float(a), 2) for a in v], t=[round(float(a), 3) for a in t],
        at=[round(float(a), 2) for a in U[:, 0]],
        an=[round(float(a), 2) for a in U[:, 1]],
        amax=round(float(res["ocp"].a_max), 3),
        xl=[round(float(a), 1) for a in xl], yl=[round(float(a), 1) for a in yl],
        xr=[round(float(a), 1) for a in xr], yr=[round(float(a), 1) for a in yr])
    with open(fname, "w") as f:
        f.write("const LAP = "); json.dump(data, f, separators=(",", ":")); f.write(";\n")
    import os
    print(f"  saved {fname} ({os.path.getsize(fname)/1024:.0f} kB)")


def main():
    mu = 1.5
    track = Track.grand_circuit(width=9.0, ds=2.0)
    print(f"=== {track.name}: L={track.L:.0f} m, mu={mu} ===")
    print("Solving racing line (mesh continuation)...")
    rl = solve_lap(track, mu=mu, N_levels=(120, 200))
    lap_summary(rl, mu)
    print("Solving centreline baseline...")
    cl = solve_lap(track, mu=mu, N_levels=(120, 200), n_cap=0.1, verbose=False)
    print(f"  centreline lap = {cl['T']:.3f} s")
    gain = cl["T"] - rl["T"]
    print(f"\n  >>> racing line is {gain:.2f} s ({gain/cl['T']*100:.1f}%) faster "
          f"than the centreline <<<")
    plot_lap(rl, cl, track, mu)
    export_web(rl, track)


if __name__ == "__main__":
    main()
