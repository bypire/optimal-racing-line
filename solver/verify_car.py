"""
Checks for the forward car model and RK4 integrator.

  (1) Coasting (no grip used): straight line, speed and kinetic energy constant.
  (2) Steady cornering at the friction limit: with full lateral grip a_n = mu*g
      the car traces a circle of radius R = v^2/(mu g) -- the maximum-cornering
      relation v_max = sqrt(mu g R).
  (3) RK4 is 4th-order: the circle-radius error falls like dt^4 under mesh
      refinement.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from car import Car, G


def test_coasting():
    car = Car(mu=1.3)
    v0 = 30.0
    traj, _ = car.simulate([0, 0, 0.0, v0], lambda t, q: [0.0, 0.0],
                            dt=0.01, n_steps=500)
    dv = np.abs(traj[:, 3] - v0).max()
    dy = np.abs(traj[:, 1]).max()               # should stay on the x-axis
    print(f"  [coasting]  max |v-v0| = {dv:.2e} m/s,  max |y| = {dy:.2e} m  "
          f"-> {'OK' if dv < 1e-9 and dy < 1e-9 else 'FAIL'}")
    assert dv < 1e-9 and dy < 1e-9, "coasting should conserve speed and stay on axis"


def circle_radius(traj):
    """Fit the radius of a (closed) circular path by least squares."""
    x, y = traj[:, 0], traj[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = cx / 2, cy / 2
    R = np.sqrt(c + cx**2 + cy**2)
    return R, (cx, cy)


def test_corner(mu=1.3, v=30.0, dt=0.002):
    car = Car(mu=mu)
    a_n = car.a_max                              # full lateral grip
    R_exact = v**2 / (mu * G)                    # v^2 = mu g R
    omega = a_n / v                              # heading rate
    T = 2 * np.pi / omega                         # one full lap
    n = int(round(T / dt))
    traj, ctrl = car.simulate([0, 0, 0.0, v], lambda t, q: [0.0, a_n],
                              dt=dt, n_steps=n)
    R_num, _ = circle_radius(traj)
    err = abs(R_num - R_exact) / R_exact * 100
    print(f"  [corner]    v={v} m/s, mu={mu}: R_exact={R_exact:.3f} m, "
          f"R_num={R_num:.3f} m  ({err:+.3f}%)   grip used={car.grip_usage([0,a_n]):.2f}")
    assert err < 1.0, "steady-corner radius must match v^2 = mu g R"
    return traj, R_exact


def test_rk4_order(v=30.0, mu=1.3):
    """Radius error vs dt should scale ~ dt^4 (RK4)."""
    car = Car(mu=mu)
    a_n = car.a_max
    R_exact = v**2 / (mu * G)
    omega = a_n / v
    Tlap = 2 * np.pi / omega
    dts, errs = [], []
    for dt in [0.02, 0.01, 0.005, 0.0025]:
        n = int(round(Tlap / dt))
        traj, _ = car.simulate([0, 0, 0.0, v], lambda t, q: [0.0, a_n], dt, n)
        R_num, _ = circle_radius(traj)
        dts.append(dt); errs.append(abs(R_num - R_exact))
    dts, errs = np.array(dts), np.array(errs)
    # slope on log-log
    p = np.polyfit(np.log(dts), np.log(np.maximum(errs, 1e-16)), 1)[0]
    print(f"  [rk4 order] radius-error slope d(log e)/d(log dt) = {p:.2f}  "
          f"(expect ~4 for RK4)")
    assert p > 3.5, "RK4 should show ~4th-order convergence"
    return dts, errs, p


def main():
    print("M1 -- forward car model verification")
    test_coasting()
    traj, R = test_corner()
    dts, errs, p = test_rk4_order()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].plot(traj[:, 0], traj[:, 1], lw=2)
    ax[0].plot(traj[0, 0], traj[0, 1], "go", label="start")
    ax[0].set_aspect("equal"); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"steady corner at the grip limit\n"
                    f"R = v$^2$/($\\mu$g) = {R:.1f} m (traced, not imposed)")
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]"); ax[0].legend()

    ax[1].loglog(dts, errs, "o-", lw=2, label="radius error")
    ax[1].loglog(dts, errs[0] * (dts / dts[0])**4, "k--", label="$\\propto dt^4$")
    ax[1].set_xlabel("time step dt [s]"); ax[1].set_ylabel("radius error [m]")
    ax[1].set_title(f"RK4 is 4th order (slope {p:.2f})")
    ax[1].legend(); ax[1].grid(alpha=0.3, which="both")

    fig.suptitle("M1: point-mass car + RK4, verified", fontweight="bold")
    fig.tight_layout()
    fig.savefig("output/m1_car_verify.png", dpi=130)
    print("  saved output/m1_car_verify.png")


if __name__ == "__main__":
    main()
