"""
Point-mass race car -- planar dynamics on the friction (traction) circle.

The forward model: the car you can integrate. The optimiser (ocp.py) later
solves for the controls that take this car round a track in minimum time; this
module just builds the physics.

Model: a point mass moving in the plane. State

    q = [x, y, theta, v]      position, heading, speed

driven by two accelerations the tyres can deliver:

    a_t  -- longitudinal (throttle / brake), along the heading
    a_n  -- lateral (cornering), perpendicular to the heading

Equations of motion:

    x'     = v cos(theta)
    y'     = v sin(theta)
    theta' = a_n / v          (lateral accel curves the path; path curvature = a_n/v^2)
    v'     = a_t

The tyres can only deliver so much grip. The KEY physics -- the thing that makes
a racing line exist -- is the FRICTION CIRCLE: the combined acceleration cannot
exceed mu*g.

    a_t^2 + a_n^2  <=  (mu g)^2

Brake hard and you have no grip left to turn; turn hard and you cannot also
accelerate. The optimal lap is a constant negotiation of that budget.

numpy only. Time integration: classical RK4.
"""

import numpy as np

G = 9.81


class Car:
    """Point-mass car with a friction-circle grip limit."""

    def __init__(self, mu=1.3, g=G):
        self.mu = mu
        self.g = g
        self.a_max = mu * g            # radius of the friction circle [m/s^2]

    # --- dynamics ----------------------------------------------------------
    def deriv(self, q, u):
        """Time derivative q' = f(q, u).  q=[x,y,theta,v], u=[a_t,a_n]."""
        x, y, theta, v = q
        a_t, a_n = u
        v_safe = max(v, 1e-6)          # avoid divide-by-zero at standstill
        return np.array([
            v * np.cos(theta),
            v * np.sin(theta),
            a_n / v_safe,
            a_t,
        ])

    def grip_usage(self, u):
        """Fraction of the friction circle used: |a| / (mu g). 1.0 = at the limit."""
        return np.hypot(u[0], u[1]) / self.a_max

    # --- integrator --------------------------------------------------------
    def rk4_step(self, q, u, dt):
        """One classical Runge-Kutta-4 step (controls held constant over dt)."""
        k1 = self.deriv(q, u)
        k2 = self.deriv(q + 0.5 * dt * k1, u)
        k3 = self.deriv(q + 0.5 * dt * k2, u)
        k4 = self.deriv(q + dt * k3, u)
        return q + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def simulate(self, q0, control_fn, dt, n_steps):
        """Roll the car forward. control_fn(t, q) -> [a_t, a_n]."""
        q = np.array(q0, float)
        traj = np.empty((n_steps + 1, 4))
        ctrl = np.empty((n_steps, 2))
        traj[0] = q
        for k in range(n_steps):
            t = k * dt
            u = np.asarray(control_fn(t, q), float)
            q = self.rk4_step(q, u, dt)
            traj[k + 1] = q
            ctrl[k] = u
        return traj, ctrl
