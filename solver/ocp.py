"""
Minimum-lap-time optimal control by direct collocation.

We solve for the controls that take the car round the track in least time: a
trajectory-optimisation problem transcribed to a nonlinear program (NLP) by
direct collocation and handed to a gradient-based solver.

Formulation -- the standard minimum-lap-time trick: use ARC LENGTH s along the
track centreline as the independent variable (so the finish is a fixed boundary
s=L, not a free final time). States as functions of s:

    n    lateral offset from the centreline
    xi   heading of the car relative to the track tangent
    v    speed

Controls: longitudinal accel a_t and lateral accel a_n, limited by the FRICTION
CIRCLE  a_t^2 + a_n^2 <= (mu g)^2.

Dynamics (d/ds), with curvature k = kappa(s) and  D = 1 - n k:

    dn/ds  = D tan(xi)
    dxi/ds = D a_n / (v^2 cos xi)  -  k
    dv/ds  = D a_t / (v cos xi)
    dt/ds  = D / (v cos xi)          <-  the lap-time integrand we minimise

Transcription: trapezoidal collocation on N intervals -- states+controls at the
nodes are the unknowns; the dynamics become equality "defect" constraints
between adjacent nodes; the objective is the trapezoidal sum of dt/ds. Solved
with scipy SLSQP. The racing line is an OUTPUT, not something we draw.
"""

import numpy as np
from scipy.optimize import minimize

G = 9.81


class MinLapOCP:
    def __init__(self, track, mu=1.3, N=60, vmax=85.0, vmin=3.0,
                 xi_max=1.2, closed=None, drag=0.0, downforce=0.0, power=None):
        """drag, downforce: aero coefficients k = 0.5*rho*Cd_or_Cl*A/m (units
        1/m) so a_drag = drag*v^2 and the grip limit grows as
        a_grip(v) = mu*(g + downforce*v^2). power = max power-to-weight P/m
        (m^2/s^3); drive accel is then capped at power/v. All default OFF, so
        the baseline is the pure point-mass friction circle."""
        self.track = track
        self.mu = mu
        self.a_max = mu * G
        self.drag = drag
        self.c_dn = mu * downforce          # grip grows by mu*downforce*v^2
        self.P_lim = power
        self.aero = (drag != 0.0 or downforce != 0.0 or power is not None)
        self.N = N
        self.vmax = vmax
        self.vmin = vmin
        self.xi_max = xi_max
        self.closed = track.closed if closed is None else closed

        # uniform s-grid of N+1 nodes over the track
        self.sg = np.linspace(0.0, track.L, N + 1)
        self.ds = np.diff(self.sg)
        self.kap = track.kappa(self.sg)
        self.wid = track.width(self.sg)
        self.nv = 3                      # states per node
        self.nu = 2                      # controls per node

        # characteristic scales -> dimensionless decision variables so the NLP
        # is well conditioned (states/controls span 2 orders of magnitude).
        w_char = max(np.mean(self.wid), 1.0)
        self.Sx = np.array([w_char, 1.0, vmax])      # [n, xi, v]
        self.Su = np.array([self.a_max, self.a_max])  # [a_t, a_n]
        M = N + 1
        self.S = np.concatenate([np.tile(self.Sx, M), np.tile(self.Su, M)])
        # row scaling for the defect constraints (each row has its state's units)
        rs = np.tile(self.Sx, N)
        if self.closed:
            rs = np.concatenate([rs, self.Sx])
        self.row_scale = rs

    # --- quasi-steady-state friction-limited speed profile (warm start) ----
    def qss_speed_profile(self):
        """The classic g-g-limited speed profile if the car stayed on the
        centreline: cornering limit v=sqrt(a_max/|kappa|), then a forward
        (accel) pass and a backward (braking) pass. A shippable sub-method and
        an excellent initial guess -- the NLP then only has to find the LINE."""
        k = np.abs(self.kap)
        v_corner = np.where(k > 1e-6, np.sqrt(self.a_max / np.maximum(k, 1e-9)),
                            self.vmax)
        v = np.minimum(v_corner, self.vmax)
        # forward pass: limited acceleration
        for i in range(self.N):
            vmax_acc = np.sqrt(v[i]**2 + 2 * self.a_max * self.ds[i])
            v[i + 1] = min(v[i + 1], vmax_acc)
        # backward pass: limited braking
        for i in range(self.N - 1, -1, -1):
            vmax_brake = np.sqrt(v[i + 1]**2 + 2 * self.a_max * self.ds[i])
            v[i] = min(v[i], vmax_brake)
        if self.closed:                  # one more sweep for periodicity
            for i in range(self.N):
                v[i + 1] = min(v[i + 1], np.sqrt(v[i]**2 + 2*self.a_max*self.ds[i]))
        return np.clip(v, self.vmin, self.vmax)

    # --- (un)packing the decision vector ----------------------------------
    def unpack(self, z):
        M = self.N + 1
        X = z[:M * self.nv].reshape(M, self.nv)
        U = z[M * self.nv:].reshape(M, self.nu)
        return X, U

    # --- per-node dynamics and time integrand ------------------------------
    def _f(self, x, u, k):
        n, xi, v = x
        at, an = u
        D = 1.0 - n * k
        c = np.cos(xi)
        return np.array([
            D * np.tan(xi),
            D * an / (v * v * c) - k,
            D * (at - self.drag * v * v) / (v * c),     # drag decel in long. dyn
        ])

    def _dtds(self, x, k):
        n, xi, v = x
        return (1.0 - n * k) / (v * np.cos(xi))

    # --- analytic derivatives (so the NLP solver gets exact Jacobians) ------
    def _dfdx(self, x, u, k):
        """d f / d[n,xi,v]  (3x3)."""
        n, xi, v = x; at, an = u
        D = 1.0 - n * k; c = np.cos(xi); t = np.tan(xi)
        g2 = at - self.drag * v * v
        J = np.zeros((3, 3))
        J[0, 0] = -k * t;            J[0, 1] = D / c**2
        J[1, 0] = -k * an / (v*v*c); J[1, 1] = D * an * t / (v*v*c); J[1, 2] = -2*D*an/(v**3*c)
        J[2, 0] = -k * g2 / (v*c);   J[2, 1] = D * g2 * t / (v*c)
        J[2, 2] = -D * (at + self.drag * v * v) / (v*v*c)
        return J

    def _dfdu(self, x, u, k):
        """d f / d[a_t,a_n]  (3x2)."""
        n, xi, v = x
        D = 1.0 - n * k; c = np.cos(xi)
        J = np.zeros((3, 2))
        J[1, 1] = D / (v*v*c)        # df1/dan
        J[2, 0] = D / (v*c)          # df2/dat
        return J

    def _dgdx(self, x, k):
        """d(dt/ds) / d[n,xi,v]  (3,)."""
        n, xi, v = x
        D = 1.0 - n * k; c = np.cos(xi); t = np.tan(xi)
        return np.array([-k/(v*c), D*t/(v*c), -D/(v*v*c)])

    # --- vectorised dynamics over all nodes (the solver hot path) ----------
    def _f_all(self, X, U):
        n, xi, v = X[:, 0], X[:, 1], X[:, 2]
        at, an = U[:, 0], U[:, 1]
        k = self.kap
        D = 1.0 - n * k; c = np.cos(xi); t = np.tan(xi)
        f = np.empty_like(X)
        f[:, 0] = D * t
        f[:, 1] = D * an / (v * v * c) - k
        f[:, 2] = D * (at - self.drag * v * v) / (v * c)
        return f

    def _dtds_all(self, X):
        n, xi, v = X[:, 0], X[:, 1], X[:, 2]
        return (1.0 - n * self.kap) / (v * np.cos(xi))

    # --- objective: total lap time (trapezoidal) ---------------------------
    def objective(self, z):
        X, _ = self.unpack(z)
        g = self._dtds_all(X)
        return np.sum(0.5 * self.ds * (g[:-1] + g[1:]))

    # --- equality constraints: trapezoidal defects (+ periodicity) ---------
    def defects(self, z):
        X, U = self.unpack(z)
        f = self._f_all(X, U)
        d = X[1:] - X[:-1] - 0.5 * self.ds[:, None] * (f[:-1] + f[1:])
        cons = d.reshape(-1)
        if self.closed:
            cons = np.concatenate([cons, X[0] - X[-1]])    # periodic lap
        return cons

    # --- inequality: friction limit (grip grows with downforce v^2) --------
    def a_grip(self, v):
        return self.a_max + self.c_dn * v * v

    def friction(self, z):
        X, U = self.unpack(z)
        ag = self.a_grip(X[:, 2])
        return ag**2 - (U[:, 0]**2 + U[:, 1]**2)

    # --- inequality: engine power limit  drive accel <= power/v ------------
    def power(self, z):
        X, U = self.unpack(z)
        return self.P_lim / X[:, 2] - U[:, 0]

    def power_jac(self, z):
        X, _ = self.unpack(z)
        Jp = np.zeros((self.N + 1, z.size))
        for i in range(self.N + 1):
            Jp[i, self._ix(i)] = [0.0, 0.0, -self.P_lim / X[i, 2]**2]
            Jp[i, self._iu(i)] = [-1.0, 0.0]
        return Jp

    # --- analytic Jacobians/gradient for the solver ------------------------
    def _ix(self, i):
        return slice(i * self.nv, i * self.nv + self.nv)

    def _iu(self, i):
        base = (self.N + 1) * self.nv
        return slice(base + i * self.nu, base + i * self.nu + self.nu)

    def objective_grad(self, z):
        X, _ = self.unpack(z)
        g = np.zeros_like(z)
        for i in range(self.N + 1):
            w = 0.0
            if i > 0:        w += 0.5 * self.ds[i - 1]
            if i < self.N:   w += 0.5 * self.ds[i]
            g[self._ix(i)] = w * self._dgdx(X[i], self.kap[i])
        return g

    def _dfdx_all(self, X, U):
        n, xi, v = X[:, 0], X[:, 1], X[:, 2]
        at, an = U[:, 0], U[:, 1]; k = self.kap
        D = 1 - n * k; c = np.cos(xi); t = np.tan(xi)
        g2 = at - self.drag * v * v
        J = np.zeros((self.N + 1, 3, 3))
        J[:, 0, 0] = -k * t;            J[:, 0, 1] = D / c**2
        J[:, 1, 0] = -k*an/(v*v*c); J[:, 1, 1] = D*an*t/(v*v*c); J[:, 1, 2] = -2*D*an/(v**3*c)
        J[:, 2, 0] = -k*g2/(v*c);   J[:, 2, 1] = D*g2*t/(v*c)
        J[:, 2, 2] = -D*(at + self.drag*v*v)/(v*v*c)
        return J

    def _dfdu_all(self, X, U):
        n, xi, v = X[:, 0], X[:, 1], X[:, 2]; k = self.kap
        D = 1 - n * k; c = np.cos(xi)
        J = np.zeros((self.N + 1, 3, 2))
        J[:, 1, 1] = D / (v*v*c)
        J[:, 2, 0] = D / (v*c)
        return J

    def defects_jac(self, z):
        X, U = self.unpack(z)
        nrow = 3 * self.N + (3 if self.closed else 0)
        Jc = np.zeros((nrow, z.size))
        I3 = np.eye(3)
        Adx = self._dfdx_all(X, U); Adu = self._dfdu_all(X, U)
        for k in range(self.N):
            r = slice(3 * k, 3 * k + 3); h = 0.5 * self.ds[k]
            Jc[r, self._ix(k)]     = -I3 - h * Adx[k]
            Jc[r, self._ix(k + 1)] =  I3 - h * Adx[k + 1]
            Jc[r, self._iu(k)]     =      - h * Adu[k]
            Jc[r, self._iu(k + 1)] =      - h * Adu[k + 1]
        if self.closed:
            r = slice(3 * self.N, 3 * self.N + 3)
            Jc[r, self._ix(0)] = I3
            Jc[r, self._ix(self.N)] = -I3
        return Jc

    def friction_jac(self, z):
        X, U = self.unpack(z)
        ag = self.a_grip(X[:, 2])
        Jf = np.zeros((self.N + 1, z.size))
        for i in range(self.N + 1):
            Jf[i, self._iu(i)] = [-2 * U[i, 0], -2 * U[i, 1]]
            if self.c_dn != 0.0:                         # d/dv of a_grip(v)^2
                Jf[i, self._ix(i)] = [0.0, 0.0, 4 * self.c_dn * X[i, 2] * ag[i]]
        return Jf

    # --- bounds on n (track width), xi, v, controls ------------------------
    def bounds(self):
        M = self.N + 1
        bnds = []
        for i in range(M):                              # states
            w = self.wid[i] - 0.1                        # small safety margin
            bnds.append((-w, w))                         # n
            bnds.append((-self.xi_max, self.xi_max))     # xi
            bnds.append((self.vmin, self.vmax))          # v
        a_ctl = self.a_max + self.c_dn * self.vmax**2    # max grip with downforce
        for i in range(M):                              # controls
            bnds.append((-a_ctl, a_ctl))                 # a_t
            bnds.append((-a_ctl, a_ctl))                 # a_n
        return bnds

    # --- initial guess: QSS speed profile on the centreline ----------------
    def initial_guess(self):
        M = self.N + 1
        v = self.qss_speed_profile()
        X = np.zeros((M, self.nv)); X[:, 2] = v
        U = np.zeros((M, self.nu))
        # rough controls: a_t from the speed gradient, a_n centripetal
        dvds = np.gradient(v, self.sg)
        U[:, 0] = np.clip(v * dvds, -self.a_max, self.a_max)
        U[:, 1] = np.clip(v**2 * self.kap, -self.a_max, self.a_max)
        return np.concatenate([X.reshape(-1), U.reshape(-1)])

    # --- solve (scaled / dimensionless variables for conditioning) ---------
    def solve(self, z0=None, maxiter=300, method="SLSQP", verbose=True):
        if z0 is None:
            z0 = self.initial_guess()
        S, rs, a2 = self.S, self.row_scale, self.a_max**2

        obj = lambda zh: self.objective(S * zh)
        grad = lambda zh: self.objective_grad(S * zh) * S
        dfn = lambda zh: self.defects(S * zh) / rs
        dfj = lambda zh: (self.defects_jac(S * zh) * S[None, :]) / rs[:, None]
        fri = lambda zh: self.friction(S * zh) / a2
        frij = lambda zh: (self.friction_jac(S * zh) * S[None, :]) / a2

        b = self.bounds()
        bnds_s = [(lo / S[i], hi / S[i]) for i, (lo, hi) in enumerate(b)]
        cons = [{"type": "eq", "fun": dfn, "jac": dfj},
                {"type": "ineq", "fun": fri, "jac": frij}]
        if self.P_lim is not None:                       # engine power limit
            am = self.a_max
            cons.append({"type": "ineq",
                         "fun": lambda zh: self.power(S * zh) / am,
                         "jac": lambda zh: (self.power_jac(S * zh) * S[None, :]) / am})
        res = minimize(obj, z0 / S, method=method, jac=grad,
                       bounds=bnds_s, constraints=cons,
                       options={"maxiter": maxiter, "ftol": 1e-8,
                                "disp": False})
        zp = S * res.x
        X, U = self.unpack(zp)
        maxdef = np.abs(self.defects(zp)).max()
        if verbose:
            print(f"  [{method}] success={res.success}  T={res.fun:.4f} s  "
                  f"iters={res.get('nit','?')}  maxdefect={maxdef:.1e}  "
                  f"| {res.message}")
        return dict(success=res.success, T=res.fun, X=X, U=U, z=zp,
                    s=self.sg, kappa=self.kap, width=self.wid, maxdef=maxdef,
                    ocp=self)


def grip_usage(U, a_max):
    return np.hypot(U[:, 0], U[:, 1]) / a_max
