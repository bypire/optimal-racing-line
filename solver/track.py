"""
Track geometry in the curvilinear (Frenet) frame.
================================================================================
A track is described by its centreline CURVATURE kappa(s) and half-WIDTH w(s) as
functions of arc length s. That is all the optimal-control problem needs: the
car's lateral offset n from the centreline and its heading relative to the track
tangent evolve against kappa(s); staying on track is just |n| <= w(s).

The (x, y) centreline -- only needed for drawing -- is recovered by integrating

    dphi/ds = kappa(s),   dx/ds = cos(phi),   dy/ds = sin(phi).
"""

import numpy as np


class Track:
    def __init__(self, s, kappa, width, closed=False, name=""):
        self.s = np.asarray(s, float)
        self._kappa = np.asarray(kappa, float)
        self._width = (np.full_like(self.s, width) if np.isscalar(width)
                       else np.asarray(width, float))
        self.L = self.s[-1]
        self.closed = closed
        self.name = name
        self._build_centreline()

    # --- interpolators -----------------------------------------------------
    def kappa(self, s):
        return np.interp(s, self.s, self._kappa,
                         period=self.L if self.closed else None)

    def width(self, s):
        return np.interp(s, self.s, self._width,
                         period=self.L if self.closed else None)

    # --- centreline geometry (for plotting / converting (s,n) -> (x,y)) ----
    def _build_centreline(self):
        ds = np.diff(self.s, prepend=self.s[0])
        ds[0] = self.s[1] - self.s[0]
        phi = np.cumsum(self._kappa * ds)
        phi -= phi[0]
        self.phi = phi
        self.cx = np.cumsum(np.cos(phi) * ds)
        self.cy = np.cumsum(np.sin(phi) * ds)
        self.cx -= self.cx[0]; self.cy -= self.cy[0]

    def to_xy(self, s, n):
        """Map curvilinear (s, n) to global (x, y) using the centreline normal."""
        phi = np.interp(s, self.s, self.phi,
                        period=self.L if self.closed else None)
        cx = np.interp(s, self.s, self.cx, period=self.L if self.closed else None)
        cy = np.interp(s, self.s, self.cy, period=self.L if self.closed else None)
        # left normal = (-sin phi, cos phi)
        return cx - n * np.sin(phi), cy + n * np.cos(phi)

    def edges(self):
        """Left/right track boundary polylines (for drawing)."""
        xl, yl = self.to_xy(self.s, self._width)
        xr, yr = self.to_xy(self.s, -self._width)
        return (xl, yl), (xr, yr)

    # --- factory: a single corner (straight - arc - straight) --------------
    @classmethod
    def single_corner(cls, R=60.0, angle_deg=90.0, straight_in=80.0,
                      straight_out=80.0, width=6.0, ds=0.5):
        angle = np.radians(angle_deg)
        arc_len = R * angle
        segs = [(0.0, straight_in), (1.0 / R, arc_len), (0.0, straight_out)]
        s_list, k_list = [0.0], [0.0]
        s = 0.0
        for kap, length in segs:
            n = max(2, int(round(length / ds)))
            for _ in range(n):
                s += length / n
                s_list.append(s); k_list.append(kap)
        return cls(np.array(s_list), np.array(k_list), width,
                   closed=False, name=f"corner R={R:g} {angle_deg:g}deg")

    # --- factory: a closed loop from (curvature, length) segments ----------
    @classmethod
    def from_segments(cls, segments, width=6.0, ds=0.5, name="loop"):
        """segments = list of (curvature, length). For a closed track the
        signed curvatures*lengths should sum to +/-2*pi and the path should
        close; we don't force it, we just build kappa(s)."""
        s_list, k_list = [0.0], [segments[0][0]]
        s = 0.0
        for kap, length in segments:
            n = max(2, int(round(length / ds)))
            for _ in range(n):
                s += length / n
                s_list.append(s); k_list.append(kap)
        closed = True
        return cls(np.array(s_list), np.array(k_list), width,
                   closed=closed, name=name)

    # --- factory: a smooth closed circuit from a parametric loop -----------
    @classmethod
    def from_xy_closed(cls, x, y, width=8.0, ds=1.5, name="circuit"):
        """Build a closed track from a densely-sampled closed centreline (x,y).
        Curvature kappa(s) and arc length s are computed from periodic finite
        differences, then resampled to a uniform s-grid."""
        x = np.asarray(x, float); y = np.asarray(y, float)
        dx = (np.roll(x, -1) - np.roll(x, 1)) / 2
        dy = (np.roll(y, -1) - np.roll(y, 1)) / 2
        ddx = np.roll(x, -1) - 2 * x + np.roll(x, 1)
        ddy = np.roll(y, -1) - 2 * y + np.roll(y, 1)
        sp = np.hypot(dx, dy)                          # ds/dt (per sample step)
        kappa = (dx * ddy - dy * ddx) / np.maximum(sp**3, 1e-12)
        s = np.concatenate([[0], np.cumsum(sp)[:-1]])
        L = s[-1] + sp[-1]
        s_uni = np.arange(0, L, ds)
        # periodic interpolation of kappa onto the uniform grid
        s_ext = np.concatenate([s, [L]])
        kap_ext = np.concatenate([kappa, [kappa[0]]])
        kap_uni = np.interp(s_uni, s_ext, kap_ext)
        t = cls(s_uni, kap_uni, width, closed=True, name=name)
        return t

    @classmethod
    def grand_circuit(cls, width=9.0, ds=1.5):
        """A stylised GP-style circuit: a long main straight, a tight hairpin, a
        sequence of medium corners and a fast sweeper. Parametric closed loop
        (harmonics of an ellipse) so it is smooth and self-closing."""
        th = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
        x = (430 * np.cos(th) + 55 * np.cos(2 * th) - 20 * np.cos(3 * th))
        y = (250 * np.sin(th) + 95 * np.sin(2 * th) + 45 * np.sin(3 * th)
             - 25 * np.sin(4 * th))
        return cls.from_xy_closed(x, y, width=width, ds=ds, name="Grand Circuit")

    @classmethod
    def skidpad(cls, R=50.0, width=2.0, ds=0.5):
        """A constant-radius circular track. Pure cornering limit -> the
        minimum lap time has a closed form: T = 2*pi*sqrt(R/(mu g)). Used as a
        ground-truth check on the optimal-control solver."""
        segs = [(1.0 / R, 2 * np.pi * R)]
        return cls.from_segments(segs, width=width, ds=ds, name=f"skidpad R={R:g}")

    @classmethod
    def oval(cls, straight=120.0, R=50.0, width=7.0, ds=0.5):
        """A closed oval: two straights joined by two 180-deg bends."""
        half = np.pi * R
        segs = [(0.0, straight), (1.0 / R, half),
                (0.0, straight), (1.0 / R, half)]
        return cls.from_segments(segs, width=width, ds=ds, name="oval")


if __name__ == "__main__":
    t = Track.single_corner()
    print(f"{t.name}: L={t.L:.1f} m, {len(t.s)} samples, "
          f"kappa max={t._kappa.max():.4f} (1/m)")
    o = Track.oval()
    print(f"{o.name}: L={o.L:.1f} m, closed={o.closed}, "
          f"end-start gap=({o.cx[-1]:.2f},{o.cy[-1]:.2f}) m")
