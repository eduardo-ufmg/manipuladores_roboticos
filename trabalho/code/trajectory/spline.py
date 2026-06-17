import numpy as np
from scipy.interpolate import make_interp_spline, PchipInterpolator
from scipy.integrate import cumulative_trapezoid
from typing import TypeAlias

Stroke: TypeAlias = list[tuple[float, float]]


class StrokeSpline:
    def __init__(self, stroke: Stroke, reparam_resolution: int = 2000):
        """
        Constructs a smooth spline from a discrete stroke and reparameterizes
        it by true arc length.

        Steps:
          1. Fit a B-spline interpolant using chord-length parameterization t.
          2. Numerically integrate ||dr/dt|| to obtain arc length s(t).
          3. Build a monotone PCHIP interpolant t(s) as the inverse map.
        """
        pts = np.array(stroke, dtype=float)
        n_points = len(pts)

        if n_points < 2:
            raise ValueError("A stroke must contain at least 2 points.")

        diffs = np.diff(pts, axis=0)
        dists = np.linalg.norm(diffs, axis=1)

        if np.any(dists < 1e-6):
            raise ValueError("Stroke contains duplicate consecutive points.")

        # --- Step 1: B-spline in chord-length parameter t ---
        t = np.concatenate(([0.0], np.cumsum(dists)))
        degree = min(3, n_points - 1)
        self._raw_spline = make_interp_spline(t, pts, k=degree)
        self._raw_deriv = self._raw_spline.derivative()

        # --- Step 2: Integrate speed ||dr/dt|| to obtain s(t) ---
        t_dense = np.linspace(0.0, t[-1], reparam_resolution)
        speeds = np.linalg.norm(self._raw_deriv(t_dense), axis=1)
        arc = np.concatenate(([0.0], cumulative_trapezoid(speeds, t_dense)))

        self.length: float = float(arc[-1])

        # --- Step 3: Monotone map s -> t ---
        self._s_to_t = PchipInterpolator(arc, t_dense)
        self._ds_to_dt = self._s_to_t.derivative()

    def evaluate_position(self, s_vals: np.ndarray) -> np.ndarray:
        """Returns the (x, y) position at arc length s."""
        t_vals = self._s_to_t(np.clip(s_vals, 0.0, self.length))
        return self._raw_spline(t_vals)

    def evaluate_derivative(self, s_vals: np.ndarray) -> np.ndarray:
        """
        Returns (dx/ds, dy/ds) at arc length s.

        By the chain rule:
            dr/ds = (dr/dt) * (dt/ds)
        where dt/ds is the derivative of the reparameterization map.
        """
        s_clipped = np.clip(s_vals, 0.0, self.length)
        t_vals = self._s_to_t(s_clipped)
        dt_ds = self._ds_to_dt(s_clipped)  # shape (N,)
        dr_dt = self._raw_deriv(t_vals)  # shape (N, 2)
        return dr_dt * dt_ds[:, np.newaxis]
