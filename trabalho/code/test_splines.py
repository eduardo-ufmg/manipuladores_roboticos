import matplotlib.pyplot as plt
import numpy as np
import pytest

from trajectory.digits import DIGITS
from trajectory.spline import StrokeSpline


@pytest.fixture
def sample_spline():
    """Provides a multi-point spline representing digit '3', first stroke."""
    stroke = DIGITS["3"][0]
    return StrokeSpline(stroke)


def test_spline_interpolation_correctness():
    """Test 3.1 — Spline interpolation correctness (Visual & Endpoints)"""
    stroke = DIGITS["2"][0]
    spline = StrokeSpline(stroke)

    # Mathematical assertion: Endpoints must match exactly
    np.testing.assert_allclose(
        spline.evaluate_position(np.asarray([0.0]))[0], stroke[0]
    )
    np.testing.assert_allclose(
        spline.evaluate_position(np.asarray([spline.length]))[0], stroke[-1]
    )

    # Visual assertion
    s_dense = np.linspace(0, spline.length, 100)
    smooth_pts = spline.evaluate_position(s_dense)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(
        [p[0] for p in stroke],
        [p[1] for p in stroke],
        "ro--",
        label="Original Polyline",
    )
    ax.plot(smooth_pts[:, 0], smooth_pts[:, 1], "b-", label="Cubic Spline")
    ax.set_title("Spline Smoothing Verification")
    ax.legend()
    ax.grid(True)

    output_path = "/tmp/spline_interpolation.png"
    plt.savefig(output_path)
    plt.close(fig)

    import os

    assert os.path.exists(output_path)


def test_velocity_continuity(sample_spline):
    """Test 3.2 — Velocity continuity"""
    spline = sample_spline
    s_dense = np.linspace(0, spline.length, 500)
    derivs = spline.evaluate_derivative(s_dense)

    # Compute numerical second derivative (acceleration)
    ds = s_dense[1] - s_dense[0]
    accel = np.diff(derivs, axis=0) / ds

    # Acceleration magnitudes should not contain infinite spikes
    accel_magnitudes = np.linalg.norm(accel, axis=1)
    max_accel = np.max(accel_magnitudes)

    # Bounded curvature limit for these specific normalized digit geometries
    assert (
        max_accel < 500.0
    ), f"Detected velocity discontinuity. Max acceleration: {max_accel}"


def test_arc_length_stability(sample_spline):
    """Test 3.3 — Arc-length stability"""
    spline = sample_spline
    num_samples = 1000
    s_dense = np.linspace(0, spline.length, num_samples)

    positions = spline.evaluate_position(s_dense)
    step_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    mean_dist = np.mean(step_distances)
    std_dist = np.std(step_distances)

    # Standard deviation of inter-sample distance should be extremely low
    # if the parameterization genuinely maps to arc length.
    assert std_dist < (
        mean_dist * 0.1
    ), f"Arc-length parameterization unstable. Std: {std_dist}"

    # Generate histogram for manual inspection
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(step_distances, bins=50, color="purple", alpha=0.7)
    ax.set_title("Inter-sample Distance Histogram")
    ax.set_xlabel("Euc. Dist. between consecutive samples", loc="left")
    ax.set_ylabel("Frequency")

    output_path = "trajectory/arc_length_histogram.png"
    plt.savefig(output_path)
    plt.close(fig)
