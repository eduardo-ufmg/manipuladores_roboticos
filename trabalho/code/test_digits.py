import pytest
import math
import matplotlib.pyplot as plt
from trajectory.digits import DIGITS


def test_digit_existence():
    """Test 2.1 — Digit existence"""
    for digit in "0123456789":
        assert digit in DIGITS, f"Digit {digit} is missing."


def test_stroke_validity():
    """Test 2.2 — Stroke validity"""
    for digit_char, strokes in DIGITS.items():
        assert len(strokes) > 0, f"Digit {digit_char} has no strokes."

        for stroke in strokes:
            assert (
                len(stroke) >= 2
            ), f"Digit {digit_char} has a stroke with fewer than 2 points."

            for x, y in stroke:
                assert not math.isnan(x) and not math.isnan(
                    y
                ), f"Digit {digit_char} contains NaN coordinates."
                assert (
                    0.0 <= x <= 1.0
                ), f"Digit {digit_char} x-coordinate {x} out of bounds."
                assert (
                    0.0 <= y <= 1.0
                ), f"Digit {digit_char} y-coordinate {y} out of bounds."


def test_digit_visualization():
    """Test 2.3 — Visualization test (saves to disk for manual inspection)"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Raw Digit Stroke Sequences")

    for idx, digit_char in enumerate("0123456789"):
        ax = axes[idx // 5, idx % 5]
        strokes = DIGITS[digit_char]

        for stroke in strokes:
            x_vals = [p[0] for p in stroke]
            y_vals = [p[1] for p in stroke]
            ax.plot(x_vals, y_vals, marker="o", markersize=3, linewidth=2)

        ax.set_title(f"Digit: {digit_char}")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    # Save instead of blocking execution with plt.show()
    output_path = "trajectory/digits_visualization.png"
    plt.savefig(output_path)
    plt.close(fig)

    # Assert successful plot generation
    import os

    assert os.path.exists(output_path), "Visualization output file was not created."
