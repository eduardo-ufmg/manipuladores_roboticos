import pytest
import random
from main import execute_writing_task

# Maximum allowable tracking error in meters
MAX_TOLERANCE = 0.01


def test_single_digit():
    """Test 7.1 — Single digit integration"""
    max_error = execute_writing_task("0", headless=True)
    assert max_error < MAX_TOLERANCE, f"Tracking failed. Max error: {max_error}"


def test_multi_digit_code():
    """Test 7.2 — Multi-digit code integration"""
    max_error = execute_writing_task("123", headless=True)
    assert max_error < MAX_TOLERANCE, f"Tracking failed. Max error: {max_error}"


def test_worst_case_code():
    """
    Test 7.3 — Worst-case code
    Digit 8 has the most complex curvature and self-intersections.
    """
    max_error = execute_writing_task("888", headless=True)
    assert max_error < MAX_TOLERANCE, f"Tracking failed. Max error: {max_error}"


def test_random_code_batch():
    """
    Test 7.4 — Random code batch
    Ensures generalized stability across unpredictable stroke sequences.
    """
    for _ in range(5):
        random_code = f"{random.randint(0, 999):03d}"
        max_error = execute_writing_task(random_code, headless=True)
        assert (
            max_error < MAX_TOLERANCE
        ), f"Failed on code {random_code} with error {max_error}"
