import pytest
import numpy as np
from geometry.board import Board
from geometry.transforms import create_orthonormal_frame


@pytest.fixture
def vertical_board():
    """Provides a valid vertical board configuration."""
    origin = np.array([0.5, 0.0, 0.5])
    normal = np.array([-1.0, 0.0, 0.0])  # Board faces backwards along X
    up_reference = np.array([0.0, 0.0, 1.0])

    x, y, z = create_orthonormal_frame(normal, up_reference)

    return Board(
        width=0.60,
        height=0.25,
        origin=origin,
        x_axis=x,
        y_axis=y,
        normal=z,
        safety_offset=0.01,
    )


def test_board_frame_orthogonality(vertical_board):
    """Test 1.1 — Board frame orthogonality"""
    b = vertical_board
    assert np.isclose(np.linalg.norm(b.x_axis), 1.0)
    assert np.isclose(np.linalg.norm(b.y_axis), 1.0)
    assert np.isclose(np.linalg.norm(b.normal), 1.0)
    assert np.isclose(np.dot(b.x_axis, b.y_axis), 0.0)
    assert np.isclose(np.dot(b.normal, b.x_axis), 0.0)
    assert np.isclose(np.dot(b.normal, b.y_axis), 0.0)


def test_board_projection_correctness(vertical_board):
    """Test 1.2 — Board projection correctness"""
    b = vertical_board

    # Test origin mapping (0,0)
    p_origin = b.board_to_world(0.0, 0.0)
    expected_origin = b.origin + b.safety_offset * b.normal
    np.testing.assert_allclose(p_origin, expected_origin)

    # Test extremity mapping (width, height)
    p_corner = b.board_to_world(b.width, b.height)
    expected_corner = (
        b.origin + b.width * b.x_axis + b.height * b.y_axis + b.safety_offset * b.normal
    )
    np.testing.assert_allclose(p_corner, expected_corner)


def test_safety_offset_correctness(vertical_board):
    """Test 1.3 — Safety offset correctness"""
    b = vertical_board
    u, v = 0.1, 0.15
    p = b.board_to_world(u, v)

    vector_from_origin = p - b.origin
    distance_to_plane = np.dot(vector_from_origin, b.normal)

    assert np.isclose(distance_to_plane, b.safety_offset)


def test_approach_pose(vertical_board):
    """Validate free-space transition points."""
    b = vertical_board
    u, v = 0.3, 0.1
    dist = 0.05
    p_approach = b.get_approach_pose(u, v, approach_distance=dist)

    vector_from_origin = p_approach - b.origin
    distance_to_plane = np.dot(vector_from_origin, b.normal)

    assert np.isclose(distance_to_plane, b.safety_offset + dist)
