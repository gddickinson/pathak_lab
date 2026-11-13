"""Skewness sign: the projection axis is the gyration tensor's dominant eigenvector, whose sign LAPACK picks
arbitrarily, so mirror-image tracks got opposite skewness. The axis is now oriented along the track's net displacement
(first -> last point), with a fixed fallback when that is perpendicular to the axis.

Run: python -m pytest -q tests/test_skewness_orientation.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from script_loader import load  # noqa: E402


def _skew(xy):
    tf = load("2_analysis/trackFeatures.py")
    return tf.RadiusGyrationAsymmetrySkewnessKurtosis(pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]}))[2]


def _track(seed):
    rng = np.random.default_rng(seed)
    steps = rng.normal(size=(15, 2)) * [1.0, 0.4] + [0.3, 0.1]
    steps[3] *= 4  # one long step: clearly skewed
    return np.cumsum(steps, axis=0)


def test_mirror_images_have_the_same_skewness():
    for seed in range(40):
        xy = _track(seed)
        s = _skew(xy)
        assert abs(s) > 1e-6
        for mirror in (xy * [-1, 1], xy * [1, -1], xy[:, ::-1]):
            assert np.isclose(_skew(mirror), s, rtol=1e-9, atol=1e-12), seed


def test_rotation_and_time_reversal_keep_the_skewness():
    for seed in range(20):
        xy = _track(seed)
        s = _skew(xy)
        for a in (0.3, 1.2, 2.5, 4.0):
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            assert np.isclose(_skew(xy @ R.T), s, rtol=1e-7, atol=1e-9)
        assert np.isclose(_skew(xy[::-1]), s, rtol=1e-9, atol=1e-12)


def test_axis_points_along_net_displacement():
    tf = load("2_analysis/trackFeatures.py")
    xy = np.array([[0, 0], [1, 0.1], [2, -0.1], [6, 0.0], [7, 0.1], [8, 0.0]])
    v = tf.orientedDominantAxis(xy, np.array([0.0, 1.0]))  # an x-dominant track
    assert np.allclose(v, [0, 1])  # (not the dominant axis: only the orientation rule is tested here)
    v = tf.orientedDominantAxis(xy, np.array([-1.0, 0.0]))
    assert np.allclose(v, [1, 0])
    closed = np.array([[0, 0], [2, 0], [4, 0.2], [2, 0.1], [0, 0]])  # net displacement 0: first nonzero component > 0
    assert np.allclose(tf.orientedDominantAxis(closed, np.array([-1.0, 0.0])), [1, 0])
    assert np.allclose(tf.orientedDominantAxis(closed, np.array([0.0, -1.0])), [0, 1])
    # a long step forward gives positive skewness
    assert _skew(xy) > 0
