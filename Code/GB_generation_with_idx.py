"""Granular-ball generation with original sample indices."""

import numpy as np
from scipy.spatial.distance import cdist


def _get_radius(ball):
    """Return the maximum distance from samples to the ball center."""
    samples = ball[:, :-1]
    center = np.mean(samples, axis=0)
    return float(np.max(np.linalg.norm(samples - center, axis=1)))


def _split_ball(ball):
    """Split a ball using its two farthest samples as division anchors."""
    samples = ball[:, :-1]
    distances = cdist(samples, samples)
    rows, cols = np.where(distances == np.max(distances))
    anchor_1 = rows[1]
    anchor_2 = cols[1]

    child_1 = []
    child_2 = []
    for i in range(len(ball)):
        if distances[i, anchor_1] < distances[i, anchor_2]:
            child_1.append(ball[i])
        else:
            child_2.append(ball[i])

    return np.asarray(child_1), np.asarray(child_2)


def _get_density(ball):
    """Compute the density-volume criterion used for ball division."""
    samples = ball[:, :-1]
    center = np.mean(samples, axis=0)
    distance_sum = np.sum(np.linalg.norm(samples - center, axis=1))
    return len(samples) / distance_sum if distance_sum != 0 else len(samples)


def _divide_balls(granular_balls):
    """Divide balls when the weighted child density is improved."""
    divided_balls = []

    for ball in granular_balls:
        if len(ball) < 4:
            divided_balls.append(ball)
            continue

        child_1, child_2 = _split_ball(ball)
        if len(child_1) == 0 or len(child_2) == 0:
            divided_balls.append(ball)
            continue

        parent_density = _get_density(ball)
        child_density = (
            len(child_1) * _get_density(child_1)
            + len(child_2) * _get_density(child_2)
        ) / len(ball)

        if child_density > parent_density:
            divided_balls.extend([child_1, child_2])
        else:
            divided_balls.append(ball)

    return divided_balls


def _normalize_balls(granular_balls, radius_threshold):
    """Refine balls whose radii exceed twice the radius threshold."""
    normalized_balls = []

    for ball in granular_balls:
        if len(ball) < 2 or _get_radius(ball) <= 2 * radius_threshold:
            normalized_balls.append(ball)
        else:
            child_1, child_2 = _split_ball(ball)
            normalized_balls.extend([child_1, child_2])

    return normalized_balls


def get_GB(data):
    """Generate granular balls while retaining original sample indices."""
    data = np.asarray(data, dtype=float)
    indices = np.arange(len(data)).reshape(-1, 1)
    granular_balls = [np.hstack((data, indices))]

    while True:
        old_count = len(granular_balls)
        granular_balls = _divide_balls(granular_balls)
        if len(granular_balls) == old_count:
            break

    radii = [_get_radius(ball) for ball in granular_balls if len(ball) >= 2]
    radius_threshold = (
        max(np.median(radii), np.mean(radii)) if radii else 0.0
    )

    while True:
        old_count = len(granular_balls)
        granular_balls = _normalize_balls(granular_balls, radius_threshold)
        if len(granular_balls) == old_count:
            break

    return granular_balls
