import numpy as np
from numpy.typing import NDArray
from typing_extensions import Optional


def _interpolant(t, degree=3):
    """
    Args:
        t: 1d array of instant to evaluate
        degree: degree of the bezier to compute
    Returns:
        the "interpolants", the coefficients use to compute the linear interpolation for the bezier curve.
    """
    if degree == 1:
        return np.array([(1 - t), t])
    if degree == 2:
        return np.array(
            [
                np.pow(1 - t, 2),
                2 * (1 - t) * t,
                np.pow(t, 2),
            ]
        )
    if degree == 3:
        return np.array(
            [
                np.pow(1 - t, 3),
                3 * np.pow(1 - t, 2) * t,
                3 * (1 - t) * np.pow(t, 2),
                np.pow(t, 3),
            ]
        )
    raise ValueError(f"degree of bezier is {degree}")


def interpolate_bezier(coeffs: NDArray, t: NDArray) -> NDArray:
    """
    compute the trajectory of a bezier-curve.

    Args:
        coeffs: a (d+1)x2 matrix, where d is the degree of the bezier.
        - row 0 for the x coordinates of control points
        - row 1 for the y coordinates of control points

        t: a 1d array of instant to evaluate

    Returns:
        The computed positions as a Nx2 matrix, where N is the length of t.
    """
    d = coeffs.shape[0] - 1
    assert coeffs.shape[1] == 2

    q = _interpolant(t, d)
    return q.T @ coeffs


def fitting_error(traj, traj_target, weights: Optional[NDArray] = None) -> float:
    if weights is None:
        weights = np.ones(len(traj))

    x, y = traj.T
    x_target, y_target = traj_target.T
    d_x = np.linalg.norm(weights * (x - x_target))
    d_y = np.linalg.norm(weights * (y - y_target))
    return float(d_x + d_y)


def fit_bezier(traj, t=None, degree=3, weights: Optional[np.ndarray] = None):
    """
    fit a bezier of degree d on a sequence of points.

    traj: a Nx2 matrix of positions.

    t: a 1d array of corresponding instants

    degree: the degree of the bezier to fit.

    returns a (degree+1)x2 matrix for the bezier control points
        - row 1: the x positions
        - row 2: the y positions

    weights: optional weights for fitting
    """
    if t is None:
        t = np.linspace(0, 1, len(traj[0]))

    if weights is None:
        weights = np.ones(t.shape)

    traj_x, traj_y = traj.T
    k = degree + 1
    m = np.zeros((k, k))
    bx = np.zeros(k)
    by = np.zeros(k)

    # this equation follows from the minimization of the pixel standard distance
    for i in range(k):
        q = _interpolant(t, degree)
        m[i] = np.sum(weights * (q * q[i]), axis=1)
        bx[i] = np.sum(weights * (traj_x * q[i]))
        by[i] = np.sum(weights * (traj_y * q[i]))

    # We use lstsq because the matrix is not always full-rank.
    # For example, if the degree of the bezier is greater than the length of "traj"
    x_fit = np.linalg.lstsq(m, bx)[0]
    y_fit = np.linalg.lstsq(m, by)[0]
    return np.vstack([x_fit, y_fit]).T
