from typing_extensions import Optional
import numpy as np


def interpolant(t):
    """
    t: 1d array of instant to evaluate
    returns the "interpolants", the coefficients use to compute the linear interpolation for the bezier curve.
    """
    return np.array(
        [
            np.pow(1 - t, 3),
            3 * np.pow(1 - t, 2) * t,
            3 * (1 - t) * np.pow(t, 2),
            np.pow(t, 2),
        ]
    )


def interpolate_bezier(coeffs, t):
    """
    compute the trajectory of a bezier-curve.

    coeffs: a 2x4 matrix
        - row 1 for x
        - row 2 for y

    t: a 1d array of instant to evaluate
    """
    assert coeffs.shape == (2, 4)

    q = interpolant(t)
    return coeffs @ q


def std_distance(traj, traj_target, weights: Optional[np.ndarray] = None):
    """
    return the standard distance between a point in `traj` and a point in `traj_target`

    they can be weighted with weights

    This error is symmetric.
    """
    if weights is None:
        weights = np.ones(traj.shape)

    x, y = traj
    x_target, y_target = traj_target
    d_x = np.var(weights * (x - x_target))
    d_y = np.var(weights * (y - y_target))
    return np.sqrt(d_x + d_y)


def fit_bezier(traj, t = None, weights: Optional[np.ndarray] = None):
    """
    fit a bezier of degree 3 on a sequence if points.

    traj: a 2xN matrix of positions.
        - the first row is x-position
        - the second row is y-position

    t: a 1d array of corresponding instants

    returns a 2x4 matrix for the bezier control points
        - row 1: the x positions
        - row 2: the y positions

    weights: optional weights for fitting
    """
    if t is None :
        t = np.linspace(0, 1, len(traj[0]))

    if weights is None:
        weights = np.ones(t.shape)

    traj_x, traj_y = traj
    m = np.zeros((4, 4))
    bx = np.zeros(4)
    by = np.zeros(4)

    # this equation follows from the minimization of the pixel standard distance
    for i in range(4):
        q = interpolant(t)
        m[i] = np.sum(weights * (q * q[i]), axis=1)
        bx[i] = np.sum(weights * (traj_x * q[i]))
        by[i] = np.sum(weights * (traj_y * q[i]))

    x_fit = np.linalg.solve(m, bx)
    y_fit = np.linalg.solve(m, by)
    return np.array([x_fit, y_fit])


def fit_bezier_T(traj, t = None, weights: Optional[np.ndarray] = None):
    traj = np.array(traj).T
    return fit_bezier(traj, t, weights)