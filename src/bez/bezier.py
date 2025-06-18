from typing_extensions import Optional
import numpy as np


def interpolant(t, degree=3):
    """
    t: 1d array of instant to evaluate
    degree: degree of the bezier to compute
    returns the "interpolants", the coefficients use to compute the linear interpolation for the bezier curve.
    """
    if degree == 1:
        return np.array( [(1-t), t] )
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
                np.pow(t, 2),
            ]
        )
    raise ValueError(f"degree of bezier is {degree}")


def interpolate_bezier(coeffs, t):
    """
    compute the trajectory of a bezier-curve.

    coeffs: a 2x(d+1) matrix, where d is the degree of the bezier.
        - row 0 for the x coordinates of control points
        - row 1 for the y coordinates of control points

    t: a 1d array of instant to evaluate
    """
    d = coeffs.shape[1] - 1
    assert coeffs.shape[0] == 2

    q = interpolant(t, d)
    return coeffs @ q

def fitting_error(traj, traj_target, weights: Optional[np.ndarray] = None):
    # TODO: use same logic for std_distance and fitting_error
    if weights is None:
        weights = np.ones(traj.shape)

    x, y = traj
    x_target, y_target = traj_target
    d_x = np.linalg.norm(weights * (x - x_target))
    d_y = np.linalg.norm(weights * (y - y_target))
    return d_x + d_y

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


def fit_bezier(traj, t = None, degree=3, weights: Optional[np.ndarray] = None):
    """
    fit a bezier of degree d on a sequence if points.

    traj: a 2xN matrix of positions.
        - the first row is x-position
        - the second row is y-position

    t: a 1d array of corresponding instants

    degree: the degree of the bezier to fit.

    returns a 2x(degree+1) matrix for the bezier control points
        - row 1: the x positions
        - row 2: the y positions

    weights: optional weights for fitting
    """
    if t is None :
        t = np.linspace(0, 1, len(traj[0]))

    if weights is None:
        weights = np.ones(t.shape)

    traj_x, traj_y = traj
    k = degree+1
    m = np.zeros((k, k))
    bx = np.zeros(k)
    by = np.zeros(k)

    # this equation follows from the minimization of the pixel standard distance
    for i in range(k):
        q = interpolant(t, degree)
        m[i] = np.sum(weights * (q * q[i]), axis=1)
        bx[i] = np.sum(weights * (traj_x * q[i]))
        by[i] = np.sum(weights * (traj_y * q[i]))

    x_fit = np.linalg.solve(m, bx)
    y_fit = np.linalg.solve(m, by)
    return np.array([x_fit, y_fit])


def fit_bezier_T(traj, t = None, weights: Optional[np.ndarray] = None):
    traj = np.array(traj).T
    return fit_bezier(traj, t, weights)
