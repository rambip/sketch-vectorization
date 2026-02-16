import numpy as np
from tqdm import tqdm

from .bezier import fit_bezier, fitting_error, interpolate_bezier


def fit_hyperedge(hyperedge):
    p = np.array(hyperedge.pixels).T
    instants = np.linspace(0, 1, p.shape[1])
    control_points = fit_bezier(p, instants, degree=hyperedge.degree)
    traj = interpolate_bezier(control_points, instants)
    u_fidelity = fitting_error(traj, p)
    hyperedge.control_points = control_points
    hyperedge.fitting_error = u_fidelity


def score(hyperedge, lam, mu=0.3):
    u_simplicity = 1 + mu * hyperedge.degree

    try:
        if not hasattr(hyperedge, "fitting_error"):
            fit_hyperedge(hyperedge)
        return (1 - lam) * hyperedge.fitting_error + lam * u_simplicity
    except np.linalg.LinAlgError:
        return float("inf")


def optimisation(
    hyper, lam=0.5, mu=0.2, temp=0.5, t_decrease=0.99995, t_min=0.05, MAX_IT=500_000
):
    for h in hyper.all_hyperedges():
        fit_hyperedge(h)
    energy = sum([score(x, lam, mu) for x in hyper.all_hyperedges()])
    error = []

    n_it = int(np.log(t_min / temp) / np.log(t_decrease))

    for i in tqdm(range(n_it)):
        choice, old, new = hyper.propose_random_perturbation()
        delta = sum(score(x, lam, mu) for x in new) - sum(
            score(x, lam, mu) for x in old
        )
        p = np.random.random()
        if p < np.exp(-delta / temp):
            hyper.perturbate(old, new)
            energy += delta
        else:
            error.append(energy)
        temp = temp * t_decrease

    return error
