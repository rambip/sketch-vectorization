import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from bez.bezier import fit_bezier, interpolate_bezier, fitting_error


def score(hyperedge, lam, mu=0.3):
    u_simplicity = 1 + mu * hyperedge.degree
    p = np.array(hyperedge.pixels).T

    try:
        instants = np.linspace(0, 1, p.shape[1])
        control_points = fit_bezier(p, instants, degree=hyperedge.degree)
        traj = interpolate_bezier(control_points, instants)
        u_fidelity = fitting_error(traj, p)
        return (1 - lam) * u_fidelity + lam * u_simplicity
    except np.linalg.LinAlgError:
        return float("inf")


def optimisation(hyper, lam, mu, temp, t_decrease, t_min, MAX_IT=500_000):
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
