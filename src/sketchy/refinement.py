from typing import List

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike

from .bezier import fit_bezier, interpolate_bezier, std_distance


def fit_bezier_error(pixel_c: ArrayLike):
    """
    arg :
        pixel_c (nd.array): 2 x N
    """
    assert pixel_c.shape[0] == 2

    instants = np.linspace(0, 1, pixel_c.shape[1])

    control_points = fit_bezier(pixel_c, instants)
    traj_fit = interpolate_bezier(control_points, instants)

    error = std_distance(traj_fit, pixel_c)

    return control_points, error


def fit_bezier_error_T(pixel_c: ArrayLike):
    """
    arg :
        pixel_c (nd.array): N x 2
    """
    assert pixel_c.shape[1] == 2

    traj = pixel_c.T
    return fit_bezier_error(traj)


def devide(pixel_c: np.ndarray, precision: float = 1) -> np.ndarray:
    _, e = fit_bezier_error_T(pixel_c)
    if e <= precision:
        return np.array([])

    N = len(pixel_c)

    _, e1 = fit_bezier_error_T(pixel_c[: N // 3])
    _, e2 = fit_bezier_error_T(pixel_c[N // 3 :])
    _, e3 = fit_bezier_error_T(pixel_c[: 2 * N // 3])
    _, e4 = fit_bezier_error_T(pixel_c[2 * N // 3 :])

    if e1 + e2 < e3 + e4:
        first = devide(pixel_c[: N // 3])
        second = devide(pixel_c[N // 3 :])
        idx = np.array([N // 3])

        if first.size == 0:
            first = np.array([])
        if second.size == 0:
            second = np.array([])
        else:
            second = second + idx  # ajout de idx à chaque élément

        return np.concatenate([first, idx, second])

    else:
        first = devide(pixel_c[: 2 * N // 3])
        second = devide(pixel_c[2 * N // 3 :])
        idx = np.array([2 * N // 3])

        if first.size == 0:
            first = np.array([])
        if second.size == 0:
            second = np.array([])
        else:
            second = second + idx  # ajout de idx à chaque élément

        return np.concatenate([first, idx, second])


def refine(topo_g: nx.MultiGraph, precision: float = 0.1):
    # TODO: why precision needs to be this small ? It's supposed to be equal to 2
    new_topo_graph = nx.MultiGraph()

    for u, v, data in topo_g.edges(data=True):
        pixels = data["pixels"]

        if pixels.shape[0] < 2:
            continue

        indices = devide(pixels, precision)

        if indices is None or len(indices) == 0:
            # Rien à changer
            new_topo_graph.add_node(u)
            new_topo_graph.add_node(v)
            new_topo_graph.add_edge(u, v, pixels=pixels)
            continue

        # Créer les nouveaux points d'interpolation
        indices = list(indices.astype(int))
        cuts = [0] + indices + [len(pixels) - 1]

        for i in range(len(cuts) - 1):
            node_a = tuple(map(int, pixels[cuts[i]]))
            node_b = tuple(map(int, pixels[cuts[i + 1]]))
            new_topo_graph.add_node(node_a)
            new_topo_graph.add_node(node_b)
            new_topo_graph.add_edge(
                node_a, node_b, pixels=pixels[cuts[i] : cuts[i + 1] + 1]
            )

    # Remplacer topo_c par les courbes mises à jour
    return new_topo_graph
