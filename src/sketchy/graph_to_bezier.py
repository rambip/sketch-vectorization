import numpy as np

from .bezier import fit_bezier


def bezier_from_graph(topo_graph):
    """
    Fit une bezier sur chaque edges du graphe et renvoie une liste de parametre de courbe de beziers
    arg :
        topo_graph (nx.Multigraph) : graphe topologique
    return :
        bezier_fits (List[(u, v), paramètre]) : une liste des parametre des courbe de bezier pour chaque edges du graphe topo
    """
    bezier_fits = []
    for i, edge in enumerate(topo_graph.edges(data=True, keys=True)):
        u, v, key, data = edge
        pixels = data["pixels"]
        if len(pixels) < 3:
            continue
        # Fit a Bezier curve to the pixel line
        p = np.vstack((pixels[:, 0], pixels[:, 1]))
        instants = np.linspace(0, 1, p.shape[1])
        control_points = fit_bezier(p, instants)
        bezier_fits.append(((u, v), control_points))

    return bezier_fits
