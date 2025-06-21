import matplotlib.pyplot as plt
from bez.hypergraph import HyperGraph
from bez.bezier import interpolate_bezier, fit_bezier
import networkx as nx
import numpy as np


def visualize_hyper(hyper: HyperGraph, offset=2):
    for h in hyper.all_hyperedges():
        p = np.array(h.pixels).T
        instants = np.linspace(0, 1, p.shape[1])
        control_points = fit_bezier(p, instants, degree=h.degree)
        traj = interpolate_bezier(control_points, instants)
        plt.plot(
            traj[1][offset:-offset], traj[0][offset:-offset], linewidth=1, color="red"
        )

        # Get near-extremity points
        start_y, start_x = traj[0][offset], traj[1][offset]
        end_y, end_x = traj[0][-offset - 1], traj[1][-offset - 1]

        plt.scatter(
            [start_x, end_x],
            [start_y, end_y],
            color="blue",
            s=3,
            alpha=0.5,
        )


def visualize_topo(G, visu_i=1335, visu_j=1112, visu_radius=50, full_graph=False):
    if full_graph:
        subgraph = G
    else:
        sub_nodes = [
            (i, j)
            for i in range(visu_i - visu_radius, visu_i + visu_radius)
            for j in range(visu_j - visu_radius, visu_j + visu_radius)
            if (i, j) in G
        ]
        subgraph = G.subgraph(sub_nodes)
    pos = {node: (node[1], node[0]) for node in subgraph.nodes}

    plt.gca().invert_yaxis()

        # Draw nodes
    nx.draw_networkx_nodes(subgraph, pos=pos, node_size=5, node_color=(0, 0, 1, 0.2))
    # Draw edges with curvature for multiplicity
    for u, v, keys in subgraph.edges(keys=True):
        num_edges = subgraph.number_of_edges(u, v)
        if num_edges == 1:
            nx.draw_networkx_edges(
                subgraph, pos, edgelist=[(u, v)], edge_color="gray"
            )
        else:
            # Draw each edge with a different curvature
            for k, key in enumerate(subgraph[u][v]):
                rad = 0.1 * (k - (num_edges - 1) / 2)
                nx.draw_networkx_edges(
                    subgraph,
                    pos,
                    edgelist=[(u, v)],
                    edge_color="gray",
                    connectionstyle=f"arc3,rad={rad}",
                )


# mauvais affichage
def hyper_final_visualisation(hyper: HyperGraph, offset=2):
    for h in hyper.all_hyperedges():
        if h.control_points is None:
            continue  # Skip if control points weren't set

        # Interpolation de la trajectoire à partir des points de contrôle
        p = interpolate_bezier(h.control_points, np.linspace(0, 1, 100))

        # Affichage de la courbe
        plt.plot(
            p[1][offset:-offset],  # y
            p[0][offset:-offset],  # x
            linewidth=1,
            color="red"
        )

        # Points bleus près des extrémités
        start_y, start_x = p[0][offset], p[1][offset]
        end_y, end_x = p[0][-offset - 1], p[1][-offset - 1]

        plt.scatter(
            [start_x, end_x],
            [start_y, end_y],
            color="blue",
            s=3,
            alpha=0.5
        )

    plt.axis('equal')
    plt.title("Interpolated Bezier Curves (HyperEdges)")
    plt.show()