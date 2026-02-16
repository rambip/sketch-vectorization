import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .bezier import interpolate_bezier
from .hypergraph import HyperGraph


def visualize_hyper(hyper: HyperGraph, offset=0):
    plt.axis("equal")
    plt.gca().invert_yaxis()
    for h in hyper.all_hyperedges():
        p = np.array(h.pixels).T
        instants = np.linspace(0, 1, p.shape[1])
        if h.control_points is None:
            raise ValueError("Hypergraph contains hyperedges that have not been fit")
        else:
            control_points = h.control_points
        traj = interpolate_bezier(control_points, instants)
        plt.plot(
            traj[1][offset : -offset - 1],
            traj[0][offset : -offset - 1],
            linewidth=1,
            color="red",
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


def visualize_topo_and_save(G, fig_size=6, save_path=None):
    subgraph = G
    pos = {node: (node[1], node[0]) for node in subgraph.nodes}
    node_color = "blue"

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.invert_yaxis()

    # Dessiner les nœuds
    nx.draw_networkx_nodes(subgraph, pos=pos, node_size=5, node_color=node_color, ax=ax)

    # Dessiner les arêtes avec courbure
    for u, v, keys in subgraph.edges(keys=True):
        num_edges = subgraph.number_of_edges(u, v)
        if num_edges == 1:
            nx.draw_networkx_edges(
                subgraph, pos, edgelist=[(u, v)], edge_color="gray", ax=ax
            )
        else:
            for k, key in enumerate(subgraph[u][v]):
                rad = 0.1 * (k - (num_edges - 1) / 2)
                nx.draw_networkx_edges(
                    subgraph,
                    pos,
                    edgelist=[(u, v)],
                    edge_color="gray",
                    connectionstyle=f"arc3,rad={rad}",
                    ax=ax,
                )

    ax.axis("off")

    if save_path:  # Si un chemin est fourni, on enregistre
        plt.savefig(
            save_path, dpi=300, bbox_inches="tight", pad_inches=0, transparent=True
        )

    plt.show()
    plt.close(fig)


def visualize_topo(G, visu_i=1335, visu_j=1112, visu_radius=50, opacity=1.0):
    subgraph = G
    pos = {node: (node[1], node[0]) for node in subgraph.nodes}
    node_color = "blue"

    plt.gca().invert_yaxis()
    plt.axis("equal")

    # Draw nodes
    nx.draw_networkx_nodes(
        subgraph, pos=pos, node_size=5, node_color=node_color, alpha=opacity
    )
    # Draw edges with curvature for multiplicity
    for u, v, keys in subgraph.edges(keys=True):
        num_edges = subgraph.number_of_edges(u, v)
        if num_edges == 1:
            nx.draw_networkx_edges(subgraph, pos, edgelist=[(u, v)], edge_color="green")
        else:
            # Draw each edge with a different curvature
            for k, key in enumerate(subgraph[u][v]):
                rad = 0.1 * (k - (num_edges - 1) / 2)
                nx.draw_networkx_edges(
                    subgraph,
                    pos,
                    edgelist=[(u, v)],
                    edge_color="green",
                    connectionstyle=f"arc3,rad={rad}",
                )


# mauvais affichage
def visualisation_bezier_hyper_final(hyper: HyperGraph, offset=2):
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
            color="red",
        )

        # Points bleus près des extrémités
        start_y, start_x = p[0][offset], p[1][offset]
        end_y, end_x = p[0][-offset - 1], p[1][-offset - 1]

        plt.scatter([start_x, end_x], [start_y, end_y], color="blue", s=3, alpha=0.5)

    plt.axis("equal")
    plt.title("Interpolated Bezier Curves (HyperEdges)")
    plt.show()


def visualize_graph_hypergraph(H: HyperGraph, fig_size=6, save_path=None):
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.invert_yaxis()

    node_color = "blue"  # (0, 0, 1, 0.2)

    pos = {node: (node[1], node[0]) for node in H.topo.nodes}

    # Pour chaque hyperedge, tracer une ligne droite entre ses deux extrémités
    for h in H.all_hyperedges():
        start = h.first()
        end = h.last()

        # Créer un mini-graph temporaire avec une seule arête (start, end)
        temp_g = nx.MultiGraph()
        temp_g.add_edge(start, end)

        # Dessiner cette arête en rouge, épaisseur plus forte, sans courbure (rad=0)
        nx.draw_networkx_edges(
            temp_g,
            pos,
            edgelist=[(start, end)],
            edge_color="gray",
            connectionstyle="arc3,rad=0",
            # arrows=False,
            ax=ax,
        )

        # Draw nodes
        nx.draw_networkx_nodes(temp_g, pos=pos, node_size=5, node_color=node_color)

    if save_path:  # Si un chemin est fourni, on enregistre
        plt.savefig(
            save_path, dpi=300, bbox_inches="tight", pad_inches=0, transparent=True
        )

    plt.show()
    plt.close(fig)
