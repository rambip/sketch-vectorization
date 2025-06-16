import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def image_to_skeleton(img):
    pass


def visualize_subgraph(G, visu_i=0, visu_j=0, visu_radius=100000):
    # Code fait par Copilot, marche mais pas testé

    plt.figure(figsize=(8, 8))
    sub_nodes = [
        (i, j)
        for i in range(visu_i - visu_radius, visu_i + visu_radius)
        for j in range(visu_j - visu_radius, visu_j + visu_radius)
        if (i, j) in G
    ]
    subgraph = G.subgraph(sub_nodes)
    pos = {node: (node[1], node[0]) for node in subgraph.nodes}

    if isinstance(subgraph, nx.MultiGraph):
        # Draw nodes
        nx.draw_networkx_nodes(subgraph, pos=pos, node_size=10, node_color="blue")
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
    else:
        nx.draw(
            subgraph,
            pos=pos,
            node_size=10,
            edge_color="gray",
            node_color="blue",
            with_labels=False,
        )
    plt.gca().invert_yaxis()
    plt.show()


def extract_topology_from_skeleton(skeleton_img):
    """
    Extrait la topologie d'un squelette binaire (image) en un graphe.
    arg:
        skeleton_img (np.ndarray): Image binaire du squelette (shape: H x W, dtype=bool)
    return:
        topological_graph (nx.Graph): Graphe topologique extrait du squelette
    """

    # Pour faciliter l'extraction on transforme le squelette en un graphe identique
    # On pourrait effectuer les opérations directement sur le squelette mais c'est moins pratique

    assert skeleton_img.dtype == bool

    sk_graph = nx.Graph()  # V = pixels blancs du squelette, E = 8 connexité

    for i in range(skeleton_img.shape[0]):
        for j in range(skeleton_img.shape[1]):
            if skeleton_img[i, j]:
                sk_graph.add_node((i, j))

    for node in sk_graph.nodes:
        i, j = node
        for ni in range(max(0, i - 1), min(skeleton_img.shape[0], i + 2)):
            for nj in range(max(0, j - 1), min(skeleton_img.shape[1], j + 2)):
                if (ni, nj) != (i, j) and skeleton_img[ni, nj]:
                    sk_graph.add_edge(
                        (i, j), (ni, nj), weight=(i - ni) ** 2 + (j - nj) ** 2
                    )

    # Graphe intermédiaire généré, maintenant on extrait la topologie
    # On fait un bfs depuis chaque intersection ou feuille jusqu'à avoir couvert tout le graphe

    topological_graph = (
        nx.MultiGraph()
    )  # Graphe topologique, V = intersections et feuilles, E = lignes de pixels

    for node in sk_graph.nodes:
        sk_graph.nodes[node]["closed"] = False

    def is_junction(node):
        return sk_graph.degree(node) > 2

    def is_leaf(node):
        return sk_graph.degree(node) == 1

    for st in sk_graph.nodes:
        if sk_graph.nodes[st]["closed"]:
            continue
        # otherwise, we explore the graph from this node
        # Until we either connect 2 leafs or junctions or loop
        neighbors = list(sk_graph.neighbors(st))
        pixel_line = [st]
        found_first_endpoint = is_leaf(st) or is_junction(st)
        for neigh in neighbors:
            if sk_graph.nodes[neigh]["closed"]:
                continue
            current_node = neigh
            while True:
                pixel_line.append(current_node)

                if is_leaf(current_node) or is_junction(current_node):
                    # On ne pourrait les fermer que si ce sont des leaf, mais pas gravissime de pas le faire
                    # Par contre faut surtout pas fermer une jonction parce que il pourrait rester des trucs à explorer depuis là
                    if not found_first_endpoint:
                        found_first_endpoint = True
                        pixel_line.reverse()  # Comment ça l'endpoint est bien au début de la liste
                        current_node = st  # On repart de l'autre coté
                    else:
                        topological_graph.add_edge(
                            pixel_line[0], pixel_line[-1], pixels=np.array(pixel_line)
                        )
                        pixel_line = [
                            st
                        ]  # pas vide car si c'est une jonction on va à la prochaine itération du for repartir sur une autre branche
                        break

                elif current_node == st:
                    # We are back to the starting point, we close the circle
                    assert (
                        not found_first_endpoint
                    )  # Si on est dans un cercle impossible d'avoir trouvé un endpoint
                    sk_graph.nodes[current_node]["closed"] = True
                    topological_graph.add_edge(
                        pixel_line[0], pixel_line[-1], pixels=np.array(pixel_line)
                    )
                    pixel_line = [
                        st
                    ]  # cf au dessus même si ici ce sera normalement jamais une jonction
                    break

                # Dans ce cas là on continue juste de suivre le chemin
                # assert sk_graph.degree(current_node) == 2, f"Le noeud courant doit avoir exactement 2 voisins, il en a {sk_graph.degree(current_node)}"

                nexts = list(sk_graph.neighbors(current_node))
                next_node = nexts[1] if nexts[0] == pixel_line[-2] else nexts[0]
                sk_graph.nodes[current_node]["closed"] = True
                current_node = next_node
            # Ici, on a soit trouvé une courbe en entier, soit qu'un endpoint, et on va continuer de l'autre coté.
        sk_graph.nodes[st]["closed"] = True

    return topological_graph
