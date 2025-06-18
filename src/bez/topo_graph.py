import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from collections import deque

from skimage.morphology import skeletonize, dilation, erosion, remove_small_holes
from skimage.morphology import disk
from skimage.filters import threshold_otsu




def image_to_skeleton(img):
    img = img[:, :, 0]  

    t = threshold_otsu(img)
    img_binary = img < t

    thicknesses = np.zeros_like(img_binary, dtype=int)

    c = 1
    max_line_width_detection = img_binary.copy()
    while np.sum(max_line_width_detection) > 0:
        # iterate the erosion
        tmp = erosion(max_line_width_detection, disk(1))
        # find the difference
        diff = max_line_width_detection ^ tmp > 0
        thicknesses[diff] = c
        max_line_width_detection = tmp
        c += 1

    img_binary = dilation(img_binary, disk(c//2))
    img_binary = remove_small_holes(img_binary)
    skeleton = skeletonize(img_binary, method="zhang")

    return skeleton, thicknesses



def visualize_subgraph(G, visu_i=1335, visu_j=1112, visu_radius=50, full_graph=False): 
    plt.figure(figsize=(8, 8))
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
        thicknesses (np.ndarray): Thicknesses (shape: H x W, dtype=int)
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
                    sk_graph.add_edge((i, j), (ni, nj), weight=(i - ni) ** 2 + (j - nj) ** 2)
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





def pixel_graphe(skeleton_img):
    sk_graph = nx.Graph()
    H, W = skeleton_img.shape

    # Ajout des noeuds pour pixels blancs
    for i in range(H):
        for j in range(W):
            if skeleton_img[i, j]:
                sk_graph.add_node((i, j))

    # Ajout des arêtes 8-connexité
    for node in sk_graph.nodes:
        i, j = node
        for ni in range(max(0, i - 1), min(H, i + 2)):
            for nj in range(max(0, j - 1), min(W, j + 2)):
                if (ni, nj) != (i, j) and skeleton_img[ni, nj]:
                    sk_graph.add_edge((i, j), (ni, nj), weight=(i - ni) ** 2 + (j - nj) ** 2)
    
    return sk_graph


def create_clusters(pix_graph):
    # Fonction pour détecter si un noeud est feuille ou jonction
    def is_junction_or_leaf(node):
        deg = pix_graph.degree(node)
        return deg == 1 or deg > 2

    visited = set()
    clusters = []

    for node in pix_graph.nodes:
        if node in visited:
            continue
        if not is_junction_or_leaf(node):
            continue

        # BFS pour trouver cluster des noeuds voisins qui sont eux aussi feuilles/jonctions
        queue = deque([node])
        cluster = []
        while queue:
            curr = queue.popleft()
            if curr in visited:
                continue
            if not is_junction_or_leaf(curr):
                continue
            visited.add(curr)
            cluster.append(curr)
            for neigh in pix_graph.neighbors(curr):
                if neigh not in visited and is_junction_or_leaf(neigh):
                    queue.append(neigh)

        if cluster:
            clusters.append(cluster)

    return clusters


def merge_clusters_nodes(clusters):
    # Étape 2 : réduire chaque cluster à un seul sommet (le "centre")
    # On choisit le pixel avec la coordonnée médiane dans cluster
    merged_nodes = {}  # mapping de noeud original vers noeud fusionné

    for cluster in clusters:
        # Trouver le pixel médian (central)
        coords = np.array(cluster)
        median_i = int(np.median(coords[:, 0]))
        median_j = int(np.median(coords[:, 1]))
        center_node = (median_i, median_j)

        # S’assurer que center_node fait bien partie du cluster (sinon prendre le plus proche)
        if center_node not in cluster:
            # Trouver le plus proche dans cluster
            dist = lambda x: (x[0] - median_i)**2 + (x[1] - median_j)**2
            center_node = min(cluster, key=dist)

        for n in cluster:
            merged_nodes[n] = center_node

    return merged_nodes



def fusion_clusters_in_graph(pix_graph, merged_nodes):
    # 4. Construire un graphe simplifié avec fusion des noeuds
    fused_graph = nx.Graph()
    for n in pix_graph.nodes:
        fused_n = merged_nodes.get(n, n)
        fused_graph.add_node(fused_n)

    for u, v, d in pix_graph.edges(data=True):
        u_fused = merged_nodes.get(u, u)
        v_fused = merged_nodes.get(v, v)
        if u_fused != v_fused:
            fused_graph.add_edge(u_fused, v_fused, weight=d['weight'])
    
    return fused_graph


def extract_simple_topology_from_skeleton(skeleton_img):
    assert skeleton_img.dtype == bool

    pix_graph = pixel_graphe(skeleton_img)

    clusters = create_clusters(pix_graph)

    merged_nodes = merge_clusters_nodes(clusters)

    fused_graph = fusion_clusters_in_graph(pix_graph, merged_nodes)

    def is_junction(node):
        return fused_graph.degree(node) > 2

    def is_leaf(node):
        return fused_graph.degree(node) == 1

    for node in fused_graph.nodes:
        fused_graph.nodes[node]["closed"] = False

    topological_graph = nx.MultiGraph()

    for st in fused_graph.nodes:
        if fused_graph.nodes[st]["closed"]:
            continue

        neighbors = list(fused_graph.neighbors(st))
        pixel_line = [st]
        found_first_endpoint = is_leaf(st) or is_junction(st)

        for neigh in neighbors:
            if fused_graph.nodes[neigh]["closed"]:
                continue

            current_node = neigh
            while True:
                pixel_line.append(current_node)

                if is_leaf(current_node) or is_junction(current_node):
                    if not found_first_endpoint:
                        found_first_endpoint = True
                        pixel_line.reverse()
                        current_node = st
                    else:
                        topological_graph.add_edge(
                            pixel_line[0], pixel_line[-1], pixels=np.array(pixel_line)
                        )
                        pixel_line = [st]
                        break

                elif current_node == st:
                    assert not found_first_endpoint
                    fused_graph.nodes[current_node]["closed"] = True
                    topological_graph.add_edge(
                        pixel_line[0], pixel_line[-1], pixels=np.array(pixel_line)
                    )
                    pixel_line = [st]
                    break

                nexts = list(fused_graph.neighbors(current_node))
                # choix du prochain noeud à suivre
                next_node = nexts[1] if nexts[0] == pixel_line[-2] else nexts[0]
                fused_graph.nodes[current_node]["closed"] = True
                current_node = next_node

        fused_graph.nodes[st]["closed"] = True

    return topological_graph

