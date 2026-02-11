import marimo

__generated_with = "0.19.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from meta import ROOT_DIR
    from bez.segments import segments
    from skimage import io
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy
    import networkx as nx

    return ROOT_DIR, io, np, nx, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Skeleton extraction
    ## Trapped-ball segmentation
    ### Decorative Line Detection
    #### Detection de points centraux des lignes :
    + Convolution avec un "second derivative of a Gaussian"
    + detections des maximas locaux : nonmax-ima suppression as in the original Canny edge detector
    + Récupère en même temps la direction de la ligne

    ####
    """)
    return


@app.cell
def _(ROOT_DIR, io, np, plt):
    from skimage.morphology import skeletonize, dilation, erosion, remove_small_holes
    from skimage.morphology import disk
    from skimage.filters import threshold_otsu
    test_image = io.imread(ROOT_DIR / 'data/original_paper/figure_2/input.png')
    test_image = test_image[:, :, 0]
    plt.imshow(test_image, cmap='gray')
    plt.axis('off')
    plt.show()
    t = threshold_otsu(test_image)
    test_image_binary = test_image < t
    thicknesses = np.zeros_like(test_image_binary, dtype=int)
    _c = 1
    max_line_width_detection = test_image_binary.copy()
    while np.sum(max_line_width_detection) > 0:
        tmp = erosion(max_line_width_detection, disk(1))
        diff = max_line_width_detection ^ tmp > 0
        thicknesses[diff] = _c
        max_line_width_detection = tmp
        _c = _c + 1
    print(f'Max line width: {_c}')
    plt.imshow(thicknesses, cmap='gray')
    plt.axis('off')
    plt.show()
    test_image_binary = dilation(test_image_binary, disk(_c // 2))
    plt.imshow(test_image_binary, cmap='gray')
    plt.axis('off')
    plt.show()
    skeleton = skeletonize(test_image_binary, method='zhang')
    print(np.unique(skeleton))
    plt.imshow(skeleton, cmap='gray')
    plt.axis('off')
    plt.show()
    return skeleton, test_image, thicknesses


@app.cell
def _(np, plt, skeleton):
    import scipy.signal
    kernel = np.ones((3, 3), dtype=int)
    n = _scipy.signal.convolve2d(skeleton.astype(int), kernel, mode='same')
    n = n * skeleton
    h = np.histogram(n)
    print(h)
    plt.plot(h[1][:-1], h[0])
    plt.xlabel('Number of neighbors')
    plt.ylabel('Number of pixels')
    plt.title('Histogram of the number of neighbors')
    plt.show()
    _c = np.where(n == 8)
    _c = (1335, 1112)
    _sub_image = skeleton[_c[0] - 100:_c[0] + 100, _c[1] - 100:_c[1] + 100]
    plt.imshow(_sub_image, cmap='gray')
    plt.axis('off')
    plt.show()
    return (n,)


@app.cell
def _(n, np, plt):
    from matplotlib.colors import ListedColormap

    plt.scatter(np.where(n == 2)[1], np.where(n == 2)[0], color="green", s=1)
    plt.scatter(np.where(n >= 4)[1], np.where(n >= 4)[0], color="red", s=1)
    plt.imshow(n, cmap=ListedColormap(["white", "white", "green", "black", "red", "white", "white", "white"]))
    plt.axis("off")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Topoligical graph extraction
    """)
    return


@app.cell
def _(nx, plt, skeleton):
    (visu_i, visu_j) = (1335, 1112)
    visu_radius = 50
    _sub_image = skeleton[visu_i - visu_radius:visu_i + visu_radius, visu_j - visu_radius:visu_j + visu_radius]
    plt.imshow(_sub_image, cmap='gray')
    plt.axis('off')
    plt.show()

    def visualize_subgraph(G, visu_i=1335, visu_j=1112, visu_radius=50, full_graph=False):
        plt.figure(figsize=(8, 8))
        if full_graph:
            subgraph = G
        else:
            sub_nodes = [(_i, j) for _i in range(visu_i - visu_radius, visu_i + visu_radius) for j in range(visu_j - visu_radius, visu_j + visu_radius) if (_i, j) in G]
            subgraph = G.subgraph(sub_nodes)
        pos = {_node: (_node[1], _node[0]) for _node in subgraph.nodes}
        if isinstance(subgraph, nx.MultiGraph):
            nx.draw_networkx_nodes(subgraph, pos=pos, node_size=10, node_color='blue')
            for (_u, _v, keys) in subgraph.edges(keys=True):
                num_edges = subgraph.number_of_edges(_u, _v)
                if num_edges == 1:
                    nx.draw_networkx_edges(subgraph, pos, edgelist=[(_u, _v)], edge_color='gray')
                else:
                    for (k, key) in enumerate(subgraph[_u][_v]):
                        rad = 0.1 * (k - (num_edges - 1) / 2)
                        nx.draw_networkx_edges(subgraph, pos, edgelist=[(_u, _v)], edge_color='gray', connectionstyle=f'arc3,rad={rad}')
        else:
            nx.draw(subgraph, pos=pos, node_size=1, edge_color='gray', node_color='blue', with_labels=False)
        plt.gca().invert_yaxis()  # Draw nodes
        plt.show()  # Draw edges with curvature for multiplicity  # Draw each edge with a different curvature

    return visu_i, visu_j, visu_radius, visualize_subgraph


@app.cell
def _(plt, thicknesses, visu_i, visu_j, visu_radius):
    from scipy.ndimage import maximum_filter

    thicknesses_max = maximum_filter(thicknesses, size=3)

    sub_thicknesses_max = thicknesses_max[
        visu_i-visu_radius:visu_i+visu_radius,
        visu_j-visu_radius:visu_j+visu_radius
    ]

    plt.figure(figsize=(6, 6))
    im = plt.imshow(sub_thicknesses_max, cmap="viridis")
    plt.axis("off")
    plt.title("Thicknesses_max (faux couleurs)")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Valeur de l'épaisseur")
    plt.show()
    return (thicknesses_max,)


@app.cell
def _(nx, skeleton, thicknesses_max, visualize_subgraph):
    thicknesses_1 = thicknesses_max
    skeleton_1 = skeleton
    print(skeleton_1.dtype, skeleton_1.shape)
    print(thicknesses_1.dtype)
    cluster_G = nx.Graph()
    for _i in range(skeleton_1.shape[0]):
        for j in range(skeleton_1.shape[1]):
            if skeleton_1[_i, j]:
                cluster_G.add_node((_i, j), thickness=thicknesses_1[_i, j])
    for _node in cluster_G.nodes:
        (_i, j) = _node
        for ni in range(max(0, _i - 1), min(skeleton_1.shape[0], _i + 2)):
            for nj in range(max(0, j - 1), min(skeleton_1.shape[1], j + 2)):
                if (ni, nj) != (_i, j) and skeleton_1[ni, nj]:
                    cluster_G.add_edge((_i, j), (ni, nj), weight=(_i - ni) ** 2 + (j - nj) ** 2)
    visualize_subgraph(cluster_G, full_graph=True)
    return (cluster_G,)


@app.cell
def _(cluster_G, np, nx, visualize_subgraph):
    from tqdm import tqdm
    # Extract topological graph from the MST
    topological_graph = nx.MultiGraph()
    leafs = [_node for (_node, degree) in cluster_G.degree() if degree == 1]
    junctions = [_node for (_node, degree) in cluster_G.degree() if degree > 2]
    # on ne garde pas les noeuds de degré 0 qui sont des pixels isolés
    for leaf in leafs:
        topological_graph.add_node(leaf)
    for junction in junctions:
        topological_graph.add_node(junction)
    visualize_subgraph(topological_graph, full_graph=True)
    for _node in cluster_G.nodes:
        cluster_G.nodes[_node]['closed'] = False

    def is_junction(node):
        return cluster_G.degree(_node) > 2

    def is_leaf(node):
        return cluster_G.degree(_node) == 1
    for st in cluster_G.nodes:
        if cluster_G.nodes[st]['closed']:
            continue
        neighbors = list(cluster_G.neighbors(st))
        pixel_line = [st]
        found_first_endpoint = is_leaf(st) or is_junction(st)
        for neigh in neighbors:  # otherwise, we explore the graph from this node
            if cluster_G.nodes[neigh]['closed']:  # Until we either connect 2 leafs or junctions or loop
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
                        topological_graph.add_edge(pixel_line[0], pixel_line[-1], pixels=np.array(pixel_line))
                        pixel_line = [st]  # On ne pourrait les fermer que si ce sont des leaf, mais pas gravissime de pas le faire
                        break  # Par contre faut surtout pas fermer une jonction parce que il pourrait rester des trucs à explorer depuis là
                elif current_node == st:
                    assert not found_first_endpoint
                    cluster_G.nodes[current_node]['closed'] = True  # Comment ça l'endpoint est bien au début de la liste
                    topological_graph.add_edge(pixel_line[0], pixel_line[-1], pixels=np.array(pixel_line))  # On repart de l'autre coté
                    pixel_line = [st]
                    break
                nexts = list(cluster_G.neighbors(current_node))  # pas vide car si c'est une jonction on va à la prochaine itération du for repartir sur une autre branche
                next_node = nexts[1] if nexts[0] == pixel_line[-2] else nexts[0]
                cluster_G.nodes[current_node]['closed'] = True
                current_node = next_node
        cluster_G.nodes[st]['closed'] = True  # We are back to the starting point, we close the circle
    max_edges = 0  # Si on est dans un cercle impossible d'avoir trouvé un endpoint
    for (_u, _v) in topological_graph.edges():
        num_edges = topological_graph.number_of_edges(_u, _v)
        if num_edges > max_edges:  # cf au dessus même si ici ce sera normalement jamais une jonction
            max_edges = num_edges
    print("Nombre maximal d'arêtes entre deux sommets :", max_edges)
    # for st in leafs + junctions:
    #     if cluster_G.nodes[st]["closed"]:
    #         # Devrait pas arriver
    #         continue
    #     # otherwise, we explore the graph from this node (bfs)
    #     #print(f"Exploring from start node: {st}")
    #     neighbors = list(cluster_G.neighbors(st))
    #     for neigh in neighbors:
    #         #print("starting line with ", st, neigh)
    #         # On explore les différentes directions
    #         if cluster_G.nodes[neigh]["closed"] :
    #             continue
    #         current_node = neigh
    #         pixels_list = [st, current_node]
    #         while not (is_leaf(current_node) or is_junction(current_node)) :
    #             #print("current line :",pixels_list)
    #             assert cluster_G.degree(current_node) == 2
    #             nexts = list(cluster_G.neighbors(current_node))
    #             next_node = nexts[1] if nexts[0] == pixels_list[-2] else nexts[0]
    #             pixels_list.append(next_node)
    #             cluster_G.nodes[current_node]["closed"] = True
    #             current_node = next_node
    #         topological_graph.add_edge(st, current_node, pixels=np.array(pixels_list))
    #     #print("Closing node ", st)
    #     cluster_G.nodes[st]["closed"] = True
    visualize_subgraph(topological_graph, full_graph=True)  # Dans ce cas là on continue juste de suivre le chemin  # assert cluster_G.degree(current_node) == 2, f"Le noeud courant doit avoir exactement 2 voisins, il en a {cluster_G.degree(current_node)}"  # Ici, on a soit trouvé une courbe en entier, soit qu'un endpoint, et on va continuer de l'autre coté.
    return (topological_graph,)


@app.cell
def _():
    # # Pruning the MST

    # def prune_mst(mst, thicknesses):
    #     # prune in place 
    #     # Leaves are nodes with degree 1 in the MST
    #     leaves = [node for node in mst.nodes if mst.degree(node) == 1] # voir ce qu'on fait avec les noeuds de degre 0
    #     for leaf in leaves:
    #         mst.nodes[leaf]["removed_branch_length"] = 0
    #     while leaves:
    #         leaf = leaves.pop()
    #         neighbor = mst.neighbors(leaf) # there should be only one neighbor for a leaf
    #         if not neighbor:
    #             continue
    #         if mst.nodes[leaf]["removed_branch_length"] + 1 < thicknesses[leaf]:
    #             if mst.degree(neighbor) == 2:
    #                 leaves.append(neighbor)
    #                 mst.nodes[neighbor]["removed_branch_length"] = mst.nodes[leaf]["removed_branch_length"] + 1
    #             mst.remove_node(leaf)

    # prune_mst(mst, thicknesses)

    # visualize_subgraph(mst)
    return


@app.cell
def _(np, plt, test_image, topological_graph):
    from bez.bezier import fit_bezier, interpolate_bezier
    bezier_fits = []
    for (_i, edge) in enumerate(topological_graph.edges(data=True, keys=True)):
        (_u, _v, key, data) = edge
        pixels = data['pixels']
        if len(pixels) < 3:
            continue
        p = np.vstack((pixels[:, 0], pixels[:, 1]))
        instants = np.linspace(0, 1, p.shape[1])
        control_points = fit_bezier(p, instants)  # Fit a Bezier curve to the pixel line
        bezier_fits.append(((_u, _v, key), control_points))

    def visualize_bezier_fits(bezier_fits, topological_graph):
        plt.figure(figsize=(8, 8))
        plt.imshow(test_image, cmap='gray', alpha=0.2)
    # Visualize the Bezier fits
        for (_, control_points) in bezier_fits:
            t = np.linspace(0, 1, 100)
            bezier_curve = interpolate_bezier(control_points, t)  # Superpose l'image originale
            plt.plot(bezier_curve[1], bezier_curve[0], color='red', linewidth=1)
        plt.axis('equal')
        plt.title("Bezier Fits on Topological Graph (superposé à l'image originale)")
        plt.show()
    visualize_bezier_fits(bezier_fits, topological_graph)
    return


@app.cell
def _():
    """
    Pour la perturbation : choisir une des 3 opération possible
    + Changement de degré : choix d'une arrête aléatoire
    + Merge / split
    + Overlap / Dissociation


    Merge : choisir un noeud aléatoire pondérés par degré - 1 
    Split : Choix d'une hyperedge de long >= 3 puis d'un noeud non extrémité dedans, et split là dessus

    Overlap : Trouve noeud tq : extrémité d'un hyperedge appartenant à un autre, uniforme sur les autres auquel il appartient,
    puis uniforme sur 1 ou 2 voisins ce qui identifie une edge



    Si matrice non inversible renvoyer +inf



    """
    return


if __name__ == "__main__":
    app.run()
