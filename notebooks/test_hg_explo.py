import marimo

__generated_with = "0.19.9"
app = marimo.App()


app._unparsable_cell(
    r"""
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo

    from meta import ROOT_DIR
    from bez.segments import segments
    from skimage import io
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy
    import networkx as nx
    from bez.topo_graph import * 

    from bez.hypergraph import HyperGraph, HYPER, EDGE,NODE
    """,
    name="_"
)


@app.cell
def _(extract_topology_from_skeleton, np):
    skeleton = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=bool
    )

    topo = extract_topology_from_skeleton(skeleton)

    print(*topo.edges, sep='\n')
    print(topo.edges[(3,1), (4,4),0])
    return (topo,)


@app.cell
def _(EDGE, HYPER, HyperGraph, node_nodes, nx, plt, topo):
    h = HyperGraph(topo)
    print(*list(topo.adjacency()),sep='\n')
    print(*h.table.items(),sep="\n" )

    tp = h.hyper2edge

    # Get node types
    nodes = tp.nodes()
    # Separate nodes by type
    hyper_nodes = [n for n in nodes if n[0] == HYPER]
    edge_nodes = [n for n, t in nodes if t == EDGE]

    print("Hyper nodes:", hyper_nodes)
    # Assign x positions for each type
    pos = {}
    for i, nodes in enumerate([hyper_nodes, edge_nodes, node_nodes]):
        for j, n in enumerate(nodes):
            print(n)
            pos[n] = (i, -j)


    plt.figure(figsize=(8, 8))
    nx.draw(tp, pos, with_labels=True, node_size=500, font_size=10, font_color='white', node_color='blue')
    plt.title("Tripartite Graph (columns by type)")

    plt.show()
    return


if __name__ == "__main__":
    app.run()
