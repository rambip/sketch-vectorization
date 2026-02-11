import marimo

__generated_with = "0.19.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Refinement
    Fonction qui va permettre de refine le topolical graphe initial, avec marge d'erreur de 2px
    """)
    return


@app.cell
def _():
    from skimage import io
    from meta import ROOT_DIR
    import matplotlib.pyplot as plt

    from bez.topo_graph import extract_topology_from_skeleton, extract_simple_topology_from_skeleton, image_to_skeleton
    from bez.refinement import  refine
    from bez.hypergraph import HyperGraph
    from bez.global_optim import optimisation
    from bez.viz import visualize_topo

    return (
        HyperGraph,
        ROOT_DIR,
        extract_simple_topology_from_skeleton,
        extract_topology_from_skeleton,
        image_to_skeleton,
        io,
        optimisation,
        plt,
        refine,
        visualize_topo,
    )


@app.cell
def _(ROOT_DIR, image_to_skeleton, io, plt):
    # Skeleton extraction
    img = io.imread(ROOT_DIR /"data/original_paper/figure_2/input.png") # [100:250, 600:780, :]

    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()

    skeletone, thikness = image_to_skeleton(img)
    plt.imshow(skeletone, cmap="gray")
    plt.axis("off")
    plt.show()
    return (skeletone,)


@app.cell
def _(
    extract_simple_topology_from_skeleton,
    extract_topology_from_skeleton,
    refine,
    skeletone,
):
    topo_graph = extract_simple_topology_from_skeleton(skeletone)
    topo_graph_ = extract_topology_from_skeleton(skeletone)
    refined = refine(topo_graph)
    return refined, topo_graph, topo_graph_


@app.cell
def _(plt, refined, topo_graph, topo_graph_, visualize_topo):
    plt.figure(figsize=(18, 6))
    _ax = plt.subplot(1, 3, 1)
    visualize_topo(topo_graph_)
    plt.title('Topology (naive)')
    _ax.text(0.5, -0.1, f'number of nodes: {len(topo_graph_.nodes())}', fontsize=10, ha='center', va='top', transform=_ax.transAxes)
    _ax = plt.subplot(1, 3, 2)
    visualize_topo(topo_graph)
    plt.title('Topology (with clustering)')
    _ax.text(0.5, -0.1, f'number of nodes: {len(topo_graph.nodes())}', fontsize=10, ha='center', va='top', transform=_ax.transAxes)
    _ax = plt.subplot(1, 3, 3)
    visualize_topo(refined)
    plt.title('Topology (refined)')
    _ax.text(0.5, -0.1, f'number of nodes: {len(refined.nodes())}', fontsize=10, ha='center', va='top', transform=_ax.transAxes)
    plt.show()
    return


@app.cell
def _(HyperGraph, refined):
    hyper = HyperGraph(refined)

    # param
    lam = 0.8
    temp = 0.5
    t_min = 0.05
    mu = 0.3
    t_decrease = 0.999 ** (1/len(refined.nodes))
    return hyper, lam, mu, t_decrease, t_min, temp


@app.cell
def _(hyper, lam, mu, optimisation, t_decrease, t_min, temp):
    error = optimisation(hyper, lam=lam, mu=mu, temp=temp, t_decrease=t_decrease, t_min=t_min)
    return (error,)


@app.cell
def _(error, plt):
    plt.plot(range(len(error)), error)
    return


@app.cell
def _(hyper, plt, skeletone):
    hyper.fit_beziers()
    plt.figure(figsize=(12, 6))
    _ax = plt.subplot(1, 2, 1)
    # Superpose l'image originale
    plt.imshow(skeletone, cmap='gray', alpha=0.2)
    hyper.visualize_fiting()
    plt.axis('equal')
    plt.title('Bezier Fits on Topological Graph')
    hyper.finition()
    _ax = plt.subplot(1, 2, 2)
    plt.imshow(skeletone, cmap='gray', alpha=0.2)
    hyper.visualize_fiting()
    plt.axis('equal')
    plt.title('Bezier Fits finilize on Topological Graph')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
