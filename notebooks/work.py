import marimo

__generated_with = "0.19.9"
app = marimo.App()


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo

    from skimage import io
    from meta import ROOT_DIR
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    from IPython.display import SVG

    from bez.topo_graph import extract_topology_from_skeleton, extract_simple_topology_from_skeleton, image_to_skeleton
    from bez.viz import visualize_topo, visualize_hyper
    from bez.refinement import  refine
    from bez.hypergraph import HyperGraph
    from bez.global_optim import optimisation
    from bez.app import generate_svg_str

    return (
        HyperGraph,
        ROOT_DIR,
        SVG,
        extract_simple_topology_from_skeleton,
        extract_topology_from_skeleton,
        generate_svg_str,
        image_to_skeleton,
        io,
        optimisation,
        plt,
        refine,
        visualize_hyper,
        visualize_topo,
    )


@app.cell
def _(
    ROOT_DIR,
    extract_simple_topology_from_skeleton,
    extract_topology_from_skeleton,
    image_to_skeleton,
    io,
    plt,
    visualize_topo,
):
    # Skeleton extraction
    img = io.imread(ROOT_DIR / 'data/original_paper/figure_1/input.png')[100:350, 300:, :]
    (skeleton, thikness) = image_to_skeleton(img)
    plt.figure(figsize=(8, 8))
    plt.imshow(1 - skeleton, cmap='gray')
    plt.title('skeleton')
    plt.show()
    topo_graph_ = extract_topology_from_skeleton(skeleton)
    topo_graph = extract_simple_topology_from_skeleton(skeleton)
    plt.figure(figsize=(12, 6))
    _ax = plt.subplot(1, 2, 1)
    visualize_topo(topo_graph_, full_graph=True)
    plt.title('Topology (naive)')
    _ax.text(0.5, -0.1, f'number of nodes: {len(topo_graph_.nodes())}', fontsize=10, ha='center', va='top', transform=_ax.transAxes)
    _ax = plt.subplot(1, 2, 2)
    visualize_topo(topo_graph, full_graph=True)
    plt.title('Topology (with clustering)')
    _ax.text(0.5, -0.1, f'number of nodes: {len(topo_graph.nodes())}', fontsize=10, ha='center', va='top', transform=_ax.transAxes)
    plt.show()
    return img, topo_graph


@app.cell
def _(plt, refine, topo_graph, visualize_topo):
    topo_graph_1 = refine(topo_graph)
    plt.figure(figsize=(6, 6))
    _ax = plt.gca()
    plt.title('After refine')
    visualize_topo(topo_graph_1, full_graph=True)
    plt.text(0.5, -0.1, f'number of nodes: {len(topo_graph_1.nodes())}', fontsize=10, ha='center', va='top', transform=_ax.transAxes)
    plt.show()
    return (topo_graph_1,)


@app.cell
def _(HyperGraph, topo_graph_1):
    hyper = HyperGraph(topo_graph_1)
    return (hyper,)


@app.cell
def _(ROOT_DIR, hyper, img, plt, visualize_hyper):
    _fig = plt.figure(figsize=(6, 6))
    plt.imshow(img, alpha=0.5)
    visualize_hyper(hyper)
    _ax = plt.gca()
    _ax.get_xaxis().set_visible(False)
    _ax.get_yaxis().set_visible(False)
    plt.title('Bezier fit, after first refine')
    _ax.text(0, -0.1, f'number of bezier: {len(hyper)}', ha='center', transform=_ax.transAxes)
    plt.savefig(ROOT_DIR / 'images' / 'bezier_fit_hut_initial.svg')
    plt.show()
    return


@app.cell
def _(hyper, optimisation, topo_graph_1):
    lam = 0.8
    temp = 0.5
    t_min = 0.05
    mu = 0.3
    t_decrease = 0.999 ** (1 / len(topo_graph_1.nodes))
    error = optimisation(hyper, lam=lam, mu=mu, temp=temp, t_decrease=t_decrease, t_min=t_min)
    return (error,)


@app.cell
def _(ROOT_DIR, error, plt):
    plt.plot(range(len(error)), error)
    plt.ylabel("Energy")
    plt.xlabel("steps")
    plt.title("evolution of energy during optimization")
    plt.savefig(ROOT_DIR / "images" / "evolution_energy_hut.svg")
    plt.show()
    return


@app.cell
def _(ROOT_DIR, hyper, img, plt, visualize_hyper):
    _fig = plt.figure(figsize=(6, 6))
    plt.imshow(img, alpha=0.5)
    visualize_hyper(hyper)
    _ax = plt.gca()
    _ax.get_xaxis().set_visible(False)
    _ax.get_yaxis().set_visible(False)
    plt.title('Bezier fit, after global optimization')
    _ax.text(0, -0.1, f'number of bezier: {len(hyper)}', ha='center', transform=_ax.transAxes)
    plt.savefig(ROOT_DIR / 'images' / 'bezier_fit_hut_opti.svg')
    plt.show()
    return


@app.cell
def _(SVG, generate_svg_str, hyper, img):
    SVG(generate_svg_str(img.shape, hyper))
    return


if __name__ == "__main__":
    app.run()
