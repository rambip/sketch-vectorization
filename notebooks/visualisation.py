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
    import os
    import numpy as np
    from PIL import Image

    from bez.topo_graph import extract_topology_from_skeleton, extract_simple_topology_from_skeleton, image_to_skeleton
    from bez.refinement import  refine
    from bez.hypergraph import HyperGraph
    from bez.global_optim import optimisation
    from bez.viz import visualize_topo, visualize_topo_and_save, visualize_graph_hypergraph

    return (
        HyperGraph,
        Image,
        ROOT_DIR,
        extract_simple_topology_from_skeleton,
        extract_topology_from_skeleton,
        image_to_skeleton,
        io,
        np,
        optimisation,
        os,
        plt,
        refine,
        visualize_graph_hypergraph,
        visualize_topo,
        visualize_topo_and_save,
    )


@app.cell
def _(Image, ROOT_DIR, image_to_skeleton, io, np, plt):
    # Skeleton extraction
    img = io.imread(ROOT_DIR /"data/original_paper/figure_2/input.png") [170:210, 680:730, :]

    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()

    skeletone, thikness = image_to_skeleton(img)


    plt.imshow(skeletone, cmap="gray")  # ou cmap="binary", ou cmap="Greens", selon ce que tu veux
    plt.axis("off")
    plt.show()


    rgba = np.zeros((skeletone.shape[0], skeletone.shape[1], 4), dtype=np.uint8)
    rgba[skeletone] = [0, 0, 0, 255] 

    out_path = ROOT_DIR / "images/squelette_transparent.png"
    Image.fromarray(rgba).save(out_path)
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
def _(ROOT_DIR, os, plt, refined, topo_graph, topo_graph_, visualize_topo):
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
    _folder = os.path.join(ROOT_DIR / 'images')
    os.makedirs(_folder, exist_ok=True)
    _save_path = os.path.join(_folder, 'topo_graph_evolution.png')
    plt.savefig(_save_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()
    return


@app.cell
def _(ROOT_DIR, os, refined, visualize_topo_and_save):
    _folder = os.path.join(ROOT_DIR / 'images')
    os.makedirs(_folder, exist_ok=True)
    _save_path = os.path.join(_folder, 'refined_topo_graph.png')
    visualize_topo_and_save(refined, save_path=_save_path)
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
def _(ROOT_DIR, hyper, os, visualize_graph_hypergraph):
    _folder = os.path.join(ROOT_DIR / 'images')
    os.makedirs(_folder, exist_ok=True)
    _save_path = os.path.join(_folder, 'hyper_graph.png')
    visualize_graph_hypergraph(hyper, save_path=_save_path)
    return


@app.cell
def _(ROOT_DIR, hyper, os, plt, skeletone):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    hyper.fit_beziers()
    plt.figure(figsize=(12, 6))
    _ax = plt.subplot(1, 2, 1)
    plt.imshow(skeletone, cmap='gray', alpha=0.2)
    # Superpose l'image originale
    hyper.visualize_fiting()
    plt.axis('equal')
    plt.title('Basic Bezier Fits')
    axins1 = inset_axes(_ax, width='20%', height='20%', loc='lower left')
    hyper.visualize_fiting(axins1)
    axins1.set_xlim(160, 200)
    axins1.set_ylim(380, 420)
    axins1.get_xaxis().set_visible(False)
    axins1.get_yaxis().set_visible(False)
    axins1.invert_yaxis()
    mark_inset(_ax, axins1, loc1=2, loc2=4, fc='none', ec='0.5')
    axins2 = inset_axes(_ax, width='20%', height='20%', loc='upper right')
    hyper.visualize_fiting(axins2)
    axins2.set_xlim(580, 620)
    axins2.set_ylim(480, 520)
    axins2.get_xaxis().set_visible(False)
    axins2.get_yaxis().set_visible(False)
    axins2.invert_yaxis()
    mark_inset(_ax, axins2, loc1=2, loc2=4, fc='none', ec='0.5')
    hyper.finition()
    _ax = plt.subplot(1, 2, 2)
    plt.imshow(skeletone, cmap='gray', alpha=0.2)
    hyper.visualize_fiting()
    plt.axis('equal')
    plt.title('finilized Bezier Fits')
    axins1 = inset_axes(_ax, width='20%', height='20%', loc='lower left')
    hyper.visualize_fiting(axins1)
    axins1.set_xlim(160, 200)
    axins1.set_ylim(380, 420)
    axins1.get_xaxis().set_visible(False)
    axins1.get_yaxis().set_visible(False)
    axins1.invert_yaxis()
    mark_inset(_ax, axins1, loc1=2, loc2=4, fc='none', ec='0.5')
    axins2 = inset_axes(_ax, width='20%', height='20%', loc='upper right')
    hyper.visualize_fiting(axins2)
    axins2.set_xlim(580, 620)
    axins2.set_ylim(480, 520)
    axins2.get_xaxis().set_visible(False)
    axins2.get_yaxis().set_visible(False)
    axins2.invert_yaxis()
    mark_inset(_ax, axins2, loc1=2, loc2=4, fc='none', ec='0.5')
    _folder = os.path.join(ROOT_DIR / 'images')
    os.makedirs(_folder, exist_ok=True)
    _save_path = os.path.join(_folder, 'topo_graph_evolution.png')
    plt.savefig(_save_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
