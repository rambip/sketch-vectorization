import marimo

__generated_with = "0.19.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Walkthrough of the pipeline

    This project aims to (re-)implement this paper: [Fidelity vs. Simplicity: a Global Approach to Line Drawing Vectorization](https://www-sop.inria.fr/reves/Basilic/2016/FLB16/fidelity_simplicity.pdf)

    In this notebook, we will try to give a complete overview of the techniques used and the implementation choices in order to do it.

    We strongly advise to read the paper a first time before looking at the notebook. Except for some details, it is well written and have nice illustrations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pipeline

    Let's start with the central idea: we want to take a photography of a sketch as input, and return a SVG made up of a few curves

    ![](../images/illustration_butterfly.png)

    In order, we
    - apply **preprocessing** to have a information about the pixels
    - create a **skeleton** and convert it to a graph (chains of pixels)
    - use **hypergraph optimization** to find the optimal curves to approximate these chains of pixels
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preprocessing

    Before anything, we need a binary representation of our drawing.

    The paper uses advanced trapped-ball techniques, but we chose to implement something simple. We expected to get results a lot worse than the paper, but most of the time the results are ok. For this reason, we did not spend more time implementing the contour detection phase.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from meta import ROOT_DIR
    from scipy import ndimage as ndi
    from bez.data import load_normalized

    from skimage import io, data, morphology
    from skimage.morphology import dilation, erosion, remove_small_holes, remove_small_objects
    from skimage.morphology import disk
    from skimage.filters import threshold_otsu, threshold_local, threshold_mean, rank, gaussian, gabor_kernel

    return ROOT_DIR, dilation, disk, erosion, load_normalized, ndi, np, plt


@app.cell
def _(ROOT_DIR, load_normalized, plt):
    img = load_normalized(ROOT_DIR / "data/sketches/house.png")
    plt.imshow(img, cmap="binary")
    plt.colorbar()
    plt.show()
    return (img,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are 3 challenges when we want to find the contours of the drawing:
    - the pen creates a complex texture on the image (and this depends on the pen)
    - the thickness of the line varies a lot in the drawing
    - some lines are "frayed", with multiple strokes approximately at the same place.

    Let's zoom in at one particular location:
    """)
    return


@app.cell
def _():
    from bez.cnn import load_trained_model

    return (load_trained_model,)


@app.cell
def _(load_trained_model):
    model = load_trained_model()
    return (model,)


@app.cell
def _(img, model, np, plt):
    img_binary = model.run(["output"], {"input": img[np.newaxis, np.newaxis, :, :]})[0][0][0] > 0.5
    plt.imshow(img_binary)
    plt.colorbar()
    plt.show()
    return (img_binary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now have black and white pixels, but we did not solve the issue of multiple strokes. And a new artefact appeared: small dots everywhere.
    We follow the methodoloy of the paper and perform a dilation.

    But we don't know what diameter to use !
    In order to get it, we first it to compute the thickness map.

    We erode the image iteratively, and for each pixel we keep track of the moment at which it disappeared.
    """)
    return


@app.cell
def _(erosion, img_binary, np, plt):
    thicknesses = np.zeros_like(img_binary, dtype=int)
    max_line_width_detection = img_binary.copy()
    i = 0
    while np.sum(max_line_width_detection) > 0:
        tmp = erosion(max_line_width_detection)
        diff = max_line_width_detection ^ tmp > 0  # iterate the erosion
        thicknesses[diff] = i
        max_line_width_detection = tmp  # find the difference
        i = i + 1
    plt.imshow(thicknesses)
    plt.title('Thickness of the drawing')
    plt.show()
    return (thicknesses,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We barely see the thickness. We will make it more visible by finding the maximum values of thickness locally.
    """)
    return


@app.cell
def _(disk, ndi, plt, thicknesses):
    thickness_for_display = ndi.maximum_filter(thicknesses, footprint=disk(5))
    plt.imshow(thickness_for_display)
    plt.title("Thickness of the drawing (after local maxima)")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we can compute the value of the dilation parameter.

    We could chose the maximum thickness, but if the thickness of the drawing is not uniform, we will a lot of details in regions where the thickness is very small.

    After some experimentation, we decided to take the median of all thickness values.
    """)
    return


@app.cell
def _(dilation, disk, img_binary, plt):
    c = 1

    # remove holes, objects and dilate
    img_binary_dilated = dilation(img_binary, disk(c))
    plt.imshow(img_binary_dilated)
    plt.title("final drawing after dilation")
    plt.show()
    return (img_binary_dilated,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Skeleton and topological graph

    Once we have a nice black and white image, we can create the skeleton very easily using iterative thining.
    """)
    return


@app.cell
def _():
    from skimage.morphology import skeletonize
    import scipy
    from bez.topo_graph import extract_simple_topology_from_skeleton, extract_topology_from_skeleton
    from bez.viz import visualize_topo

    return (
        extract_simple_topology_from_skeleton,
        extract_topology_from_skeleton,
        scipy,
        skeletonize,
        visualize_topo,
    )


@app.cell
def _(img_binary_dilated, plt, skeletonize):
    skeleton = skeletonize(img_binary_dilated, method="zhang")

    plt.imshow(1-skeleton, cmap="gray")
    plt.title("skeleton")
    plt.show()
    return (skeleton,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once we have this graph, we traverse it in order to get the pixel chains.

    This is not explained explicitely in the original paper, but the trick is to look at the **number of neighbours** in the skeleton. Indeed, most nodes in the skeleton have only 3 neighbours. If it is not the case, we call it a "special" pixel
    """)
    return


@app.cell
def _(np, plt, scipy, skeleton):
    kernel = np.ones((3, 3), dtype=int)
    neighbours = scipy.signal.convolve2d(skeleton.astype(int), kernel, mode='same')
    neighbours = neighbours * skeleton
    plt.imshow(1 - skeleton, cmap='gray')
    plt.scatter(np.where(neighbours == 2)[1], np.where(neighbours == 2)[0], label='2 neighbours', s=1)
    plt.scatter(np.where(neighbours >= 4)[1], np.where(neighbours >= 4)[0], label='4 neighbours', s=1)
    plt.scatter(np.where(neighbours > 4)[1], np.where(neighbours > 4)[0], label='>4 neighbours', s=1)
    plt.legend()
    plt.title('skeleton with number of neighbours')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use this information to perform a graph traversal, from arbitrary pixels to "special" pixels.
    For the special case of loops, we break them into a chain that ends where it starts.
    """)
    return


@app.cell
def _(extract_topology_from_skeleton, plt, skeleton, visualize_topo):
    topo_graph = extract_topology_from_skeleton(skeleton)
    visualize_topo(topo_graph, visu_radius=5, opacity=0.2)
    print(f"number of nodes: {len(topo_graph.nodes())}")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This graph can be used for the next steps. Unfortunately, we can see that some groups of pixels are clustered together, which will cause a lot of problems.

    With a modified algorithm, we are able to merge some of these clusters together.
    """)
    return


@app.cell
def _(extract_simple_topology_from_skeleton, plt, skeleton, visualize_topo):
    topo_graph_1 = extract_simple_topology_from_skeleton(skeleton)
    visualize_topo(topo_graph_1, visu_radius=5, opacity=0.2)
    print(f'number of nodes: {len(topo_graph_1.nodes())}')
    plt.show()
    return (topo_graph_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hyper graph and bezier fiting

    Once we have our hypergraph, we can start to use bezier curves to approximate it.
    Since it is a simple least square problem, we have derived the closed solution in the form of a linear system to solve.
    The code below shows this logic.
    Here $q(t)$ is a vector defined as
    $$
    q(t) = \begin{pmatrix}
    (1-t)^3 \\ 3 t (1-t)^2 \\ 2 t^2 (1-t) \\ t^3
    \end{pmatrix}
    $$

    ```python
    def fit_bezier(traj, t):
        traj_x, traj_y = traj
        m = np.zeros((4, 4))
        bx = np.zeros(4)
        by = np.zeros(4)

        # this equation follows from the minimization of the pixel standard distance
        for i in range(k):
            q = interpolant(t, degree)
            m[i] = np.sum(weights * (q * q[i]), axis=1)
            bx[i] = np.sum(weights * (traj_x * q[i]))
            by[i] = np.sum(weights * (traj_y * q[i]))

        # We use lstsq because the matrix is not always full-rank.
        # For example, if the degree of the bezier is greater than the length of "traj"
        x_fit = np.linalg.lstsq(m, bx)[0]
        y_fit = np.linalg.lstsq(m, by)[0]
        return np.array([x_fit, y_fit])
    ```
    """)
    return


@app.cell
def _():
    from bez.refinement import  refine
    from bez.hypergraph import HyperGraph
    from bez.viz import visualize_hyper
    from bez.global_optim import optimisation, fit_hyperedge

    return HyperGraph, fit_hyperedge, optimisation, refine, visualize_hyper


@app.cell
def _(HyperGraph, topo_graph_1):
    hyper = HyperGraph(topo_graph_1)
    return (hyper,)


@app.cell
def _(fit_hyperedge, hyper, img, plt, visualize_hyper):
    for _h in hyper.all_hyperedges():
        fit_hyperedge(_h)
    visualize_hyper(hyper, offset=0)
    plt.imshow(img, alpha=0.5, cmap='gray')
    _ax = plt.gca()
    _ax.axis('off')
    plt.title('Initial hyper-graph')
    _ax.text(0.5, -0.1, f'number of bezier: {len(hyper)}', ha='center', va='top', transform=_ax.transAxes)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The main issue with this initial hypergraph is that some curves should be broken down to better fit the image.
    We implement the next part of the paper, a dichotomy-based refinement.
    """)
    return


@app.cell
def _(
    HyperGraph,
    fit_hyperedge,
    img,
    plt,
    refine,
    topo_graph_1,
    visualize_hyper,
):
    topo_graph_refine = refine(topo_graph_1, 0.1)
    hyper_1 = HyperGraph(topo_graph_refine)
    for _h in hyper_1.all_hyperedges():
        fit_hyperedge(_h)
    visualize_hyper(hyper_1, offset=0)
    plt.imshow(img, alpha=0.5, cmap='gray')
    _ax = plt.gca()
    _ax.axis('off')
    plt.title('After refine')
    plt.text(0.5, -0.1, f'number of bezier: {len(hyper_1)}', fontsize=10, ha='center', va='top', transform=_ax.transAxes)
    plt.show()
    return (hyper_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we have to implement what is probably the most difficult part of the paper: the global optimisation.


    We will manipulate sequences of edges, that we will call **hyperedges**. The intuition if they are multiple chains of pixels aligned with each other in the topological graph, they should be grouped together to form a longer chain.

    This optimisation process is based on the idea of doing **perturbations** on these sequences.

    Since the paper is not very explicit about these perturbations, we implemented it in a slightly different (but similar) way.

    We first sample pairs of hyperedges $(U, V)$ such that $U$ end at a node that is anywhere inside $V$. We call such a configuation a "T" configuration.

    ![](../images/schema_t.svg)

    Note that hyperedges are oriented. Once we sample one "T", we can apply 6 different transformations:

    - increase or decrease the degree of $V$
    - splitting V into 2 (if it has more than 3 nodes)
    - overlaping V onto U (here, $V$ will become $(y, x, c, d)$)
    - dissociating V from U (for that, the last edge of V must be in U)
    - merging V and U. This is only possible when the T is in the special configuration showed below:

    ![](../images/schema_t2.svg)

    We add a special transformation: **reverse**. As the name suggests, we reverse the order of all nodes in the sequence.

    With the addition of this transformation, this is completely equivalent to the transformations proposed by the paper.


    We try to chose the same parameters as the paper each time we can, to be able to compare results.
    """)
    return


@app.cell
def _(hyper_1, optimisation, topo_graph_1):
    lam = 0.2
    temp = 0.5
    t_min = 0.05
    mu = 0.2
    t_decrease = 0.99 ** (1 / len(topo_graph_1.nodes))
    error = optimisation(hyper_1, lam=lam, mu=mu, temp=temp, t_decrease=t_decrease, t_min=t_min)
    return (error,)


@app.cell
def _(error, plt):
    plt.plot(range(len(error)), error)
    plt.ylabel("Energy")
    plt.xlabel("steps")
    plt.title("evolution of energy during optimization")
    plt.show()
    return


@app.cell
def _(hyper_1, img, plt, visualize_hyper):
    visualize_hyper(hyper_1)
    plt.imshow(img, alpha=0.5, cmap='gray')
    _ax = plt.gca()
    _ax.axis('off')
    plt.title('Bezier fit, after global optimization')
    _ax.text(0.5, -0.1, f'number of bezier: {len(hyper_1)}', ha='center', transform=_ax.transAxes)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we look closely, we see that some lines do not join exactly at the same point. It is logical, since every curve fitting is independent.
    The way to fix this problem is specified in the paper, but not exactly.

    We chose to implement what seemed like a reasonnable solution.

    First for the point-edges junctions, we look again at all our "T" configurations (see above) and force the control point of the incoming curve to fall exactly on the other curve.

    For the point-point junctions, we compute the barycenter $B$ of all the points that are supposed to arrive at the same position, and we force the control point of each incoming edge to be $B$.
    """)
    return


@app.cell
def _(hyper_1, img, plt, visualize_hyper):
    hyper_1.finition()
    visualize_hyper(hyper_1)
    plt.imshow(img, alpha=0.5, cmap='gray')
    _ax = plt.gca()
    _ax.axis('off')
    plt.title('Bezier fit, after finition')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SVG export
    """)
    return


@app.cell
def _():
    from IPython.display import SVG
    from bez.app import generate_svg_str

    return SVG, generate_svg_str


@app.cell
def _(SVG, generate_svg_str, hyper_1, img):
    out = generate_svg_str(img.shape, hyper_1, stroke_width=2)
    SVG(out)
    return


@app.cell
def _():
    from bez.app import show_example

    return (show_example,)


@app.cell
def _(ROOT_DIR, show_example):
    show_example(ROOT_DIR / "data/original_paper/figure_2/input.png")
    return


@app.cell
def _(ROOT_DIR, show_example):
    show_example(ROOT_DIR / "data/original_paper/figure_1/input.png")
    return


@app.cell
def _(ROOT_DIR, show_example):
    show_example(ROOT_DIR / "data/sketches/triangle.png")
    return


@app.cell
def _(ROOT_DIR, show_example):
    show_example(ROOT_DIR / "data/sketches/dress.png")
    return


@app.cell
def _(ROOT_DIR, show_example):
    show_example(ROOT_DIR / "data/CAD_dataset/Dataset_B/ESB_Sketches/90 degree elbows/001_1.png")
    return


@app.cell
def _(ROOT_DIR, show_example):
    show_example(ROOT_DIR / "data/CAD_dataset/Dataset_B/ESB_Sketches/U shaped parts/010_1.png")
    return


@app.cell
def _(ROOT_DIR, show_example):
    show_example(ROOT_DIR / "data/original_paper/figure_10/archi.png")
    return


@app.cell
def _(ROOT_DIR, show_example):
    show_example(ROOT_DIR / "data/original_paper/figure_14/bag/input.png")
    return


if __name__ == "__main__":
    app.run()
