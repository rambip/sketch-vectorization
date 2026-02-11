import marimo

__generated_with = "0.19.9"
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

    from skimage import io, data, morphology
    from skimage.morphology import dilation, erosion, remove_small_holes, remove_small_objects
    from skimage.morphology import disk
    from skimage.filters import threshold_otsu, threshold_local, rank, gaussian, gabor_kernel
    from skimage.color import rgb2gray
    from skimage.transform import rescale

    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    return (
        ROOT_DIR,
        dilation,
        disk,
        erosion,
        gabor_kernel,
        gaussian,
        io,
        morphology,
        ndi,
        np,
        plt,
        remove_small_holes,
        remove_small_objects,
        rescale,
        rgb2gray,
        threshold_otsu,
    )


@app.cell
def _(ROOT_DIR, io):
    img_raw = io.imread(ROOT_DIR /"data/sketches/butterfly.png")
    img_raw.shape
    return (img_raw,)


@app.cell
def _(img_raw, plt, rescale, rgb2gray):
    # depenging on the encoding of the image, we don't convert it into gray scale the same way.
    if len(img_raw.shape) == 2:
        img = img_raw
    elif img_raw.shape[2] == 2:
        img = img_raw[:, :, 0]
    elif img_raw.shape[2] == 3:
        img = (rgb2gray(img_raw)*255).astype(int)
    elif img_raw.shape[2] == 4:
        img = (rgb2gray(img_raw[:, :, :3]) * (img_raw[:, :, 3])).astype(int)
    else:
        raise ValueError(f"invalid image shape: {img_raw.shape}")

    size = min(img.shape[0], img.shape[1])
    img = 1-rescale(img, 200/size)
    print(f"shape: {img.shape}")

    plt.imshow(img, cmap="binary")
    plt.colorbar()
    plt.show()
    return img, size


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
def _(gaussian, gray_scaled, ndi, np, rotate_hessian):
    def response(Hxx, Hyy, Hxy, angle_deg):
        """
        Rotate the Hessian matrix to detect lines at any inclination.
    
        For a rotation by angle θ, the rotated Hessian components are:
        H'xx = Hxx*cos²θ + Hyy*sin²θ + 2*Hxy*sinθ*cosθ
        H'yy = Hxx*sin²θ + Hyy*cos²θ - 2*Hxy*sinθ*cosθ  
        H'xy = (Hyy-Hxx)*sinθ*cosθ + Hxy*(cos²θ-sin²θ)
        """
        theta = np.radians(angle_deg)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos2_theta = cos_theta**2
        sin2_theta = sin_theta**2
        sin_cos_theta = sin_theta * cos_theta
    
        # Rotated Hessian components
        return -(Hxx * cos2_theta + Hyy * sin2_theta + 2 * Hxy * sin_cos_theta)

    angles_to_test = np.arange(0, 180, 10)
    
    def clean(img, sigma):
        if sigma is not None:
            blured = gaussian(img, sigma=sigma)
        else:
            blured = img
    
        gx = ndi.sobel(blured, axis=1)
        gy = ndi.sobel(blured, axis=0)
        hxx = ndi.sobel(gx, axis=1)
        hyy = ndi.sobel(gy, axis=0)
        hxy = 0.5*(ndi.sobel(gx, axis=0) + ndi.sobel(gy, axis=1))
    
        max_resp = np.zeros_like(gray_scaled)
        for angle in angles_to_test:
            response = rotate_hessian(hxx, hyy, hxy, angle)
            max_resp = np.maximum(max_resp, response)
        return max_resp

    return (clean,)


@app.cell
def _(clean, gaussian, img, ndi, plt, threshold_otsu):
    plt.imshow(img)
    plt.show()
    ci = clean(img, 1.5)
    plt.imshow(ci)
    plt.colorbar()
    plt.show()

    ci = gaussian(ci)
    gx = ndi.sobel(ci, axis=1)
    gy = ndi.sobel(ci, axis=0)
    hxx = ndi.sobel(gx, axis=1)
    hyy = ndi.sobel(gy, axis=0)
    hxy = ndi.sobel(gx, axis=0)
    test = -(hxx+hyy)
    t = threshold_otsu(test)
    plt.imshow(test)
    plt.colorbar()
    plt.show()
    plt.imshow(test > t)
    plt.show()
    return


@app.cell
def _(filters, img, ndi, np, plt, rotate_hessian, threshold_otsu):
    #def detect_lines_at_angle(Hxx, Hyy, Hxy, angle_deg, threshold_factor=2.0):
    #    """
    #    Detect lines at a specific angle using rotated Hessian.
    #    Lines perpendicular to the rotation angle will be enhanced in Hxx_rot.
    gray_image = filters.gaussian(img, sigma=1.5)
    #    Hxx_rot, Hyy_rot, Hxy_rot = rotate_hessian(Hxx, Hyy, Hxy, angle_deg)
    #    
    #    # For line detection, we want strong negative values in the direction 
    #    # perpendicular to the line (indicating the line is a ridge)
    #    threshold = -np.std(Hxx_rot) * threshold_factor
    #    line_response = np.where(Hxx_rot < threshold, -Hxx_rot, 0)
    grad_x = ndi.sobel(gray_image, axis=1)
    #    return line_response, Hxx_rot, Hyy_rot, Hxy_rot
    grad_y = ndi.sobel(gray_image, axis=0)
    Hxx = ndi.sobel(grad_x, axis=1)
    # Apply Gaussian smoothing
    Hyy = ndi.sobel(grad_y, axis=0)
    Hxy = ndi.sobel(grad_x, axis=0)
    # Calculate original Hessian components
    angles = [0, 30, 45, 60, 90, 120, 135, 150]
    (fig, axes) = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    axes[0].imshow(gray_image, cmap='gray')  # Detects vertical lines
    axes[0].set_title('Original Image')  # Detects horizontal lines  
    axes[0].axis('off')  # Mixed derivative
    for (i, angle) in enumerate([0, 30, 45, 60, 90, 120, 135]):
    # Test different angles
        response_1 = rotate_hessian(Hxx, Hyy, Hxy, angle)
        axes[i + 3].imshow(response_1)
    # Create comprehensive visualization
        axes[i + 3].set_title(f'Lines at {angle}°\n(Rotated Hessian)')
        axes[i + 3].axis('off')
    max_response = 0.5 * np.ones(gray_image.shape)
    # Original image
    angle_map = np.zeros_like(gray_image)
    for angle in range(0, 180, 10):
        response_1 = rotate_hessian(Hxx, Hyy, Hxy, angle)
        mask = response_1 > max_response
    # Line detection at different angles
        max_response[mask] = response_1[mask]
        angle_map[mask] = angle
    axes[9].imshow(max_response)
    axes[9].set_title('Maximum Line Response\n(All Angles)')
    axes[9].axis('off')
    axes[10].imshow(angle_map, cmap='hsv', vmin=0, vmax=180)
    axes[10].set_title('Dominant Line Orientation\n(Color = Angle)')
    # Multi-angle line detection (maximum response across all angles)
    axes[10].axis('off')
    (fig2, axes2) = plt.subplots(2, 3, figsize=(18, 12))
    sigmas = [0.5, 1.0, 2.0]
    angles_to_test_1 = [0, 30, 60, 90, 120, 150]  # Test every 10 degrees
    for (i, sigma) in enumerate(sigmas):
        gray_scaled = filters.gaussian(img, sigma=sigma)
        gx_1 = ndi.sobel(gray_scaled, axis=1)  # Keep track of maximum response and corresponding angle
        gy_1 = ndi.sobel(gray_scaled, axis=0)
        hxx_1 = ndi.sobel(gx_1, axis=1)
        hyy_1 = ndi.sobel(gy_1, axis=0)
        hxy_1 = ndi.sobel(gx_1, axis=0)
        max_resp = np.zeros_like(gray_scaled)
        for angle in angles_to_test_1:
            response_1 = rotate_hessian(hxx_1, hyy_1, hxy_1, angle)
            max_resp = np.maximum(max_resp, response_1)
        axes2[0, i].imshow(gray_scaled, cmap='gray')
        axes2[0, i].set_title(f'Smoothed (σ={sigma})')
        axes2[0, i].axis('off')
        p = axes2[1, i].imshow(max_resp)
    # Enhanced multi-scale approach
        fig.colorbar(p)
        axes2[1, i].set_title(f'Multi-angle Lines (σ={sigma})')
    # Different scales (sigma values)
        axes2[1, i].axis('off')
    plt.tight_layout()
    plt.show()
    gray_scaled = filters.gaussian(img, sigma=1)
    gx_1 = ndi.sobel(gray_scaled, axis=1)  # Apply Gaussian smoothing at different scales
    gy_1 = ndi.sobel(gray_scaled, axis=0)
    hxx_1 = ndi.sobel(gx_1, axis=1)
    hyy_1 = ndi.sobel(gy_1, axis=0)  # Calculate Hessian at this scale
    hxy_1 = ndi.sobel(gx_1, axis=0)
    max_resp = np.zeros_like(gray_scaled)
    for angle in angles_to_test_1:
        response_1 = rotate_hessian(hxx_1, hyy_1, hxy_1, angle)
        max_resp = np.maximum(max_resp, response_1)
    t_1 = threshold_otsu(max_resp)
    plt.imshow(max_resp > t_1)  # Multi-angle detection
    plt.show()
    print('Rotation Matrix Theory:')
    print('For a rotation by angle θ, the transformation is:')
    print('R(θ) = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]')
    print()
    print("The rotated Hessian H' = R(θ)ᵀ * H * R(θ) gives:")
    print("H'xx = Hxx*cos²θ + Hyy*sin²θ + 2*Hxy*sinθ*cosθ")
    print("H'yy = Hxx*sin²θ + Hyy*cos²θ - 2*Hxy*sinθ*cosθ")
    print("H'xy = (Hyy-Hxx)*sinθ*cosθ + Hxy*(cos²θ-sin²θ)")
    print()
    print('Lines at angle α are detected by rotating the coordinate system by α')
    # Demonstrate the rotation matrix effect
    print('and examining the Hxx component of the rotated Hessian.')
    return (gray_scaled,)


@app.cell
def _(img, ndi, np, plt, threshold_otsu):
    # Load and normalize image
    image = (255 - 255 * img).astype(int)
    plt.imshow(image[20:60, 130:170])
    plt.show()
    shapes = [np.array([-1, 0, 1, 0, -1, -1, 0, 1, 0, -1, -1, 0, 0, 0, -1, -1, 0, 1, 0, -1, -1, 0, 1, 0, -1]).reshape(5, 5)]
    # 8-connected neighborhood directions (excluding center)
    new_shapes = []
    for s in shapes:
        new = np.rot90(s)
        new_shapes.append(new.copy())
    shapes.extend(new_shapes)
    out = np.max([ndi.convolve(image, s) for s in shapes], axis=0)
    out = np.max([ndi.convolve(out, s) for s in shapes], axis=0)
    t_2 = threshold_otsu(out)
    plt.imshow(out)
    plt.show()
    plt.imshow(out > t_2)
    return image, shapes


@app.cell
def _(image, morphology, np, plt, shapes):
    current = image.copy()
    for i_1 in range(10):
        current = np.max([morphology.erosion(current, s) for s in shapes] + [current], axis=0)
    plt.imshow(current)
    return


app._unparsable_cell(
    r"""
    dilate(
    shifts = np.array([
    ])
    current = image
    for i in range(10): 
        mins = []
        for shift in shifts:
            mins.append(np.min([np.roll(current, s, axis=(0, 1)) for s in shift], axis=0))
        current = current + np.max(mins, axis=0)
        t = threshold_otsu(current)
        plt.imshow(current)
        plt.colorbar()
        plt.show()
    """,
    name="_"
)


@app.cell
def _(image, np, plt):
    dirs8 = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if i or j]
    dirs25 = [(i, j) for i in [-2, -1, 0, 1, 2] for j in [-2, -1, 0, 1, 2] if i or j if i == 0 or j == 0]

    def edge_detect(img, dirs, sign):
        """Apply edge detection using 8-connected neighbors"""
        neighbor_diffs = [np.sign(np.roll(img, shifts, axis=(0, 1)) - img) for shifts in dirs]
        return np.sign(((-1 + np.sum(neighbor_diffs, axis=0)) / 5).round())
    current_1 = image
    for iteration in range(10):
    # Apply edge detection iteratively
        current_1 = current_1 + edge_detect(current_1, dirs8, 1)
    plt.imshow(current_1)
    plt.show()
    for iteration in range(10):
        current_1 = current_1 + edge_detect(current_1, dirs25, 1)
    plt.imshow(current_1)
    plt.show()
    return


@app.cell
def _(np):
    np.sign(1)
    return


@app.cell
def _(gabor_kernel, img, ndi, np, plt):
    from scipy import ndimage as ndk
    kernels = [np.real(gabor_kernel(1/3, theta=np.pi/6*k)) for k in range(6)]
    img_gab = np.any(np.array([ndi.convolve(img, ke) < 0 for ke in kernels]), axis=0)
    sub_image = img_gab[20:60, 130:170]
    plt.imshow(sub_image, cmap="gray")
    plt.colorbar()
    plt.title("Same detail, blured")
    plt.show()
    return


@app.cell
def _(img, plt):
    sub_image_1 = img[20:60, 130:170]
    plt.imshow(sub_image_1, cmap='gray')
    plt.colorbar()
    plt.title('detail of the wing')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To solve the problem of the texture, we use a gaussian filter. This will blur our image to get a more uniform texture.

    Then, we use a local thresholding: for each pixel, we look at a small region around it, we compute a threshold, and we look if the value is greater than it.
    """)
    return


@app.cell
def _(gabor_kernel, img, ndi, np, plt):
    kernels_1 = [np.real(gabor_kernel(1 / 2, theta=np.pi / 12 * k)) for k in range(12)]
    img_gab_1 = np.mean(np.array([ndi.convolve(1 - img, ke) for ke in kernels_1]), axis=0)
    sub_image_2 = img_gab_1[20:60, 130:170]
    plt.imshow(sub_image_2, cmap='gray')
    plt.colorbar()
    plt.title('Same detail, blured')
    plt.show()
    return


@app.cell
def _(block_size, img, plt, threshold_otsu):
    #block_size = 2*(size // 20)+1
    t_3 = threshold_otsu(img, block_size)
    img_binary = img < t_3
    plt.imshow(img_binary)
    plt.title('Image converted to black and white')
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
    i_2 = 0
    while np.sum(max_line_width_detection) > 0:
        tmp = erosion(max_line_width_detection)
        diff = max_line_width_detection ^ tmp > 0  # iterate the erosion
        thicknesses[diff] = i_2
        max_line_width_detection = tmp  # find the difference
        i_2 = i_2 + 1
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
def _(disk, ndi, plt, size, thicknesses):
    thicknesses_1 = ndi.maximum_filter(thicknesses, footprint=disk(size / 100))
    plt.imshow(thicknesses_1)
    plt.title('Thickness of the drawing (after local maxima)')
    plt.show()
    return (thicknesses_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we can compute the value of the dilation parameter.

    We could chose the maximum thickness, but if the thickness of the drawing is not uniform, we will a lot of details in regions where the thickness is very small.

    After some experimentation, we decided to take the median of all thickness values.
    """)
    return


@app.cell
def _(
    dilation,
    disk,
    img_binary,
    np,
    plt,
    remove_small_holes,
    remove_small_objects,
    thicknesses_1,
):
    c = int(np.median(thicknesses_1[thicknesses_1 > 0]))
    img_binary_1 = remove_small_objects(img_binary, 4 * c * c)
    # remove holes, objects and dilate
    img_binary_1 = remove_small_holes(img_binary_1, 4 * c * c)
    img_binary_1 = dilation(img_binary_1, disk(c))
    plt.imshow(img_binary_1)
    plt.title('final drawing after dilation')
    plt.show()
    return (img_binary_1,)


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
def _(img_binary_1, plt, skeletonize):
    skeleton = skeletonize(img_binary_1, method='zhang')
    plt.imshow(1 - skeleton, cmap='gray')
    plt.title('skeleton')
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
    for h in hyper.all_hyperedges():
        fit_hyperedge(h)
    visualize_hyper(hyper, offset=0)
    plt.imshow(img, alpha=0.5, cmap="gray")
    ax = plt.gca()
    ax.axis("off")
    plt.title("Initial hyper-graph")
    ax.text(0.5, -0.1, f"number of bezier: {len(hyper)}", ha="center", va="top", transform=ax.transAxes)
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
    topo_graph_2 = refine(topo_graph_1)
    hyper_1 = HyperGraph(topo_graph_2)
    for h_1 in hyper_1.all_hyperedges():
        fit_hyperedge(h_1)
    visualize_hyper(hyper_1, offset=0)
    plt.imshow(img, alpha=0.5, cmap='gray')
    ax_1 = plt.gca()
    ax_1.axis('off')
    plt.title('After refine')
    plt.text(0.5, -0.1, f'number of bezier: {len(hyper_1)}', fontsize=10, ha='center', va='top', transform=ax_1.transAxes)
    plt.show()
    return hyper_1, topo_graph_2


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
def _(hyper_1, optimisation, topo_graph_2):
    lam = 0.9
    temp = 0.5
    t_min = 0.05
    mu = 0.3
    t_decrease = 0.99 ** (1 / len(topo_graph_2.nodes))
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
    ax_2 = plt.gca()
    ax_2.axis('off')
    plt.title('Bezier fit, after global optimization')
    ax_2.text(0.5, -0.1, f'number of bezier: {len(hyper_1)}', ha='center', transform=ax_2.transAxes)
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
    ax_3 = plt.gca()
    ax_3.axis('off')
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
    out_1 = generate_svg_str(img.shape, hyper_1, stroke_width=10)
    SVG(out_1)
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
