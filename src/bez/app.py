import sys

import matplotlib.pyplot as plt
import numpy as np
from skimage import io, morphology
from skimage.morphology import skeletonize

from bez.cnn import load_trained_model
from bez.data import load_normalized
from bez.global_optim import optimisation
from bez.hypergraph import HyperGraph
from bez.preprocessing import preprocess
from bez.refinement import refine
from bez.topo_graph import extract_simple_topology_from_skeleton
from bez.viz import visualize_hyper, visualize_topo


def generate_svg_str(image_shape, hyper: HyperGraph, stroke_width=5):
    svg_paths = []
    for h in hyper.all_hyperedges():
        control_points = h.control_points
        if h.degree == 1:
            (y1, y2), (x1, x2) = control_points
            d = f"M {x1},{y1} L {x2},{y2}"

        elif h.degree == 2:
            (y1, y2, y3), (x1, x2, x3) = control_points
            d = f"M {x1},{y1} Q {x2},{y2} {x3},{y3}"

        elif h.degree == 3:
            (y1, y2, y3, y4), (x1, x2, x3, x4) = control_points
            d = f"M {x1},{y1} C {x2},{y2} {x3},{y3} {x4},{y4}"

        else:
            raise ValueError("invalid degree")

        svg_paths.append(
            f'  <path fill="none" stroke="black" stroke-width="{stroke_width}" d="{d}"/>'
        )

    paths_str = "\n".join(svg_paths)
    svg_content = f'''<svg width="{image_shape[1]}" height="{image_shape[0]}" xmlns="http://www.w3.org/2000/svg">
{paths_str}
</svg>'''

    return svg_content


def show_example(image_path):
    """
    Complete network analysis pipeline with visualization.

    Args:
        image_path: Path to the input image
    """

    # Process the image
    img = load_normalized(image_path)
    model = load_trained_model()
    img_binary = model.run(["output"], {"input": img[np.newaxis, :, :]})[0][0] > 0.5
    img_binary = morphology.dilation(img_binary)
    # todo: dilation
    skeleton = skeletonize(img_binary, method="zhang")

    # Extract and refine topology
    topo_graph = extract_simple_topology_from_skeleton(skeleton)
    topo_graph_refine = refine(topo_graph)

    # Create hypergraph and optimize
    hyper = HyperGraph(topo_graph_refine)
    errors = optimisation(hyper)
    # FIXME: finition does not work
    # hyper.finition()

    # Create subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(img, cmap="binary")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Plot 2: Binary image
    axes[1].imshow(img_binary, cmap="binary")
    axes[1].set_title("Binary Image")
    axes[1].axis("off")

    # Plot 4: Topology visualization
    plt.sca(axes[2])  # Set current axes
    visualize_topo(topo_graph_refine)
    axes[2].set_title("Refined Topology")
    axes[2].axis("equal")  # Ensure equal scaling
    axes[2].axis("off")

    # Plot 3: HyperGraph visualization
    plt.sca(axes[3])  # Set current axes
    visualize_hyper(hyper)
    axes[3].set_title("Result")
    axes[3].axis("equal")  # Ensure equal scaling
    axes[3].axis("off")

    # Add legend (assuming your visualization functions create legend-worthy elements)
    # You may need to modify this based on what your visualize_* functions actually plot
    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        axes[2].legend(handles, labels, loc="upper right", fontsize="small")

    handles, labels = axes[3].get_legend_handles_labels()
    if handles:
        axes[3].legend(handles, labels, loc="upper right", fontsize="small")

    plt.tight_layout()
    plt.show()

    return img_binary, hyper


def cli():
    if len(sys.argv) == 1:
        print("Please provide an image file to convert")
        exit(1)

    path = sys.argv[1]
    img_raw = io.imread(path)
    img_binary, thickness = preprocess(img_raw)
    skeleton = skeletonize(img_binary, method="zhang")

    # Extract and refine topology
    topo_graph = extract_simple_topology_from_skeleton(skeleton)
    topo_graph_refine = refine(topo_graph)

    # Create hypergraph and optimize
    hyper = HyperGraph(topo_graph_refine)
    errors = optimisation(hyper)
    hyper.finition()
    result = generate_svg_str(img_binary.shape, hyper)
    with open(path + ".svg", "w") as f:
        f.write(result)
    print(f"svg generated at {path}.svg")


if __name__ == "__main__":
    cli()
