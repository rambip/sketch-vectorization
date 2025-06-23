import numpy as np
import svg
from bez.hypergraph import HyperGraph
from bez.bezier import fit_bezier
from bez.viz import visualize_topo, visualize_hyper
from bez.preprocessing import preprocess
from bez.global_optim import optimisation, fit_hyperedge
from bez.topo_graph import extract_simple_topology_from_skeleton
from bez.refinement import refine
from skimage import io
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import sys


def generate_svg_str(image_shape, hyper: HyperGraph, stroke_width=5):
    svg_elements = []
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

        svg_elements.append(
            svg.Path(fill="none", stroke="black", stroke_width=stroke_width, d=d)
        )
    result = svg.SVG(width=image_shape[1], height=image_shape[0], elements=svg_elements)
    return str(result)


def show_example(image_path):
    """
    Complete network analysis pipeline with visualization.

    Args:
        image_path: Path to the input image
    """

    # Process the image
    img_raw = io.imread(image_path)
    img_binary, thickness = preprocess(img_raw)
    skeleton = skeletonize(img_binary, method="zhang")

    # Extract and refine topology
    topo_graph = extract_simple_topology_from_skeleton(skeleton)
    topo_graph_refine = refine(topo_graph)

    # Create hypergraph and optimize
    hyper = HyperGraph(topo_graph_refine)
    errors = optimisation(hyper)
    hyper.finition()

    # Create subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Plot 1: Original image
    if len(img_raw.shape) == 2:
        axes[0].imshow(img_raw, cmap="gray")
    elif img_raw.shape[-1] == 2:
        axes[0].imshow(img_raw[:, :, 0], cmap="gray")
    else:
        axes[0].imshow(img_raw)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Plot 2: Binary image
    axes[1].imshow(~img_binary, cmap="gray")
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


if __name__ == "__main__":
    if len(sys.argv[0]) == 1:
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
