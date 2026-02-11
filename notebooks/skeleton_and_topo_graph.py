import marimo

__generated_with = "0.19.9"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import io, color
    from skimage.filters import threshold_otsu
    from skimage.morphology import skeletonize

    return color, io, np, plt, skeletonize, threshold_otsu


@app.cell
def _(color, io):
    # Load the image
    image = io.imread('path_to_your_image.png')
    # Convert to grayscale if it's a color image
    if image.ndim == 3:
        image = color.rgb2gray(image)
    return (image,)


@app.cell
def _(image, skeleton_to_graph, skeletonize, threshold_otsu):
    # Binarize the image using Otsu's threshold
    thresh = threshold_otsu(image)
    binary = image > thresh

    # Perform skeletonization
    skeleton = skeletonize(binary)

    # Convert the skeleton to a graph
    graph = skeleton_to_graph(skeleton)
    return graph, skeleton


@app.cell
def _(graph, plt, skeleton):
    # Visualize the graph
    plt.figure(figsize=(8, 8))
    for edge in graph.edges():
        y0, x0 = edge[0]
        y1, x1 = edge[1]
        plt.plot([x0, x1], [y0, y1], 'k-')
    plt.imshow(skeleton, cmap='gray')
    plt.title('Skeleton Graph')
    plt.axis('off')
    plt.show()
    return


@app.cell
def _(np):
    from skeleton_tracing import Skeleton
    binary_img = np.array([[0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0], [1, 1, 0, 0, 1, 1], [0, 1, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0]], dtype=bool)
    skeleton_1 = Skeleton()
    # Load your binary image as a numpy array, dtype=bool or 0/1
    # For example, let's say `binary_img` is a 2D numpy array with True for foreground pixels
    polylines = skeleton_1.trace(binary_img)
    for (i, polyline) in enumerate(polylines):
        print(f'Polyline {i}:')
        for point in polyline:
    # Create the skeleton tracing object
    # Compute the skeleton polylines
    # polylines is a list of polylines, each polyline is a list of (x, y) points
            print(point)
    return


if __name__ == "__main__":
    app.run()
