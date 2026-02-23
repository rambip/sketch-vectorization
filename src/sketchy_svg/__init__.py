"""This is a WIP module that skethes function signatures for the new implementation of sketchy."""

__all__ = [
    "prepare",
    "optim",
    "topology",
    "export",
    "demo",
    "patch_onnx",
    "load_normalized",
    "sketch2svg",
]

import numpy as np
from numpy.typing import NDArray
from skimage import morphology

from . import demo, export, optim, prepare, topology
from .onnxruntime_compat import patch_onnx
from .prepare import load_normalized


async def sketch2svg(
    img: NDArray[np.float32], status_function=lambda x, title: x
) -> str:
    classifier = prepare.BinarySketchPredictor()
    prediction = await classifier.predict(img)
    skeleton = morphology.skeletonize(prediction)
    thickness_map = prepare.compute_thickness_map(prediction)
    chains = topology.extract_chains(skeleton)
    chains_clean = topology.remove_parasite_chains(chains)
    chains_refined = topology.refine_all_chains(chains_clean)

    optimizer = optim.SketchOptimizer(status_function=status_function)
    curves = optimizer.fit_transform(chains_refined, thickness_map)
    final_curves = optim.align_boundaries(
        curves, optimizer.endpoint_mapping_, optimizer.interior_mapping_
    )
    return export.export_svg(final_curves, img.shape[0], img.shape[1])
