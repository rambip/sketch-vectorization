from typing import List

from .utils import Curve


def export_svg(curves: List[Curve], height: int, width: int) -> str:
    """
    Export the list of curves to an SVG string.
    Args:
        curves: List of Curve objects to be exported.
        image_shape: Tuple representing the height and width of the original image (height, width).
    Returns:
        A string containing the SVG representation of the curves.
    """
    svg_paths = "\n".join(curve.to_svg(color="black") for curve in curves)
    svg_content = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"> {svg_paths} </svg>'
    return svg_content
