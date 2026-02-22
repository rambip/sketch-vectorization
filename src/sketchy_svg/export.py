from typing import List

from .utils import Curve


def export_svg(curves: List[Curve], image_shape: tuple[int, int]) -> str:
    """
    Export the list of curves to an SVG string.
    Args:
        curves: List of Curve objects to be exported.
        image_shape: Tuple representing the height and width of the original image (height, width).
    Returns:
        A string containing the SVG representation of the curves.
    """
    svg_paths = "\n".join(curve.to_svg(color="black") for curve in curves)
    svg_content = f'<svg width="{image_shape[1]}" height="{image_shape[0]}" xmlns="http://www.w3.org/2000/svg"> {svg_paths} </svg>'
    return svg_content
