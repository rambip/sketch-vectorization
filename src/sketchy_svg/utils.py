import numpy as np
from numpy.typing import NDArray

DEFAULT_SIZE = 256

Pixel = tuple[int, int]

"""
A pair of pixels used to locate the start and end of a pixel chain or curve.
Note that the 2 pixels can be identical, in which case the edge represents a loop (a curve that starts and ends at the same pixel).
"""
PixelEdge = tuple[Pixel, Pixel]

Point = tuple[float, float]

"""
A chain of pixels as a numpy array of shape (N, 2) where N is the number of pixels.
Each row is [row, col] or [y, x] pixel coordinates.
"""
PixelChain = NDArray[np.int_]


class Curve:
    def __init__(self, control_points: NDArray[np.float32], stroke_width: float):
        self.control_points = control_points
        n_control_points = len(control_points)
        self.stroke_width = stroke_width
        if n_control_points < 2 or n_control_points > 4:
            raise ValueError(f"invalid number of control points: {n_control_points}")

    @property
    def degree(self) -> int:
        return len(self.control_points) - 1

    def to_svg(self, color: str) -> str:
        """
        Return the svg path representation depending on the degree
        Arg:
            color: the stroke color
        """
        if self.degree == 1:
            symbol = "L"
        elif self.degree == 2:
            symbol = "Q"
        elif self.degree == 3:
            symbol = "C"
        else:
            raise ValueError("invalid degree")
        # first control point
        y1, x1 = self.control_points[0]
        # rest of control points
        rest = " ".join(f"{x},{y}" for y, x in self.control_points[1:])
        return f'  <path fill="none" stroke="{color}" stroke-width="{self.stroke_width}" d="M {x1},{y1} {symbol} {rest}"/>'
