import numpy as np
import svg
from bez.hypergraph import HyperGraph
from bez.bezier import fit_bezier


def generate_svg_str(image_shape, hyper: HyperGraph, stroke_width=5):
    svg_elements = []
    for h in hyper.all_hyperedges():
        p = np.array(h.pixels).T
        instants = np.linspace(0, 1, p.shape[1])
        control_points = fit_bezier(p, instants, degree=h.degree)
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

        svg_elements.append(svg.Path(fill="none", stroke="black", stroke_width=stroke_width, d=d))
    result = svg.SVG(width=image_shape[1], height=image_shape[0], elements=svg_elements)
    return str(result)
