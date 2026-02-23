from collections import defaultdict
from enum import IntEnum
from typing import Callable, Generator, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .bezier import fit_bezier, fitting_error, interpolate_bezier
from .utils import Curve, Pixel, PixelChain, PixelEdge

SuperEdgeId = int  # NewType("SuperEdgeId", int)


class SuperEdge:
    parts: List[PixelChain]
    parts_w: List[NDArray]  # weights of each pixel in each part

    def __init__(self, chains: List[PixelChain], parts_w: List[NDArray], degree: int):
        self.parts = chains
        self.parts_w = parts_w
        self.degree = degree

        # Concatenate parts into a single chain and weight array, deduplicating
        # the shared boundary pixel between consecutive parts.
        self.chain: NDArray = np.concatenate(
            [self.parts[0]] + [c[1:] for c in self.parts[1:]], axis=0
        )
        thicknesses: NDArray = np.concatenate(
            [self.parts_w[0]] + [w[1:] for w in self.parts_w[1:]], axis=0
        )
        self.weights: NDArray = 1 - 0.5 * thicknesses

        # Precompute pixel-level structural attributes.
        self.start_pixel: Pixel = (int(chains[0][0][0]), int(chains[0][0][1]))
        self.end_pixel: Pixel = (int(chains[-1][-1][0]), int(chains[-1][-1][1]))
        self.interior_pixels: list[Pixel] = [
            (int(part[-1][0]), int(part[-1][1])) for part in chains[:-1]
        ]

        # Fit the Bezier curve and store control points and fitting error.
        self._refit()

    def _refit(self) -> None:
        """Fit the Bezier curve from the precomputed chain and weights."""
        time = np.linspace(0, 1, len(self.chain))
        self.control_points: NDArray = fit_bezier(
            self.chain, time, degree=self.degree, weights=self.weights
        )
        self.fitting_error: float = fitting_error(
            interpolate_bezier(self.control_points, time), self.chain
        )

    def endpoints(self) -> Generator[Pixel, None, None]:
        for p in self.parts:
            yield (int(p[0][0]), int(p[0][1]))
        yield self.end_pixel

    def merge(self, other: "SuperEdge") -> "SuperEdge":
        return SuperEdge(
            self.parts + other.parts,
            self.parts_w + other.parts_w,
            max(self.degree, other.degree),
        )

    def reverse(self) -> "SuperEdge":
        return SuperEdge(
            [c[::-1] for c in reversed(self.parts)],
            [c[::-1] for c in reversed(self.parts_w)],
            self.degree,
        )

    def split(self, pixel: Pixel) -> Tuple["SuperEdge", "SuperEdge"]:
        # split the chain into two parts, the prefix and the rest
        i_part = [i for i, c in enumerate(self.parts) if tuple(c[-1]) == pixel][0]
        prefix = self.parts[: i_part + 1]
        rest = self.parts[i_part + 1 :]
        prefix_weight = self.parts_w[: i_part + 1]
        rest_weight = self.parts_w[i_part + 1 :]
        return SuperEdge(prefix, prefix_weight, self.degree), SuperEdge(
            rest, rest_weight, self.degree
        )

    def overlap_on(self, other: "SuperEdge"):
        """
        Return this superedge with a part from another added.
        The added part is the one which starts at the end of this superedge.
        If there is no such part, an assertion error is raised.
        """
        candidates = [
            i for i, c in enumerate(other.parts) if (tuple(c[0]) == self.end_pixel)
        ]
        assert len(candidates) >= 1, (
            "The other superedge does not have a part that starts at the end of this superedge"
        )
        part_other = other.parts[candidates[0]]
        weight_other = other.parts_w[candidates[0]]
        return SuperEdge(
            self.parts + [part_other], self.parts_w + [weight_other], self.degree
        )

    def dissociate_on(self, other: "SuperEdge"):
        """
        Inverse of overlap_on: return this superedge with the last part removed.
        The removed part must be a part of the other superedge
        """
        removed_part = self.parts[-1]
        assert any(np.array_equal(removed_part, c) for c in other.parts), (
            "The last part of this superedge is not a part of the other superedge and cannot be removed"
        )
        return SuperEdge(self.parts[:-1], self.parts_w[:-1], self.degree)

    def change_degree(self, new_degree: int) -> "SuperEdge":
        """Return a new superedge with the same chain but a different degree."""
        return SuperEdge(self.parts, self.parts_w, new_degree)

    def score(self, lam=0.5, mu=0.3) -> float:
        u_simplicity = 1 + mu * self.degree
        return (1 - lam) * self.fitting_error + lam * u_simplicity

    def __repr__(self):
        path = " -> ".join(f"({p[0]},{p[1]})" for p in self.endpoints())
        return f"SuperEdge[degree:{self.degree}, error:{self.fitting_error:.2f}, path:{path}]"


class Perturbation(IntEnum):
    INCREASE_DEGREE = 0
    DECREASE_DEGREE = 1
    MERGE = 2
    SPLIT = 3
    OVERLAP = 4
    DISSOCIATE = 5
    REVERSE = 6


class SuperGraph:
    superedges: List[SuperEdge]
    # when the node is first
    incidence: defaultdict[Pixel, List[SuperEdgeId]]

    def __init__(self, chains: List[PixelChain], thickness_map: NDArray[np.float32]):
        self.max_thickness = thickness_map.max()
        self.superedges = [
            SuperEdge([c], [thickness_map[c[:, 0], c[:, 1]] / self.max_thickness], 3)
            for c in chains
        ]
        self.incidence = defaultdict(list)
        for i, c in enumerate(self.superedges):
            self.incidence[c.start_pixel].append(i)
            self.incidence[c.end_pixel].append(i)

    def check_valid_incidence(self, msg):
        for e, occurences in self.incidence.items():
            for s_id2 in occurences:
                assert s_id2 < len(self.superedges), (
                    f"{msg}: node {e} is connected to the new id {s_id2} () but len is {len(self.superedges)}"
                )

        for i, s in enumerate(self.superedges):
            for e in s.endpoints():
                assert i in self.incidence[e], (
                    f"{msg}: incidence is not up to date: endpoint {e} should be registered in {s} (id={i}) but does not"
                )
                assert e in list(s.endpoints()), (
                    f"incidence has a lier: endpoint {e} is registered as en enpoint of {s} (id={i}) but is not"
                )
        for e, chains in self.incidence.items():
            for s in chains:
                assert e in list(self.superedges[s].endpoints())

    def add(self, superedge: SuperEdge) -> int:
        new_id = len(self.superedges)
        self.superedges.append(superedge)
        for e in superedge.endpoints():
            self.incidence[e].append(new_id)
        return new_id

    def remove(self, s_id: int):
        # self.check_valid_incidence(f"BEFORE REMOVE {s_id}")
        superedge = self.superedges[s_id]
        for e in superedge.endpoints():
            occurences = self.incidence[e].count(s_id)
            for _ in range(occurences):
                self.incidence[e].remove(s_id)
        if s_id == len(self.superedges) - 1:
            self.superedges.pop(-1)
        else:
            id_removed = len(self.superedges) - 1
            self.superedges[s_id] = self.superedges.pop(-1)
            # move the dangling references of the moved superedge to its new id
            for e in self.superedges[s_id].endpoints():
                for i in range(len(self.incidence[e])):
                    if self.incidence[e][i] == id_removed:
                        self.incidence[e][i] = s_id

        # self.check_valid_incidence(f"AFTER REMOVE {s_id}")

    def remove_many(self, ids: List[int]):
        for s_id in sorted(ids, reverse=True):
            self.remove(s_id)

    def add_incidence(self, s_id: int, pixel: Pixel):
        self.incidence[pixel].append(s_id)

    def remove_incidence(self, s_id: int, pixel: Pixel):
        occurences = self.incidence[pixel].count(s_id)
        assert occurences > 0, "The superedge is not incident to this pixel"
        self.incidence[pixel].remove(s_id)

    def increase_degree(self, s_id: int):
        """
        Increase the degree of the superedge with id s_id by 1
        """
        assert self.superedges[s_id].degree < 3, (
            "Degree must be less than 3 to increase"
        )
        self.superedges[s_id] = self.superedges[s_id].change_degree(
            self.superedges[s_id].degree + 1
        )

    def decrease_degree(self, s_id: int):
        """
        Decrease the degree of the superedge with id s_id by 1, if possible (i.e. if the degree is greater than 1)
        """
        assert self.superedges[s_id].degree > 1, (
            "Degree must be greater than 1 to decrease"
        )
        self.superedges[s_id] = self.superedges[s_id].change_degree(
            self.superedges[s_id].degree - 1
        )

    def reverse(self, s_id: int):
        """
        Reverse the order of the pixels in this superedge
        """
        self.superedges[s_id] = self.superedges[s_id].reverse()

    def merge(self, s_id1: int, s_id2: int) -> int:
        """
        Merge the two superedges with id s_id1 and s_id2, by concatenating their list of edges.
        The new superedge will have the same id as s_id1, and s_id2 will be removed from the supergraph.
        """
        join_pixel = self.superedges[s_id2].start_pixel

        assert self.superedges[s_id1].end_pixel == join_pixel, (
            "The two superedges are not connected and cannot be merged"
        )

        a = self.superedges[s_id1]
        b = self.superedges[s_id2]

        self.remove_many([s_id1, s_id2])
        return self.add(a.merge(b))

    def split(self, s_id: int, pixel: Pixel) -> Tuple[int, int]:
        """
        Split the superedge with id s_id into two superedges, by splitting the list of edges at the given split point.
        The new superedge will have a new id, and the original superedge will keep its id but with a shorter list of edges.
        """
        e1, e2 = self.superedges[s_id].split(pixel)
        self.remove(s_id)
        id1 = self.add(e1)
        id2 = self.add(e2)
        return id1, id2

    def overlap(self, s_id1: int, s_id2: int):
        """
        Extend superedge s_id1 by adding a part from s_id2 that starts at the end of s_id1.
        The new superedge will have the same id as s_id1.
        s_id2 remains in the supergraph but loses the overlapped part.
        """
        # self.check_valid_incidence("BEFORE OVERLAP")
        # Get the end pixel of s_id1
        a = self.superedges[s_id1]
        b = self.superedges[s_id2]

        # Use overlap_on to extend s_id1 with a part from s_id2
        self.superedges[s_id1] = a.overlap_on(b)

        # Get the new end pixel after overlap
        new_end_pixel = self.superedges[s_id1].end_pixel

        # Update incidence: old end pixel becomes internal, new end pixel is the endpoint
        # Note: old_end_pixel is no longer an endpoint, but we keep it in incidence
        # as an internal point could be tracked separately. For now, we don't remove it
        # from incidence since incidence tracks all endpoints.
        # Actually, after overlap, old_end_pixel becomes an internal point, not an endpoint
        self.add_incidence(s_id1, new_end_pixel)
        # self.check_valid_incidence("AFTER OVERLAP")

    def dissociate(self, s_id1: int, s_id2: int):
        """
        Remove the last part from superedge s_id1 that was added from s_id2.
        This is the inverse operation of overlap.
        The removed part is restored to s_id2.
        """
        # Get the end pixel of s_id1 before dissociation
        old_end_pixel = self.superedges[s_id1].end_pixel

        # Use dissociate_on to remove the last part from s_id1
        self.superedges[s_id1] = self.superedges[s_id1].dissociate_on(
            self.superedges[s_id2]
        )

        # Update incidence: remove old end, add new end
        self.remove_incidence(s_id1, old_end_pixel)
        # self.check_valid_incidence("AFTER DISSOCIATE")

    def __len__(self):
        return len(self.superedges)

    def sample_connected_pair(self) -> Tuple[SuperEdgeId, Pixel, Optional[SuperEdgeId]]:
        """
        Sample a random pair of connected superedges (s_a, s_b) with their joining pixel, or a single superedge with one contained pixel.

        Returns:
            One of two possibilities:
            - `(s_a, pixel, s_b)` where `s_a` and `s_b` are two different superedges, `pixel` is an endpoint of `s_a` and the end pixel of `s_b`.
            - `(s_a, pixel, None)` where `s_a` is a superedge and `pixel` is an endpoint of `s_a`
        """
        id_a = np.random.choice(len(self.superedges))
        endpoints = list(self.superedges[id_a].endpoints())
        i_pixel = np.random.choice(len(endpoints))
        pixel = endpoints[i_pixel]
        candidates_b = [
            s_id
            for s_id in self.incidence[pixel]
            if self.superedges[s_id].end_pixel == pixel and s_id != id_a
        ]
        if len(candidates_b) == 0:
            return (id_a, pixel, None)
        id_b = np.random.choice(candidates_b)
        return id_a, pixel, id_b

    def is_valid_perturbation(
        self, perturbation: Perturbation, s_a: int, pixel: Pixel, s_b: Optional[int]
    ) -> bool:
        """
        Check if a perturbation is valid for the given superedges.
        Returns True if the perturbation can be applied, False otherwise.
        """
        if perturbation == Perturbation.INCREASE_DEGREE:
            return self.superedges[s_a].degree < 3
        elif perturbation == Perturbation.DECREASE_DEGREE:
            return self.superedges[s_a].degree > 1
        elif perturbation == Perturbation.MERGE:
            if s_b is None:
                return False
            end_b = self.superedges[s_b].end_pixel
            start_a = self.superedges[s_a].start_pixel
            return end_b == start_a
        elif perturbation == Perturbation.SPLIT:
            return (
                len(self.superedges[s_a].parts) >= 2
                and self.superedges[s_a].start_pixel != pixel
                and self.superedges[s_a].end_pixel != pixel
            )
        elif perturbation == Perturbation.OVERLAP:
            if s_b is None:
                return False
            end_pixel = self.superedges[s_b].end_pixel
            return any(tuple(c[0]) == end_pixel for c in self.superedges[s_a].parts)
        elif perturbation == Perturbation.DISSOCIATE:
            if s_b is None:
                return False
            if len(self.superedges[s_b].parts) < 2 or s_a == s_b:
                return False
            part_to_remove = self.superedges[s_b].parts[-1]
            return any(
                np.array_equal(c, part_to_remove) for c in self.superedges[s_a].parts
            )
        elif perturbation == Perturbation.REVERSE:
            return True
        else:
            raise ValueError("Invalid perturbation type")

    def compute_delta_score(
        self,
        perturbation: Perturbation,
        s_a: int,
        pixel: Pixel,
        s_b: Optional[int],
        lam=0.5,
        mu=0.3,
    ) -> float:
        """
        Compute the change in score that would result from applying the given perturbation to the superedges with id s_id1 and s_id2 (if applicable).
        This is used to decide whether to accept or reject a proposed perturbation during optimization.
        """
        if not self.is_valid_perturbation(perturbation, s_a, pixel, s_b):
            return float("inf")

        if perturbation == Perturbation.INCREASE_DEGREE:
            new_score = (
                self.superedges[s_a]
                .change_degree(self.superedges[s_a].degree + 1)
                .score(lam, mu)
            )
            return new_score - self.superedges[s_a].score(lam, mu)
        elif perturbation == Perturbation.DECREASE_DEGREE:
            new_score = (
                self.superedges[s_a]
                .change_degree(self.superedges[s_a].degree - 1)
                .score(lam, mu)
            )
            return new_score - self.superedges[s_a].score(lam, mu)
        elif perturbation == Perturbation.MERGE:
            assert s_b is not None
            new_score = self.superedges[s_b].merge(self.superedges[s_a]).score(lam, mu)
            return (
                new_score
                - self.superedges[s_b].score(lam, mu)
                - self.superedges[s_a].score(lam, mu)
            )
        elif perturbation == Perturbation.SPLIT:
            prefix, updated = self.superedges[s_a].split(pixel)
            new_score = prefix.score(lam, mu) + updated.score(lam, mu)
            return new_score - self.superedges[s_a].score(lam, mu)
        elif perturbation == Perturbation.OVERLAP:
            assert s_b is not None
            new_score = (
                self.superedges[s_b].overlap_on(self.superedges[s_a]).score(lam, mu)
            )
            return new_score - self.superedges[s_b].score(lam, mu)
        elif perturbation == Perturbation.DISSOCIATE:
            assert s_b is not None
            new_score = (
                self.superedges[s_b].dissociate_on(self.superedges[s_a]).score(lam, mu)
            )
            return new_score - self.superedges[s_b].score(lam, mu)
        elif perturbation == Perturbation.REVERSE:
            return 0
        else:
            raise ValueError("Invalid perturbation type")

    def perturbate(
        self, perturbation: Perturbation, s_a: int, pixel: Pixel, s_b: Optional[int]
    ) -> List[int]:
        """
        Apply the given perturbation to the superedges with id s_id1 and s_id2 (if applicable).
        This is used to update the supergraph after accepting a proposed perturbation during optimization.
        Returns the list of ids for the superedges that were newly created or modified by the operation.
        """
        if perturbation == Perturbation.INCREASE_DEGREE:
            self.increase_degree(s_a)
            return [s_a]
        elif perturbation == Perturbation.DECREASE_DEGREE:
            self.decrease_degree(s_a)
            return [s_a]
        elif perturbation == Perturbation.MERGE:
            assert s_b is not None
            new_id = self.merge(s_b, s_a)
            return [new_id]
        elif perturbation == Perturbation.SPLIT:
            id1, id2 = self.split(s_a, pixel)
            return [id1, id2]
        elif perturbation == Perturbation.OVERLAP:
            assert s_b is not None
            self.overlap(s_b, s_a)
            return [s_a]
        elif perturbation == Perturbation.DISSOCIATE:
            assert s_b is not None
            self.dissociate(s_b, s_a)
            return [s_a]
        elif perturbation == Perturbation.REVERSE:
            self.reverse(s_a)
            return [s_a]
        else:
            raise ValueError("Invalid perturbation type")


CHOICE_DISTRIBUTION = [
    # change degree
    1 / 8,
    1 / 8,
    # merge or split
    2 / 8,  # we strongly bias merging, since it's the most usefull
    1 / 8,
    # overlap or dissociate
    1 / 8,
    1 / 8,
    # reverse edge
    1 / 8,
]


class SketchOptimizer:
    def __init__(
        self,
        lam=0.5,
        mu=0.3,
        t_decrease=0.9995,
        t_min=0.001,
        status_function: Callable = lambda x, title: x,
    ):
        self.lam = lam
        self.mu = mu
        self.error_ = []
        self.t_min = t_min
        self.t_decrease = t_decrease
        self.history_ = []
        self.status_function = status_function

    def fit_transform(
        self, X: List[PixelChain], y: NDArray[np.float32], **fit_params
    ) -> List[Curve]:
        """
        Fit and transform pixel chains into Bezier curves.

        After calling this method the following attributes are set:

        - ``endpoint_mapping_``: maps each Curve to its ``(start_pixel, end_pixel)``.
        - ``interior_mapping_``: maps each Curve to its list of interior junction
          pixels (pixels between consecutive parts of the superedge that lie in
          the interior of the fitted Bezier).

        Returns:
            curves: List of fitted Curve objects.
        """
        chains = X
        self.supergraph_ = SuperGraph(chains, y)

        energy = sum(x.score(self.lam, self.mu) for x in self.supergraph_.superedges)
        self.error_ = [energy]
        temp = 1

        n_it = int(np.log(self.t_min / temp) / np.log(self.t_decrease))

        perturbations = list(Perturbation)  # allocated once, reused each iteration

        for _ in self.status_function(range(n_it), "Optimizing curve network"):
            perturbation = np.random.choice(perturbations, p=CHOICE_DISTRIBUTION)
            a, pix, b = self.supergraph_.sample_connected_pair()
            delta = self.supergraph_.compute_delta_score(
                perturbation, a, pix, b, lam=self.lam, mu=self.mu
            )
            p = np.random.random()
            if p < np.exp(-delta / temp):
                self.history_.append((perturbation, a, pix, b))
                self.supergraph_.perturbate(perturbation, a, pix, b)
                self.error_.append(energy)
                energy += delta
            else:
                temp = temp * self.t_decrease

        max_w = self.supergraph_.max_thickness
        curves = [
            Curve(
                control_points=s.control_points,
                stroke_width=np.hstack(s.parts_w).mean() * max_w,
            )
            for s in self.supergraph_.superedges
        ]
        self.endpoint_mapping_ = {
            curves[i]: (
                self.supergraph_.superedges[i].start_pixel,
                self.supergraph_.superedges[i].end_pixel,
            )
            for i in range(len(curves))
        }
        self.interior_mapping_ = {
            curves[i]: self.supergraph_.superedges[i].interior_pixels
            for i in range(len(curves))
        }
        return curves


def align_boundaries(
    chains: List[Curve],
    endpoint_mapping: dict[Curve, PixelEdge],
    interior_mapping: dict[Curve, list[Pixel]],
) -> List[Curve]:
    """
    Align curve endpoints at shared pixel junctions.

    Two passes:

    - **Phase 1 – C0 at endpoints**: for each pixel that is the endpoint of
      two or more curves, snap all those endpoint control points to their mean
      position.
    - **Phase 2 – snap to interior**: for each pixel that is the endpoint of
      exactly one curve *and* lies in the interior of exactly one other curve,
      find the parameter ``t`` on the interior curve that is closest to the
      endpoint (by bisection) and move the endpoint control point there.

    Control points follow the ``[row, col]`` (i.e. ``[y, x]``) convention.

    Args:
        chains: Curve objects whose control points are mutated in place.
        endpoint_mapping: Maps each Curve to its ``(start_pixel, end_pixel)``.
        interior_mapping: Maps each Curve to the list of interior junction
            pixels (between its consecutive parts).
    """
    # ------------------------------------------------------------------
    # Build lookup tables
    # ------------------------------------------------------------------

    # endpoint_curves[pixel] = list of (curve, "start"|"end")
    endpoint_curves: dict[Pixel, list[tuple[Curve, str]]] = defaultdict(list)
    for curve in chains:
        if curve not in endpoint_mapping:
            continue
        start_pix, end_pix = endpoint_mapping[curve]
        endpoint_curves[start_pix].append((curve, "start"))
        endpoint_curves[end_pix].append((curve, "end"))

    # interior_curves[pixel] = list of curves that pass through pixel internally
    interior_curves: dict[Pixel, list[Curve]] = defaultdict(list)
    for curve, pixels in interior_mapping.items():
        for pix in pixels:
            interior_curves[pix].append(curve)

    # ------------------------------------------------------------------
    # Phase 1: C0 – average endpoints when multiple curves terminate here
    # ------------------------------------------------------------------
    for pixel, incidents in endpoint_curves.items():
        if len(incidents) < 2:
            continue

        anchors = [
            np.array(curve.control_points[0 if pos == "start" else -1], dtype=float)
            for curve, pos in incidents
        ]
        avg_anchor = np.mean(np.stack(anchors), axis=0)

        for curve, pos in incidents:
            cp = curve.control_points
            if pos == "start":
                cp[0] = avg_anchor
            else:
                cp[-1] = avg_anchor

    # ------------------------------------------------------------------
    # Phase 2: snap lone endpoint to the closest point on a passing curve
    # ------------------------------------------------------------------
    # A pixel qualifies when exactly one curve ends there and exactly one
    # other curve passes through it as an interior point.
    for pixel in set(endpoint_curves) & set(interior_curves):
        ep_incidents = endpoint_curves[pixel]
        int_incidents = interior_curves[pixel]

        if len(ep_incidents) != 1 or len(int_incidents) != 1:
            continue

        ep_curve, ep_pos = ep_incidents[0]
        through_curve = int_incidents[0]

        # Bisection search for the t on through_curve closest to the endpoint.
        # Each iteration evaluates 5 points and zooms into the best interval.
        t_lo, t_hi = 0.0, 1.0
        for _ in range(2):
            ts = np.linspace(t_lo, t_hi, 5)
            points = interpolate_bezier(through_curve.control_points, ts)
            # endpoint control point (already updated by Phase 1 if applicable)
            ep_pt = np.array(
                ep_curve.control_points[0 if ep_pos == "start" else -1],
                dtype=float,
            )
            dists = np.linalg.norm(points - ep_pt, axis=1)
            best = int(np.argmin(dists))
            t_lo = ts[max(0, best - 1)]
            t_hi = ts[min(len(ts) - 1, best + 1)]

        t_best = (t_lo + t_hi) / 2.0
        closest_pt = interpolate_bezier(
            through_curve.control_points, np.array([t_best])
        )[0]

        cp = ep_curve.control_points
        if ep_pos == "start":
            cp[0] = closest_pt
        else:
            cp[-1] = closest_pt

    return chains
