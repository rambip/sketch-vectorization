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
    chain: PixelChain
    control_points: NDArray
    fitting_error: float

    def __init__(self, chains: List[PixelChain], degree: int):
        self.parts = chains
        self.degree = degree
        # concatenate all chains without the endpoints
        self.chain = np.concatenate([chains[0]] + [c[1:] for c in chains[1:]], axis=0)
        self._fit()

    def endpoints(self) -> Generator[Pixel, None, None]:
        for p in self.parts:
            yield (int(p[0][0]), int(p[0][1]))
        yield self.end_pixel

    def _fit(self):
        time = np.linspace(0, 1, len(self.chain))
        self.control_points = fit_bezier(self.chain, time, degree=self.degree)
        self.fitting_error = fitting_error(
            interpolate_bezier(self.control_points, time), self.chain
        )

    def merge(self, other: "SuperEdge") -> "SuperEdge":
        return SuperEdge(self.parts + other.parts, max(self.degree, other.degree))

    def reverse(self) -> "SuperEdge":
        return SuperEdge([c[::-1] for c in reversed(self.parts)], self.degree)

    def split(self, pixel: Pixel) -> Tuple["SuperEdge", "SuperEdge"]:
        # split the chain into two parts, the prefix and the rest
        i_part = [i for i, c in enumerate(self.parts) if tuple(c[-1]) == pixel][0]
        prefix = self.parts[: i_part + 1]
        rest = self.parts[i_part + 1 :]
        return SuperEdge(prefix, self.degree), SuperEdge(rest, self.degree)

    def overlap_on(self, other: "SuperEdge"):
        """
        Return this superedge with a part from another added.
        The added part is the one which starts at the end of this superedge.
        If there is no such part, an assertion error is raised.
        """
        end_pixel = self.end_pixel
        singleton = [c for c in other.parts if (tuple(c[0]) == end_pixel)]
        assert len(singleton) >= 1, (
            "The other superedge does not have a part that starts at the end of this superedge"
        )
        part_other = singleton[0]
        return SuperEdge(self.parts + [part_other], self.degree)

    def dissociate_on(self, other: "SuperEdge"):
        """
        Inverse of overlap_on: return this superedge with the last part removed.
        The removed part must be a part of the other superedge
        """
        removed_part = self.parts[-1]
        assert any(np.array_equal(removed_part, c) for c in other.parts), (
            "The last part of this superedge is not a part of the other superedge and cannot be removed"
        )
        return SuperEdge(self.parts[:-1], self.degree)

    def change_degree(self, new_degree: int):
        """
        Return a new superedge with the same chain but with a new degree
        """
        return SuperEdge(self.parts, new_degree)

    @property
    def start_pixel(self) -> Pixel:
        """Return the first pixel of the chain as a tuple."""
        return (int(self.chain[0][0]), int(self.chain[0][1]))

    @property
    def end_pixel(self) -> Pixel:
        """Return the last pixel of the chain as a tuple."""
        return (int(self.chain[-1][0]), int(self.chain[-1][1]))

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

    def __init__(self, chains: List[PixelChain]):
        self.superedges = [SuperEdge([c], 3) for c in chains]
        self.incidence = defaultdict(list)
        for i, c in enumerate(self.superedges):
            self.incidence[c.start_pixel].append(i)
            self.incidence[c.end_pixel].append(i)

    def add(self, superedge: SuperEdge) -> int:
        new_id = len(self.superedges)
        self.superedges.append(superedge)
        for e in superedge.endpoints():
            self.incidence[e].append(new_id)
        return new_id

    def remove(self, s_id: int):
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

    def remove_many(self, ids: List[int]):
        for s_id in sorted(ids, reverse=True):
            self.remove(s_id)

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
        # Get values BEFORE merging
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
        # Get the end pixel of s_id1
        old_end_pixel = self.superedges[s_id1].end_pixel

        # Use overlap_on to extend s_id1 with a part from s_id2
        self.superedges[s_id1] = self.superedges[s_id1].overlap_on(
            self.superedges[s_id2]
        )

        # Get the new end pixel after overlap
        new_end_pixel = self.superedges[s_id1].end_pixel

        # Update incidence: old end pixel becomes internal, new end pixel is the endpoint
        # Note: old_end_pixel is no longer an endpoint, but we keep it in incidence
        # as an internal point could be tracked separately. For now, we don't remove it
        # from incidence since incidence tracks all endpoints.
        # Actually, after overlap, old_end_pixel becomes an internal point, not an endpoint
        self.remove_incidence(s_id1, old_end_pixel)
        self.incidence[new_end_pixel].append(s_id1)

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

        # Get the new end pixel after dissociation
        new_end_pixel = self.superedges[s_id1].end_pixel

        # Update incidence: remove old end, add new end
        self.remove_incidence(s_id1, old_end_pixel)
        self.incidence[new_end_pixel].append(s_id1)

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
    1 / 4,  # we strongly bias merging, since it's the most usefull
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
        t_decrease=0.9999,
        t_min=0.05,
        status_function: Callable = lambda x, title: x,
    ):
        self.lam = lam
        self.mu = mu
        self.mapping = None
        self.error_ = []
        self.t_min = t_min
        self.t_decrease = t_decrease
        self.history_ = []
        self.status_function = status_function

    def fit_transform(
        self, X: List[PixelChain], y: None = None, **fit_params
    ) -> tuple[List[Curve], dict[Curve, PixelEdge]]:
        chains = X
        self.supergraph_ = SuperGraph(chains)

        energy = sum(x.score(self.lam, self.mu) for x in self.supergraph_.superedges)
        self.error_ = [energy]
        temp = 0.2

        n_it = int(np.log(self.t_min / temp) / np.log(self.t_decrease))

        n = len(self.supergraph_)

        for _ in self.status_function(range(n_it), "Optimizing curve network"):
            assert sum(len(x.parts) for x in self.supergraph_.superedges) >= n, (
                "Some chains have disappeared !"
            )
            perturbation = np.random.choice(list(Perturbation), p=CHOICE_DISTRIBUTION)
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

        curves = [
            Curve(
                control_points=self.supergraph_.superedges[i].control_points, width=1.0
            )
            for i in range(len(self.supergraph_))
        ]
        mapping = {
            curves[i]: (
                self.supergraph_.superedges[i].start_pixel,
                self.supergraph_.superedges[i].end_pixel,
            )
            for i in range(len(curves))
        }
        return curves, mapping


def align_boundaries(
    chains: List[Curve], mapping: dict[Curve, PixelEdge]
) -> List[Curve]:
    """
    Join the curves in `chains` according to the mapping from curves to pixel edges.
    Args:
        chains: List of Curve objects to be joined.
        mapping: A dictionary mapping each Curve to a PixelEdge that indicates how the curves should be joined.
    """
    import math

    # Group incident endpoints per pixel
    by_pixel: dict[Pixel, list[tuple[Curve, str]]] = defaultdict(list)
    for curve in chains:
        if curve not in mapping:
            continue
        start_pix, end_pix = mapping[curve]
        by_pixel[start_pix].append((curve, "start"))
        by_pixel[end_pix].append((curve, "end"))

    # Phase 1: C0 position alignment at endpoints (average anchors)
    for pixel, incidents in by_pixel.items():
        if len(incidents) < 2:
            continue  # nothing to align

        # Collect current anchor positions in (x,y)
        anchors = []
        for curve, pos in incidents:
            cp = curve.control_points
            if pos == "start":
                anchors.append(np.array([cp[0][0], cp[0][1]], dtype=float))
            else:  # end
                anchors.append(np.array([cp[-1][0], cp[-1][1]], dtype=float))

        if len(anchors) == 0:
            continue

        avg_anchor = np.mean(np.stack(anchors, axis=0), axis=0)

        # Set anchors to the averaged value
        for curve, pos in incidents:
            cp = curve.control_points
            if pos == "start":
                cp[0] = (float(avg_anchor[0]), float(avg_anchor[1]))
            else:
                cp[-1] = (float(avg_anchor[0]), float(avg_anchor[1]))

    # Phase 2: C1 tangent alignment when angles are small
    ANGLE_THRESHOLD_DEG = 30.0
    ANGLE_THRESHOLD_RAD = math.radians(ANGLE_THRESHOLD_DEG)
    EPS = 1e-9

    def norm(v: np.ndarray) -> float:
        return float(np.linalg.norm(v))

    def unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < EPS:
            return np.zeros_like(v)
        return v / n

    for pixel, incidents in by_pixel.items():
        if len(incidents) < 2:
            continue

        # Build outgoing tangent directions from the junction
        dirs: list[tuple[Curve, str, np.ndarray, float]] = []  # (curve,pos,dir,mag)
        for curve, pos in incidents:
            cp = curve.control_points
            # Ensure we have enough control points
            if len(cp) < 2:
                continue
            p0 = np.array(cp[0], dtype=float)
            p1 = np.array(cp[1], dtype=float)
            pe = np.array(cp[-1], dtype=float)
            p_e1 = np.array(cp[-2], dtype=float) if len(cp) >= 2 else pe

            if pos == "start":
                t = p1 - p0
                mag = norm(t)
                d = unit(t)
            else:  # end
                t = pe - p_e1
                mag = norm(t)
                # Outgoing direction away from the end anchor is opposite of the end tangent
                d = unit(-t)

            if mag < EPS:
                continue
            dirs.append((curve, pos, d, mag))

        if len(dirs) < 2:
            continue

        # Compute a common direction if they are sufficiently aligned
        mean_dir = unit(np.sum([d for (_, _, d, _) in dirs], axis=0))
        # Check maximum angular deviation
        ok = True
        for _, _, d, _ in dirs:
            dot = float(np.clip(np.dot(d, mean_dir), -1.0, 1.0))
            ang = math.acos(dot)
            if ang > ANGLE_THRESHOLD_RAD:
                ok = False
                break

        if not ok or np.allclose(mean_dir, 0.0):
            continue

        # Apply C1 alignment: keep magnitudes, set directions to mean_dir
        for curve, pos, _, mag in dirs:
            cp = curve.control_points
            if pos == "start":
                p0 = np.array(cp[0], dtype=float)
                new_p1 = p0 + mean_dir * mag
                cp[1] = (float(new_p1[0]), float(new_p1[1]))
            else:  # end
                pe = np.array(cp[-1], dtype=float)
                new_p_e1 = pe - mean_dir * mag
                cp[-2] = (float(new_p_e1[0]), float(new_p_e1[1]))

    # No explicit mid-curve projection implemented here due to endpoint-only mapping input.

    return chains
