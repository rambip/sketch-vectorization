from collections import deque
from typing import Dict, Iterator, List, NewType

import numpy as np
from networkx import Graph, MultiGraph
from numpy.typing import NDArray

from .bezier import fit_bezier, fitting_error, interpolate_bezier
from .utils import Pixel, PixelChain

PixelGraph = NewType("PixelGraph", Graph)


def pixel_graph(skeleton: NDArray[np.bool_]) -> PixelGraph:
    """
    Convert binary skeleton image to 8-connected pixel graph.

    Args:
        skeleton: Binary image where True = skeleton pixels

    Returns:
        Graph where nodes are pixels and edges connect 8-neighbors with euclidean distance weights
    """
    g = Graph()
    h, w = skeleton.shape

    # Add nodes for all skeleton pixels
    for i in range(h):
        for j in range(w):
            if skeleton[i, j]:
                g.add_node((i, j))

    # Add edges for 8-connectivity
    for i, j in g.nodes:
        for ni in range(max(0, i - 1), min(h, i + 2)):
            for nj in range(max(0, j - 1), min(w, j + 2)):
                if (ni, nj) != (i, j) and skeleton[ni, nj]:
                    weight = (i - ni) ** 2 + (j - nj) ** 2
                    g.add_edge((i, j), (ni, nj), weight=weight)

    return PixelGraph(g)


def find_clusters(g: PixelGraph) -> List[List[Pixel]]:
    """
    Find clusters of adjacent junction/leaf nodes using BFS.

    A junction is a node with degree > 2, a leaf has degree == 1.

    Returns:
        List of clusters, where each cluster is a list of adjacent junction/leaf pixels
    """

    def is_endpoint(node):
        deg = g.degree(node)
        return deg == 1 or deg > 2

    visited = set()
    clusters = []

    for node in g.nodes:
        if node in visited or not is_endpoint(node):
            continue

        # BFS to find all adjacent endpoints
        cluster = []
        queue = deque([node])

        while queue:
            curr = queue.popleft()
            if curr in visited:
                continue
            visited.add(curr)
            cluster.append(curr)

            for neigh in g.neighbors(curr):
                if neigh not in visited and is_endpoint(neigh):
                    queue.append(neigh)

        clusters.append(cluster)

    return clusters


def _get_chain_from(
    g: PixelGraph,
    start: Pixel,
    visited: NDArray[np.bool_],
    node_map: Dict[Pixel, Pixel],
) -> List[Pixel]:
    """
    Compute a pixel chain starting from a given pixel by following unvisited neighbors until reaching an endpoint or cluster representative.
    Note that p is in `node_map` if and only if the degree of p is not 2.
    """
    chain = []
    current = start
    # if current in visited:
    #    # weird case
    #    return chain + [start]
    while True:
        if current in node_map:
            # add representative at the end
            return chain + [node_map[current]]

        chain.append(current)
        visited[current] = True

        unvisited = [n for n in g.neighbors(current) if not visited[n]]

        if len(unvisited) == 0:
            # only possible for self-loop
            return chain

        assert len(unvisited) == 1
        current = unvisited[0]


def extract_chains(img: NDArray[np.bool_]) -> List[PixelChain]:
    """
    Extract pixel chains from graph by BFS traversal between endpoints.

    If clusters are provided, they are merged to single representatives (median position).
    Chains will start and end at junctions, leaves, or cluster representatives.

    Args:
        g: Pixel graph with 8-connectivity
        clusters: Optional clusters to merge

    Returns:
        List of pixel chains (numpy arrays of shape (N, 2))
    """
    # Build cluster mapping: pixel -> representative
    g = pixel_graph(img)
    node_map: Dict[Pixel, Pixel] = {}
    cluster_members = set()  # All pixels that are part of a cluster

    clusters = find_clusters(g)

    for cluster in clusters:
        # Find mean position as representative
        coords = np.array(cluster).astype(float)
        mean_point = np.mean(coords, axis=0)

        rep = min(
            cluster,
            key=lambda p: (p[0] - mean_point[0]) ** 2 + (p[1] - mean_point[1]) ** 2,
        )
        for node in cluster:
            cluster_members.add(node)
            node_map[node] = rep

    def get_repr(node):
        return node_map.get(node, node)

    def is_endpoint(node):
        deg = g.degree(node)
        return deg == 1 or deg > 2

    # Initialize closed dict: pre-close all cluster nodes except representatives
    chains = []
    visited = np.zeros(img.shape, dtype=bool)

    for start in g.nodes:
        neighbors = list(g.neighbors(start))
        if visited[start] or start in node_map:
            continue
        if len(neighbors) != 2:
            continue

        visited[start] = True
        left_chain = _get_chain_from(g, neighbors[0], visited, node_map)
        right_chain = _get_chain_from(g, neighbors[1], visited, node_map)
        left_chain.reverse()
        full_chain = left_chain + [start] + right_chain
        chains.append(np.array(full_chain))

    return chains


def remove_parasite_chains(
    chains: List[PixelChain], min_length: int = 10
) -> List[PixelChain]:
    """
    Remove shorts chains that are not meaningful.
    Seing the chains endpoints as a graph, we:
        1. Remove all chains that are shorter that `min_length` and end at a node of degree one
        2. Remove all self-loops that are shorter than `min_length`
        3. Remove all chains that are shorter than `min_length` and part of a multi-edge (except for one)
    Args:
        chains: List of pixel chains to filter
        min_length: Minimum number of pixels for a chain to be kept

    Returns:
        Filtered list of pixel chains
    """
    # Build endpoint graph
    g = Graph()
    mg = MultiGraph()
    for chain in chains:
        assert len(chain) >= 2
        start, end = tuple(chain[0]), tuple(chain[-1])
        mg.add_edge(start, end, chain=chain)
        g.add_edge(start, end)

    def first_chain(u, v):
        return mg[u][v][0]["chain"]  # type: ignore

    filtered_chains = []
    for u, v in g.edges:
        chain = first_chain(u, v)
        if u == v:
            # self-loop case
            if len(chain) >= min_length:
                filtered_chains.append(chain)
            continue
        elif mg.number_of_edges(u, v) > 1:
            multichains = [mg[u][v][k]["chain"] for k in mg[u][v]]
            multichains.sort(key=len)
            main_chain = multichains[-1]
            filtered_chains.append(main_chain)
            filtered_chains.extend(c for c in multichains[:-1] if len(c) >= min_length)
        elif g.degree[u] == 1 or g.degree[v] == 1:
            if len(chain) >= min_length:
                filtered_chains.append(chain)
        else:
            filtered_chains.append(chain)
    return filtered_chains


def _fit_error(chain: PixelChain, degree: int = 3) -> float:
    """
    Compute the fitting error when approximating a pixel chain with a Bézier curve.

    Args:
        chain: Numpy array of shape (N, 2) with pixel coordinates
        degree: Degree of the Bézier curve to fit

    Returns:
        The standard distance error between the fitted curve and original pixels
    """
    assert len(chain) >= 2

    # Fit Bézier curve
    instants = np.linspace(0, 1, len(chain))
    control_points = fit_bezier(chain, instants, degree=degree)

    # Evaluate fitted curve at same instants
    fitted = interpolate_bezier(control_points, instants)

    # Compute error
    return fitting_error(fitted, chain)


def _find_best_split(chain: PixelChain, degree: int = 3, margin=2) -> int:
    """
    Find the optimal split index for a chain using binary search.

    Compares adjacent split points using their total fitting error to guide
    the search direction (gradient-like approach).

    Args:
        chain: Numpy array of shape (N, 2) with pixel coordinates
        degree: Degree of Bézier curves to fit

    Returns:
        The index where the chain should be split (1 <= index < len(chain))
    """
    n = len(chain)
    left, right = margin, n - 1 - margin

    while left < right:
        mid = (left + right) // 2

        # Compare error at mid vs mid+1
        error_mid = _fit_error(chain[: mid + 1], degree) + _fit_error(
            chain[mid:], degree
        )
        error_mid_plus_1 = _fit_error(chain[: mid + 2], degree) + _fit_error(
            chain[mid + 1 :], degree
        )

        if error_mid < error_mid_plus_1:
            right = mid  # Optimum is at or to the left of mid
        else:
            left = mid + 1  # Optimum is to the right of mid

    return left


def refine_chain(
    chain: PixelChain, tolerance: float = 1.0, degree: int = 3, margin=2
) -> Iterator[PixelChain]:
    """
    Subdivide a pixel chain into smaller chains where each can be accurately
    fitted by a single Bézier curve.

    Uses recursive subdivision: if the chain fits within tolerance, yield it as-is.
    Otherwise, find the optimal split point using binary search and recursively refine.

    Args:
        chain: Numpy array of shape (N, 2) with pixel coordinates
        tolerance: Maximum allowed fitting error
        degree: Degree of Bézier curves to fit

    Yields:
        Sub-chains that each fit within the tolerance
    """
    if len(chain) <= 2 * margin + 1:
        yield chain
        return

    error = _fit_error(chain, degree)

    # Base case: chain fits well enough
    if error <= tolerance:
        yield chain
        return

    # Find optimal split point using binary search
    split_idx = _find_best_split(chain, degree)
    assert split_idx >= margin
    assert split_idx <= len(chain) - margin

    # Recursively refine both parts
    yield from refine_chain(chain[: split_idx + 1], tolerance, degree)
    yield from refine_chain(chain[split_idx:], tolerance, degree)


def refine_all_chains(
    chains: List[PixelChain], tolerance: float = 2.0, degree: int = 3
) -> List[PixelChain]:
    """
    Refine all chains by subdividing them where necessary for accurate Bézier fitting.

    Args:
        chains: List of pixel chains to refine
        tolerance: Maximum allowed fitting error for each sub-chain
        degree: Degree of Bézier curves to fit

    Returns:
        Flattened list of all refined sub-chains
    """
    result = []
    for chain in chains:
        result.extend(refine_chain(chain, tolerance, degree))
    return result
