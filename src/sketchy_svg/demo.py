"""Visualization widgets for sketchy_new."""

import copy
import io

import anywidget
import matplotlib.pyplot as plt
import numpy as np
import traitlets
from matplotlib.patches import ConnectionStyle, FancyArrowPatch
from numpy.typing import NDArray
from skimage.morphology import skeletonize

from .bezier import interpolate_bezier
from .optim import Perturbation, SketchOptimizer, SuperGraph, align_boundaries
from .prepare import BinarySketchPredictor, compute_thickness_map, load_normalized
from .topology import extract_chains, refine_all_chains, remove_parasite_chains
from .utils import PixelChain


def plot_directed_edge(
    ax, control_points: NDArray[np.float32], instants=None, color="gray"
):
    if instants is None:
        instants = np.linspace(0, 1, 100)
    bez = interpolate_bezier(control_points, instants)
    ax.plot(bez[:, 1], bez[:, 0], "-", linewidth=1, alpha=0.6, color=color)
    # Orientation caret (gray)
    i0 = max(1, int(0.7 * (len(bez) - 1)))
    i1 = min(len(bez) - 1, i0 + 2)
    ax.annotate(
        "",
        xy=(bez[i1, 1], bez[i1, 0]),
        xytext=(bez[i0, 1], bez[i0, 0]),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1, shrinkA=0, shrinkB=0),
    )


class OptimPlayBack(anywidget.AnyWidget):
    """Widget to playback optimization steps."""

    _css = """
    .optim-widget {
      font-family: system-ui, -apple-system, sans-serif;
      padding: 16px;
      border: 1px solid #ddd;
      border-radius: 8px;
      width: 450px;
    }
    .optim-chart {
      margin-bottom: 16px;
      width: 100%;
      min-height: 300px;
    }
    .optim-chart svg {
      width: 100%;
      height: auto;
      display: block;
    }
    .optim-info-container {
      flex: 1;
    }
    .optim-row {
      margin-bottom: 3px;
      font-size: 14px;
    }
    .optim-label {
      font-weight: bold;
    }
    .optim-value {
      font-weight: normal;
    }
    .optim-perturbation {
      font-style: italic;
    }
    .optim-delta {
      font-weight: bold;
    }
    .optim-delta.positive {
      color: #f44336;
    }
    .optim-delta.negative {
      color: #4CAF50;
    }
    .optim-delta.zero {
      color: #999999;
    }
    .optim-controls {
      display: flex;
      gap: 16px;
      align-items: flex-start;
    }
    .optim-button-container {
      display: flex;
      align-items: center;
    }
    .optim-btn {
      padding: 8px 16px;
      border: 1px solid #ccc;
      border-radius: 4px;
      cursor: pointer;
    }
    .optim-btn:disabled {
      background-color: #f5f5f5;
      cursor: not-allowed;
    }
    .optim-btn-next {
      background-color: #4CAF50;
      color: white;
      font-weight: bold;
    }
    .optim-btn-next:disabled {
      background-color: #ccc;
    }
    """

    _esm = """
    function render({ model, el }) {
      // Add main container class
      el.classList.add('optim-widget');

      // Chart container
      const chartContainer = document.createElement('div');
      chartContainer.classList.add('optim-chart');

      // Info display container
      const infoContainer = document.createElement('div');
      infoContainer.classList.add('optim-info-container');

      // Step info
      const stepDiv = document.createElement('div');
      stepDiv.classList.add('optim-row');

      const stepLabel = document.createElement('span');
      stepLabel.classList.add('optim-label');
      stepLabel.textContent = 'Step: ';

      const stepValue = document.createElement('span');
      stepValue.classList.add('optim-value');

      stepDiv.appendChild(stepLabel);
      stepDiv.appendChild(stepValue);

      // Energy info
      const energyDiv = document.createElement('div');
      energyDiv.classList.add('optim-row');

      const energyLabel = document.createElement('span');
      energyLabel.classList.add('optim-label');
      energyLabel.textContent = 'Energy: ';

      const energyValue = document.createElement('span');
      energyValue.classList.add('optim-value');

      energyDiv.appendChild(energyLabel);
      energyDiv.appendChild(energyValue);

      // Delta energy info
      const deltaDiv = document.createElement('div');
      deltaDiv.classList.add('optim-row');

      const deltaLabel = document.createElement('span');
      deltaLabel.classList.add('optim-label');
      deltaLabel.textContent = 'Delta: ';

      const deltaValue = document.createElement('span');
      deltaValue.classList.add('optim-delta');

      deltaDiv.appendChild(deltaLabel);
      deltaDiv.appendChild(deltaValue);

      // Perturbation info
      const pertDiv = document.createElement('div');
      pertDiv.classList.add('optim-row');

      const pertLabel = document.createElement('span');
      pertLabel.classList.add('optim-label');
      pertLabel.textContent = 'Perturbation: ';

      const pertValue = document.createElement('span');
      pertValue.classList.add('optim-perturbation');

      pertDiv.appendChild(pertLabel);
      pertDiv.appendChild(pertValue);

      // Assemble info container
      infoContainer.appendChild(stepDiv);
      infoContainer.appendChild(energyDiv);
      infoContainer.appendChild(deltaDiv);
      infoContainer.appendChild(pertDiv);

      // Buttons container
      const buttonContainer = document.createElement('div');
      buttonContainer.classList.add('optim-button-container');

      // Next button
      const nextButton = document.createElement('button');
      nextButton.textContent = 'Next';
      nextButton.classList.add('optim-btn', 'optim-btn-next');

      buttonContainer.appendChild(nextButton);

      // Controls row: info on the left, button on the right
      const controlsRow = document.createElement('div');
      controlsRow.classList.add('optim-controls');
      controlsRow.appendChild(infoContainer);
      controlsRow.appendChild(buttonContainer);

      // Assemble widget
      el.appendChild(chartContainer);
      el.appendChild(controlsRow);

      // Update function - use direct references to elements
      function updateDisplay() {
        const currentStep = model.get('current_step');
        const energy = model.get('current_energy');
        const delta = model.get('current_delta');
        const perturbation = model.get('current_perturbation');
        const historyLength = model.get('history_length');
        const chartHtml = model.get('chart_html');

        // Update chart
        chartContainer.innerHTML = chartHtml || '';

        stepValue.textContent = `${currentStep} / ${historyLength}`;
        energyValue.textContent = energy.toFixed(4);

        if (delta === null) {
          deltaValue.textContent = '-';
          deltaValue.classList.remove('positive', 'negative', 'zero');
        } else {
          const sign = delta > 0 ? '+' : '';
          deltaValue.textContent = `${sign}${delta.toFixed(4)}`;
          deltaValue.classList.remove('positive', 'negative', 'zero');
          if (delta > 0) {
            deltaValue.classList.add('positive');
          } else if (delta < 0) {
            deltaValue.classList.add('negative');
          } else {
            deltaValue.classList.add('zero');
          }
        }

        pertValue.textContent = perturbation || '-';

        // Update button states
        if (currentStep >= historyLength) {
          nextButton.disabled = true;
        } else {
          nextButton.disabled = false;
        }
      }

      // Event listeners
      nextButton.addEventListener('click', () => {
        model.set('next_triggered', model.get('next_triggered') + 1);
        model.save_changes();
      });

      // Initial update and watch for changes
      updateDisplay();
      model.on('change:current_step', updateDisplay);
      model.on('change:current_energy', updateDisplay);
      model.on('change:current_delta', updateDisplay);
      model.on('change:current_perturbation', updateDisplay);
      model.on('change:history_length', updateDisplay);
      model.on('change:chart_html', updateDisplay);
    }
    export default { render };
    """

    # Widget state
    current_step = traitlets.Int(0).tag(sync=True)
    current_energy = traitlets.Float(0.0).tag(sync=True)
    current_delta = traitlets.Float(allow_none=True, default_value=None).tag(sync=True)
    current_perturbation = traitlets.Unicode("").tag(sync=True)
    history_length = traitlets.Int(0).tag(sync=True)
    next_triggered = traitlets.Int(0).tag(sync=True)
    chart_html = traitlets.Unicode("").tag(sync=True)

    def __init__(
        self,
        initial_chains: list[PixelChain],
        thickness_map: NDArray,
        history: list[tuple],
        lam: float = 0.6,
        mu: float = 0.3,
        **kwargs,
    ):
        """Initialize the playback widget.

        Args:
            initial_chains: Initial list of pixel chains
            history: List of perturbations (perturbation, s_a, pixel, s_b)
            lam: Lambda parameter for energy computation
            mu: Mu parameter for energy computation
        """
        super().__init__(**kwargs)
        self._initial_chains = initial_chains
        self._thickness_map = thickness_map
        self._history = history
        self._lam = lam
        self._mu = mu

        # Initialize supergraph and state
        self._supergraph = SuperGraph(initial_chains, thickness_map)
        self._current_idx = 0
        self._energies = [self._compute_energy()]

        # Store previous state for visualization
        self._prev_supergraph = None
        self._last_new_ids: list[int] = []

        # Set initial display
        self.history_length = len(history)
        self._update_display()

        # Watch for button clicks
        self.observe(self._on_next, names="next_triggered")

    def _compute_energy(self) -> float:
        """Compute current energy from supergraph."""
        return sum(
            edge.score(self._lam, self._mu) for edge in self._supergraph.superedges
        )

    def _format_perturbation(
        self, perturbation: Perturbation | int, s_a: int, pixel: tuple, s_b: int | None
    ) -> str:
        """Pretty print perturbation information."""
        pert_name = Perturbation(perturbation).name

        if perturbation in (Perturbation.INCREASE_DEGREE, Perturbation.DECREASE_DEGREE):
            return f"{pert_name}: superedge {s_a}"
        elif perturbation == Perturbation.MERGE:
            return f"{pert_name}: superedge {s_a} → superedge {s_b} at {pixel}"
        elif perturbation == Perturbation.SPLIT:
            return f"{pert_name}: superedge {s_a} at {pixel}"
        elif perturbation == Perturbation.OVERLAP:
            return f"{pert_name}: superedge {s_a} ← superedge {s_b} at {pixel}"
        elif perturbation == Perturbation.DISSOCIATE:
            return f"{pert_name}: superedge {s_a} removes part from superedge {s_b} at {pixel}"
        elif perturbation == Perturbation.REVERSE:
            return f"{pert_name}: superedge {s_a}"
        else:
            return f"{pert_name}: superedge {s_a}"

    def _create_chart(self):
        """Create matplotlib chart and convert to inline SVG."""
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")

        instants = np.linspace(0, 1, 100)

        def plot_edge_colored(idx: int, color: str):
            if 0 <= idx < len(self._supergraph.superedges):
                e = self._supergraph.superedges[idx]
                bez = interpolate_bezier(e.control_points, instants)
                ax.plot(bez[:, 1], bez[:, 0], "-", linewidth=2, alpha=0.95, color=color)
                # Orientation caret (colored)
                i0 = max(1, int(0.7 * (len(bez) - 1)))
                i1 = min(len(bez) - 1, i0 + 2)
                ax.annotate(
                    "",
                    xy=(bez[i1, 1], bez[i1, 0]),
                    xytext=(bez[i0, 1], bez[i0, 0]),
                    arrowprops=dict(
                        arrowstyle="-|>", color=color, lw=1.5, shrinkA=0, shrinkB=0
                    ),
                )
                # Label offset away from the curve using local normal at mid
                mid = len(bez) // 2
                p_prev = bez[max(0, mid - 1)]
                p_next = bez[min(len(bez) - 1, mid + 1)]
                tangent = p_next - p_prev
                normal = np.array([tangent[1], -tangent[0]])
                nrm = np.linalg.norm(normal) or 1.0
                tnorm = np.linalg.norm(tangent) or 1.0
                normal = (normal / nrm) * 12.0
                tangent_unit = tangent / tnorm
                tshift = (1.0 if (idx % 2 == 0) else -1.0) * 4.0 * tangent_unit
                ax.text(
                    bez[mid, 1] + normal[1] + tshift[1],
                    bez[mid, 0] + normal[0] + tshift[0],
                    str(idx),
                    fontsize=9,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
                )

        # Determine what to show based on current step
        if self._current_idx == 0 or self._prev_supergraph is None:
            # Initial state - draw all edges in gray, no numbers
            for edge in self._supergraph.superedges:
                plot_directed_edge(ax, edge.control_points, instants, color="gray")
        else:
            # Background: previous state in gray, no labels
            for edge in self._prev_supergraph.superedges:
                plot_directed_edge(ax, edge.control_points, instants, color="gray")

            # Foreground: only the changed edge(s) in color with labels
            # For merge, split, overlap, dissociate this list will be size 1 or 2
            colors = ["C1", "C2", "C3", "C4"]
            for j, idx in enumerate(self._last_new_ids):
                plot_edge_colored(idx, color=colors[j % len(colors)])

            # Add specific highlights based on perturbation type
            prev_idx = self._current_idx - 1
            if prev_idx < len(self._history):
                pert, s_a, pixel, s_b = self._history[prev_idx]

                if pert == Perturbation.MERGE:
                    # Highlight merge pixel
                    ax.plot(pixel[1], pixel[0], "ro", markersize=10, alpha=0.8)
                elif pert == Perturbation.SPLIT:
                    # Highlight split pixel
                    ax.plot(pixel[1], pixel[0], "ro", markersize=10, alpha=0.8)
                elif pert == Perturbation.OVERLAP:
                    # Highlight the connection point
                    ax.plot(pixel[1], pixel[0], "go", markersize=10, alpha=0.8)
                elif pert == Perturbation.DISSOCIATE:
                    # Highlight the dissociation point
                    ax.plot(pixel[1], pixel[0], "ro", markersize=10, alpha=0.8)

        plt.tight_layout()
        # Save to SVG string
        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
        buf.seek(0)
        svg_content = buf.read()
        buf.close()
        plt.close(fig)
        # Extract just the SVG element content (remove XML declaration if present)
        if "<svg" in svg_content:
            svg_content = svg_content[
                svg_content.find("<svg") : svg_content.find("</svg>") + 6
            ]
        return svg_content

    def _update_display(self):
        """Update the widget display."""
        self.current_step = self._current_idx
        self.current_energy = self._energies[-1] if self._energies else 0.0

        if self._current_idx == 0:
            self.current_delta = None
            self.current_perturbation = "Initial state"
        else:
            self.current_delta = self._energies[-1] - self._energies[-2]
            pert, s_a, pixel, s_b = self._history[self._current_idx - 1]
            self.current_perturbation = self._format_perturbation(pert, s_a, pixel, s_b)

        # Update chart
        self.chart_html = self._create_chart()

    def _on_next(self, change):
        """Handle next button click."""
        if self._current_idx < len(self._history):
            # Save current state before applying perturbation
            self._prev_supergraph = copy.deepcopy(self._supergraph)

            # Apply next perturbation
            pert, s_a, pixel, s_b = self._history[self._current_idx]
            self._last_new_ids = self._supergraph.perturbate(pert, s_a, pixel, s_b)

            # Update state
            self._current_idx += 1
            self._energies.append(self._compute_energy())
            self._update_display()

    def reset(self):
        """Reset to initial state."""
        self._supergraph = SuperGraph(self._initial_chains, self._thickness_map)
        self._prev_supergraph = None
        self._last_new_ids = []
        self._current_idx = 0
        self._energies = [self._compute_energy()]
        self._update_display()


MPL_WRAPPER = None
STATUS_FUNCTION = None


def set_options(options: dict):
    global STATUS_FUNCTION
    global MPL_WRAPPER
    STATUS_FUNCTION = options.get("status_function", lambda x, title: x)
    MPL_WRAPPER = options.get("mpl_wrapper", None)


async def show_example(path):
    img = load_normalized(path)
    classifier = BinarySketchPredictor()
    prediction = await classifier.predict(img)
    thickness_map = compute_thickness_map(prediction)
    skeleton = skeletonize(prediction)
    chains = extract_chains(skeleton)
    pixel_chains_refine = refine_all_chains(remove_parasite_chains(chains))

    assert STATUS_FUNCTION is not None
    optim = SketchOptimizer(status_function=STATUS_FUNCTION)
    curves = optim.fit_transform(pixel_chains_refine, thickness_map)
    final_curves = align_boundaries(
        curves, optim.endpoint_mapping_, optim.interior_mapping_
    )

    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    axes[0, 0].imshow(img, cmap="binary")
    axes[0, 0].set_title("Original image")
    axes[0, 0].axis(False)
    axes[0, 0].axis("equal")

    axes[0, 1].imshow(prediction, cmap="binary")
    axes[0, 1].set_title("Black and white prediction")
    axes[0, 1].axis(False)
    axes[0, 1].axis("equal")

    axes[0, 2].imshow(skeleton, cmap="binary")
    axes[0, 2].set_title("Skeleton")
    axes[0, 2].axis(False)
    axes[0, 2].axis("equal")

    for c in pixel_chains_refine:
        axes[1, 2].plot(c[:, 1], c[:, 0], linewidth=1)
    axes[1, 2].set_title("Refined pixel chains")
    axes[1, 2].invert_yaxis()
    axes[1, 2].axis(False)
    axes[1, 2].axis("equal")

    for cu in curves:
        plot_directed_edge(axes[1, 1], cu.control_points)
    axes[1, 1].set_title("Curves after optimization")
    axes[1, 1].invert_yaxis()
    axes[1, 1].axis(False)
    axes[1, 1].axis("equal")

    time = np.linspace(0, 1, 100)
    for cu in final_curves:
        bezier_points = interpolate_bezier(cu.control_points, time)
        axes[1, 0].plot(
            bezier_points[:, 1],
            bezier_points[:, 0],
            color="black",
            linewidth=cu.stroke_width,
        )
    axes[1, 0].set_title("Final SVG")
    axes[1, 0].invert_yaxis()
    axes[1, 0].axis(False)
    axes[1, 0].axis("equal")

    plt.tight_layout()
    fig.canvas.draw()

    arrow_kw = dict(
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=25,
        lw=2.5,
        color="tomato",
        clip_on=False,
        zorder=10,
    )

    def addarrow(start, end, connectionstyle=None):
        fig.add_artist(
            FancyArrowPatch(start, end, connectionstyle=connectionstyle, **arrow_kw)  # type: ignore
        )

    # Top row: left to right
    addarrow((0.320, 0.720), (0.350, 0.720))
    addarrow((0.633, 0.720), (0.663, 0.720))

    # Right column: top to bottom, bendy
    addarrow(
        (0.95, 0.6),
        (0.95, 0.4),
        connectionstyle=ConnectionStyle("Arc3", rad=-0.4),
    )

    # Bottom row: right to left
    addarrow((0.680, 0.235), (0.650, 0.235))
    addarrow((0.350, 0.235), (0.320, 0.235))

    if MPL_WRAPPER is not None:
        return MPL_WRAPPER(fig)
    return fig
