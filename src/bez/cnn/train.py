import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import trange
from bez.bezier import interpolate_bezier
import random
from bez.cnn.model import Rasterizer


def softmax(x, axis):
    exps = np.exp(x)
    return exps / exps.sum(axis)


def create_pattern(theta_a, theta_b, size, sigma=1, r=1):
    t = np.linspace(0, 1, 30, endpoint=True)
    """Create a single pattern using vectorized operations."""
    # Pre-compute coordinate grids
    i_coords, j_coords = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    x_coords = 2 * i_coords / (size - 1) - 1
    y_coords = 2 * j_coords / (size - 1) - 1

    # Create mask for circular region
    circle_mask = (x_coords**2 + y_coords**2) < 1.2

    # Create coefficients for Bezier curve
    coeffs = r * np.array(
        [[np.cos(theta_a), 0, np.cos(theta_b)], [np.sin(theta_a), 0, np.sin(theta_b)]]
    )

    # Compute trajectory
    traj = interpolate_bezier(coeffs, t)  # Shape: (2, 30)

    # Vectorized computation for all trajectory points at once
    # Reshape coordinates for broadcasting: (size, size, 1) and trajectory (1, 1, 30)
    x_grid = x_coords[:, :, np.newaxis]
    y_grid = y_coords[:, :, np.newaxis]
    traj_x = traj[0, :][np.newaxis, np.newaxis, :]
    traj_y = traj[1, :][np.newaxis, np.newaxis, :]

    # Compute distances from all grid points to all trajectory points
    d_square = (x_grid - traj_x) ** 2 + (y_grid - traj_y) ** 2

    # Apply Gaussian and sum over trajectory points
    gaussian_values = np.exp(-1 / (2 * sigma) * 2 * size * d_square)
    pattern = np.sum(gaussian_values, axis=2)

    # Apply circular mask
    pattern = pattern * circle_mask

    # Normalize pattern
    pattern = pattern / np.max(pattern)

    return pattern, pattern > 0.5


def create_pattern_bank(sizes):
    """Create bank with patterns of different sizes efficiently."""
    bank = []
    bank_bin = []

    # Define parameter ranges
    theta_as = np.linspace(0, np.pi, 6, endpoint=False)
    sigmas = [0.15, 0.3]
    d_thetas = [-np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3]

    # Generate all combinations
    for size in sizes:
        for theta_a in theta_as:
            for sigma in sigmas:
                for d_theta in d_thetas:
                    theta_b = theta_a + np.pi + d_theta
                    pattern, pattern_bin = create_pattern(theta_a, theta_b, size, sigma)
                    bank.append(pattern)
                    bank_bin.append(pattern_bin)

    return bank, bank_bin


# %%
def gen_synthetic_images(bank, bank_bin, batch_size, H=200, W=200, num_patterns=10):
    # Initialize output tensors
    components = torch.zeros((batch_size, 1, H, W))
    components_bin = torch.zeros((batch_size, 1, H, W))

    # Convert bank to torch tensors (keeping as list since different sizes)
    bank_tensor = [torch.FloatTensor(pattern) for pattern in bank]
    bank_bin_tensor = [torch.FloatTensor(pattern) for pattern in bank_bin]

    # Find max pattern size for padding
    max_size = max(pattern.shape[0] for pattern in bank)
    pad_size = max_size // 2

    for batch_idx in range(batch_size):
        # Add random patterns to each image
        for _ in range(num_patterns):
            # Select random pattern from bank
            pattern_idx = random.randint(0, len(bank_tensor) - 1)
            pattern = bank_tensor[pattern_idx]
            pattern_bin = bank_bin_tensor[pattern_idx]

            # Get pattern size
            pattern_h, pattern_w = pattern.shape

            # Choose random position (ensure pattern fits within image bounds)
            max_y = H - pattern_h
            max_x = W - pattern_w

            if max_y > 0 and max_x > 0:
                y = random.randint(0, max_y)
                x = random.randint(0, max_x)

                # Add pattern to the image
                intensity = 0.5 + 0.5 * random.random()
                components[batch_idx, 0, y : y + pattern_h, x : x + pattern_w] += (
                    intensity * pattern
                )
                components_bin[batch_idx, 0, y : y + pattern_h, x : x + pattern_w] += (
                    pattern_bin
                )

    components += 0.2 * torch.randn((batch_size, 1, H, W))
    return components.clip(0, 1), components_bin.clip(0, 1)


# %%
def train_model(model, show_images=False):
    sizes = [31, 45, 61, 75]
    # size of the patterns in the pattern bank
    bank, bank_bin = create_pattern_bank(sizes)
    criterion = torch.nn.BCELoss()
    opti = torch.optim.Adam(model.parameters(), lr=0.5e-1)
    losses = []
    N_iter = 150

    for step in trange(N_iter):
        x, y = gen_synthetic_images(bank, bank_bin, 10, 300, 300, 20)
        opti.zero_grad()
        y_pred = model.forward(x)
        loss = criterion(y_pred, y)
        loss.backward()
        opti.step()
        if show_images and step % 30 == 0:
            plt.imshow(y_pred[0][0].detach())
            plt.colorbar()
            plt.show()

        losses.append(loss.detach())
    return losses


# %%

if __name__ == "__main__":
    model = Rasterizer(32)
    losses = train_model(model)
    plt.plot(range(len(losses)), losses, label="reconstruction loss")
    plt.legend()
    plt.show()
    dummy_input = torch.rand((1, 256, 256), dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy_input,),
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        dynamic_axes={
            "input": {1: "height", 2: "width"},  # Variable H, W
            "output": {1: "height", 2: "width"},  # Variable H, W
        },
    )
