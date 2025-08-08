import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import sparse
from scipy import ndimage as ndi
import torch
from tqdm import tqdm, trange
from skimage import data, color, io, filters
import skimage as ski
import torchvision.transforms as transforms


# %%
def interpolate_bezier(coeffs, t):
    """
    compute the trajectory of a bezier-curve.

    coeffs: a 2x(d+1) matrix, where d is the degree of the bezier.
        - row 0 for the x coordinates of control points
        - row 1 for the y coordinates of control points

    t: a 1d array of instant to evaluate
    """
    d = coeffs.shape[1] - 1
    assert coeffs.shape[0] == 2

    q = np.array([2 * (1 - t) * (0.5 - t), 4 * t * (1 - t), 2 * t * (t - 0.5)])

    return coeffs @ q


def softmax(x, axis):
    exps = np.exp(x)
    return exps / exps.sum(axis)


# %%




# Create patterns efficiently
bank, bank_bin = create_bank_efficient(sizes=[21, 31, 41, 51])
print(f"Created {len(bank)} patterns")

plt.figure(figsize=(12, 4))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(bank[i*64])
    plt.title(f"Size: {bank[i*64].shape[0]}")
plt.show()


# %%
def gen_synthetic_images(batch_size, H=200, W=200, num_patterns=20):
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
                intensity = random.random()
                components[batch_idx, 0, y:y+pattern_h, x:x+pattern_w] += intensity * pattern
                components_bin[batch_idx, 0, y:y+pattern_h, x:x+pattern_w] += pattern_bin

    # Add noise
    components += 0.02 * torch.randn((batch_size, 1, H, W))

    return components, components_bin.clip(0, 1)


test, test_bin = gen_synthetic_images(1, num_patterns=50)
plt.imshow(test[0][0])
plt.colorbar()
plt.show()
plt.imshow(test_bin[0][0])
plt.show()

# %%
d_dict = len(bank)


class Rasterizer(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, d_model, 7, padding=4)
        self.reg1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(d_model, d_model, 5, dilation=3, padding=5)
        self.reg2 = torch.nn.ReLU()

        self.mlp = torch.nn.Conv2d(d_model, 1, 1)
        self.final = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.reg1(self.conv1(x))
        x = self.reg2(self.conv2(x))
        x = self.final(self.mlp(x))
        return x


# %%
model = Rasterizer(32)
criterion = torch.nn.BCELoss()
opti = torch.optim.Adam(model.parameters(), lr=1e-1)
losses = []
N_iter = 151

for step in trange(N_iter):
    x, y = gen_synthetic_images(10, 150, 150)
    batch_size = x.shape[0]
    opti.zero_grad()
    y_pred = model.forward(x)
    loss = criterion(y_pred, y)
    loss.backward()
    opti.step()
    if step % 30 == 0:
        plt.imshow(y_pred[0][0].detach())
        plt.colorbar()
        plt.show()

    losses.append(loss.detach())

# %%
plt.plot(range(N_iter), losses, label="reconstruction loss")
plt.legend()
plt.show()


# %%
def load(path):
    image = io.imread(path)[:, :, 0]
    image = ski.transform.rescale(image, 0.2, order=2)
    image = 1 - image
    image = image / np.max(image)
    return image


paths = [
    "https://raw.githubusercontent.com/rambip/sketch-vectorization/refs/heads/main/data/sketches/butterfly.png",
    "https://raw.githubusercontent.com/rambip/sketch-vectorization/refs/heads/main/data/sketches/dress.png",
    "https://raw.githubusercontent.com/rambip/sketch-vectorization/refs/heads/main/data/sketches/football.png",
    "https://raw.githubusercontent.com/rambip/sketch-vectorization/refs/heads/main/data/sketches/house.png",
    "https://raw.githubusercontent.com/rambip/sketch-vectorization/refs/heads/main/data/sketches/piano.png",
    "https://raw.githubusercontent.com/rambip/sketch-vectorization/refs/heads/main/data/sketches/triangle.png",
]
images = [load(p) for p in paths]
plt.imshow(images[0])
plt.colorbar()

# %%
i = 2
plt.imshow(images[i])
plt.colorbar()
plt.show()
test_input = torch.FloatTensor(images[i]).unsqueeze(0).unsqueeze(1)
out = model.forward(test_input)[0]
print(out.shape)
plt.imshow(out.detach()[0] > 0.3)
plt.colorbar()
plt.show()


# %%
import torch.onnx

dummy_input = torch.randn(1, 200, 200)
torch.onnx.export(
    model,
    dummy_input,
    "rasterizer.onnx",
    export_params=True,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {1: 'height', 2: 'width'},    # Variable H, W
        'output': {1: 'height', 2: 'width'}    # Variable H, W
    }
)

# %%
dynamic axes
