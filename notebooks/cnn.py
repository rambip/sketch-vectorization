import marimo

__generated_with = "0.19.9"
app = marimo.App()


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import trange
    from bez.cnn.model import Rasterizer
    from bez.data import load_normalized
    import torch
    from meta import ROOT_DIR
    from torchvision.transforms import v2
    import random

    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    return (
        ROOT_DIR,
        Rasterizer,
        load_normalized,
        plt,
        random,
        torch,
        trange,
        v2,
    )


@app.cell
def _(ROOT_DIR, load_normalized, random, torch, v2):
    transforms = v2.Compose([v2.RandomRotation((-180, 180)), v2.RandomCrop(size=(128, 128)), v2.RandomHorizontalFlip(p=0.5)])
    N_IMAGES = 4

    def load_data_pair(path_x, path_y, device):
        source_x = load_normalized(path_x)
        source_y = 1 - load_normalized(path_y, only_alpha=True)
        return (torch.tensor(source_x, device=device, dtype=torch.float32), torch.tensor(source_y, device=device, dtype=torch.float32))
    sources = [(load_normalized(ROOT_DIR / 'data' / 'train' / f'x_00{i}.png'), 1 - load_normalized(ROOT_DIR / 'data' / 'train' / f'y_00{i}.png', only_alpha=True)) for i in range(N_IMAGES)]

    def load_original_dataset(device='cuda'):
        base = ROOT_DIR / 'data' / 'train'
        return [load_data_pair(base / f'x_00{i}.png', base / f'y_00{i}.png', device) for i in range(N_IMAGES)]
    preloaded_sources = load_original_dataset()

    def random_labeled_img(batch_size, device='cuda'):
        result_x = []
        result_y = []
        for _ in range(batch_size):
            source_id = random.randint(0, N_IMAGES - 1)
            (source_x_t, source_y_t) = preloaded_sources[source_id]
            while True:
                input_tensor = torch.cat([source_x_t.unsqueeze(0), source_y_t.unsqueeze(0)], dim=0)
                transformed = transforms(input_tensor)
                (_x, _y) = (transformed[0], transformed[1])
                if _y.sum() > 0:
                    break
            noise = 0.02 * torch.randn(128, 128, device=device, dtype=torch.float32)
            _x = _x + noise
            result_x.append(_x)
            result_y.append(_y)
        batch_x = torch.stack(result_x).unsqueeze(1)
        batch_y = torch.stack(result_y).unsqueeze(1)
        return (batch_x, batch_y)  # Randomly choose one of the preloaded source images.  # Keep applying transforms until a valid label (non-zero sum) is produced.  # Concatenate image and label along a new batch dimension.  # Shape: [1, H, W]  # Combined shape: [2, H, W]  # Apply transforms (ensure transforms support GPU tensors).  # Add noise directly on the GPU.  # Stack and add an extra channel dimension.

    return (random_labeled_img,)


@app.cell
def _(plt, random_labeled_img):
    (_x, _y) = random_labeled_img(1)
    plt.imshow(_x.cpu()[0][0])
    plt.colorbar()
    plt.show()
    plt.imshow(_y.cpu()[0][0])
    plt.colorbar()
    plt.show()
    return


@app.cell
def _(torch):
    def total_variation_loss_l1(image):
        """L1 version of TV loss - often less aggressive"""
        batch_size, channels, height, width = image.size()
    
        tv_h = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]).sum()
        tv_w = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]).sum()
    
        return (tv_h + tv_w) / (batch_size * channels * height * width)

    return (total_variation_loss_l1,)


@app.cell
def _(
    Rasterizer,
    plt,
    random_labeled_img,
    torch,
    total_variation_loss_l1,
    trange,
):
    N_iter = 300
    tv_weight = 1
    model = Rasterizer(32).to('cuda')
    criterion = torch.nn.BCELoss()
    opti = torch.optim.AdamW(model.parameters(), lr=0.005)
    losses = []
    for step in trange(N_iter):
        (_x, _y) = random_labeled_img(300)
        opti.zero_grad()  # Generate synthetic data
        y_pred = model.forward(_x)
        pred_loss = criterion(y_pred, _y)
        tv_loss = total_variation_loss_l1(y_pred)  # Training step
        loss = pred_loss + tv_weight * tv_loss
        loss.backward()
        opti.step()
        if step % 30 == 0:
            (fig, axes) = plt.subplots(1, 3, figsize=(15, 5))
            im1 = axes[0].imshow(_y[0][0].detach().cpu(), cmap='gray')
            axes[0].set_title('Original (Binary)')
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)  # Display triplets every 20 steps
            im2 = axes[1].imshow(_x[0][0].detach().cpu(), cmap='gray')
            axes[1].set_title('Noisy')
            axes[1].axis('off')  # Original (binary)
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)  # Original (binary)
            im3 = axes[2].imshow(y_pred[0][0].detach().cpu(), cmap='gray')
            axes[2].set_title('Reconstructed')
            axes[2].axis('off')
            plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()  # Noisy
        losses.append(loss.detach().cpu())  # Reconstructed
    return losses, model


@app.cell
def _(losses, plt):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(losses)), losses, label="reconstruction loss")
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return


@app.cell
def _(ROOT_DIR, load_normalized, plt):
    test = load_normalized(ROOT_DIR / "data/sketches/house.png")
    plt.imshow(test, cmap="binary")
    plt.show()
    return (test,)


@app.cell
def _(model, plt, test, torch):
    pred = model.forward(torch.tensor(test).cuda().unsqueeze(0))
    plt.imshow(pred.detach().cpu()[0])
    plt.colorbar()
    plt.show()
    return


@app.cell
def _(ROOT_DIR, model, torch):
    dummy_input = torch.rand((1, 256, 256), dtype=torch.float32)
    torch.onnx.export(
        model.cpu(),
        (dummy_input,),
        ROOT_DIR / "src" / "bez" / "cnn" / "model.onnx",
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        dynamic_axes={
            "input": {1: "height", 2: "width"},  # Variable H, W
            "output": {1: "height", 2: "width"},  # Variable H, W
        },
    )
    return


if __name__ == "__main__":
    app.run()
