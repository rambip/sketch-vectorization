import marimo

__generated_with = "0.19.10"
app = marimo.App()


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import trange
    from bez.cnn.model import SketchDenoiser
    from bez.data import load_normalized
    import torch
    from meta import ROOT_DIR
    from torchvision.transforms import v2
    import random
    from pathlib import Path

    return Path, ROOT_DIR, SketchDenoiser, load_normalized, plt, torch


@app.cell
def _(Path, load_normalized, torch):
    N_IMAGES = 3000


    class SVGDataset:
        def __init__(self, path: Path, std_noise=0.1):
            self.std_noise = std_noise
            self.x_images = [
                load_normalized(path / f"x_{i}.png", only_alpha=True)
                for i in range(N_IMAGES)
            ]
            self.y_images = [
                load_normalized(path / f"y_{i}.png", only_alpha=True)
                for i in range(N_IMAGES)
            ]

        def __len__(self):
            return N_IMAGES

        def __getitem__(self, idx):
            x = self.x_images[idx]
            y = self.y_images[idx]
            return (
                torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                + self.std_noise *
                torch.randn((1, 200, 200)),
                torch.tensor(y, dtype=torch.float32).unsqueeze(0),
            )

    return (SVGDataset,)


@app.cell
def _(ROOT_DIR, SVGDataset):
    dataset = SVGDataset(ROOT_DIR / "data" / "train", std_noise=0.05)
    return (dataset,)


@app.cell
def _(dataset, plt):
    _img_x = dataset[0][0][0].numpy()
    _img_y = dataset[0][1][0].numpy()
    # stack horizontally
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(_img_x, cmap="binary")
    axs[0].set_title("Input Image")
    axs[0].axis("off")
    axs[1].imshow(_img_y, cmap="binary")
    axs[1].set_title("Ground truth")
    axs[1].axis("off")
    plt.show()
    return


@app.cell
def _(dataset, torch):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)
    return (dataloader,)


@app.cell
def _(torch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return (device,)


@app.cell
def _(SketchDenoiser, dataloader, device, torch):
    N_EPOCH = 10
    model = SketchDenoiser(16).to(device)
    criterion = torch.nn.BCELoss()
    opti = torch.optim.AdamW(model.parameters(), lr=0.005)
    losses = []
    for epoch in range(N_EPOCH):
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            opti.zero_grad()
            y_pred = model.forward(batch_x)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            opti.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{N_EPOCH}, Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "model.pt2")
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
def _(dataset, device, model, plt):
    _fig, _axs = plt.subplots(5, 3, figsize=(5, 12))
    for i in range(5):
        test = dataset[i][0]
        gold = dataset[i][1]
        predicted = model.forward(test.to(device).unsqueeze(0))
        _axs[i][0].imshow(test.detach().cpu().squeeze(0), cmap="binary")
        _axs[i][0].axis("off")
        _axs[i][1].imshow(predicted.detach().cpu().squeeze(0).squeeze(0), cmap="binary")
        _axs[i][1].axis("off")
        _axs[i][2].imshow(gold.detach().cpu().squeeze(0), cmap="binary")
        _axs[i][2].axis("off")
    plt.show()
    return


@app.cell
def _(ROOT_DIR, device, load_normalized, model, plt, torch):
    for name in ["butterfly", "dress", "piano"]:
        _test = load_normalized(ROOT_DIR / "data" / "sketches" / f"{name}.png")
        _predicted = model.forward(torch.tensor(_test).to(device).unsqueeze(0).unsqueeze(0))
        # stack horizontally
        _fig, _axs = plt.subplots(1, 2, figsize=(10, 5))
        _axs[0].imshow(_test, cmap="binary")
        _axs[0].set_title("Input Image")
        _axs[0].axis("off")
        _axs[1].imshow(_predicted.detach().cpu().squeeze(0).squeeze(0), cmap="binary")
        _axs[1].set_title("predicted")
        _axs[1].axis("off")
        plt.show()
    return


@app.cell
def _(model, torch):
    def export_onnx(out_path):
        dummy_input = torch.rand((1, 1, 256, 256), dtype=torch.float32)
        torch.onnx.export(
            model.cpu(),
            (dummy_input,),
            out_path,
            input_names=["input"],
            output_names=["output"],
            export_params=True,
            dynamic_axes={
                "input": {2: "height", 3: "width"},  # Variable H, W
                "output": {2: "height", 3: "width"},  # Variable H, W
            },
        )

    return (export_onnx,)


@app.cell
def _(ROOT_DIR, export_onnx):
    export_onnx(ROOT_DIR / "src" / "bez" / "cnn" / "model.onnx")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
