import marimo

__generated_with = "0.19.10"
app = marimo.App(auto_download=["ipynb"])


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
    import marimo as mo
    import polars as pl

    return Path, ROOT_DIR, SketchDenoiser, load_normalized, mo, pl, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load generated images

    See svg_dataset.py for information about how we generated the images
    """)
    return


@app.cell
def _(Path, load_normalized, torch):
    N_IMAGES = 3000
    D_IMAGE = 256


    class SVGDataset:
        def __init__(self, path: Path, std_noise=0.1):
            self.std_noise = std_noise
            self.x_images = [
                load_normalized(path / f"x_{i}.png", only_alpha=True, d=256)
                for i in range(N_IMAGES)
            ]
            self.y_images = [
                load_normalized(path / f"y_{i}.png", only_alpha=True, d=256)
                for i in range(N_IMAGES)
            ]

        def __len__(self):
            return N_IMAGES

        def __getitem__(self, idx):
            x = self.x_images[idx]
            y = self.y_images[idx]
            noise = self.std_noise * torch.randn((1, D_IMAGE, D_IMAGE))
            return (
                torch.tensor(x, dtype=torch.float32).unsqueeze(0) + noise,
                # very important for numerical stability (BCE loss)
                0.99 * torch.tensor(y, dtype=torch.float32).unsqueeze(0),
            )

    return D_IMAGE, SVGDataset


@app.cell
def _(ROOT_DIR, SVGDataset):
    dataset = SVGDataset(ROOT_DIR / "data" / "train", std_noise=0.05)
    return (dataset,)


@app.cell
def _(D_IMAGE, plt):
    def figure(data, w=D_IMAGE, h=D_IMAGE, cmap="binary"):
    
        ax = plt.figure().gca()
        ax.imshow(data.reshape(h, w), cmap=cmap)
        ax.axis(False)
        return ax

    return (figure,)


@app.cell
def _(mo, pl):
    def table(data):
        df = pl.DataFrame(data)
        # Build the table HTML
        html_parts = ['<table>']
    
        # Add table header with centered text
        html_parts.append('<thead>')
        html_parts.append('<tr>')
        for col in df.columns:
            html_parts.append(f'<th style="text-align: center;">{mo.as_html(col).text}</th>')
        html_parts.append('</tr>')
        html_parts.append('</thead>')
    
        # Add table body
        html_parts.append('<tbody>')
        for row in df.iter_rows(named=False):
            html_parts.append('<tr>')
            for cell_value in row:
                html_parts.append(f'<td>{mo.as_html(cell_value).text}</td>')
            html_parts.append('</tr>')
        html_parts.append('</tbody>')
    
        html_parts.append('</table>')
    
        return mo.Html(''.join(html_parts))

    return (table,)


@app.cell
def _(dataset, figure, table):
    _x, _y = dataset[0]
    table([{"Input": figure(_x), "Ground Truth": figure(_y)}])
    return


@app.cell
def _(dataset, torch):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)
    return (dataloader,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model training

    This model is a 7-layer CNN

    Each convolution kernel is of size 3x3 with 32 channels.
    """)
    return


@app.cell
def _(torch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return (device,)


@app.cell
def _(SketchDenoiser, dataloader, device, torch):
    N_EPOCH = 30
    model = SketchDenoiser(32).to(device)
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
    plt.title("Reconstruction loss")
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.gca()
    return


@app.cell
def _(model):
    # we convert
    model.eval()
    model_cpu = model.cpu()
    return (model_cpu,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visualisation of model outputs
    """)
    return


@app.cell
def _(dataloader, figure, model_cpu, table):
    _sample_x, _sample_y = next(iter(dataloader))
    _sample_x = _sample_x[:6]
    _sample_y = _sample_y[:6]
    table(
        {
            "Input": [figure(x) for x in _sample_x],
            "Predicted": [
                figure(p.detach()) for p in model_cpu.forward(_sample_x)
            ],
            "Ground truth": [figure(y) for y in _sample_y],
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On the test images (not synthetic):
    """)
    return


@app.cell
def _(ROOT_DIR, dataloader, figure, load_normalized, model_cpu, table, torch):
    _sample_x, _sample_y = next(iter(dataloader))
    test_x = [
        torch.tensor(
            load_normalized(ROOT_DIR / "data" / "sketches" / f"{name}.png", d=256)
        ).unsqueeze(0)
        for name in ["butterfly", "dress", "piano"]
    ]
    table(
        [
            {
                "Input": figure(x),
                "Predicted": figure(model_cpu.forward(x).detach()),
            }
            for x in test_x
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model export

    To use the model easily in the main pipeline, we convert it to [ONNX](https://onnx.ai/), a standard format for neural networks.
    """)
    return


@app.cell
def _(torch):
    def export_onnx(model, out_path):
        dummy_input = torch.rand((1, 256, 256), dtype=torch.float32)
        torch.onnx.export(
            model.cpu(),
            (dummy_input,),
            out_path,
            input_names=["input"],
            output_names=["output"],
            export_params=True,
            dynamic_axes={
                "input": {1: "height", 2: "width"},  # Variable H, W
                "output": {1: "height", 2: "width"},  # Variable H, W
            },
            dynamo=True
        )

    return (export_onnx,)


@app.cell
def _(ROOT_DIR, export_onnx, model_cpu):
    export_onnx(model_cpu, ROOT_DIR / "src" / "bez" / "cnn" / "model.onnx")
    return


if __name__ == "__main__":
    app.run()
