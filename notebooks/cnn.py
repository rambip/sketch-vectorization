import marimo

__generated_with = "0.20.1"
app = marimo.App()


@app.cell
def _():
    from sketchy_cnn.dataset import ensure_hf_dataset
    from pathlib import Path
    import polars as pl
    import matplotlib.pyplot as plt
    import copy

    ROOT_DIR = Path("sketch-vectorization")
    csv_path = ROOT_DIR / "data" / "svg_dataset.csv"
    csv_bytes = csv_path.read_bytes()
    return ROOT_DIR, copy, csv_bytes, pl, plt


@app.cell
def _():
    from sketchy_cnn.dataset import SVGDataset

    return (SVGDataset,)


@app.cell
def _():
    D_IMAGE=256
    STD_NOISE=0.02
    return D_IMAGE, STD_NOISE


@app.cell
def _(D_IMAGE, STD_NOISE, SVGDataset, csv_bytes, pl, plt):
    demo_df = pl.read_csv(csv_bytes)
    demo_dataset = SVGDataset(demo_df, n=10, d=D_IMAGE, std_noise=STD_NOISE)
    _fig, _axes = plt.subplots(3, 2, figsize=(8, 8))
    for _i in range(3):
        for j in range(2):
            _axes[_i, j].imshow(demo_dataset[_i][j][0], cmap='binary')
            _axes[_i, j].axis(False)
    _fig
    return demo_dataset, demo_df


@app.cell
def _(D_IMAGE, ROOT_DIR, STD_NOISE, SVGDataset, demo_df):
    import torch # Import torch for saving/loading

    # Define a path to save/load the dataset
    dataset_path = ROOT_DIR / "data" / "cached_dataset.pt" # Using ROOT_DIR defined earlier

    if dataset_path.exists():
        print(f"Loading dataset from {dataset_path}")
        dataset = torch.load(dataset_path)
    else:
        print("Creating dataset and saving to disk...")
        # Assuming demo_df, D_IMAGE, STD_NOISE are defined in previous cells
        dataset = SVGDataset(demo_df, n=2048, d=D_IMAGE, std_noise=STD_NOISE)
        dataset_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        torch.save(dataset, dataset_path)
        print(f"Dataset saved to {dataset_path}")
    return dataset, torch


@app.cell
def _():
    from sketchy_cnn import SketchDenoiser
    import torch_xla

    return (SketchDenoiser,)


@app.cell
def _(torch):
    def loss_function(pred, target, lambda_weight=0.3, epsilon=1e-2):
        # False negative term (mean over batch)
        fn_loss = -(target * torch.log(pred + epsilon)).mean()

        # Energy term (mean over batch)
        energy = pred.mean()

        return fn_loss + lambda_weight * energy

    return (loss_function,)


@app.cell
def _(SketchDenoiser, copy, loss_function, torch):
    import torch_xla.core.xla_model as xm
      # Explicitly import copy
    class Trainer:

        def __init__(self, d_model, lr, init_state_dict=None):
            self.device = xm.xla_device()  # Use xm.xla_device() for consistency
            self.model = SketchDenoiser(d_model).to(self.device)
            if init_state_dict is not None:
                self.model.load_state_dict({k: v.to(self.device) for k, v in init_state_dict.items()})
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
            self.losses = []
            self.checkpoints = []

        def train(self, dataset, batch_size, n_epoch):
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            print(f'start training on {self.device}')
            for epoch in range(n_epoch):
                epoch_loss = sum((self._step(batch_x, batch_y) for batch_x, batch_y in dataloader)) / len(dataloader)
                self.losses.append(epoch_loss)
                state_dict = self.model.state_dict()
                cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
                self.checkpoints.append(copy.deepcopy(cpu_state_dict))
                print(f'Epoch {epoch + 1}/{n_epoch}, Loss: {epoch_loss:.4f}')
                if len(self.losses) >= 3 and self.losses[-1] > self.losses[-2] > self.losses[-3]:
                    print('Loss increasing for 2 epochs, early stopping')
                    break
            return (self.checkpoints, self.losses)  # Get state_dict from the model which is on XLA device

        def _step(self, batch_x, batch_y):  # Move all tensors in state_dict to CPU for checkpointing if desired
            batch_x, batch_y = (batch_x.to(self.device), batch_y.to(self.device))
            self.optimizer.zero_grad()
            loss = loss_function(self.model(batch_x), batch_y)
            loss.backward()
            xm.optimizer_step(self.optimizer)
            xm.mark_step()
            return loss.item()

        def get_state(self):
            return (self.checkpoints, self.losses)  # Use xm.optimizer_step  # Add xm.mark_step

    return (Trainer,)


@app.cell
def _():
    N_EPOCH = 50
    BATCH_SIZE = 32
    LR = 0.005
    D_MODEL = 32
    return BATCH_SIZE, D_MODEL, LR, N_EPOCH


@app.cell
def _(BATCH_SIZE, D_MODEL, LR, N_EPOCH, Trainer, dataset):
    trainer = Trainer(D_MODEL, LR)
    state_dict, losses = trainer.train(dataset, BATCH_SIZE, N_EPOCH)
    return losses, state_dict


@app.cell
def _(losses, plt):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(losses)), losses, label="reconstruction loss")
    plt.title("Reconstruction loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.gca()
    return


@app.cell
def _(D_IMAGE, mo):
    import base64
    from io import BytesIO

    def figure(data, w=D_IMAGE, h=D_IMAGE, cmap="binary"):
        import matplotlib.pyplot as _plt

        fig = _plt.figure()
        ax = fig.gca()
        ax.imshow(data.reshape(h, w), cmap=cmap)
        ax.axis(False)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        _plt.close(fig)
        return mo.Html(f'<img src="data:image/png;base64,{img_b64}" />')

    def table(data):
        import polars as pl

        df = pl.DataFrame(data)
        parts = ["<table><thead><tr>"]
        for col in df.columns:
            parts.append(f'<th style="text-align:center">{col}</th>')
        parts.append("</tr></thead><tbody>")
        for row in df.iter_rows(named=False):
            parts.append("<tr>")
            for cell in row:
                parts.append(f"<td>{cell}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table>")
        return mo.Html("".join(parts))

    return


@app.cell
def _(D_MODEL, SketchDenoiser, demo_dataset, plt, state_dict, torch):
    import numpy as np
    model_viz = SketchDenoiser(D_MODEL)
    model_viz.load_state_dict({k: v.cpu() for k, v in state_dict[-1].items()})
    model_viz.eval().cpu()
    _samples = [demo_dataset[_i] for _i in range(6)]
    _xs = torch.stack([s[0] for s in _samples])
    _ys = torch.stack([s[1] for s in _samples])
    with torch.no_grad():
        _preds = model_viz.forward(_xs)
    num_samples = len(_samples)
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))
    titles = ['Input', 'Predicted', 'Ground Truth']
    for _i in range(num_samples):
        axes[_i, 0].imshow(_xs[_i].squeeze().cpu().numpy(), cmap='binary')
        axes[_i, 0].axis('off')
        if _i == 0:
            axes[_i, 0].set_title(titles[0])
        axes[_i, 1].imshow(_preds[_i].squeeze().cpu().numpy(), cmap='binary')
        axes[_i, 1].axis('off')
        if _i == 0:
            axes[_i, 1].set_title(titles[1])
        axes[_i, 2].imshow(_ys[_i].squeeze().cpu().numpy(), cmap='binary')
        axes[_i, 2].axis('off')
        if _i == 0:
            axes[_i, 2].set_title(titles[2])
    plt.tight_layout()
    plt.show()
    return (model_viz,)


@app.cell
def _(ROOT_DIR, model_viz, torch):
    # we remove the batch dimension for onnx (we will infer one image at a time)
    _dummy = torch.rand((1, 256, 256), dtype=torch.float32)
    _out_path = ROOT_DIR / "src" / "sketchy_svg" / "cnn" / "model.onnx"

    torch.onnx.export(
        model_viz,
        (_dummy,),
        _out_path,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        dynamic_axes={
            "input": {1: "height", 2: "width"},
            "output": {1: "height", 2: "width"},
        },
        dynamo=True,
    )
    print(f"Model exported to {_out_path}")
    return


if __name__ == "__main__":
    app.run()
