import marimo

__generated_with = "0.19.9"
app = marimo.App(width="columns")


@app.cell
def _():
    from io import BytesIO
    from os import environ

    import dotenv
    import marimo as mo
    import numpy as np
    import polars as pl
    from cairosvg import svg2png
    from perlin_numpy import generate_perlin_noise_2d
    from PIL import Image, ImageChops, ImageDraw

    from meta import ROOT_DIR

    return (
        BytesIO,
        Image,
        ImageChops,
        ImageDraw,
        ROOT_DIR,
        dotenv,
        environ,
        generate_perlin_noise_2d,
        mo,
        np,
        pl,
        svg2png,
    )


@app.cell
def _(dotenv, environ, mo, pl, ROOT_DIR):
    dotenv.load_dotenv()

    def collect_hg_dataset():
        df = pl.scan_parquet(
            "hf://datasets/OmniSVG/MMSVG-Icon/data/train-*-of-*.parquet",
            storage_options={"token": environ["HF_TOKEN"]},
        )
        df.select("svg", "keywords").head(1000).collect().write_csv(
            ROOT_DIR / "data" / "svg_dataset.csv"
        )

    mo.ui.run_button(
        label="download SVGs from huggingface", on_change=collect_hg_dataset
    )
    return


@app.cell
def _(pl):
    df = pl.read_csv("svg_dataset.csv")
    return (df,)


@app.cell
def _():
    # number of images
    N = 100
    return (N,)


@app.cell
def _(BytesIO, Image, N, ROOT_DIR, df, svg2png):
    import re

    svg_raw, keywords = df.sample(1).row()
    out_path = ROOT_DIR / "data" / "train"
    out_path.mkdir(exist_ok=True)

    def svg2image(raw_svg, stroke_width):
        svg_source = re.sub(
            r'fill="#?\w*"',
            f'stroke="black" stroke-width="{stroke_width}" fill="none"',
            raw_svg,
        )
        png_data = BytesIO(svg2png(svg_source))
        return Image.open(png_data)

    x_images = []
    y_images = []
    for i, svg_item in enumerate(df["svg"].sample(N)):
        x_images.append(svg2image(svg_item, 3))
        y_images.append(svg2image(svg_item, 5))
    x_images[0], y_images[0]
    return i, out_path, x_images, y_images


@app.cell
def _(mo, x_images):
    mo.inspect(x_images[0])
    return


@app.cell
def _(Image, ImageChops, ImageDraw, generate_perlin_noise_2d, np):
    def post_process(image):
        noise = generate_perlin_noise_2d((200, 200), (5, 5))
        noise += generate_perlin_noise_2d((200, 200), (20, 20))
        noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
        noise_img = Image.fromarray((noise * 255).astype(np.uint8), mode="L")

        # Extract alpha and multiply with noise
        r, g, b, a = image.split()
        result = ImageChops.multiply(a, noise_img)

        # Add random line
        draw = ImageDraw.Draw(result)
        x1, y1 = np.random.randint(0, 200, 2)
        x2, y2 = np.random.randint(0, 200, 2)
        draw.line([(x1, y1), (x2, y2)], fill=100, width=1)

        # Convert to RGBA: white RGB, result as alpha
        rgba = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        rgba.putalpha(result)

        return rgba

    return (post_process,)


@app.cell
def _(mo, post_process, x_images, y_images):
    mo.hstack([mo.vstack([x_images[i], post_process(y_images[i])]) for i in range(10)])
    return


@app.cell
def _(N, i, mo, out_path, post_process, x_images, y_images):
    for j in mo.status.progress_bar(range(N), title="Generating png images ..."):
        x = x_images[i]
        y = y_images[i]
        x.save(out_path / f"x_{j}.png")
        post_process(y).save(out_path / f"y_{j}.png")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
