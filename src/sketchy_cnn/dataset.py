from io import BytesIO
from pathlib import Path

import numpy as np
import polars as pl
import torch
from cairosvg import svg2png
from perlin_numpy import generate_perlin_noise_2d
from PIL import Image, ImageChops
from torch.utils.data import Dataset


def collect_hf_dataset(path: Path, hf_token: str, n: int = 5000) -> None:
    """Download SVG dataset from HuggingFace and save as CSV.

    Requires the `hf_token` from HuggingFace with read access to OmniSVG/MMSVG-Icon.
    """
    df = pl.scan_parquet(
        "hf://datasets/OmniSVG/MMSVG-Icon/data/train-*-of-*.parquet",
        storage_options={"token": hf_token},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Started download...")
    df.select("svg", "keywords").head(n).collect().write_csv(path)
    print(f"Dataset saved at {path}")


def ensure_hf_dataset(path: Path, hf_token: str, n: int = 5000) -> None:
    """Download the dataset only if the CSV does not already exist."""
    if not path.exists():
        collect_hf_dataset(path, hf_token, n)


def svg2image(source: str) -> Image.Image:
    png_data = BytesIO(svg2png(source))
    return Image.open(png_data)


def to_binary(image: Image.Image, d: int) -> Image.Image:
    rgba = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    rgba.putalpha(Image.fromarray(np.asarray(image).sum(-1) > 1))
    return rgba


def post_process(image: Image.Image, d: int) -> Image.Image:
    noise = 1.0 * generate_perlin_noise_2d((d, d), (2, 2))
    noise += 1.0 * generate_perlin_noise_2d((d, d), (8, 8))
    noise += 1.0 * generate_perlin_noise_2d((d, d), (32, 32))
    noise = np.exp(noise)
    noise = noise / np.max(noise)
    noise_img = Image.fromarray((noise * 255).astype(np.uint8), mode="L")

    r, g, b, a = image.split()
    result = ImageChops.multiply(a, noise_img)

    # macro lighting: low-frequency Perlin added to alpha after detail noise
    lighting = generate_perlin_noise_2d((d, d), (1, 1))
    lighting = (lighting - lighting.min()) / (lighting.max() - lighting.min())  # [0, 1]
    lighting_img = Image.fromarray(
        (lighting * 200).astype(np.uint8), mode="L"
    )  # 200: tunable intensity
    result = ImageChops.add(result, lighting_img)

    rgba = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    rgba.putalpha(result)
    return rgba


def build_synthetic_svg(df: pl.DataFrame, n: int, d: int) -> list[str]:
    paths = (
        df.sample(n)
        .select(path=pl.col("svg").str.extract_all(r"<path .*></path>\s?"))
        .with_row_index("svg_idx")
        .explode("path")
    )
    synthetic_svg = (
        paths.with_columns(width=0.3 * np.exp(3 * np.random.random(size=len(paths))))
        .group_by("svg_idx")
        .agg(
            pl.col("path").str.replace(
                r'fill="#?\w*" *fill-opacity="1.0" *filling="0',
                pl.format(
                    'stroke="black" stroke-width="{}" fill="none',
                    pl.col("width"),
                ),
            )
        )
        .select(
            svg=pl.format(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="-11.0 -11.0 222.0 222.0" height="{}.0px" width="{}.0px">{}</svg>',
                d,
                d,
                pl.col("path").list.join(""),
            )
        )["svg"]
    )
    return synthetic_svg.to_list()


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.getchannel("A")).astype(np.float32) / 255.0
    return torch.tensor(arr).unsqueeze(0)


class SVGDataset(Dataset):
    def __init__(self, df: pl.DataFrame, n: int, d: int = 256, std_noise: float = 0.05):
        svg_strings = build_synthetic_svg(df, n, d)
        self.svg_strings = svg_strings
        self.x_images = [
            image_to_tensor(post_process(svg2image(svg), d)) for svg in svg_strings
        ]
        self.y_images = [
            image_to_tensor(to_binary(svg2image(svg), d)) for svg in svg_strings
        ]
        self.std_noise = std_noise

    def __len__(self) -> int:
        return len(self.x_images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.y_images[idx]
        x = self.x_images[idx]
        noise = self.std_noise * torch.randn_like(x)
        # 0.99 factor: important for numerical stability with BCE loss
        return x + noise, 0.99 * y
