# sketch-vectorization

![](./images/illustration_pipeline.png)

## Goal

The goal of this project is to implement the following paper: <https://www-sop.inria.fr/reves/Basilic/2016/FLB16/fidelity_simplicity.pdf>

We also implemented several new ideas, such as a [convolutional neural network](./notebooks/cnn.ipynb) with synthetic data augmentation for preprocessing.

## Use

To install the dependencies, install [uv](https://github.com/astral-sh/uv) and run `uv sync`.

Then, you can run `uv run src/bez/app.py some_image.png`. This will create the png in the current directory.

## Documentation

You can take a look at:
- the [Walkthrough](./documentation/walkthrough.ipynb)
- the [Presentation](./documentaion/report.pdf)
