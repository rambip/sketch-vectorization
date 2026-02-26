import marimo

__generated_with = "0.20.1"
app = marimo.App(width="columns")


@app.cell
async def _():
    from importlib.util import find_spec

    import marimo as mo
    import matplotlib.pyplot as plt

    if find_spec("js"):
        import micropip
        await micropip.install("sketchy-svg")

        from sketchy_svg import patch_onnx

        await patch_onnx()

    from sketchy_svg import demo, load_normalized, sketch2svg

    return demo, load_normalized, mo, plt, sketch2svg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Skecth to SVG converter

    Draw something on paper, take a photograph, upload it here, and get a SVG !
    """)
    return


@app.cell
def _(mo):
    picker = mo.ui.file(label="Upload your image here (png, jpeg ...)")
    picker
    return (picker,)


@app.cell
def _(load_normalized, mo, picker, plt):
    mo.stop(len(picker.value) == 0)
    img = load_normalized(picker.value[0].contents)
    plt.imshow(img, cmap="binary")
    plt.title("Your image:")
    plt.axis(False)
    plt.gca()
    return (img,)


@app.cell
async def _(img, mo, sketch2svg):
    result = await sketch2svg(
        img, status_function=lambda x, title: mo.status.progress_bar(x, title=title)
    )
    mo.Html(result)
    return


@app.cell
def _(demo, mo):
    demo.set_options(
        dict(
            status_function=lambda x, title: mo.status.progress_bar(x, title=title),
        ),
    )
    button = mo.ui.run_button(label="show each step")
    button
    return (button,)


@app.cell
async def _(button, demo, mo, picker):
    mo.stop(not button.value)
    await demo.show_example(picker.value[0].contents)
    return


if __name__ == "__main__":
    app.run()
