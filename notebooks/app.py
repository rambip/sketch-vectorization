import marimo

__generated_with = "0.20.1"
app = marimo.App(width="columns")


@app.cell
def _():
    from sketchy_svg import load_normalized, sketch2svg, Demo
    import marimo as mo
    import matplotlib.pyplot as plt

    return Demo, load_normalized, mo, plt, sketch2svg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Skecth to SVG converter

    Draw something on paper, take a photograph, upload it here, and get a SVG !
    """)
    return


@app.cell
def _(mo):
    browser = mo.ui.file(label="Upload your image here (png, jpeg ...)")
    browser
    return (browser,)


@app.cell
def _(browser, load_normalized, plt):
    img = load_normalized(browser.value[0].contents)
    plt.imshow(img, cmap="binary")
    plt.title("Your image:")
    plt.axis(False)
    plt.gca()
    return (img,)


@app.cell
async def _(img, mo, sketch2svg):
    result = await sketch2svg(img, status_function=lambda x, title: mo.status.progress_bar(x, title=title))
    mo.Html(result)
    return


@app.cell
def _(mo):
    button = mo.ui.run_button(label="show each step")
    button
    return (button,)


@app.cell
async def _(Demo, browser, button, mo):
    mo.stop(not button.value)
    demo = Demo(status_function=lambda x, title: mo.status.progress_bar(x, title=title))
    await demo.show_example(browser.value[0].contents)
    return


if __name__ == "__main__":
    app.run()
