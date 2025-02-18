import marimo

__generated_with = "0.10.17"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    1

    return (mo,)


@app.cell
def _():
    import numpy as np

    return (np,)


if __name__ == "__main__":
    app.run()
