import marimo

__generated_with = "0.12.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    import jax.nn as nn
    import jax.random as jr
    import jax.nn as nn
    import jax.random as jr
    import seaborn as sns
    import numpy as np

    return jax, jnp, jr, mo, nn, np, sns


@app.cell
def _(mo):
    x=mo.ui.slider(0,100,1)
    x


    return (x,)


@app.cell
def _(x):
    x.value
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # discrete operators and their continous analogs, as used in neural networks
    
        # discrete operators
        # 1. convolution
        # 2. pooling
        # 3. upsampling
        # 4. downsampling
        # 5.
        """
    )
    return


@app.cell
def _(mo):
    x1=mo.ui.slider(0,100,1)
    x2=mo.ui.slider(0,100,1)
    x3=mo.ui.slider(0,100,1)
    x4=mo.ui.slider(0,100,1)
    x5=mo.ui.slider(0,100,1)
    temperature=mo.ui.slider(1,100,1)

    x1,x2,x3,x4,x5, temperature






    return temperature, x1, x2, x3, x4, x5


@app.cell
def _(np, temperature, x1, x2, x3, x4, x5):
    # max equivalent
    def soft_max(x, temperature):
        x=np.array(x)
        return np.exp(x/temperature) / np.sum(np.exp(x/temperature))
    soft_max([x1.value,x2.value,x3.value,x4.value,x5.value], temperature.value)


    return (soft_max,)


@app.cell
def _(mo):
    mo.md(r"# multilabel classification")
    return


@app.cell
def _(mo):
    mo.md(r"Smooth max (LogSumExp): $\displaystyle \frac{1}{\lambda}\log\sum_i e^{\lambda x_i}$Â  Softmax-weighted sum: $\displaystyle \sum_i \text{softmax}_\tau(x)_i , x_i$")
    return


@app.cell
def _(np, soft_max, temperature, x1, x2, x3, x4, x5):
    def smoothmax(x, lbda):
        return np.log(np.sum(np.exp(lbda*x))) / lbda

    def softmax_weighted_sum(x, lbda):
        return np.sum(soft_max(x, lbda)*x)

    smoothmax([x1.value,x2.value,x3.value,x4.value,x5.value], temperature.value)
    return smoothmax, softmax_weighted_sum


@app.cell
def _(softmax_weighted_sum, temperature, x1, x2, x3, x4, x5):
    softmax_weighted_sum([x1.value,x2.value,x3.value,x4.value,x5.value], temperature.value)
    return


@app.cell
def _(np, sns, temperature):
    def sigmoid(x, temperature):
        return 1 / (1 + np.exp(-x/temperature))

    sns.lineplot(x=np.linspace(-10,10,100), y=sigmoid(np.linspace(-10,10,100), temperature.value))
    # 1/(1-exp(x))


    return (sigmoid,)


@app.cell
def _():
    # let's compute some symbolic derivatives
    import sympy as sp
    xs=sp.symbols('x')

    f=1./(1+sp.exp(-xs))
    sp.diff(f,xs)

    # 

    return f, sp, xs


@app.cell
def _(mo):
    mo.md(
        r"""
        Gating / if-else branch: e.g. $y = A$ if $c$ else $B$ (choose between two pathways)
    
        Soft gate: $y = \alpha \cdot A + (1-\alpha)\cdot B$ with $\alpha=\sigma(s)$ for some learned score $s$Â  Attention weights: $y = \alpha \cdot A + \beta \cdot B$, $\alpha+\beta=1$
    
    
        Rather than a hard branch, the network learns a gate value $\alpha\in[0,1]$ (via a sigmoid) to blend the two outcomes . If $\alpha$ saturates near 0 or 1, it behaves like a discrete switch, but during training it can take intermediate values so gradients flow to both $A$ and $B$ paths. Attention mechanisms generalize this idea by gating a weighted combination of many options rather than just two.
        """
    )
    return


@app.cell
def _(np, sigmoid, temperature, x1, x2, x3, x4, x5):
    # soft-gating
    def soft_gate(a,b, x, temperature):
        alpha=sigmoid(x, temperature)
        return alpha*a + (1-alpha)*b

    soft_gate(np.array([1,2,3,4,5]), np.array([6,7,8,9,10]), np.array([x1.value,x2.value,x3.value,x4.value,x5.value]), temperature.value)




    return (soft_gate,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ```mermaid
        graph LR;
        rea
    
        ```
        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
