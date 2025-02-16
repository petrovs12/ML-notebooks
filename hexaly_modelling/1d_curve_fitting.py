import sys
import numpy as np

import hexaly.optimizer as hx

"""
This script performs 1D curve fitting using the Hexal optimizer.

Modules:
    hexal.optimizer: Provides optimization functionalities from the Hexal library.
    sys: Provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.

Example:
    To use this script, you need to generate an instance of the optimizer and fit the curve with the following parameters:
    
    nb_observations: int
        The number of observations to generate for the curve fitting.
    inputs: array-like
        The input data for the curve fitting.
    outputs: array-like
        The output data for the curve fitting.

    Example usage:
        optimizer = hx.Optimizer()
        optimizer.fit(nb_observations, inputs, outputs)
"""

# Generate sample data
nb_observations = 100


inputs = np.linspace(0, 10, nb_observations)
outputs = (
    3 * np.sin(2 - inputs)
    + 0.5 * inputs**2
    + 1
    + np.random.normal(0, 0.5, nb_observations)
)

# Initialize the optimizer
with hx.HexalyOptimizer() as optimizer:
    # optimizer = hx.Optimizer()
    model = optimizer.model

    # Define the curve fitting function
    # def curve_fitting_model():
    a = model.float(-100, 100)
    b = model.float(-100, 100)
    c = model.float(-100, 100)
    d = model.float(-100, 100)

    predictions = [a * model.sin(b - x) + c * x**2 + d for x in inputs]
    errors = [pred - obs for pred, obs in zip(predictions, outputs)]
    square_error = sum(model.pow(err, 2) for err in errors)

    model.minimize(square_error)
    model.close()
    optimizer.stop()
    optimizer.solve()

    # Fit the model
    # optimizer.fit(curve_fitting_model)
    # Output the results
    print("Optimal parameters:")
    print(f"a = {optimizer.get_value('a')}")
    print(f"b = {optimizer.get_value('b')}")
    print(f"c = {optimizer.get_value('c')}")
    print(f"d = {optimizer.get_value('d')}")
