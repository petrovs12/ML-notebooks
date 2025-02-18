# ...existing code...
from typing import Tuple
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd


def f(y: float) -> None:
    x = np.arange(1, int(np.floor(y)) + 1)
    sns.lineplot(x=x, y=x**3)
    plt.xlabel("x")
    plt.ylabel("x^3")
    plt.title("Plot of x vs x^3")
    plt.show()


def generate_dataset(
    t_start: float, t_end: float, t_step: float
) -> Tuple[np.ndarray, np.ndarray]:
    t = np.arange(t_start, t_end, t_step)
    x = t
    y = np.sin(t)
    return x, y


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int):
    X = np.vander(x, degree + 1, increasing=True)
    model = sm.OLS(y, X).fit()
    return model


def run_simulation():
    train_x, train_y = generate_dataset(-10, 10, 0.1)
    test_x, test_y = generate_dataset(10, 15, 0.1)

    results = []

    for degree in range(11):
        model = fit_polynomial(train_x, train_y, degree)
        train_pred = model.predict(np.vander(
          train_x, degree + 1, increasing=True)
            )
        test_pred = model.predict(
            np.vander(test_x, degree + 1, increasing=True)
        )

        train_rmse = rmse(train_y, train_pred)
        test_rmse = rmse(test_y, test_pred)
        avg_abs_coeff = np.mean(np.abs(model.params))
        print(model.summary())
        results.append(
            {
                "degree": degree,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "avg_abs_coeff": avg_abs_coeff,
            }
        )

    results_df = pd.DataFrame(results)

    # Plot the fits for degree 2 and degree 8 polynomials
    degrees_to_plot = [2, 8]
    plt.figure(figsize=(10, 6))
    plt.scatter(train_x, train_y, label='Training Data', color='blue', alpha=0.5)

    x_plot = np.linspace(min(train_x), max(train_x), 200)
    for degree in degrees_to_plot:
        model = fit_polynomial(train_x, train_y, degree)
        y_plot = model.predict(np.vander(x_plot, degree + 1, increasing=True))
        plt.plot(x_plot, y_plot, label=f'Degree {degree} Polynomial Fit')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparison of Degree 2 and 8 Polynomial Fits')
    plt.legend()
    plt.show()

    return results_df


from snorkel.labeling import LFAnalysis