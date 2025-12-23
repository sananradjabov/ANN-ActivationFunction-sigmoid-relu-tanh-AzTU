import os
from typing import Callable, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

if not os.path.exists("./static"):
    os.makedirs("./static")


class ActivationItem(TypedDict):
    func: Callable[[NDArray], NDArray]
    deriv_func: Callable[[NDArray], NDArray]
    color: str


def single_plot(
    func: Callable[[NDArray], NDArray],
    show: bool = True,
    save: bool = True,
):
    x = np.linspace(-5, 5, 100)
    y = func(x)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, linewidth=3)

    ax.set_xlabel("x")
    ax.set_ylabel(f"{' '.join(func.__name__.split('_')).title()}(x)")
    ax.set_title(f"{' '.join(func.__name__.split('_')).title()} Function")

    ax.set_xlim(-5, 5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True)

    if save:
        pic_path: str = f"./static/img/{func.__name__}_function.png"

        fig.tight_layout()
        fig.savefig(pic_path, dpi=200)

        print(f"Save: {pic_path}")

    if show:
        plt.show()


def plot_all_activations_and_derivatives(
    activation_functions: list[ActivationItem],
    show: bool = True,
    save: bool = True,
    filename: str = "all_activations_and_derivatives.png",
):
    x = np.linspace(-5, 5, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.axvline(0, color="black", linewidth=0.5)
    ax1.set_title("Activation Functions")
    ax1.set_xlabel("Input value (x)")
    ax1.set_ylabel("Output value f(x)")
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Right
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axvline(0, color="black", linewidth=0.5)
    ax2.set_title("Derivatives (Gradients)")
    ax2.set_xlabel("Input value (x)")
    ax2.set_ylabel("The Gradient f'(x)")
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, linestyle="--", alpha=0.6)

    for item in activation_functions:
        func = item["func"]
        title = func.__name__
        deriv_func = item["deriv_func"]
        color = item["color"]

        ax1.plot(x, func(x), label=title, color=color, linewidth=2)
        ax2.plot(
            x, deriv_func(x), label=f"Derivative {title}", color=color, linewidth=2
        )

        if title == "sigmoid":
            ax2.axhline(
                0.25,
                color="gray",
                linestyle=":",
                linewidth=0.8,
                xmin=0.45,
                xmax=0.55,
                label="Max Sigmoid Grad (0.25)",
            )
        elif title == "tanh":
            ax2.axhline(
                1.0,
                color="gray",
                linestyle=":",
                linewidth=0.8,
                xmin=0.45,
                xmax=0.55,
                label="Max Tanh/ReLU Grad (1.0)",
            )

    ax1.legend()
    ax2.legend()

    if save:
        pic_path: str = f"./static/img/{filename}"

        fig.tight_layout()
        fig.savefig(pic_path, dpi=300)

        print(f"Save: {pic_path}")

    if show:
        plt.show()


if __name__ == "__main__":
    from activation_functions import relu, sigmoid, tanh

    single_plot(func=relu)
    single_plot(func=sigmoid)
    single_plot(func=tanh)
