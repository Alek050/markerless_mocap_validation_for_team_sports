import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from markerless_mocap_validation_for_team_sports.statistics.utils import \
    adjust_spines


def lin_reg_stats_with_plot(
    all_data: np.ndarray,
    ax: plt.Axes = None,
    colors: list[str] = None,
) -> plt.Axes:
    """Function to calculate the linear regression statistics and plot the results.

    Args:
        all_data (np.ndarray): The input array with 3 columns: Vicon data, Theia data,
            and participant number.
        ax (plt.Axes, optional): The ax to plot the regression on. Defaults to None.
        colors (list[str], optional): The colors for the different participants.
            Defaults to None.

    Raises:
        ValueError: If the input array does not have 3 columns.

    Returns:
        plt.Axes: The axes with the linear regression plot.
    """
    if not all_data.shape[1] == 3:
        raise ValueError(
            "The input array should have 3 columns: Vicon data, Theia data, "
            "and participant number"
        )

    vicon_data = all_data[:, 0]
    theia_data = all_data[:, 1]
    participants = all_data[:, 2]

    rmsd = np.sqrt(np.mean((vicon_data - theia_data) ** 2))

    reg = linregress(vicon_data, theia_data)
    stee = np.sqrt((1 / (reg.rvalue**2)) - 1)

    if ax is None:
        _, ax = plt.subplots()

    adjust_spines(ax, ["left", "bottom"])

    if colors is None:
        seed = 42
        np.random.seed(seed)
        colors = list(mcolors.CSS4_COLORS.keys())
        colors = np.random.permutation(colors)

    for i, participant in enumerate(np.sort(np.unique(participants))):
        idxs = np.where(participants == participant)[0]
        ax.scatter(
            vicon_data[idxs], theia_data[idxs], color=colors[i], s=0.5, alpha=0.5
        )

    low = np.min([np.percentile(vicon_data, 1), np.percentile(theia_data, 1)])
    high = np.max([np.percentile(vicon_data, 99), np.percentile(theia_data, 99)])
    x = np.linspace(low, high, 100)
    ax.plot(x, x, color="black", linestyle="--", label="perfect agreement")
    ax.plot(x, reg.slope * x + reg.intercept, color="red", label="Regression line")

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)

    ax.set_xlabel("Vicon data")
    ax.set_ylabel("Theia data")
    ax.set_title("Linear regression")
    ax.legend(loc="lower right", frameon=True)

    x_start = ax.get_xlim()[0] + 0.05 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    y_start = ax.get_ylim()[1] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.text(
        x_start,
        y_start,
        f"y = {reg.slope:.2f}x + {reg.intercept:.2f}\nR = "
        f"{reg.rvalue:.2f}\nRMSD = {rmsd:.2f}\nsTEE = {stee:.2f}",
        horizontalalignment="left",
        verticalalignment="top",
    )

    return ax
