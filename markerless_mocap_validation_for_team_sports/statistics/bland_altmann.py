import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro

from markerless_mocap_validation_for_team_sports.statistics.utils import \
    adjust_spines


def bland_altmann_statistics(
    data1: np.ndarray,
    data2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """function to calculate the Bland-Altman statistics. If the data is normally
    distrubuted: the bias is the mean difference, and the upper limit of agreement
    (1.96*std + bias), and the lower limit of agreement (bias - 1.96*std).
    If the data is not normally distributed: the bias is the median difference, and the
    upper limit of agreement (97.5 percentile), and the lower limit of agreement (2.5
    percentile).

    Args:
        data1 (np.ndarray): The fist data array
        data2 (np.ndarray): The second data array

    Raises:
        ValueError: If the input arrays are not the same shape.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The bias, upper limit of agreement,
        and lower limit of agreement for every axis.
    """

    if (
        data1.ndim == 2
        and data1.shape[1] == 4
        and np.all((data1[:, 3] == 1.0) | (np.isnan(data1[:, 3])))
    ):
        data1 = data1[:, :-1]
    if (
        data2.ndim == 2
        and data2.shape[1] == 4
        and np.all((data2[:, 3] == 1.0) | (np.isnan(data2[:, 3])))
    ):
        data2 = data2[:, :-1]

    if data1.shape != data2.shape:
        raise ValueError("The input arrays should have the same shape")

    if data1.ndim == 1:
        data1 = data1[:, np.newaxis]
        data2 = data2[:, np.newaxis]

    diffs = data1 - data2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if shapiro(diffs.flatten())[1] < 0.05:
            bias = np.nanmedian(diffs, axis=0)
            lloa, uloa = np.percentile(diffs, [2.5, 97.5], axis=0)
            return bias[0], uloa[0], lloa[0]

        else:
            bias = np.nanmean(diffs, axis=0)
            std = np.nanstd(diffs, axis=0)
            uloa = 1.96 * std + bias
            lloa = bias - 1.96 * std
            return bias[0], uloa[0], lloa[0]


def bland_altmann_statistics_with_plot(
    all_data: np.ndarray,
    ax: plt.Axes | None = None,
    colors: list[str] | None = None,
) -> plt.Axes:
    """Function to calculate the Bland-Altman statistics and plot the results.

    Args:
        all_data (np.ndarray): The data array containing the Vicon, Theia data, and
            participant number data.
        ax (plt.Axes | None, optional): The axes to plot the data on. Defaults to None.
        colors (list[str] | None, optional): The colors to use for every participant.
            Defaults to None.

    Returns:
        plt.Axes: The axes with the Bland-Altman plot.
    """

    if not all_data.shape[1] == 3:
        raise ValueError(
            "The input array should have 3 columns: Vicon data, Theia data, "
            "and participant number"
        )

    vicon_data = all_data[:, 0]
    theia_data = all_data[:, 1]
    participants = all_data[:, 2]

    bias, uloa, lloa = bland_altmann_statistics(vicon_data, theia_data)
    means = (vicon_data + theia_data) / 2
    conf_interval = (uloa - lloa) / 2
    uloa = bias + conf_interval
    lloa = bias - conf_interval

    if colors is None:
        seed = 42
        np.random.seed(seed)
        colors = list(mcolors.CSS4_COLORS.keys())
        colors = np.random.permutation(colors)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 5))
    adjust_spines(ax, ["left", "bottom"])
    for i, participant in enumerate(np.sort(np.unique(participants))):
        idxs = np.where(participants == participant)[0]
        difference = vicon_data[idxs] - theia_data[idxs]
        ax.scatter(
            means[idxs],
            difference,
            s=0.5,
            alpha=0.5,
            label=participant,
            zorder=2,
            color=colors[i],
        )

    bias_line = ax.axhline(bias, color="black")
    ax.axhline(uloa, color="grey")
    ax.axhline(lloa, color="grey")

    ci_95 = ax.fill_between(
        np.linspace(-360, 360, 100),
        lloa,
        uloa,
        color="green",
        alpha=0.1,
        zorder=-2,
        interpolate=True,
    )
    tot_x = np.max(means) - np.min(means)
    ax.set_xlim(
        np.percentile(means, 1) - 0.2 * tot_x, np.percentile(means, 99) + 0.2 * tot_x
    )
    tot_y = np.abs(uloa - lloa)
    ax.set_ylim(lloa - 0.2 * tot_y, uloa + 0.2 * tot_y)
    x_loc = ax.get_xlim()[1] - 0.01
    ax.text(
        x_loc,
        uloa,
        f"ULOA: {uloa:.2f}",
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    ax.text(
        x_loc,
        lloa,
        f"LLOA: {lloa:.2f}",
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    ax.text(
        x_loc,
        bias,
        f"Bias: {bias:.2f}",
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    ax.set_xlabel("Mean Angle (degrees)")
    ax.set_ylabel("Difference (degrees)")
    ax.set_title("Bland-Altman plot")
    ax.legend(
        [bias_line, ci_95],
        ["Bias", "95% Confidence Interval"],
        loc="lower right",
        frameon=True,
    )

    return ax
