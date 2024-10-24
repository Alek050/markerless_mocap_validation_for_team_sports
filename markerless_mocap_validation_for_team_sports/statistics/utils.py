from matplotlib.axes import Axes


def adjust_spines(ax: Axes, visible_spines: list[str]) -> None:
    """Simple function to adjust the splines of a plot.

    Args:
        ax (Axes): The axes to adjust the splines of.
        visible_spines (list[str]): The splines to make visible.
    """
    ax.grid(color="0.9", zorder=-10)
    ax.set_axisbelow(True)

    for loc, spine in ax.spines.items():
        if loc in visible_spines:
            spine.set_position(("outward", 10))
        else:
            spine.set_visible(False)
