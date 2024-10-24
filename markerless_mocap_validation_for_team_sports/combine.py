import warnings

import kineticstoolkit as ktk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import SpanSelector
from scipy import signal

from markerless_mocap_validation_for_team_sports.constants import \
    KNOWN_TIMESERIES
from markerless_mocap_validation_for_team_sports.logging import create_logger

LOGGER = create_logger(__name__)


def get_combined_time_series(
    vicon: dict[str, ktk.TimeSeries],
    theia: dict[str, ktk.TimeSeries],
    exercise: str,
    start_time: float = np.nan,
    end_time: float = np.nan,
    special_side: str | None = None,
) -> tuple[dict[str, ktk.TimeSeries], dict[str, ktk.TimeSeries], dict[str, float]]:
    """Function to combine different TimeSeries from Vicon and Theia. This function
    processes the following steps:

    1. Align the timeseries using crosscorrelation. For this the knee flexion angles
        are used.
    2. Manually select the time of interest, if start and end time are np.nan.
    3. Crop the TimeSeries to the selected time of interest.
    4. Optimally unwrap the angles.

    Args:
        vicon (dict[str, ktk.TimeSeries]): A dictionary of TimeSeries from Vicon.
        theia (dict[str, ktk.TimeSeries]): A dictionary of TimeSeries from Theia.
        exercise (str): The exercise that is being analyzed.
        start_time (float, optional): The start time of interest. Defaults to np.nan.
            if np.nan, the time of interest is selected manually, else the value is
            used.
        end_time (float, optional): The end time of interest. Defaults to np.nan.
            if np.nan, the time of interest is selected manually, else the value is
            used.
        special_side (str|None, optional): The special side for the exercise. For
            instance the side of the hand that is used for the high five. Defaults
            to None. If None, the special side is selected manually. If not None, the
            value is used.

    Returns:
        tuple[dict[str, ktk.TimeSeries], dict[str, ktk.TimeSeries], dict]: A tuple of
        the cropped Vicon and Theia dictionaries with TimeSeries and dict with extra
        info.
    """

    for data in vicon, theia:
        if not isinstance(data, dict):
            raise TypeError("The input data should be a dictionary.")
        if not all(isinstance(value, ktk.TimeSeries) for value in data.values()):
            raise TypeError("The values of the input dictionary should be TimeSeries.")
        if not all(isinstance(key, str) for key in data.keys()):
            raise TypeError("The keys of the input dictionary should be strings.")
        if not all(key in KNOWN_TIMESERIES for key in data.keys()):
            raise ValueError(
                f"The keys of the input dictionary should be in {KNOWN_TIMESERIES}"
                f" but are {data.keys()}"
            )
        if not all(
            np.allclose(data[key].time, data["angles"].time) for key in data.keys()
        ):
            raise ValueError("The time of the different TimeSeries should be equal.")

    # Check if the time series are equal
    if not all(np.allclose(vicon[key].time, theia[key].time) for key in vicon.keys()):
        raise ValueError("The time of the vicon and theia TimeSeries should be equal.")

    # Step 1: align the timeseries using crosscorrelation
    vicon_lag = sync_timeseries(vicon, theia)[-1]

    # Step 2: manually select the time of interest
    if np.isnan(start_time) or np.isnan(end_time) or special_side is None:
        start_time, end_time, special_side = select_time_of_interest(
            vicon,
            theia,
            exercise,
        )
    LOGGER.info(
        f"Selected time of interest: {start_time} - {end_time}, "
        f"special side: {special_side}"
    )

    # Step 3: crop the TimeSeries to the selected time of interest
    shortened_vicon = {}
    shortened_theia = {}

    start_time, end_time = round(start_time, 4), round(end_time, 4)
    for data_type in KNOWN_TIMESERIES:
        shortened_vicon[data_type] = vicon[data_type].get_ts_between_times(
            start_time, end_time, inclusive=[True, False]
        )
        shortened_theia[data_type] = theia[data_type].get_ts_between_times(
            start_time, end_time, inclusive=[True, False]
        )

    # Step 4: optimally unwrap the angles
    for data_type in ["angles", "rotations"]:
        for key in shortened_vicon[data_type].data.keys():
            for i in range(3):
                (
                    shortened_vicon[data_type].data[key][:, i],
                    shortened_theia[data_type].data[key][:, i],
                ) = cap_range(
                    shortened_vicon[data_type].data[key][:, i],
                    shortened_theia[data_type].data[key][:, i],
                )

    extra_info = {
        "vicon_lag": vicon_lag,
        "start_time": start_time,
        "end_time": end_time,
    }
    if any([x in exercise for x in ["shot", "pass", "five", "sidestep"]]):
        extra_info["special_side"] = special_side

    return shortened_vicon, shortened_theia, extra_info


def cap_range(vicon: np.ndarray, theia: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ensure angles are within -360 and 360 degrees, and
    the maximal difference between the two angles is less than 180 degrees."""

    vicon_unwrapped, theia_unwrapped = np.unwrap(vicon, period=360), np.unwrap(
        theia, period=360
    )

    for idx in range(len(vicon_unwrapped)):
        if theia_unwrapped[idx] - vicon_unwrapped[idx] > 180:
            theia_unwrapped[idx:] -= 360
        elif theia_unwrapped[idx] - vicon_unwrapped[idx] < -180:
            theia_unwrapped[idx:] += 360

    if np.mean(vicon_unwrapped) > 180:
        vicon_unwrapped -= 360
        theia_unwrapped -= 360
    if np.mean(vicon_unwrapped) < -180:
        vicon_unwrapped += 360
        theia_unwrapped += 360

    def adjust_angle_limits_between_abs_360(data1, data2):
        if data1.min() < -180 or data2.min() < -180:
            mask = (data1 < -180) & (data2 < -180)
            new_data1 = np.where(mask, data1 + 360, data1)
            new_data2 = np.where(mask, data2 + 360, data2)

        else:
            mask = (data1 > 180) & (data2 > 180)
            new_data1 = np.where(mask, data1 - 360, data1)
            new_data2 = np.where(mask, data2 - 360, data2)

        return new_data1, new_data2

    def adjust_angle_limits_between_abs_180(data1, data2):
        mask = (data1 > 180) & (data2 > 180)
        new_data1 = np.where(mask, data1 - 360, data1)
        new_data2 = np.where(mask, data2 - 360, data2)

        mask = (new_data1 < -180) & (new_data2 < -180)
        newer_data1 = np.where(mask, new_data1 + 360, new_data1)
        newer_data2 = np.where(mask, new_data2 + 360, new_data2)

        return newer_data1, newer_data2

    def adjust_angle_limits_between_0_360(data1, data2):
        mask = (data1 > 360) & (data2 > 360)
        new_data1 = np.where(mask, data1 - 360, data1)
        new_data2 = np.where(mask, data2 - 360, data2)

        mask = (new_data1 < 0) & (new_data2 < 0)
        newer_data1 = np.where(mask, new_data1 + 360, new_data1)
        newer_data2 = np.where(mask, new_data2 + 360, new_data2)

        return newer_data1, newer_data2

    def final_capping(data1, data2, max_val):
        mask = (data1 < max_val - 360) & (data1 + 180 < -data2 - 180)
        data1 = np.where(mask, data1 + 360, data1)
        data2 = np.where(mask, data2 + 360, data2)

        # low values for vicon
        mask = (data2 < max_val - 360) & (data2 + 180 < -data1 - 180)
        data2 = np.where(mask, data2 + 360, data2)
        data1 = np.where(mask, data1 + 360, data1)

        # high values for theia
        mask = (data1 > max_val) & (data1 - 180 > -data2 + 180)
        data1 = np.where(mask, data1 - 360, data1)
        data2 = np.where(mask, data2 - 360, data2)

        # high values for vicon
        mask = (data2 > max_val) & (data2 - 180 > -data1 + 180)
        data2 = np.where(mask, data2 - 360, data2)
        data1 = np.where(mask, data1 - 360, data1)

        return data1, data2

    vicon_capped = vicon_unwrapped
    theia_capped = theia_unwrapped
    counter = 0
    impossible = False
    while (
        np.abs(vicon_capped).max() > 360 or np.abs(theia_capped).max() > 360
    ) and counter < 5:
        theia_capped, vicon_capped = adjust_angle_limits_between_abs_360(
            theia_capped, vicon_capped
        )
        counter += 1
    if (
        vicon_capped.max() - vicon_capped.min() > 360
        or theia_capped.max() - theia_capped.min() > 360
    ):
        vicon_capped_180, theia_capped_180 = adjust_angle_limits_between_abs_180(
            vicon_capped, theia_capped
        )
        vicon_capped_360, theia_capped_360 = adjust_angle_limits_between_0_360(
            vicon_capped, theia_capped
        )
        if (
            vicon_capped_360.max() - vicon_capped_360.min()
            < vicon_capped_180.max() - vicon_capped_180.min()
        ):
            vicon_capped, theia_capped = final_capping(
                vicon_capped_360, theia_capped_360, 360
            )
        else:
            vicon_capped, theia_capped = final_capping(
                vicon_capped_180, theia_capped_180, 180
            )

    # Final validation
    if (
        np.max(np.abs(vicon_capped)) > 360 or np.max(np.abs(theia_capped)) > 360
    ) and not impossible:
        raise ValueError("Something went wrong with the unwrapping of the angles.")
    if (
        vicon_capped.max() - vicon_capped.min() > 360
        or theia_capped.max() - theia_capped.min() > 360
    ):
        print(
            "Vicon range:",
            vicon_capped.max() - vicon_capped.min(),
            "Theia range:",
            theia_capped.max() - theia_capped.min(),
        )
    if any(np.abs(vicon_capped - theia_capped) > 180):
        raise ValueError("Something went wrong with the unwrapping of the angles.")

    return vicon_capped, theia_capped


def sync_timeseries(
    vicon: dict[str, ktk.TimeSeries], theia: dict[str, ktk.TimeSeries]
) -> tuple[int, int, int, int, float]:
    """Function to sync the Vicon and Theia TimeSeries. This function
    processes the following steps:

    1. Get the start and end index of the longest period without NaN's.
    2. Get the optimal lag based on the knee flexion angles.

    Args:
        vicon (dict[str, ktk.TimeSeries]): The Vicon angles, rotations,
            and points TimeSeries.
        theia (dict[str, ktk.TimeSeries]): The Theia angles, rotations,
            and points TimeSeries.

    Returns:
        tuple[int, int, int, int, float]: The start and end index of the
            longest period without NaN's for Vicon and Theia, and the optimal
            lag in seconds.
    """

    vicon_start_idx, vicon_end_idx = get_start_end_without_nan(vicon)
    theia_start_idx, theia_end_idx = get_start_end_without_nan(theia)
    initial_lag = vicon_start_idx - theia_start_idx
    extra_lag = get_lag(
        vicon["angles"].get_ts_between_indexes(
            vicon_start_idx, vicon_end_idx, inclusive=[True, False]
        ),
        theia["angles"].get_ts_between_indexes(
            theia_start_idx, theia_end_idx, inclusive=[True, False]
        ),
        vicon["points"].get_ts_between_indexes(
            vicon_start_idx, vicon_end_idx, inclusive=[True, False]
        ),
        theia["points"].get_ts_between_indexes(
            theia_start_idx, theia_end_idx, inclusive=[True, False]
        ),
    )

    fs = 1 / np.mean(np.diff(vicon["angles"].time))
    for data_type in KNOWN_TIMESERIES:
        vicon[data_type].time -= (initial_lag + extra_lag) / fs

    return (
        vicon_start_idx,
        vicon_end_idx,
        theia_start_idx,
        theia_end_idx,
        (initial_lag + extra_lag) / fs,
    )


class CustomSpanSelector:
    def __init__(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        vicon: dict[str, ktk.TimeSeries],
        theia: dict[str, ktk.TimeSeries],
    ):
        self.fig = fig
        self.ax = ax
        self.vicon = vicon
        self.theia = theia
        self.t_min = None
        self.t_max = None
        self.span_selector = None
        self.please_close = False
        self.show_player = True

    def onselect(self, t_min, t_max):
        self.t_min = t_min
        self.t_max = t_max

    def allowed_to_close(self):
        if self.t_min is None or self.t_max is None:
            return False

        for data_type in KNOWN_TIMESERIES:
            for key in self.theia[data_type].data.keys():
                if (
                    self.theia[data_type]
                    .get_ts_between_times(self.t_min, self.t_max)
                    .isnan(key)
                    .any()
                    or self.vicon[data_type]
                    .get_ts_between_times(self.t_min, self.t_max)
                    .isnan(key)
                    .any()
                ):
                    return False
        return True

    def key_press(self, event):
        if event.key == "q":
            self.please_close = True
        if event.key == "enter":
            self.show_player = not self.show_player

    def setup_span_selector(self):
        self.span_selector = SpanSelector(
            ax=self.ax,
            onselect=self.onselect,
            direction="horizontal",
            useblit=True,
            props=dict(alpha=0.5, facecolor="tab:blue"),
            interactive=True,
            drag_from_anywhere=True,
        )

        self.fig.canvas.mpl_connect("key_press_event", self.key_press)


def get_start_end_without_nan(data: dict[str, ktk.TimeSeries]) -> tuple[int, int]:
    """Function to get the start and end index of the longest period without NaN's
    in all the TimeSerieses.

    Args:
        data (dict[str, ktk.TimeSeries]): The dict with TimeSerieses.

    Returns:
        tuple[int, int]: The start and end index of the longest period
            without NaN's.
    """
    nans = np.array([False for _ in range(len(data["angles"].time))])
    for data_type in data.keys():
        for key in data[data_type].data.keys():
            nans |= data[data_type].isnan(key)

    nans = np.concatenate((np.array([True]), nans.copy(), np.array([True])))
    isnan = np.where(nans)[0]
    argmax = np.argmax(np.diff(isnan))

    return isnan[argmax], isnan[argmax + 1] - 2


def get_lag(
    vicon_angles: ktk.TimeSeries,
    theia_angles: ktk.TimeSeries,
    vicon_points: ktk.TimeSeries,
    theia_points: ktk.TimeSeries,
    joints: list = ["knee"],
) -> int:
    """Funtion to get the lag of vicon relative to theia based on the joint angles. The
    optimal lag for each angle is calculated based on the cross correlation. The max of
    the mean of all cross correlations is used to determine the optimal lag.

    Note: it is assumed that the angels timeseries of both providers do not containt
        any NaN's

    Args:
        vicon_angles (ktk.TimeSeries): The vicon angles
        theia_angles (ktk.TimeSeries): The theia angles
        vicon_points (ktk.TimeSeries): The vicon points
        theia_points (ktk.TimeSeries): The theia points
        joints (list, optional): The joints to include in the cross correlation.
            Defaults to ["knee"].

    Returns:
        int: The optimal lag.
    """

    lags = signal.correlation_lags(
        vicon_angles.data["right_knee"].shape[0],
        theia_angles.data["right_knee"].shape[0],
        mode="full",
    )

    rmse_com_inv = np.array(
        [
            1
            / lagged_root_mean_square(
                vicon_points.data["centre_of_mass"][:, 0],
                theia_points.data["centre_of_mass"][:, 0],
            ),
            1
            / lagged_root_mean_square(
                vicon_points.data["centre_of_mass"][:, 1],
                theia_points.data["centre_of_mass"][:, 1],
            ),
        ]
    ).T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rmse_com_inv = np.nan_to_num(rmse_com_inv) / np.nanmax(
            np.abs(rmse_com_inv), axis=0
        )

    peak = np.argmax(np.nanmean(rmse_com_inv[30:-30], axis=1)) + 30
    start_idx, end_idx = max(peak - 20, 0), min(peak + 20, len(lags) - 1)
    lagged_corrcoefs = np.zeros((end_idx - start_idx, len(joints) * 2))
    i = 0
    for joint in joints:
        for side in ["left", "right"]:
            lagged_corrcoefs[:, i] = lagged_corrcoef(
                vicon_angles.data[f"{side}_{joint}"][:, 0],
                theia_angles.data[f"{side}_{joint}"][:, 0],
                lags[start_idx:end_idx],
            )
            i += 1
    return lags[start_idx] + np.argmax(np.mean(lagged_corrcoefs, axis=1))


def lagged_corrcoef(arr1: np.ndarray, arr2: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Function to calculate the correlation coefficient between two arrays with
    different lags. The function calculates the correlation coefficient for each lag
    between the two arrays.

    Args:
        arr1 (np.ndarray): The first timeseries
        arr2 (np.ndarray): The second timeseries
        lags (np.ndarray): The lags to calculate the correlation coefficient for.

    Returns:
        np.ndarray: The correlation coefficients for each lag.
    """

    all_corrcoefs = np.zeros(lags.size)
    all_corrcoefs[:] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, lag in enumerate(lags):
            if lag < 0:
                current_arr1 = arr1[: min(arr1.size, arr2.size + lag)]
                current_arr2 = arr2[-lag : min(-lag + arr1.size, arr2.size)]
            elif lag == 0:
                current_arr1 = arr1[: min(arr1.size, arr2.size)]
                current_arr2 = arr2[: min(arr1.size, arr2.size)]
            else:
                current_arr1 = arr1[lag : min(lag + arr2.size, arr1.size)]
                current_arr2 = arr2[: min(arr1.size - lag, arr2.size)]

            all_corrcoefs[i] = np.corrcoef(current_arr1, current_arr2)[0, 1]
    return all_corrcoefs


def lagged_root_mean_square(arr2: np.ndarray, arr1: np.ndarray) -> np.ndarray:
    """Function to calculate the root mean square of the difference between two arrays
    with different lags. The function calculates the root mean square for each lag
    between the two arrays.

    Args:
        arr1 (np.ndarray): The first timeseries
        arr2 (np.ndarray): The second timeseries

    Returns:
        np.ndarray: The root mean square for each lag.
    """

    all_rms = np.zeros(arr1.size + arr2.size - 1)
    all_rms[:] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, lag in enumerate(range(-(arr1.size - 1), arr2.size)):
            if lag < 0:
                arr1_start_idx = arr1.size - (lag + arr1.size)
                arr1_end_idx = min(arr1.size, arr2.size + arr1_start_idx)
                all_rms[i] = np.nanmean(
                    (arr1[arr1_start_idx:arr1_end_idx] - arr2[: arr1.size + lag]) ** 2
                )
            elif lag + arr1.size <= arr2.size:
                all_rms[i] = np.nanmean((arr1 - arr2[lag : lag + arr1.size]) ** 2)
            else:
                all_rms[i] = np.nanmean((arr1[: arr2.size - lag] - arr2[lag:]) ** 2)
    return all_rms


def select_time_of_interest(
    vicon: dict[str, ktk.TimeSeries],
    theia: dict[str, ktk.TimeSeries],
    exercise: str,
) -> tuple[float, float, str | None]:
    """Function to manually select the time of interest. The function plots the
    knee flexion angle and the elbow flexion angle of the Vicon and Theia systems.
    The user can select the time of interest by selecting a time range in the plot.

    Args:
        vicon (dict[str, ktk.TimeSeries]): The Vicon data.
        theia (dict[str, ktk.TimeSeries]): The Theia data.
        exercise (str): The exercise that is being analyzed.

    Returns:
        tuple[float, float]: The selected start and end time range.
    """
    valid_selection = False
    please_close = False
    show_graph = True
    special_side = None
    while not (valid_selection or please_close):
        if show_graph:
            start_time, end_time = plot_time_graph(
                exercise,
                vicon,
                theia,
            )

        if start_time == "q":
            special_side = input(
                "What is the special side? (shot, pass, step-out side) [N/L/R/S]: "
            )

        if (
            any([x in exercise for x in ["sidestep", "shot", "five"]])
            and not start_time == "q"
        ):
            segment = None
            if "five" in exercise:
                segment = "hand"
            elif "shot" in exercise:
                segment = "foot"

            if segment:
                right_mean = np.nanmean(vicon["points"].data[f"right_{segment}"][:, 2])
                left_mean = np.nanmean(vicon["points"].data[f"left_{segment}"][:, 2])
                special_side = "r" if right_mean > left_mean else "l"

            elif "sidestep" in exercise:
                valid_idxs = np.where(
                    ~np.isnan(vicon["points"].data["centre_of_mass"][:, 1])
                )[0]
                first_idx = valid_idxs[0]
                perc_75_idx = int(np.percentile(valid_idxs, 75))

                if vicon["points"].data["centre_of_mass"][first_idx, 1] > 0:
                    if (
                        vicon["points"].data["centre_of_mass"][first_idx, 0]
                        < vicon["points"].data["centre_of_mass"][perc_75_idx, 0]
                    ):
                        special_side = "l"
                    elif (
                        vicon["points"].data["centre_of_mass"][first_idx, 0]
                        > vicon["points"].data["centre_of_mass"][perc_75_idx, 0]
                    ):
                        special_side = "r"
                else:
                    if (
                        vicon["points"].data["centre_of_mass"][first_idx, 0]
                        < vicon["points"].data["centre_of_mass"][perc_75_idx, 0]
                    ):
                        special_side = "r"
                    elif (
                        vicon["points"].data["centre_of_mass"][first_idx, 0]
                        > vicon["points"].data["centre_of_mass"][perc_75_idx, 0]
                    ):
                        special_side = "l"

            else:
                special_side = input(
                    "What is the special side? (shot, pass, step-out side) [N/L/R/S]: "
                )
            print(f"Special side: {special_side}")
        elif isinstance(start_time, float):
            special_side = "n"

        if special_side == "s":
            LOGGER.info("The trial has been skipped due to NaN's.")
            raise ValueError("The trial has been skipped due to NaN's.")

        valid_selection = _assert_propper_input(
            start_time, end_time, theia, vicon, special_side
        )

        if not valid_selection:
            LOGGER.info(
                f"The selected time range ({start_time} - "
                f"{end_time}) contains NaN's. Please select a different "
                "range or specify the special side."
            )

        if please_close:
            LOGGER.info("The user has closed the manual time selection window.")
            raise UserWarning("The user has closed the manual time selection window.")

    side = {"l": "left", "r": "right", "n": None, "s": "Skipped due to NaNs"}.get(
        special_side.lower(), None
    )
    start_time = float(start_time)
    end_time = float(end_time)

    return start_time, end_time, side


def _assert_propper_input(
    t_min: float,
    t_max: float,
    theia: dict[str, ktk.TimeSeries],
    vicon: dict[str, ktk.TimeSeries],
    special_side: str | None,
):
    if special_side and special_side not in ["n", "l", "r", "s"]:
        print("The special side should be n, l, s, or r.")
        return False
    elif special_side == "s":
        # The trial has been skipped due to NaN's
        return True

    try:
        t_min = float(t_min)
        t_max = float(t_max)
    except ValueError:
        print("The input should be a float.")
        return False

    if t_min >= t_max:
        print("The start time should be smaller than the end time.")
        return False

    for data_type in ["angles", "rotations", "points"]:
        for key in theia[data_type].data.keys():
            if (
                theia[data_type].get_ts_between_times(t_min, t_max).isnan(key).any()
                or vicon[data_type].get_ts_between_times(t_min, t_max).isnan(key).any()
            ):
                print(
                    "The selected time range contains NaN's. "
                    "Please select a different range."
                )
                return False

    return True


def plot_time_graph_fig(
    exercise: str,
    vicon: dict[str, ktk.TimeSeries],
    theia: dict[str, ktk.TimeSeries],
):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 10))
    fig.suptitle("Select the time of interest, press 'q' to quit.")
    ax1.set_title(exercise)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Angle (deg)")

    ax2.set_ylabel("Velocity (m/s)")

    ax1.plot(
        vicon["angles"].time,
        vicon["angles"].data["right_knee"][:, 0],
        label="Vicon",
    )
    ax1.plot(
        theia["angles"].time,
        theia["angles"].data["right_knee"][:, 0],
        label="Theia",
    )

    left_foot_velocity = (
        np.diff(vicon["points"].data["left_foot"][:, :3], axis=0)
        / np.diff(vicon["points"].time)[:, np.newaxis]
    )
    left_foot_total_velocity = np.sqrt(np.sum(left_foot_velocity**2, axis=1))

    right_foot_velocity = (
        np.diff(vicon["points"].data["right_foot"][:, :3], axis=0)
        / np.diff(vicon["points"].time)[:, np.newaxis]
    )
    right_foot_total_velocity = np.sqrt(np.sum(right_foot_velocity**2, axis=1))

    ax2.plot(
        vicon["points"].time[:-1]
        + (vicon["points"].time[1] - vicon["points"].time[0]) / 2,
        left_foot_total_velocity,
        label="left foot",
        color="tab:blue",
    )
    ax2.plot(
        vicon["points"].time[:-1]
        + (vicon["points"].time[1] - vicon["points"].time[0]) / 2,
        right_foot_total_velocity,
        label="right foot",
        color="tab:orange",
    )

    ax3.plot(
        vicon["points"].data["centre_of_mass"][:, 1],
        vicon["points"].data["centre_of_mass"][:, 0] * -1,
        label="CoM xy",
    )
    ax3.plot(
        theia["points"].data["centre_of_mass"][:, 1],
        theia["points"].data["centre_of_mass"][:, 0] * -1,
        label="CoM xy theia",
    )

    not_nan_idx_vicon = np.where(
        ~np.isnan(vicon["points"].data["centre_of_mass"][:, 0])
    )[0][0]
    not_nan_idx_theia = np.where(
        ~np.isnan(theia["points"].data["centre_of_mass"][:, 0])
    )[0][0]

    ax3.scatter(
        [vicon["points"].data["centre_of_mass"][not_nan_idx_vicon, 1]],
        [vicon["points"].data["centre_of_mass"][not_nan_idx_vicon, 0] * -1],
        label="start",
        s=100,
    )

    ax3.scatter(
        [theia["points"].data["centre_of_mass"][not_nan_idx_theia, 1]],
        [theia["points"].data["centre_of_mass"][not_nan_idx_theia, 0] * -1],
        label="start",
        s=100,
    )

    ax3.set_xlabel("Position (m)")
    ax3.set_ylabel("Position (m)")
    ax3.legend()

    min_t = min(vicon["angles"].time[0], theia["angles"].time[0])
    max_t = max(vicon["angles"].time[-1], theia["angles"].time[-1])
    ax2.set_xlim(min_t, max_t)
    ax1.set_xlim(min_t, max_t)

    if "sidestep" in exercise:
        ax4.plot(
            vicon["points"].time,
            vicon["points"].data["left_foot"][:, 0],
            label="left foot y",
        )
        ax4.plot(
            vicon["points"].time,
            vicon["points"].data["right_foot"][:, 0],
            label="right foot y",
        )

        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Position (m)")
        ax4.legend()
        ax4.set_xlim(min_t, max_t)

    elif "five" in exercise:
        ax4.plot(
            vicon["points"].time,
            vicon["points"].data["right_hand"][:, 2],
            label="right hand z",
        )
        ax4.plot(
            vicon["points"].time,
            vicon["points"].data["left_hand"][:, 2],
            label="left hand z",
        )
        ax4.legend()
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Position (m)")
        ax4.set_xlim(min_t, max_t)

    theia_nans = np.array([False for _ in range(len(theia["angles"].time))])
    vicon_nans = np.array([False for _ in range(len(vicon["angles"].time))])
    for data_type in theia.keys():
        for key in theia[data_type].data.keys():
            vicon_nans |= vicon[data_type].isnan(key)
            theia_nans |= theia[data_type].isnan(key)

    com_condition = (
        np.abs(vicon["points"].data["centre_of_mass"][:, 0]) < 2.5
    ) & (  # x close to origin
        vicon["points"].data["centre_of_mass"][:, 1] < 1.3
    )  # y close to origin
    for axis in [ax1, ax2]:
        y_min = axis.get_ylim()[0] * 0.9
        y_max = axis.get_ylim()[1] * 0.9
        axis.fill_between(
            vicon["angles"].time,
            y_min,
            y_max,
            where=vicon_nans,
            color="gray",
            alpha=0.3,
            label="vicon_has NaN's",
        )
        axis.fill_between(
            theia["angles"].time,
            y_min,
            y_max,
            where=theia_nans,
            color="darkred",
            alpha=0.3,
            label="theia has NaN's",
        )
        axis.fill_between(
            vicon["angles"].time,
            y_min,
            y_max,
            where=com_condition,
            color="green",
            alpha=0.1,
            label="CoM close to origin",
        )
    ax1.legend()
    ax2.legend()

    return fig, (ax1, ax2, ax3, ax4)


def plot_time_graph(
    exercise: str,
    vicon: dict[str, ktk.TimeSeries],
    theia: dict[str, ktk.TimeSeries],
):
    fig, (ax1, ax2, ax3, ax4) = plot_time_graph_fig(exercise, vicon, theia)
    custom_selector = CustomSpanSelector(
        fig,
        ax2,
        vicon,
        theia,
    )
    custom_selector.setup_span_selector()

    plt.show()

    allowed_to_close = custom_selector.allowed_to_close()
    while not allowed_to_close and not custom_selector.please_close:
        print("Please select a valid time range without NaN's.")
        fig, (ax1, ax2, ax3, ax4) = plot_time_graph_fig(exercise, vicon, theia)
        custom_selector = CustomSpanSelector(
            fig,
            ax2,
            vicon,
            theia,
        )
        custom_selector.setup_span_selector()
        plt.show()

        if custom_selector.please_close:
            break
        allowed_to_close = custom_selector.allowed_to_close()

    plt.close()

    return (
        custom_selector.t_min if not custom_selector.please_close else "q",
        custom_selector.t_max,
    )
