import os
import re
import sys
import traceback

import kineticstoolkit as ktk
import numpy as np
import pandas as pd
from constants import THEIA_FOLDER_NAME, VICON_FOLDER_NAME

sys.path.append(".")
from markerless_mocap_validation_for_team_sports.combine import (  # noqa: E402
    get_combined_time_series, sync_timeseries)
from markerless_mocap_validation_for_team_sports.logging import \
    create_logger  # noqa: E402
from markerless_mocap_validation_for_team_sports.preprocess_c3d import \
    get_preprocessed_timeseries  # noqa: E402

LOGGER = create_logger(__name__)


def main(
    data_loc: str,
    output_dir: str,
    fill_missing_samples: bool = False,
    preprocessing_file_loc: str = "preprocessing_df.csv",
    regenerate: bool = False,
) -> None:
    """Function to generate the valid dataset. This function reads the data from the
    given data_loc and generates the valid dataset by combining the theia and vicon
    data. The generated dataset is saved in the output_dir.

    Args:
        data_loc (str): The location of the data directory.
        output_dir (str): The location where the generated dataset should be saved.
        fill_missing_samples (bool, optional): Whether to fill missing samples.
            Defaults to False.
        preprocessing_file_loc (str, optional): The location of the preprocessing file.
            Defaults to "preprocessing_df.csv".
        regenerate (bool, optional): Whether to re-calculate the data while using info
            of the preprocessing df or to skip it. Defaults to False.

    Returns:
        None
    """

    assert os.path.exists(
        preprocessing_file_loc
    ), f"{preprocessing_file_loc} should exist"
    preprocessing_df = pd.read_csv(preprocessing_file_loc)

    try:
        LOGGER.info(
            f"Generating valid dataset from '{data_loc}' and saving to '{output_dir}',"
            f"fill_missing_samples: {fill_missing_samples}"
        )
        all_dirs = os.listdir(data_loc)
        if THEIA_FOLDER_NAME not in all_dirs or VICON_FOLDER_NAME not in all_dirs:
            raise ValueError(
                f"The {data_loc} should contain {THEIA_FOLDER_NAME} and "
                f"{VICON_FOLDER_NAME} directories, but only found the following"
                f" directories: {all_dirs}"
            )

        if not os.path.exists(output_dir):
            LOGGER.info(f"Creating output directory: '{output_dir}'")
            os.makedirs(output_dir)

        vicon_files = [
            file
            for file in sorted(os.listdir(f"{data_loc}/{VICON_FOLDER_NAME}"))
            if re.search(r"\d\d\d\.c3d$", file)
        ]

        for i, vicon_file_name in enumerate(np.sort(vicon_files)):
            print(f"Processing {i+1}/{len(vicon_files)}: {vicon_file_name}")
            try:
                LOGGER.info(f"Creating right order for trial '{vicon_file_name}'")
                # get the subject, trial number and trial name
                file_subject = vicon_file_name.split(" ")[0]
                trial_number = vicon_file_name.split()[-1].split(".")[0]
                trial_name = " ".join(vicon_file_name.split(" ")[1:-1])
                theia_file = (
                    f"{data_loc}/{THEIA_FOLDER_NAME}/{file_subject} "
                    f"{trial_name} {trial_number} pose_filt_0.c3d"
                )
                vicon_file = f"{data_loc}/{VICON_FOLDER_NAME}/{vicon_file_name}"

                if "both" in file_subject:
                    both_num = int(file_subject[4:])
                    participant_numbers = [f"pp{both_num* 2 - 1}", f"pp{both_num * 2}"]

                else:
                    participant_numbers = [file_subject]

                if "both" in file_subject:
                    LOGGER.info(f"Processing both participants for {vicon_file_name}")
                    if not len(participant_numbers) == 2:
                        raise NotImplementedError(
                            "Subject is 'both' but len(participant_numbers) "
                            f"is not 2: {participant_numbers}"
                        )

                    mask = (preprocessing_df["vicon_file"].str.contains(vicon_file.split(VICON_FOLDER_NAME)[-1])) & (
                        preprocessing_df["participant_number"].isin(participant_numbers)
                    )
                    if (
                        mask.sum() >= 2 and not regenerate
                    ):  # both participants have been processed
                        LOGGER.info(
                            "Both participants have been processed for "
                            f"'{vicon_file_name}'. Skipping..."
                        )
                        continue

                    LOGGER.info("Trying to get preprocessed data...")
                    theia_data = get_preprocessed_timeseries(theia_file, "theia")
                    vicon_data_1 = get_preprocessed_timeseries(
                        vicon_file,
                        "vicon",
                        vicon_person=participant_numbers[0],
                        new_time_vicon=theia_data["points"].time,
                    )
                    vicon_data_2 = get_preprocessed_timeseries(
                        vicon_file,
                        "vicon",
                        vicon_person=participant_numbers[1],
                        new_time_vicon=theia_data["points"].time,
                    )
                    LOGGER.info("Successfully loaded preprocessed data...")

                    if fill_missing_samples:
                        LOGGER.info("Filling missing samples...")
                        for data_type in ["angles", "rotations", "points"]:
                            vicon_data_1[data_type].fill_missing_samples(
                                40, method="linear", in_place=True
                            )
                            vicon_data_2[data_type].fill_missing_samples(
                                40, method="linear", in_place=True
                            )
                            theia_data[data_type].fill_missing_samples(
                                int(60 * 0.2), method="linear", in_place=True
                            )
                        LOGGER.info("Successfully filled missing samples...")

                    subject_list = get_right_ordered_subject_list(
                        vicon_data_1, vicon_data_2, theia_data, participant_numbers
                    )
                    LOGGER.info(
                        "Right ordered subject list: theia_filt_0 with "
                        f"{subject_list[0]} and theia_filt_1 with {subject_list[1]}"
                    )

                    theia_files = [
                        f"{data_loc}/{THEIA_FOLDER_NAME}/{file_subject} {trial_name}"
                        f" {trial_number} pose_filt_0.c3d",
                        f"{data_loc}/{THEIA_FOLDER_NAME}/{file_subject} {trial_name}"
                        f" {trial_number} pose_filt_1.c3d",
                    ]

                else:  # single person
                    subject_list = [file_subject]
                    theia_files = [theia_file]

                    mask = (preprocessing_df["vicon_file"].str.contains(vicon_file.split(VICON_FOLDER_NAME)[-1])) & (
                        preprocessing_df["participant_number"].isin(subject_list)
                    )

                    if mask.any() and not regenerate:
                        LOGGER.info(
                            f"Subject {subject_list[0]} has been processed for "
                            f"'{vicon_file_name}'. Skipping..."
                        )
                        continue

                    elif (
                        mask.any()
                        and regenerate
                        and "skipped"
                        in str(preprocessing_df[mask]["additional_info"].values[0])
                    ):
                        LOGGER.info(
                            f"Subject {subject_list[0]} has been processed for "
                            f"'{vicon_file_name}' but was skipped due to NaN's. No "
                            "regeneration necessary..."
                        )
                        continue

                for current_subject, current_theia_file in zip(
                    subject_list, theia_files
                ):
                    LOGGER.info(
                        f"Processing {current_subject} for '{vicon_file_name}'"
                        f" and '{current_theia_file}'"
                    )
                    current_output_dir = f"{output_dir}/{current_subject}"
                    current_info = {
                        "date": pd.Timestamp.now(),
                        "trial_name": trial_name,
                        "trial_number": int(trial_number),
                        "participant_number": current_subject,
                        "vicon_file": vicon_file,
                        "theia_file": current_theia_file,
                        "output_dir": output_dir,
                        "fill_missing_samples": fill_missing_samples,
                    }

                    mask = (preprocessing_df["vicon_file"].str.contains(vicon_file.split(VICON_FOLDER_NAME)[-1])) & (
                        preprocessing_df["participant_number"] == current_subject
                    )
                    if mask.sum() >= 1 and not regenerate:
                        current_info = preprocessing_df[mask].to_dict(orient="records")[
                            0
                        ]
                        LOGGER.info(
                            f"Found existing data for {current_subject} for "
                            f"'{vicon_file_name}'. Skipping..."
                        )
                        continue
                    elif (
                        mask.sum() >= 1
                        and regenerate
                        and "skipped"
                        in str(preprocessing_df[mask]["additional_info"].values[0])
                    ):
                        LOGGER.info(
                            f"Found existing data for {current_subject} for "
                            f"'{vicon_file_name}' but was skipped due to NaN's. "
                            "Skipping regeneration of data...."
                        )
                        continue

                    if regenerate and mask.sum() >= 1:
                        start_time = preprocessing_df[mask]["start_time"].values[0]
                        end_time = preprocessing_df[mask]["end_time"].values[0]
                        special_side = preprocessing_df[mask]["special_side"].values[0]
                    else:
                        start_time = np.nan
                        end_time = np.nan
                        special_side = None
                    
                    current_info = save_1_persion_data(
                        vicon_file,
                        current_theia_file,
                        trial_name,
                        trial_number,
                        current_output_dir,
                        fill_missing_samples,
                        vicon_prefix=current_subject,
                        current_info=current_info,
                        start_time=start_time,
                        end_time=end_time,
                        special_side=special_side,
                    )

                    if regenerate and mask.sum() >= 1:
                        preprocessing_df = preprocessing_df.drop(
                            preprocessing_df[mask].index
                        )

                    if len(preprocessing_df) == 0:
                        preprocessing_df = pd.DataFrame([current_info])
                    else:
                        preprocessing_df = pd.concat(
                            [preprocessing_df, pd.DataFrame([current_info])],
                            ignore_index=True,
                        )
                    preprocessing_df.to_csv("preprocessing_df.csv", index=False)

            except Exception:
                tb = traceback.format_exc()
                LOGGER.error(f"Error: {tb}")
                continue
    except Exception as e:
        tb = traceback.format_exc()
        LOGGER.error(f"Error: {e}\n{tb}")


def get_right_ordered_subject_list(
    vicon_data_1: dict[str, ktk.TimeSeries],
    vicon_data_2: dict[str, ktk.TimeSeries],
    theia_data: dict[str, ktk.TimeSeries],
    participant_numbers: list[str],
) -> list[str]:
    """Get the right ordered subject list based on the root mean square error between
    the vicon and theia data.

    Args:
        vicon_data_1 (dict[str, ktk.TimeSeries]): The vicon data for the first person
        vicon_data_2 (dict[str, ktk.TimeSeries]): The vicon data for the second person
        theia_data (dict[str, ktk.TimeSeries]): The theia data
        participant_numbers (list[str]): The participant numbers

    Returns:
        list[str]: The right ordered participant numbers
    """
    assert np.allclose(theia_data["points"].time, vicon_data_1["points"].time)

    vic_start1, vic_end1, theia_start1, theia_end1, _ = sync_timeseries(
        vicon_data_1, theia_data
    )
    vic_start2, vic_end2, theia_start2, theia_end2, _ = sync_timeseries(
        vicon_data_2, theia_data
    )

    T = np.mean(np.diff(vicon_data_1["points"].time))

    start_t_1 = (
        max(
            vicon_data_1["points"].time[vic_start1],
            theia_data["points"].time[theia_start1],
        )
        + 0.5 * T
    )
    end_t_1 = (
        min(
            vicon_data_1["points"].time[vic_end1], theia_data["points"].time[theia_end1]
        )
        - 0.5 * T
    )
    start_t_2 = (
        max(
            vicon_data_2["points"].time[vic_start2],
            theia_data["points"].time[theia_start2],
        )
        + 0.5 * T
    )
    end_t_2 = (
        min(
            vicon_data_2["points"].time[vic_end2], theia_data["points"].time[theia_end2]
        )
        - 0.5 * T
    )

    if round(start_t_1, 4) >= round(end_t_1, 4) and not round(start_t_2, 4) >= round(
        end_t_2, 4
    ):
        # Best sync found when there was no allignment between vicon1 and theia
        return list(participant_numbers[::-1])
    elif round(start_t_2, 4) >= round(end_t_2, 4) and not round(start_t_1, 4) >= round(
        end_t_1, 4
    ):
        # Best sync found when there was no allignment between vicon2 and theia
        return list(participant_numbers)

    vicon_data_1 = vicon_data_1["points"].get_ts_between_times(
        start_t_1, end_t_1, inclusive=[True, False]
    )
    theia_data_1 = theia_data["points"].get_ts_between_times(
        start_t_1, end_t_1, inclusive=[True, False]
    )
    vicon_data_2 = vicon_data_2["points"].get_ts_between_times(
        start_t_2, end_t_2, inclusive=[True, False]
    )
    theia_data_2 = theia_data["points"].get_ts_between_times(
        start_t_2, end_t_2, inclusive=[True, False]
    )

    mse_error1 = root_mean_square_error_com(vicon_data_1, theia_data_1)
    mse_error2 = root_mean_square_error_com(vicon_data_2, theia_data_2)

    if mse_error1 < mse_error2:
        return list(participant_numbers)
    else:
        return list(participant_numbers[::-1])


def root_mean_square_error_com(vicon_data, theia_data) -> float:
    """Calculate the root mean square error between the centre of mass of the vicon and
    theia data.

    Args:
        vicon_data (ktk.TimeSeries): The vicon data
        theia_data (ktk.TimeSeries): The theia data

    Returns:
        float: The root mean square error
    """
    vicon_com = vicon_data.data["centre_of_mass"]
    theia_com = theia_data.data["centre_of_mass"]
    return np.sqrt(np.nanmean((vicon_com - theia_com) ** 2))


def save_1_persion_data(
    vicon_loc: str,
    theia_loc: str,
    trial_name: str,
    trial_number: str,
    current_output_dir: str,
    fill_missing_samples: bool,
    vicon_prefix: str,
    current_info: dict = {},
    start_time: float = np.nan,
    end_time: float = np.nan,
    special_side: str = None,
) -> None:
    """Save the actual files for a single person.

    Args:
        vicon_loc (str): location of the vicon data
        theia_loc (str): location of the theia data
        trial_name (str): The trial name
        trial_number (str): The trial number
        current_output_dir (str): The output dir to save the data to
        fill_missing_samples (bool): whether to fill missing values of vicon
        vicon_prefix (str): The vicon person prefix. See get_prerpocessed_timeseries.
        current_info (dict, optional): The current info. Defaults to {}.
    """

    LOGGER.info(f"Saving data for {trial_name} {trial_number}")

    theia_data = get_preprocessed_timeseries(theia_loc, "theia")
    vicon_data = get_preprocessed_timeseries(
        vicon_loc,
        "vicon",
        vicon_person=vicon_prefix,
        new_time_vicon=theia_data["points"].time,
    )

    if fill_missing_samples:
        for data_type in ["angles", "rotations", "points"]:
            vicon_data[data_type].fill_missing_samples(
                20, method="linear", in_place=True
            )
            theia_data[data_type].fill_missing_samples(
                20, method="linear", in_place=True
            )

    LOGGER.info(f"Combining data for {trial_name} {trial_number}")

    try:
        vicon_data, theia_data, extra_info = get_combined_time_series(
            vicon_data, theia_data, trial_name, start_time, end_time, special_side
        )
    except ValueError as e:
        if "The trial has been skipped due to NaN's." in str(e):
            LOGGER.warning(
                f"Skipping {trial_name} {trial_number} due to NaN's in"
                f" {vicon_loc} or {theia_loc}"
            )
            return current_info | {
                "additional_info": "skipped due to NaN's",
                "saved": False,
            }
        else:
            raise e

    # check correlation in knee angles
    for side in ["left", "right"]:
        vicon_knee_angle = vicon_data["angles"].data[f"{side}_knee"][:, 0]
        theia_knee_angle = theia_data["angles"].data[f"{side}_knee"][:, 0]
        correlation = np.corrcoef(vicon_knee_angle, theia_knee_angle)[0, 1]
        if correlation < 0.9:
            LOGGER.warning(
                f"Low correlation between {side} knee angles for {trial_name} "
                f"{trial_number}: {correlation} File: {vicon_loc} and {theia_loc}"
            )

    current_info["time_offset"] = extra_info.pop("vicon_lag")
    current_info = {**current_info, **extra_info}
    current_info["data_length_s"] = (
        vicon_data["angles"].time[-1] - vicon_data["angles"].time[0]
    )
    current_info["saved"] = False

    if not os.path.exists(current_output_dir):
        os.makedirs(current_output_dir)

    # save the data
    for data_type in ["angles", "rotations", "points"]:
        vicon_data[data_type].to_dataframe().to_parquet(
            f"{current_output_dir}/{trial_name} {trial_number} "
            f"{data_type} vicon.parquet"
        )
        theia_data[data_type].to_dataframe().to_parquet(
            f"{current_output_dir}/{trial_name} {trial_number} "
            f"{data_type} theia.parquet"
        )
        LOGGER.info(
            f"Saved {data_type} data for {trial_name} {trial_number}"
            f" in loc {current_output_dir}"
        )
        current_info["saved"] = True

    return current_info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/preprocessed_dataset")
    parser.add_argument("--input_dir", type=str, default="data")
    parser.add_argument("--regenerate", type=bool, default=False)
    parser.add_argument("--fill_missing_samples", type=bool, default=False)
    parser.add_argument(
        "--preprocessing_file_loc",
        type=str,
        default="preprocessing_df.csv",
    )
    args = parser.parse_args()

    if args.preprocessing_file_loc == "preprocessing_df.csv" and not os.path.exists(
        args.preprocessing_file_loc
    ):
        preprocessing_df = pd.DataFrame(
            columns=[
                "date",
                "trial_name",
                "trial_number",
                "participant_number",
                "vicon_file",
                "theia_file",
                "output_dir",
                "fill_missing_samples",
                "time_offset",
                "start_time",
                "end_time",
                "data_length_s",
                "saved",
                "special_side",
                "additional_info",
            ]
        )
        preprocessing_df.to_csv("preprocessing_df.csv", index=False)

    main(
        args.input_dir,
        args.output_dir,
        args.fill_missing_samples,
        args.preprocessing_file_loc,
        args.regenerate,
    )
