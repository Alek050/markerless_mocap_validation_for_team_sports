import sys

import kineticstoolkit as ktk
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

from constants import INTERCONNECTIONS, EXERCISES

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from markerless_mocap_validation_for_team_sports.combine import \
    get_combined_time_series  # noqa
from markerless_mocap_validation_for_team_sports.preprocess_c3d import \
    get_preprocessed_timeseries  # noqa

matplotlib.use("qt5agg")


def main(
    theia_c3d: str | None = None,
    vicon_c3d: str | None = None,
    vicon_person: str | None = None,
    preprocessing_df: str | None = None,
):

    if theia_c3d:
        theia_data = get_preprocessed_timeseries(theia_c3d, "theia")
        leading_c3d = theia_c3d

    if vicon_c3d:
        new_time = theia_data["points"].time if theia_c3d is not None else None
        vicon_person = vicon_person if "both" in vicon_c3d else None
        vicon_data = get_preprocessed_timeseries(
            vicon_c3d, "vicon", new_time_vicon=new_time, vicon_person=vicon_person
        )
        leading_c3d = vicon_c3d

    kwargs = {}
    if preprocessing_df is not None:
        raise NotImplementedError("works only when theia and vicon are specified")
        preprocessing_df = pd.read_csv(preprocessing_df)
        mask = (preprocessing_df["vicon_file"].str.contains(vicon_c3d)) & (
            preprocessing_df["theia_file"].str.contains(theia_c3d)
        )
        if len(preprocessing_df[mask]) == 1:
            kwargs = {
                "exercise": preprocessing_df[mask]["exercise"].values[0],
                "start_time": preprocessing_df[mask]["start_time"].values[0],
                "end_time": preprocessing_df[mask]["end_time"].values[0],
                "special_side": preprocessing_df[mask]["special_side"].values[0],
            }

    if "exercise" not in kwargs:
        exercises = [x for x in EXERCISES if x in leading_c3d]
        kwargs["exercise"] = exercises[0] if len(exercises) == 1 else "unknown"

    if theia_c3d and vicon_c3d:
        vicon_data, theia_data, _ = get_combined_time_series(
            vicon_data, theia_data, **kwargs
        )

        time = np.round(theia_data["points"].time, 4)
        translated_vicon_data = {
            "points": ktk.TimeSeries(data={}, time=time),
            "transformation_matrices": ktk.TimeSeries(
                data={}, time=time
            ),
        }
        for point in vicon_data["points"].data.keys():
            translated_vicon_data["points"].data[f"{point}_vicon"] = vicon_data[
                "points"
            ].data[point]
            translated_vicon_data["points"].data[f"{point}_vicon"][:, 0] += 1
        for rotation in vicon_data["transformation_matrices"].data.keys():
            translated_vicon_data["transformation_matrices"].data[
                f"{rotation}_vicon"
            ] = vicon_data["transformation_matrices"].data[rotation]
            translated_vicon_data["transformation_matrices"].data[f"{rotation}_vicon"][
                :, 0, 3
            ] += 1

        theia_data["points"].time = time
        theia_data["transformation_matrices"].time = time

        timeseries_to_plot = [theia_data, translated_vicon_data]
    else:
        timeseries_to_plot = [vicon_data] if vicon_c3d else [theia_data]

    combined_contents = [
        ts["points"] for ts in timeseries_to_plot
    ]
    while len(combined_contents) > 1:
        new_combined_contents = [combined_contents[0].merge(combined_contents[1])]
        if len(combined_contents) > 2:
            new_combined_contents += combined_contents[2:]

        combined_contents = new_combined_contents

    player = ktk.Player(up="z")
    player.set_contents(combined_contents[0])
    player.set_interconnections(INTERCONNECTIONS)
    player.play()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize the movement of the players."
    )
    parser.add_argument(
        "--theia_c3d", type=str, help="The path to the theia c3d file.", required=False
    )
    parser.add_argument(
        "--vicon_c3d", type=str, help="The path to the vicon c3d file.", required=False
    )
    parser.add_argument(
        "--vicon_person",
        type=str,
        help="The person in the vicon c3d file.",
        required=False,
    )
    parser.add_argument(
        "--preprocessing_df",
        type=str,
        help="The path to the preprocessing df file.",
        required=False,
    )

    args = parser.parse_args()
    main(args.theia_c3d, args.vicon_c3d, args.vicon_person, args.preprocessing_df)
