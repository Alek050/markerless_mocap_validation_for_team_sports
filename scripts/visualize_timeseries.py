import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd

from constants import EXERCISES


def plot_timeseries(
    vicon: str,
    theia: str,
    data_name: str,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the time series data from the Vicon and Theia systems.

    Args:
        vicon (str) the path to the vicon parquet data
        theia (str) the path to the theia parquet data

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The figure and
            axes objects.
    """

    vicon_data = pd.read_parquet(vicon)
    theia_data = pd.read_parquet(theia)

    columns = [x for x in vicon_data.columns if data_name in x]
    if len(columns) == 0:
        raise ValueError(f"Data name {data_name} not found in columns")
    vicon_data = vicon_data[columns]
    theia_data = theia_data[columns]

    y_label_map = {
        "rotations": "Angle (deg)",
        "angles": "Angle (deg)",
        "points": "Position (m)",
    }
    y_label = [x for x in y_label_map.keys() if x in vicon][0]

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    for i, ax in enumerate(axs):
        current_vicon_data = vicon_data.iloc[:, i]
        current_theia_data = theia_data.iloc[:, i]
        time = current_vicon_data.index
        ax.plot(
            time,
            current_vicon_data,
            label="Vicon",
            marker="o",
            alpha=0.5,
        )

        ax.plot(
            time,
            current_theia_data,
            label="Theia",
            marker="x",
            alpha=0.5,
        )
        replace_map = {
            "[0]": f" x {y_label_map[y_label]} (flexion/extension)",
            "[1]": f" y {y_label_map[y_label]} (abduction/adduction)",
            "[2]": f" z {y_label_map[y_label]} (internal/external rotation)",
        }
        title = vicon_data.columns[i]
        for k, v in replace_map.items():
            title = title.replace(k, v)


        ax.set_title(title.lower())
        ax.set_ylabel(y_label_map[y_label])

    axs[0].legend()
    axs[2].set_xlabel("Time (s)")
    plt.tight_layout()

    return fig, axs


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_data_dir", type=str, default="data/preprocessed_dataset"
    )
    parser.add_argument("--exercise_name", type=str, default="double_sidestep")
    parser.add_argument("--participant_number", type=int, default=1)
    parser.add_argument("--trial_number", type=int, default=1)
    parser.add_argument("--data_type", type=str, default="joint_angles")
    parser.add_argument("--data_name", type=str, default="left_knee")

    args = parser.parse_args()
    exercises = [x.replace(" ", "_") for x in EXERCISES]

    # validate input
    if not os.path.exists(args.processed_data_dir):
        raise ValueError(f"processed_data_dir {args.processed_data_dir} does not exist")
    if args.exercise_name not in exercises:
        raise ValueError(f"exercise_name {args.exercise_name} is not in {exercises}")
    if args.participant_number not in range(1, 25):
        raise ValueError(
            f"participant_number {args.participant_number} should be in [1, 24]"
        )
    if args.trial_number not in range(1, 7):
        raise ValueError(f"trial_number {args.trial_number} should be in [1, 6]")
    if args.data_type not in ["joint_angles", "points", "segment_angles"]:
        raise ValueError(
            f"data_type {args.data_type} should be in ['joint_angles',"
            " 'points', 'segment_angles']"
        )

    data_type_map = {
        "joint_angles": "angles",
        "points": "points",
        "segment_angles": "rotations",
    }
    exercises_map = {x.replace(" ", "_"): x for x in EXERCISES}

    vicon_file = os.path.join(
        args.processed_data_dir,
        f"pp{args.participant_number}",
        f"{exercises_map[args.exercise_name]} {args.trial_number:03d}"
        f" {data_type_map[args.data_type]} vicon.parquet",
    )
    theia_file = vicon_file.replace("vicon", "theia")

    fig, ax = plot_timeseries(vicon_file, theia_file, args.data_name)
    plt.show()
