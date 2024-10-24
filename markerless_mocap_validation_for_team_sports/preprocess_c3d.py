import os
import traceback
import warnings

import kineticstoolkit as ktk
import numpy as np
import pandas as pd

from markerless_mocap_validation_for_team_sports.constants import (
    CORRECT_COM_TRIALS, THEIA_MAP, VICON_MAP, VICON_SEGMENT_ABREVIATIONS)
from markerless_mocap_validation_for_team_sports.logging import create_logger

LOGGER = create_logger(__name__)


def get_preprocessed_timeseries(
    loc: str,
    data_provider: str,
    vicon_person: str | None = None,
    new_time_vicon: np.ndarray | None = None,
) -> dict[str, ktk.TimeSeries]:
    """Function to preprocess the data provided by the data provider. The preprocessing
    steps include:
    1. Reading the data from the location provided.
    2. Converting the data into a ktk.TimeSeries object.
    3. Calculating/renaming the angles of the joints.
    4. Calculating/renaming the rotation matrices of the segments.
    5. Adding the centre of mass to the points data.

    Args:
        loc (str): The location of the data.
        data_provider (str): The data provider of the data. Can be 'vicon' or 'theia'.
        vicon_person (str|None, optional): Only used when there are multiple persons in
            the vicon data. Defaults to None.
        new_time_vicon (np.ndarray, optional): The new time array for the Vicon data.
            Effictevely resamples the data to the new time array. Defaults to None.
            If None, the original time array is used.

    Returns:
        dict[str, ktk.TimeSeries]: The preprocessed angles and rotation matrices.

    Notes:
        Due to the set origin. The x axis of theia is to the right, y is to the
        anterior, for Vicon that is the other way around. The output of should be the
        same for both. X to the right, y to the anterior, and z to the superior.
    """
    try:
        LOGGER.info(
            f"Preprocessing data from {loc} using {data_provider} "
            f"data provider, vicon_person={vicon_person}."
        )
        if not isinstance(loc, str):
            raise TypeError("The location must be a string.")
        if not isinstance(data_provider, str):
            raise TypeError("The data provider must be a string.")
        if data_provider not in ["vicon", "theia"]:
            raise ValueError("The data provider must be either 'vicon' or 'theia'.")
        if (
            vicon_person is None
            and data_provider == "vicon"
            and loc.split("/")[-1][:4] == "both"
        ):
            raise ValueError(
                "The vicon_person must be specified when there are multiple persons."
            )

        c3d = ktk.read_c3d(loc, convert_point_unit=False)

        # Analyze vicon data
        if data_provider == "vicon":
            if loc.split("/")[-1][:4] == "both":
                prefix = vicon_person + ":"
            else:
                prefix = ""

            # normalize from mm to m and normalize origin to the theia origin
            for k, v in c3d["Points"].data.items():
                if "Angles" not in k:
                    v[:, :3] = v[:, :3] / 1000

            points, rotations, angles, time = get_vicon_single_person_data(
                c3d, prefix=prefix, new_time_vicon=new_time_vicon
            )
            points_data_info = {k: {"Unit": "m"} for k in points.keys()}

        else:  # data_provider == "theia"
            rotations = {
                THEIA_MAP.get(k): v
                for k, v in c3d["Rotations"].data.items()
                if THEIA_MAP.get(k) is not None
            }
            # rotate 90 degrees around z to make the x-axis to the right
            rot_mat = np.array(
                [
                    [0, -1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            for k in rotations.keys():
                rotations[k] = np.matmul(rot_mat, rotations[k])

            points = {k: rotations[k][:, :, 3] for k in rotations.keys()}

            # add the centre of mass
            com_loc = loc.replace(".c3d", " CoM.txt")
            if not os.path.exists(com_loc):
                raise FileNotFoundError(f"The CoM file {com_loc} does not exist.")
            centre_of_mass = pd.read_csv(
                com_loc, sep="\t", index_col=0, header=None, names=["x", "y", "z"]
            ).to_numpy()
            centre_of_mass = np.concatenate(
                [centre_of_mass, np.ones((centre_of_mass.shape[0], 1))], axis=1
            )
            points["centre_of_mass"] = ktk.geometry.rotate(
                centre_of_mass, "z", angles=[90], degrees=True
            )

            # Correct for wrong placement of lab origin for pp13 - pp18
            if any([x in loc for x in CORRECT_COM_TRIALS]):
                points["centre_of_mass"][:, 0] += 0.1

            angles = get_joint_angles(rotations)
            time = c3d["Rotations"].time
            points_data_info = {k: {"Unit": "m"} for k in points.keys()}

        rotation_angles = {}
        for transformation in rotations.keys():
            rotation_angles[transformation] = np.concatenate(
                [
                    get_angles_from_transform_matrix(
                        rotations[transformation], intrinsic=False
                    ),
                    np.ones((time.size, 1)),
                ],
                axis=1,
            )
            # make the angles consistent with the anatomical joint conventions
            if "left" in transformation:
                rotation_angles[transformation][:, 1:] *= -1

        # # make the angles consistent with the orientation relative to the lab origin
        if np.nanmean(np.abs(rotation_angles["head"][:, 2])) > 90:
            for key in rotation_angles.keys():
                rotation_angles[key][:, :2] *= -1

        output = {
            "angles": ktk.TimeSeries(data=angles, time=time),
            "rotations": ktk.TimeSeries(
                data=rotation_angles,
                time=time,
            ),
            "points": ktk.TimeSeries(
                data=points, time=time, data_info=points_data_info
            ),
            "transformation_matrices": ktk.TimeSeries(data=rotations, time=time),
        }

        LOGGER.info(f"Data preprocessing of {loc} completed.")
        return output
    except Exception as e:
        tb = traceback.format_exc()
        LOGGER.error(f"An error occurred while preprocessing the data: {e}\n{tb}")
        raise e


def get_vicon_single_person_data(
    vicon_c3d: dict[str, ktk.TimeSeries],
    prefix: str = "",
    new_time_vicon: np.ndarray | None = None,
) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray
]:
    """Function to get the vicon points, rotations, angles, and time of a single
    person. Prefix is used to specify that person.

    This function has the following steps:
    1. Resample the Vicon point data if needed.
    2. Get the points of interest, as defined in the VICON_MAP.
    3. Get the 4 by 4 transformation matrices based on the points.
    4. Get the joint angles based on the rotation matrices.

    Args:
        vicon_c3d (dict[str, ktk.TimeSeries]): The loaded c3d file with raw data.
        prefix (str, optional): Which particpant to include. Defaults to "".
            For example `pp1:`, `pp2:` or "".
        new_time_vicon (np.ndarray, optional): The new time array for the Vicon data.
            Defaults to None. If None, the original time array is used.

    Returns:
        tuple: The points, rotations, angles, and time.
    """
    if new_time_vicon is not None:
        vicon_c3d["Points"] = vicon_c3d["Points"].resample(new_time_vicon)

    time = vicon_c3d["Points"].time
    points = {
        VICON_MAP.get(k[-4:]): v
        for k, v in vicon_c3d["Points"].data.items()
        if VICON_MAP.get(k[-4:]) is not None and prefix in k
    }
    points["centre_of_mass"] = vicon_c3d["Points"].data[f"{prefix}CentreOfMass"]
    points["lab"] = np.repeat(np.array([[0, 0, 0, 1]]), time.size, axis=0)

    rotations = get_vicon_transformation_matrices(vicon_c3d["Points"], prefix=prefix)
    angles = get_joint_angles(rotations)

    return points, rotations, angles, time


def get_joint_angles(rotations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Get the joint angles from the transformation matrices provided.
    The rotation matrices are in the form of a dict. The ktk.geometry.get_angles()
    function is used to extract the joint angles from the rotation matrices.
    The joint angles are then returned as a dict.

    Args:
        rotations (dict[str, np.ndarray]): A dict containing the rotation matrices.

    Returns:
        dict[str, np.ndarray]: A dict containing the joint angles.
    """
    if "lab" not in rotations.keys():
        rotations["lab"] = np.repeat(
            np.eye(4)[np.newaxis], repeats=rotations["pelvis"].shape[0], axis=0
        )

    ref_seg_ang_conv_mult_add = [
        ["pelvis", "left_thigh", "left_hip"],
        ["left_thigh", "left_shank", "left_knee"],
        [
            "left_shank",
            "left_foot",
            "left_ankle",
        ],
        [
            "pelvis",
            "right_thigh",
            "right_hip",
        ],
        [
            "right_thigh",
            "right_shank",
            "right_knee",
        ],
        [
            "right_shank",
            "right_foot",
            "right_ankle",
        ],
        ["lab", "pelvis", "pelvis"],
        ["lab", "torso", "thorax"],
        ["lab", "head", "head"],
        [
            "torso",
            "left_upper_arm",
            "left_shoulder",
        ],
        [
            "left_upper_arm",
            "left_lower_arm",
            "left_elbow",
        ],
        [
            "left_lower_arm",
            "left_hand",
            "left_wrist",
        ],
        [
            "torso",
            "right_upper_arm",
            "right_shoulder",
        ],
        [
            "right_upper_arm",
            "right_lower_arm",
            "right_elbow",
        ],
        [
            "right_lower_arm",
            "right_hand",
            "right_wrist",
        ],
    ]

    angles = {}
    for (
        reference_segment_rot,
        segment_rot,
        name,
    ) in ref_seg_ang_conv_mult_add:
        local_rot_matrix = ktk.geometry.get_local_coordinates(
            rotations[segment_rot], rotations[reference_segment_rot]
        )

        current_angles = get_angles_from_transform_matrix(
            local_rot_matrix, intrinsic=True
        )
        if np.all(current_angles == 1):
            LOGGER.warning(f"Gimbal lock detected for {name}, setting all angles to 1.")
        angles[name] = current_angles

        # make anatomical joint conventions stable over each side. i.e.,
        # higher angles in the hip rotation should be external rotation
        # for both left and right side.
        if "left" in name:
            angles[name][:, 1:] *= -1

    return angles


def get_angles_from_transform_matrix(
    transformation_matrix: np.ndarray, intrinsic: bool = True
) -> np.ndarray:
    """Function to get the angles from the transformation matrix. The angles are
    extracted using the ktk.geometry.get_angles() function. The angles are then
    returned in the form of an array. The angles are in degrees, and a
    zyx rotation order is used. The angles are returned in x, y ,z order.

    Args:
        rotation_matrix (np.ndarray): The rotation matrix.

    Raises:
        UserWarning: If a gimbal lock is detected.

    Returns:
        np.ndarray: The angles in degrees in x, y, and z direction
    """
    angle_convention = "XYZ" if intrinsic else "zyx"
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=UserWarning)
        try:
            current_angles = ktk.geometry.get_angles(
                transformation_matrix, angle_convention, degrees=True
            )

        except UserWarning as w:
            if "Gimbal lock detected." in str(w):
                return np.ones((transformation_matrix.shape[0], 3))
            else:
                raise w

    # swap x and z axis back
    if not intrinsic:
        return np.array(
            [current_angles[:, 2], current_angles[:, 1], current_angles[:, 0]]
        ).T
    return current_angles


def get_vicon_transformation_matrices(
    points: ktk.TimeSeries, prefix: str = ""
) -> dict[str, np.ndarray]:
    """Function to get the rotation matrices of the segments from the Vicon data.
    The rotation matrices are calculated using the ktk.geometry.create_frames()
    function. The convention is to use x to the right, y to the anterior, and z
    to the superior of that segment. The distal point is the origin.

    Args:
        points (ktk.TimeSeries): The TimeSeries of the points data. Note that the
            points must include proximal and origin points of all segments
        prefix (str, optional): The prefix to add before the wanted point,
            for instance "pp1:". Defaults to "".


    Returns:
        dict[str, np.ndarray[4x4]]: A dict containing the rotation matrices of
            the segments.
    """

    rotations = {}
    for side in ["L", "R"]:
        side_name = "left" if side == "L" else "right"
        for segment, abbreviation in VICON_SEGMENT_ABREVIATIONS.items():
            if len(abbreviation) == 2:
                abbreviation = f"{side}{abbreviation}"

            if abbreviation in points.data.keys():
                # there is no left/right head, torso, or pelvis
                continue

            origin = points.data[f"{prefix}{abbreviation}O"]
            proximal_ax = points.data[f"{prefix}{abbreviation}P"]
            lateral_ax = points.data[f"{prefix}{abbreviation}L"]
            anterior_ax = points.data[f"{prefix}{abbreviation}A"]

            # create the right transforms
            if segment in ["torso"]:
                x = lateral_ax - origin
                xz = origin - proximal_ax
            elif f"{side_name}_{segment}" in ["left_clavicle"]:
                x = proximal_ax - origin
                xz = lateral_ax - origin
            elif f"{side_name}_{segment}" in ["right_clavicle"]:
                x = origin - proximal_ax
                xz = origin - lateral_ax
            elif f"{side_name}_{segment}" in [
                "left_upper_arm",
                "left_hand",
                "right_upper_arm",
                "right_hand",
            ]:
                x = lateral_ax - origin
                xz = proximal_ax - origin
            elif f"{side_name}_{segment}" in [
                "left_lower_arm",
            ]:
                x = anterior_ax - origin
                xz = proximal_ax - origin
            elif f"{side_name}_{segment}" in ["right_lower_arm"]:
                x = origin - anterior_ax
                xz = proximal_ax - origin
            elif f"{side_name}_{segment}" in [
                "left_foot",
                "left_toes",
                "right_foot",
                "right_toes",
            ]:
                x = origin - lateral_ax
                xz = anterior_ax - origin
            else:
                x = origin - lateral_ax
                xz = proximal_ax - origin

            rotation = ktk.geometry.create_frames(origin, x=x, xz=xz)

            rotation[:, 3, :] = [0.0, 0.0, 0.0, 1.0]

            if abbreviation[0] == side:
                rotations[f"{side_name}_{segment}"] = rotation
            else:
                rotations[f"{segment}"] = rotation

    return rotations
