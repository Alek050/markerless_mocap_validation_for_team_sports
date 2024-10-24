KNOWN_TIMESERIES: list[str] = [
    "rotations",
    "angles",
    "points",
    "transformation_matrices",
]
"""The timeseries that is returned by the preprocess function."""


THEIA_MAP: dict[str, str] = {
    "worldbody_4X4": "lab",
    "pelvis_4X4": "pelvis",
    "l_thigh_4X4": "left_thigh",
    "l_shank_4X4": "left_shank",
    "l_foot_4X4": "left_foot",
    "l_toes_4X4": "left_toes",
    "r_thigh_4X4": "right_thigh",
    "r_shank_4X4": "right_shank",
    "r_foot_4X4": "right_foot",
    "r_toes_4X4": "right_toes",
    "torso_4X4": "torso",
    "l_uarm_4X4": "left_upper_arm",
    "l_larm_4X4": "left_lower_arm",
    "l_hand_4X4": "left_hand",
    "r_uarm_4X4": "right_upper_arm",
    "r_larm_4X4": "right_lower_arm",
    "r_hand_4X4": "right_hand",
    "head_4X4": "head",
}
"""
To map the names of the rotation matrices of
segments to global names and vica versa.
"""

VICON_MAP: dict[str, str] = {
    "PELO": "pelvis",
    "LFEO": "left_thigh",
    "RFEO": "right_thigh",
    "RTIO": "right_shank",
    "LTIO": "left_shank",
    "LFOO": "left_foot",
    "RFOO": "right_foot",
    "LTOO": "left_toes",
    "RTOO": "right_toes",
    "HEDO": "head",
    "TRXO": "torso",
    "CSPO": "spine",
    "SACO": "sacrum",
    "RCLO": "right_clavicle",
    "LCLO": "left_clavicle",
    "RHUO": "right_upper_arm",
    "LHUO": "left_upper_arm",
    "RRAO": "right_lower_arm",
    "LRAO": "left_lower_arm",
    "RHNO": "right_hand",
    "LHNO": "left_hand",
}
"""
To map the names of to global names and vica versa.
"""

VICON_SEGMENT_ABREVIATIONS: dict[str, str] = {
    "thigh": "FE",
    "shank": "TI",
    "foot": "FO",
    "toes": "TO",
    "upper_arm": "HU",
    "lower_arm": "RA",
    "hand": "HN",
    "pelvis": "PEL",
    "torso": "TRX",
    "head": "HED",
}
"""
To map the segment names to abreviations. For every segment you can add 'O' for the
origi, 'P' for the proximal, 'A' for anterior, and 'L' for the lateral. For instance,
the left thigh origi is 'LFEO'.
"""

CORRECT_COM_TRIALS = [
    "both7",
    "both8",
    "both9",
    "pp13",
    "pp14",
    "pp15",
    "pp16",
    "pp17",
    "pp18",
]
"""In these trials the center of mass in x direction was off by 10 cm.
This is corrected in the preprocess function."""
