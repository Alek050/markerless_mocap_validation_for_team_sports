THEIA_FOLDER_NAME = "theia_c3d_files"
VICON_FOLDER_NAME = "vicon_c3d_files"

EXERCISES = [
    "double sidestep",
    "dribble simulation",
    "high five",
    "sidestep",
    "slow dribble",
    "pass",
    "shot",
    "walk rotation",
]

INTERCONNECTIONS = dict()
INTERCONNECTIONS["center"] = {
    "Color": (1, 1, 0),
    "Links": [
        ["*head", "*torso", "*pelvis"],
        ["*left_clavicle", "*torso", "*right_clavicle"],
    ],
}
INTERCONNECTIONS["left"] = {
    "Color": (0, 1, 1),
    "Links": [
        [
            "*torso",
            "*left_upper_arm",
            "*left_lower_arm",
            "*left_hand",
        ],
        [
            "*pelvis",
            "*left_thigh",
            "*left_shank",
            "*left_foot",
            "*left_toes",
        ],
    ],
}
INTERCONNECTIONS["right"] = {
    "Color": (1, 0, 1),
    "Links": [
        [
            "*torso",
            "*right_upper_arm",
            "*right_lower_arm",
            "*right_hand",
        ],
        [
            "*pelvis",
            "*right_thigh",
            "*right_shank",
            "*right_foot",
            "*right_toes",
        ],
    ],
}
INTERCONNECTIONS["center_vicon"] = {
    "Color": (1, 1, 0),
    "Links": [
        [x + "_vicon" for x in INTERCONNECTIONS["center"]["Links"][0]],
        [x + "_vicon" for x in INTERCONNECTIONS["center"]["Links"][1]],
    ],
}
INTERCONNECTIONS["left_vicon"] = {
    "Color": (0, 1, 1),
    "Links": [
        ["*left_clavicle_vicon"]
        + [x + "_vicon" for x in INTERCONNECTIONS["left"]["Links"][0][1:]],
        [x + "_vicon" for x in INTERCONNECTIONS["left"]["Links"][1]],
    ],
}
INTERCONNECTIONS["right_vicon"] = {
    "Color": (1, 0, 1),
    "Links": [
        ["*right_clavicle_vicon"]
        + [x + "_vicon" for x in INTERCONNECTIONS["right"]["Links"][0][1:]],
        [x + "_vicon" for x in INTERCONNECTIONS["right"]["Links"][1]],
    ],
}
