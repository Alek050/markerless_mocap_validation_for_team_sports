# markerless_mocap_validation_for_team_sports

Welcome to the github repository related to the paper "Examining the Concurrent Validity of Markerless Motion Capture in Dual-Athlete Team Sports Movements". This repository consitsts of the code used to analyse the data, and generate the results of the paper. The data is available ...

## Getting Started

To be able to run the code, and tweak it for your own purposes, you need to follow the steps below.

1. First, clone the repository:
```bash
git clone https://github.com/Alek050/markerless_mocap_validation_for_team_sports.git
```

2. Install the required packages using conda:
```bash
conda env create -f environment.yaml
```

3. Activate the conda environment:
```bash
conda activate markerless_mocap_env
```

From here on, you can run the code in the repository. We have created different folders indicating different use cases. In general, the `markerless_mocap_validation_for_team_sports` folder contains the main code for the project. The `data` folder is empty in GitHub, but should contain the data you want to use for the project. The `scripts` folder contains specific code used to process all the data and get to the results (`generate_valid_dataset.py`) and (`generate_statistics.py`), and some extra scripts to visualize the data (`vizualise_movement.py`) and (`visualize_timeseries.py`).

## Data availability

...

## Running the scripts

### Generate preprocessed dataset
To get the preprocessed dataset, you can download the data from the data availability section. If you want to create your own dataset, you can download the raw data, or use your own. To run the script, you can use the following command:
```bash
python3 scripts/generate_valid_dataset.py 
```
By default, the follwing arguments are used:
```bash
--input_dir data
--output_dir data/preprocessed_dataset
--fill_missing_samples False
--preprocessing_file_loc preprocessing_df.csv
--regenerate False
```
The `input_dir` refers to the directory with the input raw data folders of theia and vicon. The `input_dir` should include two folders: `theia_c3d_files` and `vicon_c3d_files`. The `output_dir` refers to the directory where the preprocessed dataset will be saved. The `fill_missing_samples` argument is used to fill the missing samples in the dataset. The `preprocessing_file_loc` argument is used to save the preprocessing information. The `regenerate` argument is used to regenerate the preprocessed dataset, while using the preprocessing information to skip the manual selection of the start and end time for every sample, if availble in the preprocessing file.

### Generate statistics
To save the statistics of the preprocessed dataset in excel files, you can run the following command:

```bash
python3 scripts/generate_statistics.py
```
By default, the follwing arguments are used:
```bash
--input_dir data/preprocessed_dataset
--exercise all
--preprocessing_file_loc preprocessing_df.csv
```
The `input_dir` refers to the directory with the preprocessed dataset. The `exercise` argument is used to select the exercise you want to generate the statistics for. The value can be "all", "double sidestep", "dribble simulation", "high five", "sidestep", "slow dribble", "pass", "shot", and "walk rotation", if any trials of that are found in the `input_dir` directory. The `preprocessing_file_loc` argument is used to load the preprocessing information, which is used to normalize the statistics for the special side of an exercise (e.g. the hand used for the high five (left or right) can be different for every couple).

### Visualize movement

To better grasp the data, we have added a script to visualize the 3D movements of the athletes. To run the script, you can use the following command:

```bash
python3 scripts/visualize_movement.py --theia_c3d "data/theia_c3d_files/both11 double sidestep 002 pose_filt_1.c3d" --vicon_c3d "data/vicon_c3d_files/both11 double sidestep 002.c3d" --vicon_person pp22
```

For the theia data, two files are generated, one for each person (`pose_filt_0` and `pose_filt_1`), while vicon adds them in one file. You have to specify the right vicon person (`pp22`) with the right theia file (`pose_filt_1`). If you want to visualize the data of the other person, you need to change both the vicon person (`pp21`) and the theia file (`pose_filt_0`).

### Visualize timeseries

To compare the timeseries of a segment rotation (e.g. pelvic tilt), joint angle (e.g. knee flexion/extension angle), or a point (e.g. centre of mass), you can use the following command:

```bash
python3 scripts/visualize_timeseries.py
```

The default arguments used are:
```bash
--processed_data_dir data/preprocessed_dataset
--exercise_name double_sidestep
--particpant_number 1
--trial_number 1
--data_type joint_angles
--data_name left_knee
```

`exercise_name` should be one of the following: "double_sidestep", "dribble_simulation", "high_five", "sidestep", "slow_dribble", "pass", "shot", "walk_rotation". `particpant_number` should be an integer between 1 and 24. `trial_number` should be an integer between 1 and 6. `data_type` should be one of the following: "segment_angles", "joint_angles", "points". `data_name` should be the name of the segments, joint angle, or points you want to visualize.

## Reference

If you use the code in this repository, or the data associated with the paper, please cite the following paper:

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

We would like to thank [Tamar](...) and [Dave](...) for their help in collecting the data for this project. Furthermore, we would like to thank [Felix](https://github.com/felixchenier) for creating and open sourcing the [kineticstoolkit](https://github.com/kineticstoolkit/kineticstoolkit) which was used to processin the data.

## Contact

If you have any questions regarding the paper, data, and/or the code, please reach out to the correspoding author of the paper: ...

