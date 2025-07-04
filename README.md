# Markerless Mocap Validation for Team Sports

Welcome to the github repository related to the paper "Examining the Concurrent Validity of Markerless Motion Capture in Dual-Athlete Team Sports Movements". This repository consitsts of the code used to analyse the data, and generate the results of the paper. The paper is open access available [here](https://www.tandfonline.com/doi/full/10.1080/02640414.2025.2497678).

> [!IMPORTANT]
> If you use this repository, the dataset, or the paper associated to it, please refer to it accordingly:
> ```shell
> @article{oonk2025markerless,
>   title     = {Examining the concurrent validity of markerless motion capture in dual-athlete team sports movements},
>   author    = {Oonk, G. A. and Kempe, M. and Lemmink, K. A. P. M. and Buurke, T. J. W.},
>   journal   = {Journal of Sports Sciences},
>   year      = {2025},
>   publisher = {Taylor & Francis},
>   doi       = {10.1080/02640414.2025.2497678},
>   url       = {https://doi.org/10.1080/02640414.2025.2497678}
> }
> ```


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

The data is available [here](https://doi.org/10.34894/LZPY3B). You can download both the raw .c3d files as well as the processed data, which are in .parquet format.

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

> [!IMPORTANT]
> For creating a complete dataset, it is assumed that in the theia files are in a in a folder `theia_c3d_files`, the vicon files in a folder `vicon_c3d_files` in `--input_dir`. This can be changed in in `scripts/constants`. The naming of the files should be as follows: (1) participant/group, (2) exercise name (3) trial number. So for Vicon a valid file could be `"both1 double sidestep 001.c3d"` or `"pp2 walk rotation 003.c3d"`. For Theia a suffix is added. For the first person `pose_filt_0` is added, while for the second `pose_filt_1` is added. So the corresponding similar files for Theia are `"both1 double sidestep 001 pose_filt_0.c3d"`, `"both1 double sidestep 001 pose_filt_1.c3d"`, and `"pp2 walk rotation 003 pose_filt_0.c3d"`. Also, a corresponding `.txt` file should be added with the center of mass data: `"pp2 walk rotation 003 pose_filt_0 CoM.txt"`.

### Generate statistics
To generate all the statistics (sTEE, r, RMSD, and BA CI) for every measured timeseries (segment orientations, joint angles, center of mass) into an excel table, you can run the following command:

```bash
python3 scripts/generate_statistics.py
```
By default, the follwing arguments are used:
```bash
--input_dir data/preprocessed_dataset
--exercise all
--preprocessing_file_loc preprocessing_df.csv
```
This command will save the requested statistical results in an excel file in your current working directory.

The `input_dir` refers to the directory with the preprocessed dataset. The `exercise` argument is used to select the exercise you want to generate the statistics for. The value can be "all", "double sidestep", "dribble simulation", "high five", "sidestep", "slow dribble", "pass", "shot", and "walk rotation", if any trials of that are found in the `input_dir` directory. The `preprocessing_file_loc` argument is used to load the preprocessing information, which is used to normalize the statistics for the special side of an exercise (e.g. the hand used for the high five (left or right) can be different for every couple).

> [!IMPORTANT]
> For creating the statistics, the `input_dir` should contain folders for every individual participant (`pp1`, `pp2`, ..., `pp24`). Every participant folder, should contain 6 parquet files per movement trial saved in the following format: (1) exercise name, (2) trial number, (3) data type, and (4) data provider. For instance, you could have `"pp1/double sidestep 001 angles theia.parquet"`, `"pp11/slow dribble 003 points vicon.parquet"`, etc.

### Visualize movement

To better grasp the data, we have added a script to visualize the 3D movements of the athletes. To run the script, you can use the following command:

```bash
python3 scripts/visualize_movement.py --theia_c3d "data/theia_c3d_files/both11 double sidestep 002 pose_filt_1.c3d" --vicon_c3d "data/vicon_c3d_files/both11 double sidestep 002.c3d" --vicon_person pp22
```


https://github.com/user-attachments/assets/ea756bf3-c55f-44f1-b296-68362f4e75c6


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
![Example TimeSeries](https://github.com/user-attachments/assets/65b6fce3-3158-4165-a33a-f331957fd555)

`exercise_name` should be one of the following: "double_sidestep", "dribble_simulation", "high_five", "sidestep", "slow_dribble", "pass", "shot", "walk_rotation". `particpant_number` should be an integer between 1 and 24. `trial_number` should be an integer between 1 and 6. `data_type` should be one of the following: "segment_angles", "joint_angles", "points". `data_name` should be the name of the segments, joint angle, or points you want to visualize.

## Reference

If you use the code in this repository, or the data associated with the paper, please cite the following paper:
```
@article{oonk2025markerless,
  title     = {Examining the concurrent validity of markerless motion capture in dual-athlete team sports movements},
  author    = {Oonk, G. A. and Kempe, M. and Lemmink, K. A. P. M. and Buurke, T. J. W.},
  journal   = {Journal of Sports Sciences},
  year      = {2025},
  publisher = {Taylor & Francis},
  doi       = {10.1080/02640414.2025.2497678},
  url       = {https://doi.org/10.1080/02640414.2025.2497678}
}
```

## License

The code of this project is licensed under the CCBY License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

We would like to thank [Tamar](https://github.com/TFoppen) and [Dave](https://github.com/DaveHanegraaf) for their help in collecting the data for this project. Furthermore, we would like to thank [Felix](https://github.com/felixchenier) for creating and open sourcing the [kineticstoolkit](https://github.com/kineticstoolkit/kineticstoolkit) which was used to processing the data.

## Contact

If you have any questions regarding the data and/or the code, please open an issue on GitHub.

