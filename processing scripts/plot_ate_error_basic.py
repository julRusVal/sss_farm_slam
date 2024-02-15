import trajectory_analysis
import os

#
offset = 0.0
scale = 1.0
max_difference = 0.0001
verbose = True
plot = True
data_set_directory = "/Users/julian/KTH/Degree project/sam_slam/processing scripts/data/iros_method_2"


if not os.path.isdir(data_set_directory):
    raise ValueError("Invalid data set directory")

first_file_path = data_set_directory + "/analysis_final_3d.csv"
second_file_path = data_set_directory + "/analysis_dr_3d.csv"

ate_analysis = trajectory_analysis.TrajectoryAnalysis(first_file_path,
                                                      second_file_path,
                                                      output_directory_path=data_set_directory,
                                                      verbose=verbose)

if plot:
    print("plotting")
    ate_analysis.plot_trajectories()