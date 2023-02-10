# sam_slam
Work towards degree project, using sidescan sonar in an algae farm environment for SLAM.

# Status
Currently subscribes the odometry topic filtered and simulated and the simulated detection.
The data is saved to CSVs. Only coordinates, in the map frame, are saved for now. 

# Plans
Working to expand the simulatin script in the other repo to process this data. Also need to do more processing on the
data to extract range and bearing from the detections.

# Notes on running
I'm not sure if I structured this correctly to be a ROS package or if the git is set up correctly. 
