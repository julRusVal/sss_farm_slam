# SSS Farm SLAM

A ROS package for Simultaneous Localization and Mapping (SLAM) using Side-Scan Sonar (SSS) in algae farm environments.

**Paper**: [Side Scan Sonar-based SLAM for Autonomous Algae Farm Monitoring](https://arxiv.org/abs/2509.26121)

## Overview

This project implements a SLAM system specifically designed for underwater algae farm mapping using side-scan sonar data. The system combines odometry data with sonar detections to build maps of underwater environments while simultaneously tracking the vehicle's position.

## Demonstration

### Real-time SLAM Operation

![SLAM Demo](assets/method.gif)

*Real-time underwater SLAM demonstration showing vehicle trajectory estimation and landmark detection in an algae farm environment*

### Sample Results
<!-- Add result images here when available -->
<!-- ![Trajectory Results](assets/trajectory_result.png) -->
<!-- ![Map Output](assets/map_result.png) -->

## Features

- **Side-Scan Sonar Processing**: Real-time processing of SSS data for landmark detection
- **GTSAM Integration**: Uses Georgia Tech Smoothing and Mapping library for factor graph optimization
- **Multiple Detection Methods**: Supports both manual and automated buoy/landmark detection
- **Data Association**: Intelligent association of detections with known landmarks
- **Visualization**: Real-time visualization using RViz
- **Multiple Operating Modes**: Supports both simulated and real-world data processing

## Project Structure

```
sss_farm_slam/
├── CMakeLists.txt              # CMake build configuration
├── package.xml                 # ROS package configuration
├── setup.py                    # Python package setup
├── README.md                   # This file
├── launch/                     # ROS launch files
│   ├── run_sam_slam_real.launch
│   ├── pipeline_sim.launch
│   └── ...
├── scripts/                    # Executable ROS nodes
│   ├── sam_listener_online_slam_node.py
│   ├── sam_listener_saver_node.py
│   └── ...
├── src/sam_slam_utils/         # Core Python modules
│   ├── sam_slam_mapping.py     # Main mapping algorithms
│   ├── sam_slam_helpers.py     # Utility functions
│   ├── sam_slam_proc_classes.py # Processing classes
│   └── sam_slam_ros_classes.py # ROS interface classes
├── processing scripts/         # Data analysis and visualization
│   ├── plot_*.py              # Various plotting utilities
│   ├── process_*.py           # Data processing scripts
│   └── data/                  # Output data directory
├── testing scripts/           # Development and testing scripts
├── rviz/                      # RViz configuration files
└── [data directories]         # Raw data (should be excluded from repo)
```

## Dependencies

### ROS Dependencies
- `roscpp`
- `rospy` 
- `std_msgs`
- `cv_bridge`
- `geometry_msgs`
- `nav_msgs`
- `sensor_msgs`

### Python Dependencies
- `numpy`
- `matplotlib`
- `opencv-python`
- `gtsam` (Georgia Tech Smoothing and Mapping)
- `scikit-image`
- `mayavi` (for 3D visualization)
- `PIL` (Pillow)

### System Dependencies
- ROS Melodic/Noetic
- Python 3.6+
- OpenCV 4.x

## Installation

1. **Clone the repository** into your catkin workspace:
   ```bash
   cd ~/catkin_ws/src
   git clone <repository-url> sss_farm_slam
   ```

2. **Install Python dependencies**:
   ```bash
   pip install numpy matplotlib opencv-python scikit-image pillow
   pip install gtsam  # or build from source for better performance
   ```

3. **Build the package**:
   ```bash
   cd ~/catkin_ws
   catkin_make
   source devel/setup.bash
   ```

## Usage

### Real-Time SLAM with Recorded Data

To run SLAM with real sonar data:

```bash
roslaunch sam_slam run_sam_slam_real.launch
```

This launch file:
- Starts the SLAM node
- Plays back recorded ROS bag data
- Launches RViz for visualization
- Starts supporting nodes for GPS and detection

### Simulation Mode

For testing with simulated data:

```bash
roslaunch sam_slam pipeline_sim.launch
```

### Processing Recorded Data

To process and analyze data offline:

```bash
rosrun sam_slam sam_listener_saver_node.py
```

## Configuration

Key parameters can be configured in the launch files:

### Detection Parameters
- `detect_update_time`: Frequency of detection processing (Hz)
- `da_distance_threshold`: Distance threshold for data association
- `manual_associations`: Enable manual landmark association
- `simulated_detections`: Use simulated vs real detections

### SLAM Parameters
- `prior_ang_sig_deg`: Prior angle uncertainty (degrees)
- `prior_dist_sig`: Prior position uncertainty (meters)
- `odo_ang_sig_deg`: Odometry angle noise (degrees)
- `odo_dist_sig`: Odometry distance noise (meters)
- `detect_ang_sig_deg`: Detection angle uncertainty (degrees)
- `detect_dist_sig`: Detection distance uncertainty (meters)

### Output Parameters
- `path_name`: Directory for saving output data
- `verbose_*`: Enable verbose output for different components

## Core Components

### SLAM Node (`sam_listener_online_slam_node.py`)
Main SLAM processing node that:
- Subscribes to odometry and detection topics
- Maintains factor graph using GTSAM
- Publishes estimated trajectory and landmarks
- Saves results to CSV files

### Mapping Module (`sam_slam_mapping.py`)
Contains core algorithms for:
- Image registration and processing
- Sea-thru underwater image enhancement
- 3D point cloud processing
- Landmark detection and tracking

### Processing Classes (`sam_slam_proc_classes.py`)
Defines data structures for:
- Pose management
- Detection handling
- Graph optimization
- Data association

### Visualization Scripts
Multiple scripts for analyzing and plotting results:
- `plot_ate_error_*.py`: Absolute Trajectory Error analysis
- `plot_online_*.py`: Real-time performance metrics
- `plot_pipeline_map.py`: Map visualization

## Data Formats

### Input Data
- **Odometry**: `nav_msgs/Odometry` messages
- **Detections**: Custom detection messages with range/bearing
- **Images**: `sensor_msgs/Image` for sonar data
- **Point Clouds**: `sensor_msgs/PointCloud2` for 3D data

### Output Data
- **Trajectories**: CSV files with pose estimates
- **Landmarks**: CSV files with landmark positions
- **Performance Metrics**: Analysis of accuracy and consistency
- **Maps**: Visualization images and 3D models

## Research Context

This work is part of a degree project focusing on underwater SLAM in structured environments like algae farms. The system addresses challenges specific to underwater robotics:

- **Limited visibility**: Using sonar instead of visual sensors
- **Structured environments**: Leveraging regular patterns in algae farms
- **Data association**: Robust matching of detections to landmarks
- **Uncertainty quantification**: Proper noise modeling for underwater sensors

## Publication References

Methods implemented are based on research presented at:
- ICRA 2024: Baseline methods and proposed improvements
- IROS: Multiple method comparisons and analysis

## Troubleshooting

### Common Issues

1. **GTSAM Import Error**:
   - Ensure GTSAM is properly installed with Python bindings
   - Try building GTSAM from source if pip installation fails

2. **ROS Topic Issues**:
   - Check that all required topics are being published
   - Verify topic names match between publishers and subscribers

3. **Visualization Problems**:
   - Ensure RViz configuration file exists
   - Check that all visualization messages are being published

4. **Performance Issues**:
   - Reduce update frequencies in launch files
   - Consider using optimized GTSAM build
   - Monitor CPU/memory usage during operation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request with clear description

## License

This project is licensed under the BSD License - see the LICENSE file for details.

## Contact

- **Maintainer**: Julian Valdez (jvaldez@kth.se)
- **Institution**: KTH Royal Institute of Technology
- **Project**: SSS Farm SLAM for Algae Farm Mapping

## Acknowledgments

### Research Collaborators
- **Dr. John Folkesson** - KTH Royal Institute of Technology (Supervisor)
- **Dr. Ignacio Torroba** - KTH Royal Institute of Technology (Advisor)

### Industry Support
- **Ocean Infinity** - PhD student funding and industry partnership

### Technical Acknowledgments
- Georgia Tech for the GTSAM library
- ROS community for the robotics framework
