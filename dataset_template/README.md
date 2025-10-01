# SSS Farm SLAM Dataset

## Overview
This dataset contains side-scan sonar data collected for SLAM research in underwater algae farm environments. The data supports the research presented in "Side Scan Sonar-based SLAM for Autonomous Algae Farm Monitoring" (https://arxiv.org/abs/2509.26121).

## Citation
If you use this dataset in your research, please cite:

```bibtex
@misc{valdez2025scansonarbasedslamautonomous,
      title={Side Scan Sonar-based SLAM for Autonomous Algae Farm Monitoring}, 
      author={Julian Valdez and Ignacio Torroba and John Folkesson and Ivan Stenius},
      year={2025},
      eprint={2509.26121},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.26121}, 
}
```

## Dataset Description

### Data Collection
- **Location**: [Specify location if appropriate]
- **Date**: [Collection dates]
- **Vehicle**: SAM AUV (Autonomous Underwater Vehicle)
- **Sensors**: Side-scan sonar, IMU, GPS, depth sensor
- **Environment**: Structured algae farm with rope lines and buoys

### File Structure
```
├── rosbags/                    # ROS bag files with sensor data
├── ground_truth/              # Ground truth data for evaluation
├── detection_results/         # Processed detection files
└── documentation/             # Additional documentation
```

### Data Formats
- **ROS Bags**: Standard ROS bag format containing sensor messages
- **CSV Files**: Comma-separated values for detection results and ground truth
- **Coordinate System**: [Specify coordinate system used]

## File Descriptions

### ROS Bag Files
- `sam_real_algae_*.bag`: Real underwater data collection runs
- `sim_*.bag`: Simulated data for algorithm development

### CSV Files
- `image_process_buoys.csv`: Buoy detection results with timestamps and positions
- `image_process_ropes_*.csv`: Rope detection results for port/starboard sonar
- `ground_truth_trajectory.csv`: Reference trajectory data

## Usage Examples

### Loading ROS Bag Data
```bash
# Play back data at normal speed
rosbag play sam_real_algae_1.bag

# Play at reduced speed for processing
rosbag play -r 0.5 sam_real_algae_1.bag
```

### Processing Detection Data
```python
import pandas as pd
import numpy as np

# Load detection results
detections = pd.read_csv('detection_results/image_process_buoys.csv')
print(f"Total detections: {len(detections)}")
```

## Data Quality and Validation
- All bag files have been validated for completeness
- Detection results manually verified against sonar imagery
- Ground truth collected using differential GPS when surfaced

## License
This dataset is released under Creative Commons Attribution 4.0 International (CC BY 4.0).
You are free to share and adapt the data with appropriate attribution.

## Contact
- **Author**: Julian Valdez (jvaldez@kth.se)
- **Institution**: KTH Royal Institute of Technology
- **Project Page**: [Link to GitHub repository]

## Acknowledgments
- Ocean Infinity for PhD student funding
- KTH Royal Institute of Technology
- Research collaborators