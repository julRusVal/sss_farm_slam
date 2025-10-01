# Data Management

## Dataset Availability

The full dataset used in this research is available on Zenodo:

**📊 Dataset**: [SSS Farm SLAM Dataset](https://zenodo.org/record/17246234) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17246234.svg)](https://doi.org/10.5281/zenodo.17246234)

## Quick Start with Data

### Option 1: Download via Script
```bash
cd sss_farm_slam
python scripts/download_dataset.py --data-dir data/
```

### Option 2: Manual Download
1. Visit the [Zenodo dataset page](https://zenodo.org/record/YOUR_RECORD_ID)
2. Download the dataset archive
3. Extract to `data/` directory in your workspace

### Sample Data
Small sample files are included in this repository for testing:
```bash
# Use sample data for quick testing
roslaunch sam_slam run_sam_slam_real.launch bag_file:=data/samples/sample_run.bag
```

## Dataset Contents

- **Real underwater data**: 15+ hours of algae farm surveys
- **Simulation data**: Controlled test scenarios
- **Ground truth**: GPS waypoints and manual annotations
- **Detection results**: Processed buoy and rope detections

## Citation

If you use this dataset, please cite both the paper and the dataset:

```bibtex
@article{valdez2024sss,
  title={Side Scan Sonar-based SLAM for Autonomous Algae Farm Monitoring},
  author={Valdez, Julian and Folkesson, John},
  journal={arXiv preprint arXiv:2509.26121},
  year={2024}
}

@dataset{valdez2024dataset,
  author={Valdez, Julian and Folkesson, John},
  title={SSS Farm SLAM Dataset for Underwater Algae Farm Monitoring},
  year={2024},
  publisher={Zenodo},
  doi={10.5281/zenodo.YOUR_RECORD_ID}
}
```