# Data Directory

This directory contains the datasets used for SSS Farm SLAM experiments.

## Structure

```
data/
├── rosbags/           # ROS bag files (tracked with Git LFS)
│   ├── real/          # Real underwater farm data
│   ├── simulation/    # Simulated data
│   └── samples/       # Small sample files for testing
├── detections/        # CSV files with detection results
├── ground_truth/      # Ground truth data for evaluation
└── processed/         # Processed/intermediate data files
```

## File Types and Storage

- **Small files** (< 100MB): CSV, configs, small images → Regular Git
- **Large files** (> 100MB): ROS bags, videos, large datasets → Git LFS
- **Huge datasets** (> 1GB): Consider external repositories (Zenodo, etc.)

## Usage

### Downloading ROS Bag Files from Zenodo

The complete dataset including large ROS bag files is available on Zenodo:

**Dataset**:[SSS Farm SLAM Dataset](https://zenodo.org/record/17246234) 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17246234.svg)](https://doi.org/10.5281/zenodo.17246234)

#### Option 1: Automated Download Script
```bash
# Run the download script from the repository root
python scripts/download_dataset.py --data-dir data/

# Or specify a custom directory
python scripts/download_dataset.py --data-dir /path/to/your/data/
```

#### Option 2: Manual Download from Zenodo
1. Visit the [Zenodo dataset page](https://zenodo.org/record/YOUR_RECORD_ID)
2. Download the dataset archive (usually `sss_farm_slam_dataset.zip`)
3. Extract to your data directory:
   ```bash
   cd data/
   unzip sss_farm_slam_dataset.zip
   ```

### Using Downloaded ROS Bag Files

Once downloaded, you can use the bag files with the launch files:

```bash
# Use with real-world launch files
roslaunch sam_slam run_sam_slam_real.launch bag_file:=data/rosbags/real/sam_real_algae_1.bag

# Use with experiment launch files
roslaunch sam_slam experiments/iros_method_1.launch bag_file:=data/rosbags/real/sam_real_algae_2.bag

# List available bag files
ls data/rosbags/real/*.bag
```

### Verifying Downloaded Data

```bash
# Check bag file info
rosbag info data/rosbags/real/sam_real_algae_1.bag

# Verify file integrity (if checksums provided)
md5sum data/rosbags/real/*.bag
```

### Adding New Data Files

```bash
# For regular files
git add data/detections/new_results.csv

# For large files (automatically handled if tracked)
git add data/rosbags/new_experiment.bag
git commit -m "Add new experiment data"
git push
```

### Repository Clone with Data

```bash
# Clone repository 
git clone https://github.com/julRusVal/sss_farm_slam.git
cd sss_farm_slam

# Download the dataset
python scripts/download_dataset.py

# Pull any Git LFS files (for sample data)
git lfs pull
```

## Data Sources

- Real data collected at [Location/Date]
- Simulation data generated using [Simulator Name]
- Ground truth obtained via [Method]

## Citation

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