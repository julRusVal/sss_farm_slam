# Data Management and Cleanup Guide

## Overview
This repository contains significant amounts of research data that should be managed properly for maintainability and version control efficiency.

## Current Data Distribution

### Large Data Directories (Should be removed from repository)

1. **`processing scripts/data/`** (~827MB)
   - Contains experimental results, CSV files, and image outputs
   - Includes subdirectories for different methods: `iros_method_1`, `iros_method_2`, `iros_method_3`, `icra_2024_*`
   - Contains thousands of SSS images (`.jpg` files)
   - Should be moved to external storage or data repository

2. **`processing scripts/sea_thru_data/`** (~13MB)
   - Sea-thru algorithm test data and results
   - Image processing intermediate files
   - Should be moved to external storage

3. **`testing scripts/`** (Contains CSV files)
   - `trajectory.csv`, `trajectory_gt.csv`, `landmarks.csv`
   - Small test datasets for development
   - Can remain but should be documented

### File Types Found
- **4087+ data files** total including:
  - CSV files (trajectory data, landmarks, analysis results)
  - Image files (JPG, PNG - sonar images and plots)
  - Potentially ROS bag files
  - Log files and processing outputs

## Recommended Cleanup Actions

### 1. Immediate Removal
Remove these large data directories from version control:
```bash
# Add to .gitignore first, then remove
echo "processing scripts/data/" >> .gitignore
echo "processing scripts/sea_thru_data/" >> .gitignore
echo "*.bag" >> .gitignore
echo "*.csv" >> .gitignore
echo "*.jpg" >> .gitignore
echo "*.png" >> .gitignore

# Remove from repository
git rm -r "processing scripts/data/"
git rm -r "processing scripts/sea_thru_data/"
git rm "testing scripts/*.csv"
```

### 2. Create Data Archive
Set up external data storage:
```bash
# Create archive directory outside repository
mkdir ~/sss_farm_slam_data
mv "processing scripts/data/" ~/sss_farm_slam_data/
mv "processing scripts/sea_thru_data/" ~/sss_farm_slam_data/
```

### 3. Update Documentation
- Document where data is stored externally
- Provide instructions for obtaining research datasets
- Create sample/test data that's minimal for CI/testing

### 4. Configure .gitignore
Create comprehensive `.gitignore` to prevent future data commits:
```
# Data directories
processing scripts/data/
processing scripts/sea_thru_data/
testing scripts/data/

# Data files
*.bag
*.csv
*.jpg
*.jpeg
*.png
*.tiff
*.mp4
*.avi

# Logs and temporary files
*.log
*.tmp
*.temp

# Python cache
__pycache__/
*.pyc
*.pyo

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db
```

## Data Management Best Practices

### For Researchers
1. **Store large datasets externally** (university storage, cloud, etc.)
2. **Document data sources** and provide access instructions
3. **Use symbolic links** if local data access is needed
4. **Version control only code**, not data
5. **Provide sample datasets** that are small enough for testing

### For Development
1. **Create mock/test data** that's minimal but representative
2. **Use environment variables** for data paths in code
3. **Make data paths configurable** in launch files
4. **Test with minimal datasets** in CI/CD

### For Publication
1. **Archive datasets separately** with DOIs
2. **Reference external datasets** in papers
3. **Provide data access instructions** in README
4. **Include metadata** about dataset characteristics

## Recovery Instructions

If data needs to be accessed after cleanup:
1. Check git history: `git log --name-only --follow -- "processing scripts/data/"`
2. Restore from backup archive
3. Use git checkout on specific commits if needed
4. Contact maintainer for access to external data storage

## External Data Storage Options

### For Academic Use
- **University storage systems** (KTH Box, Google Drive)
- **Research data repositories** (Zenodo, Figshare)
- **Cloud storage** with academic discounts

### For Long-term Archival
- **Institutional repositories**
- **Data DOI services** (for citation)
- **Backup systems** with versioning