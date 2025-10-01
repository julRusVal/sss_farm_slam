#!/usr/bin/env python
"""
Download script for SSS Farm SLAM dataset from Zenodo.
"""

import os
import requests
import zipfile
from pathlib import Path

# Zenodo record information (update after uploading)
ZENODO_RECORD_ID = "17246234"  # Will be provided after upload
ZENODO_DOI = "10.5281/zenodo.YOUR_RECORD_ID"

def download_dataset(data_dir="data"):
    """Download and extract the dataset from Zenodo."""
    
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    # Zenodo API URL for latest version
    api_url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
    
    print(f"Fetching dataset information from Zenodo...")
    response = requests.get(api_url)
    
    if response.status_code != 200:
        print(f"Error: Could not fetch dataset info (status {response.status_code})")
        return False
    
    data = response.json()
    files = data['files']
    
    for file_info in files:
        filename = file_info['key']
        download_url = file_info['links']['self']
        file_size = file_info['size']
        
        file_path = data_path / filename
        
        # Skip if file already exists
        if file_path.exists() and file_path.stat().st_size == file_size:
            print(f"✓ {filename} already exists")
            continue
        
        print(f"Downloading {filename} ({file_size / 1024**2:.1f} MB)...")
        
        # Download with progress
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Extract if it's a zip file
        if filename.endswith('.zip'):
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(data_path)
    
    print("\nDataset download complete!")
    print(f"Data available in: {data_path.absolute()}")
    print(f"\nPlease cite: {ZENODO_DOI}")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download SSS Farm SLAM dataset")
    parser.add_argument("--data-dir", default="data", 
                       help="Directory to download data to (default: data)")
    
    args = parser.parse_args()
    
    success = download_dataset(args.data_dir)
    
    if not success:
        exit(1)