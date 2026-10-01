#!/usr/bin/env python3
"""
Script to create a subset of training data for Stage 2 training.

This script copies 500 random unique samples from the 'train' directory 
to a new 'train_stage2' directory for faster Stage 2 experimentation.
"""

import os

# Repository root (parent of src/); override with the SNS_ROOT environment variable.
PROJECT_ROOT = os.environ.get("SNS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def create_stage2_subset(src_dir: str, dst_dir: str, num_samples: int = 500, seed: int = 42):
    """
    Copy random samples from source train directory to destination directory.
    
    Args:
        src_dir: Source training directory path
        dst_dir: Destination directory path  
        num_samples: Number of samples to copy (default: 500)
        seed: Random seed for reproducibility
    """
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Convert to Path objects
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    # Validate source directory
    if not src_path.exists():
        raise ValueError(f"Source directory does not exist: {src_path}")
    
    # Get all sample directories
    sample_dirs = [d for d in src_path.iterdir() if d.is_dir()]
    
    if len(sample_dirs) == 0:
        raise ValueError(f"No sample directories found in: {src_path}")
    
    print(f"Found {len(sample_dirs)} samples in source directory")
    
    # Check if we have enough samples
    if len(sample_dirs) < num_samples:
        print(f"Warning: Only {len(sample_dirs)} samples available, copying all of them")
        num_samples = len(sample_dirs)
    
    # Randomly select samples
    selected_samples = random.sample(sample_dirs, num_samples)
    
    print(f"Selected {len(selected_samples)} samples for Stage 2 training")
    
    # Create destination directory
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # Copy selected samples
    print(f"Copying samples to: {dst_path}")
    
    for sample_dir in tqdm(selected_samples, desc="Copying samples"):
        dst_sample_dir = dst_path / sample_dir.name
        
        # Skip if already exists
        if dst_sample_dir.exists():
            print(f"Skipping {sample_dir.name} (already exists)")
            continue
            
        # Copy entire sample directory
        try:
            shutil.copytree(sample_dir, dst_sample_dir)
        except Exception as e:
            print(f"Error copying {sample_dir.name}: {e}")
            # Clean up partial copy if it exists
            if dst_sample_dir.exists():
                shutil.rmtree(dst_sample_dir)
    
    # Verify copied samples
    copied_samples = [d for d in dst_path.iterdir() if d.is_dir()]
    print(f"\nSuccessfully copied {len(copied_samples)} samples")
    
    # Validate a few sample directories have expected files
    expected_files = ['vp_model.npy', 'receiver_data_src_1.npy', 'receiver_data_src_75.npy', 
                     'receiver_data_src_150.npy', 'receiver_data_src_225.npy', 'receiver_data_src_300.npy']
    
    validation_samples = random.sample(copied_samples, min(3, len(copied_samples)))
    
    for sample_dir in validation_samples:
        missing_files = []
        for expected_file in expected_files:
            if not (sample_dir / expected_file).exists():
                missing_files.append(expected_file)
        
        if missing_files:
            print(f"Warning: Sample {sample_dir.name} missing files: {missing_files}")
        else:
            print(f"✓ Sample {sample_dir.name} validation passed")


def main():
    """Main function to create Stage 2 subset."""
    
    # Default paths
    base_dir = os.path.join(PROJECT_ROOT, "data")
    src_dir = os.path.join(base_dir, "train")
    dst_dir = os.path.join(base_dir, "train_stage2")
    
    # Configuration
    num_samples = 2000
    seed = 42
    
    print("=== Stage 2 Training Subset Creation ===")
    print(f"Source directory: {src_dir}")
    print(f"Destination directory: {dst_dir}")
    print(f"Number of samples: {num_samples}")
    print(f"Random seed: {seed}")
    print()
    
    try:
        create_stage2_subset(src_dir, dst_dir, num_samples, seed)
        print("\n✅ Stage 2 subset creation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error creating Stage 2 subset: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
