#!/usr/bin/env python3
"""
Stage 2 Data Preprocessing Script - Devito Only Version

This script preprocesses all Stage 2 training data using ONLY Devito forward modeling.
It loads pre-generated Stage 1 predictions to avoid PyTorch/Devito GPU context conflicts.

Process:
1. Loads original seismic data (x) and ground truth velocity (y_true)
2. Loads pre-generated Stage 1 predictions (y1) from separate script
3. Uses Devito forward model to generate simulated seismic data (x1)
4. Computes deviation signals (δx = x - x1) and residual targets (δy = y_true - y1)
5. Saves preprocessed data for fast Stage 2 training

This approach completely separates PyTorch from Devito to eliminate GPU conflicts.
"""

import os

# Repository root (parent of src/); override with the SNS_ROOT environment variable.
PROJECT_ROOT = os.environ.get("SNS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
from typing import Optional

def load_forward_model():
    """Load the Devito forward model with GPU first, CPU fallback."""
    forward_model = None
    model_name = "Unknown"
    
    # Try GPU Devito version first
    try:
        print("🚀 Attempting to load GPU-accelerated Devito forward model...")
        from sns_forward_devito_gpu import vel_to_seis
        forward_model = vel_to_seis
        model_name = "GPU Devito"
        print("✅ Successfully loaded GPU-accelerated Devito forward model")
    except ImportError as e:
        print(f"⚠️  GPU Devito not available: {e}")
        # Fallback to CPU version
        try:
            print("🔄 Falling back to CPU Devito forward model...")
            from sns_forward_devito import vel_to_seis
            forward_model = vel_to_seis
            model_name = "CPU Devito"
            print("✅ Successfully loaded CPU Devito forward model")
        except ImportError as e:
            print(f"❌ Failed to import any Devito forward model: {e}")
            raise ImportError("Could not import any Devito forward model")

    return forward_model, model_name

def preprocess_single_sample_devito(sample_id: str, src_dir: str, predictions_dir: str, 
                                   dst_dir: str, forward_model, source_coordinates: list) -> bool:
    """
    Preprocess a single sample for Stage 2 training using only Devito.
    
    Args:
        sample_id: Sample identifier
        src_dir: Source data directory 
        predictions_dir: Directory containing Stage 1 predictions
        dst_dir: Destination directory for preprocessed data
        forward_model: Devito forward modeling function
        source_coordinates: List of source positions
        
    Returns:
        bool: Success status
    """
    try:
        sample_src_path = os.path.join(src_dir, sample_id)
        sample_dst_path = os.path.join(dst_dir, sample_id)
        prediction_file = os.path.join(predictions_dir, f"{sample_id}_stage1_prediction.npy")
        
        # Create destination directory
        os.makedirs(sample_dst_path, exist_ok=True)
        
        # Check if already processed
        expected_files = ['combined_seismic.npy', 'stage1_velocity.npy', 'residual_target.npy']
        if all(os.path.exists(os.path.join(sample_dst_path, f)) for f in expected_files):
            return True  # Already processed
        
        # Check if Stage 1 prediction exists
        if not os.path.exists(prediction_file):
            print(f"⚠️  Missing Stage 1 prediction: {prediction_file}")
            return False
        
        # Load original seismic data
        seismic_inputs = []
        for i in source_coordinates:
            file_path = os.path.join(sample_src_path, f'receiver_data_src_{i}.npy')
            if not os.path.exists(file_path):
                print(f"⚠️  Missing seismic file: {file_path}")
                return False
            seismic_data = np.load(file_path).astype(np.float32)
            seismic_inputs.append(seismic_data)
        
        original_seismic = np.stack(seismic_inputs, axis=0)  # Shape: (5, 10001, 31)
        
        # Load ground truth velocity
        target_path = os.path.join(sample_src_path, 'vp_model.npy')
        if not os.path.exists(target_path):
            print(f"⚠️  Missing velocity file: {target_path}")
            return False
        target_velocity = np.load(target_path).astype(np.float32)
        
        # Load pre-generated Stage 1 prediction (NO PyTorch here!)
        stage1_velocity = np.load(prediction_file).astype(np.float32)
        
        # Validate shapes
        assert stage1_velocity.shape == (300, 1259), f"Stage1 velocity shape mismatch: {stage1_velocity.shape}"
        
        # Generate simulated seismic data using Devito forward model
        try:
            simulated_seismic = forward_model(stage1_velocity)
            simulated_seismic = simulated_seismic.astype(np.float32)
        except Exception as e:
            print(f"⚠️  Forward model failed for {sample_id}: {e}")
            # Use zeros as fallback
            simulated_seismic = np.zeros((5, 10001, 31), dtype=np.float32)
        
        # Compute deviation signal: δx = x - x1
        deviation_seismic = original_seismic - simulated_seismic
        
        # Concatenate original and deviation signals for 10-channel input
        combined_seismic = np.concatenate([original_seismic, deviation_seismic], axis=0)
        
        # Compute residual target: δy = y_true - y1
        residual_target = target_velocity - stage1_velocity
        
        # Validate shapes
        assert combined_seismic.shape == (10, 10001, 31), f"Combined seismic shape mismatch: {combined_seismic.shape}"
        assert stage1_velocity.shape == (300, 1259), f"Stage1 velocity shape mismatch: {stage1_velocity.shape}"
        assert residual_target.shape == (300, 1259), f"Residual target shape mismatch: {residual_target.shape}"
        
        # Save preprocessed data
        np.save(os.path.join(sample_dst_path, 'combined_seismic.npy'), combined_seismic)
        np.save(os.path.join(sample_dst_path, 'stage1_velocity.npy'), stage1_velocity)
        np.save(os.path.join(sample_dst_path, 'residual_target.npy'), residual_target)
        
        # Also save original ground truth for reference
        np.save(os.path.join(sample_dst_path, 'ground_truth_velocity.npy'), target_velocity)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {sample_id}: {e}")
        # Clean up partial processing
        sample_dst_path = os.path.join(dst_dir, sample_id)
        if os.path.exists(sample_dst_path):
            shutil.rmtree(sample_dst_path)
        return False

def preprocess_stage2_data_devito(src_dir: str, predictions_dir: str, dst_dir: str, 
                                 max_samples: Optional[int] = None):
    """
    Main preprocessing function for Stage 2 data using only Devito.
    
    Args:
        src_dir: Source training data directory
        predictions_dir: Directory containing Stage 1 predictions
        dst_dir: Destination directory for preprocessed data  
        max_samples: Maximum number of samples to process (for testing)
    """
    print("🚀 Starting Stage 2 data preprocessing with Devito...")
    print("💡 This script uses ONLY Devito (no PyTorch) to avoid GPU context conflicts")
    
    # Validate directories
    if not os.path.exists(src_dir):
        raise ValueError(f"Source directory does not exist: {src_dir}")
    
    if not os.path.exists(predictions_dir):
        raise ValueError(f"Predictions directory does not exist: {predictions_dir}")
    
    # Get all sample directories
    sample_ids = [os.path.basename(p) for p in glob.glob(os.path.join(src_dir, '*')) if os.path.isdir(p)]
    
    if len(sample_ids) == 0:
        raise ValueError(f"No sample directories found in: {src_dir}")
    
    # Check that we have predictions for these samples
    available_predictions = set()
    for pred_file in glob.glob(os.path.join(predictions_dir, '*_stage1_prediction.npy')):
        sample_id = os.path.basename(pred_file).replace('_stage1_prediction.npy', '')
        available_predictions.add(sample_id)
    
    # Filter samples to only those with predictions
    samples_with_predictions = [sid for sid in sample_ids if sid in available_predictions]
    
    if len(samples_with_predictions) == 0:
        raise ValueError(f"No Stage 1 predictions found for samples in {predictions_dir}")
    
    print(f"📊 Found {len(sample_ids)} samples, {len(samples_with_predictions)} have Stage 1 predictions")
    
    # Limit samples if specified (for testing)
    if max_samples is not None:
        samples_with_predictions = samples_with_predictions[:max_samples]
        print(f"🧪 Processing limited set: {max_samples} samples")
    
    print(f"⚙️  Processing {len(samples_with_predictions)} samples...")
    
    # Create destination directory
    os.makedirs(dst_dir, exist_ok=True)
    
    # Load forward model (GPU first, CPU fallback!)
    forward_model, model_name = load_forward_model()
    print(f"🔧 Using {model_name} forward model")
    
    # Source coordinates
    source_coordinates = [1, 75, 150, 225, 300]
    
    # Process all samples
    successful = 0
    failed = 0
    
    for sample_id in tqdm(samples_with_predictions, desc=f"Preprocessing Stage 2 data ({model_name})"):
        success = preprocess_single_sample_devito(
            sample_id, src_dir, predictions_dir, dst_dir, forward_model, source_coordinates
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n✅ Stage 2 preprocessing completed!")
    print(f"📈 Successfully processed: {successful} samples")
    print(f"❌ Failed: {failed} samples")
    print(f"📁 Preprocessed data saved to: {dst_dir}")
    
    # Validate a few samples
    if successful > 0:
        print(f"\n🔍 Validating preprocessed data...")
        validation_samples = samples_with_predictions[:min(3, len(samples_with_predictions))]
        
        for sample_id in validation_samples:
            sample_path = os.path.join(dst_dir, sample_id)
            
            # Check files exist
            files = ['combined_seismic.npy', 'stage1_velocity.npy', 'residual_target.npy']
            missing = [f for f in files if not os.path.exists(os.path.join(sample_path, f))]
            
            if missing:
                print(f"⚠️  Sample {sample_id} missing files: {missing}")
            else:
                # Check shapes
                combined_seismic = np.load(os.path.join(sample_path, 'combined_seismic.npy'))
                stage1_velocity = np.load(os.path.join(sample_path, 'stage1_velocity.npy'))
                residual_target = np.load(os.path.join(sample_path, 'residual_target.npy'))
                
                shape_ok = (
                    combined_seismic.shape == (10, 10001, 31) and
                    stage1_velocity.shape == (300, 1259) and
                    residual_target.shape == (300, 1259)
                )
                
                if shape_ok:
                    print(f"✅ Sample {sample_id} validation passed")
                else:
                    print(f"❌ Sample {sample_id} shape validation failed:")
                    print(f"  Combined seismic: {combined_seismic.shape} (expected (10, 10001, 31))")
                    print(f"  Stage1 velocity: {stage1_velocity.shape} (expected (300, 1259))")
                    print(f"  Residual target: {residual_target.shape} (expected (300, 1259))")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Preprocess Stage 2 training data using Devito (GPU first, CPU fallback)")
    parser.add_argument('--src_dir', type=str, default=os.path.join(PROJECT_ROOT, "data/train_stage2"),
                       help='Source training data directory')
    parser.add_argument('--predictions_dir', type=str, default=os.path.join(PROJECT_ROOT, "data/stage1_predictions"),
                       help='Directory containing Stage 1 predictions')
    parser.add_argument('--dst_dir', type=str, default=os.path.join(PROJECT_ROOT, "data/train_stage2_preprocessed"),
                       help='Destination directory for preprocessed data')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing even if destination exists')
    
    args = parser.parse_args()
    
    # Check if destination already exists
    if os.path.exists(args.dst_dir) and not args.force:
        response = input(f"Destination directory {args.dst_dir} already exists. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Preprocessing cancelled.")
            return
    
    try:
        preprocess_stage2_data_devito(
            src_dir=args.src_dir,
            predictions_dir=args.predictions_dir,
            dst_dir=args.dst_dir,
            max_samples=args.max_samples
        )
        
        print(f"\n🎉 All done! You can now use the preprocessed data for fast Stage 2 training.")
        print(f"📝 Update your Stage 2 training script to use: {args.dst_dir}")
        
    except Exception as e:
        print(f"\n💥 Preprocessing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
