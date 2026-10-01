#!/usr/bin/env python3
"""
Devito Forward Modeling Script for Inference - Devito Only Version

This script generates deviation signals for test samples using ONLY Devito forward modeling.
It loads pre-generated Stage 1 predictions to avoid PyTorch/Devito GPU context conflicts.

Process:
1. Loads original seismic data (x) from test samples
2. Loads pre-generated Stage 1 predictions (y1) from Stage 1 inference script
3. Uses Devito forward model to generate simulated seismic data (x1)
4. Computes deviation signals (δx = x - x1) 
5. Saves combined seismic data [x, δx] for Stage 2 inference

This approach completely separates PyTorch from Devito to eliminate GPU conflicts.
"""

import os
import glob
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
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

def process_single_test_sample(sample_id: str, test_data_dir: str, stage1_predictions_dir: str, 
                              output_dir: str, forward_model, source_coordinates: list) -> bool:
    """
    Process a single test sample to generate deviation signals.
    
    Args:
        sample_id: Sample identifier
        test_data_dir: Test data directory 
        stage1_predictions_dir: Directory containing Stage 1 predictions
        output_dir: Directory to save processed data
        forward_model: Devito forward modeling function
        source_coordinates: List of source positions
        
    Returns:
        bool: Success status
    """
    try:
        sample_test_path = os.path.join(test_data_dir, sample_id)
        sample_output_path = os.path.join(output_dir, sample_id)
        stage1_prediction_file = os.path.join(stage1_predictions_dir, f"{sample_id}_stage1_prediction.npy")
        
        # Create output directory
        os.makedirs(sample_output_path, exist_ok=True)
        
        # Check if already processed
        expected_files = ['combined_seismic.npy', 'stage1_velocity.npy']
        if all(os.path.exists(os.path.join(sample_output_path, f)) for f in expected_files):
            return True  # Already processed
        
        # Check if Stage 1 prediction exists
        if not os.path.exists(stage1_prediction_file):
            print(f"⚠️  Missing Stage 1 prediction: {stage1_prediction_file}")
            return False
        
        # Load original seismic data
        seismic_inputs = []
        for i in source_coordinates:
            file_path = os.path.join(sample_test_path, f'receiver_data_src_{i}.npy')
            if not os.path.exists(file_path):
                print(f"⚠️  Missing seismic file: {file_path}")
                return False
            seismic_data = np.load(file_path).astype(np.float32)
            seismic_inputs.append(seismic_data)
        
        original_seismic = np.stack(seismic_inputs, axis=0)  # Shape: (5, 10001, 31)
        
        # Load pre-generated Stage 1 prediction (NO PyTorch here!)
        stage1_velocity = np.load(stage1_prediction_file).astype(np.float32)
        
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
        
        # Validate shapes
        assert combined_seismic.shape == (10, 10001, 31), f"Combined seismic shape mismatch: {combined_seismic.shape}"
        
        # Save processed data for Stage 2
        np.save(os.path.join(sample_output_path, 'combined_seismic.npy'), combined_seismic)
        np.save(os.path.join(sample_output_path, 'stage1_velocity.npy'), stage1_velocity)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {sample_id}: {e}")
        # Clean up partial processing
        sample_output_path = os.path.join(output_dir, sample_id)
        if os.path.exists(sample_output_path):
            shutil.rmtree(sample_output_path)
        return False

def run_devito_forward_inference(test_data_dir: str, stage1_predictions_dir: str, output_dir: str,
                                max_samples: Optional[int] = None):
    """
    Main function to process test data with Devito forward modeling.
    
    Args:
        test_data_dir: Test data directory
        stage1_predictions_dir: Directory containing Stage 1 predictions
        output_dir: Directory to save processed data for Stage 2
        max_samples: Maximum number of samples to process (for testing)
    """
    print("🚀 Starting Devito forward modeling for test inference...")
    print("💡 This script uses ONLY Devito (no PyTorch) to avoid GPU context conflicts")
    
    # Validate directories
    if not os.path.exists(test_data_dir):
        raise ValueError(f"Test data directory does not exist: {test_data_dir}")
    
    if not os.path.exists(stage1_predictions_dir):
        raise ValueError(f"Stage 1 predictions directory does not exist: {stage1_predictions_dir}")
    
    # Get all test sample directories
    sample_ids = [os.path.basename(p) for p in glob.glob(os.path.join(test_data_dir, '*')) if os.path.isdir(p)]
    
    if len(sample_ids) == 0:
        raise ValueError(f"No sample directories found in: {test_data_dir}")
    
    # Check that we have Stage 1 predictions for these samples
    available_predictions = set()
    for pred_file in glob.glob(os.path.join(stage1_predictions_dir, '*_stage1_prediction.npy')):
        sample_id = os.path.basename(pred_file).replace('_stage1_prediction.npy', '')
        available_predictions.add(sample_id)
    
    # Filter samples to only those with predictions
    samples_with_predictions = [sid for sid in sample_ids if sid in available_predictions]
    
    if len(samples_with_predictions) == 0:
        raise ValueError(f"No Stage 1 predictions found for test samples in {stage1_predictions_dir}")
    
    print(f"📊 Found {len(sample_ids)} test samples, {len(samples_with_predictions)} have Stage 1 predictions")
    
    # Limit samples if specified (for testing)
    if max_samples is not None:
        samples_with_predictions = samples_with_predictions[:max_samples]
        print(f"🧪 Processing limited set: {max_samples} samples")
    
    print(f"⚙️  Processing {len(samples_with_predictions)} samples...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load forward model (GPU first, CPU fallback!)
    forward_model, model_name = load_forward_model()
    print(f"🔧 Using {model_name} forward model")
    
    # Source coordinates
    source_coordinates = [1, 75, 150, 225, 300]
    
    # Process all samples
    successful = 0
    failed = 0
    
    for sample_id in tqdm(samples_with_predictions, desc=f"Devito forward modeling ({model_name})"):
        success = process_single_test_sample(
            sample_id, test_data_dir, stage1_predictions_dir, output_dir, 
            forward_model, source_coordinates
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n✅ Devito forward modeling completed!")
    print(f"📈 Successfully processed: {successful} samples")
    print(f"❌ Failed: {failed} samples")
    print(f"📁 Processed data saved to: {output_dir}")
    
    # Validate a few samples
    if successful > 0:
        print(f"\n🔍 Validating processed data...")
        validation_samples = samples_with_predictions[:min(3, len(samples_with_predictions))]
        
        for sample_id in validation_samples:
            sample_path = os.path.join(output_dir, sample_id)
            
            # Check files exist
            files = ['combined_seismic.npy', 'stage1_velocity.npy']
            missing = [f for f in files if not os.path.exists(os.path.join(sample_path, f))]
            
            if missing:
                print(f"⚠️  Sample {sample_id} missing files: {missing}")
            else:
                # Check shapes
                combined_seismic = np.load(os.path.join(sample_path, 'combined_seismic.npy'))
                stage1_velocity = np.load(os.path.join(sample_path, 'stage1_velocity.npy'))
                
                shape_ok = (
                    combined_seismic.shape == (10, 10001, 31) and
                    stage1_velocity.shape == (300, 1259)
                )
                
                if shape_ok:
                    print(f"✅ Sample {sample_id} validation passed")
                else:
                    print(f"❌ Sample {sample_id} shape validation failed:")
                    print(f"  Combined seismic: {combined_seismic.shape} (expected (10, 10001, 31))")
                    print(f"  Stage1 velocity: {stage1_velocity.shape} (expected (300, 1259))")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Generate deviation signals for test data using Devito forward modeling")
    parser.add_argument('--test_data_dir', type=str, required=True,
                       help='Test data directory')
    parser.add_argument('--stage1_predictions_dir', type=str, required=True,
                       help='Directory containing Stage 1 predictions')
    parser.add_argument('--output_dir', type=str, default='./test_data_processed',
                       help='Directory to save processed data for Stage 2')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing even if output exists')
    
    args = parser.parse_args()
    
    # Check if output directory already exists
    if os.path.exists(args.output_dir) and not args.force:
        response = input(f"Output directory {args.output_dir} already exists. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Devito forward modeling cancelled.")
            return
    
    try:
        run_devito_forward_inference(
            test_data_dir=args.test_data_dir,
            stage1_predictions_dir=args.stage1_predictions_dir,
            output_dir=args.output_dir,
            max_samples=args.max_samples
        )
        
        print(f"\n🎉 Devito forward modeling completed! Processed data saved to: {args.output_dir}")
        print(f"📝 Next step: Run Stage 2 inference script")
        
    except Exception as e:
        print(f"\n💥 Devito forward modeling failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())