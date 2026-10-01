#!/usr/bin/env python3
"""
Stage 1 Prediction Generation Script

This script generates Stage 1 velocity predictions for all training samples using PyTorch.
It runs completely separately from Devito to avoid GPU context conflicts.

The predictions are saved as .npy files that can be loaded by the Devito preprocessing script.
"""

import os

# Repository root (parent of src/); override with the SNS_ROOT environment variable.
PROJECT_ROOT = os.environ.get("SNS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import argparse
from typing import Optional

# Import required components
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_stage1_model(model_path: str, device: torch.device):
    """Load the trained Stage 1 model."""
    try:
        from sns_caformer_enhanced_notebook import MultiTaskSnsCaformerModel
        
        model = MultiTaskSnsCaformerModel(
            backbone_name="caformer_b36.sail_in22k_ft_in1k",
            pretrained=False,
            in_channels_stem=5,
            out_channels_stem=128,
            use_multi_task=True,
            use_tta=False  # Disable TTA for consistency in preprocessing
        ).to(device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"✅ Successfully loaded Stage 1 model from {model_path}")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load Stage 1 model: {e}")
        raise

def generate_stage1_prediction_single_sample(sample_id: str, src_dir: str, predictions_dir: str, 
                                           stage1_model, device: torch.device, 
                                           source_coordinates: list) -> bool:
    """
    Generate Stage 1 prediction for a single sample.
    
    Args:
        sample_id: Sample identifier
        src_dir: Source data directory 
        predictions_dir: Directory to save predictions
        stage1_model: Loaded Stage 1 model
        device: Computing device
        source_coordinates: List of source positions
        
    Returns:
        bool: Success status
    """
    try:
        sample_src_path = os.path.join(src_dir, sample_id)
        prediction_file = os.path.join(predictions_dir, f"{sample_id}_stage1_prediction.npy")
        
        # Check if already processed
        if os.path.exists(prediction_file):
            return True  # Already processed
        
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
        
        # Generate Stage 1 prediction
        with torch.no_grad():
            seismic_tensor = torch.from_numpy(original_seismic).unsqueeze(0).to(device)
            velocity_pred = stage1_model(seismic_tensor, use_tta=False)
            stage1_velocity = velocity_pred.squeeze(0).squeeze(0).cpu().numpy()
        
        # Validate shape
        assert stage1_velocity.shape == (300, 1259), f"Stage1 velocity shape mismatch: {stage1_velocity.shape}"
        
        # Save prediction
        np.save(prediction_file, stage1_velocity)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {sample_id}: {e}")
        # Clean up partial processing
        prediction_file = os.path.join(predictions_dir, f"{sample_id}_stage1_prediction.npy")
        if os.path.exists(prediction_file):
            os.remove(prediction_file)
        return False

def generate_all_stage1_predictions(src_dir: str, predictions_dir: str, stage1_model_path: str, 
                                  max_samples: Optional[int] = None, device: Optional[str] = None):
    """
    Generate Stage 1 predictions for all samples.
    
    Args:
        src_dir: Source training data directory
        predictions_dir: Directory to save predictions
        stage1_model_path: Path to trained Stage 1 model
        max_samples: Maximum number of samples to process (for testing)
        device: Computing device ('cuda' or 'cpu')
    """
    print("🚀 Starting Stage 1 prediction generation...")
    
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"💻 Using device: {device}")
    
    # Validate source directory
    if not os.path.exists(src_dir):
        raise ValueError(f"Source directory does not exist: {src_dir}")
    
    # Get all sample directories
    sample_ids = [os.path.basename(p) for p in glob.glob(os.path.join(src_dir, '*')) if os.path.isdir(p)]
    
    if len(sample_ids) == 0:
        raise ValueError(f"No sample directories found in: {src_dir}")
    
    # Limit samples if specified (for testing)
    if max_samples is not None:
        sample_ids = sample_ids[:max_samples]
        print(f"🧪 Processing limited set: {max_samples} samples")
    
    print(f"📊 Found {len(sample_ids)} samples to process")
    
    # Create predictions directory
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Load Stage 1 model
    stage1_model = load_stage1_model(stage1_model_path, device)
    
    # Source coordinates
    source_coordinates = [1, 75, 150, 225, 300]
    
    # Process all samples
    successful = 0
    failed = 0
    
    print(f"⚙️  Generating predictions for {len(sample_ids)} samples...")
    
    for sample_id in tqdm(sample_ids, desc="Generating Stage 1 predictions"):
        success = generate_stage1_prediction_single_sample(
            sample_id, src_dir, predictions_dir, stage1_model, device, source_coordinates
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n✅ Stage 1 prediction generation completed!")
    print(f"📈 Successfully processed: {successful} samples")
    print(f"❌ Failed: {failed} samples")
    print(f"📁 Predictions saved to: {predictions_dir}")
    
    # Validate a few predictions
    if successful > 0:
        print(f"\n🔍 Validating generated predictions...")
        validation_samples = sample_ids[:min(3, len(sample_ids))]
        
        for sample_id in validation_samples:
            prediction_file = os.path.join(predictions_dir, f"{sample_id}_stage1_prediction.npy")
            
            if not os.path.exists(prediction_file):
                print(f"⚠️  Missing prediction file: {prediction_file}")
            else:
                try:
                    prediction = np.load(prediction_file)
                    
                    if prediction.shape == (300, 1259):
                        print(f"✅ Sample {sample_id} prediction validation passed")
                    else:
                        print(f"❌ Sample {sample_id} prediction shape validation failed: {prediction.shape}")
                        
                    # Check for NaN/inf values
                    if np.isnan(prediction).any():
                        print(f"⚠️  Sample {sample_id} contains NaN values")
                    if np.isinf(prediction).any():
                        print(f"⚠️  Sample {sample_id} contains Inf values")
                        
                except Exception as e:
                    print(f"❌ Failed to validate prediction for {sample_id}: {e}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Generate Stage 1 predictions for Stage 2 preprocessing")
    parser.add_argument('--src_dir', type=str, default=os.path.join(PROJECT_ROOT, "data/train_stage2"),
                       help='Source training data directory')
    parser.add_argument('--predictions_dir', type=str, default=os.path.join(PROJECT_ROOT, "data/stage1_predictions"),
                       help='Directory to save Stage 1 predictions')
    parser.add_argument('--stage1_model', type=str, default=os.path.join(PROJECT_ROOT, "models/caformer_sns_latest.pth"),
                       help='Path to trained Stage 1 model')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                       help='Computing device (auto-detect if not specified)')
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration even if predictions exist')
    
    args = parser.parse_args()
    
    # Check if predictions directory already exists
    if os.path.exists(args.predictions_dir) and not args.force:
        existing_files = len(glob.glob(os.path.join(args.predictions_dir, '*_stage1_prediction.npy')))
        if existing_files > 0:
            response = input(f"Predictions directory {args.predictions_dir} contains {existing_files} predictions. Continue? (y/N): ")
            if response.lower() != 'y':
                print("Prediction generation cancelled.")
                return
    
    try:
        generate_all_stage1_predictions(
            src_dir=args.src_dir,
            predictions_dir=args.predictions_dir,
            stage1_model_path=args.stage1_model,
            max_samples=args.max_samples,
            device=args.device
        )
        
        print(f"\n🎉 All done! Stage 1 predictions saved to: {args.predictions_dir}")
        print(f"📝 Next step: Run the Devito preprocessing script to complete Stage 2 data preparation")
        
    except Exception as e:
        print(f"\n💥 Prediction generation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())