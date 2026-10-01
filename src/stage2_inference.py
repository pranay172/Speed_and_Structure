#!/usr/bin/env python3
"""
Stage 2 Inference Script - PyTorch Only

This script generates final two-stage predictions using PyTorch only.
It loads preprocessed data (combined seismic + Stage 1 predictions) to avoid Devito conflicts.

Process:
1. Loads combined seismic data [x, δx] and Stage 1 predictions from preprocessing
2. Uses Stage 2 model to predict residual corrections δy
3. Computes final predictions: y_final = y1 + δy
4. Creates submission file with all predictions

This approach completely separates PyTorch from Devito to eliminate GPU conflicts.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from typing import Tuple

# Import required components
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ProcessedTestDataset(Dataset):
    """Dataset for loading preprocessed test data for Stage 2 inference."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.sample_ids = [os.path.basename(p) for p in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(p)]
        
        if len(self.sample_ids) == 0:
            raise ValueError(f"No processed samples found in directory: {data_dir}")
        
        # Validate that required files exist for first few samples
        validation_samples = self.sample_ids[:min(3, len(self.sample_ids))]
        required_files = ['combined_seismic.npy', 'stage1_velocity.npy']
        
        for sample_id in validation_samples:
            sample_path = os.path.join(data_dir, sample_id)
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(sample_path, f))]
            
            if missing_files:
                raise ValueError(f"Sample {sample_id} missing files: {missing_files}")
        
        print(f"Loaded {len(self.sample_ids)} processed test samples for Stage 2 inference")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        sample_id = self.sample_ids[idx]
        sample_path = os.path.join(self.data_dir, sample_id)

        try:
            # Load preprocessed data
            combined_seismic = np.load(os.path.join(sample_path, 'combined_seismic.npy')).astype(np.float32)
            stage1_velocity = np.load(os.path.join(sample_path, 'stage1_velocity.npy')).astype(np.float32)
            
            # Validate shapes
            assert combined_seismic.shape == (10, 10001, 31), f"Combined seismic shape mismatch: {combined_seismic.shape}"
            assert stage1_velocity.shape == (300, 1259), f"Stage1 velocity shape mismatch: {stage1_velocity.shape}"
            
            # Convert to tensors
            combined_seismic_tensor = torch.from_numpy(combined_seismic)
            stage1_velocity_tensor = torch.from_numpy(stage1_velocity).unsqueeze(0)  # Add channel dimension
            
            return combined_seismic_tensor, stage1_velocity_tensor, sample_id
            
        except Exception as e:
            raise RuntimeError(f"Error loading sample {sample_id}: {e}")

def load_stage2_model(model_path: str, device: torch.device):
    """Load the trained Stage 2 model."""
    try:
        from sns_caformer_stage2 import Stage2Model
        
        model = Stage2Model(
            backbone_name="caformer_b36.sail_in22k_ft_in1k",
            pretrained=False,
            in_channels_seismic=10,
            in_channels_velocity=1,
            out_channels_stem=128,
            use_tta=True
        ).to(device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"✅ Successfully loaded Stage 2 model from {model_path}")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load Stage 2 model: {e}")
        raise

def create_submission(sample_id: str, prediction: np.ndarray, submission_path: str):
    """Create or update submission file with prediction."""
    try:
        submission = dict(np.load(submission_path))
    except FileNotFoundError:
        submission = dict({})
    except Exception as e:
        print(f"Warning: Error loading existing submission file: {e}. Creating new file.")
        submission = dict({})
    
    submission.update({sample_id: prediction})
    
    try:
        os.makedirs(os.path.dirname(submission_path) if os.path.dirname(submission_path) else '.', exist_ok=True)
        np.savez(submission_path, **submission)
    except Exception as e:
        print(f"Error saving submission: {e}")
        raise

def validate_prediction_quality(prediction: np.ndarray, sample_id: str) -> Tuple[bool, str]:
    """Validate prediction quality."""
    # Check for NaN or Inf values
    if not np.isfinite(prediction).all():
        return False, f"Non-finite values detected in {sample_id}"
    
    # Check velocity range
    if prediction.min() < 0.5 or prediction.max() > 10.0:
        return False, f"Velocity out of range in {sample_id}: [{prediction.min():.2f}, {prediction.max():.2f}]"
    
    # Check for extreme gradients
    grad_h = np.abs(np.diff(prediction, axis=0))
    grad_v = np.abs(np.diff(prediction, axis=1))
    
    if grad_h.max() > 3.0 or grad_v.max() > 3.0:
        return False, f"Extreme gradients detected in {sample_id}"
    
    return True, "Good quality prediction"

def run_stage2_inference(processed_data_dir: str, stage2_model_path: str, submission_file: str,
                        batch_size: int = 2, use_tta: bool = True, device: str = None):
    """
    Run Stage 2 inference on preprocessed test data.
    
    Args:
        processed_data_dir: Directory containing preprocessed test data
        stage2_model_path: Path to trained Stage 2 model
        submission_file: Output submission file path
        batch_size: Batch size for inference
        use_tta: Whether to use test-time augmentation
        device: Computing device ('cuda' or 'cpu')
    """
    print("🚀 Starting Stage 2 inference...")
    
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"💻 Using device: {device}")
    
    # Validate directories
    if not os.path.exists(processed_data_dir):
        raise ValueError(f"Processed data directory does not exist: {processed_data_dir}")
    
    # Load Stage 2 model
    stage2_model = load_stage2_model(stage2_model_path, device)
    
    # Setup dataset and dataloader
    test_dataset = ProcessedTestDataset(processed_data_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"⚙️  Running Stage 2 inference on {len(test_dataset)} samples...")
    print(f"🔄 TTA enabled: {use_tta}")
    print(f"📦 Batch size: {batch_size}")
    
    # Remove existing submission file
    if os.path.exists(submission_file):
        os.remove(submission_file)
    
    # Process all batches
    successful = 0
    failed = 0
    quality_warnings = 0
    
    with torch.no_grad():
        for batch_idx, (combined_seismic_batch, stage1_velocity_batch, sample_ids) in enumerate(tqdm(test_loader, desc="Stage 2 Inference")):
            combined_seismic_batch = combined_seismic_batch.to(device)
            stage1_velocity_batch = stage1_velocity_batch.to(device)
            
            try:
                # Generate Stage 2 residual predictions
                residual_predictions = stage2_model(combined_seismic_batch, stage1_velocity_batch, use_tta=use_tta)
                
                # Compute final predictions for each sample in batch
                for i, sample_id in enumerate(sample_ids):
                    # Get Stage 1 prediction and residual correction
                    stage1_velocity = stage1_velocity_batch[i].squeeze(0).cpu().numpy()  # Remove channel dimension
                    residual_correction = residual_predictions[i].squeeze(0).cpu().numpy()  # Remove channel dimension
                    
                    # Final prediction: y_final = y1 + δy
                    final_prediction = stage1_velocity + residual_correction
                    
                    # Apply velocity constraints
                    final_prediction = np.clip(final_prediction, 1.0, 8.0)
                    
                    # Quality validation
                    is_valid, message = validate_prediction_quality(final_prediction, sample_id)
                    if not is_valid:
                        print(f"⚠️  {message}")
                        quality_warnings += 1
                    
                    # Convert to required format and save
                    prediction_final = final_prediction.astype(np.float64)
                    create_submission(sample_id, prediction_final, submission_file)
                    successful += 1
                    
            except Exception as e:
                print(f"❌ Error processing batch {batch_idx}: {e}")
                failed += len(sample_ids)
    
    # Summary
    print(f"\n✅ Stage 2 inference completed!")
    print(f"📈 Successfully processed: {successful} samples")
    print(f"❌ Failed: {failed} samples")
    print(f"⚠️  Quality warnings: {quality_warnings}")
    print(f"📁 Submission file created: {submission_file}")
    
    # Final validation
    try:
        submission_data = np.load(submission_file)
        print(f"\n🔍 Final submission validation:")
        print(f"📊 Submission contains {len(submission_data)} predictions")
        
        # Check a sample prediction
        if len(submission_data) > 0:
            sample_key = list(submission_data.keys())[0]
            sample_pred = submission_data[sample_key]
            print(f"📏 Sample prediction shape: {sample_pred.shape}")
            print(f"📈 Sample prediction range: [{sample_pred.min():.3f}, {sample_pred.max():.3f}]")
        
    except Exception as e:
        print(f"❌ Error validating submission file: {e}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Stage 2 inference using preprocessed test data")
    parser.add_argument('--processed_data_dir', type=str, required=True,
                       help='Directory containing preprocessed test data')
    parser.add_argument('--stage2_model', type=str, required=True,
                       help='Path to trained Stage 2 model')
    parser.add_argument('--submission_file', type=str, default='two_stage_submission.npz',
                       help='Output submission file path')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size for inference')
    parser.add_argument('--no_tta', action='store_true',
                       help='Disable test-time augmentation')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                       help='Computing device (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    try:
        run_stage2_inference(
            processed_data_dir=args.processed_data_dir,
            stage2_model_path=args.stage2_model,
            submission_file=args.submission_file,
            batch_size=args.batch_size,
            use_tta=not args.no_tta,
            device=args.device
        )
        
        print(f"\n🎉 Two-stage inference completed! Submission saved to: {args.submission_file}")
        print(f"📝 Ready for competition submission!")
        
    except Exception as e:
        print(f"\n💥 Stage 2 inference failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())