#!/usr/bin/env python3
"""
Stage 1 Inference Script - PyTorch Only

This script generates Stage 1 velocity predictions for test samples using PyTorch.
It runs completely separately from Devito to avoid GPU context conflicts.

The predictions are saved as .npy files that can be used by the forward modeling script.
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

class TestDataset(Dataset):
    """Dataset for loading test data."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.sample_ids = [os.path.basename(p) for p in glob.glob(os.path.join(data_dir, '*'))]
        self.source_coordinates = [1, 75, 150, 225, 300]
        
        if len(self.sample_ids) == 0:
            raise ValueError(f"No samples found in directory: {data_dir}")
        
        print(f"Loaded {len(self.sample_ids)} test samples for Stage 1 inference")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        sample_id = self.sample_ids[idx]
        sample_path = os.path.join(self.data_dir, sample_id)

        # Load seismic data
        seismic_inputs = []
        for i in self.source_coordinates:
            file_path = os.path.join(sample_path, f'receiver_data_src_{i}.npy')
            seismic_data = np.load(file_path).astype(np.float32)
            seismic_inputs.append(seismic_data)
        
        seismic_tensor = torch.from_numpy(np.stack(seismic_inputs, axis=0))
        return seismic_tensor, sample_id

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
            use_tta=True
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

def run_stage1_inference(test_data_dir: str, stage1_model_path: str, predictions_dir: str,
                        batch_size: int = 4, use_tta: bool = True, device: str = None):
    """
    Run Stage 1 inference on test data.
    
    Args:
        test_data_dir: Directory containing test samples
        stage1_model_path: Path to trained Stage 1 model
        predictions_dir: Directory to save Stage 1 predictions
        batch_size: Batch size for inference
        use_tta: Whether to use test-time augmentation
        device: Computing device ('cuda' or 'cpu')
    """
    print("🚀 Starting Stage 1 inference...")
    
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"💻 Using device: {device}")
    
    # Validate directories
    if not os.path.exists(test_data_dir):
        raise ValueError(f"Test data directory does not exist: {test_data_dir}")
    
    # Create predictions directory
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Load Stage 1 model
    stage1_model = load_stage1_model(stage1_model_path, device)
    
    # Setup dataset and dataloader
    test_dataset = TestDataset(test_data_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"⚙️  Running Stage 1 inference on {len(test_dataset)} samples...")
    print(f"🔄 TTA enabled: {use_tta}")
    print(f"📦 Batch size: {batch_size}")
    
    # Process all batches
    successful = 0
    failed = 0
    
    with torch.no_grad():
        for batch_idx, (seismic_batch, sample_ids) in enumerate(tqdm(test_loader, desc="Stage 1 Inference")):
            seismic_batch = seismic_batch.to(device)
            
            try:
                # Generate Stage 1 predictions
                predictions = stage1_model(seismic_batch, use_tta=use_tta)
                
                # Save predictions for each sample in batch
                for sample_id, pred_tensor in zip(sample_ids, predictions):
                    prediction = pred_tensor.squeeze(0).cpu().numpy().astype(np.float32)
                    
                    # Validate shape
                    if prediction.shape != (300, 1259):
                        print(f"⚠️  Unexpected prediction shape for {sample_id}: {prediction.shape}")
                        failed += 1
                        continue
                    
                    # Apply velocity constraints
                    prediction = np.clip(prediction, 1.0, 8.0)
                    
                    # Save prediction
                    prediction_file = os.path.join(predictions_dir, f"{sample_id}_stage1_prediction.npy")
                    np.save(prediction_file, prediction)
                    successful += 1
                    
            except Exception as e:
                print(f"❌ Error processing batch {batch_idx}: {e}")
                failed += len(sample_ids)
    
    # Summary
    print(f"\n✅ Stage 1 inference completed!")
    print(f"📈 Successfully processed: {successful} samples")
    print(f"❌ Failed: {failed} samples")
    print(f"📁 Predictions saved to: {predictions_dir}")
    
    # Validate a few predictions
    if successful > 0:
        print(f"\n🔍 Validating generated predictions...")
        prediction_files = glob.glob(os.path.join(predictions_dir, '*_stage1_prediction.npy'))
        validation_files = prediction_files[:min(3, len(prediction_files))]
        
        for pred_file in validation_files:
            sample_id = os.path.basename(pred_file).replace('_stage1_prediction.npy', '')
            
            try:
                prediction = np.load(pred_file)
                
                if prediction.shape == (300, 1259):
                    vel_range = f"[{prediction.min():.2f}, {prediction.max():.2f}]"
                    print(f"✅ Sample {sample_id} validation passed - shape: {prediction.shape}, range: {vel_range}")
                else:
                    print(f"❌ Sample {sample_id} validation failed - shape: {prediction.shape}")
                    
                # Check for NaN/inf values
                if np.isnan(prediction).any():
                    print(f"⚠️  Sample {sample_id} contains NaN values")
                if np.isinf(prediction).any():
                    print(f"⚠️  Sample {sample_id} contains Inf values")
                    
            except Exception as e:
                print(f"❌ Failed to validate prediction for {sample_id}: {e}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Stage 1 inference for test data")
    parser.add_argument('--test_data_dir', type=str, required=True, 
                       help='Test data directory')
    parser.add_argument('--stage1_model', type=str, required=True,
                       help='Path to trained Stage 1 model')
    parser.add_argument('--predictions_dir', type=str, default='./stage1_test_predictions',
                       help='Directory to save Stage 1 predictions')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for inference')
    parser.add_argument('--no_tta', action='store_true',
                       help='Disable test-time augmentation')
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
                print("Stage 1 inference cancelled.")
                return
    
    try:
        run_stage1_inference(
            test_data_dir=args.test_data_dir,
            stage1_model_path=args.stage1_model,
            predictions_dir=args.predictions_dir,
            batch_size=args.batch_size,
            use_tta=not args.no_tta,
            device=args.device
        )
        
        print(f"\n🎉 Stage 1 inference completed! Predictions saved to: {args.predictions_dir}")
        print(f"📝 Next step: Run the Devito forward modeling script")
        
    except Exception as e:
        print(f"\n💥 Stage 1 inference failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())