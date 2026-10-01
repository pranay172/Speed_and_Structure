"""
Example code demonstrating how to use sns_forward.py to:
1. Read a velocity model from SNS competition data
2. Generate synthetic seismic data using the forward model
3. Compare with the original seismic data
"""

import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from glob import glob
import sns_forward_devito  # Import our forward model

def load_sns_sample(sample_path):
    """
    Load a single sample from the SNS competition data
    
    Args:
        sample_path: Path to a sample folder
        
    Returns:
        velocity_model: np.ndarray of shape (300, 1259) in km/s
        seismic_data: dict with 5 receiver arrays, each of shape (10001, 31)
    """
    # Load velocity model
    velocity_model = np.load(os.path.join(sample_path, "vp_model.npy"))
    
    # Load seismic data for all 5 sources
    source_positions = [1, 75, 150, 225, 300]
    seismic_data = {}
    for src_pos in source_positions:
        filename = f"receiver_data_src_{src_pos}.npy"
        seismic_data[src_pos] = np.load(os.path.join(sample_path, filename))
    
    return velocity_model, seismic_data

def prepare_velocity_for_forward(velocity_model, min_velocity=None):
    """
    Prepare velocity model for the forward model
    
    Args:
        velocity_model: np.ndarray of shape (300, 1259) in km/s
        min_velocity: Minimum velocity in km/s (if None, computed from model)
        
    Returns:
        vec: cp.ndarray of shape (300*1259+1, 1) ready for forward model
    """
    # Flatten the velocity model
    vel_flat = velocity_model.flatten()
    
    # Compute minimum velocity if not provided
    if min_velocity is None:
        min_velocity = np.min(velocity_model)
    
    # Create the input vector with velocity model + minimum velocity
    vec = np.zeros((300*1259 + 1, 1), dtype=np.float32)
    vec[:-1, 0] = vel_flat
    vec[-1, 0] = min_velocity
    
    # Convert to CuPy array
    vec = cp.array(vec, dtype=cp.float32)
    
    return vec

def reshape_seismic_output(seismic_output):
    """
    Reshape the forward model output to match original data format
    
    Args:
        seismic_output: cp.ndarray of shape (5*10001*31, 1)
        
    Returns:
        seismic_dict: dict with 5 arrays, each of shape (10001, 31)
    """
    # Convert to numpy and reshape
    seismic_array = cp.asnumpy(seismic_output).reshape(5, 10001, 31)
    
    # Create dictionary matching original format
    source_positions = [1, 75, 150, 225, 300]
    seismic_dict = {}
    for i, src_pos in enumerate(source_positions):
        seismic_dict[src_pos] = seismic_array[i]
    
    return seismic_dict

def compare_seismic_data(original, generated, source_idx=0, sample_id="sample"):
    """
    Compare original and generated seismic data
    
    Args:
        original: dict of original seismic data
        generated: dict of generated seismic data
        source_idx: which source to visualize (0-4)
        sample_id: sample identifier for filename
    """
    # Create output directory if it doesn't exist
    output_dir = "forward_model_comparison"
    os.makedirs(output_dir, exist_ok=True)
    source_positions = [1, 75, 150, 225, 300]
    src_pos = source_positions[source_idx]
    
    # Check for problematic values first
    orig_has_issues = np.any(np.isnan(original[src_pos])) or np.any(np.isinf(original[src_pos]))
    gen_has_issues = np.any(np.isnan(generated[src_pos])) or np.any(np.isinf(generated[src_pos]))
    
    if orig_has_issues:
        print(f"⚠️  WARNING: Original data contains NaN/Inf values")
    if gen_has_issues:
        print(f"⚠️  WARNING: Generated data contains NaN/Inf values")
        print(f"   Generated data range: {np.nanmin(generated[src_pos]):.2e} to {np.nanmax(generated[src_pos]):.2e}")
    
    # Calculate difference with safety checks
    diff = original[src_pos] - generated[src_pos]
    
    # Calculate robust metrics that handle NaN/Inf
    if not (orig_has_issues or gen_has_issues):
        mse = np.mean(diff**2)
        mae = np.mean(np.abs(diff))
        max_abs_error = np.max(np.abs(diff))
        relative_error = np.mean(np.abs(diff) / (np.abs(original[src_pos]) + 1e-10))
    else:
        # Use robust metrics that ignore NaN/Inf
        mse = np.nanmean(diff**2) if not np.all(np.isnan(diff**2)) else np.inf
        mae = np.nanmean(np.abs(diff)) if not np.all(np.isnan(diff)) else np.inf
        max_abs_error = np.nanmax(np.abs(diff)) if not np.all(np.isnan(diff)) else np.inf
        
        # More robust relative error calculation
        valid_mask = ~(np.isnan(original[src_pos]) | np.isnan(generated[src_pos]) | 
                      np.isinf(original[src_pos]) | np.isinf(generated[src_pos]))
        if np.any(valid_mask):
            valid_orig = original[src_pos][valid_mask]
            valid_diff = diff[valid_mask]
            relative_error = np.mean(np.abs(valid_diff) / (np.abs(valid_orig) + 1e-10))
        else:
            relative_error = np.inf
    
    print(f"\nComparison metrics for source at position {src_pos}:")
    print(f"  MSE: {mse:.6e}")
    print(f"  MAE: {mae:.6e}")
    print(f"  Max absolute error: {max_abs_error:.6e}")
    print(f"  Mean relative error: {relative_error:.6f}")
    
    # Visualization
    _, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original data
    im1 = axes[0, 0].imshow(original[src_pos].T, aspect='auto', cmap='RdBu_r')
    axes[0, 0].set_title(f'Original Seismic Data (Source {src_pos})')
    axes[0, 0].set_xlabel('Time samples')
    axes[0, 0].set_ylabel('Receiver')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Generated data
    im2 = axes[0, 1].imshow(generated[src_pos].T, aspect='auto', cmap='RdBu_r')
    axes[0, 1].set_title(f'Generated Seismic Data (Source {src_pos})')
    axes[0, 1].set_xlabel('Time samples')
    axes[0, 1].set_ylabel('Receiver')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Difference
    im3 = axes[0, 2].imshow(diff.T, aspect='auto', cmap='RdBu_r')
    axes[0, 2].set_title('Difference (Original - Generated)')
    axes[0, 2].set_xlabel('Time samples')
    axes[0, 2].set_ylabel('Receiver')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Time series comparison for middle receiver
    middle_receiver = 15
    axes[1, 0].plot(original[src_pos][:1000, middle_receiver], label='Original', alpha=0.7)
    axes[1, 0].plot(generated[src_pos][:1000, middle_receiver], label='Generated', alpha=0.7)
    axes[1, 0].set_title(f'Time Series - Receiver {middle_receiver} (first 1000 samples)')
    axes[1, 0].set_xlabel('Time samples')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error over time
    axes[1, 1].plot(np.mean(np.abs(diff), axis=1))
    axes[1, 1].set_title('Mean Absolute Error over Time')
    axes[1, 1].set_xlabel('Time samples')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Error per receiver
    axes[1, 2].plot(np.mean(np.abs(diff), axis=0))
    axes[1, 2].set_title('Mean Absolute Error per Receiver')
    axes[1, 2].set_xlabel('Receiver index')
    axes[1, 2].set_ylabel('MAE')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure to the subfolder
    filename = f"comparison_{sample_id}_source_{src_pos}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison plot: {filepath}")
    
    return mse, mae, max_abs_error, relative_error

def visualize_velocity_model(velocity_model, sample_id="sample"):
    """Visualize the velocity model"""
    # Create output directory if it doesn't exist
    output_dir = "forward_model_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # Show velocity model
    im = plt.imshow(velocity_model.T, aspect='auto', cmap='jet', origin='lower')
    plt.colorbar(im, label='Velocity (km/s)')
    plt.title('Velocity Model')
    plt.xlabel('Width (units)')
    plt.ylabel('Depth (units)')
    
    # Add source and receiver positions
    source_positions = [1, 75, 150, 225, 300]
    receiver_positions = np.arange(31) * 10 + 1
    
    # Plot sources (red triangles at depth 20)
    for src_x in source_positions:
        plt.plot(src_x-1, 20, 'rv', markersize=10, label='Source' if src_x == 1 else '')
    
    # Plot receivers (blue circles at depth 10)
    plt.plot(receiver_positions-1, np.ones_like(receiver_positions)*10, 'bo', 
             markersize=6, label='Receivers')
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the figure to the subfolder
    filename = f"velocity_model_{sample_id}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved velocity model plot: {filepath}")

def main():
    """Main function to demonstrate the forward model"""
    
    # Path to training data (adjust as needed)
    training_dataset = "./data/train/*"
    sample_paths = glob(training_dataset)
    
    if len(sample_paths) == 0:
        print("No training samples found. Please check the data path.")
        return
    
    # Use the first sample for demonstration
    sample_path = sample_paths[0]
    sample_id = os.path.basename(sample_path)
    print(f"Processing sample: {sample_id}")
    
    # Load the sample data
    print("\n1. Loading SNS sample data...")
    velocity_model, original_seismic = load_sns_sample(sample_path)
    print(f"   Velocity model shape: {velocity_model.shape}")
    print(f"   Velocity range: {velocity_model.min():.3f} - {velocity_model.max():.3f} km/s")
    print(f"   Number of sources: {len(original_seismic)}")
    
    # Visualize velocity model
    print("\n2. Visualizing velocity model...")
    visualize_velocity_model(velocity_model, sample_id=sample_id)
    
    # Prepare velocity for forward model
    print("\n3. Preparing velocity model for forward computation...")
    # vec = prepare_velocity_for_forward(velocity_model)
    vec = velocity_model
    print(f"   Input vector shape: {vec.shape}")
    
    # Run forward model
    print("\n4. Running forward model...")
    print("   This may take a few moments on first run due to CUDA compilation...")
    
    # Call the forward model
    seismic_output = sns_forward_devito.vel_to_seis(vec)
    print(f"   Output shape: {seismic_output.shape}")
    
    # Reshape output to match original format
    print("\n5. Reshaping output...")
    generated_seismic = reshape_seismic_output(seismic_output)
    
    # Compare results for each source
    print("\n6. Comparing original and generated seismic data...")
    
    all_metrics = []
    for i in range(5):
        print(f"\n--- Source {i+1} ---")
        metrics = compare_seismic_data(original_seismic, generated_seismic, source_idx=i, sample_id=sample_id)
        all_metrics.append(metrics)
    
    # Summary statistics
    print("\n\n=== SUMMARY ACROSS ALL SOURCES ===")
    metrics_array = np.array(all_metrics)
    print(f"Average MSE: {metrics_array[:, 0].mean():.6e}")
    print(f"Average MAE: {metrics_array[:, 1].mean():.6e}")
    print(f"Average Max Error: {metrics_array[:, 2].mean():.6e}")
    print(f"Average Relative Error: {metrics_array[:, 3].mean():.6f}")
    
    # Note about expected differences
    print("\n\nNote: Some differences between original and generated data are expected due to:")
    print("- Numerical precision differences")
    print("- Potential differences in the exact forward modeling parameters")
    print("- Boundary condition implementations")
    print("- The original data might have been generated with slightly different settings")

if __name__ == "__main__":
    main()