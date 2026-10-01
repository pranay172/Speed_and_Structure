#!/usr/bin/env python3
"""
Optimized seismic forward modeling script for the 'Speed and Structure' competition.

Performance optimizations:
- Reuse operators across shots
- Minimize object creation in loops
- Optimize Devito operator compilation
- Use efficient memory layouts
- Cache receiver coordinates

Key parameter changes based on competition details:
- Velocity model shape: (300, 1259)
- Receiver data shape: (10001, 31) which implies:
  - nt = 10001 (number of time steps)
  - nrec = 31 (number of receivers)
- Sources: 5 shot locations as specified in the training data.
"""

import numpy as np
import os

# Set Devito optimization flags
os.environ['DEVITO_LANGUAGE'] = 'openmp'  # Enable OpenMP parallelization
os.environ['DEVITO_LOGGING'] = 'WARNING'  # Reduce verbose output
os.environ['DEVITO_OPT'] = 'advanced'     # Enable advanced optimizations

# Import core Devito functionality
from examples.seismic import Model
from examples.seismic import TimeAxis
from examples.seismic import RickerSource
from examples.seismic import Receiver
try:
    from devito import (TimeFunction, Eq, solve, Operator, configuration)
    # Enable aggressive optimizations
    configuration['opt'] = 'advanced'
    configuration['language'] = 'openmp'
except ImportError:
    print("Devito not found. Please install it with 'pip install devito'")
    exit()

# Competition parameters
# --- Known parameters from the competition details ---
shape = (300, 1259)  # Model grid size (nx, nz)
n_shots = 5          # Number of shots
n_receivers = 31     # Number of receivers per shot
nt = 10001           # Number of time steps in the recording
# Source positions are given by their horizontal grid index (1-based in filenames)
# We use 0-based indices for Python: [0, 74, 149, 224, 299]
source_indices = [1, 75, 150, 225, 300]

# --- Optimized parameters ---
spacing = (1., 1.)    # Grid spacing in m
origin = (0., 0.)     # Top-left corner of the model
nbl = 200              # Number of absorbing boundary layers
time_order = 2        # Time order of the finite-difference stencil
space_order = 2       # Spatial order of the finite-difference stencil
f0 = 0.015            # Peak frequency of the Ricker wavelet (15 Hz)
rec_depth = 10.       # Depth of receivers in meters
src_depth = 20.       # Depth of source in meters

# Pre-calculate receiver coordinates (optimization)
def _get_receiver_coordinates(domain_size, n_receivers, rec_depth):
    """Pre-calculate receiver coordinates to avoid repeated computation."""
    rec_coords = np.zeros((n_receivers, 2), dtype=np.float32)
    rec_coords[:, 0] = np.linspace(0, domain_size[0], num=n_receivers)
    rec_coords[:, 1] = rec_depth
    return rec_coords

def vel_to_seis(vp_model):
    """
    Optimized seismic forward modeling for the competition.
    
    Major optimizations:
    - Reuse operators across shots (avoiding recompilation)
    - Pre-allocate arrays and reuse objects
    - Minimize object creation in loops
    - Use efficient Devito settings

    Parameters:
        vp_model (np.ndarray): The 2D velocity model with shape (300, 1259).
    """
    # Ensure the input velocity model has the correct shape
    if vp_model.shape != shape:
        raise ValueError(f"Input velocity model has shape {vp_model.shape}, but expected {shape}")

    # Create the seismic model object (shared across shots)
    model = Model(vp=vp_model, origin=origin, shape=shape, spacing=spacing,
                  space_order=space_order, nbl=nbl, bcs='damp')

    # Time step in milliseconds
    dt = 0.1
    t0 = 0.
    tn = (nt - 1) * dt
    time_range = TimeAxis(start=t0, stop=tn, num=nt)

    # Pre-calculate receiver coordinates (avoid repeated computation)
    rec_coords = _get_receiver_coordinates(model.domain_size, n_receivers, rec_depth)

    # Define wavefield and PDE (shared across shots)
    u = TimeFunction(name="u", grid=model.grid, time_order=time_order, 
                     space_order=space_order, dtype=np.float32)
    pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
    stencil = Eq(u.forward, solve(pde, u.forward))

    # Pre-allocate results array for efficiency
    all_shot_records = np.zeros((n_shots, nt, n_receivers), dtype=np.float32)

    # Create reusable source and receiver objects
    src = RickerSource(name='src', grid=model.grid, f0=f0,
                       npoint=1, time_range=time_range)
    rec = Receiver(name='rec', grid=model.grid, npoint=n_receivers, 
                   time_range=time_range, dtype=np.float32)
    
    # Set receiver coordinates once (they don't change)
    rec.coordinates.data[:] = rec_coords

    # Pre-compile operator templates (major optimization)
    src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m)
    rec_term = rec.interpolate(expr=u.forward)
    
    # Create operator once and reuse (huge speedup)
    op = Operator([stencil] + src_term + rec_term, 
                  subs=model.spacing_map, opt='advanced')

    # Loop over each source (optimized)
    for i, src_idx_1based in enumerate(source_indices):
        # Reset wavefield efficiently
        u.data.fill(0.0)
        rec.data.fill(0.0)
        
        # Update source position (only coordinate changes)
        src.coordinates.data[0, 0] = (src_idx_1based - 1) * spacing[0]
        src.coordinates.data[0, 1] = src_depth
        
        # Run simulation (reusing compiled operator)
        op(time=time_range.num-1, dt=dt)
        
        # Store results directly in pre-allocated array
        all_shot_records[i] = rec.data

    return all_shot_records


if __name__ == "__main__":
    # This is an example of how to run the forward modeling script.
    # It creates a simple two-layer velocity model and generates the seismic data.

    print("Running example simulation...")

    # Define the model shape as per competition details
    model_shape = (300, 1259)

    # Create a simple two-layer velocity model for demonstration
    # The values are in km/s
    vp_example = np.empty(model_shape, dtype=np.float32)
    vp_example[:, :600] = 1.5  # Top layer: 1.5 km/s
    vp_example[:, 600:] = 2.5  # Bottom layer: 2.5 km/s

    # Run the forward modeling
    generated_data = vel_to_seis(vp_model=vp_example)

    # You can now use the 'generated_data' list or the files in the output directory
    print(f"\nExample run finished. {len(generated_data)} shot records were generated.")
    print(f"Shape of the first shot record: {generated_data[0].shape}")