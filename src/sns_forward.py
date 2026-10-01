'''
SNS Competition Forward Model - Corrected and Optimized
Implements the vel_to_seis forward model for Speed and Structure competition.

This version fixes critical bugs related to memory management and CUDA graph usage.
- Memory usage is reduced from >46 GB to <50 MB by not storing the full wavefield.
- State is correctly reset for each source simulation.
- CUDA Graphs are now captured correctly for each source, ensuring valid results.
'''

import numpy as np
import cupy as cp

# Type configuration - using float32 for GPU efficiency
base_type = np.float32
base_type_gpu = cp.float32
base_type_str = 'float'

# SNS Competition Parameters
nz = 1259  # depth dimension
nx = 300   # width dimension  
dx = 1    # spatial discretization in units
nbc = 200   # number of boundary cells for ABC
nt = 10001 # number of time steps
dt = 0.1e-3  # time step (0.1 ms)
freq = 15  # dominant frequency for Ricker wavelet

# --- CUDA Kernels (unchanged) ---
# CUDA kernel to compute Laplacian (2nd-order)
lapg_kernel_code = r'''
extern "C" __global__
void lapg(
              const floattype* __restrict__ input,
              floattype* __restrict__ output,
              const int    nx,
              const int    ny,
              const floattype dx2_inv) { // 1/(dx^2) scaling factor
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    int idx = iy * nx + ix;

    // Apply boundary conditions properly - use input values at boundaries
    floattype term_x1 = (ix > 0) ? input[idx - 1] : 0.0;
    floattype term_x2 = (ix < nx - 1) ? input[idx + 1] : 0.0;
    floattype term_y1 = (iy > 0) ? input[idx - nx] : 0.0;
    floattype term_y2 = (iy < ny - 1) ? input[idx + nx] : 0.0;

    // 2nd Order Laplacian: (1/dx^2) * [u(i+1,j) + u(i-1,j) + u(i,j+1) + u(i,j-1) - 4*u(i,j)]
    output[idx] = dx2_inv * (term_x1 + term_x2 + term_y1 + term_y2 - 4.0f * input[idx]);
}
'''

# CUDA kernel to update p and add source
update_p_kernel_code = r'''
extern "C" __global__
void update_p(
              const floattype* __restrict__ temp1,
              const floattype* __restrict__ temp2,
              const floattype* __restrict__ alpha,
              const floattype* __restrict__ p0,
              const floattype* __restrict__ p1,
              floattype* __restrict__ p2,
              const floattype* __restrict__ lapg_store,
              const int    nx,
              const int    ny,
              const int    it,
              const int* __restrict__ src_idx,
              const floattype* __restrict__ s_mod) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    int idx = iy * nx + ix;

    floattype temp1_local = __ldg(&temp1[idx]);
    floattype temp2_local = __ldg(&temp2[idx]);
    floattype p1_local    = __ldg(&p1[idx]);
    floattype p0_local    = __ldg(&p0[idx]);
    floattype alpha_local = __ldg(&alpha[idx]);
    floattype out = temp1_local * p1_local - temp2_local * p0_local + alpha_local * lapg_store[idx];

    if (idx == src_idx[0]) {
        out += s_mod[it];
    }
    p2[idx] = out;
}
'''

# Compile Kernels
lapg_module = cp.RawModule(code=lapg_kernel_code.replace('floattype', base_type_str))
lapg = lapg_module.get_function('lapg')
update_p_module = cp.RawModule(code=update_p_kernel_code.replace('floattype', base_type_str))
update_p = update_p_module.get_function('update_p')


# --- Precalculations and helper functions (mostly unchanged) ---
def ricker(f, dt, nt=None):
    """Generate Ricker wavelet"""
    nw = int(2.2 / f / dt)
    nw = 2 * (nw // 2) + 1
    nc = nw // 2 + 1 
    k = np.arange(1, nw + 1)  
    alpha = (nc - k) * f * dt * np.pi
    beta = alpha ** 2
    w0 = (1.0 - 2.0 * beta) * np.exp(-beta)    
    w = np.zeros(nt or len(w0))
    w[0:len(w0)] = w0
    return w, np.arange(1, len(w)) * dt

def AbcCoef2D(nzbc, nxbc, nbc, dx):
    """Generate absorbing boundary condition coefficients"""
    nz = nzbc - 2 * nbc; nx = nxbc - 2 * nbc
    a = (nbc - 1) * dx
    kappa = 3.0 * np.log(1e7) / (2.0 * a)
    damp1d = kappa * (((np.arange(1, nbc + 1) - 1) * dx / a) ** 2)
    damp = np.zeros((nzbc, nxbc))
    for iz in range(nzbc):
        damp[iz, :nbc] = damp1d[::-1]
        damp[iz, nx + nbc : nx + 2 * nbc] = damp1d
    for ix in range(nbc, nbc + nx):
        damp[:nbc, ix] = damp1d[::-1]
        damp[nz + nbc: nz + 2 * nbc, ix] = damp1d
    return cp.array(damp, dtype=base_type_gpu)

def adjust_sr(coord, dx, nbc):
    """Adjust source and receiver coordinates for padded domain"""
    isx = int(round(coord['sx'] / dx)) + nbc
    isz = int(round(coord['sz'] / dx)) + nbc
    igx = (np.round(np.array(coord['gx']) / dx) + nbc).astype(int)
    igz = (np.round(np.array(coord['gz']) / dx) + nbc).astype(int)
    if abs(coord['sz']) < 0.5: isz += 1
    igz += (np.abs(np.array(coord['gz'])) < 0.5).astype(int)
    return isx, isz, igx, igz

# --- Global Setup (constants and pre-calculated source/receiver info) ---
dx2_inv = base_type(1.0 / (dx * dx))  # 1/(dx^2) for Laplacian scaling
s, _ = ricker(freq, dt, nt)
s_gpu = cp.array(s, dtype=base_type_gpu)

source_positions = [1, 75, 150, 225, 300]
ng = 31 # number of receivers
receiver_x_positions = np.arange(ng) * 10 + 1 # [1, 11, 21, ..., 301]

# Padded dimensions and damping coefficients
nzbc, nxbc = nz + 2 * nbc, nx + 2 * nbc
damp = AbcCoef2D(nzbc, nxbc, nbc, dx)

# Pre-calculate source and receiver info
src_info_list = []
for i_source in range(5):
    coord = {
        'sx': (source_positions[i_source] - 1),
        'sz': 20, # Depth in grid units (20 * 10 = 200 meters)
        'gx': (receiver_x_positions - 1),
        'gz': np.ones(ng) * 10 # Depth in grid units (10 * 10 = 100 meters)
    }
    isx, isz, igx, igz = adjust_sr(coord, dx, nbc)
    src_info_list.append({
        'isx': isx, 'isz': isz, 'igx': cp.array(igx), 'igz': cp.array(igz),
        'src_idx_scalar': np.int32(isz * nxbc + isx)
    })

# CUDA kernel launch parameters
tx, ty = 16, 16
bx = (nxbc + tx - 1) // tx
by = (nzbc + ty - 1) // ty
grid_dims = (bx, by)
block_dims = (tx, ty)


def vel_to_seis(vec):
    """
    Forward model: velocity to seismogram (Corrected Implementation)
    """
    assert vec.shape == (300*1259+1, 1) and vec.dtype == base_type_gpu

    # --- Prepare velocity-dependent coefficients ---
    vv = cp.reshape(vec[:-1, 0], (nx, nz)) * 1000.0 # Reshape to (width, depth) and convert to m/s
    vv = vv.T  # Transpose to (depth, width) for internal use
    min_vel = vec[-1, 0] * 1000.0
    v = cp.pad(vv, nbc, mode='edge')
    
    abc = min_vel * damp
    alpha = (v * dt) ** 2 / (dx ** 2)  # Correct alpha for 2nd order Laplacian
    kappa = abc * dt
    temp1 = 2.0 - kappa
    temp2 = 1.0 - kappa

    # --- Run simulation for all sources ---
    seis_combined = cp.zeros((5, nt, ng), dtype=base_type_gpu)
    stream = cp.cuda.Stream(non_blocking=True)
    
    # Allocate wavefield buffers OUTSIDE the loop to avoid CUDA graph issues
    p0 = cp.zeros((nzbc, nxbc), dtype=base_type_gpu)
    p1 = cp.zeros((nzbc, nxbc), dtype=base_type_gpu)
    p2 = cp.zeros((nzbc, nxbc), dtype=base_type_gpu)
    lapg_store = cp.zeros((nzbc, nxbc), dtype=base_type_gpu)
    
    # Pre-allocate source-specific variables that will be updated per source
    src_idx_dev = cp.zeros(1, dtype=cp.int32)
    s_mod = cp.zeros(nt, dtype=base_type_gpu)
    igz_dev = cp.zeros(ng, dtype=cp.int32)
    igx_dev = cp.zeros(ng, dtype=cp.int32)
    
    # Create separate graphs for each source to capture source-specific parameters
    seis_temp = cp.zeros((nt, ng), dtype=base_type_gpu)
    graphs = []
    
    for i_source in range(5):
        # Reset wavefield state for new source simulation
        p0.fill(0)
        p1.fill(0)
        p2.fill(0)
        lapg_store.fill(0)
        seis_temp.fill(0)

        # Get source/receiver info for this run and update GPU arrays
        info = src_info_list[i_source]
        igz_dev[:] = info['igz']
        igx_dev[:] = info['igx']
        
        # Prepare source injection term
        bdt = (v[info['isz'], info['isx']] * dt) ** 2
        s_mod[:] = bdt * s_gpu            
        src_idx_dev[0] = info['src_idx_scalar']
        
        # Capture a unique graph for this source with its specific parameters
        stream.begin_capture()
        for it in range(nt):
            # Launch kernels with rotating buffers
            lapg(grid_dims, block_dims, (p1, lapg_store, nxbc, nzbc, dx2_inv))
            update_p(grid_dims, block_dims, (temp1, temp2, alpha, p0, p1, p2, lapg_store,
                                             nxbc, nzbc, it, src_idx_dev, s_mod))
            
            # Record seismogram to temporary buffer
            seis_temp[it, :] = p2[igz_dev, igx_dev]
            
            # Rotate buffers (efficient pointer swap)
            p0, p1, p2 = p1, p2, p0
        
        graph = stream.end_capture()
        graphs.append(graph)
        
        # Launch the graph for this source
        graph.launch(stream)
        stream.synchronize() # Wait for this source to finish
        
        # Copy result from temporary buffer to final output
        seis_combined[i_source, :, :] = seis_temp

    # Finalize and flatten the result as required
    assert seis_combined.shape == (5, 10001, 31)
    
    return cp.copy(seis_combined.flatten()[:, None])