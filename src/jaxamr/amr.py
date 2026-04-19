# Copyright © 2025 Haocheng Wen, Faxuan Luo, Hanbing Zou
# SPDX-License-Identifier: MIT
#                ___         ___           ___           ___           ___           ___           ___     
#       /\  \       /\  \         /\__\         /\  \         /\  \         /\__\         /\  \    
#       \:\  \     /::\  \       /::|  |       /::\  \       /::\  \       /::|  |       /::\  \   
#   ___ /::\__\   /:/\:\  \     /:|:|  |      /:/\:\  \     /:/\:\  \     /:|:|  |      /:/\:\  \  
#  /\  /:/\/__/  /::\~\:\  \   /:/|:|  |__   /:/  \:\  \   /::\~\:\  \   /:/|:|__|__   /::\~\:\  \ 
#  \:\/:/  /    /:/\:\ \:\__\ /:/ |:| /\__\ /:/__/ \:\__\ /:/\:\ \:\__\ /:/ |::::\__\ /:/\:\ \:\__\
#   \::/  /     \/__\:\/:/  / \/__|:|/:/  / \:\  \  \/__/ \/__\:\/:/  / \/__/~~/:/  / \/_|::\/:/  /
#    \/__/           \::/  /      |:/:/  /   \:\  \            \::/  /        /:/  /     |:|::/  / 
#                    /:/  /       |::/  /     \:\  \           /:/  /        /:/  /      |:|\/__/  
#                   /:/  /        /:/  /       \:\__\         /:/  /        /:/  /       |:|  |    
#                   \/__/         \/__/         \/__/         \/__/         \/__/         \|__|   
import jax.numpy as jnp
import jax.lax
from jax import jit, vmap
from jax.scipy.signal import convolve2d
from functools import partial
from jax import debug
from .simple_solver import boundary_conditions_2D
Lx = None
Ly = None
n_block=None
n_grid=None
refinement_tolerance=None
template_node_num=None
grid_mask_buffer_kernel=None
buffer_num = None


def set_amr(amr_config):
    global Lx, Ly, n_block, n_grid, refinement_tolerance, template_node_num, grid_mask_buffer_kernel,buffer_num
    n_block = amr_config['n_block']
    refinement_tolerance = amr_config['refinement_tolerance']
    template_node_num = amr_config['template_node_num']
    buffer_num = amr_config['buffer_num']
    Nx = amr_config['base_grid']['Nx']
    Ny = amr_config['base_grid']['Ny']
    Lx = amr_config['base_grid']['Lx']
    Ly = amr_config['base_grid']['Ly']
    n_grid = [[Nx // n_block[0][0], Ny // n_block[0][1]]]
    dx = [Lx/Nx]
    dy = [Ly/Ny]
    for i, (bx, by) in enumerate(n_block[1:], 1):
        px, py = n_grid[-1]
        mult = 1 if i == 1 else 2
        if (px * mult) % bx != 0 or (py * mult) % by != 0:
            raise ValueError(f"Initial grid not divisible: {(px * mult)}%{bx}={(py * mult)%bx}, {(py * mult)}%{by}={(py * mult)%by}")
            break
        n_grid.append([(px * mult // bx) , (py * mult// by) ])
        dx.append(Lx/Nx / (2.0**i))
        dy.append(Ly/Ny / (2.0**i))

    grid_mask_buffer_kernel = (
    jnp.zeros((2 * buffer_num + 1, 2 * buffer_num + 1))
        .at[buffer_num, :].set(1)
        .at[:, buffer_num].set(1)
        .at[buffer_num, buffer_num].set(0)
    )

@jit
def vectorized_cross_conv2d(mask):
    rows, cols = mask.shape
    result = jnp.zeros_like(mask)
    
    for offset in range(-buffer_num, buffer_num + 1):
        if offset != 0:
            shifted = jnp.roll(mask, shift=offset, axis=1)
            if offset > 0:
                shifted = shifted.at[:, :offset].set(0)
            else:
                shifted = shifted.at[:, offset:].set(0)
            result += shifted
    for offset in range(-buffer_num, buffer_num + 1):
        if offset != 0:
            shifted = jnp.roll(mask, shift=offset, axis=0)
            if offset > 0:
                shifted = shifted.at[:offset, :].set(0)
            else:
                shifted = shifted.at[offset:, :].set(0)
            result += shifted
    return result

@partial(jit, static_argnames=('level', 'nx', 'ny'))
def get_block_coordinates(level, blk_info, nx, ny):
    list_dx = []
    list_dy = []
    current_dx = Lx
    current_dy = Ly
    for k in range(level + 1):
        div_x = n_block[k][0]
        div_y = n_block[k][1]
        current_dx /= div_x
        current_dy /= div_y
        list_dx.append(current_dx)
        list_dy.append(current_dy)
    scales_x = jnp.array(list_dx)
    scales_y = jnp.array(list_dy)
    block_width_x = scales_x[level]
    block_width_y = scales_y[level]
    cell_dx = block_width_x / nx
    cell_dy = block_width_y / ny
    def compute_single_blk_cor(idx_row):
        x_indices = idx_row[0::2]
        y_indices = idx_row[1::2]
        x_min = jnp.dot(x_indices, scales_x)
        y_min = jnp.dot(y_indices, scales_y)
        grid_x, grid_y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing='ij')
        phys_x = x_min + (grid_x + 0.5) * cell_dx
        phys_y = y_min + (grid_y + 0.5) * cell_dy
        return phys_x, phys_y
    glob_index = blk_info['glob_index']
    X, Y = vmap(compute_single_blk_cor)(glob_index)
    return X, Y


@partial(jit, static_argnames=('level'))
def get_refinement_grid_mask(level, blk_data, blk_info, dx, dy):
    
    data_level = level - 1 if level > 0 else 0
    is_valid_block = jnp.all(blk_info['glob_index'] >= 0, axis=1)
    valid_mask = is_valid_block[:, None, None]
    mask_geo = 0
    if 'geometry' in refinement_tolerance: 
        nx = n_grid[data_level][0]
        ny = n_grid[data_level][1]
        
        if data_level >= 1:
            nx = nx * 2
            ny = ny * 2
        X, Y = get_block_coordinates(data_level, blk_info, nx, ny)
        dis_field = Y
        mask_dis = dis_field < refinement_tolerance['geometry']
        mask_geo = jnp.where(mask_dis, 1, 0)
    mask_phys = 0
    phys_keys = [k for k in refinement_tolerance.keys() if k != 'geometry']
    
    if len(phys_keys) > 0 and level >= 1:
        num = template_node_num
        
        if level == 1:
            src_data = blk_data
            is_ghost = False
        else:
            src_data = get_ghost_block_data(blk_data, blk_info)
            is_ghost = True

        for crit in phys_keys:
            threshold = refinement_tolerance[crit]
            
            valid_channel = False
            if crit == 'density':
                idx = 0; valid_channel = True
            elif crit == 'velocity':
                idx = 1; valid_channel = True
            
            if valid_channel:
                data_component = src_data[:, idx]
                grad_x, grad_y = vmap(jnp.gradient, in_axes=0)(data_component)
                
                if is_ghost:
                    grad_x = jnp.nan_to_num(grad_x[:, num:-num, num:-num])
                    grad_y = jnp.nan_to_num(grad_y[:, num:-num, num:-num])
                
                m_x = jnp.maximum(jnp.abs(grad_x) - threshold, 0)
                m_y = jnp.maximum(jnp.abs(grad_y) - threshold, 0)                
                mask_phys += (m_x + m_y)

    total_mask = jnp.sign(mask_phys + mask_geo)
    total_mask = total_mask * valid_mask
    # 若 jax 版本支持 convolve2d，请使用下面的函数 / If your JAX version supports convolve2d, please use the following function
    def extension_mask(mask):
        extended_mask = jnp.sign(convolve2d(mask, grid_mask_buffer_kernel, mode='same')) 
        return extended_mask
    ref_grid_mask = vmap(extension_mask, in_axes=0)(total_mask)
    # 若jax版本不支持convolve2d，请使用下面的函数 / If convolve2d is unavailable, use the following function
    # def extension_mask(mask):
    #     extended_mask = jnp.sign(vectorized_cross_conv2d(mask)) 
    #     return extended_mask
    # ref_grid_mask = vmap(extension_mask, in_axes=0)(total_mask)
    return ref_grid_mask


@partial(jit, static_argnames=('level'))
def get_refinement_block_mask(level, ref_grid_mask):

    ref_grid_mask = ref_grid_mask.reshape(ref_grid_mask.shape[0],
                        n_block[level][0], n_grid[level][0],
                        n_block[level][1], n_grid[level][1]).transpose(0, 1, 3, 2, 4) 

    ref_blk_mask = jnp.sign(ref_grid_mask.sum(axis=(3, 4)))

    return ref_blk_mask




@partial(jit, static_argnames=('max_blk_num'))
def get_refinement_block_info(blk_info, ref_blk_mask, max_blk_num):

    mask = ref_blk_mask != 0
    flat_mask = mask.ravel() 
    flat_indices = jnp.cumsum(flat_mask) * flat_mask
    indices_matrix = flat_indices.reshape(ref_blk_mask.shape)

    indices_matrix = get_ghost_mask(blk_info, indices_matrix)

    up = jnp.pad(indices_matrix, ((0, 0), (1, 0), (0, 0)), mode="constant")[:, 1:-2, 1:-1] 
    down = jnp.pad(indices_matrix, ((0, 0), (0, 1), (0, 0)), mode="constant")[:, 2:-1, 1:-1]
    left = jnp.pad(indices_matrix, ((0, 0), (0, 0), (1, 0)), mode="constant")[:, 1:-1, 1:-2]
    right = jnp.pad(indices_matrix, ((0, 0), (0, 0), (0, 1)), mode="constant")[:, 1:-1, 2:-1]

    blks, rows, cols = jnp.nonzero(mask, size = max_blk_num, fill_value = -1)

    up_vals = up[blks, rows, cols] - 1
    down_vals = down[blks, rows, cols] - 1
    left_vals = left[blks, rows, cols] - 1
    right_vals = right[blks, rows, cols] - 1

    ref_glob_blk_index = jnp.column_stack([blk_info['glob_index'][blks], rows, cols])
    ref_blk_index = jnp.column_stack([blks, rows, cols])
    ref_blk_number = jnp.sum(jnp.sign(ref_blk_mask))
    ref_blk_neighbor = jnp.column_stack([up_vals, down_vals, left_vals, right_vals])

    row_indices = jnp.arange(ref_blk_neighbor.shape[0])
    mask_nonzero = row_indices < ref_blk_number
    mask_nonzero = mask_nonzero[:, jnp.newaxis]

    ref_blk_neighbor = jnp.where(mask_nonzero, ref_blk_neighbor, -1)

    ref_blk_info = {
        'number': ref_blk_number.astype(int),
        'index': ref_blk_index,
        'glob_index': ref_glob_blk_index,
        'neighbor_index': ref_blk_neighbor
    }

    return ref_blk_info



@partial(jit, static_argnames=('level'))
def get_refinement_block_data(level, blk_data, ref_blk_info,):

    blk_data = blk_data.reshape(blk_data.shape[0], blk_data.shape[1],
                n_block[level][0], n_grid[level][0],
                n_block[level][1], n_grid[level][1]).transpose(0, 1, 2, 4, 3, 5)

    blks = ref_blk_info['index'][:, 0]
    rows = ref_blk_info['index'][:, 1]
    cols = ref_blk_info['index'][:, 2]
    ref_blk_data = blk_data[blks, :, rows, cols, :, :]

    ref_blk_data = ref_blk_data.at[-1].set(jnp.nan)


    ref_blk_data = interpolate_coarse_to_fine(ref_blk_data)

    return ref_blk_data

@partial(jit, static_argnames=('level'))
def extract_refined_blocks_with_ghost(level, blk_data, ref_blk_info):

    n_grid_x = n_grid[level][0]
    n_grid_y = n_grid[level][1]
    halo = template_node_num
    
    blks = ref_blk_info['index'][:, 0]
    rows = ref_blk_info['index'][:, 1]
    cols = ref_blk_info['index'][:, 2]

    def extract_patches(data, b_idx, r_idx, c_idx):
        start_x = r_idx * n_grid_x
        start_y = c_idx * n_grid_y

        slice_h = n_grid_x + 2* halo
        slice_w = n_grid_y + 2* halo

        slice_out = jax.lax.dynamic_slice(data, (b_idx,0,start_x,start_y), (1,data.shape[1],slice_h,slice_w))

        return slice_out.squeeze(0)
    extractor_vmap = jax.vmap(extract_patches, in_axes=(None,0,0,0))
    ref_blk_data = extractor_vmap(blk_data,blks,rows,cols)
    ref_blk_data = interpolate_coarse_to_fine(ref_blk_data)
    return ref_blk_data


@jit
def interpolate_coarse_to_fine(ref_blk_data):

    kernel = jnp.ones((2, 2))

    ref_blk_data = jnp.kron(ref_blk_data, kernel)

    return ref_blk_data


@partial(jit, static_argnames=('level'))
def interpolate_fine_to_coarse(level, blk_data, ref_blk_data, ref_blk_info):

    updated_blk_data = blk_data

    ref_blk_data = ref_blk_data.reshape(ref_blk_data.shape[0], ref_blk_data.shape[1],
                        ref_blk_data.shape[2]//2, 2,
                        ref_blk_data.shape[3]//2, 2).mean(axis=(3, 5))


    updated_blk_data = updated_blk_data.reshape(updated_blk_data.shape[0], updated_blk_data.shape[1],
                    n_block[level][0], n_grid[level][0],
                    n_block[level][1], n_grid[level][1]).transpose(0, 1, 2, 4, 3, 5)

    blks = ref_blk_info['index'][:, 0]
    rows = ref_blk_info['index'][:, 1]
    cols = ref_blk_info['index'][:, 2]
    updated_blk_data = updated_blk_data.at[blks, :, rows, cols, :, :].set(ref_blk_data)

    if level == 1:
        updated_blk_data = (
                    updated_blk_data.at[-1, :, -1, -1, :, :]
                    .set(blk_data[-1, :, -n_grid[level][0]:, -n_grid[level][1]:])
                    .transpose(0, 1, 2, 4, 3, 5)
                    .reshape(updated_blk_data.shape[0], updated_blk_data.shape[1],
                        n_block[level][0] * n_grid[level][0],
                        n_block[level][1] * n_grid[level][1])
        )
    else:
        updated_blk_data = (
                    updated_blk_data
                    .transpose(0, 1, 2, 4, 3, 5)
                    .reshape(updated_blk_data.shape[0], updated_blk_data.shape[1],
                        n_block[level][0] * n_grid[level][0],
                        n_block[level][1] * n_grid[level][1])
        )
    return updated_blk_data

@jit
def compute_morton_index(coords):
    coords = jnp.asarray(coords, dtype=jnp.uint32) & 0xFFFF 
    d = coords.shape[0]

    shift = 8
    while shift >= 1:
        mask = 0
        for i in range(0, 32, shift * d):
            mask |= ((1 << shift) - 1) << i
        coords = (coords | (coords << (shift * (d - 1)))) & mask
        shift = shift // 2

    shifts = jnp.arange(d, dtype=jnp.uint32)
    index = jnp.bitwise_or.reduce(coords << shifts[:, None], axis=0)
    return index.astype(jnp.uint32)



@jit
def compare_coords(A, B):

    matches = (A[:, None, :] == B[None, :, :])
    full_match = matches.all(axis=-1)

    return full_match.any(axis=1)


@jit
def find_unaltered_block_index(blk_info, prev_blk_info):

    index_A, num_A = prev_blk_info['glob_index'], prev_blk_info['number']
    index_B, num_B = blk_info['glob_index'], blk_info['number']

    '''
    morton_A = compute_morton_index(index_A.transpose(1,0))
    morton_B = compute_morton_index(index_B.transpose(1,0))

    mask_A = jnp.isin(morton_A, morton_B)
    mask_B = jnp.isin(morton_B, morton_A)

    '''
    mask_A = compare_coords(index_A, index_B)
    mask_B = compare_coords(index_B, index_A)

    rows_A = jnp.nonzero(mask_A, size=index_A.shape[0], fill_value=-1)[0]
    rows_B = jnp.nonzero(mask_B, size=index_B.shape[0], fill_value=-1)[0]

    unaltered_num = jnp.sum(jnp.sign(rows_A+1)) + num_A - index_A.shape[0]

    return rows_A, rows_B, unaltered_num



from jax import debug
@jit
def get_ghost_mask(blk_info, mask):
  
    num = 1
    neighbor = blk_info['neighbor_index']
    valid_neighbor = neighbor >= 0
    safe_neighbor = jnp.where(valid_neighbor, neighbor, 0)
    upper = mask[safe_neighbor[:,0], -num:, :]
    lower = mask[safe_neighbor[:,1], :num, :]
    left = mask[safe_neighbor[:,2], :, -num:]
    right = mask[safe_neighbor[:,3], :, :num]

    upper = upper * valid_neighbor[:, 0, None, None]
    lower = lower * valid_neighbor[:, 1, None, None]
    left  = left  * valid_neighbor[:, 2, None, None]
    right = right * valid_neighbor[:, 3, None, None]

    padded_horizontal = jnp.concatenate([left, mask, right], axis=2)

    pad_upper = jnp.pad(upper, ((0,0), (0,0), (num,num)), mode='constant', constant_values=0)
    pad_lower = jnp.pad(lower, ((0,0), (0,0), (num,num)), mode='constant', constant_values=0)

    ghost_mask = jnp.concatenate([pad_upper, padded_horizontal, pad_lower], axis=1)
    return ghost_mask


@partial(jit, static_argnames=('level', 'nx', 'ny'))
def get_level_scales(level, nx, ny):
    list_dx = []
    list_dy = []
    current_dx = Lx
    current_dy = Ly
    for k in range(level + 1):
        div_x = n_block[k][0]
        div_y = n_block[k][1]
        current_dx /= div_x
        current_dy /= div_y
        list_dx.append(current_dx)
        list_dy.append(current_dy)
    scales_x = jnp.array(list_dx)
    scales_y = jnp.array(list_dy)
    block_width_x = scales_x[level]
    block_width_y = scales_y[level]
    cell_dx = block_width_x / nx
    cell_dy = block_width_y / ny
    return scales_x, scales_y, cell_dx, cell_dy

@partial(jit, static_argnames=('nx', 'ny'))
def get_single_block_coordinates(glob_index, scales_x, scales_y, cell_dx, cell_dy, nx, ny):
    x_indices = glob_index[0::2]
    y_indices = glob_index[1::2]
    x_min = jnp.dot(x_indices, scales_x)
    y_min = jnp.dot(y_indices, scales_y)
    grid_x, grid_y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing='ij')
    phys_x = x_min + (grid_x + 0.5) * cell_dx
    phys_y = y_min + (grid_y + 0.5) * cell_dy
    return phys_x, phys_y
@partial(jit,static_argnames=('level'))
def add_boundary_condition(level, blk_data, blk_info, theta):
    nx = blk_data.shape[-2]
    ny = blk_data.shape[-1]
    scales_x, scales_y, cell_dx, cell_dy = get_level_scales(level, nx, ny)
    def add_single_boundary(single_blk_data, single_glob_index, theta):
        phys_x, phys_y = get_single_block_coordinates(single_glob_index, scales_x, scales_y, cell_dx, cell_dy, nx, ny)
        theta_local = theta.copy() if theta is not None else {}
        theta_local['bd_x'] = phys_x
        theta_local['bd_y'] = phys_y
        U = single_blk_data
        U_2d_with_ghost = boundary_conditions_2D(U, theta_local)
        return U_2d_with_ghost
    return vmap(add_single_boundary, in_axes=(0, 0, None))(blk_data, blk_info['glob_index'], theta)


@partial(jit, static_argnames=('level'))
def get_bd_mask(level,blk_info):
    list_dx = []
    list_dy = []
    current_dx = Lx
    current_dy = Ly
    for k in range(level + 1):
        div_x = n_block[k][0]
        div_y = n_block[k][1]

        current_dx /= div_x
        current_dy /= div_y

        list_dx.append(current_dx)
        list_dy.append(current_dy)
    scales_x = jnp.array(list_dx)
    scales_y = jnp.array(list_dy)
    leaf_dx = scales_x[level]
    leaf_dy = scales_y[level]

    def compute_single_bd_mask(idx_row):
        x_indices = idx_row[0::2]
        y_indices = idx_row[1::2]

        x_min = jnp.dot(x_indices,scales_x)
        y_min = jnp.dot(y_indices,scales_y)

        x_max = x_min + leaf_dx
        y_max = y_min + leaf_dy
        
        is_left = jnp.isclose(x_min,0.0)
        is_right = jnp.isclose(x_max,Lx)
        is_top = jnp.isclose(y_max,Ly)
        is_bottom = jnp.isclose(y_min,0.0)

        bool_mask = jnp.array([is_left,is_right,is_top,is_bottom])

        return jnp.where(bool_mask,1,0)
    glob_index = blk_info['glob_index']
    bd_mask = vmap(compute_single_bd_mask)(glob_index)

    return bd_mask

@partial(jit, static_argnames=('level'))
def get_refinement_block_data_withghost(level, blk_data, ref_blk_data, ref_blk_info,theta):
    num = template_node_num
    raw_blk_data = extract_refined_blocks_with_ghost(level, blk_data, ref_blk_info)
    raw_blk_data = jnp.nan_to_num(raw_blk_data, nan=1.0)
    ref_blk_data = get_ghost_block_data(ref_blk_data,ref_blk_info)
    ref_blk_data = jnp.nan_to_num(ref_blk_data, nan=1.0)
    noghost_re_blk_data = remove_ghost(ref_blk_data)
    bd_blk_data = add_boundary_condition(level,noghost_re_blk_data,ref_blk_info,theta)
    bd_blk_data = jnp.nan_to_num(bd_blk_data, nan=1.0)
    bd_mask = get_bd_mask(level,ref_blk_info)
    bd_mask = bd_mask[:, [0, 1, 3, 2]]
    bd_mask = bd_mask[:,:,None,None,None]
    neighbor = jnp.sign(ref_blk_info['neighbor_index']+1)[:,:,None,None,None]
    has_neighbor = neighbor
    mask_use_neighbor = has_neighbor
    mask_use_bd = (1.0 - has_neighbor) * bd_mask
    mask_use_inner = (1.0 - has_neighbor) * (1.0 - bd_mask)

    value = ref_blk_data[...,:num,:]*mask_use_neighbor[:,0] + raw_blk_data[...,num:2*num,num:-num] * mask_use_inner[:,0] + bd_blk_data[...,:num,:] * mask_use_bd[:,0]
    ref_blk_data_with_ghost = ref_blk_data.at[...,:num,:].set(value) 
    value = ref_blk_data[..., -num:, :] * mask_use_neighbor[:,1] + raw_blk_data[..., -2*num:-num, num:-num] * mask_use_inner[:,1] + bd_blk_data[...,-num:,:] * mask_use_bd[:,1]
    ref_blk_data_with_ghost = ref_blk_data_with_ghost.at[..., -num:, :].set(value)
    value = ref_blk_data[..., :, :num] * mask_use_neighbor[:,2] + raw_blk_data[..., num:-num, num:2*num] * mask_use_inner[:,2] + bd_blk_data[...,:,:num] * mask_use_bd[:,2]
    ref_blk_data_with_ghost = ref_blk_data_with_ghost.at[..., :, :num].set(value)
    value = ref_blk_data[..., :, -num:] * mask_use_neighbor[:,3] + raw_blk_data[..., num:-num, -2*num:-num] * mask_use_inner[:,3] + bd_blk_data[...,:,-num:] * mask_use_bd[:,3]
    ref_blk_data_with_ghost = ref_blk_data_with_ghost.at[..., :, -num:].set(value)

    ref_blk_data_with_ghost = ref_blk_data_with_ghost.at[-1].set(jnp.nan)
    return ref_blk_data_with_ghost


@jit
def remove_ghost(blk_data):
    num = template_node_num
    return blk_data[...,num:-num,num:-num]

@jit
def save_ghost(blk_data):
    num = template_node_num
    upper = blk_data[:,:,:,:num]
    lower = blk_data[:,:,:,-num:]
    left = blk_data[:,:,:num,num:-num]
    right = blk_data[:,:,-num:,num:-num]
    return upper,lower,left,right
@jit 
def add_ghost(blk_data,upper,lower,left,right):
    num = template_node_num
    padded_horizontal = jnp.concatenate([left, blk_data, right], axis=2)
    ghost_blk_data = jnp.concatenate([upper, padded_horizontal, lower], axis=3)
    return ghost_blk_data

@partial(jit, static_argnames=('level'))
def recover_ghost(level, blk_data, ref_blk_data, ref_blk_info, theta):
    def level1(arg):
        blk_data, ref_blk_data, ref_blk_info = arg
        nx0 = blk_data.shape[2]
        ny0 = blk_data.shape[3]
        scales_x, scales_y, cell_dx, cell_dy = get_level_scales(0, nx0, ny0)
        glob_index_0 = jnp.array([0, 0]) 
        phys_x, phys_y = get_single_block_coordinates(glob_index_0, scales_x, scales_y, cell_dx, cell_dy, nx0, ny0)
        theta_local = theta.copy() if theta is not None else {}
        theta_local['bd_x'] = phys_x
        theta_local['bd_y'] = phys_y
        U = blk_data
        U_2d = U[0]
        U_2d_with_ghost = boundary_conditions_2D(U_2d, theta_local)
        blk_data0_with_ghost_2d = U_2d_with_ghost
        blk_data_ready = jnp.expand_dims(blk_data0_with_ghost_2d, axis=0)
        return get_refinement_block_data_withghost(level, blk_data_ready, ref_blk_data, ref_blk_info, theta)
    def other_level(arg):
        blk_data, ref_blk_data, ref_blk_info = arg
        return get_refinement_block_data_withghost(level, blk_data, ref_blk_data, ref_blk_info, theta)
    ghost_blk_data = jax.lax.cond(level == 1, level1, other_level, (blk_data, ref_blk_data, ref_blk_info))
    return ghost_blk_data


@jit
def get_ghost_block_data(blk_data, blk_info):

    num = template_node_num

    neighbor = blk_info['neighbor_index']

    upper = blk_data[neighbor[:,0], :, -num:, :]
    lower = blk_data[neighbor[:,1], :, :num, :]
    left = blk_data[neighbor[:,2], :, :, -num:]
    right = blk_data[neighbor[:,3], :, :, :num]

    padded_horizontal = jnp.concatenate([left, blk_data, right], axis=3)

    pad_upper = jnp.pad(upper, ((0,0), (0,0), (0,0), (num,num)), mode='constant', constant_values=jnp.nan) 
    pad_lower = jnp.pad(lower, ((0,0), (0,0), (0,0), (num,num)), mode='constant', constant_values=jnp.nan)

    ghost_blk_data = jnp.concatenate([pad_upper, padded_horizontal, pad_lower], axis=2)

    return ghost_blk_data

@partial(jit, static_argnames=('level'))
def update_external_boundary(level, blk_data, ref_blk_data, ref_blk_info):
    num = template_node_num

    raw_blk_data = get_refinement_block_data(level, blk_data, ref_blk_info)

    neighbor = jnp.sign(ref_blk_info['neighbor_index'] + 1)[:, :, None, None, None]
    boundary_mask = jnp.ones_like(neighbor) - neighbor

    ref_blk_data = jnp.nan_to_num(ref_blk_data)

    value = ref_blk_data[..., :num, :] * neighbor[:,0] \
        + raw_blk_data[..., :num, :] * boundary_mask[:,0]
    ref_blk_data = ref_blk_data.at[..., :num, :].set(value)

    value = ref_blk_data[..., -num:, :] * neighbor[:,1] \
        + raw_blk_data[..., -num:, :] * boundary_mask[:,1]
    ref_blk_data = ref_blk_data.at[..., -num:, :].set(value)

    value = ref_blk_data[..., :, :num] * neighbor[:,2] \
        + raw_blk_data[..., :, :num] * boundary_mask[:,2]
    ref_blk_data = ref_blk_data.at[..., :, :num].set(value)

    value = ref_blk_data[..., :, -num:] * neighbor[:,3] \
        + raw_blk_data[..., :, -num:] * boundary_mask[:,3]
    ref_blk_data = ref_blk_data.at[..., :, -num:].set(value)

    ref_blk_data = ref_blk_data.at[-1].set(jnp.nan)

    return ref_blk_data


@jit
def pad_inact_blk(blk_data, blk_info):
    index = blk_info['index']
    index = index[:,0]
    mask = (index >= 0).astype(jnp.int32)
    mask = mask[:, None, None, None]
    blk_data = jnp.where(mask, blk_data, jnp.nan)
    return blk_data

def initialize(level, blk_data, blk_info, dx, dy):

    ref_grid_mask = get_refinement_grid_mask(level, blk_data, blk_info, dx, dy)

    ref_blk_mask = get_refinement_block_mask(level, ref_grid_mask)

    max_blk_num = initialize_max_block_number(level, ref_blk_mask)

    ref_blk_info = get_refinement_block_info(blk_info, ref_blk_mask, max_blk_num)

    ref_blk_data = get_refinement_block_data(level, blk_data, ref_blk_info)

    print(f'\nAMR Initialized at Level [{level}] with [{max_blk_num}] blocks')

    return ref_blk_data, ref_blk_info, max_blk_num

def update(level, blk_data, blk_info, dx, dy, prev_ref_blk_data, prev_ref_blk_info, max_blk_num):

    ref_grid_mask = get_refinement_grid_mask(level, blk_data, blk_info, dx, dy)

    ref_blk_mask = get_refinement_block_mask(level, ref_grid_mask)

    updated_mask, updated_max_blk_num = update_max_block_number(ref_blk_mask, max_blk_num)
    if updated_mask:
        max_blk_num = updated_max_blk_num
        print('\nAMR max_blk_num Updated as[',max_blk_num,'] at Level [',level,']')

    ref_blk_info = get_refinement_block_info(blk_info, ref_blk_mask, max_blk_num)

    ref_blk_data = get_refinement_block_data(level, blk_data, ref_blk_info)

    rows_A, rows_B, unaltered_num = find_unaltered_block_index(ref_blk_info, prev_ref_blk_info)
    
    ref_blk_data = ref_blk_data.at[rows_B[0:unaltered_num]].set(prev_ref_blk_data[rows_A[0:unaltered_num]])

    valid_blk_num = ref_blk_info['number']
    print(f'\nAMR Updated at Level [{level}] with [{valid_blk_num}/{max_blk_num}] blocks [valid/max]')

    return ref_blk_data, ref_blk_info, max_blk_num



def initialize_max_block_number(level, ref_blk_mask):

    ref_blk_num = jnp.sum(jnp.sign(ref_blk_mask))

    max_blk_num = int((ref_blk_num + 10 * 2**(level-1) )//10 * 10)# + 1000

    return max_blk_num


def update_max_block_number(ref_blk_mask, max_blk_num):

    ref_blk_num = jnp.sum(jnp.sign(ref_blk_mask))

    if (ref_blk_num + 1) > max_blk_num:
        updated_mask = True
        updated_max_blk_num = int(max_blk_num * 2.0)
    elif (ref_blk_num + 1) < (max_blk_num/2.5):
        updated_mask = True
        updated_max_blk_num = int(max_blk_num / 2.0)
    else:
        updated_mask = False
        updated_max_blk_num = max_blk_num

    return updated_mask, updated_max_blk_num
