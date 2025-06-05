# Copyright © 2025 Haocheng Wen, Faxuan Luo
# SPDX-License-Identifier: MIT
import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap
from jax.scipy.signal import convolve2d
from functools import partial

Lx = None
Ly = None
n_block=None
n_grid=None
refinement_tolerance=None
template_node_num=None
grid_mask_buffer_kernel=None
parallel = 'off'

def set_amr(amr_config):
    global Lx, Ly, n_block, n_grid, refinement_tolerance, template_node_num, grid_mask_buffer_kernel,parallel
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
    if 'parallel' in amr_config:
        parallel = amr_config['parallel']



@partial(jit, static_argnames=('level', 'criterion'))
def get_refinement_grid_mask(level, blk_data, blk_info, criterion, dx, dy):

    num = template_node_num

    if level == 0:
        pass
    elif level == 1:
        if criterion == 'density':
            data_component = blk_data[:, 0]
        elif criterion == 'schlieren':
            pass
        elif criterion == 'velocity':
            data_component = blk_data[:, 1]

        grad_x, grad_y = vmap(jnp.gradient, in_axes=0)(data_component)
    else:
        ghost_blk_data = get_ghost_block_data(blk_data, blk_info)

        if criterion == 'density':
            data_component = ghost_blk_data[:, 0]
        elif criterion == 'schlieren':
            pass
        elif criterion == 'velocity':
            data_component = ghost_blk_data[:, 1]

        grad_x, grad_y = vmap(jnp.gradient, in_axes=0)(data_component)
      
        grad_x = jnp.nan_to_num(grad_x[:, num:-num, num:-num])
        grad_y = jnp.nan_to_num(grad_y[:, num:-num, num:-num])

    mask_x = jnp.maximum(jnp.abs(grad_x / (dx*2.0)) - refinement_tolerance[criterion], 0)
    mask_y = jnp.maximum(jnp.abs(grad_y / (dy*2.0)) - refinement_tolerance[criterion], 0)

    mask = jnp.sign(mask_x + mask_y)


    def extension_mask(mask):
        extended_mask = jnp.sign(convolve2d(mask, grid_mask_buffer_kernel, mode='same')) 
        return extended_mask

    ref_grid_mask = vmap(extension_mask, in_axes=0)(mask)

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
def get_refinement_block_data(level, blk_data, ref_blk_info):

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



@jit
def interpolate_coarse_to_fine(ref_blk_data):

    kernel = jnp.ones((2, 2))

    ref_blk_data = jnp.kron(ref_blk_data, kernel)

    return ref_blk_data



@partial(jit, static_argnames=('level'))
def interpolate_fine_to_coarse_normal(level, blk_data, ref_blk_data, ref_blk_info):

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

    updated_blk_data = (
                updated_blk_data.at[:, :, -1, -1, :, :]
                .set(blk_data[:, :, -n_grid[level][0]:, -n_grid[level][1]:])
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



@jit
def get_ghost_mask(blk_info, mask):
  
    num = 1
    neighbor = blk_info['neighbor_index']

    upper = mask[neighbor[:,0], -num:, :]
    lower = mask[neighbor[:,1], :num, :]
    left = mask[neighbor[:,2], :, -num:]
    right = mask[neighbor[:,3], :, :num]

    padded_horizontal = jnp.concatenate([left, mask, right], axis=2)

    pad_upper = jnp.pad(upper, ((0,0), (0,0), (num,num)), mode='constant', constant_values=0)
    pad_lower = jnp.pad(lower, ((0,0), (0,0), (num,num)), mode='constant', constant_values=0)

    ghost_mask = jnp.concatenate([pad_upper, padded_horizontal, pad_lower], axis=1)

    return ghost_mask


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


def initialize_normal(level, blk_data, blk_info, criterion, dx, dy):

    ref_grid_mask = get_refinement_grid_mask(level, blk_data, blk_info, criterion, dx, dy)

    ref_blk_mask = get_refinement_block_mask(level, ref_grid_mask)

    max_blk_num = initialize_max_block_number(level, ref_blk_mask)

    ref_blk_info = get_refinement_block_info(blk_info, ref_blk_mask, max_blk_num)

    ref_blk_data = get_refinement_block_data(level, blk_data, ref_blk_info)

    print(f'\nAMR Initialized at Level [{level}] with [{max_blk_num}] blocks')

    return ref_blk_data, ref_blk_info, max_blk_num


def update_normal(level, blk_data, blk_info, criterion, dx, dy, prev_ref_blk_data, prev_ref_blk_info, max_blk_num):

    ref_grid_mask = get_refinement_grid_mask(level, blk_data, blk_info, criterion, dx, dy)

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

    max_blk_num = int((ref_blk_num + 10 * 2**(level-1) )//10 * 10)

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


##parallel_settings
blk_info_pmap_axis = {'number': 0,
                      'index': 0,
                      'glob_index': 0,
                      'neighbor_index': 0
                     }

@partial(pmap,axis_name='x',static_broadcasted_argnums=(0,3),in_axes=(None,0,blk_info_pmap_axis, None,None,None))
def parallel_initialize(level, blk_data, blk_info, criterion, dx, dy):
    device_idx = jax.lax.axis_index('x')
    
    ref_grid_mask = get_refinement_grid_mask(level, blk_data, blk_info, criterion, dx, dy)

    ref_blk_mask = get_refinement_block_mask(level, ref_grid_mask)

    max_blk_num = initialize_max_block_number(level, ref_blk_mask)

    ref_blk_info = get_refinement_block_info(blk_info, ref_blk_mask, max_blk_num)

    ref_blk_data = get_refinement_block_data(level, blk_data, ref_blk_info)

    print(f'\nDevice [{device_idx}]: AMR Initialized at Level [{level}] with [{max_blk_num}] blocks')

    return ref_blk_data, ref_blk_info, max_blk_num

@partial(pmap,axis_name='x',static_broadcasted_argnums=(0,3),in_axes=(None, 0, blk_info_pmap_axis, None, None, None,0,blk_info_pmap_axis,0))
def parallel_update(level, blk_data, blk_info, criterion, dx, dy, prev_ref_blk_data, prev_ref_blk_info, max_blk_num):
    device_idx = jax.lax.axis_index('x')
    
    ref_grid_mask = get_refinement_grid_mask(level, blk_data, blk_info, criterion, dx, dy)

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
    print(f'\nDevice [{device_idx}]:AMR Updated at Level [{level}] with [{valid_blk_num}/{max_blk_num}] blocks [valid/max]')

    return ref_blk_data, ref_blk_info, max_blk_num

@partial(pmap,axis_name='x',static_broadcasted_argnums=0,in_axes=(None, 0, 0, blk_info_pmap_axis))
@partial(jit, static_argnames=('level'))
def parallel_interpolate_fine_to_coarse(level, blk_data, ref_blk_data, ref_blk_info):

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

    updated_blk_data = (
                updated_blk_data.at[:, :, -1, -1, :, :]
                .set(blk_data[:, :, -n_grid[level][0]:, -n_grid[level][1]:])
                .transpose(0, 1, 2, 4, 3, 5)
                .reshape(updated_blk_data.shape[0], updated_blk_data.shape[1],
                    n_block[level][0] * n_grid[level][0],
                    n_block[level][1] * n_grid[level][1])
    )

    return updated_blk_data

initialize_dict = {'on':parallel_initialize,
                   'off':initialize_normal}
update_dict = {'on':parallel_update,
                   'off':update_normal}
interpolate_fine_to_coarse_dict = {'on':parallel_interpolate_fine_to_coarse,
                              'off':interpolate_fine_to_coarse_normal}
def initialize(level, blk_data, blk_info, criterion, dx, dy):
    return initialize_dict[parallel](level, blk_data, blk_info, criterion, dx, dy)

def update(level, blk_data, blk_info, criterion, dx, dy, prev_ref_blk_data, prev_ref_blk_info, max_blk_num):
    return update_dict[parallel](level, blk_data, blk_info, criterion, dx, dy, prev_ref_blk_data, prev_ref_blk_info, max_blk_num)

def interpolate_fine_to_coarse(level, blk_data, ref_blk_data, ref_blk_info):
    return interpolate_fine_to_coarse_dict[parallel](level, blk_data, blk_info, criterion, dx, dy, prev_ref_blk_data, prev_ref_blk_info, max_blk_num)
