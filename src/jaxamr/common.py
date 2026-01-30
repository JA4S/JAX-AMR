# Copyright © 2025 Haocheng Wen, Faxuan Luo, Hanbing Zou
# SPDX-License-Identifier: MIT

import jax.numpy as jnp
from jax import jit

@jit
def remove_ghost(blk_data):
    num = 3
    return blk_data[...,num:-num,num:-num]
@jit
def save_ghost(blk_data):
    num = 3
    upper = blk_data[:,:,:,:num]
    lower = blk_data[:,:,:,-num:]
    left = blk_data[:,:,:num,num:-num]
    right = blk_data[:,:,-num:,num:-num]
    return upper,lower,left,right

@jit 
def add_ghost(blk_data,upper,lower,left,right):
    num = 3
    padded_horizontal = jnp.concatenate([left, blk_data, right], axis=2)
    ghost_blk_data = jnp.concatenate([upper, padded_horizontal, lower], axis=3)
    return ghost_blk_data