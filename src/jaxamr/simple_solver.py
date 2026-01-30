# Copyright © 2025 Haocheng Wen, Faxuan Luo
# SPDX-License-Identifier: MIT

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

from jaxamr import amr
from jaxamr import common
from jaxamr.common import remove_ghost,save_ghost,add_ghost

def flux(U, gamma=1.4):
    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho
    E = U[3]
    p = (gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))

    F = jnp.array([
        rho * u,
        rho * u**2 + p,
        rho * u * v,
        u * (E + p)
    ])

    G = jnp.array([
        rho * v,
        rho * u * v,
        rho * v**2 + p,
        v * (E + p)
    ])

    return F, G


def lax_friedrichs_flux(U_L, U_R, gamma=1.4):
    F_L, G_L = flux(U_L, gamma)
    F_R, G_R = flux(U_R, gamma)

    # jnp.nanmax is used instead of jnp.max
    lambda_max = jnp.nanmax(jnp.abs(U_L[1]/U_L[0]) + jnp.sqrt(gamma * (gamma - 1) * (U_L[3]/U_L[0] - 0.5 * (U_L[1]**2 + U_L[2]**2)/U_L[0]**2)))

    return 0.5 * (F_L + F_R) - 0.5 * lambda_max * (U_R - U_L), \
           0.5 * (G_L + G_R) - 0.5 * lambda_max * (U_R - U_L)

@jit
def rhs(U, dx, dy, gamma=1.4):

    U_L_x = U[:, :-1, :] 
    U_R_x = U[:, 1:, :]
    F_LR_x, _ = lax_friedrichs_flux(U_L_x, U_R_x, gamma)
    F_x = jnp.zeros_like(U)
    F_x = F_x.at[:, 1:-1, :].set(- (F_LR_x[:, 1:, :] - F_LR_x[:, :-1, :]) / dx)

    U_L_y = U[:, :, :-1]
    U_R_y = U[:, :, 1:]
    _, G_LR_y = lax_friedrichs_flux(U_L_y, U_R_y, gamma)
    F_y = jnp.zeros_like(U)
    F_y = F_y.at[:, :, 1:-1].set(- (G_LR_y[:, :, 1:] - G_LR_y[:, :, :-1]) / dy)

    return F_x + F_y


def initialize(nx, ny, gamma=1.4):
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    rho = jnp.where(jnp.sqrt((X - 0.5)**2 + (Y - 0.5)**2) < 0.15, 1.0, 0.125)
    u = jnp.zeros_like(X)
    v = jnp.zeros_like(X)
    p = jnp.where(jnp.sqrt((X - 0.5)**2 + (Y - 0.5)**2) < 0.15, 1.0, 0.1)
    E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)

    U = jnp.array([rho, rho * u, rho * v, E])
    return X, Y, U


@jit
def rk2(ghost_blk_data, dx, dy, dt,theta):
    
    upper,lower,left,right = save_ghost(ghost_blk_data)
    ghost_blk_data1 = ghost_blk_data + 0.5 * dt * vmap(rhs, in_axes=(0, None, None))(ghost_blk_data, dx, dy)
    ghost_blk_data1 = remove_ghost(ghost_blk_data1)
    ghost_blk_data1 = add_ghost(ghost_blk_data1,upper,lower,left,right)

    ghost_blk_data2 = ghost_blk_data + dt * vmap(rhs, in_axes=(0, None, None))(ghost_blk_data1, dx, dy)
    ghost_blk_data2 = remove_ghost(ghost_blk_data2)
    ghost_blk_data2 = add_ghost(ghost_blk_data2,upper,lower,left,right)

    return ghost_blk_data2


@jit
def rk2_L0(blk_data, dx, dy, dt,theta=None):

    U = blk_data[0]
    U_with_ghost = boundary_conditions_2D(U,theta)

    U1_with_ghost = U_with_ghost + 0.5 * dt * rhs(U_with_ghost, dx, dy)
    U2_with_ghost = U_with_ghost + dt * rhs(U1_with_ghost, dx, dy)

    U2 = remove_ghost(U2_with_ghost)
    return jnp.array([U2])

def pad_2D(U):
    field = U
    field_periodic_x = jnp.concatenate([field[:,-4:-3,:],field[:,-3:-2,:],field[:,-2:-1,:],field,field[:,1:2,:],field[:,2:3,:],field[:,3:4,:]],axis=1)
    field_periodic_pad = jnp.concatenate([field_periodic_x[:,:,-4:-3],field_periodic_x[:,:,-3:-2],field_periodic_x[:,:,-2:-1],field_periodic_x,field_periodic_x[:,:,1:2],field_periodic_x[:,:,2:3],field_periodic_x[:,:,3:4]],axis=2)
    return field_periodic_pad

def replace_lb_2D(U_bd, padded_U):
    U = padded_U.at[:,0:3,3:-3].set(U_bd)
    return U
    
def replace_rb_2D(U_bd, padded_U):
    U = padded_U.at[:,-3:,3:-3].set(U_bd)
    return U

def replace_ub_2D(U_bd,padded_U):
    U = padded_U.at[:,3:-3,-3:].set(U_bd)
    return U

def replace_bb_2D(U_bd, padded_U):  
    U = padded_U.at[:,3:-3,0:3].set(U_bd)
    return U

def zero_gradient_left(U_bd, theta=None):
    U_bd_ghost = jnp.concatenate([U_bd[:,2:3],U_bd[:,1:2],U_bd[:,0:1]],axis=1)
    return U_bd_ghost

def zero_gradient_right(U_bd, theta=None):
    U_bd_ghost = jnp.concatenate([U_bd[:,-1:],U_bd[:,-2:-1],U_bd[:,-3:-2]],axis=1)
    return U_bd_ghost

def zero_gradient_bottom(U_bd, theta=None):
    U_bd_ghost = jnp.concatenate([U_bd[:,:,2:3],U_bd[:,:,1:2],U_bd[:,:,0:1]],axis=2)
    return U_bd_ghost

def zero_gradient_top(U_bd, theta=None):
    U_bd_ghost = jnp.concatenate([U_bd[:,:,-1:],U_bd[:,:,-2:-1],U_bd[:,:,-3:-2]],axis=2)
    return U_bd_ghost

def left_boundary(padded_U,theta=None):
    U_lb = padded_U[:,3:6,3:-3]
    U_lb = zero_gradient_left(U_lb,theta)
    U_with_lb = replace_lb_2D(U_lb,padded_U)
    return U_with_lb

def right_boundary(padded_U,theta=None):
    U_rb = padded_U[:,-6:-3,3:-3]
    U_rb = zero_gradient_right(U_rb,theta)
    U_with_rb = replace_rb_2D(U_rb,padded_U)
    return U_with_rb

def bottom_boundary(padded_U,theta=None):
    U_bb = padded_U[:,3:-3,3:6]
    U_bb = zero_gradient_bottom(U_bb,theta)
    U_with_bb = replace_bb_2D(U_bb,padded_U)
    return U_with_bb

def top_boundary(padded_U,theta=None):
    U_ub = padded_U[:,3:-3,-6:-3]
    U_ub = zero_gradient_top(U_ub,theta)
    U_with_ub = replace_ub_2D(U_ub,padded_U)
    return U_with_ub

def boundary_conditions_2D(U, theta=None):
    U_periodic_pad = pad_2D(U)
    U_with_lb = left_boundary(U_periodic_pad, theta)
    U_with_rb = right_boundary(U_with_lb,theta)
    U_with_bb = bottom_boundary(U_with_rb,theta)
    U_with_ghost_cell = top_boundary(U_with_bb,theta)
    return U_with_ghost_cell
