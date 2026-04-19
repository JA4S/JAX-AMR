# Copyright © 2026 Haocheng Wen, Faxuan Luo, Hanbing Zou
# SPDX-License-Identifier: MIT

import jax
import jax.numpy as jnp
import warp as wp

from jaxamr import amr

cfd_float = wp.float32

@wp.func
def flux(rho: cfd_float, rhou: cfd_float, rhov: cfd_float, E: cfd_float, gamma: cfd_float) -> tuple[wp.vec4, wp.vec4]:
    u = rhou / rho
    v = rhov / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * (u*u + v*v))
    F = wp.vec4(rho*u, rho*u*u + p, rho*u*v, u*(E+p))
    G = wp.vec4(rho*v, rho*u*v, rho*v*v + p, v*(E+p))
    return F, G


@wp.func
def lax_friedrichs_flux(
    U_L: wp.array(dtype=cfd_float, ndim=1),
    U_R: wp.array(dtype=cfd_float, ndim=1),
    gamma: cfd_float
) -> tuple[wp.vec4, wp.vec4]:

    rho_L = U_L[0]
    rhou_L = U_L[1]
    rhov_L = U_L[2]
    E_L = U_L[3]

    rho_R = U_R[0]
    rhou_R = U_R[1]
    rhov_R = U_R[2]
    E_R = U_R[3]

    F_L, G_L = flux(rho_L, rhou_L, rhov_L, E_L, gamma)
    F_R, G_R = flux(rho_R, rhou_R, rhov_R, E_R, gamma)

    u_L = rhou_L / rho_L
    p_L = (gamma - 1.0) * (E_L - 0.5 * rho_L * u_L*u_L)
    c_L = wp.sqrt(gamma * p_L / rho_L)
    lambda_L = wp.abs(u_L) + c_L

    u_R = rhou_R / rho_R
    p_R = (gamma - 1.0) * (E_R - 0.5 * rho_R * u_R*u_R)
    c_R = wp.sqrt(gamma * p_R / rho_R)
    lambda_R = wp.abs(u_R) + c_R

    lambda_max = wp.max(lambda_L, lambda_R)
    UL = wp.vec4(rho_L, rhou_L, rhov_L, E_L)
    UR = wp.vec4(rho_R, rhou_R, rhov_R, E_R)

    Fn = 0.5 * (F_L + F_R) - 0.5 * lambda_max * (UR - UL)
    Gn = 0.5 * (G_L + G_R) - 0.5 * lambda_max * (UR - UL)
    return Fn, Gn

@wp.kernel
def rhs_kernel(
    U: wp.array(dtype=cfd_float, ndim=3),
    out: wp.array(dtype=cfd_float, ndim=3),
    dx: cfd_float, dy: cfd_float, gamma: cfd_float, nx: int, ny: int
):
    i, j = wp.tid()
    if i <= 0 or i >= nx-1 or j <=0 or j >= ny-1:
        for idx in range(out.shape[0]):
            out[idx, i, j] = 0.0
        return

    Fx1, _ = lax_friedrichs_flux(U[:,i-1,j], U[:,i,j], gamma)
    Fx2, _ = lax_friedrichs_flux(U[:,i,j], U[:,i+1,j], gamma)
    _, Gy1 = lax_friedrichs_flux(U[:,i,j-1], U[:,i,j], gamma)
    _, Gy2 = lax_friedrichs_flux(U[:,i,j], U[:,i,j+1], gamma)

    res = (Fx2 - Fx1)/(-dx) + (Gy2 - Gy1)/(-dy)
    for idx in range(out.shape[0]):
        out[idx, i, j] = res[idx]



@wp.kernel
def vmap_rhs_kernel(
    U: wp.array(dtype=cfd_float, ndim=4),
    out: wp.array(dtype=cfd_float, ndim=4),
    dx: cfd_float, dy: cfd_float, gamma: cfd_float, nx: int, ny: int
):
    n, i, j = wp.tid()

    if i <= 0 or i >= nx-1 or j <=0 or j >= ny-1:
        for idx in range(out.shape[1]):
            out[n, idx, i, j] = 0.0
        return

    Fx1, _ = lax_friedrichs_flux(U[n,:,i-1,j], U[n,:,i,j], gamma)
    Fx2, _ = lax_friedrichs_flux(U[n,:,i,j], U[n,:,i+1,j], gamma)
    _, Gy1 = lax_friedrichs_flux(U[n,:,i,j-1], U[n,:,i,j], gamma)
    _, Gy2 = lax_friedrichs_flux(U[n,:,i,j], U[n,:,i,j+1], gamma)

    res = (Fx2 - Fx1)/(-dx) + (Gy2 - Gy1)/(-dy)

    for idx in range(out.shape[1]):
        out[n, idx, i, j] = res[idx]


@wp.kernel
def add_scaled(
    c: cfd_float, dt: cfd_float,
    x: wp.array(dtype=cfd_float), y: wp.array(dtype=cfd_float),
    out: wp.array(dtype=cfd_float)
):
    i = wp.tid()
    out[i] = x[i] + c * dt * y[i]


def rhs(U, dx, dy):
    out = wp.zeros_like(U)
    nx, ny = U.shape[1], U.shape[2]
    wp.launch(rhs_kernel, dim=(nx, ny), inputs=[U, out, dx, dy, 1.4, nx, ny])
    return out


def vamp_rhs(U, dx, dy):
    out = wp.zeros_like(U)
    nb, nx, ny = U.shape[0], U.shape[2], U.shape[3]
    wp.launch(vmap_rhs_kernel, dim=(nb, nx, ny), inputs=[U, out, dx, dy, 1.4, nx, ny])
    return out


def rk2(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info):
    num = template_node_num

    def step(U, dU, s):
        out = wp.zeros_like(U)
        wp.launch(add_scaled, dim=U.size, inputs=[s, dt, U.flatten(), dU.flatten(), out.flatten()])
        return amr.update_external_boundary(level, blk_data, wp.to_jax(out)[..., num:-num, num:-num], ref_blk_info)

    
    U = wp.from_jax(amr.get_ghost_block_data(ref_blk_data, ref_blk_info))
    b1 = step(U, vamp_rhs(U, dx, dy), 0.5)
    
    U1 = wp.from_jax(amr.get_ghost_block_data(b1, ref_blk_info))
    b2 = step(U, vamp_rhs(U1, dx, dy), 1.0)

    return b2


def rk2_L0(U, dx, dy, dt):

    U = wp.from_jax(U[0])

    dU1 = rhs(U, dx, dy)
    U1 = wp.zeros_like(U)
    wp.launch(add_scaled, dim=U1.size, inputs=[0.5, dt, U.flatten(), dU1.flatten(), U1.flatten()])

    dU2 = rhs(U1, dx, dy)
    U2 = wp.zeros_like(U)
    wp.launch(add_scaled, dim=U2.size, inputs=[1.0, dt, U.flatten(), dU2.flatten(), U2.flatten()])

    U2 = jnp.array([wp.to_jax(U2)])
    return U2
