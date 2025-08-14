# Copyright © 2025 Haocheng Wen
# SPDX-License-Identifier: MIT

import jax
import jax.numpy as jnp
from jax import jit, vmap


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


def analriem(rl, ul, pl, rr, ur, pr, gamma = 1.4):
    '''
    Simple analriem solver. Only for non-slip condition
    '''
    rho_avg = jnp.sqrt(rl * rr)
    u_avg = (jnp.sqrt(rl)*ul + jnp.sqrt(rr)*ur) / (jnp.sqrt(rl) + jnp.sqrt(rr))
    p_avg = (pl + pr) / 2.0

    mom_flux = rho_avg * jnp.power(u_avg, 2) + p_avg

    zero_flux = jnp.zeros_like(rl)

    return jnp.array([zero_flux, mom_flux, zero_flux, zero_flux])


def compute_hyp_wallflux(U, normal, gamma=1.4):
    '''
    Compute the hyperbolic flux at the embedded boundary
    '''
    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho
    E = U[3]
    p = (gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))

    un = u * normal[0] + v * normal[1]
    vn = jnp.zeros_like(rho)
    En = p / (gamma - 1) + 0.5 * rho * (un**2 + vn**2)

    flux = analriem(
        rl=rho, ul=un, pl=p,
        rr=rho, ur=-un, pr=p
    )

    flux_x = flux[1] * normal[0]
    flux_y = flux[1] * normal[1]

    zero_flux = jnp.zeros_like(rho)

    hyp_wallflux = jnp.array([zero_flux, flux_x, flux_y, zero_flux])

    return hyp_wallflux


def redistribute_flux(rhs_val, afr, cell_type, nbr_type):
    '''
    Redistribute flux for the small volum cell
    '''

    is_not_solid = jnp.sign(1 - cell_type)

    nbr = jnp.sign(1 - nbr_type[0])

    @jit
    def conv_3x3(val):
        result = (
            jnp.roll(val, -1, axis=0) * nbr[3] +
            jnp.roll(val, 1, axis=0) * nbr[1] +
            jnp.roll(val, -1, axis=1) * nbr[0] +
            jnp.roll(val, 1, axis=1) * nbr[2] +
            jnp.roll(val, shift=(-1, -1), axis=(0, 1)) * (nbr[3] * nbr[0]) +
            jnp.roll(val, shift=(-1, 1), axis=(0, 1)) * (nbr[3] * nbr[2]) +
            jnp.roll(val, shift=(1, -1), axis=(0, 1)) * (nbr[1] * nbr[0]) +
            jnp.roll(val, shift=(1, 1), axis=(0, 1)) * (nbr[1] * nbr[2])
            )

        return result

    # delta_M
    vtot = vmap(conv_3x3, in_axes=0)(afr)
    divnc = vmap(conv_3x3, in_axes=0)(rhs_val * afr) / vtot
    divnc = jnp.nan_to_num(divnc)

    optmp = (1 - afr) * (divnc - rhs_val)
    optmp = jnp.nan_to_num(optmp) * is_not_solid
    delm = - afr * optmp

    # weight
    rediswgt = jnp.ones_like(afr)

    # redistribution
    wtot = 1.0 / vmap(conv_3x3, in_axes=0)(afr * rediswgt)
    wtot = jnp.nan_to_num(wtot) * is_not_solid

    drho = delm * wtot * rediswgt
    optmp = optmp + drho
    rhs_val = rhs_val + optmp

    return rhs_val

@jit
def rhs(U, dx, dy, cell_info, gamma=1.4):

    cell_type = cell_info['cell_type'][None, ...]
    nbr = cell_info['neighbor_type'][None, ...]
    lfr = cell_info['fluid_edge_length_fraction'][None, ...]
    afr = cell_info['fluid_area_fraction'][None, ...]
    eb_l = cell_info['cut_length'][None, ...]
    eb_n = cell_info['cut_face_normal']

    flux_x, flux_y, flux_eb_hyp = jnp.zeros_like(U), jnp.zeros_like(U), jnp.zeros_like(U)

    U_L_x = U # U_i
    U_R_x = jnp.roll(U, -1, axis=1) # U_i+1
    U_L_y = U
    U_R_y = jnp.roll(U, -1, axis=2)

    F_LR_x, _ = lax_friedrichs_flux(U_L_x, U_R_x, gamma)
    _, G_LR_y = lax_friedrichs_flux(U_L_y, U_R_y, gamma)

    F_LR_x = jnp.nan_to_num(F_LR_x)
    G_LR_y = jnp.nan_to_num(G_LR_y)

    F_L = jnp.roll(F_LR_x, 1, axis=1) # F_i-1/2,j
    F_R = F_LR_x # F_i+1/2,j
    flux = - (F_R * lfr[:, 1] - F_L * lfr[:, 3]) / dx
    flux_x = flux_x.at[:, 1:-1, :].set(flux[:, 1:-1, :])

    G_L = jnp.roll(G_LR_y, 1, axis=2) # G_i,j-1/2
    G_R = G_LR_y # G_i,j+1/2
    flux = - (G_R * lfr[:, 2] - G_L * lfr[:, 0]) / dy
    flux_y = flux_y.at[:, :, 1:-1].set(flux[:, :, 1:-1])

    flux = - compute_hyp_wallflux(U, - eb_n) * eb_l / (dx * dy)
    flux_eb_hyp = flux_eb_hyp.at[:, 1:-1, 1:-1].set(flux[:, 1:-1, 1:-1])

    rhs_val = jnp.nan_to_num((flux_x + flux_y + flux_eb_hyp) / afr) 

    rhs_val = redistribute_flux(rhs_val, afr, cell_type, nbr) 

    return rhs_val


def initialize(x_min, y_min, Lx, Ly, nx, ny, cell_type, gamma=1.4):
    x = jnp.linspace(x_min + Lx/nx/2, Lx - Lx/nx/2, nx)
    y = jnp.linspace(y_min + Ly/ny/2, Ly - Ly/ny/2, ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    is_not_solid = jnp.sign(1 - cell_type)

    rho = jnp.where(X < 1.0, 1.0, 0.125)
    u = jnp.zeros_like(X)
    v = jnp.zeros_like(X)
    p = jnp.where(X < 1.0, 3.0, 0.1)
    E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)

    U = jnp.array([rho, rho * u, rho * v, E]) * is_not_solid[None, ...]

    return X, Y, U


def CFL(U, cfl, cell_info, gamma=1.4):

    rho = U[0:1,:,:]
    u = U[1:2,:,:]/rho
    v = U[2:3,:,:]/rho
    rhoE = U[3:4,:,:]
    p = (rhoE - 0.5*rho*(u**2+v**2))*(gamma-1)
    a = jnp.sqrt(gamma*p/rho)
    c = jnp.nanmax(jnp.abs(u) + a)

    return cfl / c * dx


@jit
def rk2(U, dx, dy, cfl, cell_info):

    dt = CFL(U, cfl, cell_info)
    U1 = U + 0.5 * dt * rhs(U, dx, dy, cell_info)
    U2 = U + dt * rhs(U1, dx, dy, cell_info)

    return U2
