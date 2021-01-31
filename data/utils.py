# import jax.numpy as jnp
from jax import lax
import numpy as np

import os.path
from tqdm.auto import tqdm

__all__ = ['solve_ETDRK4', 'generate_diff_kernels', 'save_dataset', 'load_dataset']

def solve_ETDRK4(L, N, v0, tspan, dt, output_func):
    """ETDRK4 method"""
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.)

    contour_radius = 1
    M = 16
    r = contour_radius*np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)

    LR = dt * L
    LR = np.expand_dims(LR, axis=-1) + r

    Q = dt*np.real(np.mean((np.exp(LR/2.)-1)/LR, axis=-1))
    f1 = dt*np.real(np.mean((-4. - LR + np.exp(LR)*(4. - 3.*LR + LR**2))/LR**3, axis=-1))
    f2 = dt*np.real(np.mean((2. + LR + np.exp(LR)*(-2. + LR))/LR**3, axis=-1))
    f3 = dt*np.real(np.mean((-4. - 3.*LR - LR**2 + np.exp(LR)*(4. - LR))/LR**3, axis=-1))

    u = []
    v = v0
    for t in tqdm(np.arange(tspan[0], tspan[1], dt)):
        u.append(output_func(v))

        Nv = N(v)
        a = E2 * v + Q * Nv
        Na = N(a)
        b = E2 * v + Q * Na
        Nb = N(b) 
        c = E2 * a + Q * (2.0*Nb - Nv)
        Nc = N(c)   
        v = E*v + Nv*f1 + 2.0*(Na + Nb)*f2 + Nc*f3

    return np.stack(u)

def generate_diff_kernels(order):
    p = int(np.floor((order+1)/2))

    rev_d1 = np.array((0.5, 0., -0.5))
    d2 = np.array((1., -2., 1.))

    even_kernels = [np.pad(np.array((1.,)), (p,))]
    for i in range(order//2):
        even_kernels.append(np.convolve(even_kernels[-1], d2, mode='same'))

    even_kernels = np.stack(even_kernels)
    odd_kernels = lax.conv(even_kernels[:,None,:], rev_d1[None,None,:], (1,), 'SAME').squeeze()

    kernels = np.stack((even_kernels, odd_kernels), axis=1).reshape(-1, 2*p+1)
    if np.fmod(order, 2) == 0:
        kernels = kernels[:-1]

    return kernels

def get_dataset(filename, generate_dataset, *args, **kwargs):
    if os.path.isfile(filename):
        scaled_data, scale = load_dataset(filename)
        # TODO: Check if input args match args from loaded dataset
    else:
        scaled_data, scale = generate_dataset(*args, **kwargs)
        # TODO: Save args
        save_dataset(filename, scaled_data, scale)
    return scaled_data, scale

def save_dataset(filename, scaled_data, scale, raw_sol=None):
    if not os.path.isfile(filename):
        print(f'Saving dataset to file: {filename}')
        if raw_sol is not None:
            np.savez(filename, scaled_data=scaled_data, scale=scale, raw_sol=raw_sol)
        else:
            np.savez(filename, scaled_data=scaled_data, scale=scale)
    else:
        raise FileExistsError(f'{filename} already exists! Dataset is not saved.')

def load_dataset(filename, raw_sol=False):
    print(f'Loading dataset from file: {filename}')
    dataset = np.load(filename)
    if raw_sol:
        return dataset['scaled_data'], dataset['scale'], dataset['raw_sol']
    else:
        return dataset['scaled_data'], dataset['scale']