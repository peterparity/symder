# import jax.numpy as jnp
from jax import lax
import numpy as np
from numpy.fft import fftfreq, fft2, ifft2
from sklearn.preprocessing import StandardScaler

from .utils import solve_ETDRK4, generate_diff_kernels

__all__ = ['generate_dataset']

def generate_dataset(l=64, mesh=64, dt=5e-2, tspan=None, init_mode='rect', seed=0, raw_sol=False):
    if tspan is None:
        tspan = (0, 50 + 4*dt)

    kx = np.expand_dims(2*np.pi * fftfreq(mesh, d=l/mesh), axis=-1)
    ky = np.expand_dims(2*np.pi * fftfreq(mesh, d=l/mesh), axis=0)

    ## Initial condition
    np.random.seed(seed)
    if init_mode == 'fourier':
        krange = 5
        envelope = np.exp(-1/(2*krange**2) * (kx**2 + ky**2) )
        v0 = envelope * (np.random.normal(loc=0, scale=1.0, size=(2, mesh, mesh)) 
                                + 1j*np.random.normal(loc=0, scale=1.0, size=(2, mesh, mesh)))
        u0 = np.real(ifft2(v0))
        # u0 = 0.5 + 0.1 * mesh * u0/np.linalg.norm(u0, axis=(-2,-1), keepdims=True) # normalize
        u0 = 0.55 + 0.45 * u0/np.max(np.abs(u0), axis=(-2,-1), keepdims=True) # normalize
        u0[..., 0, :, :] = 0.5
    elif init_mode == 'rect':
        n_rects = 50
        u0 = 0.5 * np.ones((2, mesh, mesh))
        rect_pos = (np.random.uniform(0, l, size=(n_rects, 2))*mesh/l).astype(int)
        rect_size = (np.random.uniform(0, 0.2*l, size=(n_rects, 2))*mesh/l).astype(int)
        rect_value = np.random.uniform(0.1, 1, size=(n_rects,))
        for i in range(n_rects):
            rect = np.zeros((mesh, mesh), dtype=bool)
            rect[:rect_size[i, 0], :rect_size[i, 1]] = True
            rect = np.roll(np.roll(rect, rect_pos[i, 0], axis=0), rect_pos[i, 1], axis=1)
            u0[1, :, :] = u0[1, :, :]*(1-rect) + rect_value[i]*rect
    else:
        raise ValueError(f"init_mode = '{init_mode}' is not valid. init_mode must be in ['fourier', 'rect']")

    ## Differential equation definition
    D2 = -(kx**2 + ky**2)
    L = np.stack((0.05 * D2, 0.1 * D2))

    def N(v):
        u = np.real(ifft2(v))
        u1 = u[...,0,:,:]
        u2 = u[...,1,:,:]
        du = np.stack([7/3. * u1 - 8/3. * u1*u2, 
                       -u2 + u1*u2], axis=-3)
        return fft2(du)

    ## Solve using ETDRK4 method
    print("Generating 2D reaction-diffusion (diffusive Lotka-Volterra) dataset...")
    sol_u = solve_ETDRK4(L, N, fft2(u0), tspan, dt, lambda v: np.real(ifft2(v))) 
    data = sol_u[:,0].reshape(sol_u.shape[0], 1*mesh**2)
    data = data.T

    ## Compute finite difference derivatives
    num_der = 2
    kernels = generate_diff_kernels(num_der)
    data = lax.conv(data[:,None,:], kernels[:,None,:], (1,), 'VALID')
    data = data[None, ...].transpose((3, 1, 0, 2)) # time, mesh**2, num_visible, num_der+1

    ## Rescale/normalize data
    reshaped_data = data.reshape(-1, data.shape[2]*data.shape[3])
    scaler = StandardScaler(with_mean=False)
    scaler.fit(reshaped_data)
    scaler.scale_[0] = 1
    scaled_data = scaler.transform(reshaped_data)
    scaled_data = scaled_data.reshape(-1, mesh, mesh, 1, num_der+1) # time, mesh, mesh, num_visible, num_der+1

    if raw_sol:
        return scaled_data, scaler.scale_.reshape(1, num_der+1), sol_u

    return scaled_data, scaler.scale_.reshape(1, num_der+1)