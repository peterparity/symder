import jax
import jax.numpy as jnp
import numpy as np

import haiku as hk

__all__ = ['Quadratic', 'SymModel', 'rescale_z']

class Quadratic(hk.Module):
    def __init__(self, n_dims, init=jnp.zeros):
        super().__init__()
        self.n_dims = n_dims
        
        ind = np.arange(n_dims)
        mesh = np.stack(np.meshgrid(ind, ind), -1)
        self.mask = jnp.array(mesh[..., 0] >= mesh[..., 1])

        self.init = lambda *args: self.mask * init(*args)

    def __call__(self, z, t=None):
        weights = self.mask * hk.get_parameter("w", 
                                               (self.n_dims, self.n_dims, self.n_dims), 
                                               init=self.init)
        out = (weights * z[..., None, None, :] * z[..., None, :, None]).sum((-2, -1))
        return out

class FFT2(hk.Module):
    def __init__(self, mesh, dx, init=jnp.zeros, name='fft2'):
        super().__init__(name=name)
        self.init = init

        kx = 2*np.pi * np.fft.fftfreq(mesh, d=dx)[:, None, None]
        ky = 2*np.pi * np.fft.fftfreq(mesh, d=dx)[None, :, None]
        
        # for use in 1st derivative
        kx_1 = kx.copy()
        ky_1 = ky.copy()
        if mesh % 2 == 0:
            kx_1[int(mesh/2), :, :] = 0
            ky_1[:, int(mesh/2), :] = 0

        kx = jnp.broadcast_to(kx, (mesh, mesh, 1))
        ky = jnp.broadcast_to(ky, (mesh, mesh, 1))
        kx_1 = jnp.broadcast_to(kx_1, (mesh, mesh, 1))
        ky_1 = jnp.broadcast_to(ky_1, (mesh, mesh, 1))

        self.ik_vec = jnp.stack([1j*kx_1, 1j*ky_1, 
                                 (1j*kx)**2, (1j*ky)**2, 
                                 (1j*kx_1)*(1j*ky_1)], axis=-1)

    def __call__(self, u, t=None):
        v = jnp.fft.fft2(u, axes=(-3, -2))
        w = hk.get_parameter("w", (2, self.ik_vec.shape[-1]), init=self.init)
        L = jnp.sum(w * self.ik_vec, axis=-1)
        return jnp.real(jnp.fft.ifft2(L*v, axes=(-3, -2)))

class SymModel(hk.Module):
    def __init__(self, dt, module_list, name='sym_model'):
        super().__init__(name=name)
        self.dt = dt
        self.module_list = tuple(module_list)
        self.time_dependence = False

    def __call__(self, z, t):
        if self.time_dependence:
            dz = self.dt * sum(module(z, t) for module in self.module_list)
        else:
            dz = self.dt * sum(module(z) for module in self.module_list)
        return dz

def rescale_z(sym_model_apply, scale_vec=None):
    if scale_vec is None:
        return sym_model_apply
    def rescaled_sym_model(z, t, *args, **kwargs):
        return sym_model_apply(z * scale_vec, t, *args, **kwargs) / scale_vec
    return rescaled_sym_model