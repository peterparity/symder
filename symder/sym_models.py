import jax.numpy as jnp
import numpy as np

import haiku as hk

__all__ = [
    "Quadratic",
    "PointwisePolynomial",
    "SpatialDerivative1D",
    "SpatialDerivative2D",
    "SymModel",
    "rescale_z",
]


class Quadratic(hk.Module):
    def __init__(self, n_dims, init=jnp.zeros):
        super().__init__()
        self.n_dims = n_dims

        ind = np.arange(n_dims)
        mesh = np.stack(np.meshgrid(ind, ind), -1)
        self.mask = jnp.array(mesh[..., 0] >= mesh[..., 1])

        self.init = lambda *args: self.mask * init(*args)

    def __call__(self, z, t=None):
        weights = self.mask * hk.get_parameter(
            "w", (self.n_dims, self.n_dims, self.n_dims), init=self.init
        )
        out = (weights * z[..., None, None, :] * z[..., None, :, None]).sum((-2, -1))
        return out


class PointwisePolynomial(hk.Module):
    def __init__(
        self, poly_terms=(2, 4), init=jnp.zeros, name="pointwise_polynomial",
    ):
        super().__init__(name=name)
        self.init = init
        self.poly_terms = poly_terms

    def __call__(self, z, t=None):
        w = hk.get_parameter("w", (len(self.poly_terms),), init=self.init)
        terms = jnp.stack([z ** n for n in self.poly_terms], axis=-1)
        return jnp.sum(w * terms, axis=-1)


class SpatialDerivative1D(hk.Module):
    def __init__(
        self,
        mesh,
        dx,
        deriv_orders=(1, 2),
        init=jnp.zeros,
        name="spatial_derivative_1d",
    ):
        super().__init__(name=name)
        self.init = init

        k = 2 * np.pi * np.fft.fftfreq(mesh, d=dx)[:, None]

        # for use in odd derivatives
        k_1 = k.copy()
        if mesh % 2 == 0:
            k_1[int(mesh / 2), :] = 0

        self.ik_vec = jnp.stack(
            [(1j * k) ** n if (n % 2 == 0) else (1j * k_1) ** n for n in deriv_orders],
            axis=-1,
        )

    def __call__(self, u, t=None):
        v = jnp.fft.fft(u, axis=-2)
        w = hk.get_parameter("w", (u.shape[-1], self.ik_vec.shape[-1]), init=self.init)
        L = jnp.sum(w * self.ik_vec, axis=-1)
        du = jnp.fft.ifft(L * v, axis=-2)
        return jnp.real(du) if jnp.isrealobj(u) else du


class SpatialDerivative2D(hk.Module):
    def __init__(self, mesh, dx, init=jnp.zeros, name="spatial_derivative_2d"):
        super().__init__(name=name)
        self.init = init

        kx = 2 * np.pi * np.fft.fftfreq(mesh, d=dx)[:, None, None]
        ky = 2 * np.pi * np.fft.fftfreq(mesh, d=dx)[None, :, None]

        # for use in odd derivatives
        kx_1 = kx.copy()
        ky_1 = ky.copy()
        if mesh % 2 == 0:
            kx_1[int(mesh / 2), :, :] = 0
            ky_1[:, int(mesh / 2), :] = 0

        kx = jnp.broadcast_to(kx, (mesh, mesh, 1))
        ky = jnp.broadcast_to(ky, (mesh, mesh, 1))
        kx_1 = jnp.broadcast_to(kx_1, (mesh, mesh, 1))
        ky_1 = jnp.broadcast_to(ky_1, (mesh, mesh, 1))

        self.ik_vec = jnp.stack(
            [
                1j * kx_1,
                1j * ky_1,
                (1j * kx) ** 2,
                (1j * ky) ** 2,
                (1j * kx_1) * (1j * ky_1),
            ],
            axis=-1,
        )

    def __call__(self, u, t=None):
        v = jnp.fft.fft2(u, axes=(-3, -2))
        w = hk.get_parameter("w", (u.shape[-1], self.ik_vec.shape[-1]), init=self.init)
        L = jnp.sum(w * self.ik_vec, axis=-1)
        du = jnp.fft.ifft2(L * v, axes=(-3, -2))
        return jnp.real(du) if jnp.isrealobj(u) else du


class SymModel(hk.Module):
    def __init__(self, dt, module_list, time_dependence=False, name="sym_model"):
        super().__init__(name=name)
        self.dt = dt
        self.module_list = tuple(module_list)
        self.time_dependence = time_dependence

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
