from jax import lax
import numpy as np
from numpy.fft import fftfreq, fft, ifft
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler

from .utils import solve_ETDRK4, generate_diff_kernels

__all__ = ["generate_dataset"]


def generate_dataset(
    sys_size=2 * np.pi,
    mesh=64,
    dt=1e-3,  # 10 * 1e-4
    tspan=None,
    pool=16,
    tpool=10,
    num_der=2,
    seed=0,
    squared=False,
    raw_sol=False,
):
    if tspan is None:
        tspan = (0, 0.5 + 4 * dt)

    out_mesh, mesh = mesh, pool * mesh
    dt /= tpool

    k = 2 * np.pi * fftfreq(mesh, d=sys_size / mesh)

    # Initial condition
    np.random.seed(seed)
    krange = 1.0
    envelope = np.exp(-1 / (2 * krange ** 2) * k ** 2)
    np.random.seed(0)
    v0 = envelope * (
        np.random.normal(loc=0, scale=1.0, size=(1, mesh))
        + 1j * np.random.normal(loc=0, scale=1.0, size=(1, mesh))
    )
    u0 = ifft(v0)
    u0 = (
        np.sqrt(2 * mesh) * u0 / np.linalg.norm(u0, axis=(-2, -1), keepdims=True)
    )  # normalize
    v0 = fft(u0)

    # Differential equation definition
    L = -0.5j * k ** 2

    def N(v):
        u = ifft(v)
        kappa = -1
        return -1j * kappa * fft(np.abs(u) ** 2 * u)

    # Solve using ETDRK4 method
    print("Generating 1D nonlinear Schrödinger dataset...")
    sol_u = solve_ETDRK4(L, N, v0, tspan, dt, lambda v: ifft(v))
    sol_u = resample(sol_u[::tpool], out_mesh, axis=-1)
    if squared:
        data = (np.abs(sol_u[:, 0]) ** 2).reshape(-1, 1 * out_mesh)
    else:
        data = np.abs(sol_u[:, 0]).reshape(-1, 1 * out_mesh)
    data = data.T

    # Compute finite difference derivatives
    kernels = generate_diff_kernels(num_der)
    data = lax.conv(data[:, None, :], kernels[:, None, :], (1,), "VALID")
    # time, out_mesh, num_visible, num_der+1
    data = data[None, ...].transpose((3, 1, 0, 2))

    # Rescale/normalize data
    reshaped_data = data.reshape(-1, data.shape[2] * data.shape[3])
    scaler = StandardScaler(with_mean=False)
    scaler.fit(reshaped_data)
    scaler.scale_ /= scaler.scale_[0]
    scaled_data = scaler.transform(reshaped_data)
    # time, out_mesh, num_visible, num_der+1
    scaled_data = scaled_data.reshape(-1, out_mesh, 1, num_der + 1)

    return (
        scaled_data,
        scaler.scale_.reshape(1, num_der + 1),
        sol_u if raw_sol else None,
    )
