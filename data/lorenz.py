from jax import lax
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler

from .utils import generate_diff_kernels

__all__ = ["generate_dataset"]


def generate_dataset(dt=1e-2, tmax=None, num_visible=2, num_der=2, raw_sol=False):
    if tmax is None:
        tmax = 100 + 4 * dt

    def lorenz(t, y0, sigma, beta, rho):
        """Lorenz equations"""
        u, v, w = y0[..., 0], y0[..., 1], y0[..., 2]
        up = -sigma * (u - v)
        vp = rho * u - v - u * w
        wp = -beta * w + u * v
        return np.stack((up, vp, wp), axis=-1)

    # Lorenz paramters and initial conditions
    sigma, beta, rho = 10, 8 / 3.0, 28
    u0, v0, w0 = 0, 1, 1.05

    # Integrate the Lorenz equations on the time grid t
    print("Generating Lorenz system dataset...")
    t_eval = np.arange(0, tmax, dt)
    sol = solve_ivp(
        lorenz,
        (0, tmax),
        y0=np.stack((u0, v0, w0), axis=-1),
        t_eval=t_eval,
        args=(sigma, beta, rho),
    )
    data = sol.y[range(num_visible)]

    # Compute finite difference derivatives
    kernels = generate_diff_kernels(num_der)
    data = lax.conv(data[:, None, :], kernels[:, None, :], (1,), "VALID")

    # Rescale/normalize data
    reshaped_data = data.reshape(-1, data.shape[2])
    scaler = StandardScaler(with_mean=False)
    scaled_data = scaler.fit_transform(reshaped_data.T)
    scaled_data = scaled_data.reshape(
        scaled_data.shape[0], data.shape[0], data.shape[1]
    )

    return (
        scaled_data,
        scaler.scale_.reshape(num_visible, num_der + 1),
        sol.y if raw_sol else None,
    )
