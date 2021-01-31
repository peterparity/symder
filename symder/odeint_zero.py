import jax
import jax.numpy as jnp
from functools import partial

__all__ = ['odeint_zero', 'dfunc', 'd_dt']

@partial(jax.custom_jvp, nondiff_argnums=(0,))
def odeint_zero(func, y0, t, *args):
    return y0

@odeint_zero.defjvp
def odeint_zero_jvp(func, primals, tangents):
    y0, t, *args = primals
    dy0, dt, *dargs = tangents
    y = odeint_zero(func, y0, t, *args)
    dydt = func(y, t, *args)
    return y, dy0 + dydt * jnp.broadcast_to(dt, dydt.shape) # needs to be explicitly broadcasted

def dfunc(func, order, transform=None):
    func0 = partial(odeint_zero, func) if transform is None else \
            lambda y0, t, *args: transform(odeint_zero(func, y0, t, *args))

    # TODO: Can potentially replace this with jax.experimental.jet (when it is better supported)
    #       for more efficient third or higher order derivatives
    out = [func0]
    for _ in range(order):
        out.append(d_dt(out[-1]))
    return out

def d_dt(func):
    def dfunc_dt(y0, t, *args):
        dy0 = jax.tree_map(jnp.zeros_like, y0)
        dt = jax.tree_map(jnp.ones_like, t)
        dargs = jax.tree_map(jnp.zeros_like, args)
        return jax.jvp(func, (y0, t, *args), (dy0, dt, *dargs))[1]
    return dfunc_dt