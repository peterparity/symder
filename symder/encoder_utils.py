import jax
import jax.numpy as jnp

__all__ = ['append_dzdt', 'concat_visible']

def append_dzdt(encoder_apply):
    def encoder_with_dzdt(params, x, dxdt):
        zero_params = jax.tree_map(jnp.zeros_like, params)
        z_hidden, dzdt_hidden = jax.jvp(encoder_apply, (params, x), (zero_params, dxdt)) # dz/dt = dz/dx * dx/dt
        return z_hidden, dzdt_hidden
    return encoder_with_dzdt

def concat_visible(encoder_apply, visible_transform=None):
    def encoder_concat_visible(params, x, *args, **kwargs):
        z_visible = visible_transform(x) if visible_transform is not None else x 
        z_hidden, *out_args = encoder_apply(params, x, *args, **kwargs)
        z = jnp.concatenate((z_visible, z_hidden), axis=-1)
        return z, *out_args
    return encoder_concat_visible