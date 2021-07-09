import jax
import jax.numpy as jnp

__all__ = ['append_dzdt', 'concat_visible', 'normalize_by_magnitude', 'to_complex']


def append_dzdt(encoder_apply, finite_difference=False):
    if finite_difference:
        def encoder_with_dzdt(params, x, *args, **kwargs):
            z_hidden = encoder_apply(params, x, *args, **kwargs)
            dz_hidden = jnp.diff(z_hidden, axis=1)
            dz_hidden = jnp.concatenate((dz_hidden, dz_hidden[:, [-1]]), axis=1)
            return z_hidden, dz_hidden  # jnp.gradient(z_hidden, axis=1)
    else:
        def encoder_with_dzdt(params, x, dxdt):
            zero_params = jax.tree_map(jnp.zeros_like, params)

            # dz/dt = dz/dx * dx/dt
            z_hidden, dzdt_hidden = jax.jvp(
                encoder_apply, (params, x), (zero_params, dxdt))
            return z_hidden, dzdt_hidden

    return encoder_with_dzdt


def concat_visible(encoder_apply, visible_transform=None):
    def encoder_concat_visible(params, x, *args, **kwargs):
        z_visible = (visible_transform(x)
                     if visible_transform is not None else x)
        z_hidden, *out_args = encoder_apply(params, x, *args, **kwargs)
        z = jnp.concatenate((z_visible, z_hidden), axis=-1)
        return z, *out_args
    return encoder_concat_visible


def normalize_by_magnitude(encoder_apply, pad=None, squared=False):
    def encoder_normalized(params, x, *args, **kwargs):
        z_phase = encoder_apply(params, x, *args, **kwargs)
        z_phase = z_phase/jnp.linalg.norm(z_phase, axis=-1, keepdims=True)

        if pad is not None and pad > 0:
            x = x[:, pad:-pad]
        z_mag = jnp.sqrt(x) if squared else x

        return z_mag * z_phase
    return encoder_normalized


def to_complex(encoder_apply):
    def encoder_complex(params, x, *args, **kwargs):
        z, *out_args = encoder_apply(params, x, *args, **kwargs)
        return z[..., [0]] + 1j * z[..., [1]], *out_args
    return encoder_complex
