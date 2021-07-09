import jax.numpy as jnp
import haiku as hk

__all__ = ["DirectEmbedding"]


class DirectEmbedding(hk.Module):
    def __init__(self, shape, init=jnp.zeros, concat_visible=False, get_dzdt=False):
        super().__init__()
        self.shape = shape
        self.init = init
        self.concat_visible = concat_visible
        self.get_dzdt = get_dzdt

    def __call__(self, x, *args):
        z_hidden = hk.get_parameter(
            "z_hidden", (x.shape[0], x.shape[1], *self.shape), init=self.init
        )

        z = jnp.concatenate((x, z_hidden), axis=-1) if self.concat_visible else z_hidden

        if self.get_dzdt:
            dz = jnp.diff(z, axis=1)
            dz = jnp.concatenate((dz, dz[:, [-1]]), axis=1)
            return z, dz
        else:
            return z
        # return z if not self.get_dzdt else (z, jnp.gradient(z_hidden, axis=1))

