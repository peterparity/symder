import jax.numpy as jnp
from .odeint_zero import dfunc

__all__ = ["get_symder_apply", "get_model_apply"]


def get_symder_apply(sym_model_apply, num_der=1, transform=None, get_dzdt=False):
    def func(z, t, params):
        return sym_model_apply(params, z, t)

    dfuncs = dfunc(func, num_der, transform=transform)[1:]

    def symder_apply(params, z, t=0.0):
        sym_deriv_x = jnp.stack(tuple(map(lambda f: f(z, t, params), dfuncs)), axis=-1)
        if get_dzdt:
            sym_dzdt = func(z, t, params)
            return sym_deriv_x, sym_dzdt
        else:
            return sym_deriv_x

    return symder_apply


def get_model_apply(
    encoder_apply,
    symder_apply,
    encoder_name="encoder",
    sym_model_name="sym_model",
    hidden_transform=None,
    get_dzdt=False,
):
    if get_dzdt:

        def model_apply(params, x, dxdt):
            z, dzdt_hidden = encoder_apply(params[encoder_name], x, dxdt)
            sym_deriv_x, sym_dzdt = symder_apply(params[sym_model_name], z)
            if hidden_transform is not None:
                z_hidden = hidden_transform(z)
                sym_dzdt_hidden = hidden_transform(sym_dzdt)
            else:
                z_hidden = z
                sym_dzdt_hidden = sym_dzdt
            return sym_deriv_x, z_hidden, dzdt_hidden, sym_dzdt_hidden

    else:

        def model_apply(params, x):
            z, *_ = encoder_apply(params[encoder_name], x)
            sym_deriv_x = symder_apply(params[sym_model_name], z)
            z_hidden = hidden_transform(z) if hidden_transform is not None else z
            return sym_deriv_x, z_hidden

    return model_apply
