import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import pytree

import optax

import pickle
from pathlib import Path
from typing import Union

__all__ = ["save_pytree", "load_pytree", "loss_fn", "init_optimizers"]

suffix = ".pt"


def save_pytree(path: Union[str, Path], data: pytree, overwrite: bool = False):
    path = Path(path)
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise FileExistsError(f"File {path} already exists.")
    with open(path, "wb") as file:
        pickle.dump(data, file)


def load_pytree(path: Union[str, Path]) -> pytree:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    if path.suffix != suffix:
        raise ValueError(f"Not a {suffix} file: {path}")
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data


def loss_fn(
    model_apply,
    params,
    batch,
    target,
    scale=None,
    deriv_weight=None,
    reg_dzdt=None,
    reg_dzdt_var_norm=True,
    reg_l1_sparse=None,
    sym_model_name="sym_model",
):
    num_der = target.shape[-1]

    if scale is None:
        scale = jnp.ones(1, num_der + 1)
    if deriv_weight is None:
        deriv_weight = jnp.ones(num_der)

    if reg_dzdt is not None:
        x = batch[..., 0]
        dxdt = batch[..., 1] * scale[:, 1] / scale[:, 0]
        sym_deriv_x, z_hidden, dzdt_hidden, sym_dzdt_hidden = model_apply(
            params, x, dxdt
        )
    else:
        sym_deriv_x, z_hidden = model_apply(params, batch)

    # scale to normed derivatives
    scaled_sym_deriv_x = sym_deriv_x * scale[:, [0]] / scale[:, 1:]

    # MSE loss
    mse_loss = jnp.sum(
        deriv_weight
        * jnp.mean(((target - scaled_sym_deriv_x) ** 2).reshape(-1, num_der), axis=0)
    )
    loss_list = [mse_loss]

    # dz/dt regularization loss
    if reg_dzdt is not None:
        num_hidden = dzdt_hidden.shape[-1]
        if reg_dzdt_var_norm:
            reg_dzdt_loss = reg_dzdt * jnp.mean(
                (dzdt_hidden - sym_dzdt_hidden) ** 2
                / jnp.var(z_hidden.reshape(-1, num_hidden), axis=0)
            )
        else:
            reg_dzdt_loss = reg_dzdt * jnp.mean((dzdt_hidden - sym_dzdt_hidden) ** 2)
        loss_list.append(reg_dzdt_loss)

    # L1 sparse regularization loss
    if reg_l1_sparse is not None:
        reg_l1_sparse_loss = reg_l1_sparse * jax.tree_util.tree_reduce(
            lambda x, y: x + jnp.abs(y).sum(), params[sym_model_name], 0.0
        )
        loss_list.append(reg_l1_sparse_loss)

    loss = sum(loss_list)
    loss_list.insert(0, loss)
    return loss, loss_list


def loss_fn_weighted(
    model_apply,
    params,
    batch,
    target,
    weight,
    scale=None,
    deriv_weight=None,
    reg_dzdt=None,
    reg_dzdt_var_norm=True,
    reg_l1_sparse=None,
    sym_model_name="sym_model",
):
    num_der = target.shape[-1]

    if scale is None:
        scale = jnp.ones(1, num_der + 1)
    if deriv_weight is None:
        deriv_weight = jnp.ones(num_der)

    if reg_dzdt is not None:
        x = batch[..., 0]
        dxdt = batch[..., 1] * scale[:, 1] / scale[:, 0]
        sym_deriv_x, z_hidden, dzdt_hidden, sym_dzdt_hidden = model_apply(
            params, x, dxdt
        )
    else:
        sym_deriv_x, z_hidden = model_apply(params, batch)

    # scale to normed derivatives
    scaled_sym_deriv_x = sym_deriv_x * scale[:, [0]] / scale[:, 1:]

    # MSE loss
    mse_loss = jnp.sum(
        deriv_weight
        * jnp.mean(
            (weight * (target - scaled_sym_deriv_x) ** 2).reshape(-1, num_der), axis=0
        )
    )
    loss_list = [mse_loss]

    # dz/dt regularization loss
    if reg_dzdt is not None:
        num_hidden = dzdt_hidden.shape[-1]
        if reg_dzdt_var_norm:
            reg_dzdt_loss = reg_dzdt * jnp.mean(
                (dzdt_hidden - sym_dzdt_hidden) ** 2
                / jnp.var(z_hidden.reshape(-1, num_hidden), axis=0)
            )
        else:
            reg_dzdt_loss = reg_dzdt * jnp.mean(
                weight[..., 0] * (dzdt_hidden - sym_dzdt_hidden) ** 2
            )
        loss_list.append(reg_dzdt_loss)

    # L1 sparse regularization loss
    if reg_l1_sparse is not None:
        reg_l1_sparse_loss = reg_l1_sparse * jax.tree_util.tree_reduce(
            lambda x, y: x + jnp.abs(y).sum(), params[sym_model_name], 0.0
        )
        loss_list.append(reg_l1_sparse_loss)

    loss = sum(loss_list)
    loss_list.insert(0, loss)
    return loss, loss_list


def init_optimizers(
    params,
    optimizers,
    sparsify=False,
    multi_gpu=False,
    sym_model_name="sym_model",
    pmap_axis_name="devices",
):
    # Initialize optimizers
    opt_init, opt_update, opt_state = {}, {}, {}
    for name in params.keys():
        opt_init[name], opt_update[name] = optimizers[name]
        if multi_gpu:
            opt_state[name] = jax.pmap(opt_init[name])(params[name])
        else:
            opt_state[name] = opt_init[name](params[name])

    # Define update function
    def update_params(grads, opt_state, params, sparse_mask):
        if sparsify:
            grads[sym_model_name] = jax.tree_multimap(
                jnp.multiply, sparse_mask, grads[sym_model_name]
            )

        updates = {}
        for name in params.keys():
            updates[name], opt_state[name] = opt_update[name](
                grads[name], opt_state[name], params[name]
            )
        params = optax.apply_updates(params, updates)

        if sparsify:
            params[sym_model_name] = jax.tree_multimap(
                jnp.multiply, sparse_mask, params[sym_model_name]
            )

        # TODO: This may not be necessary or can at least be reduced in frequency
        if multi_gpu:
            # Ensure params, opt_state, sparse_mask are the same across all devices
            params = lax.pmean(params, axis_name=pmap_axis_name)
            opt_state, sparse_mask = lax.pmax(
                (opt_state, sparse_mask), axis_name=pmap_axis_name
            )

        return params, opt_state, sparse_mask

    return update_params, opt_state
