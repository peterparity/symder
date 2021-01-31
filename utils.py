import jax
import jax.numpy as jnp
from jax.tree_util import pytree

import pickle
from pathlib import Path
from typing import Union

__all__ = ['save_pytree', 'load_pytree', 'loss_fn']

suffix = '.pickle'

def save_pytree(path: Union[str, Path], data: pytree, overwrite: bool = False):
    path = Path(path)
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load_pytree(path: Union[str, Path]) -> pytree:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    if path.suffix != suffix:
        raise ValueError(f'Not a {suffix} file: {path}')
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def loss_fn(model_apply, params, batch, target, 
            scale=None, deriv_weight=None, reg_dzdt=None, reg_l1_sparse=None):
    num_der = target.shape[-1]

    if scale is None:
        scale = jnp.ones(1, num_der+1)
    if deriv_weight is None:
        deriv_weight = jnp.ones(num_der)

    if reg_dzdt is not None:
        x = batch[..., 0]
        dxdt = batch[..., 1] * scale[:,1]/scale[:,0]
        sym_deriv_x, z_hidden, dzdt_hidden, sym_dzdt_hidden = model_apply(params, x, dxdt)
    else:
        sym_deriv_x, z_hidden = model_apply(params, batch)

    scaled_sym_deriv_x = sym_deriv_x * scale[:,[0]]/scale[:,1:] # scale to normed derivatives

    # MSE loss
    mse_loss = jnp.sum(deriv_weight * jnp.mean(((target - scaled_sym_deriv_x)**2
                                           ).reshape(-1, num_der), axis=0))
    loss_list = [mse_loss]

    # dz/dt regularization loss
    if reg_dzdt is not None:
        num_hidden = dzdt_hidden.shape[-1]
        reg_dzdt_loss = reg_dzdt * jnp.mean((dzdt_hidden - sym_dzdt_hidden)**2 
                                            / jnp.var(z_hidden.reshape(-1, num_hidden), axis=0))
        loss_list.append(reg_dzdt_loss)

    # L1 sparse regularization loss
    if reg_l1_sparse is not None:
        reg_l1_sparse_loss = reg_l1_sparse * \
                        jax.tree_util.tree_reduce(lambda x, y: x + jnp.abs(y).sum(), 
                            params['sym_model'], 0.)
        loss_list.append(reg_l1_sparse_loss)

    loss = sum(loss_list)
    loss_list.insert(0, loss)
    return loss, loss_list