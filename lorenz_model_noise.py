import jax
import jax.numpy as jnp
import numpy as np

import haiku as hk
import optax

import os.path
import argparse
from functools import partial

# from tqdm.auto import tqdm

from data.utils import get_dataset
from data.lorenz_noise import generate_dataset

from encoder.utils import append_dzdt, concat_visible
from symder.sym_models import SymModel, Quadratic, rescale_z
from symder.symder import get_symder_apply, get_model_apply

from utils import loss_fn, init_optimizers, save_pytree  # , load_pytree


def get_model(num_visible, num_hidden, num_der, dt, scale, get_dzdt=False):

    # Define encoder
    hidden_size = 128
    pad = 4

    def encoder(x):
        return hk.Sequential(
            [
                hk.Conv1D(hidden_size, kernel_shape=9, padding="VALID"),
                jax.nn.relu,
                hk.Conv1D(hidden_size, kernel_shape=1),
                jax.nn.relu,
                hk.Conv1D(num_hidden, kernel_shape=1),
            ]
        )(x)

    encoder = hk.without_apply_rng(hk.transform(encoder))
    encoder_apply = append_dzdt(encoder.apply) if get_dzdt else encoder.apply
    encoder_apply = concat_visible(
        encoder_apply, visible_transform=lambda x: x[:, pad:-pad]
    )

    # Define symbolic model
    n_dims = num_visible + num_hidden
    scale_vec = jnp.concatenate((scale[:, 0], jnp.ones(num_hidden)))

    @partial(rescale_z, scale_vec=scale_vec)
    def sym_model(z, t):
        return SymModel(
            1e2 * dt,
            (
                hk.Linear(n_dims, w_init=jnp.zeros, b_init=jnp.zeros),
                Quadratic(n_dims, init=jnp.zeros),
            ),
        )(z, t)

    sym_model = hk.without_apply_rng(hk.transform(sym_model))

    # Define SymDer function which automatically computes
    # higher order time derivatives of symbolic model
    symder_apply = get_symder_apply(
        sym_model.apply,
        num_der=num_der,
        transform=lambda z: z[..., :num_visible],
        get_dzdt=get_dzdt,
    )

    # Define full model, combining encoder and symbolic model
    model_apply = get_model_apply(
        encoder_apply,
        symder_apply,
        hidden_transform=lambda z: z[..., -num_hidden:],
        get_dzdt=get_dzdt,
    )
    model_init = {"encoder": encoder.init, "sym_model": sym_model.init}

    return model_apply, model_init, {"pad": pad}


def train(
    n_steps,
    model_apply,
    params,
    scaled_data,
    loss_fn_args={},
    data_args={},
    optimizers={},
    sparse_thres=None,
    sparse_interval=None,
    key_seq=hk.PRNGSequence(42),
):

    # JIT compile gradient function
    loss_fn_apply = partial(loss_fn, model_apply, **loss_fn_args)
    grad_loss = jax.jit(jax.grad(loss_fn_apply, has_aux=True))

    # Initialize sparse mask
    sparsify = sparse_thres is not None and sparse_interval is not None
    sparse_mask = jax.tree_map(
        lambda x: jnp.ones_like(x, dtype=bool), params["sym_model"]
    )

    # # TEMPORARY: Init mask
    # flat_mask, tree = jax.tree_flatten(sparse_mask)
    # flat_mask[0] = jnp.array([False, False, True])
    # flat_mask[1] = jnp.array(
    #     [[True, True, False], [True, True, False], [False, False, True]]
    # )
    # flat_mask[2] = jnp.array(
    #     [
    #         [[False, False, False], [False, False, False], [False, False, False]],
    #         [[False, False, True], [False, False, False], [False, False, False]],
    #         [[False, True, False], [False, False, False], [False, False, False]],
    #     ]
    # )
    # sparse_mask = jax.tree_unflatten(tree, flat_mask)

    # Initialize optimizers
    update_params, opt_state = init_optimizers(params, optimizers, sparsify)
    update_params = jax.jit(update_params)

    # Get batch and target
    # TODO: replace this with call to a data generator/data loader
    if loss_fn_args["reg_dzdt"] is not None:
        batch = scaled_data[None, :, :, :2]  # batch, time, num_visible, 2
    else:
        batch = scaled_data[None, :, :, 0]  # batch, time, num_visible
    pad = data_args["pad"]
    # batch, time, num_visible, num_der
    target = scaled_data[None, pad:-pad, :, 1:]

    batch = jnp.asarray(batch)
    target = jnp.asarray(target)

    # Training loop
    print(f"Training for {n_steps} steps...")

    best_loss = np.float("inf")
    best_params = None

    for step in range(n_steps):

        # Compute gradients and losses
        grads, loss_list = grad_loss(params, batch, target)

        # Save best params if loss is lower than best_loss
        loss = loss_list[0]
        if loss < best_loss:
            best_loss = loss
            best_params = jax.tree_map(lambda x: x.copy(), params)

        # Update sparse_mask based on a threshold
        if sparsify and step > 0 and step % sparse_interval == 0:
            sparse_mask = jax.tree_map(
                lambda x: jnp.abs(x) > sparse_thres, best_params["sym_model"]
            )

        # Update params based on optimizers
        params, opt_state, sparse_mask = update_params(
            grads, opt_state, params, sparse_mask
        )

        # Print loss
        if step % 1000 == 0:
            loss, mse, reg_dzdt, reg_l1_sparse = loss_list
            print(
                f"Loss[{step}] = {loss}, MSE = {mse}, "
                f"Reg. dz/dt = {reg_dzdt}, Reg. L1 Sparse = {reg_l1_sparse}"
            )
            print(params["sym_model"])

    print("\nBest loss:", best_loss)
    print("Best sym_model params:", best_params["sym_model"])
    return best_loss, best_params, sparse_mask


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run SymDer model on Lorenz system data with added noise."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./lorenz_noise_run0/",
        help="Output folder path. Default: ./lorenz_noise_run0/",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="./data/lorenz_noise.npz",
        help=(
            "Path to Lorenz system dataset (generated and saved "
            "if it does not exist). Default: ./data/lorenz_noise.npz"
        ),
    )
    parser.add_argument(
        "-v",
        "--visible",
        type=int,
        nargs="+",
        default=[0, 1],
        help="List of visible variables (0, 1, and/or 2). Default: 0 1",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Gaussian noise standard deviation. Default: 0.0",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        nargs="+",
        default=None,
        help="Smoothing window size, polynomial order. Default: None",
    )
    args = parser.parse_args()

    # Seed random number generator
    key_seq = hk.PRNGSequence(42)

    # Set SymDer parameters
    num_visible = len(args.visible)  # 2
    num_hidden = 3 - num_visible  # 1
    num_der = 2

    # Set dataset parameters and load/generate dataset
    dt = 1e-2
    tmax = 100 + 2 * dt
    scaled_data, scale, raw_sol = get_dataset(
        args.dataset,
        generate_dataset,
        get_raw_sol=True,
        dt=dt,
        tmax=tmax,
        num_visible=num_visible,
        visible_vars=args.visible,
        num_der=num_der,
        noise=args.noise,
        smoothing_params=args.smooth,
    )

    # Set training hyperparameters
    n_steps = 50000
    sparse_thres = 1e-3
    sparse_interval = 5000

    # Define optimizers
    optimizers = {
        "encoder": optax.adabelief(1e-3, eps=1e-16),
        "sym_model": optax.adabelief(1e-3, eps=1e-16),
    }

    # Set loss function hyperparameters
    loss_fn_args = {
        "scale": jnp.array(scale),
        "deriv_weight": jnp.array([1.0, 1.0]),
        "reg_dzdt": 0,
        "reg_l1_sparse": 0,
    }
    get_dzdt = loss_fn_args["reg_dzdt"] is not None

    # Check dataset shapes
    assert scaled_data.shape[-2] == num_visible
    assert scaled_data.shape[-1] == num_der + 1
    assert scale.shape[0] == num_visible
    assert scale.shape[1] == num_der + 1

    # Define model
    model_apply, model_init, model_args = get_model(
        num_visible, num_hidden, num_der, dt, scale, get_dzdt=get_dzdt
    )

    # Initialize parameters
    params = {}
    params["encoder"] = model_init["encoder"](
        next(key_seq), jnp.ones([1, scaled_data.shape[0], num_visible])
    )
    params["sym_model"] = model_init["sym_model"](
        next(key_seq), jnp.ones([1, 1, num_visible + num_hidden]), 0.0
    )

    # Train
    best_loss, best_params, sparse_mask = train(
        n_steps,
        model_apply,
        params,
        scaled_data,
        loss_fn_args=loss_fn_args,
        data_args={"pad": model_args["pad"]},
        optimizers=optimizers,
        sparse_thres=sparse_thres,
        sparse_interval=sparse_interval,
        key_seq=key_seq,
    )

    # Save model parameters and sparse mask
    print(f"Saving best model parameters in output folder: {args.output}")
    save_pytree(
        os.path.join(args.output, "best.pt"),
        {"params": best_params, "sparse_mask": sparse_mask},
    )
