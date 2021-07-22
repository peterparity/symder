import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

import haiku as hk
import optax

import os.path
import argparse
from functools import partial

# from tqdm.auto import tqdm

from data.utils import get_dataset
from data.diff_source_2d import generate_dataset

from encoder.utils import append_dzdt, concat_visible
from symder.sym_models import SymModel, Quadratic, SpatialDerivative2D, rescale_z
from symder.symder import get_symder_apply, get_model_apply

from utils import loss_fn, init_optimizers, save_pytree  # , load_pytree


def get_model(num_visible, num_hidden, num_der, mesh, dx, dt, scale, get_dzdt=False):

    # Define encoder
    hidden_size = 64
    pad = 2

    def encoder(x):
        return hk.Sequential(
            [
                lambda x: jnp.pad(
                    x, ((0, 0), (0, 0), (pad, pad), (pad, pad), (0, 0)), "wrap"
                ),
                hk.Conv3D(hidden_size, kernel_shape=5, padding="VALID"),
                jax.nn.relu,
                hk.Conv3D(hidden_size, kernel_shape=1),
                jax.nn.relu,
                hk.Conv3D(num_hidden, kernel_shape=1),
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
            1e1 * dt,
            (
                SpatialDerivative2D(mesh, np.sqrt(1e1) * dx, init=jnp.zeros),
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
    multi_gpu=False,
):

    # JIT compile/PMAP gradient function
    loss_fn_apply = partial(loss_fn, model_apply, **loss_fn_args)
    if multi_gpu:
        # Take mean of gradient across multiple devices
        def grad_loss(params, batch, target):
            grad_out = jax.grad(loss_fn_apply, has_aux=True)(params, batch, target)
            return lax.pmean(grad_out, axis_name="devices")

        grad_loss = jax.pmap(grad_loss, axis_name="devices")
    else:
        grad_loss = jax.jit(jax.grad(loss_fn_apply, has_aux=True))

    # Initialize sparse mask
    sparsify = sparse_thres is not None and sparse_interval is not None
    if multi_gpu:
        sparse_mask = jax.tree_map(
            jax.pmap(lambda x: jnp.ones_like(x, dtype=bool)), params["sym_model"]
        )
    else:
        sparse_mask = jax.tree_map(
            lambda x: jnp.ones_like(x, dtype=bool), params["sym_model"]
        )

    # Initialize optimizers
    update_params, opt_state = init_optimizers(params, optimizers, sparsify, multi_gpu)
    if multi_gpu:
        update_params = jax.pmap(update_params, axis_name="devices")
    else:
        update_params = jax.jit(update_params)

    # Get batch and target
    # TODO: replace this with call to a data generator/data loader
    if multi_gpu:
        n_devices = jax.device_count()
        pad = data_args["pad"]
        time_size = (scaled_data.shape[0] - 2 * pad) // n_devices
        batch = []
        target = []
        for i in range(n_devices):
            start, end = i * time_size, (i + 1) * time_size + 2 * pad
            if loss_fn_args["reg_dzdt"] is not None:
                # batch, time, mesh, mesh, num_visible, 2
                batch.append(scaled_data[None, start:end, :, :, :, :2])
            else:
                # batch, time, mesh, mesh, num_visible
                batch.append(scaled_data[None, start:end, :, :, :, 0])
            # batch, time, mesh, mesh, num_visible, num_der
            target.append(scaled_data[None, start + pad : end - pad, :, :, :, 1:])

        batch = jax.device_put_sharded(batch, jax.devices())
        target = jax.device_put_sharded(target, jax.devices())

    else:
        if loss_fn_args["reg_dzdt"] is not None:
            # batch, time, mesh, mesh, num_visible, 2
            batch = scaled_data[None, :, :, :, :, :2]
        else:
            # batch, time, mesh, mesh, num_visible
            batch = scaled_data[None, :, :, :, :, 0]
        pad = data_args["pad"]
        # batch, time, mesh, mesh, num_visible, num_der
        target = scaled_data[None, pad:-pad, :, :, :, 1:]

        batch = jnp.asarray(batch)
        target = jnp.asarray(target)

    # Training loop
    if multi_gpu:
        print(f"Training for {n_steps} steps on {n_devices} devices...")
    else:
        print(f"Training for {n_steps} steps...")

    best_loss = np.float("inf")
    best_params = None

    def thres_fn(x):
        return jnp.abs(x) > sparse_thres

    if multi_gpu:
        thres_fn = jax.pmap(thres_fn)

    for step in range(n_steps):
        # Compute gradients and losses
        grads, loss_list = grad_loss(params, batch, target)

        # Save best params if loss is lower than best_loss
        loss = loss_list[0][0] if multi_gpu else loss_list[0]
        if loss < best_loss:
            best_loss = loss
            best_params = jax.tree_map(lambda x: x.copy(), params)

        # Update sparse_mask based on a threshold
        if sparsify and step > 0 and step % sparse_interval == 0:
            sparse_mask = jax.tree_map(thres_fn, best_params["sym_model"])

        # Update params based on optimizers
        params, opt_state, sparse_mask = update_params(
            grads, opt_state, params, sparse_mask
        )

        # Print loss
        if step % 100 == 0:
            loss, mse, reg_dzdt, reg_l1_sparse = loss_list
            if multi_gpu:
                (loss, mse, reg_dzdt, reg_l1_sparse) = (
                    loss[0],
                    mse[0],
                    reg_dzdt[0],
                    reg_l1_sparse[0],
                )
            print(
                f"Loss[{step}] = {loss}, MSE = {mse}, "
                f"Reg. dz/dt = {reg_dzdt}, Reg. L1 Sparse = {reg_l1_sparse}"
            )
            if multi_gpu:
                print(jax.tree_map(lambda x: x[0], params["sym_model"]))
            else:
                print(params["sym_model"])

    if multi_gpu:
        best_params = jax.tree_map(lambda x: x[0], best_params)
        sparse_mask = jax.tree_map(lambda x: x[0], sparse_mask)

    print("\nBest loss:", best_loss)
    print("Best sym_model params:", best_params["sym_model"])
    return best_loss, best_params, sparse_mask


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run SymDer model on 2D diffusion with source data."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./diff_source_2d_run0/",
        help="Output folder path. Default: ./diff_source_2d_run0/",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="./data/diff_source_2d.npz",
        help=(
            "Path to 2D diffusion with source dataset (generated and saved "
            "if it does not exist). Default: ./data/diff_source_2d.npz"
        ),
    )
    args = parser.parse_args()

    # Seed random number generator
    key_seq = hk.PRNGSequence(42)

    # Set SymDer parameters
    num_visible = 1
    num_hidden = 1
    num_der = 2

    # Set dataset parameters and load/generate dataset
    sys_size = 64
    mesh = 64
    dt = 5e-2
    tspan = (0, 50 + 2 * dt)
    scaled_data, scale, raw_sol = get_dataset(
        args.dataset,
        generate_dataset,
        get_raw_sol=True,
        sys_size=sys_size,
        mesh=mesh,
        dt=dt,
        tspan=tspan,
        num_der=num_der,
    )

    # Set training hyperparameters
    n_steps = 50000
    sparse_thres = 5e-3
    sparse_interval = 1000
    multi_gpu = True

    # Define optimizers
    optimizers = {
        "encoder": optax.adabelief(1e-4, eps=1e-16),
        "sym_model": optax.adabelief(1e-4, eps=1e-16),
    }

    # Set loss function hyperparameters
    loss_fn_args = {
        "scale": jnp.array(scale),
        "deriv_weight": jnp.array([1.0, 10.0]),
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
        num_visible,
        num_hidden,
        num_der,
        mesh,
        sys_size / mesh,
        dt,
        scale,
        get_dzdt=get_dzdt,
    )

    # Initialize parameters
    params = {}
    params["encoder"] = model_init["encoder"](
        next(key_seq), jnp.ones([1, scaled_data.shape[0], mesh, mesh, num_visible])
    )
    params["sym_model"] = model_init["sym_model"](
        next(key_seq), jnp.ones([1, 1, mesh, mesh, num_visible + num_hidden]), 0.0
    )
    if multi_gpu:
        for name in params.keys():
            params[name] = jax.device_put_replicated(params[name], jax.devices())

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
        multi_gpu=multi_gpu,
    )

    # Save model parameters and sparse mask
    print(f"Saving best model parameters in output folder: {args.output}")
    save_pytree(
        os.path.join(args.output, "best.pt"),
        {"params": best_params, "sparse_mask": sparse_mask},
    )
