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
from data.nlse_1d import generate_dataset

from encoder.embedding import DirectEmbedding
from encoder.utils import append_dzdt, normalize_by_magnitude, to_complex
from symder.sym_models import (
    SymModel,
    PointwisePolynomial,
    SpatialDerivative1D,
    SpatialDerivative1D_FiniteDiff,
    rescale_z,
)
from symder.symder import get_symder_apply, get_model_apply

from utils import loss_fn_weighted, init_optimizers, save_pytree  # , load_pytree


def get_model(
    num_visible, num_hidden, num_der, mesh, dx, dt, scale, squared=False, get_dzdt=False
):
    # Define encoder
    pad = None

    def encoder(x, *args):
        return DirectEmbedding(
            (mesh, num_hidden),
            init=lambda *x: jnp.ones(*x) + 0.1 * hk.initializers.RandomNormal()(*x),
        )(x, *args)

    # # Angle encoder
    # def encoder(x, *args):
    #     theta = DirectEmbedding(
    #         (mesh, 1), init=jnp.zeros,  # hk.initializers.RandomNormal(),
    #     )(x, *args)
    #     return jnp.concatenate((jnp.cos(theta), jnp.sin(theta)), axis=-1)

    encoder = hk.without_apply_rng(hk.transform(encoder))
    encoder_apply = normalize_by_magnitude(encoder.apply, pad=pad, squared=squared)
    if get_dzdt:
        encoder_apply = append_dzdt(encoder_apply, finite_difference=True)
    encoder_apply = to_complex(encoder_apply)

    # Define symbolic model
    scale_vec = jnp.sqrt(scale[:, 0]) if squared else jnp.asarray(scale[:, 0])

    @partial(rescale_z, scale_vec=scale_vec)
    def sym_model(z, t):
        return SymModel(
            -1j * (1e1 * dt),
            (
                SpatialDerivative1D(
                    mesh, dx, deriv_orders=(1, 2, 3, 4), init=jnp.zeros
                ),
                lambda u: u
                * PointwisePolynomial(poly_terms=(2, 4, 6, 8), init=jnp.zeros)(
                    jnp.abs(u)
                ),
            ),
        )(z, t)

    sym_model = hk.without_apply_rng(hk.transform(sym_model))

    # Define SymDer function which automatically computes
    # higher order time derivatives of symbolic model
    symder_apply = get_symder_apply(
        sym_model.apply,
        num_der=num_der,
        transform=(lambda z: jnp.abs(z) ** 2) if squared else jnp.abs,
        get_dzdt=get_dzdt,
    )

    # Define full model, combining encoder and symbolic model
    model_apply = get_model_apply(
        encoder_apply,
        symder_apply,
        hidden_transform=lambda z: jnp.concatenate((jnp.real(z), jnp.imag(z)), axis=-1),
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
    loss_fn_apply = partial(loss_fn_weighted, model_apply, **loss_fn_args)
    if multi_gpu:
        # Take mean of gradient across multiple devices
        def grad_loss(params, batch, target):
            grad_out = jax.grad(loss_fn_apply, has_aux=True)(params, batch, target)
            return lax.pmean(grad_out, axis_name="devices")

        grad_loss = jax.pmap(grad_loss, axis_name="devices")
    else:
        grad_loss = jax.jit(jax.grad(loss_fn_apply, has_aux=True))
        # grad_loss = jax.grad(loss_fn_apply, has_aux=True)

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
                # batch, time, mesh, num_visible, 2
                batch.append(scaled_data[None, start:end, :, :, :2])
            else:
                # batch, time, mesh, num_visible
                batch.append(scaled_data[None, start:end, :, :, 0])
            # batch, time, mesh, num_visible, num_der
            target.append(scaled_data[None, start + pad : end - pad, :, :, 1:])

        batch = jax.device_put_sharded(batch, jax.devices())
        target = jax.device_put_sharded(target, jax.devices())

    else:
        if loss_fn_args["reg_dzdt"] is not None:
            # batch, time, mesh, mesh, num_visible, 2
            batch = scaled_data[None, :, :, :, :2]
        else:
            # batch, time, mesh, num_visible
            batch = scaled_data[None, :, :, :, 0]

        # batch, time, mesh, num_visible, num_der
        target = scaled_data[None, :, :, :, 1:]

        # scaled_data = scaled_data[:100].reshape(100 // 100, 100, *scaled_data.shape[1:])
        # if loss_fn_args["reg_dzdt"] is not None:
        #     # batch, time, mesh, num_visible, 2
        #     batch = scaled_data[:, :, :, :, :2]
        # else:
        #     # batch, time, mesh, num_visible
        #     batch = scaled_data[:, :, :, :, 0]

        # # batch, time, mesh, num_visible, num_der
        # target = scaled_data[:, :, :, :, 1:]

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
        # weight = jnp.exp(-0.05 * jnp.arange(batch.shape[1]))[:, None, None, None]
        weight = jnp.ones_like(batch[..., [0]])  # 1 / (0.1 + batch[..., [0]])
        grads, loss_list = grad_loss(params, batch, target, weight)

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

        # Normalize encoder params
        flat_params, tree = jax.tree_flatten(params["encoder"])
        flat_params[0] /= jnp.linalg.norm(flat_params[0], axis=-1, keepdims=True)
        params["encoder"] = jax.tree_unflatten(tree, flat_params)

        # # TEMPORARY: Fix params
        # flat_params, tree = jax.tree_flatten(params["sym_model"])
        # flat_params[0] = jnp.array([[0.0, -0.05]])
        # flat_params[1] = jnp.array([-0.1, 0.0])
        # params["sym_model"] = jax.tree_unflatten(tree, flat_params)

        # Print loss
        if step % 1000 == 0:
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

            save_pytree(
                os.path.join(args.output, f"{step:06}.pt"),
                {"params": best_params, "sparse_mask": sparse_mask},
                overwrite=True,
            )

    if multi_gpu:
        best_params = jax.tree_map(lambda x: x[0], best_params)
        sparse_mask = jax.tree_map(lambda x: x[0], sparse_mask)

    print("\nBest loss:", best_loss)
    print("Best sym_model params:", best_params["sym_model"])
    return best_loss, best_params, sparse_mask


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run SymDer model on 1D nonlinear Schrödinger data."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./nlse_1d_run0/",
        help="Output folder path. Default: ./nlse_1d_run0/",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="./data/nlse_1d_raw.npz",
        help=(
            "Path to 1D nonlinear Schrödinger dataset (generated and saved if it does not "
            "exist). Default: ./data/nlse_1d.npz"
        ),
    )
    args = parser.parse_args()

    # Seed random number generator
    key_seq = hk.PRNGSequence(42)

    # Set SymDer parameters
    num_visible = 1
    num_hidden = 2
    num_der = 2

    # Set dataset parameters and load/generate dataset
    sys_size = 2 * np.pi
    mesh = 64
    dt = 1e-3
    tspan = (0, 0.5 + 4 * dt)
    squared = False
    scaled_data, scale, raw_sol = get_dataset(
        args.dataset,
        generate_dataset,
        get_raw_sol=True,
        sys_size=sys_size,
        mesh=mesh,
        dt=dt,
        tspan=tspan,
        num_der=num_der,
        squared=squared,
    )
    print("raw_sol shape ", raw_sol.shape)

    # Set training hyperparameters
    n_steps = 100000
    sparse_thres = 1e-3
    sparse_interval = 10000
    multi_gpu = False

    # Define optimizers
    optimizers = {
        "encoder": optax.adabelief(1e-4, eps=1e-16),
        "sym_model": optax.adabelief(1e-4, eps=1e-16),
    }

    # Set loss function hyperparameters
    loss_fn_args = {
        "scale": jnp.array(scale),
        "deriv_weight": jnp.array([1.0, 1.0]),
        "reg_dzdt": 1e3,
        "reg_dzdt_var_norm": False,
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
        squared=squared,
        get_dzdt=get_dzdt,
    )

    # Initialize parameters
    params = {}
    params["encoder"] = model_init["encoder"](
        next(key_seq), jnp.ones([1, scaled_data.shape[0], mesh, num_visible])
    )
    # params["encoder"] = model_init["encoder"](
    #     next(key_seq), jnp.ones([100 // 100, 100, mesh, num_visible])
    # )
    # # Initialize encoder to true solution at time = 0
    # flat_params, tree = jax.tree_flatten(params["encoder"])
    # sol = raw_sol[1:-1].reshape(1, -1, 64)
    # sol = sol / np.abs(sol)
    # sol = jnp.stack((sol.real, sol.imag), axis=-1)
    # flat_params[0] = sol[:, :100]  # jnp.repeat(sol[:, [0]], 30, axis=1)
    # params["encoder"] = jax.tree_unflatten(tree, flat_params)

    # flat_params, tree = jax.tree_flatten(params["encoder"])
    # theta = jnp.arange(0, 2 * np.pi, step=2 * np.pi / mesh)
    # flat_params[0] *= np.stack((jnp.cos(theta), jnp.sin(theta)), axis=-1)
    # params["encoder"] = jax.tree_unflatten(tree, flat_params)

    params["sym_model"] = model_init["sym_model"](
        next(key_seq), jnp.ones([1, 1, mesh, num_hidden // 2], dtype=jnp.complex64), 0.0
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

    # # Save model parameters and sparse mask
    # print(f"Saving best model parameters in output folder: {args.output}")
    # save_pytree(
    #     os.path.join(args.output, "best.pt"),
    #     {"params": best_params, "sparse_mask": sparse_mask},
    # )
