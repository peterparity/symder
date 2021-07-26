# **SymDer**: **Sym**bolic **Der**ivative Approach to *Discovering Sparse Interpretable Dynamics from Partial Observations*

Implementation of a machine learning method for identifying the governing equations of a nonlinear dynamical system using using only partial observations. Our machine learning framework combines an encoder for state reconstruction with a sparse symbolic model. In order to train our model by matching time derivatives, we implement an algorithmic trick (see `symder/odeint_zero.py`) for taking higher order derivatives of a variable that is implicitly defined by a differential equation (i.e. the symbolic model).

See "**Discovering Sparse Interpretable Dynamics from Partial Observations**" (https://arxiv.org/abs/2107.10879) for more details. This is the official repository for the paper.

## Requirements
JAX, NumPy

## Usage
The individual `*_model.py` files provide examples of how to use our method on a variety of ODE and PDE systems. Data generation scripts are provided for each of the systems in the `data/` folder.
