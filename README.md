# **SymDer**: **Sym**bolic **Der**ivative Network for *Discovering Sparse Interpretable Dynamics from Partial Observations*

Implementation of a machine learning method for identifying the governing equations of a nonlinear dynamical system using using only partial observations. Our machine learning framework combines an encoder for state reconstruction with a sparse symbolic model. In order to train our model by matching time derivatives, we implement an algorithmic trick (see `symder/odeint_zero.py`) for taking higher order derivatives of a variable that is implicitly defined by a differential equation (i.e. the symbolic model).

Please cite "**Discovering sparse interpretable dynamics from partial observations**" (https://doi.org/10.1038/s42005-022-00987-z) and see the paper for more details. This is the official repository for the paper.

## Requirements

JAX >= 0.2.8, Haiku >= 0.0.4, scikit-learn, NumPy, SciPy

## Usage

Data generation scripts are contained in `data/`. Encoder models and related tools are contained in `encoder/`. Symbolic models and the tools for taking higher order symbolic time derivatives are contained in `symder/`. The individual `*_model.py` files provide examples of how to use our method on a variety of ODE and PDE systems.