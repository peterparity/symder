# **SymDer**: **Sym**bolic **Der**ivative Network for *Discovering Sparse Interpretable Dynamics from Partial Observations*

Implementation of a machine learning framework for interpretable partially observed system identification, combining an encoder for state reconstruction with a sparse symbolic model.

Please cite "**Discovering sparse interpretable dynamics from partial observations**" (https://doi.org/10.1038/s42005-022-00987-z) and see the paper for more details. This is the official repository for the paper.

## Requirements

JAX >= 0.2.8, Haiku >= 0.0.4, scikit-learn, NumPy, SciPy

## Usage

Data generation scripts are contained in "data/". Encoder models and related tools are contained in "encoder/". Symbolic models and the tools for taking higher order symbolic time derivatives are contained in "symder/".
