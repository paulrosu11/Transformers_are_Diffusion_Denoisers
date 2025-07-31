# Transformers are Effective Diffusion Denoisers, Both in Context, and Without Context

This repository powers the results and figures for the paper:
**Transformers are Effective Diffusion Denoisers, Both in Context, and Without Context.**
Anonymous Author(s)

---

## Overview

Each notebook in this repo is focused on producing one figure from the paper. Notebooks are self-contained, with some repetitive code to allow for easy standalone execution and hyperparameter sweeps.

* The code and structure are intentionally simple to support reproducibility and experimentation.
* **DenoisingTesting.py** provides core routines and is imported by several notebooks.
* Figures correspond to the notebook filenames; running a notebook should produce the associated plot from the paper.

For experiment details, setup, and motivation for each figure, please refer directly to the paper.

---

## Requirements

* Python with PyTorch (see each notebook’s imports for details).
* A GPU with **at least 20GB VRAM** is recommended.

All dependencies are minimal and listed at the top of each notebook.
No global requirements file is provided—see the first cell of each notebook for needed packages.

---

## How to Use

1. Choose the notebook corresponding to the figure you wish to reproduce.
2. Adjust hyperparameters as wanted in the first cell(s).
3. Run the notebook to generate the figure.

---

## Notes

* The code is intentionally repetitive to keep each notebook runnable in isolation.
* Figures are produced to match those in the paper by default.
* For further information on the algorithms, theoretical context, and experiment design, consult the paper.

