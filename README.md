# From Softmax to Score — Figure Notebooks

This repo contains self-contained Jupyter notebooks to reproduce the main figures from:

> **From Softmax to Score: Transformers Can Effectively Implement In-Context Denoising Steps**
> NeurIPS 2025
> Paper: [https://neurips.cc/virtual/2025/loc/san-diego/poster/119941](https://neurips.cc/virtual/2025/loc/san-diego/poster/119941)

All details about the experiments and theory are in the paper; the notebooks here just implement the corresponding figure pipelines.

---

## Requirements


The code is tuned for a **single NVIDIA A5000 (24 GB)**:

* Expect up to **24 GB VRAM** for the heaviest runs.
* **Inference / plotting**: usually **< 30 minutes** per notebook.
* **Training**: several hours per notebook, but not days.

You can reduce batch sizes, context sizes, or witness counts if you have less VRAM.

---

## Data and checkpoints

All notebooks share the same layout:

* `./data/`

  * MNIST, CIFAR-10, CIFAR-100 (downloaded automatically via `torchvision`).
* `./models/`

  * Subfolders per figure (e.g. `models/figure2`, `models/figure3`, …).
  * Each notebook saves checkpoints per seed and **reuses them** if present.
  * You can enable a flag like `FORCE_RETRAIN = True` in each notebook to retrain from scratch.

---

## Notebooks

### `Figure2.ipynb`

* Manifold denoising on MNIST using an RBF Laplacian-style Transformer.
* Reproduces **Figure 2**:

  * Test error vs **context size**.
  * Test error vs **number of layers**.

### `Figure3+4.ipynb`

* Score-based denoising using RBF vs “standard” (VPNorm) attention.
* Datasets: **MNIST**, **CIFAR-10**, **CIFAR-100**.
* Reproduces:

  * **Figure 3**: per-layer test error for different kernels and “theory vs trained” models.
  * **Figure 4**: visual denoising trajectories (train sample, test sample, pure noise).

### `Figure5+6-3.ipynb`

* In-context score denoising with **learnable witness tokens**.
* Reproduces:

  * **Figure 5**: test error vs **context length** (how much in-context information helps).
  * **Figure 6 / appendix**: FID vs context length (if `torchmetrics` is available).

### `Figure7+1-2.ipynb`

* Witness-based denoisers with:

  * **RBF (isotropic)** kernels.
  * **Anisotropic** (diagonal Q/K/V) kernels.
* Reproduces:

  * **Figure 7**: test error vs **number of witnesses**, plus an exact-score baseline.
  * **Figure 1**: visual “patch-wise” anisotropic denoising across layers.

---

## How to run

1. Create a suitable Python environment with the packages above.
2. Start Jupyter (or VS Code / Colab) on a GPU runtime.
3. Open a notebook (e.g. `Figure3+4.ipynb`) and run all cells top-to-bottom.

   * First run: trains models and saves them under `./models/…`.
   * Later runs: reuse checkpoints and only rerun evaluation/plots.

That’s it — the paper is the reference for what each figure means; the notebooks here just reproduce the experiments.
