#!/usr/bin/env python
"""Train a denoising Transformer on CIFAR-10 or MNIST, optionally with trainable
Witness tokens, batched queries, and a full-size training split.

--kernel {unit,rbf,vpnorm} to choose cosine-based (unit-sphere), exact RBF-KDE
   score update, or the new variance-preserving “VPNorm’’ approximation.
--constrained flag.  If set, γ is not trainable and is recomputed
     from the σ schedule each training step via
         γ_k = (σ_k² − σ_{k+1}²) / (2 σ_k²)
     with σ_{L} := 0, so the last layer gets γ = 0.5.
File names encode kernel choice, number of epochs, and whether γ is
constrained (“constr’’ vs “free’’) among all other things.
"""

from __future__ import annotations

import argparse, math, random, time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import torch, torch.nn as nn
import torchvision, torchvision.transforms as T
from tqdm.auto import tqdm


#  Models  


def softplus_inv(y: float) -> float:
    """Stable inverse soft-plus; clamps non-positive inputs to give ~0."""
    if y <= 0.0:
        return -20.0                         # softplus(-20) ≈ 2 × 10⁻⁹
    return math.log(math.expm1(y)) if y < 20.0 else y + math.log1p(-math.exp(-y))


class DenoiseLayer(nn.Module):
    """Single KDE / score-matching denoising layer.

    kernel ∈ {'unit', 'rbf', 'vpnorm'}.
      • unit   – cosine attention on the unit sphere
      • rbf    – exact Gaussian KDE score
      • vpnorm – “variance-preserving’’ cosine update approximating the
                 RBF kernel via the VE⇄VP change of variables
    """

    def __init__(self, *, kernel: str = "unit"):
        super().__init__()
        self.log_inv_sigma2 = nn.Parameter(torch.zeros(()))
        self.log_gamma      = nn.Parameter(torch.zeros(()))
        self.kernel         = kernel

    @staticmethod
    def _vp_coeffs(inv_sigma2: torch.Tensor, gamma: torch.Tensor):
        """Return αₖ, αₖ₊₁ required for VP ‹-› VE conversion (scalar tensors)."""
        σ2        = 1.0 / inv_sigma2
        σ2_next   = σ2 - gamma * σ2
        α_k       = torch.sqrt(1.0 / (1.0 + σ2))
        α_k_next  = torch.sqrt(1.0 / (1.0 + σ2_next))
        return α_k, α_k_next

    # NOTE: never constructs [B,M,D].  `keys` is [M,D]; `q` is [B,D].

    def forward(self, keys: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        inv_σ2 = torch.nn.functional.softplus(self.log_inv_sigma2)
        γ      = torch.nn.functional.softplus(self.log_gamma)

        if self.kernel == "unit":
            att = torch.matmul(q, keys.T)                # [B,M]
            w   = torch.softmax(inv_σ2 * att, dim=1)     # [B,M]
            q   = q + γ * (torch.matmul(w, keys) - q)
            q   = q / q.norm(dim=1, keepdim=True)

        elif self.kernel == "vpnorm":
            α_k, α_k_next = self._vp_coeffs(inv_σ2, γ)   # scalars

            att = torch.matmul(q / α_k, keys.T)          # [B,M]
            w   = torch.softmax(inv_σ2 * att, dim=1)     # [B,M]

            q_vp      = q / α_k
            q_vp_next = q_vp + γ * (torch.matmul(w, keys) - q_vp)
            q         = q_vp_next * α_k_next

        else:  # 'rbf'
            q_norm2 = (q ** 2).sum(1, keepdim=True)      # [B,1]
            k_norm2 = (keys ** 2).sum(1)                 # [M]
            dist2   = q_norm2 + k_norm2 - 2 * torch.matmul(q, keys.T)  # [B,M]
            w       = torch.softmax(-0.5 * inv_σ2 * dist2, dim=1)      # [B,M]
            q       = q + γ * (torch.matmul(w, keys) - q)

        return q                                         # [B,D]


class DenoiseTransformer(nn.Module):
    """Stack of L `DenoiseLayer`s with optional trainable witnesses."""

    def __init__(
        self,
        L: int,
        witnesses: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        *,
        kernel: str = "unit",
    ):
        super().__init__()
        self.layers = nn.ModuleList([DenoiseLayer(kernel=kernel) for _ in range(L)])

        # register witnesses
        if witnesses is None:
            self.witnesses, self.per_layer = None, False
        elif isinstance(witnesses, list):
            self.witnesses  = nn.ParameterList([nn.Parameter(w) for w in witnesses])
            self.per_layer  = True
        else:
            self.witnesses  = nn.Parameter(witnesses.clone())
            self.per_layer  = False

    def _get_witness(self, idx: int) -> Optional[torch.Tensor]:
        if self.witnesses is None:
            return None
        if self.per_layer:
            return self.witnesses[idx]              # [W,D]
        return self.witnesses if idx == 0 else None

    # forward: keys [M,D], q [B,D]   →  denoised q [B,D]
    def forward(self, keys: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            W = self._get_witness(i)
            k_aug = keys if W is None else torch.cat([W, keys], dim=0)  # [M',D]
            q     = layer(k_aug, q)
        return q


#  Helper to enforce γ-σ link  


@torch.no_grad()
def _enforce_gamma_constraint(model: DenoiseTransformer):
    """Overwrite each layer’s γ so that γ_k = (σ_k²−σ_{k+1}²)/(2σ_k²)."""
    inv_sigmas = [
        torch.nn.functional.softplus(lyr.log_inv_sigma2).detach()
        for lyr in model.layers
    ] + [torch.tensor(float("inf"), device=model.layers[0].log_inv_sigma2.device)]

    for k, lyr in enumerate(model.layers):
        σk2    = 1.0 / inv_sigmas[k]
        σk1_sq = 0.0 if torch.isinf(inv_sigmas[k + 1]) else 1.0 / inv_sigmas[k + 1]
        γ      = (σk2 - σk1_sq) / (2 * σk2)
        lyr.log_gamma.data.copy_(
            torch.tensor(softplus_inv(max(float(γ), 1e-12)),
                         device=lyr.log_gamma.device)
        )


#  Data  


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=1, keepdim=True).clamp_min(1e-8)


def load_cifar_unit(root: Path):
    ds   = torchvision.datasets.CIFAR10(root=root, train=True,
                                        download=True, transform=T.ToTensor())
    imgs = torch.stack([img for img, _ in ds])
    μ, σ = imgs.mean((0, 2, 3), keepdim=True), imgs.std((0, 2, 3), keepdim=True)
    X    = ((imgs - μ) / σ).view(len(imgs), -1)
    return _unit(X), μ, σ


def load_mnist_unit(root: Path):
    ds   = torchvision.datasets.MNIST(root=root, train=True,
                                      download=True, transform=T.ToTensor())
    imgs = ds.data.unsqueeze(1).float() / 255.0
    μ, σ = imgs.mean((0, 2, 3), keepdim=True), imgs.std((0, 2, 3), keepdim=True)
    X    = ((imgs - μ) / σ).view(len(imgs), -1)
    return _unit(X), μ, σ


#  Witness helpers  #


def make_witnesses(num: int, dim: int, layers: int,
                   scope: str, device: torch.device):
    if num == 0:
        return None
    if scope == "shared":
        return _unit(torch.randn(num, dim, device=device))
    return [_unit(torch.randn(num, dim, device=device)) for _ in range(layers)]


#  Theory init utils  #


def theory_initialise(model: DenoiseTransformer, *,
                      sigma0: float, sigma_min: float):
    L          = len(model.layers)
    rho        = (sigma_min / sigma0) ** (1 / L)
    γ_star     = (1 - rho ** 2) / 2
    for k, σ_k in enumerate(
        sigma0 * rho ** torch.arange(L, device=model.layers[0].log_gamma.device)
    ):
        lyr = model.layers[k]
        lyr.log_gamma.data.fill_(softplus_inv(γ_star))
        lyr.log_inv_sigma2.data.fill_(softplus_inv(1.0 / (σ_k ** 2)))


#  Batch sampler ──── #


def sample_batch(
    X: torch.Tensor,
    q_idx: torch.Tensor,
    dict_idx: torch.Tensor,
    noise_std: float,
    m_tokens: int,
    device: torch.device,
):
    """Return (keys[M,D], x_noisy[B,D], x_clean[B,D])."""
    x_clean = X[q_idx]                                   # [B,D]
    x_noisy = _unit(x_clean + noise_std *
                    torch.randn_like(x_clean))           # [B,D]
    sub_idx = dict_idx[
        torch.randint(len(dict_idx), (m_tokens,), device=device)
    ]
    keys    = X[sub_idx]                                 # [M,D]
    return keys.to(device), x_noisy.to(device), x_clean.to(device)


# ────── CLI & train  #


def main():
    p = argparse.ArgumentParser("Witness-augmented denoiser trainer")
    p.add_argument("--dataset", choices=["cifar10", "mnist"], default="cifar10")
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--steps_per_epoch", type=int, default=250)
    p.add_argument("--noise_std", type=float, default=0.05)
    p.add_argument("--sigma_min", type=float, default=0.01)
    p.add_argument("--init", choices=["random", "theory"], default="random")
    p.add_argument("--split", type=float, default=0.9)
    p.add_argument("--dict_size", type=int, default=0,
                   help="Limit on dictionary tokens; 0 uses full train set.")
    p.add_argument("--batch_dict_tokens", type=int)
    p.add_argument("--batch_size", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    # witness flags
    p.add_argument("--use_witnesses", action="store_true")
    p.add_argument("--num_witnesses", type=int, default=0)
    p.add_argument("--witness_scope", choices=["shared", "layer"],
                   default="shared")
    # kernel
    p.add_argument("--kernel", choices=["unit", "rbf", "vpnorm"], default="unit")
    # γ–σ constraint
    p.add_argument("--constrained", action="store_true",
                   help="If set, γ is derived from σ and **not** trainable.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str,
                   default="/home/users/par55/XiangTest/trained_models")
    args = p.parse_args()

    if not args.use_witnesses:
        args.num_witnesses = 0

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─────────────── Load & normalise data ──────────────── #
    root = Path("./data")
    X, _, _ = (load_cifar_unit(root) if args.dataset == "cifar10"
               else load_mnist_unit(root))
    X  = X.to(device)                                      # [N,D]
    D  = X.size(1)

    # train / test split
    perm       = torch.randperm(len(X), device=device)
    n_train    = int(args.split * len(X))
    train_idx  = perm[:n_train]
    test_idx   = perm[n_train:]
    dict_cap   = args.dict_size or n_train
    dict_idx   = train_idx[:dict_cap]

    # dictionary tokens per step
    m_tokens = args.batch_dict_tokens or len(dict_idx)

    # witnesses
    raw_witnesses = make_witnesses(args.num_witnesses, D,
                                   args.layers, args.witness_scope, device)

    # model
    model = DenoiseTransformer(args.layers, witnesses=raw_witnesses,
                               kernel=args.kernel).to(device)

    if args.constrained:
        for l in model.layers:
            l.log_gamma.requires_grad = False
    if args.init == "theory":
        theory_initialise(model, sigma0=args.noise_std, sigma_min=args.sigma_min)

    crit      = torch.nn.MSELoss()
    optim     = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs * args.steps_per_epoch,
        eta_min=args.lr * 0.1)

    #  Training loop  #
    pbar = tqdm(range(args.epochs), desc="Training", unit="epoch")
    for _ in pbar:
        loss_acc, t0 = 0.0, time.perf_counter()
        for _ in range(args.steps_per_epoch):
            if args.constrained:
                _enforce_gamma_constraint(model)

            q_idx = (test_idx[torch.randint(len(test_idx),
                                            (args.batch_size,),
                                            device=device)]
                     if len(test_idx) else
                     torch.randint(len(X), (args.batch_size,), device=device))

            keys, q_noisy, tgt = sample_batch(
                X, q_idx, dict_idx, args.noise_std, m_tokens, device)

            out   = model(keys, q_noisy)                 # [B,D]
            loss  = crit(out, tgt)
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
            loss_acc += loss.item()

        avg, lr_now = loss_acc / args.steps_per_epoch, scheduler.get_last_lr()[0]
        pbar.set_postfix(loss=f"{avg:.4e}",
                         lr=f"{lr_now:.2e}",
                         t=f"{time.perf_counter() - t0:.1f}s")

    model.eval()
    if args.constrained:
        _enforce_gamma_constraint(model)

    #  Save  #
    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    wit_tag = (f"w{args.num_witnesses}"
               f"{'L' if args.witness_scope=='layer' else 'S'}"
               if args.num_witnesses else "nowit")
    run_name = (f"{args.dataset}_L{args.layers}_{wit_tag}"
                f"_noise{args.noise_std}_k{args.kernel}"
                f"_{'constr' if args.constrained else 'free'}"
                f"_ep{args.epochs}_init{args.init}"
                f"_split{args.split:.2f}_{stamp}")
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)},
               out_dir / "model.pt")
    print(f"Saved model to {out_dir/'model.pt'}")


if __name__ == "__main__":
    main()
