"""
One interface for every balancing technique in the experiment matrix:

    X_bal, y_bal, elapsed_seconds = balance(name, X_train, y_train)

Only X_train/y_train are ever touched -- the test set stays untouched, so
metrics never leak resampled/synthetic points into evaluation.

ML-based / ML-hybrid techniques wrap imbalanced-learn.
DL-based / DL-hybrid techniques are small, dependency-light GAN/VAE
implementations in PyTorch, purpose-built for tabular minority oversampling
(not meant to rival CTGAN/SDV in fidelity -- meant to be a fair, fast,
reproducible comparison point).
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTETomek, SMOTEENN

from config import RANDOM_STATE

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except (ImportError, OSError):
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# ML-based / ML-hybrid (imbalanced-learn)
# ---------------------------------------------------------------------------
def _smote(X, y):
    return SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)


def _adasyn(X, y):
    return ADASYN(random_state=RANDOM_STATE).fit_resample(X, y)


def _ros(X, y):
    return RandomOverSampler(random_state=RANDOM_STATE).fit_resample(X, y)


def _smote_tomek(X, y):
    return SMOTETomek(random_state=RANDOM_STATE).fit_resample(X, y)


def _smote_enn(X, y):
    return SMOTEENN(random_state=RANDOM_STATE).fit_resample(X, y)


def _raw(X, y):
    return X, y


# ---------------------------------------------------------------------------
# DL-based / DL-hybrid: small VAE and GAN for minority-class generation
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE:

    class _Generator(nn.Module):
        def __init__(self, noise_dim, out_dim, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(noise_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, out_dim),
            )

        def forward(self, z):
            return self.net(z)

    class _Discriminator(nn.Module):
        def __init__(self, in_dim, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.LeakyReLU(0.2),
                nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
                nn.Linear(hidden, 1), nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    class _VAE(nn.Module):
        def __init__(self, in_dim, latent_dim=16, hidden=64):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
            self.mu = nn.Linear(hidden, latent_dim)
            self.logvar = nn.Linear(hidden, latent_dim)
            self.dec = nn.Sequential(
                nn.Linear(latent_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, in_dim),
            )

        def encode(self, x):
            h = self.enc(x)
            return self.mu(h), self.logvar(h)

        def reparam(self, mu, logvar):
            logvar = torch.clamp(logvar, -10.0, 10.0)
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparam(mu, logvar)
            return self.dec(z), mu, logvar


def _train_gan(minority_np: np.ndarray, n_synth: int, epochs=200, noise_dim=16, batch_size=64):
    torch.manual_seed(RANDOM_STATE)
    dim = minority_np.shape[1]
    G, D = _Generator(noise_dim, dim).to(DEVICE), _Discriminator(dim).to(DEVICE)
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4)
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4)
    bce = nn.BCELoss()
    data = torch.tensor(minority_np, dtype=torch.float32, device=DEVICE)
    n = data.shape[0]

    for _ in range(epochs):
        idx = torch.randint(0, n, (min(batch_size, n),), device=DEVICE)
        real = data[idx]
        b = real.shape[0]

        z = torch.randn(b, noise_dim, device=DEVICE)
        fake = G(z)
        opt_d.zero_grad()
        loss_d = (bce(D(real), torch.ones(b, 1, device=DEVICE))
                  + bce(D(fake.detach()), torch.zeros(b, 1, device=DEVICE)))
        loss_d.backward()
        opt_d.step()

        z = torch.randn(b, noise_dim, device=DEVICE)
        fake = G(z)
        opt_g.zero_grad()
        loss_g = bce(D(fake), torch.ones(b, 1, device=DEVICE))
        loss_g.backward()
        opt_g.step()

    with torch.no_grad():
        z = torch.randn(n_synth, noise_dim, device=DEVICE)
        synth = G(z).cpu().numpy()
    return synth


def _train_vae(minority_np: np.ndarray, n_synth: int, epochs=200, latent_dim=16, batch_size=64):
    torch.manual_seed(RANDOM_STATE)
    dim = minority_np.shape[1]
    vae = _VAE(dim, latent_dim=latent_dim).to(DEVICE)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    data = torch.tensor(minority_np, dtype=torch.float32, device=DEVICE)
    n = data.shape[0]

    for _ in range(epochs):
        idx = torch.randint(0, n, (min(batch_size, n),), device=DEVICE)
        real = data[idx]
        recon, mu, logvar = vae(real)
        recon_loss = nn.functional.mse_loss(recon, real)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + 0.01 * kld
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        z = torch.randn(n_synth, latent_dim, device=DEVICE)
        synth = vae.dec(z).cpu().numpy()
    return synth, vae


def _minority_majority_split(X: pd.DataFrame, y: pd.Series):
    counts = y.value_counts()
    minority_label = counts.idxmin()
    n_needed = counts.max() - counts.min()
    return minority_label, n_needed


def _gan_oversample(X, y, epochs=150):
    minority_label, n_needed = _minority_majority_split(X, y)
    if n_needed <= 0:
        return X, y
    minority_np = X[y == minority_label].to_numpy()
    synth = _train_gan(minority_np, n_needed, epochs=epochs)
    X_synth = pd.DataFrame(synth, columns=X.columns)
    y_synth = pd.Series([minority_label] * n_needed, name=y.name)
    return pd.concat([X, X_synth], ignore_index=True), pd.concat([y, y_synth], ignore_index=True)


def _vae_oversample(X, y, epochs=150):
    minority_label, n_needed = _minority_majority_split(X, y)
    if n_needed <= 0:
        return X, y
    minority_np = X[y == minority_label].to_numpy()
    synth, _ = _train_vae(minority_np, n_needed, epochs=epochs)
    X_synth = pd.DataFrame(synth, columns=X.columns)
    y_synth = pd.Series([minority_label] * n_needed, name=y.name)
    return pd.concat([X, X_synth], ignore_index=True), pd.concat([y, y_synth], ignore_index=True)


def _gan_tomek(X, y, epochs=150):
    """DL-hybrid: GAN-oversample, then remove Tomek-link pairs (ML cleaning)."""
    from imblearn.under_sampling import TomekLinks
    X_bal, y_bal = _gan_oversample(X, y, epochs=epochs)
    return TomekLinks().fit_resample(X_bal, y_bal)


def _vae_smote(X, y, epochs=150):
    """DL-hybrid: encode minority class into VAE latent space, SMOTE-interpolate
    in latent space, decode back -- i.e. SMOTE done in a learned manifold
    instead of raw feature space."""
    minority_label, n_needed = _minority_majority_split(X, y)
    if n_needed <= 0:
        return X, y
    minority_np = X[y == minority_label].to_numpy()
    _, vae = _train_vae(minority_np, n_needed, epochs=epochs)

    with torch.no_grad():
        data = torch.tensor(minority_np, dtype=torch.float32, device=DEVICE)
        mu, _ = vae.encode(data)
        latents = mu.cpu().numpy()

    # SMOTE-style interpolation between random pairs of latent points
    rng = np.random.RandomState(RANDOM_STATE)
    idx_a = rng.randint(0, len(latents), n_needed)
    idx_b = rng.randint(0, len(latents), n_needed)
    lam = rng.uniform(0, 1, size=(n_needed, 1))
    latent_synth = latents[idx_a] * lam + latents[idx_b] * (1 - lam)

    with torch.no_grad():
        synth = vae.dec(torch.tensor(latent_synth, dtype=torch.float32, device=DEVICE)).cpu().numpy()

    X_synth = pd.DataFrame(synth, columns=X.columns)
    y_synth = pd.Series([minority_label] * n_needed, name=y.name)
    return pd.concat([X, X_synth], ignore_index=True), pd.concat([y, y_synth], ignore_index=True)


_REGISTRY = {
    "raw": _raw,
    "ros": _ros,
    "smote": _smote,
    "adasyn": _adasyn,
    "smote_tomek": _smote_tomek,
    "smote_enn": _smote_enn,
    "gan_oversample": _gan_oversample,
    "vae_oversample": _vae_oversample,
    "gan_tomek": _gan_tomek,
    "vae_smote": _vae_smote,
}

_REQUIRES_TORCH = {"gan_oversample", "vae_oversample", "gan_tomek", "vae_smote"}


def balance(name: str, X: pd.DataFrame, y: pd.Series):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown balancing technique '{name}'. Options: {list(_REGISTRY)}")
    if name in _REQUIRES_TORCH and not TORCH_AVAILABLE:
        raise RuntimeError(f"'{name}' requires PyTorch (`pip install torch`).")

    start = time.time()
    X_bal, y_bal = _REGISTRY[name](X, y)
    elapsed = time.time() - start

    # imbalanced-learn returns numpy for some combos; normalize back to DataFrame/Series
    if not isinstance(X_bal, pd.DataFrame):
        X_bal = pd.DataFrame(X_bal, columns=X.columns)
    if not isinstance(y_bal, pd.Series):
        y_bal = pd.Series(y_bal, name=y.name)

    # Guarantee no NaNs or Infs leak to downstream feature selection or classifiers
    X_bal = X_bal.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X_bal.reset_index(drop=True), y_bal.reset_index(drop=True), elapsed
