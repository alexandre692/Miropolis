"""
Chargeur minimal des SAEs Gemma Scope (JumpReLU) directement depuis les
params.npz du hub HF — sans dépendre de sae-lens (trop fragile pour un pod
hackathon).

Repo des SAEs pour le modèle instruction-tuné : google/gemma-scope-9b-it-res
Arborescence : layer_{L}/width_{W}/average_l0_{K}/params.npz
Couches disponibles pour le 9B-it : 9, 20, 31. Défaut raisonnable : 20.

Référence : Gemma Scope (Lieberum et al. 2024). JumpReLU :
    acts = relu(x @ W_enc + b_enc) * (x @ W_enc + b_enc > threshold)
    recon = acts @ W_dec + b_dec
"""

import fnmatch

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download

DEFAULT_REPO = "google/gemma-scope-9b-it-res"


def list_available_saes(repo_id=DEFAULT_REPO, layer=None):
    """Liste les chemins params.npz disponibles dans le repo (option: filtre couche)."""
    files = HfApi().list_repo_files(repo_id)
    pattern = f"layer_{layer}/*/params.npz" if layer is not None else "*/params.npz"
    return sorted(f for f in files if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(f, "*/" + pattern))


class JumpReLUSAE(torch.nn.Module):
    def __init__(self, W_enc, W_dec, b_enc, b_dec, threshold):
        super().__init__()
        self.register_buffer("W_enc", W_enc)  # [d_model, d_sae]
        self.register_buffer("W_dec", W_dec)  # [d_sae, d_model]
        self.register_buffer("b_enc", b_enc)  # [d_sae]
        self.register_buffer("b_dec", b_dec)  # [d_model]
        self.register_buffer("threshold", threshold)  # [d_sae]

    @property
    def d_sae(self):
        return self.W_enc.shape[1]

    @classmethod
    def from_hub(cls, repo_id=DEFAULT_REPO, layer=20, width="16k", l0=None, device="cuda", dtype=torch.float32):
        """Télécharge et charge le SAE. Si l0 est None, prend l'unique variante
        disponible pour (layer, width), ou celle de l0 médian s'il y en a plusieurs."""
        prefix = f"layer_{layer}/width_{width}/"
        candidates = [f for f in list_available_saes(repo_id, layer) if f.startswith(prefix)]
        if not candidates:
            raise ValueError(
                f"Aucun SAE {prefix}* dans {repo_id}. Disponibles : " + ", ".join(list_available_saes(repo_id)[:20])
            )
        if l0 is not None:
            candidates = [f for f in candidates if f"average_l0_{l0}/" in f]
            if not candidates:
                raise ValueError(f"Pas de variante average_l0_{l0} sous {prefix}")
        path = sorted(candidates)[len(candidates) // 2]
        local = hf_hub_download(repo_id, path)
        params = np.load(local)
        t = {k: torch.from_numpy(np.asarray(params[k])).to(dtype) for k in params.files}
        sae = cls(t["W_enc"], t["W_dec"], t["b_enc"], t["b_dec"], t["threshold"])
        print(f"SAE chargé : {repo_id}/{path} (d_sae={sae.d_sae})")
        return sae.to(device)

    def encode(self, x):
        """x: [..., d_model] -> activations [..., d_sae]"""
        pre = x.to(self.W_enc.dtype) @ self.W_enc + self.b_enc
        return torch.relu(pre) * (pre > self.threshold)

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def decoder_direction(self, feature_id):
        """Direction unitaire du feature dans l'espace résiduel."""
        v = self.W_dec[feature_id]
        return v / v.norm()
