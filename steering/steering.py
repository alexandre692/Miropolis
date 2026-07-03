"""
Steering par curseurs SAE — le "stage 2" du cerveau Miropolis.

Principe : un député = le LoRA de vote/ligne politique (stage 1, partagé)
+ un petit vecteur de curseurs SAE qui rend son DISCOURS distinct (stage 2,
individuel). Sans stage 2, tous les agents d'un même groupe convergent vers
les mêmes arguments — ce qui n'est pas la réalité parlementaire.

Mécanisme : hook forward sur la sortie de la couche résiduelle L. Pour chaque
feature f du profil, on ajoute coef * direction_décodeur(f) au residual stream
de tous les tokens (mode "add"), ou on force l'activation du feature à une
valeur cible (mode "clamp", plus coûteux : encode+decode du delta).

Profil député (JSON) :
{
  "nom": "…", "groupe": "…",
  "features": [
    {"id": 12345, "label": "sécurité", "coef": 8.0},
    {"id": 6789,  "label": "ruralité", "coef": 5.0}
  ]
}
Les coefs s'entendent en multiples de la norme du vecteur décodeur (ordre de
grandeur utile : 3–20 ; trop fort casse la fluidité — calibrer à l'oreille).
"""

import contextlib

import torch


def _decoder_layers(model):
    """Retrouve la liste des couches décodeur, y compris à travers un wrapper
    PEFT et/ou la quantization bitsandbytes."""
    m = model
    for attr in ("base_model", "model"):
        while hasattr(m, attr) and not hasattr(m, "layers"):
            m = getattr(m, attr)
    if not hasattr(m, "layers"):
        raise AttributeError("Impossible de localiser model...layers — architecture inattendue")
    return m.layers


class SAESteerer:
    """Gestionnaire de steering. Un seul hook, profil échangeable à chaud —
    on peut donc faire parler 577 députés différents sans recharger quoi
    que ce soit."""

    def __init__(self, model, sae, layer=20, mode="add"):
        self.model = model
        self.sae = sae
        self.layer_idx = layer
        self.mode = mode
        self._steer_vec = None  # [d_model], précalculé par set_profile
        self._clamps = None  # {feature_id: valeur cible}
        self._handle = None

    def set_profile(self, features):
        """features: liste de {"id": int, "coef": float, ...} (coef=0 → ignoré).
        Précalcule le vecteur de steering (mode add)."""
        active = [f for f in features if f.get("coef")]
        if not active:
            self._steer_vec, self._clamps = None, None
            return
        vec = torch.zeros_like(self.sae.b_dec)
        for f in active:
            vec += float(f["coef"]) * self.sae.decoder_direction(int(f["id"]))
        self._steer_vec = vec
        self._clamps = {int(f["id"]): float(f["coef"]) for f in active}

    def clear_profile(self):
        self._steer_vec, self._clamps = None, None

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self._steer_vec is None:
            return output
        if self.mode == "add":
            hidden = hidden + self._steer_vec.to(hidden.dtype)
        else:  # clamp : force l'activation des features cibles
            x = hidden.to(self.sae.W_enc.dtype)
            acts = self.sae.encode(x)
            delta = torch.zeros_like(hidden, dtype=self.sae.W_enc.dtype)
            for fid, target in self._clamps.items():
                cur = acts[..., fid].unsqueeze(-1)  # [..., 1]
                direction = self.sae.W_dec[fid]  # [d_model]
                delta = delta + (target - cur) * direction
            hidden = hidden + delta.to(hidden.dtype)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def attach(self):
        if self._handle is None:
            layer = _decoder_layers(self.model)[self.layer_idx]
            self._handle = layer.register_forward_hook(self._hook)
        return self

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @contextlib.contextmanager
    def steering(self, features):
        """with steerer.steering(profil["features"]): model.generate(...)"""
        self.set_profile(features)
        self.attach()
        try:
            yield self
        finally:
            self.clear_profile()


def build_gemma_prompt(tokenizer, system, user):
    """Gemma-2 n'a pas de rôle system : même convention que le training
    (merge_system_into_user dans training/train_qlora.py)."""
    chat = [{"role": "user", "content": (system + "\n\n" + user) if system else user}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
