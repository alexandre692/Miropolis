# Steering SAE — stage 2 du cerveau Miropolis

Un député = **LoRA partagé** (ligne de vote du groupe, `training/`) + **curseurs SAE
individuels** (son discours à lui). Sans les curseurs, tous les agents d'un même groupe
poussent les mêmes arguments — irréaliste. Les votes convergent dans la réalité (82 %
d'unanimité de groupe), pas les discours.

SAEs : **Gemma Scope** (`google/gemma-scope-9b-it-res`), JumpReLU sur le residual
stream, couches 9/20/31 du gemma-2-9b-it. Chargés directement depuis les `params.npz`
du hub (`jumprelu_sae.py`) — pas de dépendance sae-lens.

## Fichiers

| Fichier | Rôle |
|---|---|
| `jumprelu_sae.py` | Chargeur + encode/decode des SAEs Gemma Scope |
| `steering.py` | `SAESteerer` : hook sur la couche L, profil échangeable à chaud (577 députés sans recharger) |
| `steer_demo.py` | Démo A/B : même député avec/sans curseurs, côte à côte |
| `derive_cursors.py` | Niveau B : curseurs AUTO-dérivés des vrais discours (top-k features où le député diverge de la moyenne de son groupe) |
| `profiles/` | Profils députés (JSON) — niveau A : features choisies sur Neuronpedia |

## Quickstart sur le pod (marche AVANT même la fin du QLoRA)

```bash
pip install -r requirements.txt          # + ceux de training/
huggingface-cli login                    # gemma-2-9b-it gated

# 1) Smoke test steering (base model nu, ~10 min)
python steer_demo.py --profile profiles/exemple_depute.json \
    --sujet "l'article 1er du projet de loi de finances pour 2026"

# 2) Avec le LoRA une fois entraîné
python steer_demo.py --adapter ../training/out/gemma-deputes-lora \
    --profile profiles/exemple_depute.json --sujet "..."

# 3) Niveau B : dériver les curseurs de TOUS les députés depuis leurs discours
python derive_cursors.py --max-docs 30 --top-k 5
```

## Calibration (important)

- `coef` utile : **3–20** en mode `add`. Trop fort = charabia. Commencer à 6,
  monter jusqu'à ce que le concept s'impose, redescendre d'un cran.
- Mode `clamp` : force l'activation à une valeur cible (plus précis, plus lent).
- Labels humains des features : [Neuronpedia](https://www.neuronpedia.org)
  → gemma-2-9b-it, residual couche 20. Chercher par mot-clé ("sécurité",
  "agriculture", "impôt"), noter l'id, le mettre dans le profil.
- Les ids de features dépendent de (couche, width) : un profil écrit pour
  `layer_20/width_16k` ne vaut QUE pour ce SAE.

## À vérifier au premier run sur le pod (pas testable sans GPU)

1. Arborescence exacte du repo HF (`python -c "from jumprelu_sae import list_available_saes; print(list_available_saes(layer=20))"`).
2. Le hook survit à la 4-bit quantization (il opère sur le residual, ça doit passer).
3. Qualité du français steeré à coef 6 vs 12.
