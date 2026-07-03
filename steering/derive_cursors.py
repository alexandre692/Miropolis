"""
Niveau B — dérive AUTOMATIQUEMENT les curseurs SAE de chaque député depuis
ses vrais discours (data/discours_par_depute.jsonl.gz, produit par
pipeline/extract_discours.py).

Méthode : pour chaque député, forward du modèle de base sur un échantillon de
ses interventions, activations SAE moyennes au niveau de la couche L ;
les curseurs = top-k features où le député diverge le plus de la MOYENNE DE
SON GROUPE (z-score pondéré). Personne n'assigne la personnalité à la main —
"chaque député est la projection de ses propres mots dans l'espace de
concepts du modèle".

Sortie : data/curseurs_deputes.json
    {acteur: {nom, groupe, features: [{id, score, coef}]}}
Les labels humains des features s'obtiennent sur Neuronpedia
(gemma-2-9b-it, résiduel couche L) — à faire pour les députés de la démo.

Usage (pod GPU, depuis steering/):
    python derive_cursors.py --max-docs 30 --top-k 5
    # ~30 interventions × 573 députés, batchées : dimensionner selon le temps GPU
"""

import argparse
import gzip
import json
import os
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from jumprelu_sae import JumpReLUSAE
from steering import _decoder_layers

CORPUS = os.path.join("..", "data", "discours_par_depute.jsonl.gz")
OUT = os.path.join("..", "data", "curseurs_deputes.json")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="google/gemma-2-9b-it")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--width", default="16k")
    p.add_argument("--max-docs", type=int, default=30, help="interventions max par député")
    p.add_argument("--max-tokens", type=int, default=384, help="tokens max par intervention")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--coef-scale", type=float, default=6.0, help="coef du curseur = coef-scale * z-score normalisé")
    p.add_argument("--min-docs", type=int, default=5, help="ignorer les députés avec moins d'interventions")
    return p.parse_args()


def load_corpus(max_docs):
    """{acteur: {nom, groupe, textes[:max_docs]}} — premières interventions
    suffisamment longues (déjà filtrées >= 200 chars à l'extraction)."""
    deps = {}
    with gzip.open(CORPUS, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d = deps.setdefault(r["acteur"], {"nom": r["nom"], "groupe": r["groupe"], "textes": []})
            if len(d["textes"]) < max_docs:
                d["textes"].append(r["texte"])
    return deps


def main():
    args = parse_args()
    deps = load_corpus(args.max_docs)
    print(f"{len(deps)} députés dans le corpus")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    sae = JumpReLUSAE.from_hub(layer=args.layer, width=args.width, device=model.device)

    # Capture du residual stream à la couche L via hook (pas de génération).
    captured = {}

    def capture_hook(module, inputs, output):
        captured["resid"] = output[0] if isinstance(output, tuple) else output

    handle = _decoder_layers(model)[args.layer].register_forward_hook(capture_hook)

    dep_mean = {}  # acteur -> activations SAE moyennes [d_sae]
    skipped = 0
    items = [(a, d) for a, d in deps.items() if len(d["textes"]) >= args.min_docs]
    for n, (acteur, d) in enumerate(items):
        sums, count = torch.zeros(sae.d_sae, device=model.device), 0
        for i in range(0, len(d["textes"]), args.batch_size):
            batch = d["textes"][i : i + args.batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_tokens,
            ).to(model.device)
            with torch.no_grad():
                model(**enc)
                acts = sae.encode(captured["resid"])  # [B, T, d_sae]
                mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
                sums += (acts * mask).sum(dim=(0, 1)).float()
                count += int(mask.sum())
        dep_mean[acteur] = (sums / max(count, 1)).cpu()
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(items)} députés encodés")
    skipped = len(deps) - len(items)
    handle.remove()

    # Moyenne et écart-type PAR GROUPE, puis top-k z-scores par député.
    by_group = defaultdict(list)
    for acteur, mean in dep_mean.items():
        by_group[deps[acteur]["groupe"]].append(mean)
    group_mu, group_sigma = {}, {}
    for g, means in by_group.items():
        stack = torch.stack(means)
        group_mu[g] = stack.mean(dim=0)
        group_sigma[g] = stack.std(dim=0) + 1e-6

    out = {}
    for acteur, mean in dep_mean.items():
        g = deps[acteur]["groupe"]
        if len(by_group[g]) < 3:
            continue  # z-score sans signification sur un groupe minuscule
        z = (mean - group_mu[g]) / group_sigma[g]
        top = torch.topk(z, args.top_k)
        zmax = float(top.values[0]) or 1.0
        out[acteur] = {
            "nom": deps[acteur]["nom"],
            "groupe": g,
            "features": [
                {
                    "id": int(idx),
                    "score": round(float(val), 3),
                    "coef": round(args.coef_scale * float(val) / zmax, 2),
                }
                for val, idx in zip(top.values, top.indices)
            ],
        }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"{len(out)} députés avec curseurs ({skipped} ignorés < {args.min_docs} docs)")
    print(f"-> {OUT}")
    print(
        "Labels humains des features : Neuronpedia (gemma-2-9b-it, res couche "
        f"{args.layer}) — vérifier au moins les députés utilisés en démo."
    )


if __name__ == "__main__":
    main()
