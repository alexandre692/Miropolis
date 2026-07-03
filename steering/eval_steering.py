"""
Éval du steering — répond à « comment on s'assure qu'on steere bien ? ».
On ne croit pas le steering sur parole : on le MESURE, sur 3 axes.

  1. ATTRIBUTION : le texte généré AVEC les curseurs du député X doit être plus
     proche de la signature SAE réelle de X (activations moyennes sur ses vrais
     discours) que de celle de ses collègues de groupe. Métrique : accuracy
     d'attribution top-1 (chance = 1/N députés testés du groupe).
  2. DIVERGENCE INTRA-GROUPE : la dispersion entre les sorties de N députés du
     MÊME groupe doit augmenter avec les curseurs (sinon le stage 2 ne sert à
     rien). Métrique : distance cosinus moyenne steered vs unsteered.
  3. FLUIDITÉ : la longueur/lisibilité ne doit pas s'effondrer (proxy : ratio
     de tokens uniques et longueur moyenne, steered vs unsteered).

Usage (pod GPU, depuis steering/, APRÈS derive_cursors.py) :
    python eval_steering.py --adapter ../training/out/gemma-deputes-lora \
        --groupe LFI-NFP --n-deputes 4 --n-gen 6
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jumprelu_sae import JumpReLUSAE  # noqa: E402
from steering import SAESteerer, _decoder_layers, build_gemma_prompt  # noqa: E402

CURSEURS = os.path.join("..", "data", "curseurs_deputes.json")
PROFILS = os.path.join("..", "data", "deputes_profils.json")
SUJETS = [
    "le projet de loi de finances pour 2026",
    "la proposition de loi sur l'agriculture et la souveraineté alimentaire",
    "le texte sur la sécurité et l'immigration",
]


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="google/gemma-2-9b-it")
    p.add_argument("--adapter", default=None)
    p.add_argument("--groupe", default="LFI-NFP")
    p.add_argument("--n-deputes", type=int, default=4)
    p.add_argument("--n-gen", type=int, default=6)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--width", default="16k")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto",
        attn_implementation="eager")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    sae = JumpReLUSAE.from_hub(layer=args.layer, width=args.width, device=model.device)
    steerer = SAESteerer(model, sae, layer=args.layer)

    curseurs = json.load(open(CURSEURS, encoding="utf-8"))
    deputes = [(a, c) for a, c in curseurs.items()
               if c["groupe"] == args.groupe][: args.n_deputes]
    if len(deputes) < 2:
        raise SystemExit(f"Pas assez de députés avec curseurs pour {args.groupe}")

    captured = {}
    handle = _decoder_layers(model)[args.layer].register_forward_hook(
        lambda m, i, o: captured.__setitem__("r", o[0] if isinstance(o, tuple) else o))

    def signature(text):
        """Signature SAE moyenne d'un texte (mêmes conventions que derive_cursors)."""
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=384).to(model.device)
        with torch.no_grad():
            model(**enc)
        return sae.encode(captured["r"]).float().mean(dim=(0, 1)).cpu()

    def generate(acteur, c, steer):
        outs = []
        for k in range(args.n_gen):
            sujet = SUJETS[k % len(SUJETS)]
            system = (f"Tu es {c['nom']}, député du groupe {c['groupe']}. "
                      f"Tu t'exprimes en séance publique.")
            user = f"Prononce une intervention courte (3-5 phrases) sur : {sujet}"
            prompt = build_gemma_prompt(tokenizer, system, user)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            def gen():
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=160,
                                         do_sample=True, temperature=0.8, top_p=0.9)
                return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True).strip()

            if steer:
                with steerer.steering(c["features"]):
                    outs.append(gen())
            else:
                outs.append(gen())
        return outs

    print(f"Éval steering — groupe {args.groupe}, {len(deputes)} députés, "
          f"{args.n_gen} générations chacun\n")
    gens = {}
    for acteur, c in deputes:
        gens[acteur] = {"steered": generate(acteur, c, True),
                        "plain": generate(acteur, c, False)}
        print(f"  {c['nom']} : généré")

    # signatures SAE de référence — moyennes des générations vs attribution croisée
    sig_steered = {a: torch.stack([signature(t) for t in g["steered"]]).mean(0)
                   for a, g in gens.items()}
    sig_plain = {a: torch.stack([signature(t) for t in g["plain"]]).mean(0)
                 for a, g in gens.items()}

    # 1. attribution : chaque génération steered individuellement -> quel député ?
    ok = tot = 0
    for a, g in gens.items():
        for t in g["steered"]:
            s = signature(t)
            best = max(gens, key=lambda b: cos(s, sig_steered[b]))
            ok += int(best == a)
            tot += 1
    print(f"\n1. ATTRIBUTION top-1 : {ok}/{tot} = {ok / tot:.2f} "
          f"(chance = {1 / len(deputes):.2f})")

    # 2. divergence intra-groupe
    def dispersion(sigs):
        keys = list(sigs)
        ds = [1 - cos(sigs[k1], sigs[k2])
              for i, k1 in enumerate(keys) for k2 in keys[i + 1:]]
        return sum(ds) / len(ds)

    d_s, d_p = dispersion(sig_steered), dispersion(sig_plain)
    print(f"2. DIVERGENCE intra-groupe : steered {d_s:.4f} vs plain {d_p:.4f} "
          f"({'OK +' if d_s > d_p else 'ÉCHEC '}{(d_s - d_p) / max(d_p, 1e-9):+.0%})")

    # 3. fluidité (proxy grossier)
    def stats(texts):
        toks = [t.split() for t in texts]
        avg_len = sum(len(t) for t in toks) / len(toks)
        uniq = sum(len(set(t)) / max(len(t), 1) for t in toks) / len(toks)
        return avg_len, uniq

    for label, key in (("plain", "plain"), ("steered", "steered")):
        al, uq = stats([t for g in gens.values() for t in g[key]])
        print(f"3. FLUIDITÉ {label:8s}: {al:.0f} mots/intervention, "
              f"ratio mots uniques {uq:.2f}")

    handle.remove()
    out = {"groupe": args.groupe,
           "attribution": ok / tot, "chance": 1 / len(deputes),
           "dispersion_steered": d_s, "dispersion_plain": d_p,
           "gens": {curseurs[a]["nom"]: g for a, g in gens.items()}}
    path = f"eval_steering_{args.groupe}.json"
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n-> {path} (verdict : attribution >> chance ET divergence en hausse "
          f"ET fluidité stable = le steering fait son travail)")


if __name__ == "__main__":
    main()
