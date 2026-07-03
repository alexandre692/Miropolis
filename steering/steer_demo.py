"""
Démo A/B du steering : fait parler un député AVEC et SANS ses curseurs SAE,
côte à côte. C'est le test visuel du "stage 2" (et le moment démo du pitch :
bouger un curseur → le discours change).

Usage (sur le pod GPU) :
    python steer_demo.py \
        --adapter ../training/out/gemma-deputes-lora \
        --profile profiles/exemple_depute.json \
        --sujet "l'article 1er du projet de loi de finances pour 2026"

    # sans --adapter : base model nu (utilisable AVANT la fin du QLoRA)
    # --scale 2.0 : multiplie tous les coefs du profil (exploration rapide)
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from jumprelu_sae import JumpReLUSAE
from steering import SAESteerer, build_gemma_prompt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="google/gemma-2-9b-it")
    p.add_argument("--adapter", default=None, help="chemin de l'adaptateur LoRA (optionnel)")
    p.add_argument("--profile", required=True, help="JSON du profil député")
    p.add_argument("--sujet", required=True, help="sujet de l'intervention")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--width", default="16k")
    p.add_argument("--mode", choices=["add", "clamp"], default="add")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=220)
    return p.parse_args()


def main():
    args = parse_args()
    profile = json.load(open(args.profile, encoding="utf-8"))

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
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    sae = JumpReLUSAE.from_hub(layer=args.layer, width=args.width, device=model.device)
    steerer = SAESteerer(model, sae, layer=args.layer, mode=args.mode)

    system = (
        f"Tu es {profile.get('nom', 'un député')}, député du groupe "
        f"{profile.get('groupe', '?')} à l'Assemblée nationale. Tu t'exprimes en "
        f"séance publique, dans le registre parlementaire français."
    )
    user = f"Prononce une courte intervention (4-6 phrases) sur : {args.sujet}"
    prompt = build_gemma_prompt(tokenizer, system, user)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    def generate():
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    features = [{**f, "coef": f.get("coef", 0.0) * args.scale} for f in profile.get("features", [])]

    print("=" * 80)
    print(f"DÉPUTÉ : {profile.get('nom')} ({profile.get('groupe')}) — sujet : {args.sujet}")
    print(f"Curseurs : {[(f.get('label'), f.get('coef')) for f in features]} (mode {args.mode})")
    print("=" * 80)
    print("\n--- SANS steering ---\n")
    print(generate().strip())
    print("\n--- AVEC steering ---\n")
    with steerer.steering(features):
        print(generate().strip())
    steerer.detach()


if __name__ == "__main__":
    main()
