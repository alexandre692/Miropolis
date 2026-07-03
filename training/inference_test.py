"""
Charge le modèle de base + l'adaptateur LoRA et teste quelques exemples
du jeu de validation pour vérifier que le fine-tuning a bien pris.

Usage:
    python inference_test.py --adapter ./out/gemma-deputes-lora --n 15
"""

import argparse
import json
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="google/gemma-2-9b-it")
    p.add_argument("--adapter", default="./out/gemma-deputes-lora")
    p.add_argument("--val", default="../data/finetune_val.jsonl")
    p.add_argument("--n", type=int, default=15)
    return p.parse_args()


def extract_position(text):
    m = re.search(r"position:\s*(pour|contre|abstention)", text, re.IGNORECASE)
    return m.group(1).lower() if m else None


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    is_gemma = "gemma" in args.base_model.lower()
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager" if is_gemma else "sdpa",
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    with open(args.val, encoding="utf-8") as f:
        examples = [json.loads(l) for l in f][: args.n]

    correct = 0
    for ex in examples:
        messages = ex["messages"]
        system, user, expected = messages[0]["content"], messages[1]["content"], messages[2]["content"]
        if is_gemma:
            # Gemma-2 ne supporte pas le rôle system : fusionné dans le tour user
            # (même convention que merge_system_into_user dans train_qlora.py)
            chat = [{"role": "user", "content": system + "\n\n" + user}]
        else:
            chat = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        exp_pos, gen_pos = extract_position(expected), extract_position(generated)
        ok = exp_pos == gen_pos
        correct += int(ok)
        print("-" * 80)
        print("USER:", user.replace("\n", " | "))
        print("ATTENDU :", expected)
        print("GÉNÉRÉ  :", generated.strip())
        print("MATCH   :", ok)

    print("=" * 80)
    print(f"Accuracy position (sur {len(examples)} exemples): {correct}/{len(examples)}")


if __name__ == "__main__":
    main()
