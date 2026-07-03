"""
Fine-tuning QLoRA sur le dataset de votes des groupes parlementaires
(data/finetune_train.jsonl).

Base par défaut : google/gemma-2-9b-it — choisi parce que le stage 2 du projet
(curseurs SAE par député) exige des SAEs publics pré-entraînés, et seul Gemma 2
en a (Gemma Scope / gemma-scope-9b-it-res). Mistral n'a pas de SAEs publics.

Usage (sur le pod GPU, depuis training/):
    pip install -r requirements.txt
    huggingface-cli login   # gemma-2-9b-it est gated (accepter la licence sur HF)
    python train_qlora.py \
        --train ../data/finetune_train.jsonl \
        --val ../data/finetune_val.jsonl \
        --output ./out/gemma-deputes-lora
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="google/gemma-2-9b-it")
    p.add_argument("--train", default="../data/finetune_train.jsonl")
    p.add_argument("--val", default="../data/finetune_val.jsonl")
    p.add_argument("--output", default="./out/gemma-deputes-lora")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=512)
    return p.parse_args()


def merge_system_into_user(example):
    """Le chat template Gemma-2 ne supporte pas le rôle system :
    on fusionne le system prompt dans le premier tour user."""
    msgs = example["messages"]
    if msgs and msgs[0]["role"] == "system":
        system, rest = msgs[0]["content"], list(msgs[1:])
        if rest and rest[0]["role"] == "user":
            rest[0] = {"role": "user", "content": system + "\n\n" + rest[0]["content"]}
        else:
            rest.insert(0, {"role": "user", "content": system})
        example["messages"] = rest
    return example


def main():
    args = parse_args()
    is_gemma = "gemma" in args.base_model.lower()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        # Gemma-2 : le logit soft-capping est incompatible avec flash/sdpa à l'entraînement
        attn_implementation="eager" if is_gemma else "sdpa",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    train_ds = load_dataset("json", data_files=args.train, split="train")
    val_ds = load_dataset("json", data_files=args.val, split="train")
    if is_gemma:
        train_ds = train_ds.map(merge_system_into_user)
        val_ds = val_ds.map(merge_system_into_user)

    sft_config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        max_length=args.max_seq_len,
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Adaptateur LoRA sauvegardé dans {args.output}")


if __name__ == "__main__":
    main()
