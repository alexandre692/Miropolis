# Fine-tuning QLoRA — agents-députés

Sur le pod GPU (RunPod, RTX 4090 24 Go ou plus) :

```bash
cd training
pip install -r requirements.txt
huggingface-cli login          # nécessaire pour mistralai/Mistral-7B-Instruct-v0.3 (modèle gated)

python train_qlora.py \
    --base-model mistralai/Mistral-7B-Instruct-v0.3 \
    --train ../data/finetune_train.jsonl \
    --val ../data/finetune_val.jsonl \
    --output ./out/mistral-deputes-lora
```

Sur un RTX 4090 (24 Go), 7137 exemples, 3 epochs : compter ~1-2h.

Tester le résultat :

```bash
python inference_test.py --adapter ./out/mistral-deputes-lora --n 15
```

## Notes

- Le modèle de base `Mistral-7B-Instruct-v0.3` est gated sur HuggingFace : accepter les
  conditions sur la page du modèle avec le compte utilisé pour `huggingface-cli login`.
- `--base-model mistralai/Mistral-7B-v0.3` (non-Instruct) est une alternative si l'accès
  gated pose problème, mais nécessite d'adapter le format de prompt (pas de chat template).
- Ajuster `--batch-size` / `--grad-accum` si out-of-memory (VRAM limitée).
- L'adaptateur LoRA (quelques dizaines de Mo) est sauvegardé dans `--output`, pas le modèle
  complet — pas besoin de re-uploader 7B de poids.
