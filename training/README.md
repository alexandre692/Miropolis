# Fine-tuning QLoRA — agents-députés

Base : **google/gemma-2-9b-it** (décision équipe 03/07). Raison : le stage 2 du projet
(curseurs SAE par député, via Gemma Scope / `gemma-scope-9b-it-res`) exige des SAEs
publics pré-entraînés — Gemma 2 est le seul candidat sérieux qui en a. Mistral n'a pas
de SAEs publics, donc pas de steering possible.

Sur le pod GPU (A40 48 Go — 9B en QLoRA 4-bit passe large) :

```bash
cd training
pip install -r requirements.txt
huggingface-cli login          # gemma-2-9b-it est gated : accepter la licence sur la page HF

python train_qlora.py \
    --train ../data/finetune_train.jsonl \
    --val ../data/finetune_val.jsonl \
    --output ./out/gemma-deputes-lora
```

Tester le résultat :

```bash
python inference_test.py --adapter ./out/gemma-deputes-lora --n 15
```

## Spécificités Gemma-2 (gérées automatiquement par les scripts)

- **Pas de rôle `system`** dans le chat template Gemma-2 : le system prompt du dataset
  est fusionné dans le premier tour `user` au préprocessing (`merge_system_into_user`).
- **`attn_implementation="eager"`** : le logit soft-capping de Gemma-2 est incompatible
  avec flash-attn/sdpa à l'entraînement (dégradation silencieuse sinon).
- Ces deux comportements s'activent si `gemma` apparaît dans `--base-model` ; le script
  reste compatible avec un base model Mistral/Llama en le passant explicitement.

## Notes

- Ajuster `--batch-size` / `--grad-accum` si out-of-memory.
- L'adaptateur LoRA (quelques dizaines de Mo) est sauvegardé dans `--output`, pas le
  modèle complet — pas besoin de re-uploader 9B de poids.
