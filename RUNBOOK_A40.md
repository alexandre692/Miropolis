# Runbook A40 — ordre exact des jobs GPU

> Tout le reste (pipeline data, profils, baseline, séance en mock) tourne en LOCAL.
> Le GPU ne sert qu'à 3 jobs, dans cet ordre. Baseline à battre (split temporel,
> coupure 2026-05-06) : **accuracy position groupe = 0.519** (`python brain/baseline.py`).

## 0. Setup (une fois)

```bash
git clone https://github.com/alexandre692/Miropolis && cd Miropolis
pip install -r training/requirements.txt -r steering/requirements.txt
huggingface-cli login   # accepter la licence google/gemma-2-9b-it sur HF
```

## 1. LoRA (le soir — prioritaire)

```bash
cd training
python train_qlora.py --train ../data/finetune_train.jsonl \
    --val ../data/finetune_val.jsonl --output ./out/gemma-deputes-lora
python inference_test.py --adapter ./out/gemma-deputes-lora --n 30
```

## 2. Curseurs SAE par député (la nuit, pendant que personne ne regarde)

```bash
cd steering
python derive_cursors.py --max-docs 30 --top-k 5     # -> data/curseurs_deputes.json
```

Au réveil : vérifier sur Neuronpedia (gemma-2-9b-it, res L20) les labels des
features des députés qu'on montrera en démo.

## 3. Séance complète (samedi matin)

```bash
# smoke test du cerveau complet AVANT d'y croire :
python brain/seance.py --scrutin VTANR5L17V4000 --adapter training/out/gemma-deputes-lora
# backtest : choisir un scrutin POSTÉRIEUR au 2026-05-06 (hors training),
# haute saillance, et comparer tally prédit vs réel.
```

## Gotchas connus (déjà gérés dans le code, ne pas "corriger")

- Gemma-2 : pas de rôle `system` (mergé dans user), `attn_implementation="eager"`.
- Ne PAS échantillonner les votes : `group_position()` lit les logits (P calibrée).
- Labels groupes canon = ceux du dataset (Dem, EcoS, UDR…) — voir GROUP_ALIASES.
- Le split de val d'Alexandre est par ligne groupe×scrutin → pour les CLAIMS,
  n'utiliser que le split temporel (comme la baseline), jamais la val accuracy brute.
