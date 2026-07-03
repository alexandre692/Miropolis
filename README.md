# Miropolis — simulation projective d'une séance de l'Assemblée nationale

Hackathon AN 2026 « Le parcours de la loi : vers une IA de confiance ».
577 agents-députés fondés sur les données réelles de la législature 17
(7 966 scrutins nominatifs, 71 500 interventions en séance — Licence Ouverte),
qui débattent et dont le vote est prédit **avant** de révéler le résultat réel.

## ⚡ Quickstart (n'importe quel ordi, AUCUN GPU requis)

```bash
git clone https://github.com/alexandre692/Miropolis && cd Miropolis
# Python 3.10+ ; la simulation n'utilise QUE la stdlib (aucun pip install)

# 1) Séance en mode mock (sans clé API — vérifie que tout marche)
python brain/seance.py --scrutin VTANR5L17V4000 --mock

# 2) Séance réelle (interventions générées par les modèles)
export OPENROUTER_API_KEY="sk-or-..."          # Windows : setx OPENROUTER_API_KEY "sk-or-..."
python brain/seance.py --scrutin VTANR5L17V4000 --openrouter
```

Options : `--rounds 2` (rounds de débat), `--speakers-per-group 1`,
mapping groupe→modèle dans `brain/models_map.json` (éditable à chaud).

## Architecture

```
ORCHESTRATEUR (gros modèle, temp 0.2, JSON)      AGENTS DÉPUTÉS (modèles rapides, temp 0.85)
 • position de chaque groupe sur le texte,        • identité chiffrée (loyauté, dissidences)
   ancrée au prior historique                     • 3 verbatims RÉELS sourcés au JO (date+CR)
 • résumé roulant du débat (la mémoire            • réflexion privée puis intervention
   collective, ré-injecté à chaque tour)          • ledger de ses engagements (mémoire indiv.)
                                                  • interjections spontanées (proba ∝ hostilité)

              CERVEAU DÉTERMINISTE (local, jamais un LLM — reproductible)
 • ancres : loyauté/dissidences par député mesurées sur les 7 966 scrutins (médiane 0.970)
 • affinités inter-groupes mesurées : RN↔UDR +0.80, EcoS↔GDR +0.71, LFI⊣droite −0.5/−0.7
 • influence Friedkin-Johnsen signée (λ = loyauté réelle) après chaque round
 • casting des orateurs : spécialistes du thème (mesuré sur leurs vraies interventions)
 • VOTE = somme des probabilités individuelles — le LLM ne vote jamais
```

**Pare-feu temporel** : en backtest/démo, chaque agent ne voit QUE des données
antérieures à la séance rejouée (verbatims filtrés par date). La prédiction est
réelle : « modèle gelé à J-1, voici ce qu'il prédit pour J, voici le réel. »

**Barre à battre (mesurée)** : prédire la position d'un groupe par son prior
(groupe, thème), split temporel → **0.519 accuracy**. `python brain/baseline.py`

## Fichiers

| Chemin | Rôle |
|---|---|
| `brain/seance.py` | Orchestrateur de séance (point d'entrée) |
| `brain/agent.py` | Agent-député : ancre + ledger + FJ signé |
| `brain/openrouter_brain.py` | Moteur API (orchestrateur + députés + interjections) |
| `brain/prompt_pack.py` | Verbatims réels sourcés, sélection par thème + pare-feu date |
| `brain/model_io.py` | MockBrain (sans API) + GemmaBrain (voie GPU, non-démo) |
| `brain/baseline.py` | Baseline déterministe à battre (0.519) |
| `pipeline/` | Extraction discours, profils députés, affinités, propension orateurs, backtest orateurs |
| `data/` | Datasets dérivés (profils, affinités, corpus gz, dataset fine-tune) |
| `training/`, `steering/` | Voie LoRA + SAE (thèse d'origine, hors chemin démo — voir RUNBOOK_A40.md) |
| `json/`, `json 3/` | Open data AN brut (scrutins, acteurs, organes) |

## Regénérer les datasets (optionnel, ~10 min, aucun GPU)

```bash
python pipeline/extract_discours.py        # nécessite data_cr/ (syseron.xml.zip, cf. docstring)
python pipeline/build_depute_profiles.py
python pipeline/build_group_affinities.py
python pipeline/build_orateur_propension.py
```

Tout est déjà commité dans `data/` — utile seulement pour auditer la chaîne.

## Choisir le scrutin de démo

Critères : postérieur au 2026-05-06 (hors priors de la baseline), haute
saillance, reconnaissable par le jury. Le champ `sort` + le tally réel sont
dans `json/<uid>.json` ; la comparaison s'affiche automatiquement.
