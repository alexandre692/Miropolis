# Fiche de synthèse — le pipeline Miropolis

> **En une phrase** : un texte de loi entre → le système prédit qui en débattra,
> fait débattre 577 agents-députés fondés sur les données réelles de l'Assemblée,
> calcule le vote — et se vérifie contre le résultat réel.

```
OPEN DATA AN ──► DONNÉES DÉRIVÉES ──► CERVEAU DÉTERMINISTE ──► AGENTS (LLM) ──► SÉANCE + DÉMO 3D
 6 datasets       pipelines python      casting·ancres·FJ        orchestrateur      prédit vs réel
```

---

## Étape 0 — Ingestion (fait une fois, reproductible)

| Quoi | D'où | Script | Sortie |
|---|---|---|---|
| 7 966 scrutins nominatifs | data.assemblee-nationale.fr | `build_depute_profiles.py` | profils : loyauté (méd. **0.970**), dissidences, abstention |
| 71 500 interventions (580 CR) | idem | `extract_discours.py` | corpus verbatims par député (gz) |
| Affinités entre groupes | scrutins | `build_group_affinities.py` | RN↔UDR **+0.80**, gauche⊣UDR **−0.6** |
| 121 110 amendements + 705 dossiers (rapporteurs, commissions) | idem | `build_casting_signals.py`, `build_commission_signal.py` | signaux de casting datés |
| 7 362 réunions (présences, ODJ) | idem | (inclus ci-dessus) | présences + ODJ par séance |

Tout est **daté** → le pare-feu temporel est possible partout.

## Étape 1 — Entrée

Un scrutin réel (`--scrutin VTANR5L17V7894`, mode backtest) **ou** un texte
inventé (`scrutin_fictif()`, mode prospectif). Le système reçoit : titre,
thème, dossier législatif, date.

## Étape 2 — Amont : l'orchestrateur prépare la séance

- **Casting prédictif** (`predicted_casting()`) : qui parlera — 6 signaux
  strictement antérieurs au jour J (rapporteur +25, amendements déposés,
  présences en commission, membre de la commission au fond, a déjà parlé sur
  ce dossier, activité lexicale récente). Répartition **proportionnelle**
  (max 2 orateurs d'un groupe par round). Validé : **recall 0.49 ≈ 9× le
  hasard** (`backtest_casting_final.py`).
- **Position de chaque groupe** : l'orchestrateur (grand modèle, JSON strict)
  l'estime **ancrée au prior historique** calculé sur les votes — tout écart
  doit être justifié. Repli déterministe si l'API tombe.

## Étape 3 — La séance se joue (boucle `simulate()` dans `brain/seance.py`)

À chaque tour de parole : la présidente donne la parole → l'agent-député
reçoit son prompt **assemblé depuis les données** (profil chiffré + 3
verbatims réels sourcés date+CR, filtrés avant J + ses engagements passés
[ledger] + résumé roulant du débat) → il **réfléchit en privé** puis prononce
son intervention (pools multi-modèles, ~30 modèles ∝ poids des groupes) →
interjections possibles (proba ∝ hostilité mesurée). Après chaque round, les
**577 opinions** bougent par influence Friedkin-Johnsen signée (λ = loyauté
réelle, poids = affinités mesurées).

## Étape 4 — Le vote : les données décident

Vote de chaque député = position de son groupe × sa loyauté × sa dissidence
thématique, déplacé (peu — c'est la réalité) par le débat. **Le LLM ne vote
jamais.** Tally = somme des 577 probabilités. Baseline à battre : **0.519**.

## Étape 5 — Aval : la sortie

`runs/seance_*.json` → transcript complet + tally avant/après débat + « voix
déplacées » + verdict prédit. En backtest : **comparaison au résultat réel**,
révélée à la fin. Visualisation : `python front/build_demo.py runs/…` →
`front/demo.html` (hémicycle 3D autonome, double-clic, zéro réseau).

---

## Les 4 commandes qui font tout

```bash
python brain/seance.py --scrutin VTANR5L17V7894 --mock         # séance sans API (test)
python brain/seance.py --scrutin VTANR5L17V7894 --openrouter   # séance réelle (clé requise)
python front/build_demo.py runs/seance_VTANR5L17V7894.json     # → démo 3D
python pipeline/backtest_casting_final.py                      # reproduire les chiffres
```

## Les chiffres à retenir

| Métrique | Valeur |
|---|---|
| Casting : recall orateurs engagés / p@3 | **0.49** (9× hasard) / **0.399** (plafond oracle 0.683) |
| Loyauté de vote médiane (mesurée) | **0.970** |
| Baseline vote (groupe,thème) à battre | **0.519** |
| Pare-feu temporel | modèle **gelé à J-1**, partout |
| Données | 100 % open data AN, Licence Ouverte, tout sourcé au JO |
