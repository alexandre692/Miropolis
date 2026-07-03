"""
Baseline déterministe : position d'un groupe sur un scrutin = sa position
majoritaire historique sur (groupe, thème). AUCUN modèle.

C'est la barre que le LoRA doit battre — si le fine-tune ne fait pas mieux
que ce prior, il n'apporte rien. Split TEMPOREL (on prédit les scrutins tardifs
depuis les précoces) : pas de fuite.

Tourne en local, sans GPU :
    python brain/baseline.py --test-frac 0.2
"""

import argparse
import json
import os
from collections import Counter, defaultdict

ENRICHIS = os.path.join("data", "scrutins_enrichis.jsonl")


def load_rows():
    with open(ENRICHIS, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    rows.sort(key=lambda r: r.get("date", ""))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-frac", type=float, default=0.2)
    args = p.parse_args()

    rows = load_rows()
    # split temporel par SCRUTIN (titre+date), pas par ligne groupe×scrutin
    keys = []
    seen = set()
    for r in rows:
        k = (r.get("date", ""), r.get("titre", ""))
        if k not in seen:
            seen.add(k)
            keys.append(k)
    n_test = max(1, int(len(keys) * args.test_frac))
    test_keys = set(keys[-n_test:])
    train = [r for r in rows if (r.get("date", ""), r.get("titre", "")) not in test_keys]
    test = [r for r in rows if (r.get("date", ""), r.get("titre", "")) in test_keys]
    print(
        f"{len(keys)} scrutins, train {len(train)} lignes / test {len(test)} lignes " f"(coupure au {keys[-n_test][0]})"
    )

    prior_gt = defaultdict(Counter)  # (groupe, thème) -> positions
    prior_g = defaultdict(Counter)  # groupe -> positions (repli)
    for r in train:
        pos = r.get("position")
        if not pos:
            continue
        prior_gt[(r["group"], r.get("theme", "autre"))][pos] += 1
        prior_g[r["group"]][pos] += 1

    n_ok, n_tot = 0, 0
    per_group = defaultdict(lambda: [0, 0])
    for r in test:
        real = r.get("position")
        if not real:
            continue
        c = prior_gt.get((r["group"], r.get("theme", "autre"))) or prior_g.get(r["group"])
        if not c:
            continue
        pred = c.most_common(1)[0][0]
        n_tot += 1
        per_group[r["group"]][1] += 1
        if pred == real:
            n_ok += 1
            per_group[r["group"]][0] += 1

    print(f"\nBASELINE prior (groupe, thème) — accuracy position groupe : " f"{n_ok}/{n_tot} = {n_ok / n_tot:.3f}")
    for g, (ok, tot) in sorted(per_group.items(), key=lambda x: -x[1][1]):
        print(f"  {g:10s} {ok / tot:.3f}  ({tot} scrutins)")
    print("\nC'est la barre à battre pour le LoRA (même split temporel).")


if __name__ == "__main__":
    main()
