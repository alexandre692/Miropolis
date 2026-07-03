"""
Matrice d'affinités inter-groupes MESURÉE depuis les votes réels :
affinite(g1, g2) = 2·P(même position sur un scrutin où les deux s'expriment) − 1
∈ [−1, +1]. Positif = alliés de vote, négatif = opposants systématiques.

C'est le réseau d'influence de la simulation — estimé du comportement révélé,
pas supposé. Utilisé par brain/agent.fj_update : un discours RN rapproche les
groupes à affinité positive et REPOUSSE ceux à affinité négative (effet
boomerang, signé).

Local, sans GPU : python pipeline/build_group_affinities.py
Sortie : data/group_affinites.json
"""

import json
import os
from collections import defaultdict
from itertools import combinations

ENRICHIS = os.path.join("data", "scrutins_enrichis.jsonl")
OUT = os.path.join("data", "group_affinites.json")


def main():
    par_scrutin = defaultdict(dict)  # (date, titre) -> {groupe: position}
    with open(ENRICHIS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("position"):
                par_scrutin[(r.get("date", ""), r.get("titre", ""))][r["group"]] = r["position"]

    same = defaultdict(int)
    both = defaultdict(int)
    for positions in par_scrutin.values():
        for g1, g2 in combinations(sorted(positions), 2):
            both[(g1, g2)] += 1
            if positions[g1] == positions[g2]:
                same[(g1, g2)] += 1

    groupes = sorted({g for pair in both for g in pair})
    aff = {g1: {} for g1 in groupes}
    for g1 in groupes:
        for g2 in groupes:
            if g1 == g2:
                aff[g1][g2] = 1.0
                continue
            key = tuple(sorted((g1, g2)))
            aff[g1][g2] = round(2 * same[key] / both[key] - 1, 3) if both[key] else 0.0

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(aff, f, ensure_ascii=False, indent=1)
    print(f"{len(par_scrutin)} scrutins, {len(groupes)} groupes -> {OUT}")
    for g1 in groupes:
        allies = sorted((v, g) for g, v in aff[g1].items() if g != g1)
        print(f"  {g1:8s} allié max: {allies[-1][1]} ({allies[-1][0]:+.2f})  "
              f"opposant max: {allies[0][1]} ({allies[0][0]:+.2f})")


if __name__ == "__main__":
    main()
