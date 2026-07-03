"""
Backtest de la prédiction d'orateurs — répond à « t'es sûr que les
spécialistes du thème parlent ? » par un chiffre, pas une intuition.

Protocole (split temporel, zéro fuite) :
  - propension calculée sur les séances des 80 % premiers jours ;
  - pour chaque séance de test : thème dominant de la séance, et pour chaque
    groupe on prédit ses top-3 orateurs (score thème×3 + volume×0.1, appris
    sur le train uniquement) ;
  - precision@3 = fraction des prédits qui ont RÉELLEMENT parlé dans cette
    séance. Baseline de comparaison : top-3 par volume seul (sans thème).

Usage (racine) : python pipeline/backtest_orateurs.py
"""

import gzip
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "brain"))
from prompt_pack import THEME_KEYWORDS  # noqa: E402

CORPUS = os.path.join("data", "discours_par_depute.jsonl.gz")


def classify(texte):
    t = texte.lower()
    best, best_n = None, 1
    for theme, kws in THEME_KEYWORDS.items():
        n = sum(t.count(k.lower()) for k in kws)
        if n > best_n:
            best, best_n = theme, n
    return best or "autre"


def main():
    rows = []
    with gzip.open(CORPUS, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["date"], r["seance"], r["acteur"], r["groupe"],
                         classify(r["texte"])))
    dates = sorted({r[0] for r in rows})
    cutoff = dates[int(len(dates) * 0.8)]
    print(f"cutoff temporel : {cutoff} ({len(dates)} jours de séance)")

    # propension apprise sur le train uniquement
    prop_theme = defaultdict(Counter)   # acteur -> theme -> n
    prop_total = Counter()
    groupe_of = {}
    for d, s, a, g, th in rows:
        groupe_of[a] = g
        if d < cutoff:
            prop_theme[a][th] += 1
            prop_total[a] += 1

    # séances de test : orateurs réels + thème dominant
    test_speakers = defaultdict(set)
    test_theme = defaultdict(Counter)
    for d, s, a, g, th in rows:
        if d >= cutoff:
            test_speakers[s].add(a)
            test_theme[s][th] += 1

    def top3(groupe, theme):
        cands = [(prop_theme[a].get(theme, 0) * 3 + prop_total[a] * 0.1, a)
                 for a in prop_total if groupe_of.get(a) == groupe]
        return [a for _, a in sorted(cands, reverse=True)[:3]]

    def top3_volume(groupe):
        cands = [(prop_total[a], a) for a in prop_total
                 if groupe_of.get(a) == groupe]
        return [a for _, a in sorted(cands, reverse=True)[:3]]

    groupes = sorted({g for g in groupe_of.values() if g != "NI"})
    hits = tot = hits_vol = 0
    for s, speakers in test_speakers.items():
        theme = test_theme[s].most_common(1)[0][0]
        for g in groupes:
            for a in top3(g, theme):
                hits += int(a in speakers)
                tot += 1
            for a in top3_volume(g):
                hits_vol += int(a in speakers)

    print(f"séances de test : {len(test_speakers)}")
    print(f"precision@3 (thème×volume) : {hits}/{tot} = {hits / tot:.3f}")
    print(f"precision@3 (volume seul)  : {hits_vol}/{tot} = {hits_vol / tot:.3f}")
    print("lecture : sur 3 orateurs prédits par groupe, combien ont vraiment "
          "parlé dans la séance.")


if __name__ == "__main__":
    main()
