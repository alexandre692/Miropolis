"""
Propension à prendre la parole, par député et par thème — mesurée sur les
71 500 interventions réelles. Répond à « qui va parler sur ce texte ? » :
dans l'hémicycle, ce sont les SPÉCIALISTES du sujet désignés par leur groupe
qui montent au créneau, et ce sont massivement les mêmes d'une séance à
l'autre. On prédit donc l'orateur par son historique thème×volume.

Sortie : data/orateurs_propension.json
    {acteur: {"total": n, "par_theme": {theme: n}}}

Usage (racine) : python pipeline/build_orateur_propension.py
"""

import gzip
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "brain"))
from prompt_pack import THEME_KEYWORDS  # noqa: E402

CORPUS = os.path.join("data", "discours_par_depute.jsonl.gz")
OUT = os.path.join("data", "orateurs_propension.json")


def classify(texte):
    t = texte.lower()
    best, best_n = None, 1  # au moins 2 hits pour attribuer un thème
    for theme, kws in THEME_KEYWORDS.items():
        n = sum(t.count(k.lower()) for k in kws)
        if n > best_n:
            best, best_n = theme, n
    return best or "autre"


def main():
    total = Counter()
    par_theme = defaultdict(Counter)
    with gzip.open(CORPUS, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            total[r["acteur"]] += 1
            par_theme[r["acteur"]][classify(r["texte"])] += 1

    out = {a: {"total": n, "par_theme": dict(par_theme[a])}
           for a, n in total.items()}
    with open(OUT, "w", encoding="utf-8") as fo:
        json.dump(out, fo, ensure_ascii=False, indent=1)
    print(f"{len(out)} députés -> {OUT}")


if __name__ == "__main__":
    main()
