"""
Backtest v2 de la prédiction d'orateurs — scorer « ordre du jour + activité
récente » : PRÉDIRE qui parlera, sans aucune donnée du jour J.

Ce qu'on s'autorise (légitime en prospectif) :
  - l'ORDRE DU JOUR : les titres des textes votés ce jour-là (publiés à
    l'avance dans la réalité) ;
  - tout ce qui est STRICTEMENT antérieur au jour J.

Scorers comparés (precision@3 par groupe et par séance) :
  A. volume global (baseline)
  B. propension thème×volume (v1 — 0.133)
  C. NOUVEAU : overlap lexical entre l'ordre du jour du jour J et les
     interventions du député sur les 30 jours précédents (décroissance
     temporelle) + petit prior de volume.

Usage (racine) : python pipeline/backtest_orateurs_v2.py
"""

import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "brain"))
from prompt_pack import THEME_KEYWORDS  # noqa: E402

CORPUS = os.path.join("data", "discours_par_depute.jsonl.gz")
ENRICHIS = os.path.join("data", "scrutins_enrichis.jsonl")

STOP = set("""le la les de des du un une et ou sur pour dans par avec sans que qui
au aux ce cette ces son sa ses leur leurs est sont être avoir fait faire plus
notre nos votre vos nous vous ils elles tout tous toute toutes mais donc ainsi
article amendement projet proposition loi lecture première nouvelle ensemble
suivant identique numéro monsieur madame après avant relative relatif visant
portant contre entre comme aussi alors même très bien deux trois lors donc""".split())


def tokens(text):
    return {w for w in re.findall(r"[a-zàâäéèêëîïôöùûüç\-]{5,}", text.lower())
            if w not in STOP}


def classify(texte):
    t = texte.lower()
    best, best_n = None, 1
    for theme, kws in THEME_KEYWORDS.items():
        n = sum(t.count(k.lower()) for k in kws)
        if n > best_n:
            best, best_n = theme, n
    return best or "autre"


def dt(d):
    return datetime.strptime(d, "%Y%m%d")


def main():
    rows = []
    with gzip.open(CORPUS, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["date"], r["seance"], r["acteur"], r["groupe"],
                         r["texte"]))
    # ordre du jour : titres des scrutins par date
    agenda = defaultdict(set)
    with open(ENRICHIS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d = (r.get("date") or "").replace("-", "")
            if d:
                agenda[d] |= tokens(r.get("titre", ""))

    dates = sorted({r[0] for r in rows})
    cutoff = dates[int(len(dates) * 0.8)]
    print(f"cutoff temporel : {cutoff}")

    groupe_of, prop_theme, prop_total = {}, defaultdict(Counter), Counter()
    docs_by_dep = defaultdict(list)   # acteur -> [(date, tokens)]
    for d, s, a, g, txt in rows:
        groupe_of[a] = g
        docs_by_dep[a].append((d, tokens(txt)))
        if d < cutoff:
            prop_theme[a][classify(txt)] += 1
            prop_total[a] += 1
    for a in docs_by_dep:
        docs_by_dep[a].sort()

    test_speakers = defaultdict(set)
    test_theme = defaultdict(Counter)
    test_date = {}
    for d, s, a, g, txt in rows:
        if d >= cutoff:
            test_speakers[s].add(a)
            test_theme[s][classify(txt)] += 1
            test_date[s] = d

    groupes = sorted({g for g in groupe_of.values() if g != "NI"})
    deps_by_group = defaultdict(list)
    for a, g in groupe_of.items():
        deps_by_group[g].append(a)

    def recent_topic_score(a, day_tokens, d0):
        """Overlap lexical avec l'ordre du jour, sur les 30 jours AVANT J,
        décroissance exponentielle (~10 j de demi-vie)."""
        lo = (dt(d0) - timedelta(days=30)).strftime("%Y%m%d")
        sc = 0.0
        for d, tk in docs_by_dep[a]:
            if d >= d0:
                break
            if d < lo:
                continue
            ov = len(tk & day_tokens)
            if ov:
                age = (dt(d0) - dt(d)).days
                sc += ov * (0.5 ** (age / 10))
        return sc + prop_total[a] * 0.002   # micro-prior de volume

    scores = {"A_volume": 0, "B_theme": 0, "C_agenda_recent": 0}
    tot = 0
    for s, speakers in test_speakers.items():
        d0 = test_date[s]
        theme = test_theme[s].most_common(1)[0][0]
        day_tokens = agenda.get(d0, set())
        for g in groupes:
            cands = deps_by_group[g]
            top = lambda key: [a for _, a in sorted(
                ((key(a), a) for a in cands), reverse=True)[:3]]
            tot += 3
            for a in top(lambda a: prop_total[a]):
                scores["A_volume"] += int(a in speakers)
            for a in top(lambda a: prop_theme[a].get(theme, 0) * 3 + prop_total[a] * .1):
                scores["B_theme"] += int(a in speakers)
            for a in top(lambda a: recent_topic_score(a, day_tokens, d0)):
                scores["C_agenda_recent"] += int(a in speakers)

    print(f"séances de test : {len(test_speakers)} — {tot} prédictions par scorer")
    for k, v in scores.items():
        print(f"  precision@3 {k:16s}: {v}/{tot} = {v / tot:.3f}")
    print("hasard ~= 32 orateurs / 573 députés = 0.056")


if __name__ == "__main__":
    main()
