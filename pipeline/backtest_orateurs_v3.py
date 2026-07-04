"""
Backtest v3 du casting — scorer combiné, toujours ZÉRO donnée du jour J :

  D = rapporteurs du dossier (nommés avant J)             [le rapporteur parle]
    + amendeurs du dossier (amendements déposés avant J)  [on défend ses amdts]
    + activité lexicale récente sur l'ordre du jour (v2)

Les dossiers du jour J viennent des scrutins de ce jour (= ordre du jour,
public à l'avance). Comparé aux scorers A/B/C sur le même split temporel.

Usage (racine) : python pipeline/backtest_orateurs_v3.py
"""

import gzip
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "brain"))
from prompt_pack import THEME_KEYWORDS  # noqa: E402

CORPUS = os.path.join("data", "discours_par_depute.jsonl.gz")
ENRICHIS = os.path.join("data", "scrutins_enrichis.jsonl")
SIGNALS = os.path.join("data", "casting_signals.json")
SMAP = os.path.join("data", "scrutin_dossier.json")

STOP = set("""le la les de des du un une et ou sur pour dans par avec sans que qui
au aux ce cette ces son sa ses leur leurs est sont être avoir fait faire plus
notre nos votre vos nous vous ils elles tout tous toute toutes mais donc ainsi
article amendement projet proposition loi lecture première nouvelle ensemble
suivant identique numéro monsieur madame après avant relative relatif visant
portant contre entre comme aussi alors même très bien deux trois lors""".split())


def tokens(text):
    return {w for w in re.findall(r"[a-zàâäéèêëîïôöùûüç\-]{5,}", text.lower())
            if w not in STOP}


def dt(d):
    return datetime.strptime(d, "%Y%m%d")


def main():
    rows = []
    with gzip.open(CORPUS, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["date"], r["seance"], r["acteur"], r["groupe"], r["texte"]))
    signals = json.load(open(SIGNALS, encoding="utf-8"))
    smap = json.load(open(SMAP, encoding="utf-8"))

    agenda_tokens = defaultdict(set)
    with open(ENRICHIS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d = (r.get("date") or "").replace("-", "")
            if d:
                agenda_tokens[d] |= tokens(r.get("titre", ""))
    dossiers_du_jour = defaultdict(set)
    for uid, v in smap.items():
        if v.get("dossier") and v.get("date"):
            dossiers_du_jour[v["date"]].add(v["dossier"])

    dates = sorted({r[0] for r in rows})
    cutoff = dates[int(len(dates) * 0.8)]
    print(f"cutoff temporel : {cutoff}")

    groupe_of, prop_total = {}, Counter()
    docs_by_dep = defaultdict(list)
    for d, s, a, g, txt in rows:
        groupe_of[a] = g
        docs_by_dep[a].append((d, tokens(txt)))
        if d < cutoff:
            prop_total[a] += 1
    for a in docs_by_dep:
        docs_by_dep[a].sort()

    test_speakers = defaultdict(set)
    test_date = {}
    for d, s, a, g, txt in rows:
        if d >= cutoff:
            test_speakers[s].add(a)
            test_date[s] = d

    groupes = sorted({g for g in groupe_of.values() if g != "NI"})
    deps_by_group = defaultdict(list)
    for a, g in groupe_of.items():
        deps_by_group[g].append(a)

    def score_C(a, day_tokens, d0):
        lo = (dt(d0) - timedelta(days=30)).strftime("%Y%m%d")
        sc = 0.0
        for d, tk in docs_by_dep[a]:
            if d >= d0:
                break
            if d < lo:
                continue
            ov = len(tk & day_tokens)
            if ov:
                sc += ov * (0.5 ** ((dt(d0) - dt(d)).days / 10))
        return sc + prop_total[a] * 0.002

    def score_D(a, day_tokens, d0, dossiers):
        sc = score_C(a, day_tokens, d0)
        for ref in dossiers:
            sig = signals.get(ref)
            if not sig:
                continue
            for r in sig["rapporteurs"]:
                if r["acteur"] == a and (r["date"] or "0") < d0:
                    sc += 25.0          # le rapporteur parle
            e = sig["amendeurs"].get(a)
            if e and e["d0"] < d0:
                sc += 6.0 * math.log1p(e["na"]) + 1.5 * math.log1p(e["nc"])
        return sc

    hits_C = hits_D = tot = 0
    n_with_dossier = 0
    for s, speakers in test_speakers.items():
        d0 = test_date[s]
        day_tk = agenda_tokens.get(d0, set())
        dossiers = dossiers_du_jour.get(d0, set())
        if dossiers:
            n_with_dossier += 1
        for g in groupes:
            cands = deps_by_group[g]
            tot += 3
            for _, a in sorted(((score_C(a, day_tk, d0), a) for a in cands),
                               reverse=True)[:3]:
                hits_C += int(a in speakers)
            for _, a in sorted(((score_D(a, day_tk, d0, dossiers), a) for a in cands),
                               reverse=True)[:3]:
                hits_D += int(a in speakers)

    print(f"séances de test : {len(test_speakers)} "
          f"(dont {n_with_dossier} avec dossier(s) identifié(s) au jour J)")
    print(f"  precision@3 C_agenda_recent      : {hits_C}/{tot} = {hits_C / tot:.3f}")
    print(f"  precision@3 D_+rapporteurs+amdts : {hits_D}/{tot} = {hits_D / tot:.3f}")
    print("hasard ~= 0.056")


if __name__ == "__main__":
    main()
