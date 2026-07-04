"""
Backtest FINAL du casting prédictif — version consolidée et reproductible
des expériences du 04/07 (v5→proportionnel). Chiffres gelés pour le pitch :

  p@3 par groupe        : 0.399   (plafond théorique/oracle : 0.683 → 58 %)
  precision liste       : 0.42-0.44
  recall orateurs >=2   : 0.49    (>=3 prises : 0.51)  ~ 9× le hasard (0.056)

Signaux (tous STRICTEMENT antérieurs au jour J — pare-feu temporel) :
  - rapporteurs du dossier (nomination datée)                    +25
  - amendements déposés sur le dossier (auteur/cosignataire)     6·ln/1.5·ln
  - présences en commission sur le dossier (réunions datées)     4·ln
  - membre de la commission au fond (mandat actif à J)           +3
  - a déjà parlé en séance sur CE dossier (demi-vie 21 j)        3·decay
  - micro-prior de volume                                        0.002/interv.
Jointures natives : scrutin→dossierRef, CR→seanceRef→réunion(ODJ), acteurRef.

Culs-de-sac VÉRIFIÉS : QAG (pas de date d'enregistrement ex-ante dans l'open
data) ; inscrits à la discussion générale (feuilleton interne, non publié).

Usage (racine) : python pipeline/backtest_casting_final.py
"""

import gzip
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime

D = lambda p: os.path.join("data", p)  # noqa: E731


def dtp(x):
    return datetime.strptime(x, "%Y%m%d")


def main():
    rows = []
    with gzip.open(D("discours_par_depute.jsonl.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["date"], r["seance"], r["acteur"], r["groupe"]))
    crref = json.load(open(D("cr_seanceref.json"), encoding="utf-8"))
    odjS = json.load(open(D("odj_par_seance.json"), encoding="utf-8"))
    sd = {cr: set(odjS.get(ref, [])) for cr, ref in crref.items()}
    signals = json.load(open(D("casting_signals.json"), encoding="utf-8"))
    dcom = json.load(open(D("dossier_commissions.json"), encoding="utf-8"))
    cm = json.load(open(D("comper_membership.json"), encoding="utf-8"))
    pres = json.load(open(D("presences_commission.json"), encoding="utf-8"))

    dates = sorted({r[0] for r in rows})
    cutoff = dates[int(len(dates) * 0.8)]
    gof, ptot = {}, Counter()
    spoke_on = defaultdict(lambda: defaultdict(list))
    for d, s, a, g in rows:
        gof[a] = g
        if d < cutoff:
            ptot[a] += 1
        for ref in sd.get(s, ()):
            spoke_on[ref][a].append(d)
    tsc, td = defaultdict(Counter), {}
    for d, s, a, g in rows:
        if d >= cutoff:
            tsc[s][a] += 1
            td[s] = d

    def score(a, d0, dos):
        sc = ptot[a] * 0.002
        for ref in dos:
            for dd in spoke_on.get(ref, {}).get(a, []):
                if dd < d0:
                    sc += 3.0 * (0.5 ** ((dtp(d0) - dtp(dd)).days / 21))
            sig = signals.get(ref)
            if sig:
                for r in sig["rapporteurs"]:
                    if r["acteur"] == a and (r["date"] or "0") < d0:
                        sc += 25.0
                e = sig["amendeurs"].get(a)
                if e and e["d0"] < d0:
                    sc += 6.0 * math.log1p(e["na"]) + 1.5 * math.log1p(e["nc"])
            for c in dcom.get(ref, []):
                if any(org == c and deb < d0 <= fin for org, deb, fin in cm.get(a, [])):
                    sc += 3.0
            n = sum(1 for dd in pres.get(ref, {}).get(a, []) if dd < d0)
            if n:
                sc += 4.0 * math.log1p(n)
        return sc

    legS = [s for s in tsc if sd.get(s)]
    groupes = sorted({g for g in gof.values() if g != "NI"})
    o = h = tot = 0
    ph = pt = rh = rt = 0
    for s in legS:
        d0, dos = td[s], sd[s]
        spk_all = set(tsc[s])
        spk_eng = {a for a, c in tsc[s].items() if c >= 2}
        ranked = sorted(((score(a, d0, dos), a) for a in gof), reverse=True)
        by_g = Counter(gof.get(a) for a in spk_all)
        for g in groupes:
            tot += 3
            o += min(3, by_g.get(g, 0))
            h += sum(a in spk_all for _, a in
                     [x for x in ranked if gof[x[1]] == g][:3])
        if spk_eng:
            N = max(10, min(40, int(len(spk_eng) * 1.2)))
            pred = {a for _, a in ranked[:N]}
            ph += len(pred & spk_eng); pt += N
            rh += len(pred & spk_eng); rt += len(spk_eng)

    print(f"séances législatives de test : {len(legS)} (cutoff {cutoff})")
    print(f"p@3 par groupe : {h / tot:.3f} | oracle {o / tot:.3f} "
          f"| ratio {h / o:.0%}")
    print(f"liste proportionnelle — precision {ph / pt:.3f}, "
          f"recall orateurs engagés {rh / rt:.3f} (hasard ~0.056)")


if __name__ == "__main__":
    main()
