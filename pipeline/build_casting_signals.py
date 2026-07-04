"""
Signaux de casting par dossier législatif, lus DIRECTEMENT dans les zips
open data (aucune extraction disque) :

  1. AMENDEURS  — qui a déposé/cosigné des amendements sur le dossier
     (data_amdt/Amendements.json.zip, 121 110 amendements ; le dossierRef
     est dans le chemin de chaque entrée). Un député qui a déposé des
     amendements vient les défendre en séance.
  2. RAPPORTEURS — nominations de rapporteurs par dossier, datées
     (data_amdt/Dossiers.json.zip). Le rapporteur parle toujours.
  3. SCRUTIN→DOSSIER — mapping natif depuis objet.dossierLegislatif.dossierRef
     des 7 978 scrutins.

Toutes les entrées portent leur DATE → le pare-feu temporel reste possible
(en prédiction au jour J on ne compte que ce qui est antérieur à J).

Sorties :
  data/casting_signals.json  {dossierRef: {"rapporteurs": [{acteur, date}],
                              "amendeurs": {acteur: {na, nc, d0}}}}
  data/scrutin_dossier.json  {uid: {"date": YYYYMMDD, "dossier": ref}}

Usage (racine) : python pipeline/build_casting_signals.py
"""

import glob
import json
import os
import zipfile
from collections import defaultdict

ZDIR = os.path.join("data_amdt")
OUT_SIG = os.path.join("data", "casting_signals.json")
OUT_MAP = os.path.join("data", "scrutin_dossier.json")


def clean_date(d):
    if not isinstance(d, str):  # xsi:nil → dict
        return ""
    return d.replace("-", "")[:8]


def collect_acteur_refs(obj, out):
    """Récupère récursivement tous les acteurRef d'un sous-arbre."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "acteurRef":
                if isinstance(v, str):
                    out.append(v)
                elif isinstance(v, list):
                    out.extend(x for x in v if isinstance(x, str))
            else:
                collect_acteur_refs(v, out)
    elif isinstance(obj, list):
        for x in obj:
            collect_acteur_refs(x, out)


def rapporteurs_du_dossier(dp):
    """Parcourt les actesLegislatifs (arbre) et récupère les nominations de
    rapporteurs, avec leur date."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            if "NominRapporteurs" in str(node.get("@xsi:type", "")):
                refs = []
                collect_acteur_refs(node, refs)
                date = clean_date(str(node.get("dateActe", "")))
                found.extend({"acteur": r, "date": date} for r in set(refs))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for x in node:
                walk(x)

    walk(dp.get("actesLegislatifs"))
    return found


def main():
    # --- rapporteurs ---
    zd = zipfile.ZipFile(os.path.join(ZDIR, "Dossiers.json.zip"))
    rapporteurs = {}
    for name in zd.namelist():
        if not name.endswith(".json"):
            continue
        try:
            dp = json.loads(zd.read(name)).get("dossierParlementaire") or {}
        except (json.JSONDecodeError, KeyError):
            continue
        uid = dp.get("uid")
        if not uid:
            continue
        r = rapporteurs_du_dossier(dp)
        if r:
            rapporteurs[uid] = r
    print(f"{len(rapporteurs)} dossiers avec rapporteurs")

    # --- amendeurs (121k entrées, dossierRef dans le chemin) ---
    za = zipfile.ZipFile(os.path.join(ZDIR, "Amendements.json.zip"))
    amendeurs = defaultdict(lambda: defaultdict(lambda: {"na": 0, "nc": 0, "d0": "99999999"}))
    n_ok = 0
    for name in za.namelist():
        parts = name.split("/")
        if len(parts) < 3 or not name.endswith(".json"):
            continue
        dossier = parts[1]
        try:
            a = json.loads(za.read(name)).get("amendement") or {}
        except json.JSONDecodeError:
            continue
        sig = a.get("signataires", {})
        depot = clean_date(a.get("cycleDeVie", {}).get("dateDepot", ""))
        auteur = (sig.get("auteur") or {}).get("acteurRef")
        if isinstance(auteur, str) and auteur.startswith("PA"):
            e = amendeurs[dossier][auteur]
            e["na"] += 1
            e["d0"] = min(e["d0"], depot or "99999999")
        cos = (sig.get("cosignataires") or {}).get("acteurRef") or []
        if isinstance(cos, str):
            cos = [cos]
        for c in cos:
            if isinstance(c, str) and c.startswith("PA"):
                e = amendeurs[dossier][c]
                e["nc"] += 1
                e["d0"] = min(e["d0"], depot or "99999999")
        n_ok += 1
        if n_ok % 20000 == 0:
            print(f"  {n_ok} amendements traités")
    print(f"{n_ok} amendements, {len(amendeurs)} dossiers avec amendeurs")

    signals = {}
    for d in set(list(rapporteurs) + list(amendeurs)):
        signals[d] = {"rapporteurs": rapporteurs.get(d, []),
                      "amendeurs": {a: v for a, v in amendeurs.get(d, {}).items()}}
    with open(OUT_SIG, "w", encoding="utf-8") as f:
        json.dump(signals, f, ensure_ascii=False)
    print(f"-> {OUT_SIG} ({os.path.getsize(OUT_SIG) // 1024} Ko)")

    # --- scrutin -> dossier ---
    smap = {}
    for path in glob.glob(os.path.join("json", "VTANR5L17V*.json")):
        try:
            s = json.loads(open(path, encoding="utf-8").read())["scrutin"]
        except (json.JSONDecodeError, KeyError):
            continue
        ref = ((s.get("objet") or {}).get("dossierLegislatif") or {}).get("dossierRef")
        uid = s.get("uid")
        if uid:
            smap[uid] = {"date": clean_date(s.get("dateScrutin", "")),
                         "dossier": ref}
    with open(OUT_MAP, "w", encoding="utf-8") as f:
        json.dump(smap, f, ensure_ascii=False)
    n_lie = sum(1 for v in smap.values() if v["dossier"])
    print(f"-> {OUT_MAP} : {n_lie}/{len(smap)} scrutins liés à un dossier")


if __name__ == "__main__":
    main()
