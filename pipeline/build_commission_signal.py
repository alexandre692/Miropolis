"""
Signal COMMISSION AU FOND : les membres de la commission qui a travaillé le
texte montent en séance publique bien plus que les autres.

  1. Dossiers.json.zip → commission(s) au fond par dossier (organeRef des
     actes COM-FOND, nominations de rapporteurs incluses).
  2. json 3/acteur → mandats COMPER par député, DATÉS (début/fin) → le
     pare-feu temporel tient : membre AU MOMENT du scrutin.

Sorties :
  data/dossier_commissions.json  {dossierRef: [organeRef]}
  data/comper_membership.json    {acteur: [[organeRef, debut, fin], ...]}

Usage (racine) : python pipeline/build_commission_signal.py
"""

import glob
import json
import os
import zipfile

ZDIR = "data_amdt"
OUT_DC = os.path.join("data", "dossier_commissions.json")
OUT_CM = os.path.join("data", "comper_membership.json")


def clean_date(d):
    if not isinstance(d, str):
        return ""
    return d.replace("-", "")[:8]


def commissions_du_dossier(dp):
    """organeRef des actes 'commission au fond' (COM-FOND) du dossier."""
    found = set()

    def walk(node):
        if isinstance(node, dict):
            code = str(node.get("codeActe", ""))
            if "COM-FOND" in code:
                ref = node.get("organeRef")
                if isinstance(ref, str):
                    found.add(ref)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for x in node:
                walk(x)

    walk(dp.get("actesLegislatifs"))
    return sorted(found)


def main():
    zd = zipfile.ZipFile(os.path.join(ZDIR, "Dossiers.json.zip"))
    dossier_com = {}
    for name in zd.namelist():
        if not name.endswith(".json"):
            continue
        try:
            dp = json.loads(zd.read(name)).get("dossierParlementaire") or {}
        except (json.JSONDecodeError, KeyError):
            continue
        uid = dp.get("uid")
        coms = commissions_du_dossier(dp) if uid else []
        if uid and coms:
            dossier_com[uid] = coms
    with open(OUT_DC, "w", encoding="utf-8") as f:
        json.dump(dossier_com, f, ensure_ascii=False)
    print(f"{len(dossier_com)} dossiers avec commission au fond -> {OUT_DC}")

    membership = {}
    for path in glob.glob(os.path.join("json 3", "acteur", "*.json")):
        try:
            a = json.loads(open(path, encoding="utf-8").read()).get("acteur") or {}
        except json.JSONDecodeError:
            continue
        uid = a.get("uid", {})
        uid = uid.get("#text") if isinstance(uid, dict) else uid
        if not uid:
            continue
        mandats = a.get("mandats", {}).get("mandat", [])
        if isinstance(mandats, dict):
            mandats = [mandats]
        rows = []
        for m in mandats:
            if m.get("typeOrgane") != "COMPER":
                continue
            ref = m.get("organes", {}).get("organeRef")
            if isinstance(ref, list):
                ref = ref[0] if ref else None
            if isinstance(ref, str):
                rows.append([ref, clean_date(m.get("dateDebut")),
                             clean_date(m.get("dateFin")) or "99999999"])
        if rows:
            membership[uid] = rows
    with open(OUT_CM, "w", encoding="utf-8") as f:
        json.dump(membership, f, ensure_ascii=False)
    print(f"{len(membership)} députés avec mandats COMPER -> {OUT_CM}")


if __name__ == "__main__":
    main()
