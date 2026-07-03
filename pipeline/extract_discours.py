"""
Extrait les interventions en séance publique par député depuis les comptes
rendus syceron (XML, open data AN, Licence Ouverte).

Entrées :
  - data_cr/xml/xml/compteRendu/*.xml  (580 séances leg. 17)
    → télécharger : https://data.assemblee-nationale.fr/static/openData/repository/17/vp/syceronbrut/syseron.xml.zip
  - "json 3/acteur/*.json"  (577 députés, mandats → groupe)
  - "json 3/organe/*.json"  (libellés des groupes)

Sorties :
  - data/discours_par_depute.jsonl.gz : 1 ligne = 1 intervention
      {acteur, nom, groupe, seance, date, code_grammaire, texte}
  - data/discours_stats.json : volumétrie par groupe / par député (top)

Usage (depuis la racine du repo) :
    python pipeline/extract_discours.py
"""

import glob
import gzip
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter

NS = {"an": "http://schemas.assemblee-nationale.fr/referentiel"}
CR_DIR = os.path.join("data_cr", "xml", "xml", "compteRendu")
ACTEUR_DIR = os.path.join("json 3", "acteur")
ORGANE_DIR = os.path.join("json 3", "organe")
OUT_JSONL = os.path.join("data", "discours_par_depute.jsonl.gz")
OUT_STATS = os.path.join("data", "discours_stats.json")

# En dessous de ce nombre de caractères, une prise de parole est considérée
# comme procédurale ("Très bien !", "Je retire l'amendement.") et exclue du
# corpus de style — elle reste comptée dans les stats.
MIN_CHARS = 200


def load_organes():
    """organeRef -> libelleAbrev pour les groupes politiques (GP)."""
    labels = {}
    for f in glob.glob(os.path.join(ORGANE_DIR, "*.json")):
        try:
            o = json.load(open(f, encoding="utf-8")).get("organe") or {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if o.get("codeType") == "GP":
            uid = o.get("uid")
            uid = uid.get("#text") if isinstance(uid, dict) else uid
            if uid:
                labels[uid] = o.get("libelleAbrev") or o.get("libelle") or uid
    return labels


def load_acteurs(gp_labels):
    """PAxxxx -> {nom, groupe}. Groupe = mandat GP en cours (dateFin None),
    sinon le dernier mandat GP connu."""
    deputes = {}
    for f in glob.glob(os.path.join(ACTEUR_DIR, "*.json")):
        try:
            a = json.load(open(f, encoding="utf-8")).get("acteur") or {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        uid = a.get("uid", {})
        uid = uid.get("#text") if isinstance(uid, dict) else uid
        if not uid:
            continue
        ident = a.get("etatCivil", {}).get("ident", {})
        nom = f"{ident.get('prenom', '')} {ident.get('nom', '')}".strip()

        mandats = a.get("mandats", {}).get("mandat", [])
        if isinstance(mandats, dict):
            mandats = [mandats]
        gp_current, gp_last = None, None
        for m in mandats:
            if m.get("typeOrgane") != "GP":
                continue
            ref = m.get("organes", {}).get("organeRef")
            if isinstance(ref, list):
                ref = ref[0] if ref else None
            if ref not in gp_labels:
                continue  # GP d'une autre législature / non-groupe
            if m.get("dateFin") is None:
                gp_current = ref
            else:
                gp_last = ref
        groupe_ref = gp_current or gp_last
        deputes[uid] = {
            "nom": nom,
            "groupe": gp_labels.get(groupe_ref, "NI") if groupe_ref else "NI",
        }
    return deputes


def clean_text(el):
    """Texte complet d'un élément <texte> (italiques/exposants aplatis)."""
    text = "".join(el.itertext())
    return re.sub(r"\s+", " ", text).strip()


def main():
    gp_labels = load_organes()
    deputes = load_acteurs(gp_labels)
    print(f"{len(gp_labels)} groupes, {len(deputes)} députés chargés")

    files = sorted(glob.glob(os.path.join(CR_DIR, "*.xml")))
    if not files:
        raise SystemExit(f"Aucun compte rendu dans {CR_DIR} — télécharger syseron.xml.zip d'abord")

    n_kept, n_short, n_nondep, n_chair = 0, 0, 0, 0
    per_group, per_dep = Counter(), Counter()
    chars_group, chars_dep = Counter(), Counter()
    per_dep_nom = {}

    os.makedirs("data", exist_ok=True)
    with gzip.open(OUT_JSONL, "wt", encoding="utf-8") as out:
        for path in files:
            try:
                root = ET.parse(path).getroot()
            except ET.ParseError as e:
                print(f"  ! {os.path.basename(path)} illisible : {e}")
                continue
            uid_el = root.find("an:uid", NS)
            seance = uid_el.text if uid_el is not None else os.path.basename(path)
            date_el = root.find(".//an:metadonnees/an:dateSeance", NS)
            date = (date_el.text or "")[:8] if date_el is not None else ""

            for p in root.iter(f"{{{NS['an']}}}paragraphe"):
                acteur = p.get("id_acteur") or ""
                if acteur not in deputes:
                    n_nondep += 1
                    continue
                # Parole de fauteuil (présidence de séance) : roledebat est
                # peu fiable, mais l'orateur y est nommé par sa fonction
                # ("Mme la présidente") au lieu de son nom.
                nom_el = p.find("an:orateurs/an:orateur/an:nom", NS)
                nom_orateur = (nom_el.text or "") if nom_el is not None else ""
                if p.get("roledebat") == "president" or re.match(r"^(M\.|Mme)\s+l[ae]\s+présidente?\s*$", nom_orateur):
                    n_chair += 1
                    continue
                texte_el = p.find("an:texte", NS)
                if texte_el is None:
                    continue
                texte = clean_text(texte_el)
                if not texte:
                    continue
                d = deputes[acteur]
                per_group[d["groupe"]] += 1
                per_dep[acteur] += 1
                per_dep_nom[acteur] = d["nom"]
                if len(texte) < MIN_CHARS:
                    n_short += 1
                    continue
                chars_group[d["groupe"]] += len(texte)
                chars_dep[acteur] += len(texte)
                n_kept += 1
                out.write(
                    json.dumps(
                        {
                            "acteur": acteur,
                            "nom": d["nom"],
                            "groupe": d["groupe"],
                            "seance": seance,
                            "date": date,
                            "code_grammaire": p.get("code_grammaire", ""),
                            "texte": texte,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    stats = {
        "seances": len(files),
        "interventions_gardees": n_kept,
        "interventions_courtes_exclues": n_short,
        "paroles_de_fauteuil_exclues": n_chair,
        "paragraphes_non_deputes": n_nondep,
        "par_groupe": {
            g: {"interventions": per_group[g], "chars_corpus": chars_group[g]}
            for g in sorted(per_group, key=per_group.get, reverse=True)
        },
        "deputes_couverts": len(per_dep),
        "top30_deputes": [
            {"acteur": a, "nom": per_dep_nom[a], "interventions": per_dep[a], "chars_corpus": chars_dep[a]}
            for a in sorted(per_dep, key=per_dep.get, reverse=True)[:30]
        ],
        "min_chars_corpus": MIN_CHARS,
    }
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(
        f"{n_kept} interventions gardées (>= {MIN_CHARS} chars), "
        f"{n_short} courtes exclues, {len(per_dep)} députés couverts"
    )
    print(f"-> {OUT_JSONL}\n-> {OUT_STATS}")


if __name__ == "__main__":
    main()
