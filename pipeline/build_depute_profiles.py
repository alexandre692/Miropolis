"""
Construit le profil comportemental de chaque député depuis les 7 966 scrutins
(json/VTANR5L17V*.json) : loyauté au groupe, participation, dissidences par
thème. C'est l'ANCRE déterministe du cerveau — le vote d'un agent n'est jamais
inventé par le LLM, il est tiré de ces statistiques révélées.

Thèmes : réutilise la classification d'Alexandre (data/scrutins_enrichis.jsonl,
mapping titre -> thème).

Sortie : data/deputes_profils.json
  {acteur: {nom, groupe, n_votes, participation, loyaute,
            dissidences: {theme: taux}, abstention_rate}}

Usage (racine du repo) : python pipeline/build_depute_profiles.py
"""

import glob
import json
import os
from collections import Counter, defaultdict

SCRUTINS_DIR = "json"
ENRICHIS = os.path.join("data", "scrutins_enrichis.jsonl")
ACTEUR_DIR = os.path.join("json 3", "acteur")
OUT = os.path.join("data", "deputes_profils.json")

POSITIONS = ("pours", "contres", "abstentions")
POS_LABEL = {"pours": "pour", "contres": "contre", "abstentions": "abstention"}

# Canon = les libellés du dataset de fine-tuning (data/finetune_train.jsonl).
# UDDPLR = abrev organe du groupe UDR ; PO847173 = organe UDR absent du
# référentiel json 3 (Ciotti et premiers membres).
GROUP_ALIASES = {"DEM": "Dem", "ECOS": "EcoS", "UDDPLR": "UDR", "PO847173": "UDR"}


def load_theme_map():
    """titre -> thème (depuis le dataset enrichi d'Alexandre)."""
    themes = {}
    if not os.path.exists(ENRICHIS):
        print("! scrutins_enrichis.jsonl absent — thèmes = 'autre'")
        return themes
    with open(ENRICHIS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            themes.setdefault(r.get("titre", ""), r.get("theme", "autre"))
    return themes


def load_noms():
    """PAxxxx -> nom complet (état civil)."""
    noms = {}
    for f in glob.glob(os.path.join(ACTEUR_DIR, "*.json")):
        try:
            a = json.load(open(f, encoding="utf-8")).get("acteur") or {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        uid = a.get("uid", {})
        uid = uid.get("#text") if isinstance(uid, dict) else uid
        ident = a.get("etatCivil", {}).get("ident", {})
        if uid:
            noms[uid] = f"{ident.get('prenom', '')} {ident.get('nom', '')}".strip()
    return noms


def iter_votants(decompte, key):
    block = (decompte or {}).get(key) or {}
    votants = block.get("votant")
    if votants is None:
        return
    if isinstance(votants, dict):
        votants = [votants]
    for v in votants:
        ref = v.get("acteurRef")
        if ref:
            yield ref


def main():
    theme_map = load_theme_map()
    noms = load_noms()

    n_votes = Counter()  # votes exprimés (pour/contre/abstention)
    n_scrutins_present = Counter()
    n_abst = Counter()
    n_loyal = Counter()  # vote == position majoritaire du groupe
    n_dissent = Counter()
    dissent_theme = defaultdict(Counter)
    votes_theme = defaultdict(Counter)
    groupe_seen = defaultdict(Counter)  # acteur -> {groupeRef: count}

    files = glob.glob(os.path.join(SCRUTINS_DIR, "VTANR5L17V*.json"))
    print(f"{len(files)} scrutins")
    for i, path in enumerate(files):
        try:
            scr = json.load(open(path, encoding="utf-8")).get("scrutin") or {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        titre = scr.get("titre", "")
        theme = theme_map.get(titre, "autre")
        groupes = (scr.get("ventilationVotes", {}).get("organe", {}).get("groupes", {}).get("groupe")) or []
        if isinstance(groupes, dict):
            groupes = [groupes]

        for g in groupes:
            gref = g.get("organeRef", "?")
            decompte = g.get("vote", {}).get("decompteNominatif", {})
            per_pos = {k: list(iter_votants(decompte, k)) for k in POSITIONS}
            # position majoritaire du groupe sur CE scrutin (votes exprimés)
            counts = {k: len(v) for k, v in per_pos.items()}
            majority = max(counts, key=counts.get) if any(counts.values()) else None

            for pos_key, acteurs in per_pos.items():
                for a in acteurs:
                    groupe_seen[a][gref] += 1
                    n_scrutins_present[a] += 1
                    n_votes[a] += 1
                    votes_theme[a][theme] += 1
                    if pos_key == "abstentions":
                        n_abst[a] += 1
                    if majority and pos_key == majority:
                        n_loyal[a] += 1
                    elif majority:
                        n_dissent[a] += 1
                        dissent_theme[a][theme] += 1
            for a in iter_votants(decompte, "nonVotants"):
                groupe_seen[a][gref] += 1
                n_scrutins_present[a] += 1
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(files)}")

    # libellés de groupes : réutilise le mapping GP de extract_discours
    import importlib.util

    spec = importlib.util.spec_from_file_location("extract_discours", os.path.join("pipeline", "extract_discours.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    gp_labels = mod.load_organes()

    total_scrutins = len(files)
    out = {}
    for a in n_scrutins_present:
        votes = n_votes[a]
        if votes == 0:
            continue
        gref = groupe_seen[a].most_common(1)[0][0]
        expressed_with_majority = n_loyal[a] + n_dissent[a]
        label = gp_labels.get(gref, gref)
        label = GROUP_ALIASES.get(label, label)
        out[a] = {
            "nom": noms.get(a, a),
            "groupe": label,
            "n_votes": votes,
            "participation": round(n_scrutins_present[a] / total_scrutins, 4),
            "loyaute": round(n_loyal[a] / expressed_with_majority, 4) if expressed_with_majority else None,
            "abstention_rate": round(n_abst[a] / votes, 4),
            "dissidences": {
                t: round(dissent_theme[a][t] / max(votes_theme[a][t], 1), 4) for t, _ in dissent_theme[a].most_common(5)
            },
        }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    loys = [d["loyaute"] for d in out.values() if d["loyaute"] is not None]
    print(f"{len(out)} députés profilés -> {OUT}")
    print(f"loyauté médiane {sorted(loys)[len(loys) // 2]:.3f}, " f"min {min(loys):.3f}, max {max(loys):.3f}")


if __name__ == "__main__":
    main()
