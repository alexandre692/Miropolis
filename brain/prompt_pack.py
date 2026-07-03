"""
Packs de prompts par député — le mécanisme d'individualité du pivot
prompt-engineering : au lieu de curseurs SAE, chaque agent reçoit SES VRAIS
extraits d'interventions (few-shot verbatim), choisis selon le thème du texte
débattu. Le modèle imite une voix réelle bien mieux qu'il ne suit des
adjectifs — et personne ne peut nous accuser d'avoir inventé la personnalité :
ce sont ses mots au Journal officiel.

Source : data/discours_par_depute.jsonl.gz (71 500 interventions réelles).

Usage librairie :
    packs = PromptPacks()                      # charge le corpus une fois
    bloc = packs.verbatims("PA793218", "immigration_sécurité", n=3)
"""

import gzip
import json
import os
import re
from collections import defaultdict

CORPUS = os.path.join("data", "discours_par_depute.jsonl.gz")

# Mots-clés par thème (taxonomie alignée sur scrutins_enrichis d'Alexandre).
THEME_KEYWORDS = {
    "fiscalité_finances": ["impôt", "budget", "fiscal", "taxe", "dépense", "dette",
                           "déficit", "finances", "crédit", "euros", "milliards"],
    "immigration_sécurité": ["immigration", "sécurité", "police", "frontière",
                             "étranger", "délinquance", "expulsion", "OQTF", "asile"],
    "écologie_énergie": ["climat", "écolog", "énergie", "carbone", "nucléaire",
                         "renouvelable", "biodiversité", "pollution", "transition"],
    "santé": ["santé", "hôpital", "soignant", "médecin", "sécurité sociale",
              "patient", "médicament", "Ehpad"],
    "travail_social": ["travail", "salaire", "retraite", "chômage", "emploi",
                       "smic", "syndicat", "précarité", "pouvoir d'achat"],
    "justice_institutions": ["justice", "magistrat", "constitution", "institution",
                             "prison", "peine", "état de droit", "49.3"],
    "international_défense": ["ukraine", "défense", "armée", "international",
                              "europe", "otan", "diplomatie", "guerre"],
    "agriculture_ruralité": ["agricult", "paysan", "rural", "élevage", "pêche",
                             "alimentaire", "terres", "PAC"],
    "logement": ["logement", "loyer", "locataire", "habitat", "HLM", "mal-logement"],
    "éducation_recherche": ["école", "éducation", "enseignant", "université",
                            "recherche", "étudiant", "professeur"],
    "outre_mer": ["outre-mer", "Mayotte", "Guadeloupe", "Martinique", "Réunion",
                  "Guyane", "Kanaky", "Calédonie"],
}


class PromptPacks:
    def __init__(self, corpus=CORPUS, min_len=250, max_len=900):
        self.by_dep = defaultdict(list)
        with gzip.open(corpus, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if min_len <= len(r["texte"]) <= max_len:
                    self.by_dep[r["acteur"]].append(r)
        # chronologique → les extraits récents en premier
        for docs in self.by_dep.values():
            docs.sort(key=lambda r: r.get("date", ""), reverse=True)

    def _score(self, texte, keywords):
        t = texte.lower()
        return sum(t.count(k.lower()) for k in keywords)

    def verbatims(self, acteur, theme=None, n=3):
        """Les n meilleurs extraits du député : d'abord ceux qui collent au
        thème, complétés par les plus récents (sa voix générale)."""
        docs = self.by_dep.get(acteur, [])
        if not docs:
            return []
        picked = []
        if theme and theme in THEME_KEYWORDS:
            kw = THEME_KEYWORDS[theme]
            scored = sorted(docs, key=lambda r: self._score(r["texte"], kw),
                            reverse=True)
            picked = [r for r in scored if self._score(r["texte"], kw) >= 2][: n - 1]
        seen = {id(r) for r in picked}
        for r in docs:  # complète avec du récent
            if len(picked) >= n:
                break
            if id(r) not in seen:
                picked.append(r)
        return [{"date": r["date"], "texte": r["texte"]} for r in picked[:n]]

    def bloc(self, acteur, theme=None, n=3):
        """Bloc prêt à coller dans le prompt."""
        vs = self.verbatims(acteur, theme, n)
        if not vs:
            return ""
        lines = ["Extraits RÉELS de tes interventions passées en séance "
                 "(imite ce ton, ce registre, ces marqueurs) :"]
        for v in vs:
            d = f"{v['date'][:4]}-{v['date'][4:6]}" if len(v.get("date", "")) >= 6 else ""
            lines.append(f"— ({d}) « {v['texte']} »")
        return "\n".join(lines)


def _clean(s):
    return re.sub(r"\s+", " ", s).strip()


if __name__ == "__main__":
    # smoke test local : 3 députés contrastés
    packs = PromptPacks()
    profils = json.load(open(os.path.join("data", "deputes_profils.json"),
                             encoding="utf-8"))
    tests = [(a, p) for a, p in profils.items()
             if p["nom"] in ("Ugo Bernalicis", "Charles de Courson", "Marine Le Pen")]
    for a, p in tests:
        vs = packs.verbatims(a, "fiscalité_finances", 3)
        print(f"\n=== {p['nom']} ({p['groupe']}) — {len(packs.by_dep.get(a, []))} "
              f"extraits éligibles, thème fiscalité ===")
        for v in vs:
            print(" *", _clean(v["texte"])[:130])
