"""
Orchestrateur de séance — fait tourner le cerveau complet sur UN scrutin :

  1. Le cerveau (LoRA ou Mock) prédit la position de chaque groupe (logits).
  2. Chaque député initialise son opinion = groupe modulé par SA loyauté et
     SA dissidence thématique historiques (ancre déterministe).
  3. Rounds de débat : des orateurs par groupe interviennent (steering SAE),
     tous les députés font une mise à jour Friedkin-Johnsen (λ = loyauté).
  4. Tally attendu = somme des probabilités individuelles, comparé au réel.

Local sans GPU (loop complet, priors historiques) :
    python brain/seance.py --scrutin VTANR5L17V4000 --mock
Sur le pod (LoRA + steering) :
    python brain/seance.py --scrutin VTANR5L17V4000 \
        --adapter training/out/gemma-deputes-lora
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from agent import POSITIONS, DeputyAgent  # noqa: E402
from model_io import MockBrain  # noqa: E402

PROFILS = os.path.join("data", "deputes_profils.json")
ACTIFS = os.path.join("data", "deputes_actifs.json")
CURSEURS = os.path.join("data", "curseurs_deputes.json")
ENRICHIS = os.path.join("data", "scrutins_enrichis.jsonl")


def load_scrutin(uid):
    """Métadonnées + résultat réel du scrutin depuis json/{uid}.json,
    enrichies du thème/saillance d'Alexandre si disponibles."""
    scr = json.load(open(os.path.join("json", f"{uid}.json"), encoding="utf-8"))["scrutin"]
    titre = scr.get("titre", "")
    date_scrutin = (scr.get("dateScrutin") or "").replace("-", "")[:8]  # YYYYMMDD
    meta = {
        "uid": uid,
        "titre": titre,
        "date": date_scrutin,
        "dossier": ((scr.get("objet") or {}).get("dossierLegislatif") or {}).get("dossierRef"),
        "typeVote": scr.get("typeVote", {}).get("libelleTypeVote", "scrutin public ordinaire"),
        "theme": "autre",
        "salience": "moyenne",
        "sort": (scr.get("sort") or {}).get("code", "?"),
    }
    if os.path.exists(ENRICHIS):
        with open(ENRICHIS, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("titre") == titre:
                    meta["theme"], meta["salience"] = r.get("theme", "autre"), r.get("salience", "moyenne")
                    break
    syn = scr.get("syntheseVote", {})
    dec = syn.get("decompte", {})
    meta["reel"] = {
        "pour": int(dec.get("pour", 0) or 0),
        "contre": int(dec.get("contre", 0) or 0),
        "abstention": int(dec.get("abstentions", 0) or 0),
        "nonVotant": int(dec.get("nonVotants", 0) or 0),
        "votants": int(syn.get("nombreVotants", 0) or 0),
    }
    # qui a réellement participé (pour conditionner le tally sur la
    # participation réelle — prédire l'absentéisme est un autre problème)
    participants = set()
    votes_reels = {}  # acteurRef -> position réelle (vote nominatif public)
    POS_KEY = {"pours": "pour", "contres": "contre", "abstentions": "abstention"}
    groupes = (scr.get("ventilationVotes", {}).get("organe", {}).get("groupes", {}).get("groupe")) or []
    if isinstance(groupes, dict):
        groupes = [groupes]
    for g in groupes:
        d = g.get("vote", {}).get("decompteNominatif", {})
        for k in ("pours", "contres", "abstentions"):
            v = (d.get(k) or {}).get("votant")
            if isinstance(v, dict):
                v = [v]
            for x in v or []:
                if x.get("acteurRef"):
                    participants.add(x["acteurRef"])
                    votes_reels[x["acteurRef"]] = POS_KEY[k]
    meta["participants"] = sorted(participants)
    meta["votes_reels"] = votes_reels
    return meta


def load_actifs():
    """Députés EN EXERCICE (data/deputes_actifs.json — pipeline/build_deputes_actifs.py).
    Sans ce filtre, le mode fictif prenait les 640 acteurs de la XVIIe
    législature (départs + remplaçants) : impossible, 577 sièges."""
    if os.path.exists(ACTIFS):
        return set(json.load(open(ACTIFS, encoding="utf-8")))
    print("!! data/deputes_actifs.json absent — lancer pipeline/build_deputes_actifs.py"
          " (fallback : TOUS les acteurs de la législature, sur-effectif)")
    return None


def build_agents(participants=None):
    profils = json.load(open(PROFILS, encoding="utf-8"))
    curseurs = {}
    if os.path.exists(CURSEURS):
        curseurs = json.load(open(CURSEURS, encoding="utf-8"))
    agents = []
    for acteur, p in profils.items():
        if participants and acteur not in participants:
            continue
        agents.append(DeputyAgent.from_profile(acteur, p, curseurs.get(acteur)))
    return agents


# thèmes reconnus (alignés sur data/scrutins_enrichis.jsonl et les mots-clés
# de prompt_pack.py) — un thème hors liste retombe sur "autre"
THEMES = ("fiscalité_finances", "immigration_sécurité", "écologie_énergie",
          "santé", "travail_social", "justice_institutions",
          "international_défense", "agriculture_ruralité", "logement",
          "éducation_recherche", "outre_mer", "autre")


def scrutin_fictif(titre, theme="autre", salience="haute",
                   type_vote="scrutin public solennel"):
    """Construit un « scrutin » à partir d'un texte de loi INVENTÉ : aucune
    donnée de vote réelle (reel=None, sort=None) — c'est une pure prédiction.
    Tous les députés en exercice participent. La date = aujourd'hui : la
    prédiction de position par groupe et les extraits de discours utilisent
    donc TOUT l'historique disponible (pas de pare-feu temporel, on ne triche
    sur rien puisqu'il n'y a pas de futur à ne pas fuiter)."""
    from datetime import date
    if theme not in THEMES:
        theme = "autre"
    slug = "".join(c if c.isalnum() else "_" for c in titre.lower())[:40].strip("_")
    return {
        "uid": f"FICTIF_{slug or 'texte'}",
        "titre": titre,
        "date": date.today().strftime("%Y%m%d"),
        "typeVote": type_vote,
        "theme": theme,
        "salience": salience,
        "sort": None,        # inconnu : aucune issue réelle
        "reel": None,        # aucun décompte réel à comparer
        "participants": [],  # le main filtre sur load_actifs() (577 en exercice)
    }


def load_affinites():
    path = os.path.join("data", "group_affinites.json")
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def load_propension():
    path = os.path.join("data", "orateurs_propension.json")
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def load_orateurs_reels(date_scrutin):
    """Mode REPLAY (opt-in --casting-reel) : les députés qui ont réellement
    pris la parole le jour du scrutin. Utile pour rejouer une séance à
    l'identique — mais ce n'est PAS une prédiction. Le mode par défaut est
    predicted_casting() ci-dessous."""
    import gzip
    path = os.path.join("data", "discours_par_depute.jsonl.gz")
    if not date_scrutin or not os.path.exists(path):
        return {}
    reels = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("date") == date_scrutin:
                reels[r["acteur"]] = reels.get(r["acteur"], 0) + 1
    return reels


STOP_CASTING = set("""le la les de des du un une et ou sur pour dans par avec
sans que qui au aux ce cette ces son sa ses leur leurs est sont être avoir
fait faire plus notre nos votre vos nous vous ils elles tout tous toute toutes
mais donc ainsi article amendement projet proposition loi lecture première
nouvelle ensemble suivant identique numéro monsieur madame après avant relative
relatif visant portant contre entre comme aussi alors même très bien deux
trois lors république constitutionnelle constitutionnel constitution
gouvernement assemblée nationale président présidente""".split())


def _tokens(text):
    import re
    return {w for w in re.findall(r"[a-zàâäéèêëîïôöùûüç\-]{5,}", text.lower())
            if w not in STOP_CASTING}


def predicted_casting(scrutin):
    """CASTING PRÉDIT (défaut) — validé par backtest temporel (v3) :
    precision@3 = 0.308 vs 0.205 (agenda seul) vs 0.056 (hasard) = 5.5×.
    Signaux, tous STRICTEMENT antérieurs au jour J :
      - rapporteurs du dossier (nomination datée)      [+25]
      - amendements déposés sur le dossier             [6·ln(1+auteur) + 1.5·ln(1+cosign)]
      - activité lexicale récente sur le titre (30 j, demi-vie 10 j)
    Retourne {acteur: score}."""
    import gzip
    import math
    from datetime import datetime, timedelta
    path = os.path.join("data", "discours_par_depute.jsonl.gz")
    d0 = scrutin.get("date")
    if not d0 or not os.path.exists(path):
        return {}
    day_tokens = _tokens(scrutin.get("titre", ""))
    lo = (datetime.strptime(d0, "%Y%m%d") - timedelta(days=30)).strftime("%Y%m%d")
    scores = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d = r.get("date", "")
            if not (lo <= d < d0):
                continue
            ov = len(_tokens(r["texte"]) & day_tokens)
            if ov:
                age = (datetime.strptime(d0, "%Y%m%d")
                       - datetime.strptime(d, "%Y%m%d")).days
                scores[r["acteur"]] = scores.get(r["acteur"], 0.0) \
                    + ov * (0.5 ** (age / 10))
    # rapporteurs + amendeurs du dossier (pare-feu : datés < J)
    sig_path = os.path.join("data", "casting_signals.json")
    ref = scrutin.get("dossier")
    if ref and os.path.exists(sig_path):
        sig = json.load(open(sig_path, encoding="utf-8")).get(ref)
        if sig:
            for r in sig["rapporteurs"]:
                if (r["date"] or "0") < d0:
                    scores[r["acteur"]] = scores.get(r["acteur"], 0.0) + 25.0
            for a, e in sig["amendeurs"].items():
                if e["d0"] < d0:
                    scores[a] = scores.get(a, 0.0) \
                        + 6.0 * math.log1p(e["na"]) + 1.5 * math.log1p(e["nc"])
    # v7 : qui a DÉJÀ parlé en séance sur CE dossier (ODJ exact par séance,
    # jours antérieurs, demi-vie 21 j) — le signal dominant sur les textes
    # débattus en plusieurs séances. p@3 global = 0.399 (58 % du plafond
    # théorique 0.683), recall@40 = 0.482.
    if ref:
        sp_p = os.path.join("data", "spoke_on_dossier.json")
        if os.path.exists(sp_p):
            sp = json.load(open(sp_p, encoding="utf-8")).get(ref, {})
            for a, dates_s in sp.items():
                for dd in dates_s:
                    if dd < d0:
                        age = (datetime.strptime(d0, "%Y%m%d")
                               - datetime.strptime(dd, "%Y%m%d")).days
                        scores[a] = scores.get(a, 0.0) + 3.0 * (0.5 ** (age / 21))
    # v4 : commission au fond (mandat actif à J) + présences en commission
    # sur CE dossier (réunions datées < J) → p@3 = 0.322 au backtest
    if ref:
        dc_p = os.path.join("data", "dossier_commissions.json")
        cm_p = os.path.join("data", "comper_membership.json")
        pr_p = os.path.join("data", "presences_commission.json")
        if os.path.exists(dc_p) and os.path.exists(cm_p):
            coms = json.load(open(dc_p, encoding="utf-8")).get(ref, [])
            if coms:
                cm = json.load(open(cm_p, encoding="utf-8"))
                for a, rows_m in cm.items():
                    if any(org in coms and deb < d0 <= fin
                           for org, deb, fin in rows_m):
                        scores[a] = scores.get(a, 0.0) + 3.0
        if os.path.exists(pr_p):
            pr = json.load(open(pr_p, encoding="utf-8")).get(ref, {})
            for a, dates_p in pr.items():
                n = sum(1 for dd in dates_p if dd < d0)
                if n:
                    scores[a] = scores.get(a, 0.0) + 4.0 * math.log1p(n)
    return scores


def speaker_score(agent, theme, propension):
    """Qui parle ? Le spécialiste du thème d'abord (mesuré sur ses
    interventions réelles), le gros parleur ensuite."""
    p = propension.get(agent.acteur, {})
    return p.get("par_theme", {}).get(theme, 0) * 3 + p.get("total", 0) * 0.1


CONTAGION = 0.3  # fraction des curseurs d'un orateur qui se propage à qui l'écoute


def ambient_cursors(agent, last_speakers, affinites):
    """Contagion de saillance : les concepts activés par les orateurs du round
    précédent deviennent temporairement saillants chez qui parle ensuite,
    proportionnellement à |affinité| (on reprend les thèmes d'un allié POUR,
    ceux d'un opposant CONTRE — dans les deux cas on en parle)."""
    extra = []
    for spk in last_speakers:
        if spk["acteur"] == agent.acteur or not spk.get("cursors"):
            continue
        w = abs(affinites.get(agent.groupe, {}).get(spk["groupe"], 0.0))
        if w == 0:
            continue
        for c in spk["cursors"][:2]:
            extra.append({"id": c["id"], "label": c.get("label"),
                          "coef": round(c.get("coef", 0.0) * CONTAGION * w, 2)})
    return extra


def real_first_speaker(scrutin):
    """Le VRAI premier orateur du débat sur ce dossier — pour que la séance
    simulée OUVRE avec la même personne que dans la réalité (typiquement le
    rapporteur ou le ministre qui présente le texte).

    Une "séance" (compte rendu) couvre plusieurs sujets dans la même
    journée : filtrer par seance seule ramène parfois une intervention sans
    rapport (une question d'actualité avant le débat). On croise donc :
      1. odj_par_seance.json (RUANR...) : quelles séances ont traité CE
         dossier (scrutin["dossier"], déjà résolu par load_scrutin) ;
      2. cr_seanceref.json : crosswalk RUANR... <-> CRSANR... (le format des
         comptes rendus dans data/discours_par_depute.jsonl.gz) ;
      3. dans ces séances, l'intervention la plus ANCIENNE dont le texte a
         un fort recouvrement lexical avec le TITRE du texte (comme le
         casting) — approxime le début du débat, pas une question annexe.
    Retourne l'acteurRef, ou None si les données manquent / rien trouvé."""
    import gzip
    ref = scrutin.get("dossier")
    odj_p = os.path.join("data", "odj_par_seance.json")
    cr_p = os.path.join("data", "cr_seanceref.json")
    disc_p = os.path.join("data", "discours_par_depute.jsonl.gz")
    if not ref or not (os.path.exists(odj_p) and os.path.exists(cr_p) and os.path.exists(disc_p)):
        return None
    odj = json.load(open(odj_p, encoding="utf-8"))
    ru2cr = {v: k for k, v in json.load(open(cr_p, encoding="utf-8")).items()}
    seances_cr = {ru2cr[ru] for ru, dossiers in odj.items() if ref in dossiers and ru in ru2cr}
    kw = _tokens(scrutin.get("titre", ""))
    if not seances_cr or not kw:
        return None
    best = None  # (date, index_fichier) le plus petit -> le plus ancien
    with gzip.open(disc_p, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            if r.get("seance") not in seances_cr:
                continue
            if sum(r["texte"].lower().count(k) for k in kw) < 2:
                continue
            key = (r.get("date", ""), i)
            if best is None or key < best[0]:
                best = (key, r["acteur"])
    return best[1] if best else None


def simulate(brain, scrutin, agents, rounds=2, speakers_per_group=1, verbose=True,
             casting_reel=False, calibre=False):
    groups = sorted({a.groupe for a in agents})
    affinites = load_affinites()
    propension = load_propension()
    if casting_reel:  # replay à l'identique (opt-in), PAS une prédiction
        orateurs_reels = load_orateurs_reels(scrutin.get("date"))
        if verbose:
            print(f"casting REPLAY : {sum(1 for a in agents if a.acteur in orateurs_reels)} "
                  f"orateurs du jour (CR)\n")
    else:             # défaut : casting PRÉDIT (agenda + activité récente)
        orateurs_reels = predicted_casting(scrutin)
        if verbose:
            print(f"casting PRÉDIT (rapporteurs+amendeurs+commissions+historique dossier, p@3=0.399 "
                  f"vs hasard 0.056) : {len(orateurs_reels)} députés scorés\n")
    group_probs = {g: brain.group_position(g, scrutin) for g in groups}
    for a in agents:
        a.init_opinion(group_probs[a.groupe], scrutin.get("theme", "autre"))
    if calibre and scrutin.get("votes_reels"):
        # MODE CALIBRÉ (démo/replay) : l'ancre de chaque député = son vote
        # nominatif RÉEL sur ce scrutin (données publiques AN), pas la
        # prédiction LLM. Le débat, les orateurs, le texte restent simulés ;
        # seul l'ancrage du vote est branché sur la réalité — le tally colle
        # donc au résultat réel à ~1 voix près. À dire tel quel en démo :
        # "opinions ancrées sur le vote nominatif réel".
        vr = scrutin["votes_reels"]
        n_cal = 0
        for a in agents:
            pos = vr.get(a.acteur)
            if not pos:
                continue
            a.opinion0 = {p: (0.99 if p == pos else 0.005) for p in POSITIONS}
            a.opinion = dict(a.opinion0)
            n_cal += 1
        if verbose:
            print(f"CALIBRÉ sur le vote nominatif réel : {n_cal}/{len(agents)} députés ancrés\n")
    # tally AVANT débat = les consignes de vote en entrant dans l'hémicycle ;
    # l'écart avec le tally final mesure ce que la séance elle-même a déplacé
    tally_avant = {p: round(sum(a.opinion[p] for a in agents), 1) for p in POSITIONS}

    transcript, summary = [], ""
    last_speakers = []
    # CASTING PROPORTIONNEL : classement GLOBAL par score prédit (toutes
    # couleurs mêlées) — les groupes mobilisés sur CE texte parlent plus,
    # comme dans la vraie séance. Garde-fou théâtral : max 2 orateurs d'un
    # même groupe par round. (validé : precision 0.44 / recall 0.45-0.51)
    ranked = sorted(agents, key=lambda a: (
        orateurs_reels.get(a.acteur, 0) * 1000
        + speaker_score(a, scrutin.get("theme", "autre"), propension)
    ), reverse=True)
    # le VRAI premier orateur du débat (rapporteur/ministre qui présente le
    # texte, en général) ouvre aussi la séance simulée — tri stable : bascule
    # cette personne en tête sans changer l'ordre relatif des autres.
    premier_reel = real_first_speaker(scrutin)
    if premier_reel and any(a.acteur == premier_reel for a in ranked):
        ranked.sort(key=lambda a: a.acteur != premier_reel)
        if verbose:
            nom = next(a.nom for a in ranked if a.acteur == premier_reel)
            print(f"premier orateur (réel, dossier) : {nom}\n")
    deja_parle = set()
    per_round = max(1, speakers_per_group) * len(groups)
    for r in range(rounds):
        heard, round_speakers = [], []
        par_groupe_ce_round = {}
        round_cast = []
        for a in ranked:
            if len(round_cast) >= per_round:
                break
            if a.acteur in deja_parle or par_groupe_ce_round.get(a.groupe, 0) >= 2:
                continue
            round_cast.append(a)
            par_groupe_ce_round[a.groupe] = par_groupe_ce_round.get(a.groupe, 0) + 1
        for speaker in round_cast:
            deja_parle.add(speaker.acteur)
            g = speaker.groupe
            if True:
                ctx = speaker.context_block(summary, [t["texte"] for t in transcript[-3:]])
                extra = ambient_cursors(speaker, last_speakers, affinites)
                texte = brain.intervention(speaker, scrutin, ctx, extra_cursors=extra)
                speaker.remember(r, texte)
                transcript.append({"round": r, "nom": speaker.nom, "groupe": g, "texte": texte})
                heard.append({"groupe": g, "opinion": dict(speaker.opinion)})
                round_speakers.append({"acteur": speaker.acteur, "groupe": g,
                                       "cursors": speaker.cursors})
                if verbose:
                    print(f"  [{r}] {speaker.nom} ({g}) : {texte[:110]}")
                # spontanéité : le groupe le plus opposé peut interjecter
                # (proba ∝ hostilité mesurée ; déterministe via hash → rejouable)
                # — volontairement rare : une interjection ponctue le débat,
                # elle ne doit pas suivre chaque prise de parole.
                opp = min(groups, key=lambda x: affinites.get(g, {}).get(x, 0.0))
                hostilite = -affinites.get(g, {}).get(opp, 0.0)
                if hostilite > 0.4 and hash((scrutin["uid"], r, speaker.acteur)) % 20 < int(hostilite * 3):
                    heckler = next((a for a in sorted(
                        (x for x in agents if x.groupe == opp),
                        key=lambda x: speaker_score(x, scrutin.get("theme", "autre"),
                                                    propension), reverse=True)), None)
                    if heckler and hasattr(brain, "interjection"):
                        cri = brain.interjection(heckler, texte)
                        transcript.append({"round": r, "nom": heckler.nom,
                                           "groupe": opp, "texte": cri,
                                           "interjection": True})
                        if verbose:
                            print(f"      >> {heckler.nom} ({opp}) : {cri[:90]}")
        for a in agents:
            a.fj_update(heard, affinites)
        last_speakers = round_speakers
        if hasattr(brain, "summarize"):  # résumé roulant par l'orchestrateur
            summary = brain.summarize(transcript[-len(heard):], summary)
        else:
            summary = f"round {r + 1} : {len(heard)} interventions, groupes {', '.join(groups)}"

    tally = {p: round(sum(a.opinion[p] for a in agents), 1) for p in POSITIONS}
    deplace = round(sum(abs(tally[p] - tally_avant[p]) for p in POSITIONS) / 2, 1)
    deputes = [{"acteur": a.acteur, "nom": a.nom, "groupe": a.groupe,
                "loyaute": a.loyaute,
                "opinion0": {p: round(v, 3) for p, v in (a.opinion0 or {}).items()},
                "opinion": {p: round(v, 3) for p, v in (a.opinion or {}).items()}}
               for a in agents]
    # résumé de clôture par la présidente de séance, avant la mise aux voix
    # (le fil défile vite, ça permet au public de raccrocher les wagons)
    recap = brain.recap_final(transcript, scrutin, summary) if hasattr(brain, "recap_final") else None
    return {"group_probs": group_probs, "transcript": transcript,
            "tally_avant_debat": tally_avant, "tally": tally,
            "voix_deplacees_par_le_debat": deplace,
            "deputes": deputes, "affinites": affinites,
            "recap_president": recap}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scrutin", help="uid, ex. VTANR5L17V4000 (ou --fictif --titre)")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--openrouter", action="store_true",
                   help="moteur API multi-modèles (OPENROUTER_API_KEY requise)")
    p.add_argument("--adapter", default=None)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--speakers-per-group", type=int, default=1)
    p.add_argument("--no-steer", action="store_true")
    p.add_argument("--casting-reel", action="store_true",
                   help="replay : orateurs réels du jour (défaut = casting PRÉDIT)")
    p.add_argument("--calibre", action="store_true",
                   help="démo/replay : ancre chaque député sur son vote nominatif RÉEL "
                        "(le tally colle au résultat réel ; débat toujours simulé)")
    p.add_argument("--models-map", default=None,
                   help="fichier models_map alternatif (ex. brain/models_map_premium.json)")
    # --- mode projet de loi FICTIF (prédiction pure, aucune issue réelle) ---
    p.add_argument("--fictif", action="store_true",
                   help="simule un texte de loi INVENTÉ (--titre requis, pas de --scrutin)")
    p.add_argument("--titre", help="titre du texte fictif (avec --fictif)")
    p.add_argument("--theme", default="autre",
                   help=f"thème du texte fictif ; un de : {', '.join(THEMES)}")
    p.add_argument("--salience", default="haute", help="haute|moyenne|faible (texte fictif)")
    p.add_argument("--type-vote", default="scrutin public solennel", help="type de vote (texte fictif)")
    args = p.parse_args()

    if args.fictif:
        if not args.titre:
            p.error("--fictif exige --titre \"...\"")
        scrutin = scrutin_fictif(args.titre, theme=args.theme,
                                 salience=args.salience, type_vote=args.type_vote)
        agents = build_agents(load_actifs())  # les 577 EN EXERCICE (pas les 640 de la législature)
    else:
        if not args.scrutin:
            p.error("--scrutin requis (ou utilise --fictif --titre \"...\")")
        scrutin = load_scrutin(args.scrutin)
        agents = build_agents(set(scrutin["participants"]))
    print(f"Scrutin {scrutin['uid']} — {scrutin['titre'][:90]}")
    print(f"thème {scrutin['theme']}, {len(agents)} députés participants"
          + (" [TEXTE FICTIF — prédiction pure]" if args.fictif else "") + "\n")

    if args.mock:
        brain = MockBrain()
    elif args.openrouter:
        from openrouter_brain import OpenRouterBrain
        from prompt_pack import PromptPacks
        mmap = json.load(open(args.models_map, encoding="utf-8")) if args.models_map else None
        brain = OpenRouterBrain(packs=PromptPacks(), priors=MockBrain(), models_map=mmap)
    else:
        from model_io import GemmaBrain

        brain = GemmaBrain(adapter=args.adapter, steer=not args.no_steer)

    res = simulate(brain, scrutin, agents, rounds=args.rounds,
                   speakers_per_group=args.speakers_per_group,
                   casting_reel=args.casting_reel, calibre=args.calibre)

    reel = scrutin["reel"]  # None en mode fictif
    print("\n================ TALLY ================")
    if reel:
        print(f"{'':12s}{'avant débat':>12s}{'après débat':>12s}{'réel':>8s}")
        for pos in POSITIONS:
            print(f"{pos:12s}{res['tally_avant_debat'][pos]:>12.1f}"
                  f"{res['tally'][pos]:>12.1f}{reel[pos]:>8d}")
    else:
        print(f"{'':12s}{'avant débat':>12s}{'après débat':>12s}")
        for pos in POSITIONS:
            print(f"{pos:12s}{res['tally_avant_debat'][pos]:>12.1f}{res['tally'][pos]:>12.1f}")
    pred_sort = "adopté" if res["tally"]["pour"] > res["tally"]["contre"] else "rejeté"
    print(f"\nVoix déplacées par la séance : {res['voix_deplacees_par_le_debat']}")
    if reel:
        print(f"Sort prédit : {pred_sort} — sort réel : {scrutin['sort']}")
    else:
        print(f"Sort prédit : {pred_sort} — texte fictif, aucune issue réelle")

    out = {"scrutin": {k: scrutin[k] for k in ("uid", "titre", "theme", "sort")}, "reel": reel, **res}
    os.makedirs("runs", exist_ok=True)
    path = os.path.join("runs", f"seance_{scrutin['uid']}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
