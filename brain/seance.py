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
    meta["participants"] = sorted(participants)
    return meta


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
        "participants": [],  # vide → build_agents(None) prend TOUS les députés
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
trois lors""".split())


def _tokens(text):
    import re
    return {w for w in re.findall(r"[a-zàâäéèêëîïôöùûüç\-]{5,}", text.lower())
            if w not in STOP_CASTING}


def predicted_casting(scrutin):
    """CASTING PRÉDIT (défaut) — validé par backtest temporel :
    precision@3 = 0.205 vs 0.133 (thème) vs 0.056 (hasard), soit 3.7×.
    Signal : overlap lexical entre le TITRE du texte (connu à l'avance —
    l'ordre du jour est public) et les interventions du député sur les
    30 jours précédant STRICTEMENT le scrutin, décroissance ~10 j.
    Zéro donnée du jour J. Retourne {acteur: score}."""
    import gzip
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


def simulate(brain, scrutin, agents, rounds=2, speakers_per_group=1, verbose=True,
             casting_reel=False):
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
            print(f"casting PRÉDIT (agenda+récence, p@3=0.205 vs hasard 0.056) : "
                  f"{len(orateurs_reels)} députés actifs sur ce sujet\n")
    group_probs = {g: brain.group_position(g, scrutin) for g in groups}
    for a in agents:
        a.init_opinion(group_probs[a.groupe], scrutin.get("theme", "autre"))
    # tally AVANT débat = les consignes de vote en entrant dans l'hémicycle ;
    # l'écart avec le tally final mesure ce que la séance elle-même a déplacé
    tally_avant = {p: round(sum(a.opinion[p] for a in agents), 1) for p in POSITIONS}

    transcript, summary = [], ""
    last_speakers = []
    for r in range(rounds):
        heard, round_speakers = [], []
        for g in groups:
            members = [a for a in agents if a.groupe == g]
            if not members:
                continue
            # casting : les orateurs RÉELS de la séance d'abord (s'ils sont
            # connus), sinon les spécialistes du thème (propension mesurée)
            members.sort(key=lambda a: (
                orateurs_reels.get(a.acteur, 0) * 1000
                + speaker_score(a, scrutin.get("theme", "autre"), propension)
            ), reverse=True)
            for speaker in members[:speakers_per_group]:
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
        agents = build_agents(None)  # tous les députés en exercice
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
        brain = OpenRouterBrain(packs=PromptPacks(), priors=MockBrain())
    else:
        from model_io import GemmaBrain

        brain = GemmaBrain(adapter=args.adapter, steer=not args.no_steer)

    res = simulate(brain, scrutin, agents, rounds=args.rounds,
                   speakers_per_group=args.speakers_per_group,
                   casting_reel=args.casting_reel)

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
