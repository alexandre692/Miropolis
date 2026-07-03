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


def load_affinites():
    path = os.path.join("data", "group_affinites.json")
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def load_propension():
    path = os.path.join("data", "orateurs_propension.json")
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


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


def simulate(brain, scrutin, agents, rounds=2, speakers_per_group=1, verbose=True):
    groups = sorted({a.groupe for a in agents})
    affinites = load_affinites()
    propension = load_propension()
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
            # orateurs prédits par leur historique réel : les spécialistes du
            # thème du texte parlent (médiane réelle : 32 orateurs/séance)
            members.sort(key=lambda a: speaker_score(a, scrutin.get("theme", "autre"),
                                                     propension), reverse=True)
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
                opp = min(groups, key=lambda x: affinites.get(g, {}).get(x, 0.0))
                hostilite = -affinites.get(g, {}).get(opp, 0.0)
                if hostilite > 0.3 and hash((scrutin["uid"], r, speaker.acteur)) % 10 < int(hostilite * 6):
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
                            print(f"      ↯ {heckler.nom} ({opp}) : {cri[:90]}")
        for a in agents:
            a.fj_update(heard, affinites)
        last_speakers = round_speakers
        if hasattr(brain, "summarize"):  # résumé roulant par l'orchestrateur
            summary = brain.summarize(transcript[-len(heard):], summary)
        else:
            summary = f"round {r + 1} : {len(heard)} interventions, groupes {', '.join(groups)}"

    tally = {p: round(sum(a.opinion[p] for a in agents), 1) for p in POSITIONS}
    deplace = round(sum(abs(tally[p] - tally_avant[p]) for p in POSITIONS) / 2, 1)
    return {"group_probs": group_probs, "transcript": transcript,
            "tally_avant_debat": tally_avant, "tally": tally,
            "voix_deplacees_par_le_debat": deplace}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scrutin", required=True, help="uid, ex. VTANR5L17V4000")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--openrouter", action="store_true",
                   help="moteur API multi-modèles (OPENROUTER_API_KEY requise)")
    p.add_argument("--adapter", default=None)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--speakers-per-group", type=int, default=1)
    p.add_argument("--no-steer", action="store_true")
    args = p.parse_args()

    scrutin = load_scrutin(args.scrutin)
    agents = build_agents(set(scrutin["participants"]))
    print(f"Scrutin {scrutin['uid']} — {scrutin['titre'][:90]}")
    print(f"thème {scrutin['theme']}, {len(agents)} députés participants\n")

    if args.mock:
        brain = MockBrain()
    elif args.openrouter:
        from openrouter_brain import OpenRouterBrain
        from prompt_pack import PromptPacks
        brain = OpenRouterBrain(packs=PromptPacks(), priors=MockBrain())
    else:
        from model_io import GemmaBrain

        brain = GemmaBrain(adapter=args.adapter, steer=not args.no_steer)

    res = simulate(brain, scrutin, agents, rounds=args.rounds, speakers_per_group=args.speakers_per_group)

    reel = scrutin["reel"]
    print("\n================ TALLY ================")
    print(f"{'':12s}{'avant débat':>12s}{'après débat':>12s}{'réel':>8s}")
    for pos in POSITIONS:
        print(f"{pos:12s}{res['tally_avant_debat'][pos]:>12.1f}"
              f"{res['tally'][pos]:>12.1f}{reel[pos]:>8d}")
    pred_sort = "adopté" if res["tally"]["pour"] > res["tally"]["contre"] else "rejeté"
    print(f"\nVoix déplacées par la séance : {res['voix_deplacees_par_le_debat']}")
    print(f"Sort prédit : {pred_sort} — sort réel : {scrutin['sort']}")

    out = {"scrutin": {k: scrutin[k] for k in ("uid", "titre", "theme", "sort")}, "reel": reel, **res}
    os.makedirs("runs", exist_ok=True)
    path = os.path.join("runs", f"seance_{scrutin['uid']}.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
