"""
OpenRouterBrain — moteur de génération du pivot prompt-engineering.
Même interface que MockBrain/GemmaBrain → brain/seance.py ne change pas.

Architecture :
  - ORCHESTRATEUR (gros modèle, température basse, sortie JSON stricte) :
    prédit la position de chaque groupe sur le texte (il reçoit le prior
    historique et doit motiver tout écart) + maintient le résumé roulant du
    débat (le "fil" qui évite la perte de contexte entre les tours).
  - DÉPUTÉS (modèles rapides, température plus haute) : parlent avec leur
    pack d'identité = profil chiffré + extraits VERBATIM de leurs vraies
    interventions (brain/prompt_pack.py), choisis selon le thème du texte.
  - Les VOTES restent déterministes (ancre + FJ) — l'API ne vote jamais.

Mapping groupe→pool de modèles : brain/models_map.json (éditable). Chaque
député pioche un modèle dans le pool de son groupe, de façon stable (hash de
son acteurRef). Taille du pool pondérée par le poids réel du groupe à
l'Assemblée — plus un groupe pèse de sièges, plus il a de voix différentes.
Note honnête : utiliser des modèles différents est un choix d'ingénierie pour
diversifier les styles, PAS un claim scientifique — la représentativité vient
des données (verbatims, ancres), pas de la loterie des architectures.

Prérequis : variable d'environnement OPENROUTER_API_KEY.
"""

import hashlib
import json
import os
import re
import time
import urllib.request

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_MAP = os.path.join(os.path.dirname(__file__), "models_map.json")

DEFAULTS = {
    # vérifier les ids exacts sur https://openrouter.ai/models
    "orchestrateur": "anthropic/claude-sonnet-4.5",
    "depute_default": "mistralai/mistral-small-3.2-24b-instruct",
}

POSITIONS = ("pour", "contre", "abstention")


def _get_key():
    """Clé : env OPENROUTER_API_KEY, sinon fichier local .openrouter_key
    (racine du repo ou home) — JAMAIS commité (.gitignore)."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    for p in (os.path.join(os.path.dirname(__file__), "..", ".openrouter_key"),
              os.path.join(os.path.expanduser("~"), ".openrouter_key")):
        if os.path.exists(p):
            return open(p, encoding="utf-8").read().strip()
    return None


def _call(model, messages, temperature=0.7, max_tokens=400, json_mode=False,
          retries=5):
    key = _get_key()
    if not key:
        raise RuntimeError("Clé OpenRouter introuvable : définir OPENROUTER_API_KEY "
                           "ou créer un fichier .openrouter_key (racine repo ou home)")
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    body = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                API_URL, data=body, method="POST",
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json",
                         "HTTP-Referer": "https://github.com/alexandre692/Miropolis",
                         "X-Title": "Miropolis"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            if content is None:  # refus/filtre du modèle : contenu vide renvoyé tel quel
                raise RuntimeError(f"contenu vide renvoyé par {model} (refus/filtre probable)")
            return content
        except urllib.error.HTTPError as e:  # 429 (rate limit) surtout : backoff long
            last_err = e
            time.sleep(min(3 * (attempt + 1), 15))
        except Exception as e:  # réseau/parse/contenu vide : backoff plus court
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"OpenRouter KO après {retries} essais : {last_err}")


def _extract_intervention(raw):
    """Isole le texte dans <intervention>...</intervention> et retire toute
    balise résiduelle. Tolérant à la troncature par max_tokens (balise
    ouverte jamais refermée) et au markdown que le modèle colle parfois
    autour ou À L'INTÉRIEUR des chevrons (**<...>**, <**...>**...).
    Si la réflexion elle-même est tronquée (jamais fermée), on garde son
    contenu plutôt que de renvoyer du vide — dégradé vaut mieux que blanc."""
    # normalise d'abord : tout markdown collé à un chevron disparaît, pour
    # que les regex de balises ci-dessous matchent <intervention>/<reflexion>
    # sous leur forme propre quelle que soit la fantaisie du modèle.
    clean = re.sub(r"\*+(?=[<>])|(?<=[<>])\*+", "", raw)
    m = re.search(r"<intervention>(.*?)(?:</intervention>|$)", clean, re.S)
    if m:
        text = m.group(1)
    else:
        text = re.sub(r"<reflexion>.*?</reflexion>", "", clean, flags=re.S)
    return re.sub(r"</?(?:intervention|reflexion)>", "", text, flags=re.I).strip()


def _parse_probs(text):
    """Extrait {pour, contre, abstention} d'une réponse JSON (tolérant)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    probs = {}
    for p in POSITIONS:
        v = d.get(p)
        if not isinstance(v, (int, float)) or v < 0:
            return None
        probs[p] = float(v)
    total = sum(probs.values())
    return {p: v / total for p, v in probs.items()} if total > 0 else None


class OpenRouterBrain:
    def __init__(self, packs=None, priors=None, models_map=None):
        cfg = dict(DEFAULTS)
        if models_map is None and os.path.exists(MODELS_MAP):
            models_map = json.load(open(MODELS_MAP, encoding="utf-8"))
        self.map = models_map or {}
        cfg.update(self.map.get("_config", {}))
        self.orch_model = cfg["orchestrateur"]
        self.dep_default = cfg["depute_default"]
        self.packs = packs          # PromptPacks (peut être None → pas de verbatims)
        self.priors = priors        # MockBrain, pour ancrer l'orchestrateur

    def _dep_model(self, agent):
        """Pioche dans le pool de modèles du groupe (voir models_map.json),
        de façon stable pour un même député (hash de son acteurRef) — pas de
        tirage aléatoire, un run est reproductible."""
        pool = self.map.get(agent.groupe, self.dep_default)
        if isinstance(pool, str):
            return pool
        if not pool:
            return self.dep_default
        h = int(hashlib.sha256(agent.acteur.encode("utf-8")).hexdigest(), 16)
        return pool[h % len(pool)]

    # ---------- ORCHESTRATEUR ----------

    # part du prior historique conservée dans la position finale d'un groupe.
    # Léger filet de sécurité : dampe les erreurs confiantes de l'orchestrateur
    # sans écraser ses prédictions justes. Le vrai levier est le prompt
    # (raisonnement sur le camp politique) ; ce mélange ne fait qu'amortir.
    PRIOR_BLEND = 0.25

    def group_position(self, groupe, scrutin):
        prior = self.priors.group_position(groupe, scrutin) if self.priors else None
        prior_txt = (f"Ligne de vote HISTORIQUE MESURÉE du groupe {groupe} sur ce "
                     f"thème : {json.dumps(prior, ensure_ascii=False)}. C'est ton "
                     f"point d'ancrage : ne t'en écarte que si le texte le justifie "
                     f"clairement, et de façon mesurée."
                     if prior else "")
        messages = [{"role": "user", "content": (
            "Tu es un politologue expert de l'Assemblée nationale française "
            "(XVIIe législature, 2024-2026). Estime la position du groupe "
            f"{groupe} sur ce scrutin.\n"
            f"Type : {scrutin.get('typeVote', '?')} — thème {scrutin.get('theme', '?')} "
            f"— saillance {scrutin.get('salience', '?')}\n"
            f"Titre : {scrutin.get('titre', '')}\n"
            f"{scrutin.get('texte_contexte', '')}\n{prior_txt}\n\n"
            "RAISONNE EN DEUX TEMPS.\n"
            "1) Le texte est-il CLIVANT (idéologique, porté par un camp — "
            "retraites, immigration, fiscalité...) ou plutôt TECHNIQUE / "
            "CONSENSUEL (simplification, ratification, sujet local ou "
            "d'intendance) ? Un texte technique est souvent adopté LARGEMENT, "
            "au-delà des clivages : dans ce cas la majorité des groupes vote "
            "POUR (ou s'abstient), l'opposition frontale est rare.\n"
            "2) Si le texte est CLIVANT : à quel bord appartient-il (qui le "
            "porte, qui en tire crédit) ? Un groupe vote alors presque toujours "
            "CONTRE un texte d'un camp adverse, MÊME si le thème lui est cher — "
            "la logique partisane l'emporte. Ne te fais pas piéger par un "
            "intitulé sympathique ('plus juste', 'pour tous') : regarde qui "
            "l'a écrit. Souviens-toi aussi que le groupe de la majorité "
            "présidentielle (EPR) et ses alliés soutiennent les textes du "
            "gouvernement. En cas de doute, colle au prior historique.\n"
            'Réponds en JSON strict : {"pour": p, "contre": p, "abstention": p, '
            '"raison": "une phrase"} avec p sommant à 1.')}]
        try:
            probs = _parse_probs(_call(self.orch_model, messages,
                                       temperature=0.2, json_mode=True))
        except RuntimeError as e:
            import sys
            print(f"  [orchestrateur KO sur {groupe} -> repli prior] {e}", file=sys.stderr)
            probs = None
        if not probs:
            return prior or {"pour": 1 / 3, "contre": 1 / 3, "abstention": 1 / 3}
        if not prior:
            return probs
        # ancrage : mélange prior mesuré + estimation orchestrateur, pour que
        # ce dernier ne puisse pas faire basculer un groupe contre son histoire
        # sur un raisonnement fragile.
        a = self.PRIOR_BLEND
        blended = {p: a * prior.get(p, 0.0) + (1 - a) * probs.get(p, 0.0) for p in POSITIONS}
        tot = sum(blended.values()) or 1.0
        return {p: v / tot for p, v in blended.items()}

    def summarize(self, transcript_tail, previous_summary=""):
        """Résumé roulant du débat — le contexte récurrent qui garde le fil."""
        tours = "\n".join(f"- {t['nom']} ({t['groupe']}) : {t['texte']}"
                          for t in transcript_tail)
        messages = [{"role": "user", "content": (
            "Tu es le secrétariat de séance de l'Assemblée nationale. Mets à "
            "jour le résumé du débat en 4 phrases max : positions exprimées, "
            "arguments saillants, tensions.\n"
            f"Résumé précédent : {previous_summary or '(début de séance)'}\n"
            f"Nouvelles interventions :\n{tours}")}]
        try:
            return _call(self.orch_model, messages, temperature=0.2,
                         max_tokens=250).strip()
        except RuntimeError:
            return previous_summary

    def recap_final(self, transcript, scrutin, previous_summary=""):
        """Résumé de clôture dans la voix de la présidente de séance, donné
        juste avant la mise aux voix — le fil du débat défile vite, ce
        récapitulatif permet au public de raccrocher les wagons."""
        tours = "\n".join(f"- {t['nom']} ({t['groupe']}) : {t['texte']}"
                          for t in transcript if not t.get("interjection"))
        messages = [{"role": "user", "content": (
            "Tu es Yaël Braun-Pivet, présidente de l'Assemblée nationale. Le "
            f"débat sur « {scrutin.get('titre', '')} » vient de se clore. "
            "Fais, à la première personne, un résumé de clôture en 4 à 6 "
            "phrases avant la mise aux voix : rappelle l'objet du texte, puis "
            "les grandes lignes de clivage entre groupes (qui soutient, qui "
            "s'oppose, et pourquoi, en une phrase chacun) — sans trancher "
            "toi-même. Registre solennel de présidence de séance.\n"
            f"Résumé roulant du débat : {previous_summary or '(aucun)'}\n"
            f"Interventions :\n{tours}")}]
        try:
            return _call(self.orch_model, messages, temperature=0.3,
                         max_tokens=500).strip()
        except RuntimeError:
            return previous_summary

    # ---------- DÉPUTÉS ----------

    def intervention(self, agent, scrutin, context, extra_cursors=None,
                     max_tokens=500):  # réflexion privée + intervention
        # extra_cursors (contagion SAE) sans objet en mode API : la contagion
        # thématique passe par le résumé + les dernières interventions du contexte.
        bloc_verbatims = ""
        if self.packs:
            bloc_verbatims = self.packs.bloc(agent.acteur, scrutin.get("theme"),
                                             n=3, before=scrutin.get("date"))
        system = (
            f"{context}\n\n{bloc_verbatims}\n\n"
            "RÈGLES STRICTES : 4 à 6 phrases, première personne, registre de "
            "séance publique (adresse à la présidence, interpellations). Ne "
            "cite JAMAIS un chiffre qui n'est pas dans le contexte ci-dessus. "
            "Réponds aux arguments réellement exprimés avant toi SANS "
            "reformuler ni paraphraser leur phrase d'ouverture ou leur "
            "tournure — même en cas d'accord, retrouve TON angle et TES mots, "
            "jamais un écho de ce qui vient d'être dit. Ne change "
            "pas de position par rapport à tes engagements listés. Tu n'es PAS "
            "là pour trouver un consensus : maintiens le désaccord si c'est ta "
            "position — c'est un débat parlementaire, pas une médiation. "
            "Sois CASH, pas diplomate : les député·es ont l'habitude de "
            "l'affrontement en hémicycle. Attaque frontalement les arguments "
            "adverses (jamais la personne), quitte à être dur, sarcastique ou "
            "consterné. Bannis les tournures édulcorées ('je comprends votre "
            "point de vue, mais...', 'avec tout le respect que je vous dois', "
            "'je ne suis pas sûr que...') : un vrai débat parlementaire est "
            "vif et tranchant, pas une table ronde consensuelle.")
        user = (f"Le texte en débat : {scrutin.get('titre', '')}\n"
                f"{scrutin.get('texte_contexte', '')}\n"
                "D'abord, réfléchis en privé dans <reflexion>...</reflexion> "
                "(2-3 phrases : que dit vraiment ce texte, quel est TON angle "
                "vu ton historique, à quel argument précédent répondre). Puis "
                "prononce ton intervention dans <intervention>...</intervention>.")
        try:
            raw = _call(self._dep_model(agent),
                        [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
                        temperature=0.85, max_tokens=max_tokens)
            return _extract_intervention(raw)
        except RuntimeError as e:
            return f"[intervention indisponible : {e}]"

    def interjection(self, agent, target_text, max_tokens=60):
        """Interjection spontanée (1 phrase, style 'Exclamations sur les
        bancs…') d'un opposant qui vient d'entendre `target_text`."""
        messages = [{"role": "user", "content": (
            f"Tu es {agent.nom}, député {agent.groupe} à l'Assemblée "
            f"nationale. Un orateur adverse vient de dire : "
            f"« {target_text[:300]} ». Lance UNE interjection d'une phrase, "
            "mordante et sans détour — attaque l'argument, jamais la "
            "personne (pas d'insulte), mais ne mâche pas tes mots.")}]
        try:
            return _call(self._dep_model(agent), messages,
                         temperature=1.0, max_tokens=max_tokens).strip()
        except RuntimeError:
            return "(Protestations.)"
