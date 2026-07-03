"""
DeputyAgent — l'agent-député du cerveau Miropolis.

Un agent = ancre déterministe (profil de votes révélés) + saillance (curseurs
SAE) + état (ledger de séance + opinion Friedkin-Johnsen).

Le vote n'est JAMAIS inventé par le LLM : l'opinion initiale dérive de la
position du groupe (prédite par le LoRA) modulée par la loyauté et la
dissidence thématique HISTORIQUES du député. Le débat ne déplace l'opinion
qu'à hauteur de (1 - λ), λ = loyauté (médiane réelle : 0.97).
"""

from dataclasses import dataclass, field

POSITIONS = ("pour", "contre", "abstention")


@dataclass
class DeputyAgent:
    acteur: str
    nom: str
    groupe: str
    loyaute: float  # [0.5, 1.0] observé ; None -> 0.95
    abstention_rate: float
    dissidences: dict  # theme -> taux de dissidence historique
    cursors: list = field(default_factory=list)  # features SAE [{id, coef, label?}]
    opinion0: dict = None  # distribution initiale {pour, contre, abstention}
    opinion: dict = None  # distribution courante (mise à jour FJ)
    ledger: list = field(default_factory=list)  # [{"round", "dit"}] — engagements pris

    @classmethod
    def from_profile(cls, acteur, profil, cursors=None):
        return cls(
            acteur=acteur,
            nom=profil["nom"],
            groupe=profil["groupe"],
            loyaute=profil.get("loyaute") if profil.get("loyaute") is not None else 0.95,
            abstention_rate=profil.get("abstention_rate", 0.02),
            dissidences=profil.get("dissidences", {}),
            cursors=(cursors or {}).get("features", []),
        )

    def init_opinion(self, group_probs, theme):
        """Opinion initiale = position du groupe modulée par la dissidence
        thématique du député. La masse dissidente part vers l'abstention
        (proportionnellement à son taux d'abstention) et vers la position
        opposée — direction de dissidence inconnue = heuristique V0 assumée."""
        d = min(self.dissidences.get(theme, 0.0) + (1.0 - self.loyaute) * 0.5, 0.9)
        maj = max(group_probs, key=group_probs.get)
        opp = "contre" if maj == "pour" else "pour"
        w_abst = min(max(self.abstention_rate * 5, 0.2), 0.8)
        o = {p: (1 - d) * group_probs.get(p, 0.0) for p in POSITIONS}
        o["abstention"] += d * w_abst
        o[opp] += d * (1 - w_abst)
        total = sum(o.values()) or 1.0
        self.opinion0 = {p: v / total for p, v in o.items()}
        self.opinion = dict(self.opinion0)
        return self.opinion

    def fj_update(self, heard, affinites=None):
        """Friedkin-Johnsen SIGNÉ : opinion_t = λ·ancre + (1-λ)·influence sociale.
        `heard` = [{"groupe": g, "opinion": {...}}]. Le poids de chaque orateur
        = l'affinité de vote MESURÉE entre son groupe et le mien
        (data/group_affinites.json) : positive → attraction, négative →
        répulsion (le discours d'un opposant renforce la position inverse)."""
        if not heard or self.opinion is None:
            return self.opinion
        lam = self.loyaute
        num = {p: 0.0 for p in POSITIONS}
        den = 0.0
        for h in heard:
            op = h.get("opinion", h) if isinstance(h, dict) else h
            g = h.get("groupe") if isinstance(h, dict) else None
            w = (affinites or {}).get(self.groupe, {}).get(g, 0.3) if g else 0.3
            if w >= 0:
                contrib = op
            else:  # répulsion : la masse pour/contre s'inverse (boomerang)
                contrib = {"pour": op.get("contre", 0.0),
                           "contre": op.get("pour", 0.0),
                           "abstention": op.get("abstention", 0.0)}
            for p in POSITIONS:
                num[p] += abs(w) * contrib.get(p, 0.0)
            den += abs(w)
        if den == 0:
            return self.opinion
        social = {p: num[p] / den for p in POSITIONS}
        self.opinion = {p: lam * self.opinion0[p] + (1 - lam) * social[p] for p in POSITIONS}
        return self.opinion

    def remember(self, round_idx, statement):
        self.ledger.append({"round": round_idx, "dit": statement})

    def context_block(self, debate_summary, last_turns):
        """Contexte injecté quand CE député prend la parole : identité + ancre
        + ses engagements passés + où en est le débat. Borné par construction."""
        lines = [
            f"Tu es {self.nom}, député du groupe {self.groupe}.",
            f"Ta loyauté de vote au groupe est de {self.loyaute:.0%}"
            + (" — tu es un dissident fréquent." if self.loyaute < 0.85 else "."),
        ]
        if self.cursors:
            themes = ", ".join(str(c.get("label", c["id"])) for c in self.cursors[:3])
            lines.append(f"Tes angles d'attaque récurrents : {themes}.")
        if self.ledger:
            lines.append("Tu as déjà dit dans ce débat : " + " | ".join(e["dit"][:120] for e in self.ledger[-3:]))
        if debate_summary:
            lines.append(f"Résumé du débat : {debate_summary}")
        if last_turns:
            lines.append("Dernières interventions : " + " | ".join(t[:150] for t in last_turns))
        return "\n".join(lines)
