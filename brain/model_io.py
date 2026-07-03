"""
Interface unique vers le modèle : GemmaBrain (GPU, LoRA + steering SAE) et
MockBrain (local, priors historiques) — même API, donc la séance entière se
teste SANS GPU et le swap est un flag.

Lecture des votes PAR LOGITS, pas par échantillonnage : on contraint la sortie
aux trois positions et on lit P(pour|contre|abstention) directement sur les
logits du premier token → une probabilité calibrée par passe, pas un tirage
bruité. Tally attendu = somme des probabilités.
"""

import json
import os
from collections import Counter, defaultdict

POSITIONS = ("pour", "contre", "abstention")

# Même gabarit que le dataset de fine-tuning (training/train_qlora.py) :
# le LoRA est interrogé EXACTEMENT dans le format où il a appris.
SYSTEM_TMPL = (
    "Tu es un député de l'Assemblée nationale française, membre du groupe "
    "{groupe}. On te soumet un texte au vote, dans le domaine {theme}. Réponds "
    "au format 'position: <pour|contre|abstention> | cohésion: "
    "<unanime|forte|modérée|divisé>', en cohérence avec la ligne politique de "
    "ton groupe et sa cohésion habituelle sur ce type de sujet."
)
USER_TMPL = (
    "Type de vote: {type_vote} (saillance: {salience})\n"
    "Thème: {theme}\nTitre: {titre}\n"
    "Quelle est ta position et la cohésion attendue de ton groupe ?"
)


class MockBrain:
    """Priors (groupe, thème) calculés sur data/scrutins_enrichis.jsonl.
    Permet de développer et tester toute la séance en local."""

    def __init__(self, enrichis=os.path.join("data", "scrutins_enrichis.jsonl")):
        counts = defaultdict(Counter)
        with open(enrichis, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("position"):
                    counts[(r["group"], r.get("theme", "autre"))][r["position"]] += 1
                    counts[(r["group"], None)][r["position"]] += 1
        self._priors = counts

    def group_position(self, groupe, scrutin):
        c = (
            self._priors.get((groupe, scrutin.get("theme", "autre")))
            or self._priors.get((groupe, None))
            or Counter({"contre": 1})
        )
        total = sum(c.values())
        return {p: c.get(p, 0) / total for p in POSITIONS}

    def intervention(self, agent, scrutin, context, extra_cursors=None):
        maj = max(agent.opinion or {"contre": 1}, key=(agent.opinion or {"contre": 1}).get)
        amb = f" [+{len(extra_cursors)} concepts ambiants]" if extra_cursors else ""
        return (f"[mock] {agent.nom} ({agent.groupe}) s'exprime {maj} "
                f"sur « {scrutin.get('titre', '?')[:60]}… »{amb}")


class GemmaBrain:
    """Gemma-2-9B-it 4-bit + LoRA (training/out/…) + steering SAE optionnel.
    Import torch/transformers UNIQUEMENT ici — le reste du cerveau reste pur."""

    def __init__(self, base_model="google/gemma-2-9b-it", adapter=None, sae_layer=20, sae_width="16k", steer=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb,
            device_map="auto",
            attn_implementation="eager",
        )
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

        self.steerer = None
        if steer:
            import sys

            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "steering"))
            from jumprelu_sae import JumpReLUSAE
            from steering import SAESteerer

            sae = JumpReLUSAE.from_hub(layer=sae_layer, width=sae_width, device=self.model.device)
            self.steerer = SAESteerer(self.model, sae, layer=sae_layer)

        # ids du premier token de chaque position ("pour"/"contre"/"abstention")
        self._pos_ids = {p: self.tokenizer(p, add_special_tokens=False)["input_ids"][0] for p in POSITIONS}

    def _prompt(self, system, user):
        chat = [{"role": "user", "content": system + "\n\n" + user}]
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def group_position(self, groupe, scrutin):
        """P(pour/contre/abstention) du groupe, lue sur les logits après
        l'amorce 'position: ' — format d'entraînement exact."""
        system = SYSTEM_TMPL.format(groupe=groupe, theme=scrutin.get("theme", "autre"))
        user = USER_TMPL.format(
            type_vote=scrutin.get("typeVote", "scrutin public ordinaire"),
            salience=scrutin.get("salience", "moyenne"),
            theme=scrutin.get("theme", "autre"),
            titre=scrutin.get("titre", ""),
        )
        prompt = self._prompt(system, user) + "position: "
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        sel = self.torch.tensor([self._pos_ids[p] for p in POSITIONS], device=logits.device)
        probs = self.torch.softmax(logits[sel], dim=-1).tolist()
        return dict(zip(POSITIONS, probs))

    def intervention(self, agent, scrutin, context, extra_cursors=None,
                     max_new_tokens=180):
        """Intervention en séance, steerée par les curseurs SAE du député
        + les concepts ambiants du débat (contagion de saillance)."""
        system = context  # le context_block de l'agent EST le system prompt
        user = (
            f"Prononce une intervention courte (3-5 phrases) en séance sur : "
            f"{scrutin.get('titre', '')}. Registre parlementaire, première personne."
        )
        prompt = self._prompt(system, user)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        def gen():
            with self.torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                )
            return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

        cursors = list(agent.cursors or []) + list(extra_cursors or [])
        if self.steerer and cursors:
            with self.steerer.steering(cursors):
                return gen()
        return gen()
