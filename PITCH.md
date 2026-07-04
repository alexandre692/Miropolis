# MIROPOLIS — le jumeau numérique de l'Assemblée nationale

## Le pitch (3 minutes)

**L'idée.** Quand un texte de loi arrive dans l'hémicycle, personne ne sait
exactement ce qui va s'y passer — qui montera au créneau, quels arguments
s'affronteront, comment finira le vote. Miropolis le simule **avant**. Vous
lui donnez un texte à l'ordre du jour ; il vous rend la séance : les
orateurs, le débat, le scrutin.

**Comment ? Nous n'avons rien inventé — nous avons fait parler la mémoire de
l'Assemblée.** Cinq sources open data croisées : 7 966 scrutins nominatifs,
71 500 interventions en séance, 121 110 amendements, les dossiers
législatifs, les présences en commission. De ce croisement naissent 577
agents — un par député — dont **rien n'est imaginé** : leur fidélité de vote
est mesurée (médiane réelle : 97 %), leurs sujets de dissidence sont mesurés,
et quand ils parlent, ils s'appuient sur **leurs propres mots au Journal
officiel** — chaque extrait est cité, date et compte rendu à l'appui.

**Trois prédictions, une séance.**
1. **Qui parlera ?** Le rapporteur, les auteurs d'amendements, les habitués
   du dossier : notre casting capte **la moitié des orateurs réels du débat,
   neuf fois mieux que le hasard** — sans utiliser une seule donnée du jour J.
2. **Comment le débat vivra ?** Les agents s'écoutent, se répondent et
   s'influencent selon des affinités **mesurées sur leurs votes** : un
   discours LFI rapproche les écologistes et braque la droite — c'est dans
   les données, pas dans un prompt.
3. **Comment finira le vote ?** L'IA ne vote jamais : le scrutin sort des
   statistiques individuelles des députés. Même simulation, même prédiction —
   reproductible devant vous.

**Et la confiance ?** C'est le cœur. Le modèle est **gelé à la veille de la
séance** qu'il prédit — il ne sait rien du lendemain, et nous le rejouons en
direct contre le résultat réel. Nous affichons la barre naïve à battre
(52 %), nous publions nos protocoles de test, et nous savons dire **où
s'arrête le prédictible** : nous avons calculé le plafond théorique de notre
propre tâche. Une IA de confiance, ce n'est pas une IA qui promet tout —
c'est une IA qui montre ses sources, ses limites, et qui se laisse vérifier.

*Miropolis : l'open data de l'Assemblée, transformé en intuition politique.*

---

## Sous le capot — l'architecture agentique (version développée, 5-8 min ou Q&A)

> Le fil rouge de chaque choix technique : **si l'Assemblée a la donnée, on la
> mesure ; sinon — et seulement sinon — on demande au modèle.** L'IA générative
> n'intervient que là où il faut du langage ; partout ailleurs, ce sont les
> données de l'Assemblée qui décident.

### 1. Une société d'agents, pas un chatbot

Miropolis n'est pas « une IA à qui on pose une question ». C'est une
**simulation multi-agents** : 577 agents-députés, une présidente de séance,
et un orchestrateur — qui interagissent en temps réel selon le protocole
réel de l'hémicycle (tours de parole, interpellations, interjections, mise
aux voix).

Chaque agent est construit en trois couches, **toutes dérivées des données de
l'Assemblée** :
- **Son ancre comportementale** : sa fidélité de vote (calculée sur les 7 966
  scrutins nominatifs de la législature — médiane réelle 97 %), ses thèmes de
  dissidence, son taux d'abstention. C'est elle qui vote — jamais le modèle
  de langage.
- **Sa mémoire** : à chaque prise de parole, l'agent reçoit ses engagements
  précédents dans le débat (il ne se contredit pas) et le résumé roulant de
  la séance, maintenu par l'orchestrateur (il ne perd jamais le fil). Mémoire
  individuelle + mémoire collective : c'est ce qui transforme des appels IA
  isolés en un débat continu.
- **Sa voix** : des extraits *verbatim* de ses vraies interventions au
  Journal officiel, sélectionnés selon le texte débattu.

Et entre les agents, **l'influence circule selon des poids mesurés** : nous
avons calculé la matrice d'affinités entre groupes sur l'ensemble de leurs
co-votes (RN↔UDR : +0.80 ; gauche⊣UDR : −0.6). Quand un orateur parle, il
rapproche ses alliés et braque ses opposants — avec l'intensité que
l'histoire des votes lui donne, pas celle qu'un développeur aurait choisie.

### 2. Le prompt engineering, adossé à la recherche

Nous n'avons pas « écrit des prompts » : nous avons appliqué les résultats
récents de la recherche sur les agents génératifs, chacun câblé et testé :

- **L'identité par les faits, pas par les adjectifs.** Dire à un modèle
  « sois combatif » ne produit rien de fiable. Lui donner « ta fidélité de
  vote est de 71 %, tu as dissenté 12 fois sur la santé » produit un
  comportement. Chaque prompt d'agent est **assemblé automatiquement depuis
  les données de l'Assemblée**, pas rédigé à la main.
- **Les verbatims plutôt que les descriptions.** La recherche (Stanford,
  2024) montre qu'un agent fondé sur les *propos réels* d'une personne la
  prédit bien mieux qu'un agent fondé sur son profil. Nos agents reçoivent
  3 extraits réels de *leurs* interventions — choisis selon le thème du
  texte, **cités** (date + référence du compte rendu) : chaque mot de
  l'identité est vérifiable au Journal officiel.
- **La réinjection d'identité à chaque tour.** Les personas des LLM
  « dérivent » au fil d'une conversation (persona drift) : au bout de dix
  tours, tout le monde parle pareil. Remède documenté : réinjecter
  l'identité complète à chaque prise de parole — c'est notre architecture.
- **La réflexion privée avant la parole.** Chaque agent raisonne d'abord
  dans un canal privé (quel est ce texte ? quel est MON angle ? à qui
  répondre ?) puis prononce son intervention publique. Seule la seconde
  entre au compte rendu — comme un vrai député.
- **L'anti-consensus.** La littérature montre que des agents LLM mis en
  débat convergent artificiellement vers l'accord. Double verrou : une
  consigne explicite (« c'est un débat parlementaire, pas une médiation »)
  et surtout des opinions gérées HORS du modèle de langage, par la mécanique
  d'ancres et d'influence mesurée.
- **Les garde-fous anti-hallucination.** Interdiction de citer un chiffre
  absent du contexte fourni ; sorties structurées (JSON strict) pour tout ce
  qui est machine-lisible ; repli déterministe si un appel échoue — la
  simulation ne peut pas crasher en direct.

### 3. L'approche multi-modèles, et pourquoi

La séance fait travailler **deux étages de modèles, choisis pour leur rôle** :

- **L'orchestrateur** : un grand modèle, température basse, sorties JSON
  strictes. Il tient les fonctions de fiabilité : estimer la position de
  chaque groupe sur le texte — **ancré au prior historique calculé sur les
  votes, dont il doit justifier tout écart** — et maintenir le résumé
  roulant du débat.
- **Les députés** : des pools de modèles rapides et **volontairement
  divers** (plusieurs familles, ~30 modèles répartis sur les groupes, taille
  du pool proportionnelle au poids réel du groupe dans l'hémicycle). Chaque
  député est associé à son modèle de façon stable et reproductible.

La motivation est empirique : une simulation multi-agents servie par un
modèle unique **s'homogénéise** — mêmes tournures, mêmes réflexes, écho
entre orateurs (nous l'avons observé, puis corrigé en excluant les modèles
qui paraphrasaient ou raisonnaient en anglais). La diversité des moteurs
protège la diversité des voix. Et nous le disons honnêtement : c'est un
choix d'**ingénierie** au service du réalisme — la représentativité, elle,
vient des données de l'Assemblée (verbatims, ancres, affinités), pas de la
marque des modèles.

### 4. Ce que ça motive : une méthode, pas un hack

À chaque étage, la même discipline : **qui parle** est prédit depuis les
données (rapporteurs, amendements, commissions, historique — validé par
backtest temporel) ; **ce qui se dit** est généré sous contrainte des
données (verbatims sourcés, garde-fous) ; **ce qui se décide** est calculé
depuis les données (ancres, affinités, jamais le LLM). Le résultat se
vérifie contre la réalité, séance par séance, avec un modèle gelé à la
veille. C'est notre définition opérationnelle d'une IA de confiance
appliquée au Parlement : **le génératif au service du mesurable, jamais
l'inverse.**

---

## Antisèche Q&A (les chiffres exacts)

| Question probable | Réponse |
|---|---|
| Précision du casting ? | p@3 par groupe **0.399** ; plafond théorique (oracle) **0.683** → 58 % de l'omniscience. En liste proportionnelle : precision 0.42, **recall 0.49** sur les orateurs engagés (≥2 prises) ≈ 9× le hasard (5,6 %). Protocole : split temporel, 101 séances jamais vues, `pipeline/backtest_casting_final.py`. |
| Pourquoi pas plus ? | Les listes d'inscrits à la discussion générale ne sont **pas publiées** (feuilleton interne des groupes) ; le spontané (rappels au règlement) est irréductible. Vérifié : les QAG n'ont pas de date d'enregistrement ex-ante dans l'open data. |
| Le vote, comment ? | Jamais par le LLM. Position du groupe (ancrée au prior historique, l'orchestrateur doit justifier tout écart) × loyauté individuelle mesurée × influence Friedkin-Johnsen dont les poids sont les **affinités de co-vote mesurées** (RN↔UDR +0.80, LFI⊣droite −0.5/−0.7). Baseline à battre : 0.519. |
| Fuite de données ? | Pare-feu temporel partout : verbatims, amendements, nominations, présences — tout est daté et filtré strictement avant J. La démo l'énonce : « le modèle est gelé au 29 juin, voici ce qu'il prédit pour le 30 ». |
| Les personnalités ? | Aucune écrite à la main : profil = statistiques de vote + extraits réels sourcés (date + uid du compte rendu). Auditables ligne par ligne au JO. |
| Ça tourne où ? | La simulation déterministe : n'importe quel laptop, stdlib Python. La génération : API (multi-modèles). La démo 3D : un fichier HTML autonome, sans réseau. |
| Licences ? | 100 % open data Assemblée nationale, Licence Ouverte. Aucune donnée privée. |
| Et après ? | Mode prospectif déjà codé (`scrutin_fictif`) : un texte inventé → séance complète. Calibration probabiliste du casting (régression sur les 6 signaux) et personnalisation par activations (SAE) en R&D. |
