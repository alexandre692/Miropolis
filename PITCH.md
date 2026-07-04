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
