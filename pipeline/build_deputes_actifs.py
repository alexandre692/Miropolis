"""Extrait la liste des députés EN EXERCICE depuis l'export open data AN
AMO10 (deputes_actifs_mandats_actifs_organes) → data/deputes_actifs.json
(liste triée d'acteurRef "PAxxxxx").

Pourquoi : deputes_profils.json contient TOUS les acteurs de la XVIIe
législature (640 = départs + remplaçants inclus). Le mode --fictif de
brain/seance.py prenait tout le monde → 640 députés dans l'hémicycle
(impossible, 577 sièges). Ce fichier sert de filtre.

Usage (racine) :
    python pipeline/build_deputes_actifs.py
Le zip est lu en place (pas d'extraction), comme le reste du pipeline :
    data_amdt/AMO10_deputes_actifs.json.zip
Source : https://data.assemblee-nationale.fr/static/openData/repository/17/
         amo/deputes_actifs_mandats_actifs_organes/ (Licence Ouverte)
"""

import json
import os
import re
import zipfile

ZIP = os.path.join("data_amdt", "AMO10_deputes_actifs.json.zip")
OUT = os.path.join("data", "deputes_actifs.json")


def main():
    z = zipfile.ZipFile(ZIP)
    refs = sorted({m.group(1) for n in z.namelist() if (m := re.search(r"/acteur/(PA\d+)\.json$", n))})
    json.dump(refs, open(OUT, "w", encoding="utf-8"))
    print(f"{len(refs)} députés actifs -> {OUT}")


if __name__ == "__main__":
    main()
