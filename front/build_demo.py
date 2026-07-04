"""
Injecte un ou plusieurs runs de séance (runs/seance_*.json) dans le template
hemicycle.html → front/demo.html autonome (ouvrable en double-clic, aucun
serveur requis). Avec plusieurs runs, le site en tire un au hasard à chaque
chargement de page — évite de toujours montrer le même scrutin.

Usage (racine) :
    python front/build_demo.py runs/seance_VTANR5L17V4000.json
    python front/build_demo.py runs/seance_A.json runs/seance_B.json runs/seance_C.json
"""

import base64
import json
import os
import sys

TEMPLATE = os.path.join("front", "hemicycle.html")
THREE = os.path.join("front", "lib", "three.module.min.js")
OUT = os.path.join("front", "demo.html")


def main():
    run_paths = sys.argv[1:] or [os.path.join("runs", "seance_VTANR5L17V4000.json")]
    runs = [json.load(open(p, encoding="utf-8")) for p in run_paths]
    html = open(TEMPLATE, encoding="utf-8").read()
    # toujours un TABLEAU de runs (même s'il n'y en a qu'un) : le JS pioche
    # dedans au hasard à chaque chargement de page.
    payload = json.dumps(runs, ensure_ascii=False).replace("</", "<\\/")
    html = html.replace("/*__RUN_DATA__*/null", payload, 1)
    # three.js inliné en data-URL dans l'importmap → demo.html s'ouvre en
    # DOUBLE-CLIC (file://), zéro serveur, zéro réseau. Wifi-proof.
    b64 = base64.b64encode(open(THREE, "rb").read()).decode()
    html = html.replace('"three":"./lib/three.module.min.js"',
                        f'"three":"data:text/javascript;base64,{b64}"', 1)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"{len(runs)} run(s) ({', '.join(run_paths)}) -> {OUT} "
          f"({os.path.getsize(OUT) // 1024} Ko, autonome)")


if __name__ == "__main__":
    main()
