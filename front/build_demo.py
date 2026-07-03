"""
Injecte un run de séance (runs/seance_*.json) dans le template hemicycle.html
→ front/demo.html autonome (ouvrable en double-clic, aucun serveur requis).

Usage (racine) : python front/build_demo.py runs/seance_VTANR5L17V4000.json
"""

import base64
import json
import os
import sys

TEMPLATE = os.path.join("front", "hemicycle.html")
THREE = os.path.join("front", "lib", "three.module.min.js")
OUT = os.path.join("front", "demo.html")


def main():
    run_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        "runs", "seance_VTANR5L17V4000.json")
    run = json.load(open(run_path, encoding="utf-8"))
    html = open(TEMPLATE, encoding="utf-8").read()
    payload = json.dumps(run, ensure_ascii=False).replace("</", "<\\/")
    html = html.replace("/*__RUN_DATA__*/null", payload, 1)
    # three.js inliné en data-URL dans l'importmap → demo.html s'ouvre en
    # DOUBLE-CLIC (file://), zéro serveur, zéro réseau. Wifi-proof.
    b64 = base64.b64encode(open(THREE, "rb").read()).decode()
    html = html.replace('"three":"./lib/three.module.min.js"',
                        f'"three":"data:text/javascript;base64,{b64}"', 1)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"{run_path} -> {OUT} ({os.path.getsize(OUT) // 1024} Ko, autonome)")


if __name__ == "__main__":
    main()
