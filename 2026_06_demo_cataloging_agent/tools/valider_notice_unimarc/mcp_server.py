#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ['fastmcp>=2.0']
# ///


"""
Serveur MCP exposant l'outil `valider_notice_unimarc`.
 
Le même validateur (logique métier) est ici publié via le protocole MCP : n'importe
quel agent compatible (Claude Code, etc.) peut alors le découvrir et l'appeler.
Lancement :  uv run mcp_server.py
"""

from validate import valider_notice_unimarc as _valider
from fastmcp import FastMCP

mcp = FastMCP("unimarc")
 
 
@mcp.tool()
def valider_notice_unimarc(notice_xml: str) -> dict:
    """Valide une notice Unimarc/XML.
 
    Le schéma exposé au modèle (nom, description, paramètres) est dérivé
    automatiquement par FastMCP à partir de cette signature et de la docstring.
 
    Args:
        notice_xml: la notice Unimarc au format XML.
    Returns:
        {valide: bool, erreurs: [str], avertissements: [str]}
    """
    return _valider(notice_xml)
 
 
if __name__ == "__main__":
    mcp.run()  # transport stdio par défaut
 