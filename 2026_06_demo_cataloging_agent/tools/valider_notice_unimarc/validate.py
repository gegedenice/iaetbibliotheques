"""
Outil `valider_notice_unimarc` — validateur minimal de notices Unimarc/XML.

Contient trois choses :
  1. la logique de validation (fonction `valider_notice_unimarc`) ;
  2. le SCHÉMA de l'outil (ce que voit le modèle) ;
  3. un `dispatch` générique (ce que fait le harnais quand le modèle appelle l'outil).

Aucune dépendance hors bibliothèque standard.
"""
import re
import xml.etree.ElementTree as ET

# --- Configuration métier (à adapter à votre politique de catalogage) ---------
ZONES_OBLIGATOIRES = {"200": ["a"]}          # 200 $a : titre propre
ZONES_RECOMMANDEES = {"101": ["a"], "801": []}  # absence -> avertissement


# --- 1. Logique de validation -------------------------------------------------
def _localname(tag: str) -> str:
    """Retire un éventuel namespace ({ns}record -> record)."""
    return tag.split("}", 1)[-1]


def _isbn_valide(isbn: str) -> bool:
    s = re.sub(r"[\s-]", "", isbn).upper()
    if len(s) == 10:
        if not re.match(r"^\d{9}[\dX]$", s):
            return False
        total = sum((10 - i) * (10 if c == "X" else int(c)) for i, c in enumerate(s))
        return total % 11 == 0
    if len(s) == 13:
        if not s.isdigit():
            return False
        total = sum((1 if i % 2 == 0 else 3) * int(c) for i, c in enumerate(s))
        return total % 10 == 0
    return False


def _ppn_format_ok(ppn: str) -> bool:
    # Format ABES : 8 chiffres + 1 caractère de contrôle (chiffre ou X).
    return bool(re.match(r"^\d{8}[\dX]$", ppn))


def valider_notice_unimarc(notice_xml: str) -> dict:
    """Valide une notice Unimarc/XML.

    Retourne un dict : {valide: bool, erreurs: [str], avertissements: [str]}.
    Conçu pour être consommé par un LLM : messages courts et explicites.
    """
    erreurs, avertissements = [], []

    # 1. XML bien formé
    try:
        root = ET.fromstring(notice_xml)
    except ET.ParseError as e:
        return {"valide": False, "erreurs": [f"XML mal formé : {e}"], "avertissements": []}

    if _localname(root.tag) != "record":
        erreurs.append(f"Élément racine attendu <record>, trouvé <{_localname(root.tag)}>.")

    # 2. Indexer les zones (datafields) par tag
    datafields = {}
    for df in root:
        if _localname(df.tag) != "datafield":
            continue
        tag = df.get("tag", "")
        if not re.match(r"^\d{3}$", tag):
            erreurs.append(f"Tag de zone invalide : '{tag}' (3 chiffres attendus).")
        sousz = {sf.get("code"): (sf.text or "").strip()
                 for sf in df if _localname(sf.tag) == "subfield"}
        datafields.setdefault(tag, []).append(sousz)

    # 3. Zones obligatoires
    for tag, sous_obligatoires in ZONES_OBLIGATOIRES.items():
        if tag not in datafields:
            erreurs.append(f"Zone obligatoire {tag} absente.")
            continue
        for code in sous_obligatoires:
            if not any(code in occ and occ[code] for occ in datafields[tag]):
                erreurs.append(f"Sous-zone obligatoire {tag} ${code} absente ou vide.")

    # 4. Zones recommandées
    for tag in ZONES_RECOMMANDEES:
        if tag not in datafields:
            avertissements.append(f"Zone recommandée {tag} absente.")

    # 5. Contrôles spécifiques
    for occ in datafields.get("101", []):
        code = occ.get("a", "")
        if code and not re.match(r"^[a-z]{3}$", code):
            erreurs.append(f"101 $a : code langue '{code}' invalide (ex. 'fre').")
    for occ in datafields.get("010", []):
        isbn = occ.get("a", "")
        if isbn and not _isbn_valide(isbn):
            erreurs.append(f"010 $a : ISBN '{isbn}' invalide (longueur ou clé de contrôle).")
    for occ in datafields.get("700", []):
        ppn = occ.get("3", "")
        if ppn and not _ppn_format_ok(ppn):
            avertissements.append(f"700 $3 : PPN '{ppn}' n'a pas le format attendu (8 chiffres + clé).")

    return {"valide": not erreurs, "erreurs": erreurs, "avertissements": avertissements}


# --- 2. Schéma de l'outil (ce que reçoit le modèle) ---------------------------
SCHEMA = {
    "name": "valider_notice_unimarc",
    "description": (
        "Valide une notice Unimarc/XML : zones obligatoires présentes, codes langue, "
        "ISBN et format de PPN. Retourne {valide, erreurs, avertissements}."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "notice_xml": {"type": "string", "description": "La notice Unimarc au format XML"}
        },
        "required": ["notice_xml"],
    },
}

# Table de routage nom -> fonction. Le harnais y ajoute ses autres outils.
OUTILS = {"valider_notice_unimarc": valider_notice_unimarc}


# --- 3. Dispatch générique ----------------------------------------------------
def dispatch(nom: str, arguments: dict) -> dict:
    """Exécute l'outil demandé par le modèle et renvoie son résultat."""
    if nom not in OUTILS:
        return {"erreur": f"Outil inconnu : {nom}"}
    return OUTILS[nom](**arguments)


if __name__ == "__main__":
    import json
    notice_ok = ('<record>'
                 '<datafield tag="101"><subfield code="a">fre</subfield></datafield>'
                 '<datafield tag="200"><subfield code="a">La mer</subfield></datafield>'
                 '<datafield tag="801"><subfield code="b">SCD Sciences</subfield></datafield>'
                 '</record>')
    print(json.dumps(dispatch("valider_notice_unimarc", {"notice_xml": notice_ok}),
                      ensure_ascii=False, indent=2))