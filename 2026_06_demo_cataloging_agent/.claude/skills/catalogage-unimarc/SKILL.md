---
name: catalogage-unimarc
description: Produire une notice Unimarc/XML à partir d'une description d'ouvrage
  (titre, auteur, ISBN). À utiliser pour tout catalogage ou enrichissement de notice.
---

# Catalogage Unimarc

1. Identifier les éléments fournis (titre, mention d'auteur, ISBN, langue…).
2. Aligner l'auteur sur IdRef avec `resolve-authorities-idref` ; reporter le PPN en 700 $3.
3. Construire la notice dans cet ordre de zones :
   - 010 $a — ISBN
   - 101 $a — langue du texte
   - 200 $a — titre propre, $f — mention de responsabilité
   - 700 $a/$b — auteur (nom, prénom), $3 — PPN IdRef
   - 801 — source de catalogage
4. Valider avec `valider_notice_unimarc`. Corriger les erreurs signalées.
5. Présenter d'abord la notice brute, puis, si besoin, un bref commentaire.