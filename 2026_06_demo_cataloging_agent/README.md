# Démo Assistant de catalogage Unimarc

> Post : https://iaetbibliotheques.fr/2026/06/focus-sur-lia-agentique

Agent Claude code qui assiste le catalogage courant d'une bibliothèque. À partir
de la description d'un ouvrage (titre, auteur, ISBN), il produit une notice
**Unimarc/XML** prête à être versée, après vérification d'absence dans le Sudoc
et alignement de l'auteur sur IdRef.

## Fonctionnement

Pour chaque ouvrage décrit, l'agent suit la procédure définie dans
[`AGENTS.md`](AGENTS.md) :

0. **Vérification préalable Sudoc (obligatoire).** Recherche par ISBN
   (`lookup_by_isbn`) ou, à défaut, par titre + auteur (`search_sudoc`).
   Si une notice existe déjà, l'agent s'arrête et signale le ou les PPN trouvés
   plutôt que de produire un doublon.
1. **Résolution de l'auteur** sur IdRef via la skill `resolve-authorities-idref` ;
   le PPN obtenu est reporté en `700 $3`. En cas de statut `not_found` ou ambigu,
   le `$3` reste vide et le cas est signalé — **aucun PPN n'est inventé.**
2. **Construction de la notice** avec la skill `catalogage-unimarc`.
3. **Validation** avec l'outil `valider_notice_unimarc`, puis correction des
   erreurs signalées.
4. **Présentation** de la notice brute d'abord, suivie d'un bref commentaire
   si nécessaire.

## Règles de catalogage

- Langue de catalogage : français (`101 $a fre`).
- En cas de doute (homonymie, donnée manquante), l'agent demande plutôt que de
  supposer.

## Composants

### Skills (`.claude/skills/`)

- **`catalogage-unimarc`** — construit la notice Unimarc/XML en suivant l'ordre
  des zones : `010` (ISBN), `101` (langue), `200` (titre, mention de
  responsabilité), `700` (auteur + PPN IdRef), `801` (source de catalogage).
- **`resolve-authorities-idref`** — résout une personne en PPN IdRef à partir de
  son nom et de tout indice de désambiguïsation (œuvres, domaine, affiliation,
  rôle, année). Retourne un JSON strict avec un niveau de confiance.

### Serveurs MCP (`.mcp.json`)

- **`sudoc`** — connecteur SRU interrogeant le catalogue collectif du Sudoc
  (ABES) : `lookup_by_isbn`, `lookup_by_ppn`, `search_sudoc`, `count_records`,
  `scan_index`…
- **`unimarc`** — validation des notices via `valider_notice_unimarc`
  (outil local, voir [`tools/valider_notice_unimarc/`](tools/valider_notice_unimarc/)).

## Prérequis

- [Claude Code](https://claude.com/claude-code)
- [`uv`](https://docs.astral.sh/uv/) — exécute les serveurs MCP `sudoc` et
  `unimarc` déclarés dans `.mcp.json`.

## Utilisation

Ouvrir le dossier avec Claude Code et décrire l'ouvrage à cataloguer,
par exemple :

> Catalogue cet ouvrage : *Le nom de la rose*, Umberto Eco, ISBN 9782253033134.

L'agent enchaîne alors la vérification Sudoc, la résolution de l'auteur, la
génération de la notice et sa validation.

## Structure du dépôt

```
.
├── AGENTS.md / CLAUDE.md          Instructions de l'agent (procédure, règles)
├── .mcp.json                      Déclaration des serveurs MCP (sudoc, unimarc)
├── settings.json                  Réglages du harnais Claude Code
├── .claude/skills/                Skills de catalogage et de résolution d'autorités
└── tools/valider_notice_unimarc/  Validateur Unimarc local (serveur MCP)
```
