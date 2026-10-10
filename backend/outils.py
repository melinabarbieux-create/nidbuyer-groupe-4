"""
Outils de l'agent NidBuyer.

Un outil est une fonction Python ordinaire. Le modele n'en voit que trois choses :
son NOM, sa DOCSTRING et ses PARAMETRES types. C'est son seul mode d'emploi :
une docstring vague = un outil mal utilise.

Regles :
- les calculs sont faits ici, en Python, jamais par le modele ;
- un outil renvoie des donnees simples (dict, list, nombres, texte) ;
- en cas d'entree invalide, lever ValueError avec un message que le modele peut comprendre.

Donnees : data/annonces_exemple.json et data/medianes_quartiers.csv sont des EXEMPLES
FICTIFS pour demarrer. En P1, chercher_biens passe par le RAG et les medianes viennent du DVF.
"""
import json
from pathlib import Path

from .marche import mediane_quartier as _mediane

ANNONCES_EXEMPLE = Path(__file__).resolve().parent.parent / "data" / "annonces_exemple.json"


def _annonces() -> list[dict]:
    return json.loads(ANNONCES_EXEMPLE.read_text(encoding="utf-8-sig"))


def chercher_biens(budget_max: float, quartier: str | None = None, surface_min: float | None = None,
                   mots_cles: str | None = None) -> list[dict]:
    """
    Cherche des biens a vendre a Toulon.

    budget_max : prix maximum en euros, frais d'agence inclus.
    quartier : nom du quartier (ex. "Mourillon"), ou None pour tous les quartiers.
    surface_min : surface minimale en m2, ou None.
    mots_cles : mots a retrouver dans la description (ex. "jardin ecole"), ou None.
    Retourne au plus 5 biens, les moins chers d'abord, avec id, type, surface, quartier, prix, description.
    """
    if budget_max <= 0:
        raise ValueError("budget_max doit etre positif, en euros")
    biens = [b for b in _annonces() if b["prix"] <= budget_max]
    if quartier:
        biens = [b for b in biens if b["quartier"].lower() == quartier.strip().lower()]
    if surface_min:
        biens = [b for b in biens if b["surface"] >= surface_min]
    if mots_cles:
        mots = [m.lower() for m in mots_cles.split() if len(m) >= 2]
        biens = [b for b in biens if any(m in b["description"].lower() for m in mots)]
    return sorted(biens, key=lambda b: b["prix"])[:5]


def ecart_au_marche(bien_id: str) -> dict:
    """
    Compare le prix au m2 d'un bien a la mediane des ventes DVF de son quartier.

    bien_id : identifiant d'un bien renvoye par chercher_biens (ex. "a03").
    Retourne prix (euros), surface (m2), quartier, prix_m2, mediane_quartier_m2,
    ecart_pct (negatif = moins cher que le marche).
    """
    bien = next((b for b in _annonces() if b["id"] == bien_id), None)
    if bien is None:
        raise ValueError(f"bien inconnu : {bien_id}. Utiliser un id renvoye par chercher_biens.")
    mediane = _mediane(bien["quartier"])
    prix_m2 = bien["prix"] / bien["surface"]
    resultat = {"bien_id": bien_id, "prix": bien["prix"], "surface": bien["surface"],
                "quartier": bien["quartier"], "prix_m2": round(prix_m2)}
    if mediane is None:
        return {**resultat, "mediane_quartier_m2": None, "ecart_pct": None}
    return {**resultat, "mediane_quartier_m2": round(mediane),
            "ecart_pct": round((prix_m2 - mediane) / mediane * 100, 1)}


def simuler_pret(montant: float, duree_ans: int, taux_annuel_pct: float) -> dict:
    """
    Mensualite d'un pret immobilier a taux fixe (hors assurance).

    montant : somme empruntee en euros (prix + frais - apport).
    duree_ans : duree du pret en annees (ex. 20 ou 25).
    taux_annuel_pct : taux nominal annuel en pourcentage (ex. 3.4 pour 3,4 %).
    Retourne mensualite, cout_total_credit.
    """
    if montant <= 0 or duree_ans <= 0 or taux_annuel_pct < 0:
        raise ValueError("montant et duree_ans positifs, taux_annuel_pct >= 0")
    n, t = duree_ans * 12, taux_annuel_pct / 100 / 12
    mensualite = montant / n if t == 0 else montant * t / (1 - (1 + t) ** -n)
    return {"mensualite": round(mensualite, 2), "cout_total_credit": round(mensualite * n - montant, 2)}
