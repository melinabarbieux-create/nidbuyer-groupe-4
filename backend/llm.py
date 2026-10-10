"""
Acces unique aux LLM. FOURNI PAR L'ENSEIGNANT : a utiliser, pas a reecrire.

Tout le reste du code appelle `generer()`. Le fournisseur se choisit par variable
d'environnement, sans toucher au code metier :

    LLM_PROVIDER=gemini   Gemini via une cle d'API AI Studio (P1, avant le module GCP) : defaut
    LLM_PROVIDER=vertex   Gemini via Vertex AI, sans cle (P2, deploye sur Cloud Run)
    LLM_PROVIDER=ollama   modele ouvert servi en local par Ollama (P2, version interne)
    LLM_PROVIDER=fake     reponses deterministes, sans reseau (tests, CI)

Variables :
    LLM_MODEL             ex. gemini-3.5-flash-lite (gemini), gemini-3.5-flash (vertex), qwen3:4b (ollama)
    LLM_MODEL_SECOURS     modele essaye si LLM_MODEL est sature (gemini, vertex)
    GEMINI_API_KEY        (gemini)
    GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION   (vertex)
    OLLAMA_URL            defaut http://localhost:11434 (ollama)

Les variables peuvent aussi etre mises dans un fichier .env a la racine du repo
(copie de .env.example) : il est lu automatiquement. Une variable deja exportee
dans le terminal reste prioritaire.

Exemple :
    from backend.llm import generer
    texte = generer("Resume ce bien : ...", system="Tu es NidBuyer.")
    fiche = generer(prompt, schema=FicheDecision)   # -> instance Pydantic validee
    res = executer_agent("Trouve un T3 sous 250 k€", outils=[chercher_biens, mediane_quartier])
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from typing import Literal, TypeVar, get_args, get_origin

import requests
from pydantic import BaseModel, ValidationError

try:  # .env a la racine du repo, s'il existe (n'ecrase pas les variables deja exportees)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

MENTION_IA = (
    "Réponse générée par une intelligence artificielle (NidBuyer). "
    "Elle ne remplace pas l'avis d'un professionnel."
)


class LLMError(RuntimeError):
    """Appel LLM impossible ou sortie inexploitable apres relance."""


def provider() -> str:
    return os.environ.get("LLM_PROVIDER", "gemini").lower()


def modele() -> str:
    defaut = {"gemini": "gemini-3.5-flash-lite", "vertex": "gemini-3.5-flash", "ollama": "qwen3:4b", "fake": "fake"}
    return os.environ.get("LLM_MODEL", defaut.get(provider(), ""))


# --- Journal des appels : sert en P3 pour comparer cout et latence -------------

JOURNAL: list[dict] = []


def _journaliser(debut: float, tokens_in: int | None, tokens_out: int | None, ok: bool) -> None:
    JOURNAL.append({
        "provider": provider(), "modele": modele(), "ok": ok,
        "latence_s": round(time.perf_counter() - debut, 3),
        "tokens_in": tokens_in, "tokens_out": tokens_out,
    })
    del JOURNAL[:-1000]  # garde les 1000 derniers appels


# --- Point d'entree unique ---------------------------------------------------

def generer(prompt: str, system: str | None = None, schema: type[T] | None = None,
            temperature: float = 0.2, max_essais: int = 2,
            images: list[bytes] | None = None) -> str | T:
    """
    Envoie un prompt au LLM configure.

    images : contenu brut de photos JPEG/PNG (modele multimodal requis, ex. pour vision/).
    Sans schema : retourne le texte.
    Avec schema (classe Pydantic) : retourne une instance validee. En cas de sortie
    non conforme, relance une fois en renvoyant l'erreur au modele, puis leve LLMError.
    """
    appel = {"gemini": _vertex, "vertex": _vertex, "ollama": _ollama, "fake": _fake}.get(provider())
    if appel is None:
        raise LLMError(f"LLM_PROVIDER inconnu : {provider()}")

    derniere_erreur = None
    for essai in range(max_essais):
        p = prompt if essai == 0 else (
            f"{prompt}\n\nTa reponse precedente etait invalide ({derniere_erreur}). "
            "Reponds uniquement avec un JSON conforme au schema."
        )
        debut = time.perf_counter()
        try:
            texte, t_in, t_out = appel(p, system, schema, temperature, images or [])
        except Exception as e:
            _journaliser(debut, None, None, ok=False)
            raise LLMError(f"Appel {provider()} en echec : {e}") from e
        if schema is None:
            _journaliser(debut, t_in, t_out, ok=True)
            return texte
        try:
            objet = schema.model_validate_json(_extraire_json(texte))
            _journaliser(debut, t_in, t_out, ok=True)
            return objet
        except (ValidationError, ValueError) as e:
            _journaliser(debut, t_in, t_out, ok=False)
            derniere_erreur = str(e)[:300]
            logger.warning("Sortie LLM non conforme (essai %d) : %s", essai + 1, derniere_erreur)
    raise LLMError(f"Sortie non conforme au schema apres {max_essais} essais : {derniere_erreur}")


def _extraire_json(texte: str) -> str:
    """Tolere un bloc ```json ... ``` ou du texte parasite autour de l'objet JSON."""
    debut, fin = texte.find("{"), texte.rfind("}")
    if debut == -1 or fin == -1:
        raise ValueError("aucun objet JSON dans la reponse")
    return texte[debut:fin + 1]


# --- Fournisseurs ------------------------------------------------------------

def _mime(img: bytes) -> str:
    return "image/png" if img[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"


_vertex_client = None


def _client_google():
    """Client Gemini : cle d'API (gemini) ou identite GCP (vertex)."""
    global _vertex_client
    from google import genai
    if _vertex_client is None:
        if provider() == "gemini":
            cle = os.environ.get("GEMINI_API_KEY", "").strip()
            if not cle or cle == "collez-votre-cle-ici":
                raise LLMError("GEMINI_API_KEY absente : collez votre cle AI Studio dans le fichier .env "
                               "(cp .env.example .env), ou LLM_PROVIDER=fake pour travailler sans cle.")
            _vertex_client = genai.Client(api_key=cle)
        else:
            _vertex_client = genai.Client(
                vertexai=True,
                project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "europe-west1"),
            )
    return _vertex_client


def _delai_quota(erreur) -> float | None:
    """
    Sur un 429, Gemini dit combien attendre (« retryDelay »: « 23s »). Retourne ce delai en secondes,
    ou None si c'est le quota du JOUR qui est epuise (attendre ne sert alors a rien).
    """
    texte = str(erreur)
    if "PerDay" in texte or "per_day" in texte.lower():
        return None
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", texte) \
        or re.search(r"retry in (\d+(?:\.\d+)?)\s*s", texte, re.IGNORECASE)
    return float(m.group(1)) if m else 20.0


def _generate_google(contents, config):
    """
    Appel Gemini tolerant aux pannes passageres. Retourne (reponse, modele utilise).
    - 429 quota PAR MINUTE : on attend le delai indique par Gemini, puis on relance (3 fois max).
    - 429 quota du JOUR : inutile d'attendre, on passe au modele suivant.
    - 503 sature ou coupure reseau : on reessaie une fois apres 2 s, puis LLM_MODEL_SECOURS.
    """
    import httpx
    from google.genai import errors

    client = _client_google()
    candidats = [modele(), modele()]
    secours = os.environ.get("LLM_MODEL_SECOURS", "").strip()
    if secours and secours != modele():
        candidats.append(secours)
    attente_max = float(os.environ.get("LLM_ATTENTE_QUOTA_MAX", "65"))
    derniere = None
    for i, nom in enumerate(candidats):
        if i == 1:
            time.sleep(2)
        attentes = 0
        while True:
            try:
                return client.models.generate_content(model=nom, contents=contents, config=config), nom
            except (errors.ServerError, httpx.TransportError) as e:
                derniere = e
                break
            except errors.ClientError as e:
                if getattr(e, "code", None) != 429:
                    raise
                derniere = e
                delai = _delai_quota(e)
                if delai is None or attentes >= 3 or delai > attente_max:
                    break
                attentes += 1
                logger.warning("Quota par minute atteint sur %s : attente %.0f s (essai %d/3)", nom, delai + 1, attentes)
                time.sleep(delai + 1)
        logger.warning("Modele %s indisponible (%s), essai du suivant", nom, str(derniere)[:120])
    jour = derniere is not None and _delai_quota(derniere) is None
    conseil = ("Quota du jour epuise sur cette cle : utilisez la cle d'un autre membre du groupe."
               if jour else "Reessayez dans une minute, ou changez LLM_MODEL dans .env (ex. gemini-flash-lite-latest).")
    raise LLMError(
        f"Gemini ne repond pas ({', '.join(dict.fromkeys(candidats))}) : {str(derniere)[:200]}. {conseil}"
    ) from derniere


def _vertex(prompt, system, schema, temperature, images):
    from google.genai import types

    _client_google()
    config = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
    if schema is not None:
        config.response_mime_type = "application/json"
        config.response_schema = schema
    contents = [types.Part.from_bytes(data=img, mime_type=_mime(img)) for img in images] + [prompt]
    r, _ = _generate_google(contents, config)
    u = r.usage_metadata
    return r.text or "", getattr(u, "prompt_token_count", None), getattr(u, "candidates_token_count", None)


def _ollama(prompt, system, schema, temperature, images):
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    if images:
        messages[-1]["images"] = [base64.b64encode(img).decode() for img in images]
    body = {
        "model": modele(), "messages": messages, "stream": False,
        "options": {"temperature": temperature},
        # Les modeles "a raisonnement" (Qwen3...) ajoutent sinon du texte de reflexion
        # qui casse le JSON. On le desactive.
        "think": False,
    }
    if schema is not None:
        body["format"] = schema.model_json_schema()
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    r = requests.post(f"{url}/api/chat", json=body, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"], data.get("prompt_eval_count"), data.get("eval_count")


def _fake(prompt, system, schema, temperature, images):
    """Deterministe et hors ligne. Remplit un schema avec des valeurs par defaut plausibles."""
    if schema is None:
        return f"[fake] {prompt[:120]}", len(prompt) // 4, 20
    return json.dumps(_exemple(schema)), len(prompt) // 4, 50


def _exemple(schema: type[BaseModel]) -> dict:
    valeurs = {}
    for nom, champ in schema.model_fields.items():
        valeurs[nom] = _valeur(champ.annotation)
    return valeurs


def _valeur(ann):
    origine, args = get_origin(ann), get_args(ann)
    if origine is Literal:
        return args[0]
    if origine is list:
        return []
    if args:  # Optional[X] / X | None : premier type non None
        return _valeur(next(a for a in args if a is not type(None)))
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return _exemple(ann)
    return {int: 1, float: 1.0, bool: False}.get(ann, "fake")


# --- Agent : le modele choisit et appelle des outils (fonctions Python) -------------

def executer_agent(question: str, outils: list, system: str | None = None,
                   max_tours: int = 6, temperature: float = 0.0) -> dict:
    """
    Boucle agentique : le modele decide quels outils appeler, avec quels arguments,
    lit les resultats, recommence, puis repond.

    outils : fonctions Python avec annotations de type et docstring. Le modele ne voit
    QUE le nom, la docstring et les parametres : c'est son unique mode d'emploi.

    Retourne :
        {
          "reponse": str,                      # texte final
          "appels": [{"outil": str, "arguments": dict, "resultat": ..., "erreur": str | None}, ...],
          "tours": int,                         # nombre d'allers-retours avec le modele
          "arret": "reponse" | "max_tours",
          "tokens_par_tour": [(entree, sortie), ...],   # un couple par appel au modele
          "tokens_entree": int, "tokens_sortie": int    # totaux sur toute la boucle
        }
    Une exception levee par un outil n'arrete pas l'agent : l'erreur lui est renvoyee.
    Chaque tour renvoie tout l'historique au modele : tokens_entree croit plus vite que les tours.
    """
    par_nom = {f.__name__: f for f in outils}
    boucle = {"gemini": _agent_google, "vertex": _agent_google,
              "ollama": _agent_ollama, "fake": _agent_fake}.get(provider())
    if boucle is None:
        raise LLMError(f"LLM_PROVIDER inconnu : {provider()}")
    avant = len(JOURNAL)
    resultat = boucle(question, par_nom, system, max_tours, temperature)
    # Les appels de cette boucle sont les dernieres entrees du journal
    # (approximatif seulement si le journal a atteint sa limite de 1000 entrees pendant la boucle).
    nouveaux = JOURNAL[avant:] if len(JOURNAL) > avant else []
    resultat["tokens_par_tour"] = [(j["tokens_in"] or 0, j["tokens_out"] or 0) for j in nouveaux]
    resultat["tokens_entree"] = sum(e for e, _ in resultat["tokens_par_tour"])
    resultat["tokens_sortie"] = sum(s for _, s in resultat["tokens_par_tour"])
    return resultat


def _executer_outil(par_nom: dict, nom: str, arguments: dict) -> dict:
    appel = {"outil": nom, "arguments": arguments, "resultat": None, "erreur": None}
    if nom not in par_nom:
        appel["erreur"] = f"outil inconnu : {nom}"
        return appel
    try:
        appel["resultat"] = par_nom[nom](**arguments)
    except Exception as e:  # l'agent doit pouvoir se corriger
        appel["erreur"] = f"{type(e).__name__}: {e}"
    return appel


def _pour_modele(appel: dict):
    """Resultat serialisable renvoye au modele (tronque : le contexte n'est pas infini)."""
    contenu = {"erreur": appel["erreur"]} if appel["erreur"] else {"resultat": appel["resultat"]}
    texte = json.dumps(contenu, ensure_ascii=False, default=str)
    return json.loads(texte) if len(texte) <= 8000 else {"resultat_tronque": texte[:8000]}


def _agent_google(question, par_nom, system, max_tours, temperature):
    from google.genai import types
    config = types.GenerateContentConfig(
        system_instruction=system, temperature=temperature, tools=list(par_nom.values()),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]
    appels = []
    for tour in range(1, max_tours + 1):
        debut = time.perf_counter()
        r, _ = _generate_google(contents, config)
        u = r.usage_metadata
        _journaliser(debut, getattr(u, "prompt_token_count", None), getattr(u, "candidates_token_count", None), ok=True)
        if not r.function_calls:
            return {"reponse": r.text or "", "appels": appels, "tours": tour, "arret": "reponse"}
        contents.append(r.candidates[0].content)
        reponses = []
        for fc in r.function_calls:
            if len(appels) >= 6:  # <-- Sécurité anti-abus de coût
                break
            appel = _executer_outil(par_nom, fc.name, dict(fc.args or {}))
            appels.append(appel)
            reponses.append(types.Part.from_function_response(name=fc.name, response=_pour_modele(appel)))
        contents.append(types.Content(role="user", parts=reponses))
    return {"reponse": "", "appels": appels, "tours": max_tours, "arret": "max_tours"}


def _schema_outil(f) -> dict:
    """Description JSON d'une fonction Python, au format outils d'Ollama / OpenAI."""
    import inspect
    types_json = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}
    props, requis = {}, []
    for nom, p in inspect.signature(f).parameters.items():
        ann = p.annotation
        base = next((a for a in get_args(ann) if a is not type(None)), ann) if get_args(ann) else ann
        base = get_origin(base) or base
        props[nom] = {"type": types_json.get(base, "string")}
        if p.default is inspect.Parameter.empty:
            requis.append(nom)
    return {"type": "function", "function": {
        "name": f.__name__, "description": inspect.getdoc(f) or "",
        "parameters": {"type": "object", "properties": props, "required": requis}}}


def _agent_ollama(question, par_nom, system, max_tours, temperature):
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    outils = [_schema_outil(f) for f in par_nom.values()]
    appels = []
    for tour in range(1, max_tours + 1):
        debut = time.perf_counter()
        r = requests.post(f"{url}/api/chat", timeout=300, json={
            "model": modele(), "messages": messages, "tools": outils, "stream": False,
            "think": False, "options": {"temperature": temperature}})
        r.raise_for_status()
        data = r.json()
        _journaliser(debut, data.get("prompt_eval_count"), data.get("eval_count"), ok=True)
        msg = data["message"]
        if not msg.get("tool_calls"):
            return {"reponse": msg.get("content", ""), "appels": appels, "tours": tour, "arret": "reponse"}
        messages.append(msg)
        for tc in msg["tool_calls"]:
            fn = tc["function"]
            args = fn.get("arguments") or {}
            appel = _executer_outil(par_nom, fn["name"], json.loads(args) if isinstance(args, str) else args)
            appels.append(appel)
            messages.append({"role": "tool", "tool_name": fn["name"],
                             "content": json.dumps(_pour_modele(appel), ensure_ascii=False)})
    return {"reponse": "", "appels": appels, "tours": max_tours, "arret": "max_tours"}


def _agent_fake(question, par_nom, system, max_tours, temperature):
    """Hors ligne : appelle chaque outil une fois avec des arguments d'exemple, puis repond."""
    import inspect
    appels = []
    for f in list(par_nom.values())[:max_tours]:
        args = {n: (p.default if p.default is not inspect.Parameter.empty else _valeur(p.annotation))
                for n, p in inspect.signature(f).parameters.items()}
        appels.append(_executer_outil(par_nom, f.__name__, args))
    _journaliser(time.perf_counter(), len(question) // 4, 20, ok=True)
    return {"reponse": f"[fake] {question[:120]}", "appels": appels, "tours": len(appels) + 1, "arret": "reponse"}
