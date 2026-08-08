"""Synthetic "fixture" corpus + a deterministic fake embedding model.

Purpose: let `python -m benchmark run` (the default, CI-safe invocation) work
instantly, offline, with no dependency on the real production Box index or on
downloading BAAI/bge-m3 - exactly the same trade-off the existing
`tests/test_search_relevance.py` fixture already makes (this module
intentionally mirrors it rather than importing from `tests/`, so the
benchmark package has zero dependency on the test suite).

IMPORTANT LIMITATION (same one documented in test_search_relevance.py): this
fake embedding cannot validate the semantic quality of the real BAAI/bge-m3
model, only that the pipeline *mechanics* (RRF merge, cosine rerank, dedup,
diversify) behave correctly. Real quality measurement must run with
`--environment production`, which uses the real embedding model against the
real index (see benchmark/environment.py).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

DIMENSIONS = 24
TOKEN_WEIGHT = 0.6

CATEGORY_SIGNALS = {
    "beton": ["beton", "krychl", "betonárny", "recept"],
    "vyztuz": ["výztuž", "armatur", "kladečsk", "kotevní", "prut"],
    "hydroizolace": ["hydroizolac", "izolace proti", "vlhkost", "fólie"],
    "pentaflex": ["pentaflex", "těsnicí pás", "pracovní spár", "spáry"],
    "zakladova_deska": ["základ", "desk", "geodetick", "bednění", "osazení"],
    "zmenove_listy": ["změnov", "vícepráce", "dodatek rozsahu prací", "registr změn"],
    "sod": ["smlouva o dílo", "smluvní pokut", "dodatek smlouvy"],
    "bozp": ["bozp", "bezpečnost práce", "školení pracovník", "proškolení"],
    "kontrolni_dny": ["kontrolní den", "kontrolní dny", "zápis z jednání", "harmonogram jednání"],
    "predani_dila": ["předání", "předávací", "soupis dokladů"],
}

STOPWORDS = {"a", "na", "v", "o", "k", "s", "z", "do", "po", "pro", "je", "se", "ze", "za", "ve"}

DOCUMENTS = {
    "kniha_betonu.txt": "Kniha betonů obsahuje záznamy o všech dodávkách čerstvého betonu, datu betonáže a použitém receptu.",
    "dodaci_listy_betonu.txt": "Dodací listy betonu dokládají množství, třídu betonu a čas dodání na stavbu.",
    "protokoly_zkousek_betonu.txt": "Protokoly zkoušek betonu obsahují výsledky tlakových zkoušek krychlí ve stáří 7 a 28 dní.",
    "kladecke_vykresy_vyztuze.txt": "Kladečské výkresy výztuže specifikují průměry prutů, rozteče a kotevní délky.",
    "kontrola_vyztuze_pred_betonazi.txt": "Kontrola výztuže před betonáží ověřuje krytí, počet prutů a čistotu armatury.",
    "navrh_hydroizolace_zakladu.txt": "Návrh hydroizolace základů řeší ochranu proti zemní vlhkosti a tlakové vodě.",
    "montaz_hydroizolacni_folie.txt": "Montáž hydroizolační fólie musí být provedena bez proražení a s přesahem spojů.",
    "pentaflex_technicky_list.txt": "Technický list těsnicího pásu Pentaflex popisuje způsob osazení do pracovní spáry základové desky.",
    "pentaflex_montazni_postup.txt": "Montážní postup Pentaflexu vyžaduje očištění pracovní spáry a kontrolu přítlaku pásu.",
    "geodeticke_zamereni_desky.txt": "Geodetické zaměření základové desky ověřuje výškové a polohové osazení bednění.",
    "stavebni_denik_zaklady.txt": "Stavební deník eviduje postup prací na základové desce, počasí a přítomné pracovníky.",
    "zmenovy_list_01.txt": "Změnový list číslo 1 popisuje úpravu tloušťky základové desky na základě geologického průzkumu.",
    "registr_zmen_projektu.txt": "Registr změn projektu eviduje schválené změnové listy a jejich dopad na rozpočet.",
    "smlouva_o_dilo_hlavni.txt": "Hlavní smlouva o dílo definuje rozsah prací, termíny a smluvní pokuty zhotovitele.",
    "dodatek_smlouvy_o_dilo.txt": "Dodatek smlouvy o dílo upravuje termín dokončení a cenu víceprací.",
    "plan_bozp_staveniste.txt": "Plán BOZP staveniště stanovuje bezpečnostní opatření a školení pracovníků.",
    "zaznam_o_skoleni_bozp.txt": "Záznam o školení BOZP potvrzuje proškolení pracovníků před zahájením prací.",
    "zapis_kontrolniho_dne_05.txt": "Zápis z kontrolního dne číslo 5 shrnuje postup výstavby a úkoly pro zhotovitele.",
    "harmonogram_kontrolnich_dnu.txt": "Harmonogram kontrolních dnů stanovuje pravidelné termíny jednání investora a zhotovitele.",
    "predavaci_protokol_zakladove_desky.txt": "Předávací protokol základové desky dokumentuje předání investorovi včetně příloh.",
    "soupis_dokladu_k_predani.txt": "Soupis dokladů k předání díla vyžaduje knihu betonů, protokoly zkoušek a geodetické zaměření.",
}


def _strip_diacritics(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _hashed_vector(seed: str) -> list[float]:
    digest = hashlib.sha256(seed.encode()).digest()
    return [(b / 127.5 - 1.0) for b in digest[:DIMENSIONS]]


class FakeCategoryEmbeddings:
    """Deterministic stand-in for BAAI/bge-m3 - see module docstring."""

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            folded = _strip_diacritics(text.casefold())
            vector = [0.0] * DIMENSIONS
            for category, signals in CATEGORY_SIGNALS.items():
                if any(signal in folded for signal in signals):
                    category_vector = _hashed_vector(category)
                    vector = [a + b for a, b in zip(vector, category_vector)]
            tokens = [t[:5] for t in re.findall(r"[a-z0-9]+", folded) if len(t) > 2 and t not in STOPWORDS]
            for stem in tokens:
                token_vector = _hashed_vector("tok:" + stem)
                vector = [a + TOKEN_WEIGHT * b for a, b in zip(vector, token_vector)]
            norm = sum(v * v for v in vector) ** 0.5 or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


def write_fixture_corpus(root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, text in DOCUMENTS.items():
        (root / name).write_text(text, encoding="utf-8")
    return root
