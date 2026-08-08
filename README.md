# AI Search v1.1.0

## Overview

AI Search je lokální aplikace pro hybridní vyhledávání a otázky nad českou
dokumentací. Je zaměřená na macOS a pracuje se soubory dostupnými v lokálních
složkách, například v synchronizované složce Box Drive. Box API ani cloudové
e-mailové API nepoužívá.

Vyhledávání kombinuje fulltext SQLite FTS5 a vektorové vyhledávání v LanceDB.
Výsledky se slučují, řadí podle relevance, deduplikují a mohou sloužit jako
citované podklady pro odpověď generovanou lokálním modelem přes Ollamu.

## Features

- hybridní fulltextové a vektorové vyhledávání,
- hledání dokumentů i odpovědi na otázky,
- odpovědi s citacemi a indikací jistoty,
- FTS query expansion pro českou stavební terminologii,
- OCR skenovaných PDF a obrazových souborů,
- strukturovaná extrakce XLS a XLSX,
- lokální import EML a MSG,
- samostatné inkrementální indexy pro dokumenty, e-maily a poznámky,
- filtry podle zdroje, projektu, složky, typu souboru, autora a data,
- diagnostika indexu a retrieval regression benchmark.

## Architecture

Projekt používá následující ověřené komponenty:

- `app.py` — Streamlit uživatelské rozhraní,
- `ui_services.py` — aplikační vrstva, nastavení, práce se zdroji a orchestrace
  vyhledávání,
- `ai_search.py` — indexace, chunking, SQLite FTS5, LanceDB retrieval, fusion,
  reranking a generování odpovědí,
- `document_extractors.py` a `parsing_worker.py` — extrakce a izolované parsování
  dokumentů,
- `query_expansion.py` — konzervativní rozšíření FTS dotazů,
- `diagnostics.py` — kontrola konzistence a diagnostické exporty,
- `benchmark/` — benchmark a retrieval regression suite,
- Ollama — lokální HTTP služba pro generování odpovědí.

Zjednodušený retrieval tok:

```text
dotaz
  ├─ SQLite FTS5 / BM25
  └─ LanceDB vector search
        ↓
      RRF fusion a semantic scoring
        ↓
      agregace evidence, deduplikace a diverzifikace
        ↓
      výsledky nebo lokální Ollama odpověď s citacemi
```

## Requirements

| Požadavek | Ověřený stav projektu |
|---|---|
| Operační systém | macOS; minimální verze není specifikována |
| Python | release byl ověřen na Pythonu 3.12.12; minimální verze není specifikována |
| Ollama | nutná pro AI odpovědi; minimální verze není specifikována |
| Poppler | kód očekává `pdfinfo`, `pdftotext` a `pdftoppm` v `/opt/homebrew/bin` |
| Tesseract | kód očekává `/opt/homebrew/bin/tesseract` a jazyky `ces+eng` |
| Xcode Command Line Tools | potřeba pouze pro sestavení nativního macOS launcheru přes `/usr/bin/clang` |

Python závislosti jsou uvedeny v `requirements.txt`. Projekt nemá deklarovanou
minimální verzi Pythonu v `pyproject.toml`.

## Installation

```bash
git clone https://github.com/varvaro/AI-Search.git
cd AI-Search
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Před spuštěním musí být samostatně dostupné Ollama, Poppler a Tesseract.
Projekt neobsahuje automatickou instalaci systémových nástrojů.

Ollama musí mít lokálně dostupné modely používané aplikací:

```text
qwen3:8b
qwen3:14b
```

Repozitář jejich instalaci ani stažení automaticky neprovádí.

## Running

Spusťte Ollamu:

```bash
ollama serve
```

V jiném terminálu spusťte Streamlit aplikaci:

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

Při prvním použití se přes Sentence Transformers stáhne embeddingový model
`BAAI/bge-m3`. Doba stažení a indexace závisí na připojení a velikosti zdrojové
složky.

### First indexing

1. V levém panelu vyberte lokální projektovou složku.
2. Stiskněte **Aktualizovat index**.
3. Průběh a případné chyby sledujte v aplikaci nebo v diagnostice.

Exportované e-maily a lokální poznámky lze přidat v **Nastavení**. Každý zdroj
používá samostatný SQLite a LanceDB index; výsledky se při hledání společně
seřadí.

## Models

### Active

| Účel | Model |
|---|---|
| Embedding | `BAAI/bge-m3` |
| Výchozí Ollama LLM | `qwen3:8b` |
| Hloubková analýza | `qwen3:14b` |

Produkční UI používá FTS-only query expansion (`QUERY_EXPANSION_MODE = "fts"`).

### Optional or experimental

- `BAAI/bge-reranker-v2-m3` je implementovaný pro volitelnou retrieval strategii
  `union_ce`; standardní UI cesta tuto strategii nezapíná.
- `gemma4` existuje jako konfigurační konstanta, ale aktivní použití tohoto
  modelu nebylo v projektu ověřeno.

## Data and privacy

- Zdrojové dokumenty zůstávají v lokálních složkách a aplikace je při indexaci
  čte.
- Runtime data nejsou součástí Git repozitáře.
- SQLite a LanceDB indexy obsahují metadata, extrahovaný text a embeddingy; je
  proto nutné s nimi zacházet jako s citlivými daty.
- Ollama endpoint je lokální: `http://127.0.0.1:11434`.
- Embeddingové modely se při prvním použití stahují z externího modelového
  registru a následně se používají z lokální cache.
- Benchmark a diagnostické výstupy mohou obsahovat názvy dokumentů nebo lokální
  cesty a nejsou určené k automatickému commitování.

Výchozí runtime umístění:

```text
~/Library/Application Support/AI Search/
├── database/
├── lance/
├── cache/
├── logs/
└── state/
```

Umístění lze přesměrovat environment variable:

```bash
export AI_SEARCH_HOME="/path/to/local/runtime"
```

## Supported formats

Aktivní indexační whitelist obsahuje:

```text
PDF, DOC, DOCX, XLS, XLSX, TXT, MD, CSV, RTF,
EML, MSG, PNG, JPG, JPEG, TIF, TIFF, BMP, GIF, WEBP
```

Kvalita extrakce závisí na typu dokumentu, dostupnosti systémových nástrojů a
kvalitě zdrojového souboru.

## macOS application

`Spustit AI Search.command` spouští Streamlit z existujícího checkoutu projektu.
Aktuální launcher obsahuje cestu specifickou pro původní lokální instalaci a
není přenositelný na jiný Mac bez úpravy.

`Vytvořit aplikaci pro Dock.command` sestaví `AI Search.app` s nativním oknem
WebKit. Výsledný bundle není samostatná distribuovaná aplikace:

- závisí na checkoutu projektu,
- vyžaduje existující `.venv`,
- uvnitř ukládá cestu k projektu,
- nativní launcher může spustit `ollama serve` a lokální Streamlit,
- provozní log zapisuje do `~/Library/Logs/AI Search/`.

Build skript navíc vyžaduje lokální, neverzovaný soubor
`macos/project-path.txt`. Čistý clone proto není bez lokální konfigurace
připravený k sestavení `.app`.

## Testing

Ověřený příkaz:

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```

Ověřený stav release `v1.1.0`:

```text
461 passed
```

Přímé spuštění `pytest -q` v aktuální konfiguraci selhává při importu root
modulů; používejte modulovou variantu výše.

## Benchmark

Benchmark je implementovaný v adresáři `benchmark/`. Podrobnosti o datasetech,
metrikách, produkčním prostředí a porovnávání běhů jsou v
[`benchmark/README.md`](benchmark/README.md).

Offline fixture benchmark:

```bash
source .venv/bin/activate
python -m benchmark run
```

Benchmark nad existujícím produkčním indexem:

```bash
python -m benchmark run --environment production
```

Retrieval regression suite:

```bash
python -m benchmark.run_retrieval_regression
```

Produkční benchmark vyžaduje již vytvořený lokální index a načítá skutečný
embeddingový model. Hosted CI workflow není v projektu implementované.

## Troubleshooting

- **Ollama není dostupná:** spusťte `ollama serve`. Vyhledávání zůstane funkční,
  ale AI odpověď nebude dostupná.
- **Ollama model chybí:** ověřte, že je příslušný model lokálně dostupný pod
  názvem z části [Models](#models).
- **Složka neexistuje:** ověřte, že je zdrojová složka lokálně dostupná a že má
  aplikace oprávnění ji číst.
- **PDF nebo OCR nefunguje:** ověřte dostupnost Poppleru, Tesseractu a jazyků
  `ces+eng` v cestách uvedených v [Requirements](#requirements).
- **Soubor nelze otevřít:** mohl být přesunut nebo odstraněn; aktualizujte index
  a zkontrolujte oprávnění macOS.
- **Prázdné výsledky:** zrušte filtry a ověřte stav indexu v levém panelu nebo v
  diagnostice.
- **Testy hlásí `ModuleNotFoundError`:** použijte
  `python -m pytest tests/ -q`, ne přímo `pytest -q`.

## Known limitations

- Projekt je závislý na macOS nástrojích a pevných Homebrew cestách.
- Minimální verze macOS a Pythonu není specifikovaná.
- Nativní `.app` není samostatný distribuční balíček.
- Hosted CI není implementované.
- Diagnostika stále používá interní označení `1.0.0-rc1`, které neodpovídá Git
  release `v1.1.0`.

## License

Projekt aktuálně neobsahuje soubor `LICENSE`; podmínky použití a distribuce
nejsou specifikované.
