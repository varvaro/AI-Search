# UX/UI audit AI Search – návrh redesignu v1.1

**Role:** senior product designer + senior frontend engineer, čistě auditní režim.
**Status:** žádný soubor v repozitáři nebyl změněn. Tento dokument je jediný výstup.
**Rozsah:** pouze uživatelská vrstva (`app.py`, `ui_services.py`, `diagnostics.py` – zobrazovací část). Retrieval, chunking, embeddingy, indexace a `ai_search.py`'s scoring/reranking se nemění. Jedna navržená položka (viz Fáze UI-2, bod 2) se dotýká `ai_search.py`, ale pouze **přidáním** klíče do návratového dict `answer()` – je to explicitně vyznačeno a odděleno od bezpečných změn.

---

## 1. Současný stav

### 1.1 Architektura UI

Neexistuje `ui_components.py` ani `.streamlit/config.toml`. UI vrstva je rozdělena do dvou souborů:

| Soubor | Role | Streamlit? |
|---|---|---|
| `app.py` (184 řádků) | layout, routing přes `st.session_state.page`, veškeré render funkce, jediný `<style>` blok | ano |
| `ui_services.py` (563 řádků) | čistá business logika – filtrování, dedup, diversifikace, agregace skóre, metadata, náhledy | **ne**, nulová závislost na `streamlit` |

Toto oddělení je z hlediska rizika velmi příznivé: `ui_services.py` je testovatelný bez UI a `app.py` z něj pouze čte. Znamená to ale i to, že **veškerý vizuální kód žije v jednom monolitickém `app.py`** – žádné komponenty, žádné šablony, žádný layout systém.

**Kde vzniká layout:**
- `st.set_page_config(layout="wide")` + `.stApp{max-width:1720px}` (řádek 12–14) – jediné globální omezení šířky.
- 4 „stránky" řízené stringem v `st.session_state.page`: `search` (výchozí), `settings`, `history`, `diagnostics` – žádné `st.navigation`/`st.Page`, čistě `if/elif` řetězec (řádky 101–183).
- Sidebar (`with st.sidebar:`, řádky 66–99) je vykreslován **na všech 4 stránkách stejně** – není kontextový.

**Kde jsou komponenty:** žádné znovupoužitelné komponenty v Streamlit smyslu. Existují 4 pomocné funkce v `app.py`: `select_folder`, `render_preview`, `render_result`, `highlight` – všechny volají přímo `st.*` a HTML string building, ne skutečné komponenty s vlastním stavem.

**Kde se definují styly:** výhradně jeden `st.markdown("""<style>...</style>""", unsafe_allow_html=True)` blok na řádku 13–15 (jeden dlouhý řádek, cca 900 znaků CSS bez zalomení) + tři samostatné inline `style=` atributy v diagnostické stránce (řádek 103–104) a v `diagnostics.py`'s HTML exportu. Barvy jsou z části natvrdo (`#3b82f6`, `#fde68a`, `#dcfce7`, `#fee2e2`), z části vážou na Streamlit CSS proměnné (`var(--text-color)`, `var(--secondary-background-color)`) – nekonzistentní přístup, žádný jednotný token systém.

**Co lze upravit pouze v UI vrstvě bez rizika pro backend:**
Cokoli v `app.py`, co pouze čte existující návratové hodnoty z `ui_services.py`/`ai_search.py` a nezasahuje do parametrů volání `search_all()`, `ai_search.search()`, `ai_search.answer()`, `index_source()`/`ai_search.sync()`. Konkrétně bezpečné jsou: CSS/`.streamlit/config.toml`, texty a nadpisy, pořadí a seskupení `st.*` prvků, ikony, viditelnost/skrytí polí, které se dnes vykreslují (sidebar metriky, diagnostické JSONy). **Nebezpečné** jsou změny signatur `render_result`/`render_preview`/`select_folder`, pokud je nahradí jiná knihovna volá (žádná dnes neexistuje – nízké riziko), a jakákoli změna parametrů volání `search_all(...)` (`is_question`, `expand_query`, `settings.result_count`) – to by bylo zásahem do retrievalu, což je mimo zadání.

### 1.2 Testovací povrch, který audit musí respektovat

`tests/test_ui.py` používá `streamlit.testing.v1.AppTest` a **assertuje přesné české texty a labely tlačítek**: `"Náhled"`, `"Zobrazit další výsledky"`, `"Nebyly nalezeny"`, `"Odpověď"`, `"Přejít na citovaný dokument"`, nadpisy `"Nastavení"`/`"Historie a diagnostika"`/`"Diagnostika"`, `"Výsledky (N)"`. Toto je zásadní zjištění pro plánování: **jakákoli změna textu/labelu musí být doprovázena update testu v témže kroku**, jinak se sada rozbije bez souvislosti s backendem.

### 1.3 Datový kontrakt, který UI dnes nevyužívá

`ai_search.answer()` (viz `ai_search.py`, řádek 1692–1720) už vrací:
```python
{"answer": "<vyrenderovaný markdown string včetně bloku Jistota odpovědi>",
 "citations": [...], "model": "...", "confidence": "green"|"yellow"|"red"}
```
`app.py` (řádek 169) čte pouze `response["answer"]` a `response["citations"]`. **Klíč `confidence` existuje, je spočítaný, a UI ho ignoruje** – informace o jistotě odpovědi je dnes pohřbená jako plain text na konci `answer` stringu (`confidence_block`, `ai_search.py:1695`), místo aby byla samostatný vizuální prvek. To je největší „quick win" celého auditu: nulové backendové riziko, čistě otázka čtení pole, které už existuje.

Podobně `ai_search.answer()` interně pracuje se strukturovaným JSON (`shrnuti`, `oblasti`/`body`, `nenalezeno` – schémata `STRUCTURED_ANSWER_SCHEMA`/`CONCISE_ANSWER_SCHEMA`, `ai_search.py:1506–1533`), ale **flattenuje ho na jeden markdown string dřív, než se vrátí z funkce** (`_render_structured_answer`/`_render_concise_answer`). UI proto nemůže dnes zobrazit „krátké shrnutí" oddělené od „bodů" bez zásahu do `ai_search.py` – viz Fáze UI-2.

---

## 2. Problémy UX

Hodnoceno jako aplikace pro stavbyvedoucího, projektového manažera a technický dozor investora – ne jako vývojářský dashboard.

### A) Levý panel

**Problém:** sidebar dnes na každé stránce a při každém použití zobrazuje: 4× `st.metric` (PDF/E-maily/Poznámky/Dokumenty), velikost indexu v MB, přesný timestamp poslední synchronizace, **název embedding modelu** (`BAAI/bge-m3`), **název LLM modelu**, stav Ollamy. To je směs dvou různých publik v jednom bloku:
- „Je systém funkční?" (stavbyvedoucí potřebuje vidět) – Ollama stav, zda je index připraven.
- „Jaký model/kolik MB/kolik PDF vs. e-mailů" (administrátor/vývojář) – irelevantní pro každodenní použití, vizuální šum na obrazovce, která se zobrazuje úplně pořád.

**Co skrýt do diagnostiky:** název embedding modelu, název LLM modelu, přesná velikost v MB, rozpad PDF/e-mail/poznámky na 4 samostatné metriky. Toto vše už je jinak dostupné na stránce `Diagnostika` (`diagnostics.collect_diagnostics`) – jde tedy o **odstranění duplicity**, ne o ztrátu informace.

**Co má uživatel vidět při každém použití:** 1 kompaktní indikátor „index připraven, N dokumentů, aktualizováno DD.MM." + 1 stavový indikátor Ollamy (zelený/červený bod). Tlačítko „Aktualizovat index" zůstává primární akcí sidebaru, beze změny chování.

**Vedlejší zjištění:** sidebar má **dvě samostatná tlačítka** – „Historie a diagnostika" (stránka `history`) a „Diagnostika" (stránka `diagnostics`) – s překrývajícím se obsahem (obojí zobrazuje Ollama stav a embedding/LLM info). To je zmatek v navigaci, ne jen kosmetika.

### B) Vyhledávání

**První dojem:** nadpis „Najděte informace ve svých projektech" + jednořádkový `st.text_input` s placeholderem – funkčně v pořádku, ale vizuálně nerozlišený od formulářového pole. Chybí vizuální „ukotvení" vyhledávacího pole jako primárního akčního bodu obrazovky (na rozdíl od Spotlight/ChatGPT, kde je vyhledávací/vstupní pole jednoznačně dominantní prvek stránky).

**Jasnost zadání dotazu:** rozpoznaný režim („Hledat dokumenty" vs. „Položit otázku") se zobrazuje jako `st.caption` – malý, nevýrazný text, snadno přehlédnutelný, přitom jde o informaci, která **zásadně změní tvar výstupu** (seznam výsledků vs. AI odpověď s citacemi). Toggle „Vynutit hloubkovou analýzu" je vždy vidět, i když je relevantní jen pro dotazy v režimu otázky – zbytečná kognitivní zátěž při prostém vyhledávání dokumentu.

**Nápovědy:** jediná nápověda je placeholder v poli a `help=` tooltip u toggle. Žádné navržené/oblíbené dotazy, žádné rychlé filtry viditelné bez rozkliknutí. `st.expander("Filtry")` je vždy sbalený, 6 polí (zdroj, projekt, podsložka, typ souboru, autor, datum) – vhodné pro power usera, ale bez vizuální vazby na aktuální dotaz.

**Rychlost orientace:** po odeslání dotazu se zobrazí `st.caption(f"Nalezeno: N · čas Xs · režim ...")` – informačně správné, ale opět jako malý text bez hierarchie, sloučené do jedné věty se třemi různě důležitými fakty (počet, latence, režim).

### C) AI odpověď

**Současný stav:** `st.subheader("Odpověď")` + `st.write(response["answer"])` – jeden neformátovaný blok textu, kde je jistota odpovědi zaflákaná jako text na konci (viz 1.3). Neexistuje vizuální hierarchie mezi „přímou odpovědí" a „podrobnostmi".

**Navržená struktura (viz sekce 3.4 pro detail):**
1. **Krátké shrnutí** – 1–2 věty, vizuálně odlišené (větší/tučné písmo), odpovídá poli `shrnuti` ze `STRUCTURED_ANSWER_SCHEMA` – dnes backend toto pole počítá jen pro checklist dotazy a zahazuje ho do plain textu.
2. **Body** – odrážky s typem tvrzení (fakt/požadavek/doporučení – toto pole `typ` už `_render_answer_item` zpracovává, jen ho nevykresluje jako vizuální rozlišení, pouze jako textovou předponu `"Požadavek: "`/`"Doporučení: "`).
3. **Jistota odpovědi** – barevný badge (🟢/🟡/🔴) z `response["confidence"]`, ne text v odpovědi.
4. **Počet zdrojů** – `len(response["citations"])`, dnes se dozví uživatel jen nepřímo počítáním citačních bloků pod odpovědí.

### D) Citace

**Současný stav:** citace v QA režimu (`app.py:169–172`) jsou samostatný HTML blok (`.citation` CSS třída, modrý levý okraj) + samostatné `st.button("Přejít na citovaný dokument")` **pod** barevným blokem, ne uvnitř něj – vizuálně nesouvislé (barevný rámeček skončí, tlačítko je mimo něj v obyčejném rozvržení).

**Zásadní nekonzistence:** dokument-lookup režim (`render_result`, řádek 51–59) má u každého výsledku 3 akce (Náhled / Otevřít dokument / Finder) **a** hvězdičkové hodnocení shody (`match_label`) **a** důvod shody (`match_reason` – „kombinovaná shoda", „podobnost 89 %"...). QA citace **nemá nic z toho** – žádné hodnocení shody, jen text a jedno tlačítko. Uživatel v režimu otázky tak dostává méně důvěryhodnostních signálů než v režimu vyhledávání dokumentu, přestože `match_label`/`match_reason` jsou obecné funkce v `ui_services.py`, které lze zavolat i tam – čistě chybějící volání, ne chybějící data.

**Návrh:**
- Zdroje oddělit od odpovědi jasným vizuálním blokem (karta s vlastním nadpisem „Zdroje (N)", ne jen `st.subheader`).
- Každý zdroj = jedna karta s: ikonou podle typu souboru, názvem dokumentu, projektem/složkou, hvězdičkovým hodnocením shody + důvodem (stejně jako dokument-lookup), zvýrazněným úryvkem.
- Primární akce „Otevřít dokument" jako tlačítko **uvnitř** téže karty, ne pod samostatným HTML blokem – jeden klik, jedna vizuální jednotka.
- Sekundárně „Zobrazit ve Finderu" (ikonové tlačítko), stejně jako dnes existuje u document-lookup výsledků.

### E) Stavové informace

Rozděleno podle toho, co potřebuje **běžný uživatel** (stavbyvedoucí/PM/TDI) vs. **administrátor**:

| Informace | Dnes | Návrh |
|---|---|---|
| Index připraven, kolik dokumentů, kdy naposledy | sidebar, 4 samostatné metriky + caption | sidebar, **1 kompaktní řádek** |
| Ollama běží/neběží | sidebar, `st.write` | sidebar, stavový indikátor (barevný bod) |
| Embedding model, LLM model | sidebar (vždy vidět) | **jen** Diagnostika/Nastavení |
| Velikost indexu v MB | sidebar (vždy vidět) | **jen** Diagnostika |
| Rozpad PDF/e-mail/poznámky | sidebar, 4 metriky | **jen** Diagnostika (tam je i dnes, jako `extensions`) |
| Watchdog, výkon, poslední chyby, integrity check | Diagnostika (správně) | beze změny – zůstává admin-only |
| Historie synchronizací | stránka „Historie" | beze změny, ale zvážit sloučení s Diagnostikou (viz A) |

Dnešní `Diagnostika` stránka je z hlediska obsahu **správně navržena** pro admin publikum (JSON dumpy, watchdog, per-extension tabulka, error log, „Ověřit index", export HTML/PDF) – problém není tam, problém je že sidebar tuto hranici neresepektuje a technické detaily prosakují do hlavního used flow.

---

## 3. Návrh nového UI

### 3.1 Inspirace a princip

- **Apple Spotlight:** jedno dominantní vstupní pole, minimální chrom kolem něj, výsledky jako plovoucí seznam karet, žádné zbytečné rámečky.
- **ChatGPT:** generózní bílý prostor, jasná hierarchie (shrnutí → detail → zdroje), zprávy jako samostatné vizuální jednotky, ne hustý formulářový layout.
- **Enterprise AI nástroje:** barevné stavové badge s jednoznačným sémantickým významem (ne dekorace), sekundární informace vždy odsunuta za jeden klik (expander/samostatná stránka), ne v hlavním poli vidění.

Princip pro stavbu na denní bázi: **žádná animace, žádné rozptylující prvky, vysoký kontrast pro čtení na mobilu/tabletu na světle na stavbě.**

### 3.2 Barevná paleta

Návrh cílí primárně na `.streamlit/config.toml` (nativní, podporovaný mechanismus theming), ne na rozšiřování ad-hoc CSS blobu.

| Token | Hex | Použití |
|---|---|---|
| `primaryColor` | `#2563EB` | primární tlačítka, aktivní stav, odkazy – navazuje na už používanou `#3b82f6` u citací, jen konzistentně formalizováno |
| `backgroundColor` | `#FFFFFF` | hlavní plocha |
| `secondaryBackgroundColor` | `#F5F7FA` | karty výsledků, sidebar, panely |
| textColor (základní) | `#1B2333` | primární text |
| sekundární text | `#5B6472` (opacity varianta) | metadata, popisky – zachovává dnešní `.meta{opacity:.75}` logiku, jen formalizovanou |
| hraniční linka | `#E3E7ED` | ohraničení karet místo dnešního `color-mix(...)` |
| úspěch/vysoká jistota | `#16A34A` | 🟢 badge, potvrzující stavy |
| upozornění/střední jistota | `#D97706` | 🟡 badge |
| chyba/nízká jistota | `#DC2626` | 🔴 badge, chybové stavy |
| zvýraznění shody v textu | `#FDE68A` (`mark`) | beze změny, funguje dobře |

Tato paleta **formalizuje sémantiku, která už v backendu existuje** (`CONFIDENCE_LABELS = {"green":..., "yellow":..., "red":...}`) – nejde o novou barevnou logiku, jen o její promítnutí do UI.

### 3.3 Hierarchie informací (shora dolů)

1. Vstupní pole + rozpoznaný režim (badge, ne caption) – nejvyšší váha, vždy viditelné.
2. Primární obsah: AI odpověď **nebo** seznam výsledků dokumentů.
3. Podpůrné důkazy: zdroje/citace – jasně oddělený blok pod primárním obsahem.
4. Sekundární ovládání: filtry, stránkování „Zobrazit další výsledky".
5. Systémový stav: patička sidebaru, nejnižší vizuální váha, detail schovaný za Diagnostiku.

### 3.4 Hlavní obrazovka

```
┌──────────────────────────────────────────────────────────┐
│  🔎  [ Hledaný výraz nebo otázka.......................] │  ← dominantní, zaoblené, jemný stín
│      🔤 Hledat dokumenty   (badge, ne caption)            │
├──────────────────────────────────────────────────────────┤
│  Nalezeno 12 dokumentů · 0.8 s          [▸ Filtry]        │  ← jedna řádka, filtry sbalené
├──────────────────────────────────────────────────────────┤
│  [primární obsah – odpověď NEBO výsledky]                 │
└──────────────────────────────────────────────────────────┘
```

### 3.5 Blok odpovědi AI

```
┌─ Odpověď ──────────────────────────────────── 🟢 Vysoká jistota ─┐
│  Hydroizolace musí být provedena do 15. 6. podle TP FERI.        │  ← shrnutí, tučně
│                                                                     │
│  • Požadavek: dodavatel musí předložit protokol o zkoušce [1]    │
│  • Fakt: izolace je typu Pentaflex [2]                           │
│                                                                     │
│  Odpověď vychází z 3 zdrojů                                       │
└────────────────────────────────────────────────────────────────────┘
```

### 3.6 Blok zdrojů

```
Zdroje (3)
┌────────────────────────────────────────────────────────┐
│ 📄  TP_hydroizolace_FERI_rev2.docx      ★★★★★ Vysoká   │
│ 📁 Projekt X · 📂 03_Technická dokumentace              │
│ 🔀 kombinovaná shoda · podobnost 91 %                   │
│ "…musí být provedena do 15. 6. podle technologického…" │
│                                    [Otevřít]  [Finder]  │
└────────────────────────────────────────────────────────┘
```

### 3.7 Levý panel

```
🔎 AI Search
Lokální hledání v projektových dokumentech

📁 Projekt: NDS Property           [Změnit]

●  Index připraven · 6 342 dokumentů · akt. 08.08.
●  Ollama běží

[      Aktualizovat index      ]

⚙ Nastavení    📊 Diagnostika
```

Bez `st.metric` mřížky, bez modelových názvů, bez MB – to vše zůstává v Diagnostice.

---

## 4. Priorita změn

### Fáze UI-1 – bezpečné změny (layout, texty, CSS, pořadí prvků)

| # | Změna | Soubory | Bezpečné? | Dopad na backend/retrieval? |
|---|---|---|---|---|
| 1 | `.streamlit/config.toml` s barevnou paletou (sekce 3.2) | nový soubor `.streamlit/config.toml` | ano | žádný |
| 2 | Badge jistoty odpovědi z `response["confidence"]` (pole už existuje) | `app.py` | ano | žádný – jen čtení existujícího pole |
| 3 | Řádek „Odpověď vychází z N zdrojů" | `app.py` | ano | žádný |
| 4 | Sidebar: skrýt model/MB/rozpad typů do Diagnostiky, ponechat 1 stavový řádek | `app.py` (řádky 66–99) | ano | žádný – přesun zobrazení, ne dat |
| 5 | Doplnit `match_label`/`match_reason` i do QA citací | `app.py` (řádek 169–172), volá existující `ui_services` funkce | ano | žádný |
| 6 | Ikona dokumentu podle přípony místo vždy 📄 | `app.py` | ano | žádný |
| 7 | Přepracovat úvodní wizard na `st.columns`/`st.container(border=True)` bez vlastní CSS třídy | `app.py` | ano | žádný |
| 8 | Rozpoznaný režim jako badge namísto `st.caption` | `app.py` | ano | žádný |
| 9 | Zkrácení/reorganizace CSS blobu na řádku 13–15 | `app.py` | ano, **ale** nutno souběžně ověřit vizuálně (CSS třídy se používají na více místech) | žádný |

**Riziko společné pro celou Fázi UI-1:** `tests/test_ui.py` assertuje přesné texty tlačítek/nadpisů (`"Náhled"`, `"Odpověď"`, `"Přejít na citovaný dokument"` atd. – viz 1.2). Každá textová/labelová změna vyžaduje **v témže kroku** upravit odpovídající test, jinak selže CI bez souvislosti s funkčností.

### Fáze UI-2 – lepší práce se zdroji, dokumentové karty, navigace

| # | Změna | Soubory | Bezpečné? | Dopad na backend/retrieval? |
|---|---|---|---|---|
| 1 | Nahradit raw HTML `.result`/`.citation` divy `st.container(border=True)` kartami | `app.py` (`render_result`, `render_preview`, QA citační smyčka) | ano, nízké riziko | žádný – čistě renderovací refaktor, ale vizuálně největší zásah, nutná manuální QA |
| 2 | Exponovat `shrnuti`/`oblasti`/`nenalezeno` jako strukturovaná data z `ai_search.answer()` (dnes vrací jen finální markdown string) | **`ai_search.py`** (přidání klíče do return dict `answer()`) + `app.py` (čtení nového klíče) | **částečně** – nízké backendové riziko (čistě přídavný klíč, žádná změna scoringu/promptu/rerankingu), **ale zasahuje mimo UI vrstvu**, vyžaduje explicitní schválení nad rámec „pouze UI" | riziko: nulové pro retrieval/scoring, nenulové pro kontrakt návratové hodnoty `answer()` – nutno ověřit `tests/test_answer_quality.py` |
| 3 | Jednotné akce na kartě zdroje (Otevřít/Finder/Náhled) i v QA režimu | `app.py` | ano | žádný |
| 4 | Sloučit/přejmenovat navigaci „Historie a diagnostika" vs. „Diagnostika" | `app.py` | ano, ale mění `st.session_state.page` klíče | žádný na backend, **ale** nutno upravit `test_secondary_screens_render` (parametrizovaný na `page="history"/"diagnostics"`) |

### Fáze UI-3 – budoucí funkce (mimo scope v1.1, jen návrh)

| # | Nápad | Poznámka k riziku |
|---|---|---|
| 1 | Konverzační UI pro QA režim (`st.chat_message`/`st.chat_input`, historie dotazů) | Zásadní změna `app.py` a modelu `session_state`; nutno ověřit dopad na `@st.cache_resource embeddings()` a `st.rerun()` cykly; **žádný přímý backendový risk**, ale je to nová funkce, ne redesign. |
| 2 | KPI dashboard pro PM/TDI perzonu (stav dokumentace projektu) | Nová agregační logika – čistě čtecí funkce v `ui_services.py`, bez zásahu do `search`/`sync`; nový scope, ne redesign existující obrazovky. |
| 3 | Role/oprávnění (admin vs. běžný uživatel) – fyzické schování Diagnostiky, ne jen sbalení | Vyžaduje rozhodnutí mimo UI redesign (kde se role ukládá, jak se ověřuje); `ai_search_config.py` by potřeboval nový konfigurační přepínač. |

---

## 5. Doporučený implementační plán

1. **Fáze UI-1 jako jeden PR/krok.** Všechny položky jsou nezávislé na retrievalu a lze je nasadit společně. Po implementaci: `pytest tests/test_ui.py -q` (aktualizovat asserty textů souběžně s textovými změnami) + manuální vizuální kontrola (`streamlit run app.py`), protože `AppTest` neověří CSS/vzhled, jen strukturu stromu prvků.
2. **Fáze UI-2, bod 1 a 3 (karty, parita akcí) jako druhý krok**, samostatně od bodu 2 (strukturovaná data z `ai_search.py`) – tyto dva body mají odlišný rizikový profil a odlišné schvalovací potřeby (čistě UI vs. zásah do `ai_search.py`).
3. **Fáze UI-2, bod 2 vyžaduje explicitní rozhodnutí uživatele**, zda je ochoten rozšířit zadání z „pouze UI vrstva" na „UI + jeden přídavný klíč v `ai_search.answer()`". Bez něj lze krátké shrnutí v UI-1 přiblížit jen heuristicky (např. první věta/odstavec z `response["answer"]`), což je fragilnější a nedoporučuji jako trvalé řešení.
4. **Testování po každém kroku:** `pytest tests/test_ui.py -q` vždy; pro krok s `ai_search.py` navíc `pytest tests/test_answer_quality.py -q` a defenzivně `python -m benchmark.run_retrieval_regression` (i když by se scoring nemělo dotknout, ověření je levné).
5. **Navigace (Fáze UI-2, bod 4)** až po ověření, že žádný jiný test/skript neodkazuje na `st.session_state.page` hodnoty `"history"`/`"diagnostics"` napevno (dnes jen `test_ui.py`).
6. **Rollback:** všechny navržené změny jsou čistě frontendové bez migrace dat – rollback je `git revert`, bez dopadu na SQLite/LanceDB/index.
7. **Fáze UI-3** ponechat jako samostatné budoucí zadání až po nasazení a vyhodnocení UI-1/UI-2 na reálném použití.

---

*Žádné soubory nebyly v rámci tohoto auditu změněny. Tento dokument (`UI_AUDIT_AI_SEARCH_v1_1.md`) je jediný artefakt vytvořený tímto krokem.*
