# AI Search pro macOS

Lokální česká aplikace pro hybridní hledání a otázky nad dokumenty v lokálně
synchronizované složce Box Drive. Box API ani cloudové e-mailové API nepoužívá.

## První spuštění

```bash
cd "/Users/miroslavvarvarovsky/Documents/AI Search"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama serve
```

V jiném okně Terminálu spusťte:

```bash
cd "/Users/miroslavvarvarovsky/Documents/AI Search"
source .venv/bin/activate
streamlit run app.py
```

Nebo dvakrát klikněte na `Spustit AI Search.command`.

## První indexace

V levém panelu vložte cestu k lokální projektové složce Box Drive a stiskněte
**Aktualizovat index**. Aplikace soubory pouze čte. Model `BAAI/bge-m3` se při
prvním použití automaticky stáhne, takže první spuštění trvá déle.

Exportované e-maily ve formátu EML a lokální poznámky přidejte v **Nastavení**.
Každý zdroj má samostatný inkrementální index a výsledky se společně řadí podle
relevance.

## Přidání do Docku

1. Dvakrát klikněte na `Vytvořit aplikaci pro Dock.command`.
2. Finder zobrazí vytvořenou aplikaci `AI Search.app`.
3. Přetáhněte ji do Docku.

## Instalace profesionální macOS aplikace

Hotovou aplikaci `AI Search.app` můžete přetáhnout do složky **Aplikace**
(`/Applications`) a následně ji spouštět dvojklikem bez Terminálu a Safari.
Pro připnutí spusťte aplikaci, klikněte pravým tlačítkem na její ikonu v Docku
a zvolte **Volby → Ponechat v Docku**. Aplikace používá vlastní nativní okno,
automaticky zkontroluje Ollamu i lokální server a provozní log ukládá do
`~/Library/Logs/AI Search/`.

## Nejčastější problémy

- **Ollama není dostupná:** v Terminálu spusťte `ollama serve`. Hledání funguje
  dál, pouze generování odpovědi zobrazí srozumitelnou chybu.
- **Složka neexistuje:** ověřte, že je Box Drive spuštěný a projekt je lokálně
  dostupný.
- **Model se nestáhne:** zkontrolujte internetové připojení a aplikaci spusťte
  znovu. Po stažení pracuje model z místní cache.
- **Soubor nelze otevřít:** mohl být přesunut nebo odstraněn; aktualizujte index
  a zkontrolujte oprávnění macOS.
- **Prázdné výsledky:** zrušte filtry a ověřte stav indexu v levém panelu.

## Testy

```bash
source .venv/bin/activate
pytest -q
```
