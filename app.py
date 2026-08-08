from __future__ import annotations
import base64, html, threading, time
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import ai_search
import diagnostics
from ai_search_config import APP_SUPPORT_DIR, QUERY_EXPANSION_MODE
from ui_services import Settings, apply_filters, choose_folder, classify_query, context_excerpt, display_location, ensure_runtime_layout, index_source, index_summary, indexed_root, load_settings, match_label, match_reason, ollama_status, open_path, preview_document, read_history, save_settings, search_all, state_file

ROOT=Path(__file__).resolve().parent; STATE=APP_SUPPORT_DIR; ensure_runtime_layout(STATE,ROOT/".ai-search-state"); SETTINGS=state_file(STATE,"settings.json"); PAGE_SIZE=8
st.set_page_config(page_title="AI Search",page_icon="🔎",layout="wide")
# Barevná paleta žije primárně v .streamlit/config.toml (theme.primaryColor atd. -
# viz UI_AUDIT_AI_SEARCH_v1_1.md sekce 3.2). Tento blok drží jen strukturální
# CSS, které Streamlit theming nepokrývá (karty výsledků/citací, zvýraznění,
# stavové badge v sidebaru a u odpovědi AI).
st.markdown("""<style>
.stApp{max-width:1720px;margin:auto}
.result{border:1px solid color-mix(in srgb,var(--text-color) 16%,transparent);background:var(--secondary-background-color);border-radius:10px;padding:.65rem .8rem;margin:.32rem 0}
.result-title{font-weight:700;font-size:1rem}
.meta{font-size:.8rem;opacity:.75;line-height:1.35}
.context{font-size:.9rem;line-height:1.38;margin:.35rem 0}
.citation{border-left:4px solid #2563EB;background:var(--secondary-background-color);padding:.65rem .8rem;margin:.4rem 0 .2rem 0;border-radius:0 8px 8px 0}
mark{background:#fde68a;color:#111827;border-radius:3px;padding:0 2px}
.status-line{font-size:.92rem;line-height:1.7;margin:.1rem 0}
.badge{display:inline-block;padding:.15rem .65rem;border-radius:999px;font-size:.8rem;font-weight:600;white-space:nowrap}
.badge-green{background:#DCFCE7;color:#14532D}
.badge-yellow{background:#FEF3C7;color:#78350F}
.badge-red{background:#FEE2E2;color:#7F1D1D}
.badge-blue{background:#DBEAFE;color:#1E3A8A}
@media(max-width:760px){.result{padding:.55rem}}
</style>""",unsafe_allow_html=True)

@st.cache_resource(show_spinner="Načítám významové vyhledávání…")
def embeddings(): return ai_search.Embeddings()

def highlight(text,query):
    safe=html.escape(text)
    for word in [w for w in query.split() if len(w)>2]:
        import re; safe=re.sub(re.escape(html.escape(word)),lambda m:f"<mark>{m.group(0)}</mark>",safe,flags=re.IGNORECASE)
    return safe

def select_folder(label,current,setting_name,key_prefix="main"):
    c1,c2=st.columns([1,2])
    if c1.button(f"📂 Vybrat {label}",use_container_width=True,key=f"choose-{key_prefix}-{setting_name}"):
        ok,value=choose_folder(f"Vyberte složku pro {label}")
        if ok: setattr(settings,setting_name,value); save_settings(SETTINGS,settings); st.rerun()
    if current:
        c2.caption(f"Vybráno: **{Path(current).name}**")
        with c2.expander("Zobrazit úplnou cestu"): st.code(current)
    else: c2.caption("Složka zatím není vybrána")

def render_preview(row,query):
    project,folder=display_location(row,settings); st.subheader("Náhled")
    st.markdown(f"**{row.get('title',row['document'])}**"); st.caption(f"📁 {project} · 📂 {folder}")
    preview=preview_document(Path(row["path"]),context_excerpt(row.get("quote",""),query))
    if preview["kind"]=="image":
        st.image(base64.b64decode(preview["content"]),caption="Náhled 1. strany (vyrenderováno lokálně)",use_container_width=True)
        st.download_button("💾 Stáhnout PDF",Path(row["path"]).read_bytes(),file_name=Path(row["path"]).name,mime="application/pdf",key="preview-download")
    elif preview["kind"]=="pdf":
        st.info("📄 PDF náhled se nepodařilo vyrenderovat. Otevřete soubor přímo, nebo si ho stáhněte.")
        pdf_data=base64.b64decode(preview["content"])
        st.download_button("💾 Stáhnout PDF",pdf_data,file_name=Path(row["path"]).name,mime="application/pdf",key="preview-download")
    elif preview["kind"]=="error": st.error(preview["content"])
    else: st.markdown(highlight(preview["content"][:6000],query),unsafe_allow_html=True)
    if st.button("📄 Otevřít dokument",key="preview-open"): st.toast(open_path(Path(row["path"]))[1])

def relevance_meta_html(row,best):
    """⭐-hodnocení shody + lidsky čitelný důvod (ui_services.match_label/
    match_reason), sdíleno mezi výsledky vyhledávání a citacemi AI odpovědi -
    dřív byl tento signál vidět jen u prvních, teď je konzistentní na obou
    místech (viz UI_AUDIT_AI_SEARCH_v1_1.md, sekce 2D). Vykreslení je oproti
    předchozí verzi render_result o jeden řádek kompaktnější (důvod na stejném
    řádku jako hodnocení, ne pod ním) - žádný test na přesný HTML string necílí."""
    relevance=match_label(row.get("score",0),best) if best else ""
    if not relevance: return ""
    reason=match_reason(row); reason_html=f" · {html.escape(reason)}" if reason else ""
    return f"<br>⭐ {relevance}{reason_html}"

def render_result(row,query,key,best):
    project,folder=display_location(row,settings); context=context_excerpt(row.get("quote",""),query)
    st.markdown(f"<div class='result'><div class='result-title'>📄 {html.escape(row.get('title',row['document']))}</div><div class='meta'>📁 {html.escape(project)} &nbsp;·&nbsp; 📂 {html.escape(folder)} &nbsp;·&nbsp; 📅 {html.escape(row.get('date','—'))}{relevance_meta_html(row,best)}</div><div class='context'>📝 {highlight(context,query)}</div></div>",unsafe_allow_html=True)
    a,b,c=st.columns(3)
    if a.button("Náhled",key=f"preview-{key}",use_container_width=True): st.session_state.preview=row; st.rerun()
    if b.button("📄 Otevřít dokument",key=f"open-{key}",use_container_width=True): st.toast(open_path(Path(row["path"]))[1])
    if c.button("📂 Finder",key=f"finder-{key}",use_container_width=True): st.toast(open_path(Path(row["path"]),True)[1])
    with st.expander("Zobrazit úplnou cestu"): st.code(row["path"])

def confidence_badge_html(level):
    """Barevný badge pro ai_search.answer()'s `confidence` (green/yellow/red) -
    pole už backend počítá a vrací, dosud ho ale UI nikde nečetlo (bylo vidět
    jen jako text zaflákaný na konci odpovědi). Čistě čtení existujícího pole,
    ai_search.py se nemění. `CONFIDENCE_LABELS` je existující veřejná konstanta
    modulu ai_search, ne nová závislost."""
    label=ai_search.CONFIDENCE_LABELS.get(level)
    return f"<span class='badge badge-{level}'>{label}</span>" if label else ""

def strip_confidence_footer(text):
    """Backend (ai_search.answer) připojuje na konec odpovědi lidsky čitelný
    blok o jistotě odpovědi ("\\n\\nJistota odpovědi:\\n..."). UI teď tutéž
    informaci zobrazuje jako badge nad odpovědí, takže tento blok při
    zobrazení odřezáváme, aby se nezobrazoval dvakrát. Čistě textová úprava
    zobrazení v app.py - ai_search.py se nemění a formát návratové hodnoty
    zůstává stejný. Pokud by backend formát bloku v budoucnu změnil, tento
    řez se bezpečně stane no-opem (oddělovač se nenajde, text se zobrazí
    celý) - nejde tedy o tvrdou vazbu, která by mohla něco rozbít."""
    return text.split("\n\nJistota odpovědi:")[0]

def sources_caption(count):
    return f"Odpověď vychází z {count} zdroje" if count==1 else f"Odpověď vychází ze {count} zdrojů"

settings=load_settings(SETTINGS); summary=index_summary(STATE)
if not settings.project_root: settings.project_root=indexed_root(STATE)
st.session_state.setdefault("page","search"); st.session_state.setdefault("visible_results",PAGE_SIZE)
if "index_message" in st.session_state: st.toast(st.session_state.pop("index_message"))

with st.sidebar:
    st.title("🔎 AI Search"); st.caption("Lokální hledání v projektových dokumentech")
    select_folder("projekt",settings.project_root,"project_root")
    st.divider()
    # Zjednodušený přehled systému (UI_AUDIT_AI_SEARCH_v1_1.md, sekce 2A/4-UI1):
    # název embedding/LLM modelu, velikost v MB a rozpad PDF/e-mail/poznámky
    # patří jen do Diagnostiky - zde je vidí uživatel na každém hledání, i když
    # to k práci nepotřebuje. Data samotná (index_summary) se nemění, mění se
    # jen to, co z nich sidebar vykresluje.
    index_ready=summary["ready"]; ollama_ok=ollama_status()
    st.markdown(f"<div class='status-line'>{'🟢' if index_ready else '🔴'} <b>{'Index připraven' if index_ready else 'Index není připraven'}</b>"+(f" · {summary['total']} dokumentů" if index_ready else "")+"</div>",unsafe_allow_html=True)
    if index_ready: st.caption(f"Aktualizováno: {summary['latest']}")
    st.markdown(f"<div class='status-line'>{'🟢' if ollama_ok else '🔴'} Ollama {'běží' if ollama_ok else 'není dostupná'}</div>",unsafe_allow_html=True)
    job=st.session_state.get("index_job"); running=bool(job and job["thread"].is_alive())
    if st.button("Aktualizovat index",type="primary",use_container_width=True,disabled=not bool(settings.project_root) or running):
        root=Path(settings.project_root)
        if not root.is_dir(): st.error("Složka neexistuje. Vyberte ji znovu ve Finderu.")
        else:
            job={"stop":threading.Event(),"progress":{"phase":"Připravuji indexaci","current":0,"total":0,"elapsed":0,"eta":None,"counts":{}},"result":None,"error":None}
            def run_index():
                try: job["result"]=index_source(root,"Dokument",STATE,embeddings(),progress=lambda event:job.update(progress=event),stop_event=job["stop"])
                except Exception as exc: job["error"]=str(exc)
            job["thread"]=threading.Thread(target=run_index,name="AI Search indexace",daemon=True); st.session_state.index_job=job; job["thread"].start(); st.rerun()
    if job:
        event=job["progress"]; total=max(event.get("total",0),1); percent=min(99,max(1,int(event.get("current",0)/total*100))); eta=f" · ETA {event['eta']:.0f} s" if event.get("eta") else ""; name=f" · {Path(event['path']).name}" if event.get("path") else ""; counts=event.get("counts",{})
        st.progress(percent,text=f"{event.get('phase','Indexuji')} · {event.get('current',0)} / {event.get('total',0)}{name}{eta}")
        st.caption(f"Chunk {event.get('chunk_number',0)} · fáze {event.get('phase_elapsed',0):.1f} s · {event.get('chunks_per_second',0):.2f} chunků/s · Nové {counts.get('added',0)} · Aktualizované {counts.get('changed',0)+counts.get('renamed',0)} · Přeskočené {counts.get('unchanged',0)+counts.get('skipped',0)} · Chyby {counts.get('errors',0)} · uplynulo {event.get('elapsed',0):.0f} s")
        if running:
            if st.button("Stop",use_container_width=True): job["stop"].set(); st.warning("Dokončuji právě zpracovávaný dokument…")
            time.sleep(.5); st.rerun()
        else:
            if job["error"]: st.error(f"Indexaci se nepodařilo dokončit: {job['error']}.")
            elif job["result"]:
                result=job["result"]; save_settings(SETTINGS,settings); state="zastavena" if result.get("stopped") else "dokončena"; st.success(f"Indexace {state}. Nové {result['added']} · aktualizované {result['changed']+result['renamed']} · přeskočené {result['unchanged']+result['skipped']} · chyby {result['errors']}.")
            if st.button("Zavřít průběh",use_container_width=True): del st.session_state.index_job; st.rerun()
    if st.button("Nastavení",use_container_width=True): st.session_state.page="settings"; st.rerun()
    if st.button("Historie a diagnostika",use_container_width=True): st.session_state.page="history"; st.rerun()
    if st.button("Diagnostika",use_container_width=True): st.session_state.page="diagnostics"; st.rerun()

if st.session_state.page=="diagnostics":
    st.header("Diagnostika"); data=diagnostics.collect_diagnostics(settings.project_root,STATE)
    if data["rc1"]: st.markdown("<div style='padding:1.2rem;border-radius:12px;background:#dcfce7;color:#14532d;font-size:1.3rem'><b>AI SEARCH RC1</b><br>Produkční stav: <b>PŘIPRAVENO</b></div>",unsafe_allow_html=True)
    else: st.markdown("<div style='padding:1.2rem;border-radius:12px;background:#fee2e2;color:#7f1d1d'><b>AI SEARCH RC1</b><br>Produkční stav: NEPŘIPRAVENO</div>",unsafe_allow_html=True)
    st.subheader("Systémový stav"); st.json({**data["paths"],**data["models"],"velikost databáze":data["sizes"]["database"],"velikost LanceDB":data["sizes"]["lance"],"velikost cache":data["sizes"]["cache"],"poslední indexace":data["last_index"],**data["counts"]},expanded=True)
    st.subheader("Index"); st.dataframe([{"přípona":name,**values} for name,values in data["extensions"].items()],use_container_width=True,hide_index=True)
    st.subheader("Watchdog"); st.json(data["watchdog"],expanded=False)
    st.subheader("Výkon"); st.json(data["performance"],expanded=False)
    st.subheader("Poslední chyby")
    if data["errors"]: st.dataframe(data["errors"],use_container_width=True,hide_index=True)
    else: st.success("Nejsou evidovány žádné chyby.")
    log_path=STATE/"logs/project.log"
    if st.button("Otevřít log",disabled=not log_path.exists()): st.toast(open_path(log_path)[1])
    c1,c2=st.columns(2)
    if c1.button("Ověřit index",type="primary",use_container_width=True): st.session_state.diagnostic_check=diagnostics.verify_index(STATE/"database/project.sqlite3",STATE/"lance/project",STATE/"database/.index.lock")
    if c2.button("Export diagnostiky",use_container_width=True):
        html_report,pdf_report=diagnostics.export_reports(data,STATE); st.session_state.diagnostic_export=(str(html_report),str(pdf_report))
    if "diagnostic_check" in st.session_state:
        check=st.session_state.diagnostic_check; (st.success if check["ok"] else st.error)("Kontrola indexu: OK" if check["ok"] else "Kontrola indexu zjistila problém."); st.json(check)
    if "diagnostic_export" in st.session_state: st.success(f"HTML: {st.session_state.diagnostic_export[0]}\n\nPDF: {st.session_state.diagnostic_export[1]}")
    if st.button("Zpět k hledání",key="diagnostics-back"): st.session_state.page="search"; st.rerun()
elif st.session_state.page=="settings":
    st.header("Nastavení"); select_folder("projekt",settings.project_root,"project_root","settings"); select_folder("e-maily",settings.email_root,"email_root","settings"); select_folder("poznámky",settings.notes_root,"notes_root","settings")
    settings.default_llm=st.selectbox("Výchozí jazykový model",["qwen3:8b","qwen3:14b"],index=0 if settings.default_llm=="qwen3:8b" else 1); st.caption(f"Embeddingový model: {ai_search.EMBEDDING_MODEL} (změna vyžaduje reindexaci)"); st.caption("OCR: automaticky aktivní pro skenované dokumenty"); settings.result_count=st.slider("Počet výsledků",8,50,max(8,settings.result_count))
    if st.button("Uložit nastavení",type="primary"): save_settings(SETTINGS,settings); st.success("Nastavení bylo uloženo.")
    c1,c2=st.columns(2)
    if c1.button("Importovat e-maily",disabled=not bool(settings.email_root)):
        if Path(settings.email_root).is_dir(): st.success(f"E-maily zaindexovány: {index_source(Path(settings.email_root),'E-mail',STATE,embeddings())['added']} nových.")
        else: st.error("Složka e-mailů neexistuje. Vyberte ji znovu.")
    if c2.button("Importovat poznámky",disabled=not bool(settings.notes_root)):
        if Path(settings.notes_root).is_dir(): st.success(f"Poznámky zaindexovány: {index_source(Path(settings.notes_root),'Poznámka',STATE,embeddings())['added']} nových.")
        else: st.error("Složka poznámek neexistuje. Vyberte ji znovu.")
    if st.button("Zpět k hledání"): st.session_state.page="search"; st.rerun()
elif st.session_state.page=="history":
    st.header("Historie a diagnostika"); st.write("Ollama:","běží" if ollama_status() else "není dostupná"); st.write("Embeddingový model:",ai_search.EMBEDDING_MODEL); st.write("Výchozí LLM:",settings.default_llm)
    history=read_history(STATE)
    if history: st.dataframe(history,use_container_width=True,hide_index=True)
    else: st.info("Historie je zatím prázdná. Proveďte první indexaci.")
    if st.button("Zpět k hledání"): st.session_state.page="search"; st.rerun()
else:
    if not summary["ready"]:
        st.header("Začněte ve třech krocích")
        # Nativní st.columns/st.container(border=True) namísto vlastních CSS
        # tříd .wizard/.step/.arrow - stejný obsah, méně ad-hoc CSS k údržbě.
        steps=[("1. Vyberte projekt","Pomocí Finderu vlevo v panelu"),("2. Vytvořte index","Jedním tlačítkem v postranním panelu"),("3. Začněte hledat","Dokumenty i AI odpovědi")]
        cols=st.columns([3,.4,3,.4,3])
        for i,(title,detail) in enumerate(steps):
            with cols[i*2].container(border=True): st.markdown(f"**{title}**"); st.caption(detail)
            if i<2: cols[i*2+1].markdown("<div style='text-align:center;padding-top:2.2rem;opacity:.4;font-size:1.4rem'>→</div>",unsafe_allow_html=True)
    st.header("Najděte informace ve svých projektech")
    query=st.text_input("Hledaný výraz nebo otázka",placeholder="Například: Kdy má být dokončena hydroizolace?",key="search_query")
    classification=classify_query(query) if query else {"mode":"dokument","deep":False}
    mode="Položit otázku" if classification["mode"]=="otazka" else "Hledat dokumenty"
    force_deep=st.toggle("Vynutit hloubkovou analýzu",False,help="U komplexních dotazů (požadavky, kontrola, porovnání, shrnutí, rizika) se zapíná automaticky."); deep=classification["deep"] or force_deep
    if query:
        mode_icon="🧠" if mode=="Položit otázku" else "🔤"
        st.markdown(f"<span class='badge badge-blue'>{mode_icon} {mode}</span>"+(" <span class='badge badge-blue'>🧠 hloubková analýza</span>" if deep else ""),unsafe_allow_html=True)
    with st.expander("Filtry"):
        c1,c2,c3=st.columns(3); source=c1.selectbox("Zdroj",["Vše","Dokument","E-mail","Poznámka"]); project_filter=c2.text_input("Projekt"); folder=c3.text_input("Podsložka"); c4,c5,c6=st.columns(3); extension=c4.text_input("Typ souboru"); author=c5.text_input("Autor nebo odesílatel"); date_range=c6.date_input("Datum od–do",value=[])
        if st.button("Zrušit filtry"): st.rerun()
    if query and summary["ready"]:
        if st.session_state.get("last_query")!=query: st.session_state.visible_results=PAGE_SIZE; st.session_state.last_query=query
        started=time.perf_counter()
        with st.spinner("Hledám v dostupných zdrojích…"):
            rows=search_all(query,settings,STATE,embeddings(),is_question=classification["mode"]=="otazka",expand_query=QUERY_EXPANSION_MODE); start=date_range[0] if len(date_range)>0 else None; end=date_range[-1] if len(date_range)>1 else None; rows=apply_filters(rows,source,"Vše" if not project_filter else project_filter,folder,"Vše" if not extension else extension,author,start,end)
        elapsed=time.perf_counter()-started; search_mode="AI odpověď" if mode=="Položit otázku" else "Hybrid (fulltext + význam)"
        st.caption(f"Nalezeno: {len(rows)} dokumentů · čas {elapsed:.2f} s · režim {search_mode}")
        if not rows:
            with st.container(border=True):
                st.warning("Nebyly nalezeny žádné odpovídající výsledky.")
                st.caption("Zkuste upravit klíčová slova, zkontrolovat aktivní filtry, nebo ověřit, že je projekt zaindexovaný.")
        elif mode=="Položit otázku":
            original_default,original_complex=ai_search.DEFAULT_MODEL,ai_search.COMPLEX_MODEL
            try:
                ai_search.DEFAULT_MODEL=settings.default_llm; ai_search.COMPLEX_MODEL=settings.deep_llm if deep else settings.default_llm
                response=ai_search.answer(query,rows)
            finally:
                ai_search.DEFAULT_MODEL,ai_search.COMPLEX_MODEL=original_default,original_complex
            if not response.get("citations"): st.warning("V dostupných zdrojích jsem nenašel dostatek informací pro spolehlivou odpověď.")
            else:
                citations=response["citations"]
                with st.container(border=True):
                    head_l,head_r=st.columns([5,2]); head_l.subheader("Odpověď")
                    badge=confidence_badge_html(response.get("confidence"))
                    if badge: head_r.markdown(f"<div style='text-align:right;padding-top:.5rem'>{badge}</div>",unsafe_allow_html=True)
                    st.write(strip_confidence_footer(response["answer"]))
                    st.caption(sources_caption(len(citations)))
                st.subheader(f"Zdroje ({len(citations)})")
                best_citation_score=max((row.get("score",0) for row in citations),default=0)
                for i,row in enumerate(citations,1):
                    project,_=display_location(row,settings); st.markdown(f"<div class='citation'><b>[{i}] {html.escape(row['document'])}</b><br>📁 {html.escape(project)}{relevance_meta_html(row,best_citation_score)}<br>📝 {highlight(context_excerpt(row.get('quote',''),query),query)}</div>",unsafe_allow_html=True)
                    if st.button("Přejít na citovaný dokument",key=f"citation-{i}",use_container_width=True): st.toast(open_path(Path(row["path"]))[1])
        else:
            left,right=st.columns([3,2],gap="large"); visible=rows[:st.session_state.visible_results]; best=max(row["score"] for row in rows)
            with left:
                st.subheader(f"Výsledky ({len(rows)})")
                for i,row in enumerate(visible): render_result(row,query,i,best)
                if len(visible)<len(rows) and st.button("Zobrazit další výsledky",use_container_width=True): st.session_state.visible_results+=PAGE_SIZE; st.rerun()
            with right:
                if st.session_state.get("preview"): render_preview(st.session_state.preview,query)
                else:
                    with st.container(border=True): st.info("Klikněte na Náhled u výsledku. Zde se zobrazí dokument nebo relevantní text.")
    elif query: st.info("Nejprve vyberte projekt a vytvořte index.")
    else:
        with st.container(border=True):
            st.markdown("🔎 **Zadejte hledaný výraz nebo otázku.** Vše zůstává pouze na tomto Macu.")
            st.caption("Například: „Kdy má být dokončena hydroizolace?“ nebo „změnový list GD“")
