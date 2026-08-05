#!/bin/zsh
cd "/Users/miroslavvarvarovsky/Documents/AI Search" || { osascript -e 'display dialog "Složka AI Search nebyla nalezena." buttons {"OK"}'; exit 1; }
if [[ ! -x .venv/bin/streamlit ]]; then
  osascript -e 'display dialog "Chybí projektové prostředí. V Terminálu spusťte instalaci podle README." buttons {"OK"}'
  exit 1
fi
exec .venv/bin/streamlit run app.py --server.headless=false
