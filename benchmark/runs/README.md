Generated run artifacts (`<timestamp>_<gitsha>_<environment>.json`) are written
here by `python -m benchmark run`. They are gitignored (see `.gitignore`) -
production runs can embed real Box document/folder names, so they must stay
local. Keep whichever runs you want to compare against later (e.g. copy one
aside as `baseline_production.json` before a risky change) and pass their
paths to `python -m benchmark compare BASELINE.json CURRENT.json`.
