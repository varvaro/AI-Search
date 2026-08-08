Checked-in baselines that `python -m benchmark.run_retrieval_regression`
compares every run against.

Unlike `benchmark/runs/` (gitignored), these files ARE tracked - a baseline is
only useful if it travels with the code it describes. They therefore go through
`run_retrieval_regression.baseline_payload()`, which strips each case's
`top_paths`: the comparison needs only rank/recall/MRR/nDCG/hit/error, and
committing 10 absolute Box paths per case would leak real document and folder
names into the repository. The full paths stay in the local run artifact.

Refresh a baseline only when the new numbers are the ones you want to defend:

    python -m benchmark.run_retrieval_regression --update-baseline

Refreshing hides whatever moved since the last one, so do it deliberately -
after a change you have already reviewed, not as a way to clear a red report.
Each baseline records the `index_fingerprint` it was measured against;
the runner warns when the current index differs, because rank changes then may
come from the index rather than from code.
