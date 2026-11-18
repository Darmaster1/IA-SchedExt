#!/usr/bin/env bash
# Reproduce every number, table and figure in paper/ia_schedext.pdf (about 8 minutes on one CPU core).
set -e
pip install -r requirements.txt
python -m pytest tests -q
python experiments/run_experiments.py        # writes results/*.json  (add --quick for a 1-minute smoke run)
python experiments/analyze.py                # writes paper/tables_generated.tex, paper/figs/*.pdf, results/numbers.json
cd paper && pdflatex ia_schedext.tex && pdflatex ia_schedext.tex
