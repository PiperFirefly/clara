#!/usr/bin/env bash
# typecheck.sh — run mypy on the correctness-critical modules.
# Part of the gradual-typing effort: catches wrong column/key access, None vs
# value slips, and variable-shadowing bugs at CI time instead of at runtime.
# Legacy memstore.py's argparse CLI is excluded from failure (suppressed via the
# per-file directive + benign empty-container annotations), so a clean run here
# means the modules that hold the vault/memory/emotion logic type-check.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venvs/memory/bin/python
FILES=(mailtool/operator_affect.py mailtool/logvault.py memory/person_model.py \
       memory/prediction.py memory/caliber.py)
"$PY" -m mypy "${FILES[@]}"
echo "typecheck: clean"
