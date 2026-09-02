#!/usr/bin/env bash
# codesearch — grep/rg wrapper that ALWAYS skips binary/vendor/irrelevant dirs.
# Recursively searching $HOME with plain grep hammers the HDD through
# gigabytes of binary files (models/, venvs/, learning/puzzle-lab/, node_modules/).
# Use this instead of `grep -rIn ... .` so the machine doesn't crawl.
#
# Usage: codesearch.sh <pattern> [extra rg flags...]
#   codesearch.sh "72\.39\.5\.171"
#   codesearch.sh "some-ip-or-token" -l
set -euo pipefail

PAT="${1:?usage: codesearch.sh <pattern> [rg flags...]}"
shift

EXCLUDES=(
  --glob '!models/**'
  --glob '!venvs/**'
  --glob '!venv/**'
  --glob '!*.venv/**'
  --glob '!site-packages/**'
  --glob '!dist-packages/**'
  --glob '!learning/puzzle-lab/**'
  --glob '!tools/other_llms/**'
  --glob '!tools/camera/**'
  --glob '!node_modules/**'
  --glob '!sessions/**'
  --glob '!.pi/agent/sessions/**'
  --glob '!learning/freeroam/**'
  --glob '!archive/**'
  --glob '!tools/communications/email/inbox/**'
  --glob '!tools/communications/sms/inbox/**'
  --glob '!library/**'
  --glob '!recovery/docs/**'
  --glob '!.cache/**'
  --glob '!.npm/**'
  --glob '!.local/lib/**'
  --glob '!*.gguf'
  --glob '!*.bin'
  --glob '!*.so'
  --glob '!*.safetensors'
  --glob '!*.pt'
  --glob '!*.onnx'
  --glob '!*.wasm'
  --glob '!*.pyc'
  --glob '!.git/**'
)

# Default root: $HOME unless a path is given as the pattern scope.
ROOT="${ROOT_DIR:-$HOME}"

# rg auto-respects .gitignore and binary detection; the globs above are belt-and-suspenders
# for the big binary trees that aren't git-tracked.
rg --smart-case --hidden "${EXCLUDES[@]}" "$@" -n "$PAT" "$ROOT"
