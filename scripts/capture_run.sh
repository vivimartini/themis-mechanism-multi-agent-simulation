#!/usr/bin/env bash
# Rebuild the dissertation artifacts and capture a portable provenance record.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p reference_outputs

START_EPOCH="$(date +%s)"
START_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

if GIT_SHA="$(git rev-parse HEAD 2>/dev/null)"; then
  :
else
  GIT_SHA="UNCOMMITTED"
fi
if [[ -n "$(git status --porcelain 2>/dev/null || true)" ]]; then
  GIT_STATE="dirty"
else
  GIT_STATE="clean"
fi

./run_all.sh 2>&1 \
  | sed "s|$ROOT/||g" \
  | tee reference_outputs/full_run.txt

END_EPOCH="$(date +%s)"
DURATION="$((END_EPOCH - START_EPOCH))"
PY="${PY:-.venv/bin/python}"

{
  printf 'Dissertation run provenance\n'
  printf '============================\n'
  printf 'started_utc: %s\n' "$START_UTC"
  printf 'duration_seconds: %s\n' "$DURATION"
  printf 'git_sha: %s\n' "$GIT_SHA"
  printf 'git_state: %s\n' "$GIT_STATE"
  printf 'platform: %s %s %s\n' "$(uname -s)" "$(uname -r)" "$(uname -m)"
  printf 'python: '
  "$PY" --version
  printf '\nPinned environment\n'
  printf '%s\n' '------------------'
  "$PY" -m pip freeze
  printf '\nSHA-256 artifacts\n'
  printf '%s\n' '-----------------'
} > reference_outputs/run_provenance.txt

"$PY" - <<'PY' >> reference_outputs/run_provenance.txt
import hashlib
from pathlib import Path

root = Path.cwd()
paths = [
    root / "requirements.txt",
    root / "data" / "actors_baseline.csv",
    root / "data" / "headlines.json",
    root / "reference_outputs" / "experiment_manifest.md",
    root / "reference_outputs" / "full_run.txt",
]
paths += sorted((root / "data").glob("*.npz"))
paths += sorted((root / "figures").glob("*.pdf"))
paths += sorted((root / "figures").glob("*.png"))
for path in paths:
    if path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{digest}  {path.relative_to(root)}")
PY

printf '\nWrote reference_outputs/full_run.txt and run_provenance.txt\n'
