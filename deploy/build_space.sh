#!/usr/bin/env bash
# Assemble the flat HuggingFace Space directory from the repo.
# The src/ modules are copied in (not duplicated in git) so the Space is self-contained.
# Usage: bash deploy/build_space.sh   (run from the repo root: domaingpt/)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$REPO_ROOT/deploy/_space"

rm -rf "$STAGE"
mkdir -p "$STAGE"

cp "$REPO_ROOT/deploy/app.py"          "$STAGE/app.py"
cp "$REPO_ROOT/deploy/requirements.txt" "$STAGE/requirements.txt"
cp "$REPO_ROOT/deploy/README.md"       "$STAGE/README.md"
cp "$REPO_ROOT/src/retrieval.py"       "$STAGE/retrieval.py"
cp "$REPO_ROOT/src/generate.py"        "$STAGE/generate.py"

echo "Space staged at: $STAGE"
echo "Contents:"
ls -1 "$STAGE"
