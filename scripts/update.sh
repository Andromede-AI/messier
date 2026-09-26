#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv run --group dev python -c 'from messier import DatasetBuilder; DatasetBuilder().build()'
uv run --group dev pytest -q --source=local

# if changes to raw inputs or a benchmark builder produce new or changed tasks, update their classification labels and rebuild
# uv run --group dev python -m messier.classify
# uv run --group dev python -c 'from messier import DatasetBuilder; DatasetBuilder().build()'
