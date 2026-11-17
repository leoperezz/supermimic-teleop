#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
exec uv run python - "$@" <<'PY'
import argparse
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--package-only', action='store_true')
parser.add_argument('--hub-output', type=Path)
parser.add_argument('--clips-per-shard', type=int, default=1000)
options, remaining = parser.parse_known_args()
source_parser = argparse.ArgumentParser(add_help=False)
source_parser.add_argument('--output', type=Path, default=Path('data/seed-processed'))
source, _ = source_parser.parse_known_args(remaining)
if options.clips_per_shard < 1:
    parser.error('--clips-per-shard must be positive')
if not options.package_only:
    subprocess.run([sys.executable, '-m', 'supermimic.data.prepare_seed', *remaining], check=True)
    if '--help' in remaining or '-h' in remaining:
        sys.exit(0)
destination = options.hub_output or source.output.with_name(source.output.name + '-hub')
subprocess.run([sys.executable, 'bin/package_dataset.py', '--input', str(source.output),
                '--output', str(destination), '--clips-per-shard', str(options.clips_per_shard)], check=True)
PY
