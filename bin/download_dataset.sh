#!/usr/bin/env bash
# Raw sources for GUIDE.md: paired human SMPL and robot G1 motion.
# No SOMA archives or SONIC checkpoints; preprocessing remains separate.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if ! command -v uv >/dev/null 2>&1; then
  echo 'Se requiere uv; ejecuta primero bin/install.sh.' >&2
  exit 1
fi

# Isolated download dependencies; does not change pyproject.toml or uv.lock.
exec uv run --no-project --python 3.13 \
  --with 'huggingface-hub>=1,<2' --with 'python-dotenv>=1,<2' \
  python - "$PROJECT_DIR" "$@" <<'PY'
from contextlib import ExitStack
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

from dotenv import dotenv_values
from huggingface_hub import HfApi, get_token, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

project = Path(sys.argv.pop(1))
parser = argparse.ArgumentParser(
    prog="download_dataset.sh",
    description="Descarga G1 + metadatos + SMPL para la arquitectura de GUIDE.md. "
                "No incluye SOMA ni checkpoints SONIC.",
    epilog="Acepta el acceso en https://huggingface.co/datasets/bones-studio/seed "
           "y configura HF_TOKEN en el entorno, .env.local o .env. "
           "Puedes repetir el comando tras una interrupción.",
)
parser.add_argument("--no-extract", action="store_true", help="Descargar sin extraer.")
parser.add_argument("--dry-run", action="store_true", help="Consultar tamaños sin descargar datos.")
parser.add_argument("--output", type=Path, default=project / "data" / "seed")
parser.add_argument("--revision", default="main", help="Revisión de BONES-SEED.")
parser.add_argument("--smpl-revision", default="main", help="Revisión de GEAR-SONIC.")
parser.add_argument("--workers", type=int, default=4, help="Descargas simultáneas (por defecto: 4).")
args = parser.parse_args()
if args.workers < 1:
    parser.error("--workers debe ser positivo")
# Environment > .env.local > .env > cached login. Ignore empty tokens.
# Parse dotenv files without executing them as shell code.
token = (
    os.environ.get("HF_TOKEN")
    or dotenv_values(project / ".env.local").get("HF_TOKEN")
    or dotenv_values(project / ".env").get("HF_TOKEN")
    or get_token()
)
if not token:
    sys.exit("Falta HF_TOKEN en el entorno, .env.local o .env. Acepta primero "
             "el acceso en https://huggingface.co/datasets/bones-studio/seed.")

destination = args.output.expanduser().resolve()
api = HfApi(token=token)
specs = [
    ("bones-studio/seed", "dataset", destination,
     ["g1.tar.gz", "metadata/*", "LICENSE.md", "README.md"]),
    ("nvidia/GEAR-SONIC", "model", destination / ".sources" / "gear-sonic",
     ["bones_seed_smpl/bones_seed_smpl.tar.part_*", "*LICENSE*", "README.md"]),
]
plans = []
try:
    # Check access to both sources before starting the large downloads.
    for repo, kind, local, patterns in specs:
        api.auth_check(repo_id=repo, repo_type=kind)
        requested = args.revision if kind == "dataset" else args.smpl_revision
        revision = api.repo_info(repo_id=repo, repo_type=kind, revision=requested).sha
        options = dict(repo_id=repo, repo_type=kind, revision=revision,
                       token=token, local_dir=local, allow_patterns=patterns)
        plan = snapshot_download(**options, dry_run=True)
        names = {item.filename for item in plan}
        if kind == "dataset" and "g1.tar.gz" not in names:
            sys.exit("No se encontró g1.tar.gz en BONES-SEED.")
        if kind == "model":
            parts = sorted(n for n in names if n.startswith("bones_seed_smpl/"))
            # Match the seven parts in the official download_from_hf.py.
            expected_parts = [f"bones_seed_smpl/bones_seed_smpl.tar.part_a{c}"
                              for c in "abcdefg"]
            if parts != expected_parts:
                sys.exit("Se esperaban las siete partes SMPL, desde part_aa hasta part_ag. "
                         "Revisa el listado publicado antes de continuar.")
            smpl_parts = [local / n for n in parts]
        pending = sum(item.file_size for item in plan if item.will_download)
        print(f"{repo}: {pending / 1e9:.2f} GB pendientes; revisión {revision}", flush=True)
        plans.append((options, pending))
    if args.dry_run:
        sys.exit(0)
    pending = sum(size for _, size in plans)
    probe = destination
    while not probe.exists():
        probe = probe.parent
    if shutil.disk_usage(probe).free < pending:
        sys.exit("Espacio insuficiente para los archivos de descarga pendientes.")
    print("La extracción y conversión necesitan espacio adicional. "
          "Se conservan los archivos originales para reanudar.", flush=True)
    for options, _ in plans:
        snapshot_download(**options, max_workers=args.workers)
except HfHubHTTPError as exc:
    if getattr(exc.response, "status_code", None) in (401, 403):
        sys.exit("Acceso denegado. Verifica HF_TOKEN y los permisos de lectura "
                 "para bones-studio/seed y nvidia/GEAR-SONIC; acepta las condiciones "
                 "de acceso del dataset en Hugging Face.")
    raise


class JoinedParts:
    """Read a split tar sequentially without creating another ~32 GB file."""

    def __init__(self, handles):
        self.handles = iter(handles)
        self.current = next(self.handles, None)

    def read(self, size):
        chunks = []
        while size > 0 and self.current is not None:
            chunk = self.current.read(size)
            if not chunk:
                self.current = next(self.handles, None)
                continue
            chunks.append(chunk)
            size -= len(chunk)
        return b"".join(chunks)


if not args.no_extract:
    markers = destination / ".extracted"
    markers.mkdir(exist_ok=True)
    archives = [
        ("g1.tar.gz", [destination / "g1.tar.gz"], "r|gz", destination / "g1"),
        ("bones_seed_smpl", smpl_parts, "r|", destination / "smpl_filtered"),
    ]
    for name, paths, mode, expected in archives:
        fingerprint = [dict(name=p.name, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                       for p in paths]
        marker = markers / f"{name}.json"
        if marker.exists() and expected.is_dir():
            if json.loads(marker.read_text()) == fingerprint:
                print(f"Ya extraído: {name}", flush=True)
                continue
        print(f"Extrayendo {name}...", flush=True)
        with ExitStack() as stack:
            stream = JoinedParts([stack.enter_context(p.open("rb")) for p in paths])
            with tarfile.open(fileobj=stream, mode=mode) as tar:
                tar.extractall(destination, filter="data")
        if not expected.is_dir():
            sys.exit(f"La extracción no creó el directorio esperado: {expected}")
        marker.write_text(json.dumps(fingerprint) + "\n")

print(f"Destino G1: {destination / 'g1' / 'csv'}", flush=True)
print(f"Destino SMPL: {destination / 'smpl_filtered'}", flush=True)
print("Pendiente antes de entrenar: verificar pares SMPL–G1, filtrar y convertir "
      "G1 a motion.npz, y construir las features humanas de GUIDE.md.", flush=True)
PY
