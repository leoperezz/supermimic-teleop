#!/usr/bin/env bash
# Usage: bash bin/install.sh [uv sync options, e.g. --no-dev]
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MJLAB_DIR="$(dirname -- "$PROJECT_DIR")/mjlab_research"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
Uso: bash bin/install.sh [opciones de uv sync]

Instala uv si falta, prepara Python 3.13, clona mjlab_research si no
existe e instala SuperMimic y sus dependencias en .venv usando uv.lock.
Para descargar los datos, ejecuta bin/download_dataset.sh por separado.
Incluye las dependencias de desarrollo por defecto.

Ejemplo: bash bin/install.sh --no-dev
HELP
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  # A previous installation may not yet be on this terminal's PATH.
  if [[ -x "$HOME/.local/bin/uv" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    command -v curl >/dev/null 2>&1 || {
      echo 'Falta curl; instálalo para poder descargar uv.' >&2
      exit 1
    }
    echo 'Instalando uv...'
    curl --fail --silent --show-error --location https://astral.sh/uv/install.sh \
      | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

if [[ ! -e "$MJLAB_DIR" ]]; then
  command -v git >/dev/null 2>&1 || {
    echo 'Falta git; instálalo para poder clonar mjlab_research.' >&2
    exit 1
  }
  echo "Clonando mjlab_research en $MJLAB_DIR..."
  git clone https://github.com/HybridRobotics/mjlab_research.git "$MJLAB_DIR"
fi

if [[ ! -f "$MJLAB_DIR/pyproject.toml" ]]; then
  echo "No se encontró un proyecto mjlab válido en $MJLAB_DIR." >&2
  exit 1
fi

cd -- "$PROJECT_DIR"
echo 'Preparando Python 3.13 e instalando SuperMimic...'
uv python install 3.13
uv sync --python 3.13 --locked "$@"


printf '\nInstalación completada en %s/.venv\n' "$PROJECT_DIR"
printf 'Puedes ejecutar comandos desde %s con uv run.\n' "$PROJECT_DIR"
