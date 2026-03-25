#!/bin/bash
# Pre-build heavyweight resources locally to avoid Docker OOM.
# The INDRA bio ontology build requires ~7-8 GB RAM, which exceeds
# Docker Desktop's typical memory allocation on 16 GB machines.
#
# Usage: ./docker/prebuild.sh
# Then:  docker compose up --build -d

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
RESOURCES_DIR="$SCRIPT_DIR/resources"

# Locate Python — prefer the project venv, fall back to PATH
if [ -x "$REPO_DIR/.venv/bin/python" ]; then
    PYTHON="$REPO_DIR/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "Error: No Python found. Create a venv first: python3 -m venv .venv && pip install -e ." >&2
    exit 1
fi

echo "Using Python: $PYTHON"
echo "Resources dir: $RESOURCES_DIR"

# Verify indra is importable
$PYTHON -c "import indra" 2>/dev/null || {
    echo "Error: indra not installed. Run: pip install -e . (in the project venv)" >&2
    exit 1
}

# Get package versions to match directory structure
PROTMAPPER_VERSION=$($PYTHON -c "import protmapper; print(protmapper.__version__)")
INDRA_ONT_VERSION=$($PYTHON -c "from indra.ontology.bio import bio_ontology; print(bio_ontology.version)")
GILDA_VERSION=$($PYTHON -c "import gilda; print(gilda.__version__)")
ADEFT_VERSION=$($PYTHON -c "import adeft; print(adeft.__version__)")
# adeft uses appdirs.user_data_dir() — platform-dependent
ADEFT_LOCAL=$($PYTHON -c "from adeft.locations import ADEFT_HOME; print(ADEFT_HOME)")
ONTOLOGY_LOCAL=$($PYTHON -c "from indra.ontology.bio.ontology import CACHE_DIR; print(CACHE_DIR)")
GILDA_LOCAL=$($PYTHON -c "from gilda.resources import resource_dir; print(resource_dir)")

echo "Versions: protmapper=$PROTMAPPER_VERSION, indra_ontology=$INDRA_ONT_VERSION, gilda=$GILDA_VERSION, adeft=$ADEFT_VERSION"
echo "Adeft local path: $ADEFT_LOCAL"
echo "INDRA ontology local path: $ONTOLOGY_LOCAL"
echo "Gilda local path: $GILDA_LOCAL"

mkdir -p "$RESOURCES_DIR"

# --- protmapper resources ---
PROTMAPPER_SRC=$($PYTHON -c "from protmapper.resources import resource_dir; print(resource_dir)")
PROTMAPPER_DST="$RESOURCES_DIR/protmapper/$PROTMAPPER_VERSION"
if [ -d "$PROTMAPPER_SRC" ] && [ "$(ls -A "$PROTMAPPER_SRC")" ]; then
    echo "Copying protmapper resources from cache..."
    mkdir -p "$PROTMAPPER_DST"
    cp -a "$PROTMAPPER_SRC/"* "$PROTMAPPER_DST/"
else
    echo "Building protmapper resources (downloads ~130 MB)..."
    $PYTHON -m protmapper.resources
    mkdir -p "$PROTMAPPER_DST"
    cp -a "$HOME/.data/protmapper/$PROTMAPPER_VERSION/"* "$PROTMAPPER_DST/"
fi

# --- INDRA bio ontology (the big one: ~470 MB pickle, ~8 GB RAM to build) + SQLite DB ---
ONTOLOGY_SRC="$ONTOLOGY_LOCAL/bio_ontology.pkl"
ONTOLOGY_SQLITE_SRC="$ONTOLOGY_LOCAL/bio_ontology.db"
ONTOLOGY_DST="$RESOURCES_DIR/indra_ontology/$INDRA_ONT_VERSION"
if [ -f "$ONTOLOGY_SRC" ] && [ -f "$ONTOLOGY_SQLITE_SRC" ]; then
    echo "Copying INDRA bio ontology from cache (470 MB)..."
    mkdir -p "$ONTOLOGY_DST"
    cp "$ONTOLOGY_SRC" "$ONTOLOGY_DST/bio_ontology.pkl"
    cp "$ONTOLOGY_SQLITE_SRC" "$ONTOLOGY_DST/bio_ontology.db"
else
    echo "Building INDRA bio ontology (requires ~8 GB RAM, takes ~45 min)..."
    $PYTHON -m indra.ontology.bio build
    mkdir -p "$ONTOLOGY_DST"
    cp "$ONTOLOGY_LOCAL/bio_ontology.pkl" "$ONTOLOGY_DST/bio_ontology.pkl"
    cp "$ONTOLOGY_LOCAL/bio_ontology.db" "$ONTOLOGY_DST/bio_ontology.db"
fi

# --- adeft models ---
# adeft uses appdirs: macOS=~/Library/Application Support/adeft, Linux=/root/.local/share/adeft
ADEFT_SRC="$ADEFT_LOCAL/$ADEFT_VERSION"
ADEFT_DST="$RESOURCES_DIR/adeft/$ADEFT_VERSION"
if [ -d "$ADEFT_SRC" ] && [ "$(find "$ADEFT_SRC" -type f 2>/dev/null | head -1)" ]; then
    echo "Copying adeft models from cache..."
    mkdir -p "$ADEFT_DST"
    cp -a "$ADEFT_SRC/"* "$ADEFT_DST/"
else
    echo "Downloading adeft models..."
    $PYTHON -m adeft.download
    mkdir -p "$ADEFT_DST"
    cp -a "$ADEFT_SRC/"* "$ADEFT_DST/"
fi

# --- gilda grounding terms + Gilda SQLite DB ---
GILDA_DST="$RESOURCES_DIR/gilda/$GILDA_VERSION"
mkdir -p "$GILDA_DST"

# Download terms + models if not cached
if [ -f "$GILDA_LOCAL/grounding_terms.tsv.gz" ]; then
    echo "Copying gilda resources from cache..."
    cp -a "$GILDA_LOCAL/"* "$GILDA_DST/"
else
    echo "Downloading gilda resources..."
    $PYTHON -m gilda.resources
    cp -a "$GILDA_LOCAL/"* "$GILDA_DST/"
fi

# Build Gilda SQLite DB if missing
if [ ! -f "$GILDA_DST/grounding_terms.db" ]; then
    echo "Building gilda SQLite database..."
    $PYTHON -m gilda.resources.sqlite_adapter
    cp "$GILDA_LOCAL/grounding_terms.db" "$GILDA_DST/grounding_terms.db"
fi

echo ""
echo "Pre-built resources:"
du -sh "$RESOURCES_DIR"/*
echo ""
echo "Done. Now run: docker compose up --build -d"
