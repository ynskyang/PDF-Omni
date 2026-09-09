#!/usr/bin/env bash
# Check that the OmniMVS datasets are laid out the way dataset.py expects and
# optionally symlink them under --db_root.
# Usage: bash scripts/prepare_data.sh <path_to_downloaded_omnidata> [db_root]
set -euo pipefail
SRC="${1:?usage: prepare_data.sh <src_dir> [db_root]}"
DB_ROOT="${2:-$(cd "$(dirname "$0")/.." && pwd)/../omnidata}"
DATASETS=(omnithings omnihouse sunny cloudy sunset)

mkdir -p "$DB_ROOT"
status=0
for d in "${DATASETS[@]}"; do
    src="$SRC/$d"
    if [ ! -d "$src" ]; then
        echo "[data] missing: $src" >&2
        status=1
        continue
    fi
    for req in config.yaml cam1 cam2 cam3 cam4; do
        [ -e "$src/$req" ] || { echo "[data] $d: missing $req" >&2; status=1; }
    done
    if [ ! -e "$DB_ROOT/$d" ]; then
        ln -s "$(cd "$src" && pwd)" "$DB_ROOT/$d"
        echo "[data] linked $DB_ROOT/$d -> $src"
    else
        echo "[data] $DB_ROOT/$d already exists"
    fi
    n_gt=$(find "$src" -maxdepth 1 -type d -name 'omnidepth_gt_*' | wc -l)
    echo "[data] $d: $(ls "$src/cam1" | wc -l) frames, $n_gt ground-truth folder(s)"
done
echo "[data] lookup tables are built inside each dataset folder on the first run"
exit $status
