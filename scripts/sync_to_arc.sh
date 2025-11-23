#!/bin/bash
set -euo pipefail

# ---- local source ----
SRC_BASE="$HOME/Projects/FlowIntegrator"

# ---- ARC destination ----
DEST_ALIAS="arc-htc"              # alias from ~/.ssh/config
DEST_HOME="/home/phys1997/FlowIntegrator"
DEST_DATA="/data/phys-galsim/phys1997/FlowIntegrator"

usage() {
    echo "Usage: $0 [results|data] [--delete]"
    echo "  --delete : show and remove files on remote not present locally"
    exit 1
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
fi

TARGET=$1
DELETE_FLAG=${2:-}

do_rsync() {
    local src=$1
    local dest=$2
    local desc=$3

    ssh "$DEST_ALIAS" "mkdir -p '$dest'"

    if [[ "$DELETE_FLAG" == "--delete" ]]; then
        echo "[INFO] Showing files that would be deleted from ${DEST_ALIAS}:${dest}:"
        rsync -avh --dry-run --delete --itemize-changes -e "ssh" "$src/" "$DEST_ALIAS:$dest/"
        echo
        read -p "Proceed with deleting the above files? [y/N] " answer
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            echo "[INFO] Syncing with deletions enabled"
            rsync -avh --progress --delete -e "ssh" "$src/" "$DEST_ALIAS:$dest/"
        else
            echo "[INFO] Aborted by user."
            exit 0
        fi
    else
        echo "[INFO] Syncing without deletions"
        rsync -avh --progress -e "ssh" "$src/" "$DEST_ALIAS:$dest/"
    fi
}

case "$TARGET" in
    results)
        do_rsync "$SRC_BASE/results" "$DEST_DATA/results" "results"
        ;;
    data)
        do_rsync "$SRC_BASE/data" "$DEST_DATA/data" "data"
        ;;
    *)
        usage
        ;;
esac

echo "[INFO] Sync complete."
