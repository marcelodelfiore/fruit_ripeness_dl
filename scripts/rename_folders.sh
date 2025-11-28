#!/bin/bash

# This script helps renaming folders with spaces in their names, as
# originally downloaded from Kaggle

# Usage: ./rename_folders.sh [directory_path] [run]

# 1. Select target directory (defaults to current folder if not provided)
TARGET_DIR="${1:-.}"

# 2. Check execution mode
if [ "$2" == "run" ]; then
    DRY_RUN=false
    echo "--- EXECUTING RENAMES ---"
else
    DRY_RUN=true
    echo "--- DRY RUN (No changes will be made) ---"
    echo "To actually rename files, run: ./rename_folders.sh <directory> run"
fi

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    exit 1
fi

# Counter for changes
count=0

# Loop through files in the target directory
# The glob pattern "* *" ensures we only look at items with spaces
find "$TARGET_DIR" -maxdepth 1 -type d -name "* *" | while read -r filepath; do

    # Extract directory name and parent path
    dirname=$(basename "$filepath")
    parentdir=$(dirname "$filepath")

    # Create new name: Replace all spaces with underscores
    newname="${dirname// /_}"

    # Construct full new path
    newpath="$parentdir/$newname"

    # check if destination already exists to prevent overwriting
    if [ -e "$newpath" ]; then
        echo "[SKIP] Target '$newname' already exists. Cannot rename '$dirname'."
        continue
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "[WOULD RENAME] '$dirname'  ->  '$newname'"
    else
        mv "$filepath" "$newpath"
        echo "[RENAMED] '$dirname'  ->  '$newname'"
    fi

    ((count++))
done

if [ "$count" -eq 0 ]; then
    echo "No folders with spaces found."
else
    echo "-----------------------------------"
    if [ "$DRY_RUN" = true ]; then
        echo "Found $count folder(s) to rename."
    else
        echo "Successfully renamed $count folder(s)."
    fi
fi
