#!/usr/bin/env bash
set -euo pipefail

count=0
while IFS= read -r lang_file; do
    echo "[sankey] $lang_file"
    python lang_analysis/sankey_lang_plot.py --lang-path "$lang_file"
    count=$((count + 1))
done < <(find artifacts -type f -path '*/lang/languatory.json' | sort)

echo "[sankey] Done: $count diagrams generated."
