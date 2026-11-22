#!/bin/zsh
# Weekly insider cadence (Monday 09:00 via launchd):
# 1) drop a dated Grok brief into prompts/weekly/
# 2) notify via macOS notification
# 3) run the daily watcher once (catch-up + fresh-week snapshot)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TODAY=$(date +%F)
MONDAY=$(date -v-monday +%F 2>/dev/null || date +%F)
mkdir -p "$ROOT/prompts/weekly"
OUT="$ROOT/prompts/weekly/grok_${TODAY}.md"
if [ ! -f "$OUT" ]; then
  sed "s/<WEEK_START>/${MONDAY}/; s/<WEEK_END>/${TODAY}/" \
    "$ROOT/prompts/grok_weekly.md" > "$OUT"
fi
osascript -e "display notification \"weekly brief: $OUT\" with title \"insider monitor\"" >/dev/null 2>&1
"${PYTHON:-python3}" "$ROOT/scripts/insider_watch.py" --ntfy
