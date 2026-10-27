#!/bin/bash
# LocalLlmBench launcher.
#   ./run.sh                 run in this window (waits for night + idle Mac)
#   ./run.sh --background    same, but detached; safe to close Terminal
#   ./run.sh --smoke         2-minute check on the smallest model
#   ./run.sh --now           start right away, ignore night/idle rules
#   ./run.sh --report        rebuild the report only
cd "$(dirname "$0")" || exit 1
mkdir -p data
PY=""
for c in /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 /opt/homebrew/bin/python3 /usr/local/bin/python3 "$HOME/.local/bin/python3.11" python3; do
  p="$(command -v "$c" 2>/dev/null)" || continue
  if [ "$p" = "/usr/bin/python3" ] && ! xcode-select -p >/dev/null 2>&1; then continue; fi
  PY="$p"; break
done
if [ -z "$PY" ]; then echo "No usable python3 found"; exit 1; fi
if [ "$1" = "--background" ]; then
  shift
  nohup caffeinate -ims "$PY" bench.py "$@" >> data/console.log 2>&1 &
  echo $! > data/bench.pid
  echo "LocalLlmBench started in background (PID $!). Log: $(pwd)/data/run.log"
  exit 0
fi
exec caffeinate -ims "$PY" bench.py "$@"
