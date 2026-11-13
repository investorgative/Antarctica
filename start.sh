#!/usr/bin/env bash
# Starts ANTARCTICA: paper trader (tmux "Antarctica"), dashboard on :8760 ("AntarcticaWeb"),
# free Cloudflare quick tunnel ("AntarcticaCloud"). Every process restarts itself if it crashes.
cd "$(dirname "$0")"
DIR="$(pwd)"
PY="$DIR/venv/bin/python"
[ -x "$PY" ] || { echo "venv missing - run ./setup.sh first"; exit 1; }
PORT=$("$PY" -c "import antarctica_core as c;print(c.load_config()['web'].get('port',8760))")
mkdir -p "$DIR/data/logs"
if ! tmux has-session -t Antarctica 2>/dev/null; then
  tmux new-session -d -s Antarctica "cd '$DIR' && while true; do '$PY' antarctica_trader.py; echo 'trader exited - restarting in 10s'; sleep 10; done"
  echo "started tmux session Antarctica       (paper trader)"
else echo "tmux session Antarctica already running"; fi
if ! tmux has-session -t AntarcticaWeb 2>/dev/null; then
  tmux new-session -d -s AntarcticaWeb "cd '$DIR' && while true; do '$PY' antarctica_web.py; echo 'web exited - restarting in 10s'; sleep 10; done"
  echo "started tmux session AntarcticaWeb    (dashboard http://localhost:$PORT)"
else echo "tmux session AntarcticaWeb already running"; fi
if command -v cloudflared >/dev/null 2>&1; then
  if ! tmux has-session -t AntarcticaCloud 2>/dev/null; then
    : > "$DIR/data/logs/cloudflared.log"
    tmux new-session -d -s AntarcticaCloud "cd '$DIR' && while true; do cloudflared tunnel --no-autoupdate --url http://localhost:$PORT 2>&1 | tee -a data/logs/cloudflared.log; echo 'tunnel exited - restarting in 10s'; sleep 10; done"
    echo "started tmux session AntarcticaCloud  (quick tunnel - URL appears in a few seconds)"
    sleep 8
    URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$DIR/data/logs/cloudflared.log" | tail -1)
    [ -n "$URL" ] && echo "public URL: $URL" || echo "public URL: run 'tmux attach -t AntarcticaCloud' or click the PUBLIC chip in the dashboard header"
  else echo "tmux session AntarcticaCloud already running"; fi
else
  echo "cloudflared not installed - tunnel not started (dashboard still works on your network)"
fi
