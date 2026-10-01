#!/usr/bin/env bash
# start.sh — the one-command launcher for Arcturos.
#
#   ./start.sh            install + start the dashboard as a systemd user
#                         service (default; survives logouts/reboots).
#                         On systems without systemd (macOS, containers)
#                         it falls back to running in the foreground.
#   ./start.sh run        force foreground mode (Ctrl+C to stop)
#   ./start.sh remove     stop + uninstall the service
#
# Serves on 0.0.0.0:24816 (LAN-visible). Any OpenAI-compatible inference
# server on your network works as a bench target; point arcturos.local.toml
# (or the Kick off form) at yours after starting.
set -euo pipefail

PORT="${ARCTUROS_PORT:-24816}"
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_NAME="arcturos"

ensure_deps() {
  cd "$APP_DIR"
  if [ ! -d venv ]; then
    echo "→ creating venv (first run)"
    python3 -m venv venv
  fi
  # shellcheck disable=SC1091
  source venv/bin/activate
  if ! python -c "import fastapi" 2>/dev/null; then
    echo "→ installing dependencies (first run)"
    pip install -q -r requirements.txt
  fi
}

have_systemd_user() {
  command -v systemctl >/dev/null 2>&1 &&
    systemctl --user is-system-running >/dev/null 2>&1 || \
    [ "$(systemctl --user is-system-running 2>/dev/null)" = "degraded" ]
}

write_unit() {
  local unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  mkdir -p "$unit_dir"
  cat > "$unit_dir/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=Arcturos dashboard
After=network.target

[Service]
Type=simple
WorkingDirectory=${APP_DIR}
ExecStart=${APP_DIR}/venv/bin/uvicorn arcturos.main:app --host 0.0.0.0 --port ${PORT}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
}

do_run() {
  ensure_deps
  echo "→ Arcturos on http://0.0.0.0:${PORT} (Ctrl+C to stop)"
  exec python -m uvicorn arcturos.main:app --host 0.0.0.0 --port "$PORT"
}

do_install() {
  ensure_deps
  write_unit
  systemctl --user enable --now "${SERVICE_NAME}.service"
  echo "→ Arcturos installed + running as a service"
  echo "   url:    http://0.0.0.0:${PORT}"
  echo "   logs:   journalctl --user -u ${SERVICE_NAME} -f"
  echo "   stop:   ./start.sh remove"
  if command -v loginctl >/dev/null &&
     [ "$(loginctl show-user "$USER" --property=Linger 2>/dev/null)" != "Linger=1" ]; then
    echo "   note: to auto-start after reboot run: sudo loginctl enable-linger \$USER"
  fi
}

do_remove() {
  systemctl --user disable --now "${SERVICE_NAME}.service" 2>/dev/null || true
  rm -f "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/${SERVICE_NAME}.service"
  systemctl --user daemon-reload 2>/dev/null || true
  echo "→ service removed"
}

case "${1:-install}" in
  run)     do_run ;;
  install) if have_systemd_user; then do_install; else
             echo "→ systemd user session unavailable — running in the foreground"
             do_run
           fi ;;
  remove)  do_remove ;;
  *) echo "usage: $0 [install|run|remove]"; exit 2 ;;
esac