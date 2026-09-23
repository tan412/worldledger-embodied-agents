#!/bin/bash
# organoid_service 启停（服务器 ~/organoid_service/ 下运行）: ./start.sh [start|stop|restart|status|log]
set -u
cd "$(dirname "$0")"
PORT="${ORGANOID_PORT:-8663}"
PAT="uvicorn app:app --host 0.0.0.0 --port $PORT"

do_stop() { pkill -f "$PAT" 2>/dev/null && echo "stopped" || echo "not running"; }
do_start() {
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  # Blender 不在 PATH 时自动探测 ~/apps 下的解压版
  if [ -z "${ORGANOID_BLENDER:-}" ]; then
    ORGANOID_BLENDER="$(ls -d ~/apps/blender/blender-*-linux-x64/blender 2>/dev/null | head -1 || true)"
  fi
  [ -n "${ORGANOID_BLENDER:-}" ] && export ORGANOID_BLENDER
  nohup ./venv/bin/uvicorn app:app --host 0.0.0.0 --port "$PORT" >> service.log 2>&1 &
  sleep 1
  pgrep -f "$PAT" >/dev/null && echo "started :$PORT" || { echo "FAILED, tail of service.log:"; tail -5 service.log; exit 1; }
}

case "${1:-start}" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_stop; sleep 1; do_start ;;
  status)  pgrep -fl "$PAT" || echo "not running" ;;
  log)     tail -30 service.log ;;
  *) echo "usage: $0 [start|stop|restart|status|log]"; exit 1 ;;
esac
