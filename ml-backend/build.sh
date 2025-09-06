#!/usr/bin/env bash
set -euo pipefail

CMD=${1:-help}

case "$CMD" in
	build)
		docker compose build ;;
	up)
		docker compose up -d ;;
	down)
		docker compose down ;;
	logs)
		docker compose logs -f ;;
	restart)
		docker compose down && docker compose up -d ;;
	*)
		echo "Usage: $0 {build|up|down|logs|restart}" ;;
esac

