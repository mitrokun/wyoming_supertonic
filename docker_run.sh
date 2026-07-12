#!/usr/bin/env bash
# Thin entrypoint around `python -m wyoming_supertonic`.
#
# Keeping this in a script (rather than a long CMD/command list in the
# Dockerfile or compose file) means:
#   - the Wyoming URI always has a sane default
#   - any extra flags passed to `docker run` / compose `command:` are
#     forwarded as-is (--language, --speed, --steps, --threads,
#     --no-streaming, --crop-silence, --debug, --extra ru, ...)
set -euo pipefail

WYOMING_URI="${WYOMING_URI:-tcp://0.0.0.0:10209}"

exec python3 -m wyoming_supertonic \
    --uri "${WYOMING_URI}" \
    --data-dir /data/models \
    "$@"
