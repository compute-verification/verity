#!/bin/bash
# /usr/local/lib/vy-lean-cache/run-daily: vy-lean-cache-daily.service's command on vy-nebius-1, as root. A stub until lean's
# publisher replaces it: restore the dependencies outside the sandbox, `vy-lean-sandbox checkout` main's SHA, `run` the build,
# `clean`, and only then read /etc/verity/lean-cache.env and upload.
echo "vy-lean-cache: publisher not installed (lean's PR)" >&2
exit 2
