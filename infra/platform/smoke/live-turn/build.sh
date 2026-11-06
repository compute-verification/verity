#!/usr/bin/env bash
# Bundles the live-turn smoke from the platform's own sources into
# smoke/live-turn/api/turn.mjs; packages stay external and Vercel installs them
# from this directory's package.json. Deploy the directory as a project of its
# own (never one of the platform's): `vercel deploy smoke/live-turn`, with
# SMOKE_KEY_SHA256 set to the SHA-256 hex of a key, and call /api/turn?key=KEY.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here/../.."
npx esbuild smoke/live-turn/turn.ts --bundle --platform=node --format=esm --target=node24 \
  --packages=external --outfile=smoke/live-turn/api/turn.mjs
