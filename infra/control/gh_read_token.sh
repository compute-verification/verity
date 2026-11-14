#!/bin/sh
# Print a read-only GitHub App installation token, the one every reader of GitHub on the pod takes (the sweep, the agent watch,
# the cloud watch and the watchers' gh, bin/gh): gh_app_token.sh's exchange (a JWT signed RS256 with the App's private key,
# exchanged at GitHub for a 1-hour token scoped to compute-verification/verity and compute-verification/infra), asked for with
# contents and pull requests read and nothing else, so nothing that holds it can write. Its only output on stdout is the token,
# for `GH_TOKEN=$(gh_read_token.sh --cached)`. Exit 1, with GitHub's message on stderr, when the exchange fails.
#   gh_read_token.sh --cached   the token minted within the last 45 min, else a new one: kept in $GH_READ_CACHE (default
#                               /dev/shm/verity-read-token, memory only, root 0600), a file of its own: a reader is never
#                               handed gh_app_token.sh's write token, nor a writer this one
#   GH_APP_ID (default verity-drive's 5385601), GH_APP_INSTALLATION (173153984), GH_APP_KEY (/data/state/secrets/verity-drive.pem)
app=${GH_APP_ID:-5385601} inst=${GH_APP_INSTALLATION:-173153984} key=${GH_APP_KEY:-/data/state/secrets/verity-drive.pem}
cache=${GH_READ_CACHE:-/dev/shm/verity-read-token}
if [ "${1:-}" = --cached ]; then
  if [ -s "$cache" ] && [ $(( $(date +%s) - $(stat -c %Y "$cache") )) -lt 2700 ]; then cat "$cache"; exit 0; fi
  t=$("$0") || exit 1
  (umask 077; printf '%s\n' "$t" > "$cache.$$") && mv "$cache.$$" "$cache"
  printf '%s\n' "$t"
  exit 0
fi
b64() { openssl base64 -A | tr '+/' '-_' | tr -d '='; }
now=$(date +%s)
head=$(printf '{"alg":"RS256","typ":"JWT"}' | b64)
body=$(printf '{"iat":%d,"exp":%d,"iss":"%s"}' $((now - 60)) $((now + 540)) "$app" | b64)
sig=$(printf '%s.%s' "$head" "$body" | openssl dgst -sha256 -sign "$key" -binary | b64) || exit 1
curl -sS --max-time 30 -X POST -H "Authorization: Bearer $head.$body.$sig" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/app/installations/$inst/access_tokens" \
  -d '{"repositories":["verity","infra"],"permissions":{"contents":"read","pull_requests":"read"}}' |
  python3.12 -c '
import json, sys
d = json.load(sys.stdin)
if "token" not in d:
    sys.exit("gh_read_token: GitHub refused: %s" % d.get("message", d))
print(d["token"])'
