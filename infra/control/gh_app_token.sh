#!/bin/sh
# Print a GitHub App installation token (INF-133): a JWT signed RS256 with the App's private key (openssl, no Python crypto on the
# pod), exchanged at GitHub for a 1-hour token scoped to compute-verification/verity with contents and pull requests write.
# Its only output on stdout is the token, for `GH_TOKEN=$(gh_app_token.sh)`. Exit 1, with GitHub's message on stderr, when the
# exchange fails.
#   gh_app_token.sh --cached   the token minted within the last 45 min, else a new one: kept in $GH_APP_CACHE (default
#                              /dev/shm/verity-drive-token, memory only, root 0600), so a 90-min drive tick never holds one expired
#   gh_app_token.sh git get    a git credential helper (`credential.helper=!/data/infra/gh_app_token.sh git`): username and the
#                              --cached token, read when git needs it rather than once a tick; store and erase do nothing
#   GH_APP_ID (default verity-drive's 5385601), GH_APP_INSTALLATION (173153984), GH_APP_KEY (/data/state/secrets/verity-drive.pem)
app=${GH_APP_ID:-5385601} inst=${GH_APP_INSTALLATION:-173153984} key=${GH_APP_KEY:-/data/state/secrets/verity-drive.pem}
cache=${GH_APP_CACHE:-/dev/shm/verity-drive-token}
if [ "${1:-}" = git ]; then
  cat > /dev/null
  [ "${2:-}" = get ] || exit 0
  t=$("$0" --cached) || exit 1
  printf 'username=x-access-token\npassword=%s\n' "$t"
  exit 0
fi
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
  -d '{"repositories":["verity"],"permissions":{"contents":"write","pull_requests":"write"}}' |
  python3.12 -c '
import json, sys
d = json.load(sys.stdin)
if "token" not in d:
    sys.exit("gh_app_token: GitHub refused: %s" % d.get("message", d))
print(d["token"])'
