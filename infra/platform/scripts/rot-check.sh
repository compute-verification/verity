#!/usr/bin/env bash
# Rebuilds the platform against the newest release of every SDK it pins, then
# typechecks it and runs its unit and end-to-end tests. Exit 0 means the pins
# can move; anything else names what broke. The checkout itself is untouched:
# the build happens in a copy.
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

tar -C "$here" --exclude=node_modules --exclude=.next --exclude='.workflow-*' --exclude=.data -cf - . | tar -C "$work" -xf -
cd "$work"

deps=(ai workflow next @ai-sdk/harness @ai-sdk/harness-claude-code @ai-sdk/harness-codex @ai-sdk/sandbox-vercel
  @ai-sdk/workflow-harness @vercel/sandbox @vercel/oidc @modelcontextprotocol/sdk @electric-sql/pglite)
dev=(@workflow/vitest vitest typescript)

echo "== pinned"
node -e 'const p=require("./package.json");for(const [k,v] of Object.entries({...p.dependencies,...p.devDependencies}))console.log(k,v)'
echo "== installing latest"
npm install --no-audit --no-fund --loglevel=error --save-exact "${deps[@]/%/@latest}"
npm install --no-audit --no-fund --loglevel=error --save-exact --save-dev "${dev[@]/%/@latest}"
echo "== resolved"
node -e 'const p=require("./package.json");for(const [k,v] of Object.entries({...p.dependencies,...p.devDependencies}))console.log(k,v)'
echo "== typecheck"
npx tsc --noEmit
echo "== unit"
npx vitest run --config vitest.config.ts
echo "== e2e"
npx vitest run --config vitest.integration.config.ts
echo "== rot check passed"
