# Brief: CLEANUP (soundness), cleanup window 07:00Z-09:00Z Tue 27 Oct

**Why:** top's announcement (Daniel, 11:40 PM PDT): outsiders read the repository from the morning. Clear the slop from soundness's area so what's left is the substance.

**Scope (only these):** `verity/core/protocols/{sampled_proofs,interface,composition,onsite,remote}`: Markdown, Python docstrings and comments. **No** Lean edits (Lean that runs as code waits for a Lean check). **Leave** `lean-audit.json` (tool-read data) and test-fixture run/art ids (data).

**1. Markers.** Run, over those dirs:
`rg -n --glob '!archive/**' -e '\b(TOP|CMP|LEAN|INF|TRN|PRF|NET|MEM|VLLM|LINTOOLS|ASK)-[0-9]+\b|note:[0-9]{8}T|\bart:[0-9a-f]{6}|notion\.(so|com)|slack\.com|\(#[0-9]{3,5}\)|PR #[0-9]+|\bDaniel\b|@(top|ci|infra|proofs|lean|comms)\b' DIRS`
Keep the reason; drop the id, the date and the name ("(Daniel, Sep 27)", "ASK-25", "PRF-127", "(#1494)" go; the rule they justify stays, stated as current truth). `@lean` in tests/test_proofs.py is a pytest decorator, not a handle: leave it. 36 lines before.
**2. Content.** Read each README/PROTOCOL.md/DESIGN.md there. Delete what is stale (superseded designs, retired features' histories like "two-stage sampling is retired (...)", finished studies' numbers, status logs) rather than rewriting history into it. Current truth an outsider needs stays. GLOSSARY.md (#5023) names win (e.g. "network certifier", never "warden"; "subprotocol", never "leg"; "protocol soundness", never "the one theorem").
**3. Commits/PR:** one commit `Delete stale sampled-proofs docs` (content), one for markers; one small PR, `Lane soundness` in the body, the before/after marker count in the body. Mark it ready (`research queue ready N --mark --by soundness`), then post the PR number on the soundness inbox thread? No: report it in your final message. Docs-only PRs land on any lead's verdict tonight.
**Branch:** `cursor/soundness-cleanup-<4 hex>` from `origin/main`; push only to it. Run only the touched suites' quick tier if you change any `.py` docstring (cpu-3 lane: `research run --on vy-nebius-cpu-3 --pool research --mem-gb 40 --source . --cwd source -- bash -lc "uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py --changed <main sha> --quick"`); a docstring/comment-only change may skip it if the suite runner says nothing changed.
**Done by 08:45Z:** PR marked, marker count after = 0 in scope (or each remaining line named as data).
