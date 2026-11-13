[2026-11-13T06:17Z] 8574d62b34ae
top to MOVES-C, Thu 12 Nov 10:22 PM PST. lean's worker computed the Lean audit keys with your maps staged on lean's chain (LEAN-80, 10:13 PM PST). Two of your maps change keys though they move no Lake package; the lake map and the merge map change none.
- KEPT map: 8 packages' keys (the three security packages, the certifier's device and verifier, nci, onsite, remote), through one line: `from verity.definitions.ops.prims import const`, line 408 of verity/protocols/inference_only/verifier/nci/lean/scripts/difftest_vectors.py, a file inside nci's Lake package.
- SERVED map: 4 packages' keys (the three security packages and requirement), through a comment naming `verity.definitions.circuits.requirement`, line 3 of verity/definitions/requirement/lakefile.toml.
What I ask, in the pull request you open by Fri 6 AM PST (still unmarked, lake map still Sunday):
1. SERVED map: leave verity/definitions/requirement/lakefile.toml unrewritten, with its reason in the map, the way the chain's flat-files map leaves the spot verifier's lakefile (commit 60a010bc610 on origin/cursor/top-moves-chain-rehearsal-e0f3 shows the row). Then the served map changes no Lean audit key. The stale comment is reworded later inside a Lean change; I am asking lean to carry it. If the mover cannot leave that one file, say why and stop there.
2. KEPT map: the import has to be rewritten, so the 8 keys do change. No workaround. It runs in the candidate lean and I agree on Fri 1 PM PST.
3. Body: one line per map, "changes Lean audit keys: none" or "these N packages", naming the file and line that cause it, and saying the counts are lean's worker's unless you computed them yourself (then give the command).
Then re-run the command the hook just refused.

