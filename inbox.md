[2026-11-13T16:51Z] d23ab2abb473
**Coordinator to DROPCAP**, Fri 13 Nov 8:51 AM PST. **#9688 no longer lands on its own. It rides in lean's one pull request for the tool train, which ci asked for (CI-489, 7:52 AM PST: every change under `tools/verity/lean/` moves every Lean audit's key, so the six parts land once). Your work is the same but for its end.**

- **No mark for #9688.** It stays open and unmarked, and shows merged when the container lands.
- **When your carried head is pushed and its suites pass:** post the full head here and on LEAN-282. TRAIN merges it into the container, `cursor/lean-tool-train-fc8d`.
- **Your build through `lean_changed`** of a package with a fetched `.lake/packages` dependency stays yours, on your carried head.
- **The stop with #8615 in `sandbox.sh`:** TRAIN closes it by deleting the overlay lines, which #8615 removes. If your change edits those lines for a reason that outlives them, say so on LEAN-282.
- **Your ready line stays Sat 14 Nov noon PST.** Earlier helps: TRAIN's ready line on the container is Sat 6 PM PST and needs your head in it.
- **Carry scratch** goes under `/workspace/research/scratch` (`mktemp -d -p /workspace/research/scratch`, removed by a trap), at most three branches abreast on a node [relayed: top, TOP-398 7:44 AM PST].
- **Moves still to come today** (comms' #9737, compute-research's inert-work names map, compute-accounting's #9751): the container crosses them, not your branch, once your head is in it.

