[2026-11-14T18:35Z] 42cff2daeb3c
From the kernels lead, Sat 14 Nov 18:35Z. A CHANGE OF METHOD FOR YOUR ARCHITECTURE ROW, IN FORCE NOW (top, TOP-424, 18:33Z; Daniel, 18:25Z to 18:30Z: "I really would love it, even if nothing worked, just to have the whole repo in the shape of the architecture ... A refactor branch where we just do this, and then we run one test at the end and then merge to main ... the full architecture refactor done in the next 24 hours"). It replaces "open a pull request to main and mark it" in my last brief.

THE BRANCH IS `architecture`, cut from `main` `82f6b86e8f2d`, open now; it closes SUN 15 NOV 18:30Z. Its rules:
1. Your row lands THERE: fetch, merge `origin/architecture` into your work, push to `architecture` (`git push origin HEAD:architecture`). Never rebase it, never force. If the push is refused, fetch, merge, push again. Push at least once an hour, in pieces, not one big push at the end.
2. No queue, mark, verdict or check on it. Before a push, the .py files you changed compile (`python3 -m py_compile`). Tests may fail; they are mended after the branch closes.
3. EVERY path move is made by the move tool, its map committed under `tools/move/`. No bare `git mv`.
4. Shared lists (the lint allowlists' counts, `tests/tree.toml`, every `lean-audit.json`, generated files) are not kept current by hand. On a conflict in one, keep both sides' lines.
5. `main` takes no architecture change from now. An open architecture pull request of yours is merged into `architecture` by you, its mark withdrawn, and closed with one line naming the branch's commit that holds it.
6. A question of name or place: ask on TOP-424 as kernels/YOURNAME (`research msg linear comment TOP-424 --file F --worker YOURNAME`); with no answer in 15 minutes your reading stands, and you say it there.
So: THE MOVE FIRST, pushed; then our own test, each pinned binary byte for byte (CPU node, CUDA 13.0), run AFTER the push and reported, not held as a gate. No suites before a push beyond py_compile. Still: model check at each turn's start, no timer or subscription, nothing on a card, never a secret.

YOURS, KYARM: Pearl-C4's side, `benchmarks/pouw/harness/native` with `nvfp4_sm120`, to PoUW's prover under `verity/protocols/accounting/pouw/prover/` (it has `kernels/`, `circuit/`, `clear/`, `schemes/` on `main`: fit beside `kernels/`), with `benchmarks/harness/overhead_node.sh` changed in the same push. KTILE lands the Pearl-C kernels' move (#10035) on the same branch: read what it pushed before you choose names, and merge rather than cross. Return when the move is on the branch, by 22:30Z at the latest, with the branch's commit, what does not compile or load, the byte comparison's state, and your paragraph on the harness library's sha.

