[2026-11-10T20:46Z] ba6ed968ab4a
ci (20:50Z), from top (CI-485, 20:28Z), after Daniel at 12:22 PM PST: "done today". **Item 5 (a mark checked on its merge with `main`) is due live at 6 PM PST (02:00Z), not Wed.** Your slow and regression tests come in the same PR.
- **Split with top's worker LEANWARM, as I answered top:** you hold the mark's path. `research queue ready` runs the tier on the head's merge with `main`'s tip and admits only on a pass.
- For a change that touches Lean, the path calls one function LEANWARM writes in `tools/verity/lean/`: the Lean audit on that merge, which publishes its modules and replay entries. Leave a clear call site for it, behind a flag if their function isn't on `main` yet. Agree the function's name and arguments with LEANWARM on CI-485.
- Build on #8683's branch `cursor/ci-preflight-main-7c31`, which already judges the preflight on `main`'s tip.

