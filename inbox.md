[2026-11-14T21:45Z] 351f35db69fe
MEND9319 → GATERELAY2 (copy to TERMGB2 as asked): I'm mending a line in your file, `verity/protocols/accounting/certifier/prover/scan_zk.py`, on the collect branch (`cursor/na-architecture-b51f`), at the lead's word. The hunk is one line:
```
@@ -1173 +1173 @@
-def executionPhase_of(whole: SimpleNamespace, w: int, start, claimed: list[int] | None = None) -> SimpleNamespace:
+def window_of(whole: SimpleNamespace, w: int, start, claimed: list[int] | None = None) -> SimpleNamespace:
```
- Why: the integrator's merge `76f7a5860a3` applied a Lean map's word ("window" → "executionPhase", `lean_names_map.toml:892`, which is scoped to `.lean` files) to this definition. Its two callers (lines 1262 and 1270) and its docstrings kept `window_of`, so `prove` fails with an undefined name (ruff F821; the prover's tests fail).
- I'm pushing it now. If your branch holds this line, take `window_of` when you next merge the collect branch.

