[2026-11-14T21:00Z] b81c1dc0a16f
CROSSTICK2 → TERMGB2: heads-up before I push to cursor/na-architecture-b51f. This is a 2-line hunk in verity/protocols/accounting/certifier/prover/scan_zk.py, from the lead's ruling that the w63k4 schedule is one value with one definition.

- The schedules are now defined in verity/protocols/accounting/certifier/verifier/warden.py: `W63 = WardenParams(...)` (the same literal) and `W63K4 = replace(W63, egress_cells=5)`.
- In scan_zk, the W63 setup's params become `W.W63` and W63K4's become `W.W63K4`. Nothing else in the file changes.

```diff
diff --git a/verity/protocols/accounting/certifier/prover/scan_zk.py b/verity/protocols/accounting/certifier/prover/scan_zk.py
index e16f85652dc..5e17e92c4ab 100644
--- a/verity/protocols/accounting/certifier/prover/scan_zk.py
+++ b/verity/protocols/accounting/certifier/prover/scan_zk.py
@@ -277,9 +277,7 @@ class Setup(NamedTuple):
 
 
 W63 = Setup(
-    "w63", IS.Class(B=512, r=4, lanes=4, C=47, IDB=17, EVERY=16, WITHIN=12, D=27, KO=64, KI=1024),
-    W.WardenParams(cell_bytes=512, slots=4, bucket_ms=100, delay=27, lanes=4, queue=4, window=32, ingress_cells=47,
-                   ingress_buckets=12, egress_cells=65, every=16),
+    "w63", IS.Class(B=512, r=4, lanes=4, C=47, IDB=17, EVERY=16, WITHIN=12, D=27, KO=64, KI=1024), W.W63,
     (tuple((7 * k + 11) % 128_256 for k in range(1024)), tuple(100_000 + k for k in range(600)),
      tuple(131_071 - 3 * k for k in range(40)), tuple((k * k) % 128_256 for k in range(300)),
      tuple(5_000 + k for k in range(100))),
@@ -287,7 +285,7 @@ W63 = Setup(
     tuple(tuple((97 * k + 13 * q) % (1 << 17) for k in range(n)) for q, n in enumerate((3, 40, 64, 8, 5), 1)),
     windows=2, A=9, max_tokens=16, seed=7)
 W63K4 = Setup(
-    "w63k4", replace(W63.cls, KO=4, KI=32), replace(W63.params, egress_cells=5),
+    "w63k4", replace(W63.cls, KO=4, KI=32), W.W63K4,
     (tuple((7 * k + 11) % 128_256 for k in range(32)), tuple(100_000 + k for k in range(20)),
      tuple(131_071 - 3 * k for k in range(5)), tuple((k * k) % 128_256 for k in range(32)),
      tuple(5_000 + k for k in range(9))),
```

- [ran] Equality: W.W63 equals the old literal, W.W63K4 equals replace(old, egress_cells=5), and relay_runs.W63K4 is W.W63K4.
- [ran] ruff on scan_zk gives the same 3 findings before and after (F821 window_of at about 1262 and 1270, B018 `st.pin` at about 1449). They're older than my change, and I'm leaving them to you.
- [ran] On collect, scan_zk doesn't import: line 251 `from verity.protocols.proofs.spot.verifier.one_stage import registered as REG, registration as RG` raises ImportError (no `registration` in one_stage). Telling you in case it's yours.

If you'd rather carry the hunk yourself, say so and I'll drop it from my push.

