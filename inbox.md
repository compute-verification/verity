[2026-11-17T00:06Z] 63e9201f47b7
#10564's new head is 4bed2cbd251b1bf2a26e3ae9f6bf5df6549c53dd, pushed. It is still a draft from 6f9f794db37 and lands after #10597. It is now 29 files, +9/−1,367.

New in this head:
- Deleted security/Proofs/InferenceOnly/Cover.lean and Cover/Checked.lean, and their line in Proofs/InferenceOnly.lean.
- Removed the inference-only verifier README's clause "(`theta_desc` is a proof lemma)".
- No other file uses coverTrained, theta_desc or checkedCoverTraining_of_descAgrees (ran: git grep, excluding tools/move).

The body lists the lock records to remove. The properties lock's records for InferenceOnlyCheck.Properties.SecurityProofs.CheckedCoverTraining are guarantees, unwitnessed and shape.properties.

Ran: the root Lean-name, doc-path, retired-terms and claim-id tests (91 passed), and tests/benchmarks/slowdown/test_lean_names.py (3 passed, against the unchanged lock).

Not run: the Lean build, the Lean check, the lock update and the suites whole. That the Lean builds is inferred: no remaining file imports the two files.

[2026-11-17T00:10Z] fbf40d470c5d
model: Claude Opus 5.5 High; subscriptions removed: none held (no timers, no subscriptions).

TRN-158, the inference-only verdict as the Lean verifier's: draft https://github.com/compute-verification/verity/pull/10670 (branch cursor/training-inference-only-lean-decides-4207), not marked. No count changes.

What's in it:
- θ (ran): the held weights go to Lean as registered inputs in θ, through `Program(..., registered=, theta=)`, in both `count_instance` (one-stage) and `built.instance`. population.py is unchanged.
- The one-stage row (ran): the verdict on c and k is now the Lean verifier's, and n is judged on the exact sealed count. With no --lean, or a Lean verifier that gives no verdict, the row is undecided.
  - On the 1B class it is reject on k: "the Lean verifier counts a block of X fresh bits, more than k = Y (the record's count is Y: the Lean verifier counts the rows the record's wires fill and the pinned tables' openings as fresh, because the session Program it is given carries no wires)".
  - `record_count` {held, wired, openings, fresh} is a labelled field beside the verdict. The format is v5.
- Our benchmark (ran): "The class this tree builds" prints "verdict   the Lean verifier's, ...: reject, k failed", with the record's count beside it.
  - k = 38,625 and the tied path's k = 145,568 carry ", resting on the record's count, which the Lean verifier does not confirm".
  - The value line and its 1 are unchanged.
- Tests (ran):
  - On a made circuit (GEMM and RoPE over two slots, one declared wire), the record's count is 19,840 and the Lean verifier's is 19,968. Both figures are asserted, and the test passes today.
  - With no wire and no table, both counts are 19,968 at the real k, and the Lean verifier refuses at k - 1.
  - This went in our suite, not test_block_counts.py, which is @circuits'.
- Node run (ran): tests/benchmarks/inference_only (88 passed) and tests/benchmarks/one_stage (1,739 passed), fresh with slow tests, plus ruff. The last commit after it is docstring only.

The composed record's row (read): the Lean composite takes no inference-only (c, k, n) verdict today. composite.py, CoreRemote.lean and CoreOnsite.lean combine `verity/inference-only/v0`'s own case (InferenceOnlyCheck.Protocol, the Theta/Desc rate rule at α and Γ; params alpha, gamma, log, parameters, program; evidence owner, provenance, theta).
- So the record carries an `inference_only` row and the RECORD line's `not_combined`, saying "not read by the combined verdict". BR.PRODUCTS and the Lean composite are unchanged.
- What the composite would need (inferred): a case of this Count verdict (c, k, n, the session Program with θ, the table) as a new protocol id in composite's cases, with a Lean combine rule for it in CoreRemote and CoreOnsite. Or the (c, k, n) rule folded into inference-only/v0's own case. Soundness owns the Lean, proofs the hand-off.

One more gap, the same kind (ran, on our fixture): θ can only name a whole parameter. A row the record holds for some units and wires for the rest (the norm's w in our `small`) is held in the record's count but fresh to the Lean verifier. On the 1B class θ removed exactly Python's held bits, so it doesn't arise there.

Diff of proofs' two files (benchmarks/one_stage/inference_only.py, benchmarks/one_stage/served_verdicts.py), base 220767e4865 to the branch head. The rest of the branch touches served_verdicts.sh's step-5 comment and tests/benchmarks/one_stage too.

```diff
diff --git a/benchmarks/one_stage/inference_only.py b/benchmarks/one_stage/inference_only.py
index 8ebc8d9b62b..799ba7ab123 100644
--- a/benchmarks/one_stage/inference_only.py
+++ b/benchmarks/one_stage/inference_only.py
@@ -43,27 +43,28 @@ refuses (`gate-table`).
 policy's class (``--circuit``, `served_verdicts.sh`'s step 3), whose session is the run's block Definition, its gates
 by the table, and its fresh bits at the verifier's pins (``--pins``, the weights registration) and the window's roots
 (``--registration``), as `block_counts` counts them; and the serve dir's sealed windows, each of which the class takes
-(`class_windows`) holding the class's m blocks, since a class's circuit is the same for every window it takes. Its rows
-are `InferenceOnlyCheck.Count`'s (``[definition, gates, fresh, blocks]``) at the exact counts, judged by its rule (`judge`): every
-class with a block at most c gates and k fresh bits, and the blocks at most n. Anything the rule can't decide refuses
-(`Undecided`): no policy, a policy whose ``k_by_kind`` doesn't sum to its k or isn't the run's circuit's (a policy
-written without one is read as before), a serve dir that records no start, a policy written once the serve had begun (unless
-``--retroactive`` says why it's read anyway), a record with no inference blocks, an unpriced primitive, a closed window
-with no seal, or a sealed window the class doesn't take.
-With ``--lean``, `inference-only-check --count` runs on one window's session Program (`lean_count`), with ``--detail`` where the
-binary takes it, as its usage says (`takes_detail`): such a binary prints only ``{complies}`` without it and its whole
-verdict with it, as one that doesn't take it prints by default. A block's gates that differ from its own refuse too.
-That Program's inputs are positional (`Input<w>`), so its θ is empty: every input bit, the weights' too, is fresh to it,
-and its fresh bits and its verdict there (``with_theta_empty``) are said beside the verdict's, not judged. Its whole
-verdict counts that one window's blocks and names the row that failed, so the public verdict takes only its gates a
-block, fresh bits and ``complies``, or its refusal's ``error`` and ``message`` (`lean_said`), and the private one the
-whole call.
+(`class_windows`) holding the class's m blocks, since a class's circuit is the same for every window it takes. Anything
+that can't be decided refuses (`Undecided`): no policy, a policy whose ``k_by_kind`` doesn't sum to its k or isn't the
+run's circuit's (a policy written without one is read as before), a serve dir that records no start, a policy written
+once the serve had begun (unless ``--retroactive`` says why it's read anyway), a record with no inference blocks, an
+unpriced primitive, a closed window with no seal, a sealed window the class doesn't take, no ``--lean``, or a Lean
+verifier that gives no verdict or counts other gates a block than the table does.
+The verdict is the Lean verifier's: `inference-only-check --count` on one window's session Program (`lean_count`), with
+``--detail`` where the binary takes it, as its usage says (`takes_detail`), the Program's parameters whose every row is
+a registered read of a pinned parameter bound at registration and named in θ (`held_inputs`, `count_instance`). It
+decides each block's gates against c and fresh bits against k; n is judged on the exact count, the sealed windows' blocks.
+The record's count (``record_count``: `block_counts.fresh` and the rows the record's wires and the pinned tables'
+openings fill) is carried beside it and never decides it. The session Program carries no wires,
+so the Lean verifier counts a wired row and a table opening as fresh where the record's count does not. Its whole
+verdict counts that one window's blocks, so the public verdict takes only its gates a block, fresh bits, ``complies``
+and the rule that failed, or its refusal's ``error`` and ``message`` (`lean_said`), and the private one the whole call.
 
 ``--out`` is the public verdict: the judgment against (c, k, n) and no count, of blocks or of windows, in any form. It
-holds the verdict, ``complies``, ``failure`` and ``reason``, each block class's session and its gates and fresh bits a
-block (``classes``), the class's slots and positions, and ``counts`` saying that n was judged on the exact count, which
-it doesn't publish. ``--private`` (the serve dir's, 0600) holds it with the exact counts (``blocks``, `InferenceOnlyCheck.Count`'s
-rows, and ``total``), each window the class doesn't take and why, and the `inference-only-check --count` call (``lean_detail``).
+holds the verdict, ``complies``, ``failure`` and ``reason``, each block class's session and the Lean verifier's gates
+and fresh bits a block (``classes``), the record's count a block, the class's slots and positions, and ``counts`` saying
+that n was judged on the exact count, which it doesn't publish. ``--private`` (the serve dir's, 0600) holds it with the
+exact counts (``blocks``, the record's rows, and ``total``), each window the class doesn't take and why, and the
+`inference-only-check --count` call (``lean_detail``).
 """
 from __future__ import annotations
 
@@ -78,6 +79,7 @@ import re
 import subprocess
 import sys
 import tempfile
+from collections import Counter
 from collections.abc import Mapping, Sequence
 from fractions import Fraction
 from pathlib import Path
@@ -91,10 +93,14 @@ import served_shares as SS  # noqa: E402
 import served_verdicts as SV  # noqa: E402
 
 POLICY_FORMAT = "verity-bench/inference-only-policy/v1"
-VERDICT_FORMAT = "verity-bench/inference-only-verdict/v4"
+VERDICT_FORMAT = "verity-bench/inference-only-verdict/v5"
 #: what the public verdict says of the counts: a count on the grid, floored or not, still gives the count of windows
 #: (at 4 blocks a window, exactly for 5 to 16 of them), so it publishes the judgment against n alone
 COUNTS = "none published: n judged on the blocks' exact count, which only the serve dir's private verdict holds"
+RECORD_COUNT = ("the record's count: block_counts on the record's wires and declarations, beside the Lean verifier's "
+                "verdict, which it does not decide and which does not confirm it")
+WIRED = ("the Lean verifier counts the rows the record's wires fill and the pinned tables' openings as fresh, because "
+         "the session Program it is given carries no wires")
 INFERENCE_ONLY_PACKAGE = ROOT / "verity/protocols/workload_classification/training_detection/inference_only/verifier/lean"
 INFERENCE_ONLY_REFERENCE = INFERENCE_ONLY_PACKAGE / "scripts" / "inference_only_reference.py"
 #: `prove_path.py window-circuit`'s default ``--cap``: the units a stated member holds in the class's record
@@ -411,16 +417,60 @@ def _reference():
     return module
 
 
-def count_instance(doc: Mapping[str, Any], table: Mapping[str, int], c: int, k: int, n: int) -> dict[str, Any]:
+def _held_reads(doc: Mapping[str, Any], pins: Mapping[str, str] | None,
+                roots: Mapping[str, str] | None) -> tuple[dict[str, int], dict[str, dict[str, int]], Counter]:
+    from verity.commitments.periods import interface as IF
+    from verity_vllm.program.predict import block_counts as BC
+
+    units = {t: int(n) for t, n in doc["templates"]}
+    rows = {t: dict(IF.layout(BC.resolve(t)).rows) for t in units}
+    _, cuts, reg, _ = IF.declarations(doc)
+    held = IF.held_params(pins, roots)
+    covered: Counter = Counter()
+    for r in (reg.reads if reg is not None else ()):
+        if r.param in held and r.port in rows[r.reader]:
+            covered[(r.reader, r.port)] += r.units * 16 * rows[r.reader][r.port] // int(cuts.get(r.reader, {}).get(r.port, 1))
+    return units, rows, covered
+
+
+def held_inputs(doc: Mapping[str, Any], pins: Mapping[str, str] | None, roots: Mapping[str, str] | None) -> tuple[str, ...]:
+    units, rows, covered = _held_reads(doc, pins, roots)
+    return tuple(f"{j}.{port}" for j, t in enumerate(units) for port, words in rows[t].items()
+                 if covered[(t, port)] == 16 * words * units[t])
+
+
+def record_count(doc: Mapping[str, Any], m: int, pins: Mapping[str, str] | None,
+                 roots: Mapping[str, str] | None) -> dict[str, Any]:
+    from verity.commitments.periods import interface as IF
+
+    count = IF.input_count(doc, pins=pins, roots=roots)
+    held = sum(_held_reads(doc, pins, roots)[2].values())
+    bits = {"held": held, "wired": count["wired"], "openings": count["registered"] - held,
+            "fresh": count["free"] + count["ingress"]}
+    return {key: v // m if v % m == 0 else v / m for key, v in bits.items()}
+
+
+def count_instance(doc: Mapping[str, Any], table: Mapping[str, int], c: int, k: int, n: int,
+                   held: Sequence[str] = ()) -> dict[str, Any]:
     """`InferenceOnlyCheck.Count`'s instance on one window: its session Definition activated at its m slots (the session
-    Program, `population.population_program`) with no θ, its block that Definition."""
-    from verity.circuits import codec
-    from verity.circuits.population import population_program
+    Program, `population.population`), its parameters ``held`` bound at registration and named in θ, its block that
+    Definition."""
     from verity_vllm.program.predict import block_counts as BC
 
     fn, m, _ = BC.session(doc)
-    text = codec.canonical_json(codec.encode_program(population_program(fn, m))).decode()
-    return {"program": text, "theta": [], "blocks": [codec.descriptor_id(fn)], "table": dict(table), "c": c, "k": k, "n": n}
+    return session_instance(fn, m, table, c, k, n, held)
+
+
+def session_instance(fn: Any, m: int, table: Mapping[str, int], c: int, k: int, n: int,
+                     held: Sequence[str] = ()) -> dict[str, Any]:
+    from verity.circuits import codec
+    from verity.circuits.population import population
+    from verity.circuits.program import Program
+
+    program = Program(population(fn, m), registered=tuple(held), theta=tuple(held))
+    text = codec.canonical_json(codec.encode_program(program)).decode()
+    return {"program": text, "theta": sorted(program.annotations.get("theta", [])), "blocks": [codec.descriptor_id(fn)],
+            "table": dict(table), "c": c, "k": k, "n": n}
 
 
 def takes_detail(exe: Path, timeout_s: int) -> bool:
@@ -461,8 +511,8 @@ def lean_count(exe: Path, inst: Mapping[str, Any], scratch: Path, timeout_s: int
 
 def lean_said(call: Mapping[str, Any], gates: int) -> dict[str, Any]:
     """What the public verdict takes of `lean_count`'s ``call``: why it gave no verdict, its refusal (``error`` and
-    ``message``), or its gates a block, whether they are ``gates``, and with θ empty its fresh bits and ``complies``.
-    Never its rows' counts, total or failure, nor any other text it printed."""
+    ``message``), or its gates and fresh bits a block, whether the gates are ``gates``, ``complies`` and the rule that
+    failed. Never its rows' counts or total, nor any other text it printed."""
     out = call.get("out")
     if out is None:
         return {"error": call["error"]}
@@ -472,7 +522,9 @@ def lean_said(call: Mapping[str, Any], gates: int) -> dict[str, Any]:
     if not (isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], list) and len(rows[0]) == 4):
         return {"error": f"inference-only-check {' '.join(call['flags'])} printed no row of the block"}
     _, g, x, _ = rows[0]
-    return {"gates": g, "agrees_on_gates": g == gates, "with_theta_empty": {"fresh": x, "complies": out.get("complies")}}
+    failure = out.get("failure")
+    return {"gates": g, "agrees_on_gates": g == gates, "fresh": x, "complies": out.get("complies") is True,
+            "failed": failure[1] if isinstance(failure, list) and len(failure) == 2 else None}
 
 
 def served_start(serve_dir: Path) -> str:
@@ -536,24 +588,36 @@ def verdict(pol: Mapping[str, Any] | None, doc: Mapping[str, Any] | None, serve_
         blocks = sealed * row["slots"]
         rows = [[row["definition"], row["gates"], row["fresh"], blocks]]
         got = judge(rows, c, k, n)
+        counted = record_count(doc, row["slots"], pins, roots)
         private |= {"blocks": rows, "total": got["total"]}
-        base |= {"classes": [[d, g, x] for d, g, x, _ in rows], "counts": COUNTS, "complies": got["complies"],
-                 "failure": got["failure"], "positions": row["positions"], "slots": row["slots"]}
-        if lean is not None:
-            inst = count_instance(doc, row["table"], c, k, n)
-            call = lean_count(lean, inst, scratch or Path(tempfile.gettempdir()), lean_timeout_s)
-            private["lean_detail"] = call
-            base["lean"] = said = lean_said(call, row["gates"])
-            if "gates" in said and not said["agrees_on_gates"]:
-                raise Undecided(f"inference-only-check --count counts {said['gates']} gates a block, and this {row['gates']}")
-        if got["complies"]:
+        base |= {"counts": COUNTS, "positions": row["positions"], "slots": row["slots"],
+                 "record_count": {**counted, "by": RECORD_COUNT}}
+        if lean is None:
+            raise Undecided("no Lean verifier was given (--lean), and the verdict is the Lean verifier's")
+        inst = count_instance(doc, row["table"], c, k, n, held_inputs(doc, pins, roots))
+        call = lean_count(lean, inst, scratch or Path(tempfile.gettempdir()), lean_timeout_s)
+        private["lean_detail"] = call
+        base["lean"] = said = lean_said(call, row["gates"])
+        if "error" in said:
+            raise Undecided(f"the Lean verifier gave no verdict: {said['error']}"
+                            + (f": {said['message']}" if "message" in said else ""))
+        if not said["agrees_on_gates"]:
+            raise Undecided(f"the Lean verifier counts {said['gates']} gates a block, and the gate table {row['gates']}")
+        if not said["complies"] and said["failed"] not in ("gates", "fresh", "count"):
+            raise Undecided("the Lean verifier refuses and names no rule of gates, fresh and count that failed")
+        failure = ([0, said["failed"]] if not said["complies"] and said["failed"] != "count" else
+                   [None, "count"] if blocks > n or not said["complies"] else None)
+        base |= {"classes": [[row["definition"], said["gates"], said["fresh"]]], "complies": failure is None,
+                 "failure": failure}
+        if failure is None:
             base |= {"verdict": "accept", "reason": None}
         else:
-            i, what = got["failure"]
+            beside = ("" if said["fresh"] <= counted["fresh"] else
+                      f" (the record's count is {counted['fresh']}: {WIRED})")
             base |= {"verdict": "refuse", "reason": {
-                "gates": f"a block of {row['gates']} gates, more than c = {c}",
-                "fresh": f"a block of {row['fresh']} fresh bits, more than k = {k}",
-                "count": f"the blocks' exact count is more than n = {n}"}[what]}
+                "gates": f"the Lean verifier counts a block of {said['gates']} gates, more than c = {c}",
+                "fresh": f"the Lean verifier counts a block of {said['fresh']} fresh bits, more than k = {k}{beside}",
+                "count": f"the blocks' exact count is more than n = {n}"}[failure[1]]}
     except Undecided as e:
         base |= {"verdict": "refuse", "reason": f"undecided: {e}", "complies": False}
     except Exception as e:  # noqa: BLE001 - anything else the count can't get past is undecided too, and refuses
@@ -580,13 +644,12 @@ def line(v: Mapping[str, Any]) -> str:
     if "classes" not in v:
         return f"{v['verdict']}: {v['reason']}"
     (d, g, x), = v["classes"]
-    lean = v.get("lean")
-    said = ("" if lean is None else f"; inference-only-check --count: {lean['error']}" if "error" in lean else
-            f"; inference-only-check --count: {lean['gates']} gates a block ({'agrees' if lean['agrees_on_gates'] else 'DIFFERS'})")
+    r = v["record_count"]
     return (f"{v['verdict']}{'' if v['reason'] is None else ': ' + v['reason']}: session {d[:16]}… "
-            f"({'the registered one' if v.get('registered_session') else 'not the registered one'}), {g} gates and {x} "
-            f"fresh bits a block, against c {v['c']}, k {v['k']}, n {v['n']} (no count published: n is judged on the "
-            f"blocks' exact count, held on the node){said}")
+            f"({'the registered one' if v.get('registered_session') else 'not the registered one'}), the Lean verifier's "
+            f"{g} gates and {x} fresh bits a block, against c {v['c']}, k {v['k']}, n {v['n']} (no count published: n is "
+            f"judged on the blocks' exact count, held on the node); the record's count, which decides nothing: "
+            f"{r['fresh']} fresh bits a block ({r['held']} held, {r['wired']} wired, {r['openings']} openings)")
 
 
 def main(argv: Sequence[str] | None = None) -> int:
@@ -612,7 +675,7 @@ def main(argv: Sequence[str] | None = None) -> int:
     v.add_argument("--circuit", type=Path, required=True, help="the picked window's circuit record at the policy's class")
     v.add_argument("--registration", type=Path, required=True, help="the window's registration (its roots)")
     v.add_argument("--pins", type=Path, default=None, help="the weights registration, the verifier's pins")
-    v.add_argument("--lean", type=Path, default=None, help="inference-only-check, to run --count on one window's session Program")
+    v.add_argument("--lean", type=Path, default=None, help="the Lean verifier, inference-only-check, whose --count on one window's session Program is the verdict")
     v.add_argument("--lean-timeout", type=int, default=1800, help="inference-only-check --count's limit in seconds (default 1800)")
     v.add_argument("--retroactive", default=None, help="why a policy written after the serve began is read anyway")
     v.add_argument("--out", type=Path, required=True)
diff --git a/benchmarks/one_stage/served_verdicts.py b/benchmarks/one_stage/served_verdicts.py
index 000cac50345..78b2b19a37f 100644
--- a/benchmarks/one_stage/served_verdicts.py
+++ b/benchmarks/one_stage/served_verdicts.py
@@ -262,6 +262,9 @@ PIECES: dict[str, tuple[str, str, str, str]] = {
 #: the record's subprotocol outside this chain: serve() runs its piece (WIPE_CMD) on the card before its engine loads
 WIPE = ("the wipe", "wipe", "memory-accounting", "the wipe of the serving card before its engine loads")
 NAMES = {"integrity": "sampled proofs", **{k: v[0] for k, v in PIECES.items()}, "wipe": WIPE[0]}
+INFERENCE_ONLY = "inference-only.json"
+INFERENCE_ONLY_UNREAD = ("not read by the combined verdict: the composite combines verity/inference-only/v0's own "
+                         "case, not this (c, k, n) verdict")
 #: a piece's limit unless ``<COMPONENT>_TIMEOUT`` or ``PIECE_TIMEOUT`` sets one, as `timeout` reads it
 PIECE_TIMEOUT = "60m"
 #: what `pieces` leaves beside a piece it ran: the node it ran on, its exit and its limit
@@ -2347,11 +2350,19 @@ def composite_answer(public: Path, product: str, window: str | None = None,
             f"{doc['product']}'s Lean check, compiled, in {where}")
 
 
+def inference_only_row(v: Mapping[str, Any]) -> dict[str, Any]:
+    return {"verdict": {"accept": "accept", "refuse": "reject"}.get(v.get("verdict"), "error"),
+            "protocol": v.get("format"), "reason": v.get("reason"),
+            "answer": {k: v[k] for k in ("classes", "c", "k", "n", "lean", "record_count", "failure") if k in v},
+            "answer_from": f"{AS_WRITTEN} ({INFERENCE_ONLY})", "combined": INFERENCE_ONLY_UNREAD}
+
+
 def record(rows: Mapping[str, Mapping[str, Any]], *, window: int, steps: tuple[int, int], node: str,
            run: str | None, servers: Mapping[str, tuple[str | None, str]] | None = None, product: str = "onsite",
            deferred: str | None = None, off_node: bool = False, tree: Mapping[str, str] | None = None,
            served_compare: str | None = None, answer: Mapping[str, Any] | None = None,
-           lean: str = "no composite was read") -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
+           lean: str = "no composite was read", inference_only: Mapping[str, Any] | None = None,
+           ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
     """The window report over ``rows`` (by runner name), its block record (None when a subprotocol about this node
     rejected, erred or was undecided), and the ``RECORD`` line's summary, composed for ``product``: each subprotocol it
     drops is its `dropped_row`, about no server. ``servers``: each piece's `piece_server`, by runner name; SPOT's
@@ -2361,7 +2372,9 @@ def record(rows: Mapping[str, Mapping[str, Any]], *, window: int, steps: tuple[i
     hand-off (`HANDED_FROM`); ValueError on ``node`` itself. ``tree``: the composing tree's ``tree`` (its commit) and
     ``tree_on`` (the branch its host pinned it from), which ``composed_on`` carries. ``answer``: the product's compiled
     Lean check's answer (`composite_answer`), the block record's verdict, and ``lean`` where it was read or why there is
-    none, which the report and the line carry; a rejection by it rejects whatever the rows say."""
+    none, which the report and the line carry; a rejection by it rejects whatever the rows say. ``inference_only``: the
+    inference-only verifier's verdict, which the report carries as its ``inference_only`` row and the line under
+    ``not_combined``, and which neither the combined verdict nor the block record reads."""
     from verity_e2e_demo import block_record as BR
 
     host = socket.gethostname()
@@ -2402,6 +2415,9 @@ def record(rows: Mapping[str, Mapping[str, Any]], *, window: int, steps: tuple[i
              "answers": answers, "lean": lean, **left,
              **({"dropped": {NAMES[k]: rows[k]["why"] for k in drops}} if drops else {}),
              **({"unserved": unserved} if unserved else {}), **dev}
+    io = {} if inference_only is None else {"inference_only": inference_only_row(inference_only)}
+    if io:
+        named["not_combined"] = {"the inference-only verifier": io["inference_only"]["verdict"]}
     bad = sorted(NAMES[k] for k, r in rows.items()
                  if r.get("verdict") not in ("accept", "missing", "dropped") and NAMES[k] not in elsewhere)
     if bad:
@@ -2411,6 +2427,7 @@ def record(rows: Mapping[str, Mapping[str, Any]], *, window: int, steps: tuple[i
                               "lean": lean}
         line = {"composed": composed, "by": bad, **named, "block_record": None, "reason": report["composed"]["reason"],
                 "why": "block_record carries an accepted subprotocol's certificate only"}
+        report["protocols"] = {**rows, **io}
         return report, None, line
     extra = {subprotocols[k]: {"row": r, "server": at[k][0], "runs": r.get("runs") or [run], "source": at[k][1]}
              for k, r in rows.items() if k != "integrity" and r.get("verdict") == "accept" and at[k][0] is not None}
@@ -2423,6 +2440,7 @@ def record(rows: Mapping[str, Mapping[str, Any]], *, window: int, steps: tuple[i
             "record_missing": rec["composed"]["missing"],
             "check": BR.check(rec), "steps": list(steps), "node": node,
             "deployment": rec["monitoring_period"]["deployment"]["id"]}
+    report["protocols"] = {**rows, **io}
     return report, rec, line
 
 
@@ -2766,12 +2784,18 @@ def main(argv: Sequence[str] | None = None) -> int:
                         else (None, "no composite was read"))
     except CompositeRefused as e:
         answer, lean, refused = None, str(e), True
+    try:
+        io = json.loads((a.out / INFERENCE_ONLY).read_text()) if (a.out / INFERENCE_ONLY).is_file() else None
+    except ValueError as e:
+        io = {"verdict": "error", "reason": f"{INFERENCE_ONLY} doesn't parse: {e}"}
+    if io is not None and not isinstance(io, dict):
+        io = {"verdict": "error", "reason": f"{INFERENCE_ONLY} is not an object"}
     report, rec, line = record(rows, window=a.window, steps=(steps[0], steps[1]), node=a.node, run=run,
                                servers={k: at for k, (_, at) in got.items()}, product=a.product, deferred=a.deferred,
                                off_node=a.off_node, tree={"tree": a.tree, **({"tree_on": a.tree_on} if a.tree_on else {})}
                                if a.tree else None,
                                served_compare=None if a.served_compare is None else served_compare_line(a.served_compare),
-                               answer=answer, lean=lean)
+                               answer=answer, lean=lean, inference_only=io)
     a.out.mkdir(parents=True, exist_ok=True)
     (a.out / "verdicts.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
     if rec is not None:
```

