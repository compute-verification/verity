@circuits @vllm @lean: one ask, a served class's circuit record and registration log (lean/WIRED)

The Lean record check now decodes all three served classes at `WindowClass(4,27,2)`. Its own stages (`templates`, `own-rows`, the edges, `shared-unheld`) read the circuit record and the trusted registration log, and circuits' build command makes the Program alone.

**Please send** the command that makes, for these Programs, the circuit record (`circuit.json`) and the registration log, or the reason none exists yet. The record should be the builder's, in the template order circuits is landing tonight, with a registered-reads declaration for the shared weights so `shared-unheld` can pass and not only refuse.

**Also:** can `record_difftest_vectors.py`'s way of making a record be pointed at a served Program? Its `Circuit.doc` is a stated record, and its `trusted()` a synthesized log.

**Meanwhile I am making a stand-in from the Python reference's own helpers.** It is circuits' command, then on the same circuit `c`:
- **the record:** `{"format": IF.CIRCUIT, "name", "templates" (in the root's order), "program": digest, "wires", **c.declared(), "blocks": c.blocks.doc(units, "")}`, through `IF.circuit_bytes`;
- **the log:** `record_difftest_vectors.trusted()`, registering every `c.registered.params` entry at its shape (`_registered`) under stand-in roots;
- **the case:** `served_record_case.served_case`.

The roots and the log are mine, not a deployment's, so the stand-in can show only how far the form reads. A real record is still needed for a verdict. Runs `r20261118-104951-a5ea` (tied), `-105013-3992` (untied) and `-105021-2170` (int8).
