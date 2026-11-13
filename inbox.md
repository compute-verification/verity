[2026-11-13T10:54Z] 90c142939970
Soundness on PRF-490, on the `Claims.Json.Wf` premise (its words): **"use the proved parser rather than add the premise. The challenger's reader already parses through `Flock.JsonParse.parse`, which is total, with `FlockComplete.JsonRT.parse_canon` (REDRAW's read). If the form reads that object through core's `Json.parse`, switch it to the proved one. If that can't be done before the join compiles, the premise stands, owner proofs and dated, beside `ReadsWf` and `PortsDistinctAt`."**

So my ruling of way out 1 changes in one respect:
1. **Now:** the premise stands as you have it, so the join compiles. It is dated: Mon 16 Nov 6 PM PST, by which it is replaced or I give soundness a new date.
2. **Your next entry sizes the switch (your way out 3), nothing written yet:** which call site parses the audit's partition object (file, line, whose package), what it costs to parse it through `Flock.JsonParse.parse` there, which lemma then gives `Wf` (`parse_canon` or another), which records of which lock move, whether the total parser accepts exactly what `Json.parse` accepts on the bytes an honest prover sends (it must be stricter only), and which difftest holds the Python twin. The precedent is REDRAW's `commitOf?` branch (`cursor/proofs-redraw-commitof-4e9d`): read its commit for the pattern. The same sizing for `ReadsWf`'s object and the `ports` object of `PortsDistinctAt`, in one line each: do they fall to the same switch?
3. `PartReadsBack.lean` (with `Good` as a hypothesis): keep it. `ReadsBack.registered`'s restatement: yours, after item 2.

