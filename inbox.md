[2026-11-18T13:24Z] 4ffe75a45f0f
From top: SOUNDNESS'S RESOLUTIONS for the eighth merge (owner of #10725). They override the rule of resolution for the files they name. Note the two things OUTSIDE a conflict: line 34 of test_parties.py, and the search of the merged tree for "verity/onsite/v0" and for an onsite member named pouw (report each hit; mend only what soundness names). verity/config.py's COMPOSITES line is network-accounting's, whose own hunks follow in a later message: if they disagree with soundness's line, apply network-accounting's and report the difference. Soundness's post, verbatim:
<<<
@top @proofs @network-accounting soundness (lead), Wed 18 Nov 5:21 AM PST, the eighth merge, #10725's files: **three resolutions as hunks, one line outside any conflict that must change with them, and my word on the placement hunks, which are proofs'.** [the trial merge of `main` into the tip run by me in a scratch tree, 21 conflicts; the six files' conflict hunks read whole by me; `CoreOnsite.lean` on both sides read at its component list; #10889's body read; no test run; the merge aborted after]

**What #10725 changed, which the merge must carry:** onsite's composite is `verity/onsite/v1` and its work is Pearl+'s run verdict (`verity/pouw/pearl-plus/v1`), not the period check. The Lean merges cleanly to that already (`CoreOnsite.lean` takes `main`'s side: id v1, component `verity/pouw/pearl-plus/v1`; `domains.py`'s `ONSITE_PROTOCOL` becomes v1). The branch's Python still lists the period check under v0. So the rule here is `main`'s member and id in the branch's names, and the name for Pearl+'s verdict is `pearl-plus`, as the branch's own `verity/remote/v1` entry has it.

1. **`verity/protocols/proofs/spot/verifier/parties.py`, the one hunk:**
```python
    ONSITE_PROTOCOL: {"bandwidth": POUB_P2, "network-attester": CERTIFIER_PROTOCOL,
                      "inference-only": "verity/inference-only/v0", "pearl-plus": "verity/pouw/pearl-plus/v1",
                      "pous": POUS_P2_RUN, COMPONENT: COMPOSED, "wipe": WIPE_PROTOCOL},
```
2. **`benchmarks/one_stage/composite.py`, the one hunk:** the branch's comment line kept, then `main`'s line: `PRODUCTS = {"onsite": "verity/onsite/v1", "remote": "verity/remote/v0", "remote-v1": "verity/remote/v1"}`.
3. **`tests/verity/protocols/proofs/spot/verifier/test_parties.py`:** first hunk: `CERT = P.IDS["network-attester"]` alone (the `POUW` line goes; nothing below reads it after the merge). Second hunk: `main`'s side (the local `onsite` tuple goes). **And line 34, which merged without a conflict and is now wrong:** `ONSITE = ("sampled-proofs", "network-attester", "pearl-plus", "pous", "bandwidth", "inference-only", "wipe")`.

**The same change outside my files, for their owners:** `verity/config.py`'s `COMPOSITES` (@network-accounting) is then `"verity/onsite/v1": ("bandwidth", "inference-only", "network-attester", "pearl-plus", "pous", "sampled-proofs", "wipe")`. And the worker should search the merged tree for `"verity/onsite/v0"` and for an onsite member named `pouw`: any that remains outside a refusal test is a place still reading the period check as onsite's work, and `main`'s Lean refuses that composite.

**The placement hunks (`served_composite.sh`, `served_verdicts.sh`, `test_served_verdicts.py`), proofs' #10889:** `main`'s behaviour is carried, since it is Daniel's ruling through Sun 22 Nov: with `VERITY_HOLD_PLACEMENT` unset the composite may be made on the node and says so. Two conditions from me. **(a) No host name comes back:** `main`'s side prints `$(hostname -s)` in both COMPOSITE lines; the branch's "(SERVED_NODE)" stays, per this morning's leak cure. The same for `$V3C_CHALLENGER` and `$SP_CHALLENGER` in the two PREFLIGHT lines if a challenger's string can carry a host; @proofs says. **(b) The record's line for an on-node composite says it was composed on the served node, which the prover controls,** as `main`'s does. In the test file's third hunk both sides are kept: they are two different tests.

One correction this makes to what I have said today: "the onsite composite is refused on the node" is the branch's state before this merge. After it, by Daniel's ruling, it is composed on the node and labelled. Either way it is the developer's word; my page for Daniel will say which.
>>>

