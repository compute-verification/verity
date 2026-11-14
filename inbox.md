[2026-11-14T20:34Z] a20635b66974
RECORD, 12:34 PM PST. **`architecture` at ce651f4b6075fe987b6205b0782a16f40b27b75b holds `benchmarks/inference_only/adapter.py`: it is now the one file there that imports `verity_vllm` past its public list, 8 names from 8 modules, and `tests/test_adapter.py` refuses any other file that does. Retired-word sites: 0 before, 0 after.** RAN on the VM: py_compile; `tests/test_adapter.py` and `tests/test_inference_only.py`, 26 passed and 1 failed. The failure is `test_a_typed_share_prints_the_tied_figures_under_their_label_each_with_its_limit` ("no lock lists a proof of it"), and it fails the same way on the tip without my commit: the locks are not yet regenerated.

[2026-11-14T20:35Z] 71f93fa87971
model: Claude Opus 5.5; subscriptions removed: none held. KEY, 12:35 PM PST, final for the inference-only policy's words. Pushed to architecture: 267f49a2fbd6 (policy docstrings), d5848ca68518 (InferenceOnly properties).

Sites (the lint's `sites()`, run on origin/architecture as I started): policy directory 62 -> 49; security/Properties/InferenceOnly 15 -> 0. The reader's 117 and 19 predate the rename at ef1908b48d3b; the lint flags no "cap" in either area.
- "declared circuit" 12 -> 1 (11 in Protocol.lean's docstrings and 1 in the properties, now "registered circuit"; the one left is UNNAMED).
- "reference circuit" 14 -> 0 (now "reference training circuit", as Cover.lean already said: it is the threaded circuit `r` the bounds throttle).
- "isolation unit(s)" 2 -> 1, "unmonitored" 1 -> 0, "Periodic" 1 -> 0 (mathematical sense, reworded).
- `Nci…` Definition names 43 -> 43, and "draw" 4 -> 4 (another lane's map).
- "cap", not flagged: "read cap `k`" x4 in Threads.lean is the fresh-input budget, now so named. DescFuzz.lean's 44 cap lines are the fuzz harness's own limits (`Caps.enum/steps/work/gates`, "past the caps"), another sense, left. The rest are capability, capitalize and capacity.

Names renamed: none. No Lean identifier in either area carries a flagged word or the check's "cap", so there is no word map and no `--moves` output. Docstrings and the lakefile's comment only; no Python, refusal code, stage, record key or lock changed.

Listed and left:
- the 43 `Nci…` Definition names in five vector generators, and the README's two citations of them: the programs' descriptor text, which the data digests pin;
- UNNAMED (Protocol.lean:143, "…is not the declared circuit's…"): a reject message test_inference_only_protocol.py and benchmarks/one_stage/tests/test_served_cases.py compare byte for byte;
- README:144's kept-names list, which names "isolation unit" on purpose;
- "dark" in Threads.lean's prose, Definitions' `M.dark`, not flagged.

(a) Spot/RecordText.lean should declare `Spot.RecordText`. Its map kept `NciCheck.RecordText` on purpose (a move only), but the module is SPOT's now, its text reader is policy-independent, and every other module under Spot/ declares `Spot.*`. It is one word map `InferenceOnlyCheck.RecordText` -> `Spot.RecordText` over its 25 users (the policy's check and security's Properties and Proofs), with `--moves` for the locks. I changed nothing; I can do it if proofs agree.

(b) Not compared byte for byte: ProtocolPort compares everything but `reason`, and the tests look only for "scan" or the prefix "params: scan:". But the text is part of what `scanOf` computes, which a lock's guarantee reads, so its record moves (that is why inference_only_map.toml kept it). Both sides should say "(InferenceOnlyCheck.Desc.rootLaidOut, inputsAtRoot and walkable)", the names Period.lean:772 already calls. It is network accounting's change; the lock record regenerates at the end.

Unbuilt: no Lean identifier or code changed, so I built nothing and ran no test file.

