[2026-11-12T19:12Z] b067e97484c0
FORM to training, Thu 12 Nov 19:14Z. 2 and 3 are in, compiled and reference. RAN unless marked.

Runs on vy-nebius-cpu-2: r20261112-184716-2546 (compiled form, once a case), r20261112-181129-4505 and r20261112-182947-4e85 (reference; the difftest executable on the record as served). Branch `cursor/served-record-case-3b1c` at 49c796ad6f, off 9cf54a37d7f4, draft, unmarked. Record: r20261112-101803-fbc8's own files, tied by digest to its registration. Pins: the embedding reader's closure digest. Limits: the run's α = 31564366720/359346581, Γ = 5749545296.

2. The record as served: compiled REFUSES, `inputs`. Reference: `inputs`, and equal on every compared result. Wires: 144,205 of 144,205 taken (144,199 as built, 6 into a closed row). Refusals by rule: `inputs` 1, every earlier rule 0.
Cause: `registered.formats`, a key the remote branch's record carries for the 256 Pearl+ values and the form does not read. Without that key (a stand-in, no run wrote it): compiled ACCEPTS, reference equal. 4 blocks, 1,437,386,324 gates and 337,786,912 fresh bits a block; ×4 these are the serve's own figures.

3. One weight out of the log's first record, each later `previous` re-made.
- As served: `inputs`, both. It refuses for the key, not the weight.
- Stand-in, smallest read weight (`layers.0.input_layernorm`) or largest (`embed_tokens@32768`): the rate rule ACCEPTS, compiled and reference. The count rule refuses, `fresh` (reference; compiled prints by 19:40Z).
- Why: the run fixed α with every weight counted fresh (READ, its limits file). With α = k over a block's gates, the rate rule refuses, `nci`, compiled and reference.
- A weight the record does not read (`layers.0.o_proj`): ACCEPT, nothing changes. The record reads 37 of the 101; no linear is in it.

4. Owners. `formats`: ours, the form reads the key (two lines and cases), a follow-up. α: vllm's, fix it with the weights held, which tightens it. Nothing for network-accounting.

READ: #8676 merged at 18:10Z; main's package is file for file the head's. The same nine cases at main: r20261112-191043-6003, verdicts by 19:50Z.

Yours to decide: whether the goal's check is the count rule (it refuses today) or the rate rule (needs vllm's α).

