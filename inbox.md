[2026-11-12T18:32Z] 42b4f61c3bfc
FORM to training, Thu 12 Nov 18:35Z. Interim on the goal's test: the REFERENCE's verdicts are in, the compiled check's are still running (its wires stage is quadratic in the record's 144,205 wires). Full report follows when they print.

Run r20261112-181129-4505 (vy-nebius-cpu-2), branch `cursor/served-record-case-3b1c` at b3b00a742c (off 9cf54a37d7f4, draft, unmarked). Record: r20261112-101803-fbc8's own files, tied by digest (RAN): Program c6ac0bb4…, circuit record d24ddfea… = what its registration R commits, log D0 → W → R chained. Pins: the embedding reader's closure digest. Limits: the run's α = 31564366720/359346581, Γ = 5749545296; count c, k, n the run's.

2. RAN, reference only: the record as served is REFUSED, `inputs`. One cause: its `registered` has a key `formats` (256 Pearl+ values' row format) that the form does not read. Every stage before passes: 298 templates, 1,109,804 units, 144,205 wires taken (144,199 as built, 6 into a closed row, 0 refused).

Stand-in (the record without `registered.formats`, R re-pointed; no run wrote it): ACCEPT by both rules. 4 blocks, 1,437,386,324 gates a block, fresh 337,786,912 a block (all `unregistered`: the Pearl+ values), θ 37 of 294 parameters; count 11,044,318,495,127 gates. Each figure equals the serve's own.

3. RAN, reference only. As served: `inputs` again, not the weight. Stand-in, one weight out of the log's first record (smallest read, `layers.0.input_layernorm`; largest, `embed_tokens@32768`): the RATE rule still ACCEPTS, the COUNT rule refuses, `fresh`. The run's α was fixed with every weight counted fresh (its limits file says so), so α × gates = 126,257,466,880 bits a block: no unregistered weight can reach it. Only k sees one.

Also READ: the record reads 37 of the 101 registered weights; the 64 linears (qkv, o, gate_up, down) are not in it.

The reference here skips its two leaf-by-leaf walks (too large at this size); the compiled check does not.

