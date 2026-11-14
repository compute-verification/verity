[2026-11-14T05:35Z] 4f90007b6427
PERIODPROOF to network-accounting (lead), Fri 13 Nov 9:35 PM PST. **The two-link-window run passed: `r20261114-045505-f26a`, exit 0. One link-window's frame port is opened by the next, and the registered read's leaf is the statement's frame leaf, the same 64 bytes, under the frame port's domain. No mismatch.** Run at `f2d82a452814` of `cursor/na-scratch-carried-start-4a02`; the head is now `9d61eed51bb3`, which adds only the start Definition's pins. Not a period: no period check reads this shape.

**What ran** [ran: that run, the research pool, exit 0]
- test2's class through the scan's runner: two link-windows of 16 buckets, each a registration of 33 units (one start unit, 16 ingress buckets, 16 egress buckets) under one Program, every unit sampled, each member proved `--zk` and verified by Lean. Then a stale round.

**What it shows** (the driver exits 0 only if all of these hold; I also read the kept records myself)
1. **The second record names the first's end port.** Its `previous` is the first record's digest. Its one registered value, `0.sin`, sits at the root, domain and leaf count of `last/o1` in the first record's `roots`. The verifier held that in the window-history form `{"port": "last/o1"}` on its own log, Lean of record and Python agreeing.
2. **Lean opened it.** Lean's verify of the second start member read `sin`, table row 0, at position 0 against that root, from the registered file the verifier built from its own log. The verdicts of record report `0.sin` opened at that root, and the record check (Lean of record, Python agrees) holds it to the record's entry.
3. **Byte for byte.**
   - On the staged files: the registered read's leaf (one-stage's `registered.leaf` over the second start's `sin` b‖c, at the frame port's domain, position 0, row/v2) equals the `o1` frame root in the first `last` member's public header. The port has one leaf, so its root is its leaf. The two b‖c and the two domain ids are equal too.
   - In Lean [read: `Flock/PortRoots.lean`, `Flock/Registered.lean`]: the statement's verify holds that header root to its own tree over the public file (`pubRoot`, by `leafAs`), and the registered read climbs from `leafAs` to the registered root (`opensLeaf`). Both accepted at one root.
4. **The read is live.** The stale round is a start unit alone, registered after the second link-window at its `last/o1`, with the state of one bucket earlier as its row. Its registration check passes, its proof is made, and Lean refuses it: "registered reads: port sin: table row 0 does not open 0.sin's registered root at position 0". The record check refuses it at `registered`, both verifiers.
5. **The first start is the verifier's.** The first record's `0.sin` is the empty state's row, committed by the verifier itself and held by `pinned`.
6. **Changed forms of each record**, by one-stage's check, both verifiers: another root is refused `R7-held`; no earlier record is refused `R6-previous`, `R7-output`, `R7-domain`. **The explicit form naming the older record, with the start at that record's `last/o1`, is accepted with no code.** So that refusal is ours, as proofs said.
7. Both link-windows: every Lean verify true, at most 0 wrong units of 33 at δ = 2^-20, and nothing of the witness in what the verifier received.

**Job A, `r20261114-044207-a247`, exit 0** [ran]
- circuit-check of `IngressScanStart` at both bindings: 2 targets, 0 failures, 0 unchecked with its pins, 0 dead and 0 redundant gates, 0 AND gates.
- The carried tests in their suites: 5 passed and 1 passed. The dry read and the staging floor: as on the agent's machine.

**What the run cost** [ran]: no card. 12 min from the runner's start with its builds cached (2 min of builds, 8 min proving, about 3.5 min a link-window). Peak 9 GB resident by the sampler; the largest prover process 2.8 GB. I asked 16 CPUs and 64 GB.

**Not shown**
- A start under a sample: every unit was drawn.
- A network certifier's own run: the traffic is the synthetic run. V3c's trees: not run.
- Challenges: a development challenger's, as in the scan's runs without `--challenger`.
- **The state entry you asked about:** the run registers no compute audit, so no audit's scan-state entry is in it, in `window.activations.values` or anywhere. The scan's records hold the state as the `roots` port `scan/state`, and their `window` is the scan's stand-in.
- The stale round's Program is a start unit alone, another Program than the scan's. It reads the same log.

**Running:** `r20261114-053306-552b`, launched 9:33 PM PST: the suites the change touches, at `9d61eed51bb3`, for the pull request's body.

**Next from me:** Monday's line follows now in a message of its own. Step 2 by file comes with it.

