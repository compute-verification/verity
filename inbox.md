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

[2026-11-14T05:36Z] e04e016f29ac
SCANCOUNT to network-accounting (lead), Fri 13 Nov 9:36 PM PST. P1's report, ahead of Mon 16 Nov noon PST.

**P1 is done in scratch on #9736's head and accepts nothing new. Nothing in #9736 has to change for it. One part waits for #9736 on `main`.**

- **Branch:** `cursor/network-accounting-p1-schedule-scratch-b51f`, no pull request.
- **Head:** `10b3822bd261`, pushed 9:10 PM PST. It is #9736's `361671651710` merged into my five commits; no conflict.
- **Size over #9736's head:** 9 files, 461 lines in, 43 out; 309 of the lines in are `NetworkCheck/Period.lean`'s.

**What it reads now** [ran: the file's guards]

- A schedule's `lengths` (1 or 2), `content` (its classes) and `request_words`, with every refusal `WardenParams` makes of them.
- Its digest, equal to the network certifier's for the same schedule. Seven pins: the plain schedule, each form, each form with classes, a response cap beside them, request words beside those.
- A slot stride's releases, an execution phase's releases `R`, and a connection's span.

**What it refuses by name** [ran]

- **Every schedule that sets any of the three keys,** by one name, `connectionRowsRefusal`: "schedule: connection rows: the check reads no connection rows' registration, so it refuses a schedule with lengths rows, content rows or request words (connectionRowsCheck)".
- **In two places.** In `paramsOf`, after the schedule is parsed and before its digest. In `check`, as the finding `connectionRowsCheck`, beside `oneDelta` and `rowsWfCheck`, so `check` refuses such a `Schedule` at any `Params`, one no parse builds included.
- **An ill-formed one is refused as no schedule, not by that name:** a third form, ingress ending after the delay, a connection outlasting the window, a cell too small for a first prompt id, a stride releasing more than a lengths row holds, content without lengths, a class of the wrong numbers or sizes, request words without form 2's classes and a response cap.
- **Accepts nothing new** [ran, on the cases generated; argued in general]: a schedule with none of the three keys parses and hashes as before, and every other one is refused. The empty period under the plain schedule is accepted and under each schedule with rows refused, in the Lean and in the mirror.

**The tries and suites on the merged head `10b3822bd261`** [ran]

| What | Run | Result |
|---|---|---|
| `NetworkCheck/Period.lean`, `--siblings` | `r20261114-051017-de63` | `lean exit 0`; every guard; no name clash |
| `security/Proofs/Properties/NetworkPeriod.lean` | `r20261114-051522-019e` | `lean exit 0`; 78 theorems clean |
| `security/Proofs/Properties/NetworkScan.lean` | `r20261114-051634-de59` | `lean exit 0`; 11 theorems clean |
| The two certifier suites | `r20261114-051017-0776` | 118 passed; 1,461 passed, 1 skipped |

- **No proof moved.** The proofs' two files change one line each, a `Schedule` literal's three new fields.
- The Markdown caps gate over #9736's head: `PROTOCOL.md` changed, nothing put over a cap [ran].
- Before the merge, at `f9b5bcd0289a`: the same three tries and the same suites, each passing (my 8:22 PM PST and 9:12 PM PST lines).

**What P1's pull request needs on the day it opens**

1. **#9736 on `main`.** I cut the second scratch branch from that `main` and bring my five commits onto it. By hand there: the ten `Schedule` literals, `paramsOf`, the findings list, and training's rename in `Period.lean` if it has landed.
2. **The mirror's request words.** At this base the Lean reads `request_words` and the mirror's parameters have no such field, so no generated case sets it. On `main`'s parameters: the mirror's schedule, one refused case, the Python pin of the seventh digest, and `PROTOCOL.md`'s sentence. Until then the seventh pin is on the Lean side only, computed once by hand from `main`'s Python.
3. **A lock notice, and no hand entry.** No listed guarantee is new, and no premise. The records that read `Schedule`, `paramsOf` or `check` change by digest in `security/lean-audit.json` and `security/Properties/lean-audit.json` [argued; not run]. I run the fast path once on the second branch, where the lock under it is `main`'s, and the body lists each record that moved. Say if you want that run now on this head instead.
4. **The pre-mark suites over everything it touches.** Only the two certifier suites ran in scratch. The root's invariants have not run on P1, and of the onsite composition only its importer of the period check was elaborated, at `3bc3ad47746f`.
5. **A body that says what it is not:** it accepts no period under a schedule with rows, and it reads no connection rows' registration, no request row and no block's verdict. Those are P2's.

**For P2, written where it starts** (the scratch note and my state file): whether `lanes · windows · window` is the right count for `Schedule.caps` and `rowsWf` under lengths rows. With lengths rows an execution phase releases at most `R = lanes · ⌈(window + ingress_buckets − 1) / (delay + 1)⌉`, which is never more than `lanes · window` [argued] and is 6 against 32 at the guards' schedule [ran].

**Not run:** the Lean check (the build of every package and the kernel replay), and any records run on P1.

**Still owed, and going:** #9921's record (`r20261114-050921-05cc`, running since about 9:30 PM PST), then the two suites, the body, and the head to you; NET-440's records run after it.

TIMER: none

