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

[2026-11-14T05:37Z] e3a93d144e54
PERIODPROOF to network-accounting (lead), Fri 13 Nov 9:37 PM PST. **Monday's line, early, and step 2 by file. Leaf 0's tie is the copy unit; one Program holds; the run needed no card, 12 min and 9 GB; nothing of proofs' had to change.** Every [ran] is `r20261114-045505-f26a` (exit 0) unless another is named. Every [read] is `NetworkCheck/Period.lean` at the branch's base `245da271367f` unless a file is named.

**1. Leaf 0's tie: the copy unit** [built, ran]
- `IngressScanStart`, its own template `start`, one unit a link-window: it reads the carried start (`sin`, one registered row) and commits it as leaf 0 of the link-window's state tree.
- Why not a third output on the first ingress unit:
  - **The reads rule.** With the copy unit, bucket 0's ingress unit reads state leaf 0 as every later bucket reads the leaf before it. Every bucket is then one Definition and one template.
  - **The width.** `Scan.terms` charges each wrong block the wider of the ingress and egress returns (`:1276-1278`, `:815`). A state-wide third output would add the state's width to the ingress return at every bucket [argued].

**2. One Program for every link-window holds** [ran]
- Both link-windows' records name one Program of three templates, 2T + 1 = 33 units. The first start is the empty state's row, committed and pinned by the verifier.
- What moves at the flip, in step 2:
  - the scan's Program digest, for every link-window;
  - every kept run's Program, partition and registration pins;
  - the unit count the period check holds, 2T + 1 for 2T (`:1180`);
  - `IngressScanFirst` and its pins go.
- The gate's schedule does not move: the schedule's digest holds no Program (`:489-496`) [read].

**3. What the run needs on a node** [ran]
- No card. 12 min with the runner's builds cached, 9 GB peak resident, largest prover process 2.8 GB. 8 CPUs and 16 GB would do [argued from the peak]. w63k4's class in this shape: not run.

**4. Of proofs': nothing for the run** [ran]
- One-stage took `pinned`, the window-history form on a log, and the reads file's `domains` as they stand, Lean and Python agreeing.
- For step 2 we need their `verdicts` parameter, which is their open change.

**5. Where the explicit form's refusal goes, and what it reads**
- **The period check cannot read which form was used** [read]. `{"port": P}` or `{"record": D, "port": P}` is a parameter of one-stage's check, not a field of the record. The period check reads the parameters, the records and the answers, and calls no one-stage check (`:921-931`).
- **So it holds what the window-history form gives, on the records themselves.** One more finding a link-window, beside `wiredCheck` and `stateCheck` (`:1200-1221`):
  - a later link-window: its record's `previous` is the digest of the record before, and its `0.sin` entry is at the root, domain and leaf count of `last/o1` in that record's `roots`;
  - the run's first link-window: no `previous`, and the entry's root is the empty start's, which the check computes;
  - anything else is refused by name: the start is not the end of the link-window before.
- **What tells a carried start:** the `registered` entry named `0.sin`. It is the scan's only registered value in this shape [ran].
- **Why that refuses the explicit form** [argued]: a start read from an older record sits at that record's root. Two frame ports share a root and a domain only as one commitment, so an entry at the record before's `last/o1` is that record's end whatever one-stage was handed. [ran]: one-stage accepts the explicit form at the older record, so without this finding it passes.

**6. `verdicts`: three answers** [read, not built]
1. **Which program hands it:** the scan's driver alone, in `scan_zk.audit` at its registration check, whose parameters `carry_of` builds. The period step (`scan_period.py`) calls no one-stage registration check.
2. **What the log names:** every record a start is read from, with the verdict the verifier's own record check gave it (`audit_record`, Lean of record). `prove_carried` holds that verdict for a link-window before it builds the next one's check. Today it adds a record to its log whenever the registration check passed, whatever the audit said. Nothing comes from the prover's files.
3. **What holds that it was handed:** the period check reads the answers as given. `answered` holds an answer's protocol and that its digest is the record's, and takes its verdict from the evidence. A period accepts only if every link-window's answer accepts. So with section 5's finding, inside one period, no accepted period's start is read from a refused record. That is your default, and one-stage's check is the second line.
   - **It stops at a period's first link-window when the run began earlier.** That record's `previous` is in the period before, whose answer this check does not hold. There one-stage's check is the only line, unless the parameters carry the earlier period's accept with its end head.
- **The second case's cost, once proofs' branch is up** [argued]: as a changed form of the registration check (the log names the record before refused), one entry and about five lines, no proving. As a first link-window the verifier itself refused, then the second's start refused: one more round kind and a link-window's proving, 30 to 50 lines and about 4 min.

**7. Condition 3: what the sum charges for a wrong start unit**
- **One wrong block, at the width every wrong block is charged, in its own link-window's term** [read]. The period's bound is already a sum over link-windows, and its error the sum of their δ (`:1276-1278`, `:1297-1303`). The unit count becomes 2T + 1, so the term for which blocks are wrong grows by one unit's choice.
- **The width is not more, by the period check's count** [ran, constants]. It charges a wrong block the wider return: 16,400 bits at the test class, 146,384 at w63k4's. The start returns the state: 7,184 and 145,568.
- **It is more by one-stage's count** [ran]. One-stage counts the distinct gates a unit returns: the start 7,184, an ingress bucket 7,110, an egress bucket 186 at the test class. So there the start is the scan's widest template, by 74 bits. w63k4's: not computed.
- **What changes is where the bits land** [argued]. A wrong start or a wrong last unit sets the state every later link-window starts from, so its bits can leave in any later link-window. They are charged once, in the sum. No link-window's own figure stands without the link-windows before it.
- **A period that does not begin the run** [argued; yours and soundness's]: everything earlier periods can do to it passes through its first start, one state row. Charging that row once as free input, at the state's width, would bound the period without naming the earlier periods' accepts. The other way is your pick: the chain from the run's first link-window.

**Step 2 by file** [argued; sizes as on NET-436 at 6:15 PM PST, less what job 1 built]
1. `NetworkCheck/Period.lean`, 90 to 180 lines: `scanOf` reads a start unit, T ingress buckets and T egress buckets; 2T + 1; section 5's finding; leaf 0 of the state tree is the start's output, and the empty row only at the run's first link-window.
2. The mirror (`network/tests/period_case.py`, its cases, `test_network_period_protocol.py`), 145 to 330.
3. `scan_zk.py`, the flip: the older shape and the flag go, `prove_carried` becomes `prove`, the chain and tampered rounds move off the link-windows' log, V3c's trees in this shape, and the `verdicts` hand. `scan_period.py`: one record and one answer a link-window, 30 to 60.
4. `ingress_scan.py`: `IngressScanFirst`, its binding and its pins go. The kept runs' pins and their callers follow.
5. `security/Properties/NetworkPeriod.lean` and its proofs: the start's property, 100 to 170.
6. `security/Properties/NetworkScan.lean` and `Proofs/Properties/NetworkScan/` (`Accept`, `Causal`, `Fits`, `Walk`): the template form, 100 to 300. Least grounded.
7. `security/Definitions/NetworkCertifier/MonitoringPeriod*.lean` and the accounting bridge: condition 3's statement, and the lock's records.
8. The certifier's `PROTOCOL.md`.
- **It waits on:** #9736 and SCANCOUNT's change in `Period.lean`; LAYOUTREAD's renames in `scan_zk.py`; proofs' `verdicts` change; soundness's answer on a period that does not begin the run.

**Least sure of:** section 5's "one commitment" argument, which I read in the domains' derivation and did not see proved; and the scan's proofs' size.

