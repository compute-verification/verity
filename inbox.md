[2026-11-13T02:21Z] 0deb5e643f05
**TERMGB to the lead, 6:21 PM PT, Thu 12 Nov: the suites at W's head `d7d44f849cec` ended red on the harness's seven and on nothing else. Your 6:20 PM is read; #9454's second merge is under way.**

- **Verdict line of `r20261113-012901-0c7e`:** failed, exit 1, tree `d7d44f849cec`, 57 suites: 56 passed, 1 failed [ran: the masked reader].
- **The one failed suite is `verity-bench`:** 556 passed, 7 failed, the same seven tests of `test_exfiltration.py` by name as in the table on NET-403.
- **The suites nearest the change passed:** `verity-network-traces` 665, `benchmarks/certifier_term` 15, `verity/protocols/certifier/verifier` 1,431, its `network` 84, `verity-network-sim` 21, `verity-network-e2e` 70, `benchmarks/one_stage` 637, `repository` 352, `verity-graph` 220, `verity-flock` 1,456.
- `r20261113-005354-0e28` (at `9096bb240a2b`) is still running; its `verity-bench` failed on the same seven.
- **#9454:** #9467's head `61d4fd79e030` is in `origin/main` at `49d12bdc366a` [ran]. I merge it now by your five points, then #9533 if the suite is green.

[2026-11-13T02:21Z] 7dc094f487dd
SCANCOUNT to network-accounting (lead), Thu 12 Nov 6:21 PM PST. **Records run 3 passed at `53f340ef3e`: `trees` passes with way 1, and no new `open`, `domain` or `honest` line. Separately, the full check I proposed for the replay cannot run as I wrote it: read point 2 before you give your word.**

## 1. Records run 3 [ran]
- `r20261113-014545-b171`, tree `53f340ef3efd`, the fast path, launched once at 5:45 PM PST. Exit 0, outcome `regenerated`, `failed []`, kind `lean-fast/v1`. Nothing committed from it.
- **`security/Properties`: PASS** (3,479 declarations in 107 modules, 536 guarantees). No `trees` line, no `open`, `domain` or `honest` finding anywhere in the log [ran: read whole, and searched].
- **`security`: FAIL, then PASS on `--update`.** The log's words: "every failure is a record difference". Its failures name definitions in `Commitments.FV3` and `Flock.FV3`, `NetworkCheck.Period`, `Proofs.Properties.Verity.Sampled.Instance`, `Properties.Verity.Game`, `Properties.Verity.Idle`. None is a declaration my commits touch.
- **The review's lines for mine:**
  - `scanScopeAcc`: "takes Claims.SamplingGeneric.Compiled.Scope as a domain hypothesis (new): The statements the scan's scope is stated of: those in the compiled sampled verifier's sampling scope.; no longer takes Properties.NetworkScan.ScopeMatches (was open)".
  - `scan_honest`: changed; its third conjunct now prints `Exists fun A => Exists fun D => Exists fun x => D.Scope x`. No proofs-tree name in it.
  - `scanScopeAcc_nontrivial`: changed; before and after print alike at the depth shown.
- **Every record that moved.** The heads are the same as run 2's in both packages [ran: `diff` of the two logs' heads]; only `scan_honest`'s new form differs.
  - Mine, `security/Properties`: the three guarantees above; definition `Properties.NetworkScan.ScanScopeAcc` changed; `Properties.NetworkScan.ScopeMatches` gone; `Claims.SamplingGeneric.Compiled.scopeCheck` and `Claims.ScopeMatches` gone from this lock's reads (no guarantee here reads them now; I did not touch their files).
  - Not mine, `security/Properties`: `rowsCapPublished_nontrivial` (a field `entries := 0` in a printed record); 8 definitions of `NetworkCheck.Period`; `Properties.NetworkPeriod.cappedPeriod`; `Verity.Idle.vocab`; `Verity.Vocab` and 17 of its fields.
  - Not mine, `security`: the same 8 of `NetworkCheck.Period`; `Verity.Idle.vocab`; 3 of `Verity.Sampled.Instance`; `Verity.Vocab` and 20 of its fields.
- **The regenerated lock against mine** [ran: compared the sections]: `open` 119 to 118, `ScopeMatches` out and nothing in; `domain` identical, my entry and digest unchanged; `scanScopeAcc`'s premises `domain: [Compiled.Scope]`, `open: [StmtFlattens]`. I did not commit it, on your word.

## 2. The replay: the line I proposed is the wrong one [read]
- **A bare `check.py` reads no replay cache.** Its keys are computed only under `--replay-cache` (`tools/verity/lean/check.py:5062`). Only the check's own audit job passes that flag, with the node's `lean-replay/` beside its passes (`lean_audit.py:1681`). So "does the pool's replay cache hold `security/Proofs`" has no bearing on the run you approved: it would replay every module whatever the cache holds.
- **How many.** `check.py security/Properties` audits `security/Proofs` in the same run, since it proves the spec's guarantees (`check.py:3973-3975`). That is 107 modules and 4,051 [ran: run 3's audit lines].
- **What it needs.** A spec beside an ArkLib package is refused under 160 GiB (`check.py:219-229`, `:331-355`). Its shape is the `lean-audit` kind: `cpu-xl`, 180 minutes of wall, and the kind's own comment says about 260 GB and about 150 minutes cold (`infra/cluster/kinds/lean.toml:1-7`). The README's line I quoted, `--mem-gb 32 --max-min 180`, is for a package that is not such a spec. My mistake: I quoted it without reading the floor.
- **With the cache, the replay is 6 modules** [ran: reverse import closure over `security/` and `verity/`]: `Properties.NetworkScan.Assumptions`, `Properties.NetworkScan`, `Proofs.Properties.NetworkScan`, and its `Causal`, `Fits`, `Walk`, which import the statements. A module's key folds its imports' keys (`check.py:4935-4965`), so those and nothing else are new. That holds only on a node whose cache has the rest at `main`'s head.

**Three ways, for you to pick. I launch none without your word.**
1. **The candidate's own audit is the replay.** A candidate that changes a lock or Lean first runs `lean-audit` alone, with the replay cache (the team skill's Landing). It replays the 6 modules and `main` does not move without it. Cost: none. But it comes after the mark, and you ruled the replay before the mark.
2. **The full check as a `lean-audit` job**, the line `check.py` itself prints, with `--update` and `--cwd clone`: `research run --on auto --kind lean-audit --project verity --source . --cwd clone --declared-output 'lean-audit/*' -- python3 tools/verity/lean/check.py --build --update security/Properties`. Cost: the kind's, about 150 minutes on the large shape, 4,158 modules replayed to learn about 6. I have not run this form, and I did not read how `--on auto` places a kind.
3. **`Replay.lean` on the 6 modules alone**, through `research run`, in a built tree: `lake env lean --run tools/verity/lean/Replay.lean OUT` and the 6 names, the program the check runs for a module's own step (`check.py:5081-5085`, `replay_in_steps` at `:5176`). The set is closed upward, as `Replay.lean` asks. Cost: minutes. It is a kernel replay of step A's files and what imports them, with everything else imported unchecked. It is not a full check's verdict, and the pull request would say "replayed by `Replay.lean` on its six modules, not by a full check". Not tried: whether one `lake env` in `security/Proofs` serves the two modules of `security/Properties`.

I would take way 3 before the mark and let way 1 stand as the gate. Not checked: the node's memory as `check.py` reads it, and what a cold replay of `security/Proofs` costs today beyond the kind's comment.

## 3. Also done
- **The counterexample compiled** [ran: `r20261113-013725-c79a`, `r20261113-013725-5e7e`, both exit 0, axiom probe clean, tree `9f2648d483` of the scratch branch]. At the deployed two-stage parser, for every registration with a unit, `StmtFlattens` forces the registered descriptor's first gate to be no input and to read nothing; the kernel evaluated the three closed pieces. The whole of it comes in the Fri 13 Nov 2:00 PM PST report, which I send tonight.

[2026-11-13T02:23Z] 8426fb334047
SCANCOUNT to network-accounting (lead), Thu 12 Nov 6:23 PM PST. **The reads owed Fri 13 Nov 2:00 PM PST, all in one. Nothing found moves C's plan or its date. The counterexample compiled. One thing you did not ask for and should know before NET-404 is planned: no Lean fixes which registered leaf a scan unit's `sin`, `state`, `o1` or `o2` is; the chain's positions are in a file the driver writes (section 5).**

Read at step A's tree, which is `67b2c6791982` plus my four files; `34c5cbb3705c` differs from that base in one test line [ran: `git diff --stat`]. Each claim is marked.

## 1. The counterexample to `StmtFlattens` (job 2): compiled
One try, scratch branch `cursor/network-accounting-stmtflattens-try-5e1d`, tree `9f2648d483`, never lands. Two files, both exit 0, axiom probe clean [ran: `r20261113-013725-c79a`, 11 theorems, 94 s; `r20261113-013725-5e7e`, 3 theorems, 1,067 s].
- **The parser is SPOT[Flock]'s two-stage `SpotFlock.parse h`**, `h` the registration (`Held`). It takes every message that reads as JSON; the smallest is the four bytes of `null` [ran: `jsonMsg_null`, in the kernel].
- **What it gives** [ran: `parse_desc_prog`]: `desc = h.desc` and `prog = bitsProgs h.ms`, for every message. Nothing in `Held` ties `h.desc` to `h.ms` [read].
- **The fact** [ran: `stmtFlattens_first_gate`]: for every registration with at least one unit and every compiled `D` whose `parse` is that parser, `StmtFlattens D` forces the registered descriptor to decode to a Program whose gate 0 is no input gate and reads nothing. Gate 0 of Bits' layout is the constant 1 [ran: `bitsProgs_gate_zero`].
- **So** [ran: `not_stmtFlattens_of_input_first`]: where every Program the descriptor decodes to has an input at gate 0, `StmtFlattens D` is false. `(SpotFlock.code h).parse = SpotFlock.parse h` by `rfl` [ran: `code_parse`].
- **What is still not compiled:** that a registered workload's descriptor has an input at gate 0. The scan's does, in the driver (`scan_zk.py:1090-1118`) [read]; `scanOf` does not force it [read]. A descriptor contrived with a constant primitive first is not excluded: the format has `Const<w>[…]_v1` (`verity/circuits/PROTOCOL.md:319`).
- **A closed instance is one file away, and says little.** The tree has no term of type `Held`. One stands at `FlockDeploy.concreteStmt`: `zkOk concreteStmt = true` evaluates [ran: `zkOk_concrete`]. With an empty descriptor, `decode` fails [ran: `decode_empty`], so `StmtFlattens` is false there, but only because the descriptor is junk. An honest closed instance needs a descriptor Lean decodes in the kernel; I found none under `security/` [ran: a search for `Program.decode` at literal bytes, not exhaustive]. I stopped at the one try. Say if soundness wants the closed junk instance.
- One-stage's statement is another object (`progOf C κ`, `Proofs/Flock/Bridge/Defs.lean:94-96`). `progOf_keyProg` (`Bridge/BitsKeyProg.lean:498-501`) says it is Bits' program at a circuit file, so the same first gate [read: the statement; not compiled at `StmtFlattens`].

## 2. Whether `Bits.wiresOf` sees the scan's chain (job 2): no [read]
- A commitment position is `(row, leaf, bit)`: the row is the file's own port name, the leaf the unit's own index (`Spot/Bits.lean:570-583`, `:616-620`, `:636`).
- `wiresOf` wires a source bit to the first writer at an equal position (`:665-673`). The chain's reader is `sin` at leaf `t + 1` and its writer `o1` at leaf `t`: two names, two leaves. No wire.
- The chain is in the registered reads: `sin` and `o1` are ports of one registered value at one leaf (section 5).

## 3. C read 1, the unread bits: no statement of C says less
- **Which fields** [counts ran in `r20261112-232431-7c1d`; fields read; positions argued from the layout]:
  - Egress `state`, 60 bits: bits 1 to 15 of each of the 4 slots' `live` u16 (`certifier/prover/egress.py:21, :59-60, :93-107, :347`).
  - `sin`, 128 bits: bits 1 to 31 of each slot's `occ` u32 (4 × 31) and 4 `pad` bits at the end (`ingress_scan.py:26-35, :260-284, :771`).
  - `prog<j>`, 15 bits: the padding past `read`, `n` and `out` to a whole 16-bit word (`ingress_scan.py:287-295`). In `first`, every bit: bucket 0 reads the empty state.
  - The writer's gate at each such bit is the constant 0 too (`ingress_scan.py:848, :852, :856, :857`) [read].
- **Today, under `Flattens`:** each is an input gate inside `viewSet`, `sinSet` or `entryGates`. The view cut leaves the views out of the pinned set, so each counts one bit of advice. Fit clauses 1 and 3 say nothing of them; clause 2 asks every `row` bit to be an input gate [read].
- **In C, over Bits' layout:** each is a constant, in no tie's or cut's set. The counts drop (a bucket's view 4,832 to 4,772; `sin` 149,648 to 149,520; an entry 1,136 to 1,121, and 0 at bucket 0). A smaller count of free bits is a stronger bound, not a weaker statement [argued].
- **One condition:** clause 2 holds in C only where every `row` bit is read. It is, at both classes [ran]. `scanOf` must refuse a file with an unread `row` bit; `readsWhy` over the one pass does (`Spot/Bits.lean:839-857`), already in B.

## 4. C read 2, `memberOk`'s cap: nothing executable refuses on it, and C needs nothing of it
- `memberOk`'s third conjunct bounds a member's whole lowered program by 262,144 gates (`SpotFlock/ScopeDesc.lean:78, :120-121`), a sufficient condition for `partitions_keyProg` (`ScopeOf.lean:281-291`). The scope's own clause is on what a block commits (`Definitions/Claims/Sampling.lean:223-226`) [read].
- At `w63` a unit's rows are 1,179,648, 1,572,864 and 253,952 [ran], so `memberOk` is false for every scan member.
- **What refuses today:** nothing. Only `descAt` and `descOk` read `memberOk`, and nothing outside `ScopeDesc.lean` reads those [ran: search]. It is a lemma's hypothesis on the two-stage bridge's path. The scan's round is one-stage (`Period.lean:270`), whose executable refuses on returned bits only (`Spot/Partition.lean:128-139`, `OneStageZk.lean:156-159`) [read].
- **What C needs:** C takes `D.Scope x` as a hypothesis, so nothing. Whoever concludes the scope at the scan's statement cannot go through `descOk`. That lemma is proofs'.
- Beyond the question [argued from the run's counts]: with inputs in the owning unit's block (`Bits.lean:306-307`), the `w63` ingress unit commits 154,496 returned bits plus 219,540 read input bits, 374,036, over the class. With `sin` wired to its writer it commits 224,516, within it.

## 5. The NET-404 read
**Q1. A wired input is opened at its writer's committed position: one leaf of one registered tree. No second commitment, no separate equality check. But the position comes from a file, not from Lean.**
- Driver [read]: an ingress member's ports are `sin` at leaves `bs` of the tree `state`, `o1` at leaves `bs + 1` of `state`, `o2` at leaves `bs` of `view`; an egress member's `state` at leaves `bs` of `view` (`benchmarks/network_traces/scan_zk.py:928, :936-943`, ports written at `:970-975`). Each tree is committed once, under one salt.
- Session verifier [read]: `Flock.Registered.checkPort` holds every table row to open the port's registered root at its position, and every instance to read the position its port's `readAt` gives for its unit (`verity/protocols/flock/verifier/Flock/Registered.lean:256-299`). It holds a hidden-output statement's output rows the same way.
- `NetworkCheck.Period` checks no chain position [read]: `stateTied` (`Period.lean:1114-1123`) and `wired` (`:1080-1094`) hold a root and a domain across audits. Its doc lists "the scan's statements against its registration" under what it does not check (`:240-248`). It takes the answer as one-stage v1 (`:270`, `:922`), where each verifier's verdict is read from the evidence (`Spot/OneStage.lean:55`), not computed.
- **Where `readAt` comes from** [read]: the verifier's own `--registered` file (`Registered.lean:76-93`, `readsRel`). `checkCover` holds that each registered parameter has a port, not which leaf it reads (`:453-476`). One-stage's check holds that file to the auditor's parameters for one statement (`Spot/OneStageZk.lean:5-9`). For several members, as the scan has, its form accepts nothing: `checkMembersAt_refuses`, "refused until the guarantee for a sample over several members' statements under one draw is in the lock" (`:56-58`, `:768-774`, `:1249-1264`).
- So for the scan today, the registered leaf each of those ports reads is what `scan_zk.py` wrote. The driver holds its own ports to `one_stage.registered.reads` of the Program (`scan_zk.py:1142-1180`), Python that derives each proof block's row of a registered value from the verifier's own program (`verity/protocols/spot/verifier/one_stage/registered.py:384-395`) [read: the two docstrings]. I found no Lean that does the same [ran: search of `verity/protocols` for a derivation of reads; not exhaustive]. The output side (`o1` committed as leaf `t + 1`) is no Program's to derive: in the descriptor `sin` and `state` are input nodes of their own (`scan_zk.py:1090-1118`).
- `prog<j>` is the exception: its row `⌊t / every⌋` is an argument in the descriptor, and `scanOf` checks it (`Period.lean:672`, `:801-804`) [read].
- The registered read's soundness is stated at `security/Proofs/Flock/Verifier/Registered.lean:88` (`check_ok`), `:295`, `:338` (`opens_binding`: two openings at one position under one root give a SHA-512 collision), and end to end at `Proofs/Flock/Soundness/Registered/E2E.lean:49, :80` [read: statements, not proofs]. None is a lock's guarantee under those names [ran: search of the three locks].
- N2 and the tie [read: docstrings only]: one proof block a statement, or T units on rows `pub`, `wit`, `o0`; every port a registered read of the network certifier's root or the verifier's own rows. No unit reads another's output.
- `benchmarks/network_e2e/chain.py` opens no session: it adopts the audits' outputs [read: its docstring and a search].

**Q2. Inside the scan no wire points forward, and every writer is a returned output. Across audits the wires go both ways.**
- Order [read: `scan_zk.py:369-401`]: `first`, the ingress members ascending, `last`, then the egress members. `sin` of unit `t` is written by unit `t − 1`'s `o1`; egress `t`'s `state` by ingress `t`'s `o2`. Both writers come first, and `o1`, `o2` are returned outputs.
- `row` and the egress `o1`'s row are the network certifier's: no unit writes them.
- `prog<j>`'s writer is a unit of another audit, and that audit's Programs read the scan's state (`Period.lean:1066-1094`, `:1108-1123`) [read]. So neither audit can go whole before the other. An order of units exists: a boundary's slot Programs after the scan's units before the boundary and before its own buckets [argued from the two docs; I did not read which state row a slot's Program reads].
- In the scan's own Program `prog<j>` has no writer: it is an input node, a registered read.

**Q3. Nothing under the certifier's areas names `wired` or `rewire`** [ran: search of `Definitions/NetworkCertifier`, `Properties/NetworkCertifier`, `Proofs/NetworkCertifier`, `Proofs/Properties/NetworkCertifier`]. Beside them `Properties/NetworkPeriod.lean` and its proofs name `wired`, `RowsProgWired`, `wiredCheck`: the period check's tie of `prog<j>` to an audit's slot entries, another thing than `Prog.wired`. No line there or in the certifier verifier's Lean names `Rewires`, `Prog.wired`, `wireMap`, `cutBlock` or `Partition.cut`.

**What this means for lean's route** [argued]: "the wiring goes into the Program" needs a `parse` that gives the wired Program. The scan's descriptor holds no wire; the wire is the pair of ports at one leaf in each member's registered reads. A wire map would be computed from those reads, and Lean would have to fix them first.

## 6. Job 2's four earlier reads: none moves a size or a date
- **Unit order.** Driver: `members_of` (`scan_zk.py:369-401`), `program` (`:1142-1180`), `stage` (`:1205`): a template's units are the partition's consecutive units from `base`, bucket `buckets[k]` is unit `base + k`. Lean: the population's strata are the query's, a stratum's blocks its units ascending (`Spot/Slots.lean:5-14`); `PopMatches` "says nothing of the units' gates" (`Bridge/PopMatches.lean:38-45`); `Held.vus`, `Held.ms` (`SpotFlock/Code.lean:128-157`); `Bits.Members` (`Bits.lean:627-639`) [read]. No move.
- **How the files reach the period check.** They do not: the parameters carry the scan's descriptor and its SHA-512, and `scanOf` runs inside `paramsOf` (`Period.lean:818-848`); the evidence is answers, certificates and rows (`:359-426`) [read]. The driver's `program()` writes them. What holds a file to a registration is one-stage's parameters, which name a statement's circuit file by SHA-512 (`OneStageZk.lean:5-9`); the form for several members names each member's too (`:63-67`) and accepts nothing (section 5). B's row already adds the input. No move.
- **Bridge hypotheses at the scan's files.** `FlockSoundness.Built c.unit` (`Properties/Flock/Circuit/Rows.lean:149`), `unitLog ≤ 32`, `templateOk` (`SpotFlock/Code.lean:372-383`), `memberOk` (section 4), `outRowPorts = replicate outPorts.size 1` (`Bridge/BitsRowMap.lean:265`), `Layout.KeysRow` (`:197`) [read: the hypotheses; not checked at the scan's files beyond `memberOk`]. C takes the scope as a hypothesis, so none enters C. No move.
- **The real circuit files.** All three pass `Bits.check` at `w63` and `test`, flat [ran: `r20261112-232431-7c1d`; sent 3:48 PM PST]. No move.

## When the replay can run
My 6:21 PM PST, point 2: the full check as I proposed it cannot run on the line I quoted, and it waits on your pick of three ways.

## Not checked
- The body of the Python `one_stage.registered.reads`, and how the driver's `tree_port` turns a template's own input row into a leaf of the `state` tree.
- `Layout.KeysRow` and `templateOk` at the scan's files.
- The proofs of the registered read's lemmas.
- Which state row a slot's Program reads.

