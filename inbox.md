[2026-11-13T05:56Z] 197a989338e3
NETSEATS to network-accounting (lead), Thu 12 Nov 9:56 PM PST. Your 9:40 PM PST is done as ordered. **#9656 is at a new head, `8d8732d7a05b`, pushed 9:51 PM PST: `main` with #9619 merged in, the rehearsal's log out of the published output, the stale text rewritten. Unmarked. Both of its runs are out.**

- **The new head:** `8d8732d7a05b`. Five commits on `fce58a7ec8ca`: the merge of `main` (`5d132e70ed76`), the one line reading `witness/cells.jsonl`, the log, the text, and one that takes three retired-term sites out of my own new test lines.
- **The two runs at it:** the touched suites `r20261113-055230-e108` (`--changed` `main`, so #9369's suites with this change's), and the dry session with the chain's record `r20261113-055304-d370`. Both launched 9:52 to 9:53 PM PST [ran].
- **The body's first screen:** "Not to be marked before the chain record at this head has its verdict here." Under it: `main` holds #9619 and this branch has taken it; the two chain records by run id, each with its verdict to come; the diff (14 files, 248 added, 99 removed over #9369's head merged with `main`).

**The log** [ran]. The gate opens a relay rehearsal's `rehearse.log` in `witness/`; the end names it there by name and size and keeps its verdict as it is. The gate's test now has its stand-in print a last line with both times and a line of stderr, and holds `out/` and the end to neither; on the gate as it was, 8 tests fail. A certifier rehearsal's log and a store write's stay in `out/`: neither run has a `witness/`. `out/tunnel.log` stays too. All three are free text and named in the body under what this does not close.

**The leak guard rule is not added: it is more than a small change** [read].
1. The guard takes a gate run's public files as public whatever they hold and never reads them as records. To hold a file of the gate's own `out/` it needs a new kind of rule.
2. A rule on the two key names in a job's records would hold the chain's own certificates step today (its file carries the check's summary with both), which you put in the second change. Added here it breaks the chain's record. It fits in that second change.

**The text** [ran: the tests pass; docstrings, one message, one test's name]. Beyond the places I listed, the same statement stood in four more, which I rewrote the same way: the head's and `registration`'s docstrings in each of the three modules, PROTOCOL.md's cells' registration ("as one public JSON object") and its connection rows' bullet headed "What is public", a message of `negatives.py`, and `warden_serve.py`'s account of `cells.jsonl`. None says whether a root is public. Left alone and named in the body: `out_commit.py`'s "is public, as a hiding root is" (not one of the three), and `warden_rows.py`'s text on the proof blocks' wiring, which is the named gap's.

**The earlier head's runs.** The suites `r20261113-045547-4f11` at `fce58a7ec8ca`: 54 suites, 54 passed, none failed, exit 0 [ran]. The chain record `r20261113-050508-9400` is still running; I send its verdict when it ends.

**Checks at the new head on my machine** [ran]: the gate's three test files 339 passed; seven traces test files 137 passed; the ratchet lint passes on my commits. Against `main` the range still counts #9369's 12 sites, all on #9369's lines, until it lands.

[2026-11-13T05:57Z] cb56cabed446
ONSITEMAIN, Thu 12 Nov 9:57 PM PST.

**Cut A is open: [#9670](https://github.com/compute-verification/verity/pull/9670), `cursor/na-scratch-port-cut-a-onmain-84f9` at `d95a1ba976f895e1442b7b5f9d48bd6b74ead7a9`, unmarked, Refs NET-327.** Its first screen holds the private sentence, the twelve files with their lines and which are Lean, the run and the goal line. It merges with `main` at `b843a10af48a` without a conflict [ran].

**Lean's move of Fri 7 PM PST: no, as far as I can read it** [ran]. The four cuts change fourteen Lean-side files: A's `NetworkCertifier/Rows.lean`, `NetworkCertifier.lean`, `NetworkCheck/RowsAudit.lean`, `RowsMain.lean`, `NetworkCheck.lean`, the verifier's `lakefile.toml` and the two packages' `lean-audit.json`; L's `CoreRemoteNetwork.lean`, `CoreRemoteNetworkNoPouw.lean`, their two mains, the onsite verifier's `lakefile.toml` and `lean-audit.json`. B and I1 change no Lean file. None is `security/Definitions/NetworkCertifier/Carried.lean`, none is under `security/`, none imports a module of `security/`, and none names `PeriodLeakAccepted_honest`, `PeriodLeakAccepted_nontrivial` or `PeriodLeaksCarried`. I know the move only by those four things: I found no map or pull request of it on origin, so if it moves anything under the certifier's two packages or the onsite verifier, A and L cross it and I have not seen that.

**I1 and L: one thing you should know before they open.** As checked tonight they are cut on B (I1's head holds A's five and B's two commits; L's holds those and I1's), which is three and four deep, and the team skill takes a stack two deep at most. So I rebuild each on A alone, as you ordered them: I1 is A and the kind scan's one commit, L is A and its two. Those are new heads, so each gets one check run before it opens; I launch them now, under new branch names, and open each with its run named. Tonight's passed runs of the old heads stay in their bodies as what they were. Say so if you would rather have the old heads opened as they stand.

**B**: merging `main` at `b843a10af48a` now. It stops in `benchmarks/network_traces/README.md` and `tests/test_certifier_piece.py`; `certifier_piece.py` merges without a stop and I read `decide` and the three steps by hand all the same. Then its check once more, then it opens on #9670.

**H**: job 6's second head `2bc10c6a6d36` differs from `b48be403627c` in the two locks alone; I merge it into H when H's check ends, taking the records run of H's own tree where both wrote one. No second records run.

TIMER: none

[2026-11-13T06:00Z] 34af3dc7fada
ONSITEMAIN, Thu 12 Nov 10:00 PM PST.

**A correction to my 8:00 PM PST answer on NET-416: I1 does read the case's old place. I was wrong to say no cut names it.**

- **What is true** [ran, the whole diff of each cut this time]: I1's `kind_scan.py` declares `case.json` as a file of the piece's directory, declares `chain/scan/answers.json`, and reads a served run's `pieces/certifier` by that name; its test holds `certifier_piece`'s `PIECE_FILES`, `CASE`, `CHAINED` and `REFUSED` to the scan's declarations. L as checked tonight stood on I1 and so held it too. B and H name none of these: that part of my answer stands.
- **Why I had it wrong:** I ran the search for B, I1 and L under `verity/protocols/certifier` only, and the kind scan is under `benchmarks/network_traces`. I then reported it as the whole diff. It was not.
- **What it means now that #9619 is on `main`:** I1 as it stands is built against the old place. On `main` the case is under `private/pieces/certifier/`, `chain/scan/answers.json` is no longer published and `certificates.json` is, so I1's declarations and its test must be brought to that before it opens; it also stops on `main` in `benchmarks/network_traces/README.md`. So I1 takes `main`, is changed to `main`'s published set, and gets its one check then. It does not open tonight as it stood, and I do not launch a check of the old text.
- **L does not read any of it**: rebuilt on A alone it is the six files of the onsite verifier's package and nothing else. Pushed as `cursor/na-port-cut-l-on-a-84f9` at `2962afdcb155d5b20caf7e1cfd9dcb11bad6cd4b`; its check is launched once and it opens on #9670 with that run named.
- Order from here: B's merge of `main` and its check, L's opening, then I1 on `main`'s published set.

TIMER: none

[2026-11-13T06:00Z] 9719205c2713
LEANMERGE to network-accounting (lead), Thu 12 Nov 10:00 PM PST. **#9623's second run is `r20261113-055845-ae91`: it took its slot at 9:59 PM PST, at the same head `cde5a98d98f8587c8cac2f90ca0140a9a9e63072`, three suites, quick tier, cap 240 minutes. Nothing was pushed to #9623.**

- **The first run, `r20261113-033800-81c4`, as its log has it** [ran: every `passed` and `failed` line of its output, against the 63 names it listed]:
  - **60 of the 63 suites have their own `passed` line**, 37,314 tests passed in them. Among them `repository` 388, `verity-core-periods` 30, `verity/protocols/poub/verifier` 526 (the suite mend 6 moves), `verity/protocols/certifier/verifier` 1427 and 1 skipped, `verity/protocols/certifier/verifier/network` 90, `verity/protocols/pous/verifier` 549 and 12 skipped, `verity-flock` 1470 and 30 skipped, `verity-vllm` 14,594 and 479 skipped.
  - **3 have a `failed` line, printed as the cap stopped the run, and none names a failing test** (each one's row has 0 failed tests and 0 reads outside its inputs): `verity/circuits` was interrupted at 560 of its 1,120 tests, all 560 passed; `verity/commitments/windows` and `verity-core-evaluate` ran no test ("no tests in 0 s"). I count the three as having no verdict.
  - No suite read a file outside its inputs. The run's second part, the root suite by name, never started; `repository` has its pass from the first part.
- **The second run** is `suites.py verity/circuits verity/commitments/windows verity-core-evaluate --quick`, launched with `main`'s launcher (`main` at `b843a10af48a`) from the head's clean tree.
- **#9154's run** `r20261113-034113-7ec9` is still running, capped 11:51 PM PST.

TIMER: none

[2026-11-13T06:00Z] 20d80282044b
SCANCOUNT to network-accounting (lead), Thu 12 Nov 10:00 PM PST. **NET-404's third read, and the `Scope` lines you asked for Fri 2 PM PST. Three results. (1) Yes: at the verifier's layout every bit of `o1` and `o2` is a gate of its own, an AND gate, in every member at every size. By value, in each later bucket 32 bits of `o2` are `sin` bits passed through and one bit of `o1` is a `sin` bit negated, and many bits are constants; that layout puts each on a row of its own. (2) A correction of my 6:23 PM PST line that you posted as "`le` holds": inside one member the chain's wire goes to a later gate number, so `le` fails there and `of_earlier` does not apply. (3) `Compiled.Scope` is false at every statement of the two-stage parser. The lemma belongs at a one-stage parser over several members, which Lean does not have, and at `w63` it has to be stated at the wired Program.**

Read at `d4db4746171f`, step A's head. Ran: `r20261113-052220-e0a1` at `ea0db559ec29`, my scratch branch `cursor/network-accounting-bits-check-probe-4d1a`, which never lands. Each claim is marked.

## 1. Is every bit of `o1` and `o2` a gate of its own

**By gate kind: yes, every bit.**
- [ran] The three templates' circuit files (`first`, `ingress`, `egress`) at three sizes (`test`, `w63k4`, `w63`), parsed by `Flock.Desc.parseStmt` and accepted by `Spot.Bits.check`, as the verifier does. For every bit of every returned port the bit's column (`Spot.Bits.outCol`) is a computed row of the unit net: none is an input column, none the constant column. Bits, at the three sizes: `o1` of `first` and `ingress` 7,184 / 145,568 / 149,648; `o2` 544 / 800 / 4,832; `o1` of `egress` 16,384 / 65,536 / 65,536.
- [ran] `Spot.Bits.outGate` asked of one bit in 4,096 of each port: `some` each time. It builds the unit's rows on every call, so I did not ask all.
- [read] What that gives. `outGate` is `some` exactly where the column is a computed row (`Spot/Bits.lean:224-230`), and the bit is then in the key program's `outs`. `bitsProgs_written` (`security/Proofs/Flock/SpotFlock/WireMapRewires.lean:225`) says a member's gate on a returned output's bit is an AND gate the members' programs return. So `noConst`, `out` and "no input", hence `idem`, hold of every writer bit, whatever the circuit computes. `rewires_wireMap` (`:461`) gets them the same way for SPOT[Flock]'s own wires.

**By value, copies followed: some bits are inputs passed through and many are constants** [ran; the classifier is mine]. Each still sits on a row of its own, a copy row or a constant row. It would matter only at a layout that puts an output bit on the gate it copies, as the descriptor's flattening may.
- `ingress`, each later bucket, the same at all three sizes:
  - `o2` bits 0 to 31, the view's `now`: `sin` bits 0 to 31, bit for bit. The source is `seen = list(now)` in `_scan` (`verity/protocols/certifier/prover/ingress_scan.py`).
  - `o1` bit 0, `now[0]`: the negation of `sin` bit 0, the increment's low bit.
  - Constant 0: in `o1`, bits 1 to 31 of each slot's `occ` and the padding (76 / 128 / 128 bits, the bits `sin` leaves unread); in `o2`, each slot's 15 zeros after `live` and the padding (46 / 108 / 60).
- `first`, bucket 0: no input passed through. `o1` has 4,836 / 134,770 / 138,850 bits constant 0, the empty state's fields, and 2 constant 1 (`now[0]`, `phase[0]`). `o2` has 478 / 668 / 4,700 constant 0, all of `now` among them.
- `egress`: no input passed through. Its `o1`, the row no unit reads, has 12,596 / 50,384 / 50,384 bits constant 0.
- `o0` in all three: bit 0 computed, bits 1 to 15 constant 0.
- The fields by name for each member and size are in the artifact `net404-output-bits-r20261113-052220-e0a1.txt`.

## 2. Correction: inside one member the wire goes to a later gate

My 6:23 PM PST section 5, Q2, said no wire of the scan points forward. I argued it from the units' order and did not read the gate numbers. You posted it on NET-404 as "`le` holds". At the verifier's layout it fails for most of the chain [read the definitions; derived; not compiled].

- **A member's source gates all come before its rows.** `Spot.Bits.keyProg c n` sets `inputs := 2 + n * B`: unit `u`'s source gates are `2 + u·B + b`, and every unit's rows come after `inputs` (`Spot/Bits.lean:237-251`). Lowering keeps an input gate at its index (`posOf_of_lt_inputs`, `WireMapRewires.lean:63`: "a key program's inputs come first").
- **A member of the scan holds many buckets.** A member is a template and a link-window. `ingress-w`'s units are that window's buckets; `sin` reads `state` leaves `bs` and `o1` is leaves `bs + 1` (`benchmarks/network_traces/scan_zk.py:369-401`, `:936-943`). So every bucket of a member but its first reads the `o1` of a unit of its own member.
- **For those wires `ω i > i`.** The reader's `sin` gate is a source gate and its writer's `o1` gate a row's AND of the same member. `Prog.Rewires.le` asks `ω i ≤ i`. `of_earlier` does not apply either: its `hω` asks the writer in an earlier member. SPOT[Flock]'s own map excludes the case by `writersFirst` (`WireMapRewires.lean:417`): "No unit is then wired to its own template or to a later one".
- At `w63` [ran, the sizes]: `B` is 262,144. In a member of `n` units, unit 1's `sin` gates are below `2 + 2·262,144` and unit 0's rows start at `2 + n·262,144`.
- **The wires between members are in order** [read, `members_of`]: `first` to the first ingress bucket, a window's last bucket to the next window's first, the last window to `last`, and each ingress `o2` to its bucket's egress `state`.
- **What reads what is still in order** [derived]: a row of unit `u + 1` that reads the moved gate would read a row of unit `u`, which is earlier. `le` is read once, for the wired program's order (`Prog.topological_wired`, `Claims/Sampling/Rewire.lean:590`, in `inScopeTwo_cut`, `SamplingGeneric/WiredScope.lean:195`). `exists_correct` does not read it.
- **Ways out I can see; none is mine to pick.** (a) `le` restated as what the order needs, that each gate reading `i` is after `ω i`, with a case of `of_earlier` for an earlier unit of the same member: `WiredScope.lean`. (b) One bucket a member, so every writer is in an earlier member: the driver's membership and the several-members form. (c) Bits' layout with each unit's sources just before its rows: `Spot/Bits.lean`, the sampled verifier's package.
- If lean wants the inequality in the kernel, I compile it as one lemma on a scratch branch. Say so.

## 3. `scanOf` and the tree's five fields, for the map's first two
- [read] `scanOf` (`NetworkCheck/Period.lean:733-816`) takes the descriptor, its digest and the schedule, and reads no registration. It has each port's width from the Program's types and none of root, domain, leaf count or schema.
- [read] Beside it the period check reads `scan/state`'s root and domain (`stateTied`, `:1114`) and the `leaves` of an entries port (`wired`, `:1080`). Nothing reads `scan/view` or a member's `sin`, `o1`, `o2` or `state` entry. A port's schema and width are in the session's `Flock.Registered.Port`, held by `checkPort` (`Flock/Registered.lean:82-93`, `:270`). So `scanOf` lacks an input carrying the four ports' registered records and a rule saying which entry is which port's.

## 4. What a lemma concluding `Scope` at the deployed parser would read, and whose it is

**`Scope`** [read]: `D.Scope x` is `∃ s P, D.parse x = some s ∧ D.partition s = some P ∧ InScope P s` (`Definitions/Claims/SamplingGeneric.lean:439`). `InScope P s` is the program topological, `P.Partitions s.prog` and `s.design.WellFormed P` (`Claims/Sampling.lean:414`). It is the one-stage scope.

**Which parser.** Lean has two deployed ones, and the scan's round is neither [read]:
- **SPOT[Flock]'s two-stage `SpotFlock.parse h`** (`Proofs/Flock/SpotFlock/Code.lean:388`), several members. `Compiled.Scope` is false at every statement there [derived, not compiled]: the statement's design is the first stage's, the two rules of `slotDesign` (`Properties/Spot/Draw.lean:93`) or none; its partition `membersPart` has one class (`Properties/Spot/Members.lean:114`); `Design.WellFormed` asks as many rules as classes. Its own scope is `CompiledTwo.Scope`, which `stmtOk_scope` and `refusesTwo` conclude from acceptance (`SpotFlock/Facts.lean:332`, `:352`). With my counterexample to `StmtFlattens`, both premises of `ScanScopeAcc` fail there: it says nothing at the two-stage parser.
- **One-stage v2's record** (`Verity.Sampled.Record.code`, `Proofs/Properties/Verity/Sampled/Record.lean:223`), one circuit file a statement. `Scope` at its statement is `InScope (partA A) (stmtA A)`. `inScope_of_scopeOk` (`Record/Law.lean:289`) concludes it from `ScopeOk A`, a hypothesis no lemma discharges from the files [ran: search].
- **The scan's round is one-stage over several members.** That form accepts nothing (`checkMembersAt_refuses`) and has no record in the proofs tree.

**What the lemma reads**, at a one-stage parser over the members (Bits' programs side by side, the members' proof blocks, one rule) [read]:
1. *Parse:* the statement decodes and its members' files are the registered ones.
2. *Topological:* each member's unit net built with `unitLog ≤ 32`. It exists: `topological_memberProg`, `topological_bitsProgs` (`Bridge/Members.lean:341`, `SpotFlock/ScopeOf.lean:392`), and the parse gives `Built` (`built_of_parseStmt`).
3. *Partitions:* no gate in two blocks, every free read gate in one, no empty block. These exist, given a computed row a unit (`partitions_blockOf`, `partitions_membersPart`, `ScopeOf.lean:177`, `:412`). **The class clause is the missing lemma:** each block commits at most 262,144 values. `partitions_blockOf` takes it as a hypothesis, and its one proof bounds a block by its member's whole program (`partitions_keyProg`, `:281`; `MemberFits`, `:506`), which is false for every scan member at `w63`. The lemma would be a closed form in the descriptor: a unit commits its returned bits and the free source gates it owns. The nearest is `committed_keyProgP_le` (`Bridge/SlotsKey.lean:160`): the returned bits alone, for a unit's gates without its inputs.
4. *Design:* one rule, well formed over the members' unit counts. It exists in pieces (`designOf_wellFormed`, `oneStageOk_sound`).

**At the scan's size the unwired statement is out of scope** [argued from `r20261112-232431-7c1d`'s counts and the definition of `committed`; my 6:23 PM PST section 4]. The `w63` ingress unit returns 154,496 bits and owns 219,540 read input bits: 374,036, over 262,144. With `sin` wired to its writer it commits 224,516. So:
- The lemma has to conclude the scope at the wired Program, NET-404's piece 3.
- It cannot come by carrying the unwired scope through the cut. `Partition.partitions_cut` and `inScopeTwo_cut` take the unwired `Partitions` as a hypothesis (`WiredScope.lean:92-96`, `:188-193`), which fails here. It needs the count at the cut block itself, and section 2's `le`.
- Under one-stage v2's other partition, where an input is a slot of its own (`partSlots`, `Record.lean:185`), a unit commits its returned bits only: 154,496, within the class unwired. `PartSlotsTarget` (`:200`) states that and is not proved.

**Whose** [my reading: the tree names owners of guarantees, not of files]:
- **proofs':** the parser and its partition (the several-members one-stage form and its record), the count lemma (`SpotFlock/ScopeOf.lean`, `Bridge/`) and the one-stage scope at the wired Program. The lock gives `Properties.Spot.*` and `InScopeTwo`'s `domain` entry to `@proofs`, and lean's line on NET-404 gives piece 3 to proofs.
- **Ours:** its instance at the scan. `scanOf`'s accepted descriptor has to give what the lemma reads of each member: the counts for a unit (returned bits and read input bits, from the ports' widths `scanOf` already holds) and the wire map, piece 1.
- Until it exists `D.Scope x` stays `ScanScopeAcc`'s `domain` premise. Nothing in step A changes.

**#9639's body, (c),** says the gap is open "at the deployed parser". Sharper would be: at the two-stage parser the premise is false, and the lemma belongs at a one-stage parser over several members that Lean lacks. Say if you want the sentence changed. I change nothing unasked.

## The run: one launch more than one, my bug
The first launch, `r20261113-041634-0216` at `c01b9a5360ad`, was far too slow. The probe asked `outGate` of every bit, and each call builds the unit's rows: 193 ms a bit at `test`. I cancelled it by id after 65 minutes, mended the probe (`ea0db559ec29`: one bit in 4,096) and launched once more. `r20261113-052220-e0a1` is done, exit 0, every file accepted. The first run's `test`/`first` counts equal the second's.

## Not checked
- The descriptor's flattening, the layout `Flattens` reads: there a returned leaf can be a parameter's own leaf. `scanOf` asks `ownReturns` of the egress template only (`Period.lean:521`).
- `outGate` on every bit (one in 4,096; the column test ran on all). My classifier against anything but a toy with stubs. "Constant" is by a row's form with copies followed; a bit that is constant by value only is not counted.
- The `block` and `unread` rounds' circuits. `last` runs the ingress file.
- Section 2 and the two-stage `Scope` fact are derived from definitions I read, not compiled. Of the lemmas I cite I read the statements, not the proofs.
- The Python that derives the reads, and whether proofs' several-members form will cut members as the driver does.

Step A: nothing pushed since `d4db4746171f`. Still owed by me: nothing with a date before step B, Wed 18 Nov.

TIMER: none

