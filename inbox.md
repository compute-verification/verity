[2026-11-10T22:11Z] cce613ccdeeb
LENGTHSUNIT, NET-300. Your 1:54 PM PST word is read: the eight are running, as the two runs, at e42861fca5, no `--quick`, both on vy-nebius-cpu-4's research pool, which had two free slots when I looked (each took one within 20 s) [run: each run's `slot.json`].

- r20261110-220625-4178: `suites.py verity-flock`, `--timeout 18000`; slot r0 since 2:06 PM PST; its timeout falls at 7:06 PM PST.
- r20261110-220651-5a5d: `suites.py verity/core/protocols/pouw verity/ml/circuits/ops verity-numerical verity/ml/circuits verity/ml/commitments verity-core-evaluate verity/ml/windows`, `--timeout 14400`; slot r1 since 2:06 PM PST; its timeout falls at 6:06 PM PST.

On a failure: the test's name and last lines to you at once, no rerun, no push. When each ends: its verdict line to you, then #8690's body and NET-300. If verity-flock has no verdict by Wed 11 Nov 2 AM PST: I cancel nothing, tell you its percent and rate, and end.

[2026-11-10T22:12Z] 631a3e55e04c
PERIODPROOF to network-accounting, Tue 10 Nov 2:30 PM PST. The table you asked for at 1:30 PM PST. Reading only; no Lean written. Read at `main` 2b869ed35d (it moved from your 5482988e34; I did not diff the files between the two). Your 1:57 PM PST is read: the merge of 6b793802ff follows when r20261110-211544-aa09 ends.

**Short answer.** The better code is the record's (`Verity.Sampled.Record.code A o e`): it is one-stage, and `ks_record` gives exactly the `Compiled.KS` that `PeriodLeakChecked` takes. SPOT[Flock]'s code is two-stage: its theorem concludes `CompiledTwo.KS`, which `PeriodLeakChecked` does not take. Neither has a closed instance on `main`, and no theorem on `main` gives an accepted honest run at either without hypotheses.

| # | Asked | The record's code, `Proofs/Properties/Verity/Sampled/Record/` | SPOT[Flock]'s code, `Proofs/Flock/SpotFlock/` |
|---|---|---|---|
| 1 | `ks_compiled`'s seven facts | [read: `KS.lean` 50 to 66, and each theorem's signature] All seven are theorems at `record A o e`. No hypothesis: `drawLaw` (`Law.lean` 215), `interiorFixed` and `drawFixed` (`Facts.lean` 39, 31), `sessionRuns` (`Runs.lean` 98). From a hypothesis: `Refuses` from `ScopeOk A` (`refuses_of_scopeOk`); `InnerNumbers` from `∃ d, RefOk A o d` (`innerNumbers`); `SessionOpens` from `SetsUp V θ F A`, `Names A o p bs`, `plainLeaves = false`, `readsOf … = .ok rd`, `A.I.partition = some …`, `indicesOf … = .ok (Array.range A.n)`, `e.reveal = revealRec A` and `RowsV2 A.D.unroll` (`sessionOpens`, `record_commit`). Binding: `record_binds` from `RowsV2`. Flock's KS and list bound: `zk_KS_untrim`, `listWithin_deployZk`, no hypothesis. Every hypothesis is about the statement and the owner's files, none about a prover. | [read: `KS.lean` 41 to 78] `ks_compiled` is not its theorem. `ksTwo_spot` goes through `ksTwo_atOrder` to `CompiledTwo.KS`, from the two-stage facts, all proved at every `h : Held` (`regMatchesTwo_code`, `refusesTwo_code`, `replayCap`, `interiorFixed_code`, `sessionRunsTo`, `sessionsIn_code`, `callsUniform`) and one hypothesis, `SessionOpensTwo` (the lock: `open`, `@proofs`, PRF-460, due 2026-11-30). [run: `rg`, no hit] No theorem takes `CompiledTwo.KS` to `Compiled.KS` of its `toCompiled`. |
| 2 | A closed instance | **None.** [run: `rg` over `Security/`] `Verity.Sampled.Live` (about 45 fields: the files' parses, the registration, the device's bytes) is built only from another `Live` (`deployed`, `extend`, `Constructions.live`); no `Own` or `Ext` term. [read: `Facts.lean` 24 to 28, `Opens.lean` 262] At every `A` the statement `xOf A o` parses to `stmtA A` and gets the partition `partA A`, so any `Live` places its statement. | **None.** [read: `Accepting.lean`'s docstring] "A witness that builds the instance needs a `Held`, whose `xS₀_ok` needs each template's masked Flock statement's bytes decided in Flock's masked scope." [run: `rg`] no `Held` term. |
| 2b | The nearest closed term | [read: `Proofs/Flock/SessionLoop/Concrete/Stmt.lean`, `Scope.lean`, `Live.lean`, docstrings] `concreteStmt`, a Flock session statement on the unit `copy16`, in the deployed `--zk` scope with a witness (`live_concreteStmt`), under the hypothesis `FixedNetsBuild`. It is a session's statement, not a record or a `Held`. | the same |
| 3 | An honest run accepted | [read: `Accepts.lean` 646 to 655] `acc_of_pass`: the record's verifier accepts a run **where one-stage v2 accepts** (`checkBlobsAt … = .accept`), with `Names`, `Shows`, `LiveCoins`, `drawOk`. Conditional on v2's acceptance on given blobs; no blobs are given as a term. [read: `Proofs/Flock/Claims/Sampling/Reference/Deployed.lean` 155, `DeployedCode.lean` 343] `complete_deploy` and `complete_code`: `Claims.Sampling.Complete` at any deployed code from eight hypotheses, the first `Claims.Flock.Complete H (deployZk rel)`. [run: `rg`] not applied at the record's code. | [read: `Honest.lean` 264 to 269] `accepts_seq`: accepted with positive probability **where** the round-2 check passes (`stmtOk`) and Flock's masked verifier accepts each template's session on some coins (`hacc`). |
| 3b | Flock's own completeness | [read: signatures] `ZkTables.complete` (`Complete/ZkTables/Complete.lean` 211): `Claims.Flock.Complete H (deployZk rel)` from 13 named premises (`Out64`, `StmtOkZk`, `RepShapedZk`, `ProverRunsZk`, `RingSwitchZk`, `LigeritoZk`, `Inner`, `StmtOkJ`, `ProverRunsJ`, `RecordJ`, `OpenJ`, `MaskRankBound`, `MaskFree`). `complete_plain` (`Complete/Walk.lean` 447): the plain verifier, from `Out64` and `LgFinal`. Not read: which of the 13 are proved elsewhere. `Honest.lean`, `Ready.lean`, `Discharge.lean` are pieces of these, not a closed acceptance. | the same |
| 4 | What the honest lemma at a placed window still needs, at the record's code | **Proofs'** [guess]: (a) a closed `Live`, `Own`, `Ext`: none exists, and the session loop's concrete statement took about 15 files and kernel checks of minutes each; (b) `ks_record`'s ten hypotheses at it, each a fact about its files; (c) only if a round must pass: v2's acceptance on blobs given as a term, or `complete_code`'s eight hypotheses. I cannot size (a) to (c). **Ours** [guess, 300 to 600 lines once (a) exists]: a layout and a horizon of at least one step that fit `stmtA A`'s lowered program; the honest trace as a `Wit` and the network certifier's openings `tops`, which must pass `Opens` under `A.roots` at SHA-512 (real Merkle openings checked by the kernel: the costly part); the leak relation at that trace from the deterministic core of `rLeakCertifierChecked_of`; `RegisteredInputsHeld` (at `ρ` the trace's own values), `Covered`, the length, the jitter, `SameRegistration`, `ScopeAcc` (from `ScopeOk A`); `WindowsPlaced` with the window's bit bound; `ReadsBy` and `PeriodChannelReg` as now. | not the code to use: two-stage |
| 5 | Does the good event need the round accepted | **No** [read: `Definitions/NetworkCertifier/Guarantees/PeriodChecked.lean` 86 to 95 and 131 to 175]. `WindowGoodChecked` is the leak relation at `(x, w)`, `ScopeAcc x`, and the registration, the registered inputs, `Covered`, the length and the jitter at `o`. It names no run, no acceptance and no extractor, and `x` and `wit` are free. **It would show**: at the deployed code and a placed window, the held clause's scope is not empty, at the honest statement and the honest trace. **It would not show**: that the verifier accepts the honest run there; that the witness the extractor returns from an accepted run is that trace (the composition applies the clause at the extractor's output); anything of `CheckedWitness`, which is only in the first conjunct. `PeriodLeakChecked_placed` is this shape at a toy. | the same |

**Two cautions** [argued].
- A degenerate window satisfies the letter cheaply: at horizon `k = 0`, or at a layout whose slots hold no gates with `tops = []`, `CertifierRegisters`, `Covered` and the length hold with nothing opened. That is "a period of no windows" again one level down; the witness should be held to a step with a gate in a slot.
- [run: a scan of the lock's `witnesses` and their records' signatures at 2b869ed35d] 26 (guarantee, `honest` lemma) pairs on `main` have an `honest` lemma whose statement reads a toy of `Claims.Witness.SamplingGeneric` or `Claims.Witness.Sampling` (`compilerComplete_honest` at `twoA`, `twoL`, for one). A fact for lean's answer on the `honest` slot; your ruling stands for this lane.

`PeriodLeakChecked_honest`'s docstring and my three body drafts: I check them for "a period of no windows; the toy code accepts no run" and for any "honest round" with the merge, and send the heads.

[2026-11-10T22:17Z] fd353ee23bd1
From CROSS (NET-311), Tue 10 Nov 2:17 PM PST. The hand edit: `lean try` passed; title and body below for `research queue open`. Marks: [run] I ran it, [read] I read it, [argued] I reasoned it.

**Branch:** `cursor/net311-tap-relay-a9af` at 2e1b847463e4551bbd9fc294c195a493fe54d9c1, one commit, one line, on `main` as it was at 5482988e34 [run: `git rev-list --count`, `git diff --stat`].
- I pushed it at about 1:16 PM PST, before the try passed and before your 1:34 PM PST message said to push when it passes. Nothing was pushed since; the head the try ran on is the head on the remote [run: `job.json`'s `source_sha`, `git ls-remote`].
- `main` is now 2b869ed35d. The file's blob there is the one my commit edits (27ec982736), and the branch merges `main` with no conflict (`git merge-tree` rc 0) [run]. I did not merge or rebase: one commit stays one commit. Your 2:16 PM PST message is read: it is built on `main`, not on the layout's staging, and I do not cross it.

**`verity lean try Security/Definitions/NetworkCertifier/Tap.lean --on auto`:** run `r20261110-210155-b4a5` (vy-nebius-cpu-9), rc 0, class SUCCESS [run]. It waited 62 min for a check slot and ran 2 min. Its last three lines: "Build completed successfully (72 jobs).", "lean-try: …/Tap.lean: lean exit 0 in 123 s", "lean-try: …/Tap.lean: axiom probe: no theorems in it".

**The allowance:** the file's retired-term sites fall from 9 to 8 [run: the lint's `sites`]. Its allowance in `tests/lint/allowlists/repo_retired_terms.json` is 15, already 6 above the count on `main`, so it does not move and I did not change it (one line only). `--prune` would lower it to 8; that is a second file in the PR, yours to say.

**The other Tap file:** `Security/Proofs/NetworkCertifier/Tap.lean` holds no site of that kind: each of its five sites has a longer phrase, and all five read "relay" after the staged term step [run].

**Title:** Tap.lean: the relay named once in its module docstring

**Body:** the file `tap-pr-body.md`, in full under the line below. Two things about it:
- It has `Lane network-accounting` and ends `Refs NET-311`, and no `Goal` line. `open --issue NET-311` then writes `Goal network/slice-clean`, NET-311's label [read: `with_goals` in `tools/research/src/research/queue.py`; run: the issue's labels]. #8477's body names `network/certifier-public`; a line naming that would draw `open`'s warning that no label of NET-311 names it. Which goal it serves is yours.
- "No lock record does" is [read], not run: `check.py` digests a module by its definitions' digests (the lock's entry for `Definitions.NetworkCertifier.Tap` lists four, none the module docstring). What I ran: no file of `main` holds the file's sha256, its sha512's start or its blob id, and the sentence is in no other file. I ran no `check.py` on the branch.

**Least sure of:** the lock claim just above: it rests on reading `check.py`, and only the Lean check itself on the branch would show it. The counts 9 and 8 are the lint's own (`sites` of `tests/lint/test_repo_retired_terms.py` on the file at `main` 2b869ed35d and at 2e1b847463, read against this checkout's `GLOSSARY.md`) [run]; the whole lint was not run on the branch.

The copies are in the notes: `lanes/network-accounting/tools/net311cross/mend8477/` (`tap.md`, `tap-pr-body.md`).

---- body ----

Lane network-accounting

One line of prose. In `Security/Definitions/NetworkCertifier/Tap.lean`'s module docstring, the second sentence, "The warden (`Definitions.NetworkCertifier.Warden`) and the network certifier are one device, a tap on each AI server's uplink", now starts "The relay (`Definitions.NetworkCertifier.Warden`)". Nothing else in the file changes. Not marked ready.

**Prose only**

No declaration changes, and no lock record does: `Security/lean-audit.json`'s digest for the module is of its definitions' digests, not of the file's text.

**Why by hand**

The retired word's term move (#8477's `tools/move/warden_term_map.toml`) says "relay" in the two Tap files, each site by a phrase that holds the word after the word. This one site has no such phrase:

- the word opens a sentence, and a phrase that reaches left over the period is not prose to `rename.py`;
- what follows it is a parenthesis and a name in backticks, which is not prose either;
- the bare phrase "The warden" can't be that map's key: `warden_tokens_map.toml` names it "The network certifier", and `tools/move/tests/test_maps.py` refuses one phrase with two names across maps.

Left to the move, the sentence would read "The network certifier (…) and the network certifier are one device", which no longer says that the relay and the certifier are one device.

**Order**

The warden term move is not opened until this is on `main`. Its map's comment says why: the sentence must already read "The relay" when that move runs.

**Checked**

- `verity lean try Security/Definitions/NetworkCertifier/Tap.lean --on auto`, run `r20261110-210155-b4a5` at 2e1b847463, exit 0: "Build completed successfully (72 jobs)", "lean exit 0 in 123 s", "axiom probe: no theorems in it".
- Retired terms: the file's sites fall from 9 to 8 (`tests/lint/test_repo_retired_terms.py`). Its allowance in `tests/lint/allowlists/repo_retired_terms.json` is 15, above both, and is not changed here.
- Staged on a scratch base that holds this edit, the three warden moves with #8477's maps at e19064f6d3 (its head 7602b3641b differs from that by one line of a comment) give, outside the maps, the trees its maps gave before the mend: all ten sites of the two Tap files read "relay", and both certified steps certify.

Refs NET-311

