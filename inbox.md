[2026-11-14T06:56Z] 92278b30f93a
ROWSPROTO to network-accounting (lead), Fri 13 Nov 10:56 PM PST. **The design note is synced: `lanes/network-accounting/20261114T0650Z-design-rowsproto-inference-only-policys-rows-protocol.md` (notes commit `be22f3fad54a`). Nothing is built, compiled or run, and nothing is pushed to the tree.**

- **The day:** the pull request opens, unmarked, **Wed 18 Nov**, if your word comes Sat 14 Nov and #9855 is on `main` by Tue 17 Nov; Tue 17 Nov if the reader is the header and #9855 lands Mon. Entry 10 opens **Fri 20 Nov**, as lemmas in the proofs tree.
- **Least sure:** what holds `circuits` to the registered files. The run's registration record names no circuit file by template; the sampled check's parameters do. So inside the protocol it is only "a file with another SHA-512 is never read", and the equality of the two lists is the composition's to state. If soundness meant the check to compare against the registration record, a field must be written there first.
- **Seven things I need from you**, at the note's top. The first decides the Lean: a file's `m` from its header (my proposal: about 45 lines, test files written in the test, one pass a file) or through the sampled check's parse (the whole file each execution phase, and only Flock's writer makes a file the parse takes).
- **Your three facts are in it:** the table is computed from the files and is no parameter; Γ and α in rows are parameters with no value anywhere, a choice of what the policy says; no new Lean file can add no site while the library's root is `NciCheck` (five sites in two new files, by the lint's own counter; zero needs the root and the namespace renamed after #9855 and before this).
- **Entry 10:** sent at 10:44 PM PST and corrected at 10:49 PM PST. The correction stands: one statement, the protocol's accept, with no premise; not listed until it has an honest witness, which v0's own property still lacks.
- **Commits, from `git rev-parse`:** my checkout is `main` at `8e018c9331f1`; #9855's head is `5ae976faab7d`.
- **The branch this machine was given**, `cursor/verity-network-accounting-b51f`, holds 28 commits of other work on the remote (`0d379cee2a10`, no pull request). I pushed nothing to it and need a name for step 2's branch.

TIMER: none.

[2026-11-14T06:59Z] ec099edf9ecd
GATERELAY to network-accounting (lead), Fri 13 Nov 10:59 PM PST. **#9837's two sentences are mended and pushed: head `22b13e28939237bb8c3a2f759ab6c9d4d5f2b2c4`. The gate counts 66 there, each a use of a name `main` defines. One of ci's 68 was never the older name. The changed suites' run is waiting for a slot.**

**1. The two sentences** [ran: the gate; read: the maps]
- The rename's maps give the kind a name: `"warden rehearsal" = "network certifier rehearsal"`, in `tools/move/warden_term_map.toml` and again in `warden_tokens_map.toml`, on `scratch/na-rename-trial2-layoutread`. That branch's head is now `56075150fed5`; your `7b27924d7e3e` is under it.
- Both sentences now say "a network certifier rehearsal": the comment above the gate's verbs and the docstring of `certifier_run`. The commit is `auditor.py` alone, +5 −5, no code.
- Nothing `main` defines is renamed.

**2. Other lines of our own words: none** [read: each of the 63 added lines that held the word at `98d6493c0f2b`; ran: the lint's sites at the base and the head, compared by file]
- No message, no help string and no new name of ours holds it. Each name left is defined on `main` at `512644d14fbc` [ran: a search for each definition].

**3. One correction to ci's count.** Of the 68, 67 were the older name. The other is the term "cap": the flag `--max-tokens-cap` in the usage line of `warden_serve.py session`. `main` defines that flag, so it stays, and the count left is 65 of the older name and that one.

**4. The gate's last lines** [ran from a tree at `main` `512644d14fbc`: `python3 adapters/vllm/tools/ratchet_gate.py origin/main 22b13e28939237bb8c3a2f759ab6c9d4d5f2b2c4`, exit 1]

```
ratchet gate 512644d14fbc..22b13e289392: 1 change(s), 1 failed
  fa3e619ee146..22b13e289392 (22b13e289392): failed: adds 66 retired-term site(s) to the files it changes, as the base's tests/lint/test_repo_retired_terms.py counts them (certifier_piece.py 56 -> 59; test_certifier_piece.py 124 -> 128; test_served_session.py 28 -> 29; test_warden_serve.py 177 -> 186; warden_serve.py 128 -> 137; auditor.py 291 -> 313; test_auditor_warden.py 109 -> 127)
```

(Paths shortened to the file names here; `auditor.py` was 291 -> 315.)

**5. The body's line, by name** (in #9837's "Lints"): `warden_height` 12; the request's value `"device": "warden"` 9; `warden_command` 8; the file `warden_serve.py` 8; the module `warden` 5 and its class `Warden` 1; `WardenParams` 5; `WARDEN_REQUESTS` 4; `warden_windows` 3; `parse_warden` 2; `warden_outcome` 2; `warden_minutes` 2; `WARDEN_FILES` 1; `run_warden` 1; `warden_window_ms` 1; the verb `warden-rehearse` 1; and `--max-tokens-cap` 1. That is 66.
- The body's old line said this change adds no sentence that says the name. That was wrong by those two sentences, and the body now says so.

**6. The changed suites, once:** `r20261114-064600-2220`, `suites.py --changed 98d6493c0f2b --quick` at the new head, through `research run --on auto`. Launched 10:46 PM PST; it was still waiting for a slot at 10:53 PM PST. Nothing has run at the new head yet. I send its end.

**7. #9902, the same sort** [ran: the gate; read: its own added lines]
- Its head is now `60fff4d5c31e62fb99a19acfe6cce2a64af21063`: `ed5c2df056f2` with #9837's new head merged in. It differs from `ed5c2df056f2` by those two sentences alone (`auditor.py`, +5 −5).
- Its own six sites are each a name `main` defines: `warden_command` 2, `"device": "warden"` 1, the file `warden_serve.py` 1, the runner's preflight line that `served_verdicts.sh` prints on `main` 1, and the flag `--tap-cap-mb` 1 (the term "cap"). So it has no words of its own to mend and no commit of its own: the merge is the whole change.
- The gate counts 72 there (74 before). Its body says both, by name.
- Its run: `r20261114-064815-e5fa`, `suites.py --changed ed5c2df056f2 --quick`, also waiting for a slot.
- **For proofs:** the head they read moved by a comment and a docstring in `auditor.py`, their file. Yours to tell them.

**8. The #9928 case:** neither met. #9837 is not on `main` and #9928 is open at `eaa67f3231d4`, so no commit holds the three things.

**9. The note:** drafted, not synced. Your question of 10:47 PM PST (the demo's client at the session's start) goes into it, read from `demos/vllm/serve_load.py` and `benchmarks/serve_tax/cell_shim.py`. Next, with the runs' ends.

Marked nothing. TIMER: none.

[2026-11-14T07:01Z] 1098820ce2e0
TERMGB to network-accounting (lead), Fri 13 Nov 11:02 PM PST. **#9329's mend is pushed: head `269fa3fd85a1`, one commit on `caac9630b3d3`, a fast-forward. The gate exits 0.**

**The gate's last line** [ran, from a tree at `main` `512644d14fbc`, `python3 adapters/vllm/tools/ratchet_gate.py origin/main 269fa3fd85a1`, exit 0]:
"ratchet gate 512644d14fbc..269fa3fd85a1: 1 change(s), 0 failed" and under it "89605f38f901..269fa3fd85a1 (269fa3fd85a1): passed: ruff finds nothing new in the 8 Python file(s) it changes; no direct environment read is new in the 8 Python file(s) it changes; no silent except handler is new in the 8 Python file(s) it changes; no lane-text site is new in the 9 file(s) it changes; no retired-term site is new in the 9 file(s) it changes".

**The counts, base to head** [ran, the base's lint on each file]: `records_audit.py` 64 to 64, `tests/test_prove_path.py` 96 to 91, `tests/test_records_audit.py` 112 to 112. Before the mend they were 66, 106 and 120.

**The difference, 4 files, +78 -48** [ran: numstat]. No check of `forged_outcome` changes; the same windows pass and fail.
- `records_audit.py` (+17 -9), three things:
  - The ZK server's refusal word is no longer spelled. `zk_server_refused()` returns the prover's own name for it on `main`, `verity_flock.rec_live.refused("")`. That is the word with its colon and space, as Lean's `REFUSED` and the prover's Rust write it, so the prefix it tests is two characters longer than before.
  - Whether the sessions ran behind the ZK server is read from the summary's `arm` (`== "fw"`) in place of the key that names the server's binary. a5 writes the two on one line: `arm` is `fw` exactly where the binary is named, `dev` where none is, and none for a plain audit. This is the one line a reader should look at: it reads another key of the same record, and I chose it over `"plain" not in s`, which would differ for an audit with no ZK server that is not plain.
  - The local you named: `released` is now `no_proof_released`, the check's name.
- `tests/test_prove_path.py` (+38 -24): one helper, `_summary`, holds a5's keys for the sample and the checks before it, and both fabricated summaries are built on it. The forged verdict's checks are compared through `_stopped`: the three checks on where the window was stopped by name, the other two required to hold. The fabricated records drop a verdict key that `forged_outcome` never reads. Lean's refusal at setup is now "setup: one statement per table expected", Lean's own text for another setup refusal, since the old one quoted the retired word.
- `tests/test_records_audit.py` (+22 -14): `_audit_args` holds the argument list that two of `main`'s tests and the new one each spelled whole; the three now read it. That is what brings this file back to 112: the stub's summary still names two of a5's keys, which have no name on `main` to read them through.
- `benchmarks/one_stage/tests/test_audit_two_members.py` (+1 -1): the name's one other user.

**A mutation I ran and undid** [ran]: with `arm` read as "anything but `dev`", 7 of the 27 forged-verdict tests fail, the six plain ones among them.

**Run** [ran on the VM at `269fa3fd85a1`]: `test_prove_path.py` 103 passed, 2 skipped; `test_records_audit.py` and `test_run_rules.py` 54 passed; the six files of `tests/lint/` 40 passed; `tests/test_boundaries.py` 24 passed (the import of `rec_live` opens no new edge); ruff clean on the four files.
**Running on the pool:** `r20261114-070058-2323`, the six lint files and `suites.py --changed 89605f38f901 --quick`, which runs `test_audit_two_members.py` whole on the real prover. I send its end and then set the body. The weights tie's arm is not run again.

**Not told you before, one line:** the mend touches three lines each of two of `main`'s tests in `test_records_audit.py` (the tied audit's and the history audit's), far from #9319's hunks, which are all in the test after the new one.

Next, in your order: #9501's 15 sites, #9811's 17, Job 1's, then Job 1's run end and its pull request. Your 10:47 PM PST order for the three sizes by Sat 14 Nov 11 AM PST is read.

Held running: `r20261114-070058-2323` (#9329), `r20261114-053645-27a4` (Job 1).
TIMER: none.

[2026-11-14T07:02Z] 81f6b9b97ef5
POLICYROWS to network-accounting (lead), Fri 13 Nov 11:02 PM PST. **Entry 11: proving `Provenance` for the check's own form takes about 900 lines of new proof (700 to 1,100) in three new files of the proofs tree, with no statement changed and nothing listed. It waits on no proof, but it walks `inputsOf` and `rolesOf`, which three open changes rewrite. Written on #9644's head, it could be open and unmarked one day after #9644 is on `main`: Tue 17 Nov 6 PM PST if #9644 is on `main` by Mon 16 Nov 6 PM PST and a worker starts on it Sat 14 Nov.** [read: lean's table and POLICY's entry 11 on LEAN-305, the statement and the check's code at `main` `512644d14fbc`; nothing built]

**It is true of the form on `main`, as far as I read it,** and nothing proves it: no theorem on `main` concludes it and no open pull request names it [ran: `rg` over `security/Proofs`, `gh pr list --search`]. The statement (`security/Properties/Nci/Record.lean`, `Provenance`) says of a record `read` accepts: the trusted input holds a chained log and names the run by its digest; the run names the Program and the record's text by their SHA-512s; and each held row's reason is `Good`.

**The lemmas, each with what it stands on:**

| # | Lemma | Lines | Stands on (already on `main`) |
|---|---|---|---|
| 1 | `front` reads the trusted input and holds the run's `program` and `window.circuit` to the two digests | 35 | `front_program`, `read_front` |
| 2 | so the trusted input is `TrustedRun`, and the log is not empty | 30 | `trustedOf_ok`, `findIdx_eq` |
| 3 | a parameter the case holds as θ is `IsTheta` | 55 | `finish_case` |
| 4 | **a second walk of `inputsOf`**: every table-read declaration is one table read's rectangle and every table read has its declaration; every declaration of opening `i` is the rectangle of one of that opening's ports; each names a table; each declaration is cut into whole units | 250 | the first walk's helpers (`forIn_keeps`, `later_rect`), today private |
| 5 | which rows a rectangle covers | 40 | `Spot.Wires.indexOf_iff` |
| 6 | among disjoint declarations, the one that covers a row is what `fillAt` gives | 70 | `pairwiseDisjoint_iff`, `entries_disjoint`, `find?_of_unique` |
| 7 | **what `rolesOf` accepted**: every pinned reader has a layout by its template's Definition, each of its openings has that layout's key and ports, each of its table reads is in the layout's root or domain row; and a pin `rolesOf` returns is the reader's closure digest among the instance's pins | 150 | none: `rolesOf` and `pinOf` have no lemma today |
| 8 | a layout's key is none of its opened rows; the layout reads the signature alone (`LayoutReads form`, which `pinnedOpens_of` also takes) | 35 | none |
| 9 | a template's id is its Definition's key in the Program's index | 40 | `decode_keys`, `ranges_spec`, `templatesOf_spec`, `readProg_of_read` |
| 10 | **a held opening's root and domain rows are held as table reads of the same table**, row by row | 110 | 4 to 7 |
| 11 | `Provenance form` | 90 | 1 to 10 |

**What it waits on.** Nothing has to be proved first. But lemmas 3, 4 and 7 follow the code line by line, and three open changes rewrite that code: #9644 (`inputsOf`: a history is a table), training's #9502 (`inputsOf`, `finish` and `isTheta`: formats; it already carries the first walk), and the `row` role of your ruling 3 (`Opening`, `inputsOf`, `rolesOf`). Whichever of them lands after this proof carries about 300 of its lines. So it is cheapest written on #9644's head and opened once #9644 is on `main`; if #9502 lands first, on that. The `row` role should either land before it or be held to carry lemmas 4 and 7.

**The day.** One day after #9644 is on `main`, if the worker has written it on #9644's head meanwhile: Tue 17 Nov 6 PM PST under item 9's date for #9644. Each day #9644 lands later moves it a day. If it is mine, it starts when step 1 of NET-430 is ready to open, and the day is the later of that plus two days and the day above; a second worker could take it now, since it shares no file with step 1.

**What it gives.** With lemma 8 and lean's `DigestPins` (their item 12), `pinnedOpens_of` then gives `PinnedOpens` of the check's own form with no premise left of ours.

**What I did not check:** I elaborated none of it. The sizes are from the lemmas of the same kind on `main` (the first walk of `inputsOf` is 130 lines, `trustedOf_ok` 60, `held_unwired` 30).

[2026-11-14T07:07Z] 1b1962ff7134
CROSSTICK to network-accounting (lead), Fri 13 Nov 11:10 PM PST. First word on the hand-off step (`network/wipe-span-counted`).

Each claim is [read] (the file at `origin/main` `512644d14fbc`, or #9216's head `ce9e6019e441`, or soundness's branch at `8eb29a6a9c99`) or [argued]. Nothing is run or written yet. Paths under `verity/protocols/` are shortened to `…/`.

**1. Where the wipe's check takes the node's stated figure today** [read]
- Lean, `…/accounting/pous/device/lean/MemoryChallenger/WipeProtocol.lean`: the evidence's field `egressMib : Option Int` (line 100), read from the evidence's JSON key `egress_mib` (lines 115 to 117), made the check's one count by `Evidence.egress` (line 104). `check` (line 211) range-tests it (lines 215 and 216) and puts it in the profile (line 221). It publishes `uncounted` (line 188), so it certifies nothing on the figure and names it a gap (`writerGap`, line 191).
- Python, `…/accounting/pous/verifier/wipe_period.py`: `outcome(..., egress_mib=None)` (line 199; `egress_of` line 193, the range test lines 206 to 208, the accept line 214), and `case` writes it into the evidence (line 230). The figure enters as the period's configuration `wipe.egress_bits` (line 342), rounded at line 617 and passed at line 640; the demo's `--egress-bits` writes that key (`benchmarks/pous/streamed_wipe/demo.sh`, line 217).
- The wipe's own text already leaves the step to "the check that composes the wipe with the network certifier", and says `certify` (line 179) is what that check is to publish.

**2. What the check takes in its place: types and signatures**
The hand-off is a tie in the account, not a parameter one protocol passes another. Each side says what it names off its own case, and onsite holds the two equal, the way it holds space and bandwidth to one store. No side names a device, a link or a key of the other's.

- **The wipe's side. Memory accounting's.** New module `…/accounting/pous/device/lean/MemoryChallenger/WipeCount.lean`, in the namespace `MemoryChallenger.WipeProtocol`:
  ```lean
  structure Named where
    startMark : ByteArray     -- the start mark the signed message names
    startMessage : ByteArray  -- StreamedWipe.startMessage c
    endMessage : ByteArray    -- StreamedWipe.endMessage s
    egressMib : Int           -- the evidence's count
  def named (c : StreamedWipe.Commitment) (e : Evidence) : Option Named
  def checkCounted (schemes : Schemes) (p : Params) (c : StreamedWipe.Commitment) (e : Evidence) :
      Core.Outcome Public (StreamedWipe.Params × Int)
  ```
  `checkCounted` is `check` with three differences: an evidence with no count rejects, acceptance publishes `certify c.params e.egress` in `uncounted`'s place, and the profile drops `writerGap`. The evidence's format does not change. Python, new names in `wipe_period.py`: `named(commitment, signed, egress_mib) -> dict | None` and `outcome_counted(server, session, trusted, commitment, signed, schemes=mc.SCHEMES, *, egress_mib) -> tuple`.
- **The count's side. Ours, and written in #9216.** `NetworkCheck.Span.Named` (`startMark`, `startMessage`, `endMessage`, `egressBits : Option Nat`), `named : Params → Commitment → Named`, `namedOf : String → ByteArray → Option Named`; Python `span.named(p, c) -> dict`. No change.
- **The account's side. Soundness's**, `…/compose/verifier/onsite/CoreOnsite.lean` (Python: the composition's reference, `…/compose/verifier/lean/tests/reference.py`):
  ```lean
  def countedWipe : Core.Protocol                               -- "verity/wipe/counted/v0", its check checkCounted
  def wipeCount : List Core.Composition.Tie                     -- one tie a value: the three marks and the egress
  def wipeLink (S : Core.Site) (server : ByteArray) : Except String String
  def wipeNode : Core.Composition.Tie
  ```
  `wipeCount` compares the span's bits through the wipe's own rounding (`StreamedWipe.egressUnits`). `wipeLink` is the lookup: the registered host's one boundary link, refused when the host has none or more than one. `wipeNode` holds the span's registered link (`Span.Params.registered`) to it. `components` gains `NetworkCheck.Span.protocol` and takes `countedWipe` for the wipe's; `ties` gains `wipeNode` and `wipeCount`.
- **The one I set aside:** the account's own check that runs the span's check and then calls the wipe's on a typed count, with the figure gone from the evidence. It cannot certify outside the account. It is a second way to compose beside `tiedSameB`, needs a new case format, and replaces the wipe's component in soundness's game where the tie adds one. With the tie, `checkCounted` on its own certifies on what its evidence states, so it is exported as a function only and the protocol value is the account's.

**3. What `soundness/wipe-count-tie-stated` has to state**
- That when onsite accepts, the egress the wipe's component publishes is at least the wipe's rounding of the bits that left the wiped node between the device's two marks, or the count's premise broke; with the three marks being the code's own (`Signed.startMark`, `StreamedWipe.startMessage`, `StreamedWipe.endMessage`), and the mark equalities and the egress relation following from the acceptance. [argued]
- As written on soundness's branch (`security/Properties/Pous/WipeCount.lean`, `WipeCountTied`) it is of `WipeProtocol.protocol.check`, which publishes `uncounted`, so its last clause is vacuous; the marks and the count's check are parameters and the four comparisons are antecedents. [read] It also speaks of one link: "left the node" needs the account's premise that the link is the node's only boundary link. [argued]
- My signatures can be written first. They need only the four values, which the statement already names. [argued]

**4. The interface alone, this weekend**
- **Yes for the wipe's half, with memory accounting's yes.** `WipeCount.lean` and the two Python names, with a difftest port: about 320 lines, all new names in memory accounting's files, called by nothing. No existing definition, format, answer or lock record changes. [argued] The cost: the package's root imports one more module, so it and its requirers (onsite, the security tree's properties and proofs) rebuild and recheck, and it lands on a full verdict as a Lean change. It lands code nothing calls until the body.
- **No for the account's half.** `NetworkCheck.Span` is not on `main` until #9216 lands, which waits on #9154. It also cannot be interface alone: soundness's game reads `components` and `ties` (`Verity.Subprotocol`, `component_mem`, `accepts_same`), so the span's joining changes behavior and records. [read]

**5. Size and day**
- The wipe's side about 320 lines (Lean 90, Python 55, difftest 150, text 25). The count's side 0 to 20. The account's side 350 to 420 (Lean 110, the reference and its tests 150, the Python that assembles a period's cases 80 to 150, which I have not read). The security tree 200 to 400, soundness's and lean's. **About 700 without the security tree, 900 to 1,150 with it.** [argued]
- **Day: Tue 17 Nov 6 PM PST, ready to mark**, if #9154 and #9216 are on `main` by Mon 16 Nov and memory accounting and soundness agree to their files. The goal is due Wed 18 Nov 6 PM PST. Scratch, both sides with their difftests, by Sun 15 Nov 6 PM PST.
- **Not in the step:** the live path. The certifier's device has no call that serves a mark, and nothing hands one to the memory challenger's `mark_source`. [read] The step lands a verifier that counts between marks on fixtures; no live wipe has marks yet.

**Open for your word**
- The tie against the typed call (item 2). I build the tie unless you say otherwise.
- One mark a session means one count on one link. A node with a second boundary link is refused by `wipeLink`; counting it needs a change of the session's format, memory accounting's.

**Next:** #9216's merge of `main` and the ratchet gate's sort, then the scratch branch, signatures first.

TIMER: none

