## STOPS, Fri 13 Nov 02:22Z: second reading of the period's five statements (owner PERIODREAD): no empty premise and no misplaced quantifier found; each holds in ways that say less than it reads, the widest that the lane picks which of its roots can be sampled

I read the five statements about the period's check against the code they are stated over, and changed nothing. As far as I can follow by hand each one is true of the code, none takes a premise that nothing can meet, and no quantifier is in the wrong place. What I found is places where a statement holds and still says less than its summary reads. The widest: an accepted period does not mean that every root the lane committed in the monitoring period passed the audit. It means that every root whose session was reserved with the sample's family of coins did. The lane names the families when it reserves a session, and it opens and closes its own periods on the ledger, so it can commit a root on a session without that family, outside any period, and the check still accepts, listing that root among the "other sessions", of which nothing is checked. The statements say so openly; nothing says what a reader of the answer must do with that list. Second: nobody has yet produced an input on which the check accepts with the real audit inside it; the accepted periods in the tests use a stand-in for the audit. The rest are narrower. The two statements about whose ledger it is leave each line's signature free, while the check reads the order of signing from the signatures. The accept statement is stated for parameters the decoder never produces, at which the audit can read its coins under another key than the ledger's. And "audited once" in the third statement is, in the Lean, "published once with one audit entry".

Everything below is [read] at `f2889fd3ce` unless tagged otherwise; nothing was built or run, no Lean written, no file changed. P is `security/Properties/Spot/Period.lean`, L is `security/Properties/Spot/PeriodLedger.lean`, H is `security/Properties/Challenger.lean`; under `verity/protocols/spot/verifier/lean/Spot/`, C is `Period.lean`, A is `PeriodAudit.lean`, S is `ChallengerSession.lean`; G is `verity/composite/device/live_challenger/auditor.py`. None of the five is proved or listed in a lock at this commit: the two modules are roots of the package (`security/Properties/lakefile.toml:101`) and nothing else names them.

### 1. `LedgerMeans` (P:76-91)

**What it asserts.** Whenever the ledger's half returns windows for a ledger: the ledger verifies under the verifier's key and holds an entry stamped at or after the period's end; every session the ledger opens for the lane (a v2 open under the lane's name, numbered inside the period's run of entries) is returned, among the windows when its layout holds the sample's family and among the other sessions when it does not; each returned window is such a session, has ended, and is listed by a closed period of the ledger all of whose listed sessions are windows; each returned other session is such a session without the family; there is at least one window, no root and no session is returned twice, and a window's phase is its place.

**True of the code** [derived], from `read` (C:159-175) and `windowsIn` (C:209-230): `checked` refuses a line that is no entry (S:687) and an entry whose number is not its place (S:574), so the open's `seq` and `reaches`' position are the same count; it refuses a second open of a root or of a session (S:596-597), which gives the two `Nodup`s and makes `x.ended.isSome` (P:83) the same ledger window's; a session joins one period, once (S:590, S:620), under a name used once (S:640).

**Ways it holds while its meaning fails.**

- **(a) The lane picks which roots can be sampled.** `Sampled` (P:66-67) is whether the open's `families` holds the label. An open carries its reserve's `families` (S:124, S:605), and the reserve's are the requester's argument, any distinct labels (G:1077-1085, G:2056). A lane writes its own `period-open` and `period-close` (G:95-96), and the gate takes a reserve from a lane with no period open (G:1131-1137). `windowsIn` refuses a session without the family only inside a period that holds a window (C:221-224); any other goes to `others` (C:210, C:230). So [derived] a lane that closes its period, then reserves a session without the family and commits a root on it, gets an accept with that root in `others`. P:35-39 says nothing is held of such a session. The summary the check opens with, "did every root a lane committed ... pass sampled proofs' audit" (A:2-4), needs `others` to be empty, or a rule for its reader; no statement and no step of the check has either.
- **(b) An other session's fields are free.** P:86-87 (and P:158-159) tie only `root` and `session` of an element of `others` to the ledger. Its `families`, `withheld`, `place` and `period` (C:138-144) are published (C:239-241, A:205), and a `windowsOf` returning anything in them satisfies the statement: `¬ Sampled p x` is said of the ledger's window, not of `s.families`. So a reader cannot settle (a) from the published fields on the statements' strength.
- **(c) "The lane" is a name, and v2.** `Committed` (P:61-63) reads `laneOf` of the open's body (S:498: its `for` when non-empty, else its `by`) and asks a v2 open. Who may reserve under a name is the gate's (G:1811: `--for` is the admin's): the device's behaviour, in no statement and not in `History` (H:113-115). A v1 open inside the run (S:574 lets v1 entries precede the first v2 one; its root is its `commit`, S:595) is never `Committed` and is in neither list.
- **(d) The order of the windows is not stated.** P:91 gives `phase = i`; nothing says the windows are in the order of their opens. `AcceptMeans` hands window `i` the records of the windows before it (P:141), "in the ledger's order" by A:42 alone. [guess] of low weight: whether one-stage v2's use of `log` depends on the order I did not read.
- **(e) The monitoring period is two positions that the device's stamps pick** (P:50-57, C:116-120). An entry with no time is passed over (C:107-113), and the ledger's form asks no time of an end, a period's entry or a one-step open (S:424-442). Real time is in no statement. L:25-27 says so; I name it so that nobody reads "during the monitoring period" as a time.

### 2. `AcceptMeans` (P:130-141, with `AuditedAt`, P:104-122)

**What it asserts.** Whenever the check accepts, with any audit of one registration in one-stage v2's place: it publishes the lane and the two ends it was asked; its windows are the ledger's half's, in the same order, and its other sessions the same; the evidence holds one entry a window; the profile is the audits' in order; and each window has exactly one audit, for which: the ledger's session on the window's root is a v2 one that ended closed and sealed its files; the pins log the open sealed records the commit it ran, and that commit is on the verifier's copy of main's first-parent line; the evidence's session files are that session's under the verifier's key; every open of the root withholds its coin commitment; the session's log shows the family's coins released after exactly one value besides the root, the audit's registration; the evidence holds one case, whose record hashes to that registration; the parameters the check composes for the window parse as one-stage v2's, with the records of the windows before it; the case's public file names no shared rows; the evidence the check composes parses; and the audit accepted them, publishing what the window shows.

**True of the code** [derived], from `run`, `windowAt` and `caseAt` (A:237-303): an accept has no undecided window (A:314-316), so every window passed and `before` is every record in order.

**Ways it holds while its meaning fails.**

- **(f) It is stated for parameters the decoder never produces.** P:131 quantifies over every `AuditParams`. `paramsAt` hands one-stage v2 the challenger key `p.keyJ` (A:93), while the ledger and the session's files are checked under `p.ledger.key` (C:160, P:111): two fields (A:69-73) that only `AuditParams.of` sets from one JSON (A:108, A:114). At a `p` where they differ the statement holds, and "the verifier's parameters for the window, which name the period's challenger" (P:28-29) is false of the audit. Likewise only `.of` refuses an `audit` that names `seed_salt`, `day` or `declarations` (A:110-113); `paramsAt` passes every key of `audit` but `log` and `proofs` through (A:92). The check as deployed decodes (`params := AuditParams.of`, A:347), so this is the statement's and not the check's: a theorem that composes `AcceptMeans` with one-stage v2's guarantee needs "`p.keyJ` reads as `p.ledger.key`" and `audit`'s form from somewhere, and `AcceptMeans` gives neither.
- **(g) That the audited session is the window's is not a conjunct.** `AuditedAt` binds the session by `auditedIn` on the root (P:108), which takes the first ledger window with that root (S:720); `au.session = w.session` is nowhere in P:104-122. It is true because `checked` refuses a second open of a root (S:596-597), which none of the five states: `LedgerMeans`' two `Nodup`s (P:89-90) are over the returned sessions, not over `wins`. [derived]
- **(h) No accept is known at the check's own audit.** `inner` is any (P:131), as the difftest needs. At `OneStageZk.checkBlobs` the tests hold no accepted period: the difftest says no proof there is one that one-stage v2 accepts, and that accepted periods run through the stand-in alone (`verity/protocols/spot/verifier/lean/tests/test_period_audit_difftest.py:5-11`); its test of the real audit asserts a reject or an undecided (:572-606). The check also refuses a public file whose header names `shared_rows` (A:253, P:118-119), which Flock's stager writes when rows are shared (`verity/protocols/flock/prover/python/verity_flock/circuit.py:1290`) and the vLLM adapter's drawn stager asks for (`adapters/vllm/verity_vllm/protocols/pouw/h1/drawn.py:445`). [guess] today's served registrations are of the refused form; I did not trace which registration a period would audit. Until one accept is run, `AcceptMeans` and `CommittedOnce` at the real audit speak of a case nobody has produced.
- **(i) What the window's parameters leave out.** The check never sets `seed_salt`, `day` or `declarations` (A:79, A:48-49). One-stage reads each as optional and holds nothing through one that is absent (`Spot/OneStage.lean:243-245`, :355, :375). So what one-stage holds a record to through them, the lane's declared program and the window's seed salt (`Spot/OneStage.lean:31-32`, `Spot/ChallengerDraw.lean:14-16`), is not held in a period's audit [derived]. `AuditedAt` shows the parameters only as `paramsAt`'s value (P:117), so a reader sees this only by unfolding it.
- **(j) "Exactly one registration" is of one family.** `sampledOn` reads the roots of the family `label` alone (A:183-185). A second registration after which the session released another family's coins is passed over, where A:26-27 says a second registration "is never passed over". [guess] whether a window can hold a second registration that matters; not read.

### 3. `CommittedOnce` (P:149-161)

**What it asserts.** Whenever the check accepts: the ledger verifies and reaches the period's end; each root the lane committed in the run on a session with the family is the root of exactly one published window, of that session, which has exactly one audit entry; each root it committed on a session without the family is among the published other sessions; and each published window is a root the lane committed on a session with the family.

**Ways it holds while its meaning fails.**

- **(k) "Audited" is "has one audit entry".** P:156 says `∃ a, pub.windows[i].audits = #[a]` and nothing of `a`. The title (P:143) and the module's "exactly one accepted audit" (P:31-32) read more; P:148 defers what the audit is to `AcceptMeans`. Quoted alone, `CommittedOnce` gives no audit.
- **(a) again.** The half for a root without the family (P:158-159) is membership in `others` and nothing else. `CommittedOnce` does not say that `others` holds only such roots, or none twice; both are in `LedgerMeans`, through `pub.others = ws.others` (P:136).
- `periods` is bound and unused (P:153). Harmless.

### 4. `LedgerLinesOfDevice` (L:56-63)

**What it asserts.** For any history of the device and any ledger the ledger's half reads: either the ledger is a list of lines with one stamped at or after the period's end, and the line at each place `i` is an entry numbered `i` whose text without its signature is that of some line of the device's own ledger under the key; or the fixed finder outputs a break from the ledger and the history: a signature in the ledger that verifies under the key over a message the device never signed, or two different strings with one SHA-512 digest.

**Checked and sound** [derived]: `(unsignedText line).isSome` (L:61) closes the reading where two lines that are no entry agree as `none`; a forgery is one relative to `dev` (H:331-333); the finder hashes each line's tagged text and message on both sides (H:258-269, H:294, H:320), so a line whose message the device signed says what a line of the device's says, or the finder has a collision.

**Ways it holds while its meaning fails.**

- **(l) The leaf is read from the signature the statement leaves free.** L:25-26 says a line's signature is not held to the device's. The check reads from it: the leaf each entry was signed at (S:577) orders the entries (S:580), is each window's `leaf` (S:622), bounds the record's leaf between the open's and the end's (S:631), and `sessionNamed` holds the session's record to the leaf the end names (S:815-817); and `prev` hashes each line's whole text, its signature included (S:669). A presented line that says what the device's says under a signature the device did not make is on the statement's first side and is no `Break`: a forgery is over a message the device never signed (H:333). So what the check concludes from leaves, that the session's record was signed between its open and its end, is held to the device by neither statement [derived]. It closes by asking the finder for a signature the device did not make (strong unforgeability), or by taking the device's chain (G:832 writes each line's `prev` from the line before) as a fact of the device, with the lines' texts among the strings the finder hashes; today they are not (H:258-269).
- **(m) The device's line may be at any place** (L:62). Said at L:19-21; `LedgerOfDevice` is the answer.
- **(n) `dev` is any.** At a history that lacks a line the device wrote, an honest ledger is a break; L:29-33 says so. Nothing in Lean ties `dev` to the key's holder (H:28-30), so the statement is as good as the history its reader puts in.

### 5. `LedgerOfDevice` (L:79-87)

**What it asserts.** If also every line of the device's ledger under the key is an entry numbered by its place (`Numbered`, L:69-70): either the presented lines are, their signatures aside, exactly the first lines of the device's ledger, as many as were presented and in order, and one of them is stamped at or after the period's end; or a break.

**The premise can be met** [read]: the gate numbers each line by the count of lines before it (G:832) and appends only the six kinds the reader takes (G:2059, G:2081, G:2159, G:2189, G:2364; S:457-460). **It follows from statement 4** [derived]: a presented line at place `i` says what a device's line says, that line's number is `i`, and by `Numbered` its place is `i`.

**Ways it holds while its meaning fails.**

- **(o) `Numbered` is all or nothing over the whole history.** L:69-70 asks every line of the device's ledger under the key to be an entry the reader takes, the lines after the presented ones too. One line this reader refuses anywhere under the key (a later schema's, a new kind, a body the form refuses, S:424-442) makes the premise false, and the statement is then silent about a ledger whose presented lines are sound. [derived] the argument needs only "a line of the device's that is an entry numbered `i` is at place `i`", which asks nothing of the other lines.
- **(p) `Numbered` is a hypothesis outside the named assumptions.** L:65-68 says it stays one until `History` defines the numbering. How it is listed when the statement is locked (an open premise, or part of `trusted-host`) is the owner's; I did not hold it to the lock's rules.
- **(l) again.** "None dropped and none added" is of texts without their signatures (L:85-86).

### Not established

- Nothing was built or run: every claim is from reading at `f2889fd3ce`. The branch's head has moved since (`91d05f87f5`), which I did not read.
- That any of the five is provable as stated: not attempted, and no proof is on the branch at this commit.
- (a) as a run case: I built no ledger with such a session and passed none through the difftest.
- One-stage v2's inside (`OneStageZk.checkBlobs`): not read, so the weights of (d), (i) and (j) are guesses.
- Whether a served registration's public file names `shared_rows`: two lines read, not traced.
- Whether a real ledger on the auditor's host meets `Numbered`: not run.
- `ChallengerKey.ofJson` and `paramsOfJson` on `challenger.key`: not read; (f) rests on `paramsAt` and `AuditParams.of` alone.

Pushed head of my own work: `cursor/proofs-stops-drawn-7b3e` at `7c776d1125f7`, unchanged; nothing unpushed. No run is out. To post: nothing.

## CUT, Fri 13 Nov 02:23Z: ALL THREE FILES OF THE CUT COMPILE at `bb96dbb00a` (`Cut.lean`, `CutPrints.lean`, `CutHeld.lean`: 40 theorems, three standard axioms, no `sorry`); the read STAGECOINS asks for is not provable from acceptance alone, so I give `Good` with a read, under one more named premise

Plain words. The cut is the step of the two-stage audit's proof that says the challenger's log the evidence carries, for the first stage's and the second stage's sessions, is what the auditor's device released, unless the prover forged the device's signature or found a SHA-512 collision. All of it now compiles: the statement, the bound, the reduction to what the device's history prints, and the proof that the form's own check holds the second stage's records to the device's coin trees. What is left is one gap STAGECOINS found: on a play where the check reads no log at all, the statement holds vacuously.

What compiled [run]:
- Try 6, `r20261113-014528-6f45` on vy-nebius-cpu-12 at `bb96dbb00a`: done, rc 0. `CutHeld.lean`: `lean exit 0`; axiom probe: "none of its 10 theorems uses an axiom beyond propext, Classical.choice and Quot.sound"; the sibling check passed. Its dependencies, `Cut.lean`, `CutPrints.lean` and TWOLEG's `Steps.lean` among them, built at that head in the same run.
- Try 5, `r20261113-010822-a348` at `6c782c1022` (in the 01:46Z entry): `Cut.lean`, 7 theorems, and `CutPrints.lean`, 23 theorems, the same verdicts.
- Warnings only, in all three: `first` fallbacks never taken, unused simp arguments, deprecated `if_pos`/`if_neg`. No statement depends on them.

What the three files prove, at any history of the device [run]:
- `cutBound`: from `Wired`, `Binds` and `SignsPlay`, \(\Pr[\text{verdict} \wedge \neg\,\texttt{LogIsDevice}] \le\) the read finder's term plus the forger's term.
- `binds_of_accepts : BindsOfPrints dev θ`: `Binds` holds wherever the device's history at each play prints the play's log in its two sessions (`PrintsTwo`), with no other premise. `first_session` and `second_session` are its two halves.
- With HISTORY's `PrintsTwo` and `SignsPlay` at the two-session history, the one premise left on the cut is `Wired`, the assembly's [relayed: your 01:02Z message says HISTORY's two compile against my definitions].

`CutReads`: a correction to my 01:11Z entry. I wrote there that I would prove `Accepts s → readsLogs s ≠ []`. It is not provable at this base, for three reasons [read]:
1. Where the first stage read no byte, the run's coin check accepts only `null` coin evidence (`Spot/Run.lean` 167–171, `coinsHold`). The cut then has no first-session log to read, and `readsLogs` reads both sessions or none.
2. Where it read some, the run's coin check reads under the key the evidence carries, not the parameters' (`ChallengerCoins.lean` 105–118). This is the premise RUNCOINS carries as `CheckReadsDraw`.
3. The form never reads the first session's log as coin trees. Only the device's run does (`PrintsOn`), so the first read exists on a log the device printed, and otherwise the record carries a break.

So I give `Good` with the read, and the same terms:
- New names, in a new file: `LogRead o := readsLogs o.sent ≠ []`, and the cut's bound at `fun o => LogRead o ∧ LogIsDevice o`.
- One more named premise, `FirstReads θ`: on every accepted play, `drawCoins` under the parameters' key reads the first stage's coin evidence on the boundary's window. It is RUNCOINS's `CheckReadsDraw` at the parameters' key, plus "the first stage read a byte".
- `LogIsDevice`, `cutBound`, `PrintsOn`, `PrintsTwo` and `SignsPlay` stay as they are, so nothing HISTORY or STAGECOINS built changes.
- STAGECOINS then takes the read from `Good` and drops `CutReads` [derived].
- I also wrote at 01:11Z that a `Good` with a read would make `Binds` false. That holds without `FirstReads`; with it, an accepted play with no read carries a break [derived, not compiled].

The proof needs four pieces [derived]:
- the parameters' key is at the deployed LMS parameters (the two-stage `paramsOfJson` runs v2's, which refuses another: `Limits.paramsOfJson_deployed`);
- the first read exists or its record carries a break (`Drawn.drawn_log`, `PrintsOn`, and the one-stage leg's `readLog_isSome`);
- the second stage's records are on the blobs (the per-member form reads each);
- the second read exists (`coinTreesOf_fold`, `releasedRootsOf_pairsOk`).

Hours: the statement in a try by 03:30Z; the proof compiled by about 07:00Z [guess].

STOP RULE
- Nothing [read].

NEEDS
- STAGECOINS and ASSEMBLY: say in your next entry if `Good` as `LogRead ∧ LogIsDevice` does not fit what you read of it. I write it that way unless told otherwise.
- RUNCOINS: `FirstReads` is yours to discharge, from `CheckReadsDraw` and the run's step lemmas. I state it in my terms (`firstDraws`, `firstCoins`) and take it as a hypothesis.

To post
- Nothing.

Not established
- `cutBound` at `LogRead ∧ LogIsDevice`: not written.
- A two-stage audit whose first stage reads no coin: the cut reads no log of it and says nothing of its second stage's coins. `FirstReads` excludes it. Whether the registered rule allows it is not read.
- The earlier-signature hole: at a history of one play, an earlier period's genuine signature in the evidence makes the error ⊤. Soundness's change of Tue 17 Nov closes it.

Pushed head: `bb96dbb00a`. Nothing is left unpushed.

