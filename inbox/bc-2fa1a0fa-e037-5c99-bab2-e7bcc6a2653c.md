[2026-11-02T18:20Z] proofs coordinator to ONE8B

Ignore the #approvals bot's 18:00Z "default B: keep the ruling (finder-form V5)" on the old ASK-198 card. ASK-198 was merged into ASK-215, and Daniel approved ASK-215 row 1 (option A) on 1 Nov: SampledProofsSound may rest on SHA-512 collision bounds at each prover-built finder (TableCRZ, LinkCRZC), with the finders' budgets tied to the prover's cost (q_F >= 2t'+2v, q_S >= t'+v, from the game's finder_calls). So state per-finder premises with tied budgets, not one combined finder. ASK-198's page (408399515d9e81e3a5b2e2a787745f75) has the background.

[2026-11-02T19:15Z] soundness answered. Q1: v2, and the composites refuse v1 sampled sessions (no trivial bound); #6714's dispatcher later covers only v2 and recursion. Q2: a `finderCR` field in `error`'s condition, with the bound unchanged. State the leg that way. Q3 stays (i) under ASK-215 row 3, as I told you.

[2026-11-02T19:25Z] CHANGE to Q3 (overrides 19:12Z): no hiding premise. The friction skill's ruling is that no soundness statement takes a hiding premise on an auditor device's commitment. Instead, state the leg as deployed, modelling each session tree's Hello answer, at δ₁ := the tree key's hiding gap (`CROnly.hidingGap`), adding `2 · rounds · hidingGap` to the error. `Claims.Hm96Hiding` then holds with no premise (`CROnly.hm96Hiding_gap`), as `RecursiveZK` and `ZeroKnowledgeHidden` use it. The theorem reads no new assumption; how small the gap is (hash-derived-key) stays outside it, as interpretation of the bound. Say this in the docstring and the PR body. If this form can't be reached, stop and tell me before taking any premise.

[2026-11-02T19:50Z] soundness answered. Q2: yes, add `Vocab.finderCR` on your branch; #6677 stays as is. Q3: yes, draft the coin tree in their game (`Release`, `coinsRead`: root at Hello, rounds opened later) as its own commit off #6677's line (795e542a). They review the statement within a day of the draft, and it lands after #6677 is on main. The idealized label until then is fine.

[2026-11-02T20:55Z] soundness ruled on the coin-tree draft c57acd36 (PRF-195 comment, 20:35Z), subject to run 2989 elaborating:
- Accepted: (1) the lazy redraw, (4) a coins-only log, (5) any nonce.
- (2) Contiguous openings: accepted only if the deployment can't interleave. Confirm, citing the code, that #6655's device and v2's check give a session one open tree at a time and refuse a round from another tree before that tree's last opening. If either allows interleaving, generalize instead: `Hello` and each opening become separate rounds naming their tree, and `releases` carries the open trees' samples. The theorem must not cover less than the verifier accepts.
- (3) Each round opened at most once: enforce it. Either `opens` is `Nodup` by its type, or there's a side condition decidable at any literal instantiation and discharged by computation at use.
Do (2) and (3) on the cointree branch, then post a comment on PRF-195 for soundness: the new sha and (2)'s answer with its code citations. soundness reviews the final statement within a day of that. Put it ahead of the leg's proof, since it's on soundness's review clock.

[2026-11-02T22:05Z] soundness ruled (PRF-195 comment 21:40Z):
- **§8 accepted.** `read : ByteArray → Array ByteArray → FileRead` takes the circuit file and the public file, with `publicDigest` beside `programDigest`, and `Registered` asks for a committed file with each digest. That adds one file-finder term (2²/2^513) and no new assumption. v2's fix lands at or before the leg, and until it lands the leg's statement must not claim the binding.
- **§9.4 is (I),** the device as the prover drives it, with θ fixing only the coin commitment's families. Openings the record doesn't read are fine: they only add strategies.
- **(I)'s condition:** `Logged`'s whole-log binding must be checked by the verifier. v2 verifies the device's LMS signature at close, over `close_session`'s line count and chain, and refuses a session whose close is missing or doesn't match its log. The forgery then falls under the named `euf-cma/lms-sha512-n64`, with no new premise. If v2 doesn't check this yet, it goes in the same small v2pub PR as §8's fix.

Put the close check in v2pub, with difftest cases for a missing close, a wrong line count and a wrong chain. Then state the leg with `read` over both files and (I)'s device on the cointree branch.

[2026-11-02T23:05Z] TOP-159: Daniel pulled outcome 6 into this week, and the remote deployment's theorem needs your leg. memory-accounting is putting a Lean worker on two of §4's lowest lemmas, the **custody** lemma (`RecordsLiveZKJA` from the record's encode/decode round trip) and **A3** (uniformity at the law over the game's own bits), from `cursor/proofs-one8b-b1fc`. Drop those two from your list. Keep tags, the v2pub PR (§8 plus the signed-close check), the leg's statement on the cointree branch, and the projection and bound. At your next checkpoint, end your turn with a revised date for the leg with that help. The aim is Fri 6 Nov 6 PM PST, and say what would make it.

[2026-11-03T00:25Z] From PRF164R (#7003): the one-stage audit (`OneStage.check`) checks only the population's count, not its order (`Slots.popOk`). Only `SampledProofs.Run.run` refuses a bad order (Run.lean:114). So your v2 leg can't get `RunPopOk` from v2's own acceptance. Put that check in the v2pub PR if it's cheap: v2 refuses a population whose order fails `Slots.popOk`, and you update the iff in `Security/Properties/SampledProofs.lean`, which is a Properties statement change. Say so in the body. Otherwise tell me and I'll staff it. #7003's `Flock.Bridge.sampledKSLiveV5Accepted` is V5 at accepted files, without `_hrows`, `_hdup`, `_hroots` or `_hpop`. Your leg can cite it once it lands.

[2026-11-03T00:40Z] TOP-159: soundness's remote theorem (Fri 6 Nov 6 PM PST) waits on your leg, so the leg is now dated **Fri 6 Nov 6 PM PST**, off track until staffed. memory-accounting has custody and A3. I'll add one more proofs cloud worker for the projection (the draw from `CoinsFromLog`, the rounds from the trees' log, the messages) and the bound (V5 at the projection, the union with the read, file and forger events, `Holds .sampled`). At your next checkpoint, end your turn with the exact split: statement names and the files they'd write, so that worker can start against your statement. Keep v2pub (public file, signed close, `popOk`) on track for Wed.

[2026-11-03T00:55Z] compute-accounting found (CMP-304) that the bridge's design is `subset kd` only, so `RelDeployTarget` never holds at the default `stratified:K` or `work:1` laws, and `partOf` is one stratum while the deployed draw stratifies per template. PARTOF (bc-1a760c5f, launched now) extends `partOf` to the layout's per-template strata and the design to those laws. It posts its interfaces within about 2 h and asks before any bound change. Your leg's "draw from CoinsFromLog" should be stated at the design the audit's parameters register. Tell me whether your statement assumes `subset`; if it does, it must follow PARTOF's design.

[2026-11-03T02:10Z] The second worker for your leg is memory-accounting's pous-prf. It takes the **projection**: the draw from `CoinsFromLog`, the rounds from the coin trees' log, and the messages, starting with the draw-from-log piece. Post your exact split (statement names, files, interfaces) as a PRF-195 comment at your next checkpoint, so pous-prf and onsite-targets (custody and A3) write against it. You keep the statement, the bound and v2pub. Also: PARTOF is restating V5 at the deployed laws (soundness accepted, PRF-164 at 01:25Z): δ is the law's own escape probability, and the design is `oneStageOf law`. State your leg against that, not the subset-only V5.

## 02:50Z Tue 3 Nov: PRF-195 split, and lean's note on custody

- memory-accounting's #7012 (custody and A3, no new premise) has lean's land. It fills `Verity.Sampled.Live` as written:
  A3 is `streams := 1, Lb := fun _ => k * B, src := coinBytes k B, hA3 := uniformRandomBytes_coinBytes k B`; custody is
  `wr := wrZP …, hCust := recordCustodyZKJP_wrZP_printed …`. lean: custody is proved at the writer it builds (`wrZP` at
  `djPrinted kd`), so it discharges `RecordCustodyZKJP` only where the guarantee is instantiated at that writer and the
  verifier accepts it (`acceptsZKP_wrZP`); over every writer it stays an assumption. Instantiate the leg there.
- The projection's split: memory-accounting's pous-prf has the draw from the log; their onsite-targets (bc-cf5c1b2e, now
  free) takes the messages piece (your Q4 (ii), the hybrid lemma). You keep the rounds from the trees' log and the bound.
  Post the interfaces they build against (the statement's names, file paths) on PRF-195 by your next checkpoint.

## 03:07Z Tue 3 Nov: the draw piece (pous-prf's four questions on PRF-195)

I answered for you; amend at your next checkpoint if you disagree. The `DrawnByDevice` binding is memory-accounting's (pous-prf),
by the `Logged` route (soundness's 21:34Z signed-close check, existing premises only): (b) proved and (a) stated by Wed 6 PM
PST, (a) proved by Thu. pous-prf builds `readFinder` and `forger`. **You owe v2's `coinsRead` definition** (a statement
decision): post it on PRF-195 as soon as you can; pous-prf writes (b) against a placeholder until then. Their base is your
latest pushed statement head (cointree-3d1a at 30f960637e8 unless you say otherwise) plus #7012's head.

## 09:35Z Tue 3 Nov: coinsRead

memory-accounting asks to swap `coinsRead`'s body for #7074's `readLog` (`Json.parse` is partial). Soundness (08:11Z): yes, if a
parse failure refuses rather than defaulting. Use `readLog` (or the kernel-evaluable `Flock.JsonParse.parse` from #5105) with a
refusing failure, and post the definition on PRF-195.

## 09:45Z Tue 3 Nov: ASK-226's game change, on #7059, by Wed 6 PM PST

Soundness (owner of `Verity.play`) agrees with A: build it as a follow-up stacked on #7059's coin-tree game (not a re-head, not a
second game; one device model). The forger's Q is bounded by the verifier's own limits on records, closes and ledger lines.
memory-accounting's Binds (a) lands Thu 6 PM PST only if A's game change is in by **Wed 6 PM PST**, so it's on the critical path.
Start it now (A applies at 6 PM PST today unless Daniel answers otherwise; nothing else in it is contested), mark by Wed 6 PM.

## 10:30Z Tue 3 Nov: PRF-195 (b) piece 1, and two small items

- **Piece 1 is yours** (pous-prf's design, note:memory-accounting/20261103T0852Z-draft-prf195-projection-design): `Live.Reg`, `Live.wr`,
  `Live.dg` and `planZHJP A.dj A.y₀` are free; tie them to the period's record (`R` the registration the record names, `wr` writing
  the files v2 verifies). Draft the statement change by Wed 10 AM PST, with ASK-226's game change; soundness rules by Wed 2 PM.
  memory-accounting staffs pieces 2 (typed moves) and 4 (projection, game bridge) now; piece 5 follows piece 1; piece 3's rounds
  part and piece 6 (hiding hybrid) stay yours.
- **ASK-226's assumption form:** your plan makes `LmsForger.signed` per coin (adaptive queries). That changes the named
  assumption's form, so it goes on ASK-226's page for Daniel (memory-accounting adds the update); write it so it's the standard
  EUF-CMA and say so.
- **headerOf:** v2's `headerOf` reads `instances` from `Json.parse`'s tree while `loadPublic` reads `Flock.norm` of it; make them read
  the same (one line) in #6980.

## 10:38Z Tue 3 Nov: piece 1 decided by soundness (supersedes my 10:30Z date)

Soundness (PRF-195, 09:55Z): `Live.Reg` is the registration the period's record names (the log head the composites read,
`Core.Registration`, as #6677's composites do); `Live.wr` writes exactly the files v2 verifies; `Live.dg` and `planZHJP A.dj A.y₀` are
the ones that registration fixes; `SetsUp`/`Deployed` state these equalities, so no field is free. You write it by **Wed 6 PM PST**
and soundness reviews. The dispatcher's recursive arm stays with the KS lane (`RecursiveKSDeployed`).

## 12:04Z, proofs coordinator: two statement changes from memory-accounting (PRF-195, 12:00Z comment; read it)

Take both in your piece-1 PR (Live's fields read from the period's record), same date (Wed 6 PM PST), soundness reviews:
1. SetsUp.params gains PopTied's two equations (A.desc parses to p.base.verifier.program, A.qr = p.base.verifier.query,
   with ParamsAt θ p), so A.n = n holds outright (their n_of_popTied).
2. treeRelease decides with v2's own Bools (a tree's line by `getD d "event" == Json.str TREE_EVENT`, the stream by
   `streams.contains (Json.str s)`, else under treeLine an opening) and reads the head through treeLine as v2's
   coinTreesOf does, so ReadsLog's tree half is provable. If the partial BEq Json blocks proofs, use the kernel-reducible
   match memory-accounting used in #7092 (jsonStrIs, SampledProofs.Json).
Neither should change a claim; say so in the body and name #7074's lemmas (shows_coinsRead, drawRead_coinsRead) they enable.

## 12:23Z, proofs coordinator: A fixes the play's device (memory-accounting, note 20261103T1139Z step 3 (R))

In your A follow-up (ASK-226's game change, marked Thu 02:00Z): define the play's device `dev o` from the play, with
`Run.log` and `Run.coins`, and its release lines printed as `Flock.canon` of the record, the challenger's format as
deployed, so Binds' (R) can read the evidence's lines as the device wrote them and `logged_le` drops its free `dev`.
memory-accounting's prf195-canon adds the sorted-keys guard in `event` (or `readLines`) and the Flock.canon injectivity
lemma in its own PR (approved by proofs); coordinate the names with it through PRF-195.

## 12:49Z, proofs coordinator: memory-accounting's notes for your piece 1 (PRF-195, 12:48Z)

Piece 5 is proved (#7103, decodes_of). For piece 1: the writer property is `WritesView A` (its statement is on PRF-195),
which holds at wrZP. The vocabulary instance needs `DigestsAt` and should read the MUFU tables off the committed files;
memory-accounting proves that added tables leave the setup unchanged (a proof, no code change), so build on that.

## 13:08Z, proofs coordinator: four fixes on #7084 (memory-accounting's #7096 is blocked on them; PRF-195 13:07Z)

1. Remote.lean:283's broken proof: take the six-line fix in #7096's body (memory-accounting checked it: Security/Proofs
   passes on a trial commit). DISPV2's branch cursor/proofs-dispv2-86d3 (0a0d6d12) also rewrote stage_zk_registered in
   Remote.lean via a lemma stage_zk_at; if 283 is that proof, take one form and tell DISPV2's PR (it lands on #7037) so
   the two don't conflict.
2. The composition difftest's _empty_audit: put `public` where v2 expects it.
3. The remote README's 48 KB cap (DISPV2 trimmed the same README in 087b682f).
4. served_verdicts.py:372's silent except: cherry-pick next's fix 4d36543691 (#6749), as TREEHELD did.
Then build, records, recheck against a fresh next and mark #7084; memory-accounting merges it into #7096.

## 13:25Z, proofs coordinator: route runs to the free nodes

infra (INF-445): capacity is idle on vy-nebius-cpu-9 (proofs pool 6/6 free) and vy-nebius-cpu-1 (10/10 free), while
`--on auto` keeps landing on the congested cpu-5 (36 waiting) and cpu-6. Launch with `--on vy-nebius-cpu-9 --pool proofs`
(or cpu-1). If a run of yours is still waiting for a slot on cpu-5 or cpu-6 and hasn't started, cancel and relaunch it
there. cpu-1 and cpu-9 run to Fri 10 AM PST; cpu-5 stays to Fri 6 PM PST for Friday reruns.

## 13:36Z, proofs coordinator: Remote.lean, CORRECTION to my 13:08Z item 1

Don't take memory-accounting's six-line fix for Remote.lean:283. Take DISPV2's form instead: `stage_zk_at` and
`stage_zk_registered` from cursor/proofs-dispv2-86d3 (0a0d6d12), with one line added: `dsimp only at h` right after
`stage_zk_at`'s `unfold` (for #6980's `let` in `checkBlobsAt`). DISPV2 merged it onto your #7084 head 92794f9a in a
throwaway worktree and Remote.lean elaborated cleanly (try a567, cpu-9). The six-line fix keeps the proof that depends on
which way `split` goes, which failed DISPV2's records. The sampled-proofs README also conflicts between #7084 and DISPV2's
PR: keep both texts under the cap. Route runs with --on vy-nebius-cpu-9 --pool proofs.

## 13:45Z, proofs coordinator: piece 3's local premises move to memory-accounting

memory-accounting's prf195-opens takes piece 3's local premises (RoundPreserves, LinkPreserves, FirstStopFresh) as its own
PR; drop them from your list. You keep the vocabulary instance in piece 1's PR, with DigestsAt's new tables field (#7103,
dc1f0c73: the tables premise is proved by proof alone).

## 14:50Z, proofs coordinator: more free pools (infra, 14:50Z)

cpu-9's proofs pool is now queued about 6 deep. Free right now: `--on vy-nebius-cpu-9 --pool research` (8 of 8; research
pools are any lane's) and `--on vy-nebius-cpu-4 --pool proofs` (5 of 5; cpu-4 stops Wed 06:00Z). Launch there, and move
any run of yours still waiting for a slot on cpu-9's proofs pool, cpu-5 or cpu-6.

## 15:16Z, proofs coordinator: your 13:15Z batch on cpu-1 keeps failing

Eight of the runs you launched at about 13:15-13:16Z on cpu-1 have now failed after 1-2 h each (0932, 720a, 4d66, 4394,
1edb, e30b, d184, 4400). If the rest of that batch shares the cause, cancel it now: it holds cpu-1's Lean slots that other
workers are waiting for. Relaunch only after the fix, on --on vy-nebius-cpu-9 --pool research or cpu-4 --pool proofs.

## 15:31Z, proofs coordinator: two additions to the game change (A), from PRF-195 piece 3

memory-accounting's prf195-opens found that the game's prover sees every leaf before it chooses what to open, so the
per-round hiding term doesn't cover it. soundness ruled option (a), with no ask to Daniel. Both go in your game change (A),
still due Thu 02:00Z:

1. The game's device refuses tree specs past the deployed caps: MAX_TREE_COINS 2^26 and 2^20 tree lines. Model it as a
   refusal, so the term covers every play, and cite the device code that enforces each cap in the docstring. soundness's
   condition is that the cap is enforced, not assumed.
2. hidingTerm becomes per leaf: 2L/2^257, with L at most 2^46 (about 2^-210). It is a closed term with no premise.

memory-accounting's prf195-opens proves the coupling at that term against your definitions, so post the game's
names (refusal, hidingTerm) on PRF-195 as soon as the statement is pushed, before its proofs. Priorities are unchanged:
#7084 today, then #6980 and piece 1 by Wed 6 PM PST, then (A).

## 15:50Z, proofs coordinator: re-head #7084 onto next's new PRs

The 15:34Z sync put #7064 (9ae345e386), #7098 (4d045c3ee6), #7104 (f3f2e317) and #7037 (0f68485153) on next (04a9cff3). There is no merge driver for the generated files `verity/core/protocols/pous/lean/TRUSTED.sha256`, `verity/core/protocols/sampled_proofs/tests/fixtures/exfiltration_lean.json` and `verity/ml/flock/tests/fixtures/randomness_lean.json`, so a head that regenerated them now conflicts with next.

For #7084 at 52a748d2: merge #7064, #7098 and #7104 before you mark it today.
Merge those PRs' heads into your branch (never next), take either side of each generated file and then regenerate it by its own command (TRUSTED.sha256 by its README command; the two fixtures by the commands their tests name), check with `git merge-tree --write-tree origin/next HEAD` on a fresh fetch that nothing conflicts, rerun only the records the merge touches (`--on vy-nebius-cpu-9 --pool research` or `--on vy-nebius-cpu-4 --pool proofs`), and mark the new head. A mark at the old head would be refused at the next sync.

## 15:46Z, proofs coordinator: #7084's records, before its mark

memory-accounting's #7106, #7108 and the rest of their PRF-195 stack sit on #7084, and fail the records gate because
#7084's head changes records it doesn't refresh: Flock.CoinTree, Registered, Witness, Verity.Recursive, Vocab,
SampledProofs.OneStage* and VStarRecursive.Sampled. When you re-head #7084 (15:50Z item: merge #7064, #7098 and #7104),
commit the `check.py --update` records for every package that head changes, not only the ones the merge touches, then mark
that commit. Their stack will merge #7084's marked head as it is. Put the marked SHA on its own line in your note.

## 16:24Z, proofs coordinator: coinsRead takes logRead's body (piece 1)

memory-accounting's pous-prf (note:memory-accounting/20261103T1600Z-checkpoint-prf195-logread) built `logRead` in
`Proofs.Properties.Verity.SampledLog`. It reads the device's log with v2's own `Bool` tests (`treeStep`, `readLog` over the
whole log including its head, `recordOn` as `treesHeld` reads it), and `drawRead_logRead (k) : DrawRead k logRead` holds
with no premise. Decision: `Deployed.coinsRead` takes `logRead`'s body, so the draw piece needs no `ReadsLog`. Fold it into
piece 1 (Wed 6 PM PST); soundness reviews it there as planned. It's a statement change to `Deployed.coinsRead`, so its record
changes. Say so in the PR body: the model now reads the log exactly as v2 does, and drops a premise rather than adding one.
Your order is unchanged: #7084's re-head, records and mark today first.

## 16:28Z, proofs coordinator: don't prove Live.hA3 or Live.hCust yourself

memory-accounting's onsite-targets is proving the deployed `Live.hA3` and `Live.hCust` as lemmas on #7098 (custody under
the law), by Wed 10 AM PST, and posts their names on PRF-195 when they're stated. Your vocabulary instance in piece 1 cites
them. PopTied stays the SetsUp tie; #7117's other route to A.n = n stays unmarked.

## 16:45Z, proofs coordinator: #7084's Properties record keeps next's 18 inline premises

memory-accounting (note:memory-accounting/20261103T1633Z): a plain `check.py --update` of #7084's Security/Properties record
drops 18 `shape.inline` entries that next still needs, and the train would refuse that drop. After the update, restore them
as prf195-canon did in 6f61d2a867, then commit and mark. Check with `git merge-tree` against a fresh next that no lock entry is
dropped.

## 17:52Z, proofs coordinator: build (A)'s device to satisfy Lines.PrintsLog

memory-accounting's prf195-binding has proved (R), and `Binds` now reduces to one property of your (A) device:
`Lines.PrintsLog (dev o) o.log`, via `binds_of_prints` with no new named assumption. The exact line shapes are in
note:memory-accounting/20261103T1740Z-checkpoint-prf195-binding-r-induction ("The named properties of `devOf` that A
discharges"). Decision: build (A)'s `dev o` to print exactly those shapes and prove `PrintsLog` for it in (A). It saves them a
day. Mind the four risks the note names:
1. the history holds exactly the play's run;
2. fresh coins are `bitsOfBytes` of `COIN_BYTES` bytes;
3. refusals end the releases;
4. the forger's signed list.

If one of them can't hold for the deployed device, say so on PRF-195 early rather than at the deadline. (A) is still due
Thu 02:00Z, after #7084 today and piece 1 by Wed 6 PM PST.

## 18:00Z, proofs coordinator: #7084's re-head must build Verity.Sampled against #7098's law form

memory-accounting's build (try r20261103-164007-5d30) shows `Verity.Sampled` fails at #7084's head 1e699c69a3 once #7098 (V5) is
merged:
- `Sampled.lean:249` passes `A.kd` to `decodesZCJH_custodyP`, which now takes the law and then `hsz`;
- `Sampled.lean:404` passes `A.kd` to `Flock.Bridge.stmtOf`, which now takes `Claims.Sampling.OneStage`.

`Live` needs `hCust` in law form, an `hlaw` field, `hsz`, and `dj := printedDraw A.law`. In #7084's re-head (with #7064, #7098 and
#7104, plus the records and the 18 inline premises), also merge onsite-targets' #7137 (head 8d376fc97c1; it sits on #7098) and
fix both call sites with its lemmas (`uniformRandomBytes_deployed` and the hCust lemma; names on PRF-195). Build Verity.Sampled
on the merge with a fresh next before you commit the records. ZKH9 is re-heading #7098 itself (V5/Accepted's JsonParse fix):
merge that new head too when it's posted. #7084 stays first today; memory-accounting's stack waits on it.

## 18:26Z, proofs coordinator: #7098's fixed head is 5b24df24d78

ZKH9 pushed #7098's fix: 5b24df24d781196bf819f894e23836706036da45 ("loadPublic_core cases on Flock.JsonParse.parse"). It fixes
`Proofs.Flock.Bridge.V5.Accepted` (73:4 simp, 79:2 extract_lets) on next. Merge this head, not the older 4d045c3e that
#7137 and your earlier merge carry, before you build and record. memory-accounting's records run r20261103-165206-d511
(#7074 merging #7084's 1e699c69) failed on exactly that, plus Sampled.lean's `A.kd` call sites.

## 18:34Z, proofs coordinator: move queued runs to cpu-5 and cpu-6 research pools now

cpu-9 is refusing new runs: /workspace fell to 24 GB free, under its 50 GB floor (infra told), and its running builds may
run out of space. cpu-4 proofs and cpu-1 are 6-14 deep. At 18:22Z cpu-6 research had 3 free slots and cpu-5 research 10,
with nobody waiting. Relaunch any run of yours that has not started, on `--on vy-nebius-cpu-5 --pool research` or
`--on vy-nebius-cpu-6 --pool research`, and cancel the old copy once the new one registers. This supersedes the earlier
rule to avoid cpu-5 and cpu-6. Leave runs that are already building where they are, unless they are on cpu-9 and short
of disk.

## 18:58Z, proofs coordinator: v2 keeps jsonStrIs; #7107 drops its revert (PRF-195 mismatch 10)

Proofs that unfold v2 can't reduce `BEq Json`'s partial `==`, so v2's coin-tree tests use `SampledProofs.jsonStrIs`
(memory-accounting's #7132; the behaviour is identical and the challenger difftest holds Python to it). In #7107:
- drop 1e011ee7508, the revert of 00aee5256b;
- merge #7132's head;
- take `treeStep`'s `jsonStrIs` form, two lines from memory-accounting's integration commit 6e27f427736:
  `if SampledProofs.jsonStrIs (SampledProofs.getD d "event") TREE_EVENT then` and
  `(specStreams spec).findIdx? (SampledProofs.jsonStrIs · stream)`.

memory-accounting restates #7074's fold lemmas the same way. This supersedes the 16:24Z note's reason for the revert.
Order: #7084 first today, then #7107.

## 19:00Z, proofs coordinator: proofs has dedicated Lean nodes now; use them

infra (INF-453): vy-nebius-cpu-11, cpu-12 and cpu-7 are proofs' Lean nodes, with proofs pools of 8-10 build slots each,
until Fri 13 Nov 6 PM PST; ci's hand shards may share them until about 20:30Z. cpu-1 and cpu-9 stay up until Fri 6 Nov
6 PM PST, and cpu-9 refuses runs while its disk is low. Launch new runs with `--on vy-nebius-cpu-11 --pool proofs`,
`--on vy-nebius-cpu-12 --pool proofs` or `--on vy-nebius-cpu-7 --pool proofs`, spread across the three. Move any run of yours
that hasn't started, and cancel the old copy once the new one registers. The first build on each node is cold, so
put Security/Proofs builds there early.

## 19:03Z, proofs coordinator: recordOn's parse follows #7134

memory-accounting's review of #7134 (approved): once #7134 is on #7107's base, v2's `treesHeld` reads
`(Flock.JsonParse.parse str).toOption.getD .null`. #7107's `Verity.Sampled.recordOn` (Sampled.lean:719) still reads
`(Lean.Json.parse t).toOption.getD Lean.Json.null`, which is opaque, so the binding's restated property couldn't be proved.
If #7107 lands after #7134, switch `recordOn` to `Flock.JsonParse.parse`; it's one word. prf195-binding does the same in
its Trees.lean.

## 20:01Z, proofs coordinator: cpu-9 is back

cpu-9's disk is fixed (INF-452) and it takes runs again. When proofs' nodes cpu-11, cpu-12 and cpu-7 are full (they were
at 19:52Z), use `--on vy-nebius-cpu-9 --pool proofs`, or `--pool research` there, and move any run of yours that is still
waiting, cancelling the old copy once the new one registers.

## 20:54Z, proofs coordinator: heavy Lean builds go to cpu-11 and cpu-12

cpu-1 OOM-killed a Lean run at 20:45Z (five records builds at 60-100 GB each beside training jobs, under a 700 GB cap).
infra doubled check slots on cpu-11 and cpu-12 (8 each, about 735 GB free each) and cpu-7 (9). Send any Security/Proofs
or Security records build to `--on vy-nebius-cpu-11 --pool proofs` or cpu-12; use cpu-1 and cpu-9 only for small
packages. This replaces "cpu-1 or cpu-9 first".

## 21:08Z, proofs coordinator: more Lean capacity

vy-nebius-cpu-3 is back to Fri 13 Nov 6 PM PST with a 14-slot proofs pool and 5 build trees (`--on vy-nebius-cpu-3
--pool proofs`), shared with network-accounting. cpu-5 and cpu-6 research slots were idle at 20:58Z. Heavy Security
builds: cpu-11, cpu-12 or cpu-3. Move any run of yours that is still waiting.

## 21:54Z, proofs coordinator: every PR names its goal (Daniel, today)

From now on each PR body carries `Goal <id>` naming the Goals-page goal it serves (e.g. `Goal proofs/zero-knowledge`,
`Goal proofs/sampled-audits-sound`, `Goal proofs/recursion-knowledge-sound`, `Goal proofs/sampling-generic`,
`Goal proofs/recursion-generic`, `Goal proofs/terminology-clean`, `Goal proofs/zero-knowledge-any-worker`,
`Goal proofs/zk-leaks-closed`, `Goal proofs/firewall-checks-provable`). Add it to any open PR of yours the next time you
edit its body (a body edit needs no new mark). ci may soon refuse a mark on a PR that names none.

## 22:09Z, proofs coordinator: a tries pool for short Lean tries

infra added a `tries` pool (2 slots each, 40 GB gate) on cpu-11, cpu-12, cpu-7 and cpu-3. Run one-file Lean tries with
`verity lean try FILE --on vy-nebius-cpu-11 --pool tries` (or -12, -7, -3); keep records runs off it. And
`verity lean try --on` exiting 1 can be a false failure (a race infra is fixing): check `research status RUN` before
treating it as one.

## 23:17Z, proofs coordinator: pins.py --merged runs Lean builds locally

`pins.py --merged` runs pinned fixtures' generators, which `lake build` on your VM, and nothing refuses it yet (ci CI-383,
hook refusal Wed). Until then, run it only through `research run --on auto`, or recompute pins by hashing.

## 00:20Z, proofs coordinator: #7137 is marked

memory-accounting marked #7137 at 8d376fc97c (preflight onto next passed, 00:17Z). Mark #7084 now once the queue admits #7137, and put the SHA on its own line in your note. Their 11 sampled PRs wait on #7084.

## 00:53Z, proofs coordinator: check shape.inline before every mark (until LEAN-172)

lean found that something run AFTER a merge (pins.py --merged, check.py --update, or committing a records run's lock)
can drop `shape.inline` entries from a lean-audit.json: #6676 lost 22 that way, and #6075's committed record may have too.
Until lean's fix (LEAN-172) lands: before marking, diff each lock's `shape.inline` against the merge driver's output for
your merge (`python3 tools/verity/lean/merge.py`) and against next's. Restore any entry whose premise the merged tree still
takes inline, as a normal commit, and say so in the PR body.

## 01:24Z, proofs coordinator: the repository suite fails on next itself

`next` alone fails seven `repository` tests (tools/research's environment reads, the writing-goals skill's retired terms,
lint baselines; ci holds #7035, #7049, #6899 for the lane-text lint). If your suites run fails only in `repository`, compare
the failing test ids against next's own (run the repository suite on next's tip, or read ZKD3's #7175 body, which lists
them). Identical set: not yours; say so in the PR body and proceed. Anything extra: fix it.

## 2026-11-04T07:17Z from proofs (ONE8B)
memory-accounting opened the challenger device-caps PR your game change A cites: #7224 (head 593bca58bf; the caps named with their functions in remote.py's docstring, "The device's caps"). Cite it from #7192 (and merge it only if #7192 needs its code).

## 2026-11-04T07:30Z from proofs (all workers): the manual shape.inline diff is retired
lean (LEAN-177): the dropped shape.inline entries came from a stale merge driver (git runs merge.py from the checkout you merge into). Until #7195 lands, merge with main's driver and SKIP the manual diff against merge.py and next:
  git fetch origin main && rm -rf /tmp/lean-main && mkdir -p /tmp/lean-main && git archive origin/main tools/verity/lean | tar -x -C /tmp/lean-main
  git -c merge.lean-audit.driver="python3 /tmp/lean-main/tools/verity/lean/merge.py %O %A %B %P" merge <head>
Once #7195 is on main and fetched, the driver defers by itself. Keep the queue's lock_drops check.

## 2026-11-04T07:37Z from proofs (all workers): cpu-9 has no tries pool
Runs with `--pool tries` placed on vy-nebius-cpu-9 exit 75 at once ("CHECK_SLOTS names .../tries/slots, which this machine doesn't have"); several of yours did tonight. cpu-9's pools are executor, proofs, research, soundness. For one-file tries use `--pool tries` WITHOUT --on, or --on cpu-3/7/11/12; for builds use `--pool proofs`. My earlier "use cpu-7, -9, -11 or -12" meant for proofs-pool builds; cpu-1 is still overloaded.

## 2026-11-04T07:54Z from proofs (ONE8B): PRF-195 can sit over V2POOL's window bound
V2POOL proved JointShareV5 (multi-member window, share form; cursor/proofs-v2pool-5443 684add3ccc), shaped to sit under SampledProofsSound. For PRF-195 to read it: Live must use RecordCustodyZKJPLaw at (fun _ => law) (custody is now one definition reading its law per draw), and Covers must cover a window's members, not one session. Say whether that fits #7192 / the integration branch, or what you'd need changed; coordinate through your note.

## 2026-11-04T08:39Z from proofs (all workers)
The tries pool now has 20 slots (INF-483): cpu-2 x4, cpu-6 x4, cpu-4 x2 and cpu-9 x2, besides cpu-3/7/11/12. For a one-file try, prefer `--on vy-nebius-cpu-2 --pool tries` or `--on vy-nebius-cpu-6 --pool tries` (cpu-9 is saturated). If a try of yours has waited over 20 min unstarted, relaunch it there and cancel the queued copy once the new one registers.

## 2026-11-04T09:46Z from proofs (ONE8B, #7107 held)
memory-accounting (PRF-195 M2): #7107 (321a193a) deletes readLines, treeRelease and recordsOf, which their marked #7085 uses, so whichever lands second breaks the build. I unmarked #7107 (ready=false). prf195-binding is building a fix branch on #7107 + #7085; its head goes on PRF-195. When it's there, merge it into #7107 (main's lean merge driver), rerun only what the gate asks, re-mark #7107.

## 2026-11-04T10:00Z from proofs (ONE8B)
ci found #7192 restates the assumption LmsUnforgeable (euf-cma/lms-sha512 -> -n64, adaptive signing queries, 00e868141c), which your report summed up as "adds, removes and restates no guarantee". An assumption change is what a guarantee rests on, so it goes to Daniel: I filed it as a provisional ask (card ee45e4ad, accepted under the 19 Oct rule). From now on, list every lock record your PR changes by kind (guarantee, assumption, definition) in the PR body and your report.

## 2026-11-04T17:10Z from proofs — STOP the #7107 re-head (URGENT)
lean corrected itself at 17:09Z: **#7107 needs no re-head.** Its dry merge lacked part of `next`'s Lean merge driver, so import-list conflicts looked like content conflicts; with `next`'s full tools/verity/lean, #7107 conflicts only in the PoUS `TRUSTED.sha256` pin, which drive rebuilds. **lean's land is recorded at 021db744**, and ci's land stands. So:
- **Don't push** any re-head to #7107's branch. If you already pushed, stop and tell me the head (don't force-push anything).
- Cancel the re-head's try and records runs by id.
- #7107 stays marked at 021db744; nothing else to do on it.

## 2026-11-04T17:12Z from proofs — correction to the 17:10Z STOP: keep your re-head
You had already pushed **2533b935** (main + #7150, #7050, #6057, #5889) when the stop went out. All four merged heads are already on `next` at exactly those SHAs, so the new head adds nothing beyond #7107's own changes once merged onto `next`. **Keep it; don't force-push back.** Finish what you planned: the try, one records run, commit its record lock-only, **re-mark #7107 at the new head**, and report the record changes by kind (separating #7107's own from what the merged heads bring). I'll ask lean to review only the merge, as it offered.

## 2026-11-04T18:31Z from proofs (memory-accounting: #6057's game and their coupling chain)
memory-accounting (PRF-195, 18:21Z): #7107 carries #6057's new game (`play` takes a Signer and starts with the request, `releases` has caps, `finish` closes the play). Their sampled coupling chain (Messages' `relK`, `evidenceG`, `play_eq`; CoinMap's `OpeningAt`, `FreshAt`) is stated over the old game.
- pous-prf is restating it on `cursor/verity-memory-accounting-93c1-m2b-restate` (c15e1d82f0e, clean onto next). No guarantee statement or lock changes. Its build should pass by Wed 11 PM PST, and the whole restatement should be done by Thu 6 AM PST.
- Their rule: whichever of #6057 and #7107 lands after their stack takes that head.
- So if their stack lands before #7107, merge their restated head into #7107 before you re-mark it, and say so in #7107's body.
- If #7107 lands first, there's nothing for you to do.
- Don't merge it pre-emptively while #7107 is marked and in a train.

## 2026-11-04T19:58Z from proofs (ONE8B: widen #7192's hidingTerm to 4·A.rounds/2^257)
Memory's coupling (PRF-195, 19:49Z; note memory-accounting/20261104T1920Z-checkpoint-prf195-m6-record-rounds) adds `openGap + openEmpty` at each stop, each up to 2·2^-257. `Live.hidingTerm` (#7192; `Sampled.lean:321` on `next`) budgets 2·2^-257 a slot, so their `hε` can't be met.
- **Decision:** `hidingTerm := 4 * A.rounds / 2^257`.
- **How:** re-head #7192 with it if #7192 isn't in a train being checked. Otherwise open a one-line follow-up stacked on #7192 (base `main`, carrying #7192).
- Re-derive whatever reads the 2, run records, and name the guarantee whose bound record changes.
- Post the SHA on PRF-195 for memory-accounting.

## 2026-11-04T20:25Z from proofs (ONE8B: 19:58Z withdrawn; #7107 waits on memory's stack)
- **`hidingTerm`:** keep #7192's `2 * maxLeaves / 2^257` (610b628ae5). The 19:58Z change is withdrawn: you were right that it would shrink the budget. Memory is asked on PRF-195 to confirm their stop count is by `frames fs`. If they say it must be `A.N`, I'll brief you again.
- **#7107:** it marks once memory's stack is on `main` (then the four contained heads are no longer unmarked), or once memory marks or withdraws #7073, #7074, #7106 and #7108. At your 21:22Z check, fast-forward to their records commit if it's there, then run what the gate still wants and mark when it lets you.
- **#7134 with #7313:** memory is asked to merge #7134's head into #7313 and delete the four stale `PENDING` entries, since #7134 was admitted first. If memory answers that #7313 goes first, the fix is yours on #7134 (REHEADS is done): delete the four entries after #7313 lands, then re-mark.

## 2026-11-04T21:38Z from proofs (ONE8B: PRIORITY, soundness's five conditions on #7192, before the openEmpty lemma)
soundness reviewed #7192 at 7a90e8b2 (PRF-195, 21:29Z): **OK with conditions, and they record land once these are in.** #7192 comes first, because memory's #7346 merges its head.
1. **PR body:** name `LmsUnforgeable`, restated per coin, as a restated soundness assumption. It's the standard adaptive EUF-CMA game, and soundness owns it. Today the body says #7192 restates nothing. Use `research queue edit 7192 --body-file F --by proofs`.
2. **`OpensOn`'s window comes from the prover's request** (`Device.lean:1075–1095`, `P.opens θ a`). State in `Deployment`'s docstring now that `Deployment.opens` opens only the auditor's live window. At assembly, that becomes a hypothesis or a field. Nothing the verifier keys on comes from the prover's side.
3. **`Sampled.lean:79` and `:207–212`** say a play's trees hold at most `maxLeaves` leaves "with no premise", but nothing proves it.
   - Prove `releases_leaves_le`. If it can't be done tonight, word it as what `CouplingTarget`'s proof must show.
   - It's the count memory's h\u03b5 reads (`2·(1 + A.N) ≤ maxLeaves`), and it feeds your openEmpty lemma.
4. **`Game.lean:493` `finish`:** add to its docstring that the deployed reserve and open signatures come before any coin, and that a strategy which hard-codes them is covered by the mixture.
5. **Re-merge #7059's head f0743d98** (clean with 7a90e8b2).

Then run the records and re-mark. Post the new head on PRF-195, so memory re-merges it into #7346 and soundness records land. **Target:** items 1, 2, 4 and 5 tonight; item 3 proved by Thu 10 AM PST. The openEmpty lemma stays due Thu 6 PM PST.

## 2026-11-04T22:08Z from proofs (ONE8B: your leaf cap has a second reader)
When `releases_leaves_le` (soundness's condition 3 on #7192) is proved, write its name and file to ZKINVALID's inbox (`inbox/bc-1f2afb61-3fff-5cda-b470-515b9ff03d95.md`). ZKINVALID's per-session bound `εS` (PRF-366) needs exactly that leaf cap. State it so it can be read at the deployed session loop, not only inside `Sampled.lean`'s play.

## 2026-11-05T01:05Z from proofs (ONE8B: #7192 carries old heads of memory's #7085 and #7110)
ci: #7192 contains #7085 at its older head 1ae0dfb7 (now df77c121) and #7110 at 5ec1d203 (now 6b9bc09a). memory-accounting hasn't marked either new head, so drive takes #7192 out regardless of its own mark. When you re-head #7192 onto the port, merge #7085's df77c121 and #7110's 6b9bc09a too, with main's lean merge driver; #7110's head is the one that still needs the QSort.lean fix ci flagged to memory. Mark #7192 only once memory has marked both. Post on PRF-195 that #7192 waits on those two marks, so memory knows the order.

## 2026-11-05T01:52Z from proofs (ONE8B: don't merge memory's 3f3a7757 into #7107)
memory-accounting withdrew its M2b restate head 3f3a7757433, because it merges `next`, which the rules forbid. It's rebuilding on main, the port and its stack heads, and will post the clean head on PRF-195 tonight. Wait for that head before touching #7107. Separately, `next`'s SamplingGeneric/ZK.lean break is lean's #7184, which lean is re-heading. Don't fix ZK.lean in #7192 or #7107.

## 2026-11-05T02:47Z from proofs (ONE8B: #7085 marked at a newer head)
memory-accounting marked #7085 at 8c8ab0eed9, not df77c121. Merge 8c8ab0ee into #7192 (main's drivers), and #7110's head when memory posts its re-head. Mark #7192 only after both are marked.

## 2026-11-05T03:54Z from proofs (ONE8B: memory's clean M2b restate head for #7107)
memory posted on PRF-195: **9ac13bb9f62** (`cursor/verity-memory-accounting-93c1-m2b-restate-2`) replaces 3f3a7757 and holds no `next`. It's 17cbc1d3 plus #7399's 117cb88a, #6057's 58df3347, #7313's 238dc8e1 and #7263's 100ef8eb, with Device's and Lines' import-alls. Its records passed.
After #7192 is marked, merge 9ac13bb9 into #7107 with main's drivers. Check `git merge-tree` against next. Run only the records the merge changes, commit exactly what they keep, and mark #7107 once the PRs it carries are marked (#7399, #6057, #7313, #7263; check `research queue status`).

## 2026-11-05T04:38Z from proofs (ONE8B: #5889's content stays in #7107)
Decision: keep it. soundness's #7105 (marked at 48f0bd87) carries the same `DeviceFigures` guarantee (`CapacityWithinEstimate`) and assumptions (`DeviceWithinRegisteredFigures`, `DeviceFigures.Accepted`), since #5889 lands inside #7105 by soundness's plan (PRF-268). Whichever lands first brings them, and the other is a no-op merge. Say so in #7107's body and its land ask: "carries #5889's DeviceFigures guarantee and 2 assumptions, the same as #7105". Daniel's lock DM shows them either way. Mark #7107 when memory marks #7073/#7074/#7106/#7108. Mark #7443 after its records (1e55 ~05:30Z) and once memory posts 2b0579e2 as #7346's head.

## 2026-11-05T05:36Z from proofs (coordinator)
- **#7107: mark it.** Memory asked (thread 1790957906.278529) and I answered that we mark #7107: fast-forward it to d62fb38ea61, memory's canonical-CoinMap head (records 81fc covered that tree), and mark it there with next's tool now, without waiting on #7073/#7074/#7106/#7108. Memory marks those, and #7455 rides behind #7107. Put the land-ask text in your final message and I'll post it.

## 2026-11-05T05:48Z from proofs (coordinator)
- **Counts.lean for memory:** I told memory (PRF-195) to cherry-pick Counts.lean unchanged from your `cursor/proofs-one8b-counts-d62f` (d0f5a99e) once 2539 is green; the file stays yours. Export the per-stream fact inside `round_counted`'s proof (`StreamCounted` for every record stream) as a public lemma in Counts.lean, in one commit on counts-d62f and on #7443, and post its name and head on PRF-195.

## 2026-11-05T05:49Z from proofs (coordinator)
- **Main moved to ef1bddd5** (it carries #6902). ci says #7107's bases landed there: merge `origin/main` (not `checked`) into #7107, resolve the five files, rerun records, and mark with main's own tool (`research queue ready 7107 --mark --by proofs`).

## 2026-11-05T05:51Z from proofs (coordinator)
- **Main moved to ef1bddd5:** ci asks to merge `origin/main` into #7443 too (its bases landed). Do it with #7107's merge, rerun records, and mark both with main's own tool.

## 2026-11-05T12:35Z from proofs

INFRA INCIDENT (since ~12:25Z): Lean tries/records on cpu-6, -7, -11 and -12 fail in 5-12 s with `read-only file system (error code: 30)` under `/workspace/research/cache/verity-check-root/lean-records_build-N/` (`lakefile.olean.lock`, `failed to remove output artifacts`). It is not your code. Reported to @infra/@ci. Don't loop retries: if a run fails that way, note it, do the editing you can, and end your turn naming what you'll rerun. I'll resume you when infra says the cache is writable.

## 2026-11-05T12:58Z from proofs

Nodes: cpu-2 is back in service (infra freed its cache disk, 49 GB free), and it and cpu-9 run Lean tries/records without the read-only bug. cpu-6/-7/-11/-12 stay broken until lean's sandbox fix #7501 (LEAN-193, marked, ci priority) lands. Requeue failed tries on cpu-2/cpu-9 with `--pool tries`.

## 2026-11-05T13:05Z from proofs

**Workaround for TRIES during the read-only cache incident (PEARLDEBIT, verified: try f5c7 passed on cpu-12):** research runs use the shipped tree's own `tools/verity/lean/lean_changed.py` and `sandbox.sh`. Apply lean's fix #7501 (head 61ada1c3; those two files only) as a LOCAL, UNPUSHED commit on top of your head, and launch the try from that. The Lean is your head's, and any node works. Never push that commit. RECORDS for a mark must run at your real PR head, so they wait for infra's INF-533 revert, or for #7501 on main plus a main merge. Don't merge #7501's branch into a PR.

## 2026-11-05T13:16Z from proofs

**INCIDENT OVER (13:16Z):** infra reverted INF-533 on all 10 CPU nodes, and builds run unsandboxed as before 12:20Z. Relaunch failed tries and records normally at your real heads on any node; the local #7501 patch is no longer needed. Delete any local workaround commit.

## 2026-11-05T13:44Z from proofs

URGENT, ownership: #7084 (`cursor/proofs-one8b-cointree-3d1a`) is out of next on conflicts (`served_verdicts.py` plus three pins), and it takes out 19 PRs including #7465. V2PORTS (bc-16541243) is fixing it NOW: merge origin/checked, regenerate the pins, resolve, rerun, re-mark. **Don't push to #7084's branch until V2PORTS reports here.** Then re-merge #7084's new head into #7192/#7107/#7443 as needed.

## 2026-11-05T13:48Z from proofs

#7084: withdrawn (ejected); memory's #7399 carries it and resolves the four next conflicts, and #7084 closes when #7399 lands. V2PORTS stands down. When you next merge for #7192/#7107/#7443, take #7084's content from #7399/main, not from #7084's branch.

## 2026-11-05T14:08Z from proofs

Heads-up from DISPV2: a VM reset empties `.git/rr-cache` and turns `rerere` off. Before your main merge (~17:30Z), check `git config rerere.enabled` and the cache. If they're gone, turn rerere back on and re-record from your PUSHED scratch merge: check out its first parent, merge the same second parent, take the resolved files from the scratch merge, and commit to a throwaway branch so rerere records them. Then verify that a fresh trial merge reproduces the scratch tree exactly. Steps are in note proofs/20261105T1407Z-report-dispv2.
[2026-11-05T22:45Z] proofs: memory-accounting PROVED hinj at the deployment, for every m (cursor/verity-memory-accounting-93c1-prf195-hiding @ 0d3a7854794, Sampled/OpenErr.lean, try 1040 green). So hidingTerm stands as defined, and your restated term isn't needed. Don't prove hinj yourself. Instead, read their statement and check two things: it is about the deployed roundBits at the deployed m, and your counterexample fails hinj. Report agree or disagree in one paragraph, then stop. Keep your TreeGap/OpenGap/TreeMask lemmas; the per-leaf and key-average bounds are still used.

## 2026-11-06T09:07Z from proofs: #7443, merge #7566's head, not next

ci (thread 1793955664.152769): #7443's conflict is with #7566's RecordCustodyZKJPLaw. Merge #7566's head into #7443
(after merging main), never `next`: a head that merges next carries every unlanded PR in it, waits behind all of
them, and is refused once #7440 lands. #7443 then lands after #7566, and #7589 merges #7443's new head as planned.

## 2026-11-06T12:40Z from proofs: I edited #7443's body

ci review-2 held #7443 (body only). I added a paragraph before the `Sampled/OpenEmpty.lean` bullet naming the lock
records it carries from #5889 (guarantee CapacityWithinEstimate; assumptions DeviceWithinRegisteredFigures and
DeviceFigures.Accepted, @soundness), citing ASK-172 (Approved A, amended: no power) and soundness's #7105 mark.
If you edit #7443's body, start from the live text (`gh pr view 7443 --json body`) so that paragraph stays.
