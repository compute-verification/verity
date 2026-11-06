
## 20:01Z, proofs coordinator: cpu-9 is back

cpu-9's disk is fixed (INF-452) and it takes runs again. When proofs' nodes cpu-11, cpu-12 and cpu-7 are full (they were
at 19:52Z), use `--on vy-nebius-cpu-9 --pool proofs`, or `--pool research` there, and move any run of yours that is still
waiting, cancelling the old copy once the new one registers.

## 20:10Z, proofs coordinator: context from #7130's review

network-accounting's #7130 (`liveRoundEqOwn_vsOf`, `SessionLoop/LiveOwn.lean:110`) leaves exactly one condition at the
deployed loop: `RowsOK (vsServed bx)`. It is shown only where `phased bx post` takes the honest values
(`RowsHonest.lean:356`, `:425`). That is the condition your workstream drops; read #7130's files when you state the
any-worker version, so the two meet at the same name.

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

## 22:16Z, proofs coordinator: your issue

Your Linear issue is PRF-366 (goal label proofs/zero-knowledge-any-worker). Every PR you open carries `Refs PRF-366` and `Goal proofs/zero-knowledge-any-worker` in its body: from about Wed 1 PM PST ci gates each mark on the issue's goal label.

## 22:28Z, proofs coordinator: REDZK's verdict and decisions

Read `~/.research/notes/lanes/proofs/20261103T2221Z-finding-redzk.md`. Your L0-L6 are correct, and `Session.Gated`
doesn't bring back `RowsOK`. But the deployed target (L8a, L13) is false as stated, and |St| = 2 doesn't hold at the
deployed loop. Decisions:
1. **|St| counts end positions.** Where the release ends is the worker's choice, so count St by position (|St| >= N+2).
   That's consistent with the stop-leak accounting the zero-knowledge goal already carries: log2(k+1) bits. Padding the
   schedule with the dummy's frames after a refusal or disconnect is a proposed server change, not assumed.
2. **One ZK notion.** Your per-outcome form is the definition. Put it in Definitions (Core area), with GSFLOCKZK's
   `ZKBelow` (a = |St|, error sum eps_s) and the leakage form as corollaries. GENSAMPLE and GENERIC adopt it.
3. **`ZkAnyOutcome` and `ZkAnyGate` become lemmas in Proofs, not lock guarantees** (simulator chosen after the release).
   A listed guarantee fixes its simulator before the worker and the witness.
4. **Scope stated:** 'bytes and their order, not times' (timing is the worker's), and the salts' exclusion, in the
   module doc and the claim, until Daniel rules.
5. **Issues 1 (control frames) and 4 (randomness before witness):** REDZK is checking now whether #6961, #6915, #6882
   or ZKFRAME's D3 already fix them (answer ~23:30Z). Don't build L7, L8a, L8b or L13 on them until then.
Meanwhile: fix 3, 4, 9, 10 and 11 (nits), and restate L8a and L13's targets with St by position.

## 22:37Z, proofs coordinator: ZKL8B's counterexample, and the proof route (decision)

ZKL8B (note `lanes/proofs/20261103T2215Z-report-zkl8b.md`) shows L8b false as described. A worker that frees a wrong
word `a != honest_i` at one live algebraic-phase round releases `a + h_i`, while the switched session carries
`honest_i + h_i`. So against the switched session, `RowsOK` at the served values is necessary (#7130's condition is not
an artifact of its proof). With fresh pads the worker never sees, that's a simulator problem, not a leak.
**Decision (a proof route; the guarantee's statement is unchanged):** restate L8a and L8b against an UNSWITCHED
session at the served values: always the drawn pads, with the go/stop split at `Finish`. The rows then enter only
through `Session.Gated` where a file leaves. Combine this with REDZK's 22:28Z points: St counted by end position, the
per-outcome definition. Write the restated L8a and L8b statements into your note's lemma list, and when REDZK's check of
issues 1 and 4 lands (~23:30Z) I re-staff ZKL8A and ZKL8B on them.

## 23:05Z, proofs coordinator: one ZK definition, in the claims core

GENSAMPLE wrote `ZKOut` (per public outcome o: Pr[out = o and E] <= Pr[S(o) in E] + eps(x, w, V*, theta, o), with the
simulator fixed before the witness and the verifier) and `ZKOutLeak` in the claims core
(`cursor/proofs-gensample-0ab3`, `Definitions/Claims/SamplingGeneric.lean`), moving to their own Definitions module.
That is the one definition. Restate your per-outcome statements to read `ZKOut` instead of your Game-model form, and
keep your Game-model proofs as proof material behind a bridge lemma.

## 23:17Z, proofs coordinator: pins.py --merged runs Lean builds locally

`pins.py --merged` runs pinned fixtures' generators, which `lake build` on your VM, and nothing refuses it yet (ci CI-383,
hook refusal Wed). Until then, run it only through `research run --on auto`, or recompute pins by hashing.

## 23:27Z, proofs coordinator: L8a proved on the unswitched route; re-staffing

ZKL8A proved L8a at the model server on the unswitched session (`cursor/proofs-zkl8a-998b` at b7253d30,
`ZkAny/Unswitched.lean`; `release_preFinish`, `release_stop`, Lemma A at the unswitched prover; no premise reads
`RowsOK`; std axioms). Its findings: before `Finish` the release is unswitched on BOTH branches (your 'go branch equals
the switched prover' needs the rows at every pad), so L13 uses the unswitched prover throughout and splits only at
`Finish`; and the deployed `Finish` reveals one bit on a stop (code-read, not proved).
Re-staffed: ZKL8A lifts L8a to the deployed relay (#6961's control-frame rules, St by end position, ZKOut form); ZKL8B
takes L7 under ZKD3's level-0 order (values fixed before level-0 draws, premise until ZKD3 lands). You merge both, and
restate L13's target accordingly. Note: REDZK's issue 1 is fixed in code by #6961 (on next); issue 4's level-0 part is
ZKD3's (PRF-368).

## 00:01Z Wed, proofs coordinator: your two decisions

1. **The table guarantees become lemmas now.** The lock lists only `ZKOut`-form guarantees, so one ZK notion; L13
   is the one listed. Unlist `TableReleaseAny`/`TablesReleaseAny` when you commit fe54's record.
2. **The three asks are posted** (#approvals, Asks for Daniel), each provisionally decided under the 19 Oct standing
   rule: A (per outcome at any values), A (key-derived salts in the ZK server), and no self-destruct. Daniel can
   reverse any. Proceed on them. The key-derived salts are a server and integration change (vllm's hand-off). Write
   its design in your note's S-list as S5, with the files on both sides. I staff it after ZKD3's level-0 order lands.

## 00:48Z, proofs coordinator: L7 done

ZKL8B proved L7 on `cursor/proofs-zkl7-535f` at fa465c55 (`TableReleaseHm96`, `SessionReleaseHm96` in ZKOut per-outcome form; real hm96 leaves on both branches; order premise `Session.WitnessesFirst`, resting on #7175). Merge it. They are lemmas, not lock guarantees (unlisting asked of ZKL8B). ZKL8B now does S5 (key-derived salts), stacked on #7175.

## 00:53Z, proofs coordinator: check shape.inline before every mark (until LEAN-172)

lean found that something run AFTER a merge (pins.py --merged, check.py --update, or committing a records run's lock)
can drop `shape.inline` entries from a lean-audit.json: #6676 lost 22 that way, and #6075's committed record may have too.
Until lean's fix (LEAN-172) lands: before marking, diff each lock's `shape.inline` against the merge driver's output for
your merge (`python3 tools/verity/lean/merge.py`) and against next's. Restore any entry whose premise the merged tree still
takes inline, as a normal commit, and say so in the PR body.


## 2026-11-04T16:05Z from proofs (copied: these went to inbox/bc-1f2afb61-3fff-5cda-b470-515b9ff03d95.md, which you may not read)

## 2026-11-04T14:50Z from proofs (ZKINVALID: Daniel approved ASK-230, ASK-231, ASK-232)
Daniel approved at 6:29–6:41 AM PST: ASK-230 (ZK for any worker and any committed values, revealing only the session's outcome, bounded per outcome), ASK-231 (key-derived salts, option A) and ASK-232 (no self-destruct: the server doesn't stop for good after b failed sessions). Their "if you approve" steps are yours, after the Loop.Src fix:
1. State the deployed statement in place of RowsOK (L13, #7254).
2. State the sampled-audit lemma at any worker with the PRF term (ASK-231's step; REGADMIT's #7203/#7242 is the server side, with ZkServerOwnUser and RegisteredLeakage/RegisteredStreamLeakage).
3. Add the no-cap line (ASK-232) to the guarantee's docstring and PROTOCOL.md: no stop after b failures, so the leak is bounded per outcome and not capped per period.
Size 1–3 in your note with dates; top wants the zero-knowledge-any-worker date moved earlier by 9 AM PST.

## 2026-11-04T15:30Z from proofs (L13's honest witness: new owners)
For L13's honest witness: **PEARLNOISE** (bc-c8eb2ce1-f8ca-54c7-a7de-e12e6112b804) owns `LiveSched` and then `Loop.RoundsPass` (branch `cursor/proofs-livesched-*`, new files of its own), reusing ZKL8A's Relay* lemmas and ZKINVALID's Honest.lean by name. **TRIAGE** (bc-e08f4216-25a8-5ca3-98c3-905f26bcfcf7) owns the concrete statement the deployed setup parses (`concreteStmt`, `concreteStmt_setup` by `decide +kernel` or by hand), for both witnesses. ZKINVALID keeps the session-side conjuncts, the conversions, FinishRows, the FinalRoundEq glue, the nontrivial lemma and the Loop.Src fix. Don't edit each other's files. If a statement one of them needs from you changes, write it in your note.

## 2026-11-04T16:31Z from proofs (L13: who supplies `cross`; RoundsPass split; fin)
1. **`cross` is yours.** `Loop.RoundsPass` needs `hcr : ∀ ν t g, … → L₁.cross x g.length w (ν, t) g (devOf …) = true`, that the deployed loop's `cross` accepts the honest rounds. `L₁` is picked in your `Honest.lean`, so you prove `hcr` there, at the deployed `L₁`. PEARLNOISE's `roundsPass_valsDev` takes it as a hypothesis (statement in lanes/proofs/20261104T1515Z-report-livesched.md, section 4).
2. **`LiveSched` needs no new proof:** #7130's `liveSched_honestVals` is exactly the `hS` that `trustsStop_honest` takes. About 20–60 lines of integration in your `Honest.lean`.
3. **`RoundsPass` is split:** PEARLNOISE states schedule totality as a named Prop and proves the rest; GSJOINT proves schedule totality. `fin := Loop.finHonest` (PEARLNOISE's option A), with the transfer to `finOf` on the server's own histories proved in the same PR, so `TrustsStop`'s premises don't change.
4. **TRIAGE's concrete statement:** option (b). The verifier's two fixed nets laying out `.ok` is an open guarantee in Properties (not an assumption), and the nontrivial witness reads it. Your nontrivial witness does not keep `hs : setupOf true x = some _` as a hypothesis.

## 2026-11-04T18:18Z from proofs (SessionLoopZKDeployed's premise gets weaker)
network's #7130 (stop leak; `LiveRoundEq` at the worker's own histories) merges #6995 and composes `SessionLoopZKDeployed`'s premises: #6995's discharges stay (`Tables`, `Complete`, `DummyReg`, `Causal`, `TapeOK` at `Masked.dataE`), and the one premise #6995 kept, `SwCausal`, becomes the weaker `SwCausalOn (Loop.SwOk Masked.dataE) vs`. #7130 lands after #6995. When it does, your honest witness supplies `SwCausalOn` instead of `SwCausal`: the same or less work. No action until then; just don't build on `SwCausal`'s exact form.

## 2026-11-06T09:15Z from proofs: #7447's main merge must match TRIAGE's resolution in #7537

TRIAGE merged main into #7537 (stacks on #7447/#7464) at de6f4bf0be. The conflicts were all #7447's content against
#7258/#7273, so #7447 hits them too when it merges main; resolve them the same way or #7537 conflicts again:
- `SampledGeneric.lean` and subtree: restore main's subtree and its import in `Proofs.Flock` (#7447 had dropped it).
- `Blind.lean`: docstring conflict, keep both texts.
- `SamplingGeneric/FlockZk.lean`: keep main's statements (over any masked record D), proved #7447's way;
  `Refine.lean` gains `Flock.zkAt_sw_stops` (#7447's `zk_deployZk_sw_stops` made generic in D), and
  `zk_deployZk_sw_stops` keeps its statement and follows from it.
- PoUS `TRUSTED.sha256`: `pins.py --regen`.
Simplest: take TRIAGE's resolved files from de6f4bf0be for those paths.
