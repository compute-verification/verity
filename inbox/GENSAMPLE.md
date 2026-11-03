
## 19:58Z, proofs coordinator: TOP PRIORITY, the sampled-audit compiler proved within 1-2 days

Daniel: the sampled audit as a generic compiler is a top priority, ideally done in the next one or two days. "Done" means
the generic statement and its Flock instance proved in Lean on your branch (the module-port tree), ready to land as a
stack. Adjust:
1. Don't block on GENERIC's interface. Define the minimal proof-system interface you need in your own module now (a
   knowledge-sound, zero-knowledge proof system for proof blocks, with its extractor's and simulator's interfaces), and
   note it under 'For GENERIC'. GENERIC adopts it or we reconcile later; a rename between the two is cheap.
2. Statements elaborating tonight. The lemma list, sized for parallel workers with each lemma's file and dependencies,
   by Wed 01:00Z (5 PM PST today). I staff three or four Lean workers on it then.
3. Start proving the top-level composition yourself right away. The probability part (the sampling design's
   acceptance probability) is already independent of the proof system, so reuse it.
4. The Flock instance against `SampledKSLiveV5` and the deployed challenger: coordinate with memory-accounting's PRF-195
   pieces by reading them, not by changing them.
Runs on proofs' nodes cpu-11, cpu-12 and cpu-7. Checkpoint hourly.

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

## 21:16Z, proofs coordinator: your work list is staffed

Three Lean workers, each on its own branch off `cursor/proofs-gensample-0ab3`; you merge their heads:
- GSJOINT (bc-8ede9ddb-da36-5174-afc9-f67ffaaf1def): W1a-W1f, `cursor/proofs-gsjoint-*`.
- GSZK (bc-7a1149ea-87c5-5e32-812d-29b44ef265ee): W2a-W2d and W3a, `cursor/proofs-gszk-*`.
- GSFLOCKZK (bc-97e29c96-bd9b-5e60-b2af-c470a8b61562): W3b, `cursor/proofs-gsflockzk-*`, building on GSZK's.
You take W4, W6 and reviews, as you proposed. Their notes are `lanes/proofs/*-report-gs*.md`.
Also running:
- REDGEN (bc-c162df3c-286c-5266-8e1e-34e56a13d187) red-teams your statement and GENERIC's, verdict about 23:30Z.
- GENERIC proves B1, the bridge from its claims-core inner protocol to your `ProofSystem`/`Extractor`/`Trials`. Your
  `Game` model stays the sampled audit's model.
- FDEVFIX (bc-fb0ad372-d695-5ca8-85c4-264ac2f2f014) is on GENERIC's finding: today's `FdevV` undercharges its loop.
  Check that no `Trials.CR` finder of yours has the same pattern.
Target: the composition, its Flock instance and the ZK side built with no sorry by Thu, then a PR stacked on #6902.

## 21:45Z, proofs coordinator: REDGEN's verdict. Restate before more proving (decision)

Read `~/.research/notes/lanes/proofs/20261103T2145Z-finding-redgen.md` in full. Its blocker:
- **S1:** the extractor, the trials and their costs are free (`Defs.lean:89-91, 117-128, 204-211`), and hm96 is
  statistically hiding. So an unbounded extractor opens every commitment to an honest trace, meets every hypothesis
  with kappa = 0, and makes F empty at every prover. The proved theorem is true but says far less than it reads.

**Decision: one model, the claims core.** Restate the sampled compiler by generalizing `Claims.Sampling` on main
(`Definitions/Claims/Sampling.lean:550, 581-593, 718-722`): its Flock-specific parts become parameters. The extractor
and trials become rerun programs, with costs counted from `RP.eval` and the plurality a definition nobody runs. This
replaces the `FlockSoundness.Game` interface (S1, S4, S5) and my 21:16Z 'keep the Game model'. GENERIC's B1 then maps
claims core to claims core.
Also before proving:
- S2: `LinkFinder.outOf`'s second opening, uncharged. A separate worker, LINKFIX, fixes main's copy; mirror its fix.
- S3: charge the Flock link trials `t' + v_run`.
- S6: zero knowledge with leakage, like `Claims.Sampling.SimLeak`.
- Fix the nits' retired terms.
Keep what survives: the acceptance-probability composition, and the proof skeleton of `Compose`/`Joint`.
**Deadline:** the restated statement elaborating by Wed 06:00Z, posted at the top of your note. I re-staff W-items from
it then. GSJOINT, GSZK and GSFLOCKZK are told to pause.

## 21:54Z, proofs coordinator: every PR names its goal (Daniel, today)

From now on each PR body carries `Goal <id>` naming the Goals-page goal it serves (e.g. `Goal proofs/zero-knowledge`,
`Goal proofs/sampled-audits-sound`, `Goal proofs/recursion-knowledge-sound`, `Goal proofs/sampling-generic`,
`Goal proofs/recursion-generic`, `Goal proofs/terminology-clean`, `Goal proofs/zero-knowledge-any-worker`,
`Goal proofs/zk-leaks-closed`, `Goal proofs/firewall-checks-provable`). Add it to any open PR of yours the next time you
edit its body (a body edit needs no new mark). ci may soon refuse a mark on a PR that names none.

## 21:56Z, proofs coordinator: the ZK form for the restatement (decision)

GSFLOCKZK (note `lanes/proofs/20261103T2128Z-report-gsflockzk.md`, branch `cursor/proofs-gsflockzk-818d` at 62be82e6)
found why S6 bites at Flock. The deployed server's theorem `zk_server_lean_loop` is in leakage form:
Pr_real[E] <= 2 * (Pr_sim[E] + eps), one-sided with factor 2 (the stop bit), and eps reads (x, w, V*) through Pr[Fork].
`Close` (two-sided, eps from the statement only) only fits it with eps >= 1.

**Decision:** state the generic ZK in that leakage form: for every event, Pr_real <= 2^l * (Pr_sim + eps(x, w, V*)), with
l the leaked bits and eps a function of the statement, the witness and the verifier, never an assumed bound (the
break-term ruling). Flock then instantiates it with l = 1 from `zk_server_lean_loop` directly. GSFLOCKZK's `ZKBelow`,
`server_zkBelow` and `audit_below` are that shape already; reuse them.

## 22:00Z, proofs coordinator: hm96 hiding, ready to merge

GSZK's `cursor/proofs-gszk-4b48` at 908467bba4: `FlockHide.lean` as standalone lemmas about hm96's commitment (no
`Hider`/`Close`, std axioms). `hm96_commit_ideal_close`: any trace's commitment is within N*delta1 of the ideal one, for
every [0,1] test. `hm96_commit_two_traces`: two traces within 2*N*delta1. Note: `Hiding (N*delta1)` between two traces
is FALSE as the work list had it (`hm96_commit_two_traces_not_card`); use the ideal form, or pay 2*N*delta1. Merge it into
the restated tree.

## 22:09Z, proofs coordinator: a tries pool for short Lean tries

infra added a `tries` pool (2 slots each, 40 GB gate) on cpu-11, cpu-12, cpu-7 and cpu-3. Run one-file Lean tries with
`verity lean try FILE --on vy-nebius-cpu-11 --pool tries` (or -12, -7, -3); keep records runs off it. And
`verity lean try --on` exiting 1 can be a false failure (a race infra is fixing): check `research status RUN` before
treating it as one.
