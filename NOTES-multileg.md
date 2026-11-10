# The sampled leg over a window of m members

Scratch notes for PRF-490, on `cursor/soundness-multileg-8fb2` only. This file never goes in the PR.

Base: `cursor/soundness-remote-deployed-8fb2` (#8569, `remote_soundness_deployedReady`, the one-member form). Everything
is in namespace `Verity.Sampled`, package `security_proofs`, under
`Security/Proofs/Properties/Verity/Sampled/`, registered in `Security/Proofs/lakefile.toml` and reached from the lock's
roots (`Window.Assembled`, `Window.Members`, `Members`). One worker (LEG) since the restart; the three earlier
branches are merged here.

## The statement

`Window/Assembled.lean`:

```lean
theorem windowSpotSound_assembled (e : 𝔽 ≃ (Fin 128 → Bool)) (rd : WindowReadings) (K : WindowConstructions V)
    (dev : Exchange → Properties.Challenger.History)
    (D : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      WindowDecoders V θ F c W σ)
    (label : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → String) (nc : ℕ)
    {S : ByteArray → Array ByteArray → Option Window}
    (hD : WindowDeployed V ((K.withBinding dev k).withProject V rd D
      (windowSampleMaps e V rd label (afterAt V k) fun W => W.srcCoinsAt nc)) S)
    (hCov : WindowCovers V k S)
    (I : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      (W.extend (windowSampleMaps e V rd label (afterAt V k) (fun W => W.srcCoinsAt nc) W θ F c σ).widening).L.Ω → ℕ →
        Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → Prop)
    (hP : ∀ θ F W, S θ F = some W → ∀ o : Exchange, o.params = θ → Lines.PrintsLog (dev o) o.log)
    (hS : ∀ θ F W, S θ F = some W → ∀ c, Binding.SignsPlay V dev θ F c)
    (hL : ∀ θ F W, S θ F = some W → W.Lb 0 = nc * 64)
    (hR : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) j pre fr, (rd W θ F c).round j pre fr →
      ∃ i s n, fr = .opening i s n)
    (hm : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      MemberSteps V θ F c k
        (fun o => (Binding.Logged Binding.readsV2 o ∧ TablesSent o.sent) ∧
          SampledDigest k o.sent (afterAt V k θ F c σ))
        W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ)
        (windowSampleMaps e V rd label (afterAt V k) (fun W => W.srcCoinsAt nc) W θ F c σ)
        (fun _ _ => linkEmptyAt e (finish (V.signer θ) θ F c σ.1))
        (fun _ => roundEmptyAt e (finish (V.signer θ) θ F c σ.1)) (I W θ F c σ)) :
    V.SpotSound k
```

**Status of this form.** At `388586f624` the statement carries one more hypothesis, `hε` (over every member's stops
the openings' gaps and empty classes are at most the hiding term), and is off `linkEmpty`/`roundEmpty`; that form
is built and probed (the Runs table). The form above, without `hε` and off the count-aware events, is committed from
`a8013ac42d` (`Window/Charge.lean`) and is **not yet built**: its try and its records run are queued (Runs). Until
they pass, the proved statement is the one with `hε`.

`V.SpotSound k` is the same target the one-member leg meets (`Vocab.SpotSound`): at every parameters, files,
commitment and strategy, the plays v2's sampled verdict accepts while the claim fails are at most `V.error`, which is
conditional on the vocabulary's finder premises. So the conclusion plugs into `Verity.Soundness` where the one-member
`spotSound` does.

How it is put together, each layer proved once:

| Layer | Theorem | File |
|---|---|---|
| the bound: `JointShareV5`'s, applied at `Window.joint`, not proved again | `Window.ks`, `spotSound_of_window` | `Window.lean` |
| the projection through m sessions in order, under its joint law | `prob_seqP`, `seqOff_le_prob`, `windowLogged`, `windowCouplingTarget_of_logged` | `Window/Walk.lean` |
| the run-of-record walk a member at a time | `SeqReach.chain`, `SeqChain.forall₂`, `windowDecodes_of_members` | `Window/Reach.lean` |
| what a member's trace keeps and logs (`StopsAt`, the play's start, the released coins) | `MemberReach.stopsAt`, `MemberReach.logged`, `MemberSteps`, `memberDecodes_of_stops` | `Window/Logged.lean` |
| the draw to member subsets and its marginals; the coin map's premise over the window | `Member.mem_part`, `Window.sum_part_card`, `Window.avg_design`, `Window.eps_extend`, `windowCouples_of_close`, `windowSampleMap_fresh`, `windowSampleMap_start` | `Window/Draw.lean` |
| the stops' term within the hiding term at every strategy | `seqErr_le`, `windowErr_le_of_charge`, `windowErr_sampleMap_le_at`, `windowCouplingTarget_of_sample_at` | `Window/Charge.lean` |
| the composition | `windowSpotSound_of_sample` | `Window/Leg.lean` |
| (a), (a') and the sample's coins | `windowSpotSound_assembled` | `Window/Assembled.lean` |
| v2's acceptance as m verdicts in order | `Verdicts`, `run_verdicts_at`, `Verdicts.getElem` | `Window/Accepts.lean` |

The bound it reaches is `spotSound_of_window`'s: `W.sessionDelta t = W.drawMiss + (W.joint t).eps` (the one sample's
own miss, and `Σ_k (ε_ks,k + ε_bind,k)` at member `k`'s part of the one sample), one two-continuation term per round of
every member's session (`W.rounds = Σ_m (2·2^roundsLog_m + 1)`), the record's, the bound digests', the hiding term, the
read finder's collision and the forger's forgery.

## What each premise is

Discharged inside the statement, with what each rests on:

| What | By | Rests on |
|---|---|---|
| the sample's coins are uniform | `Window.avg_srcCoinsAt`, from `Window.hA3` | uniform randomness (A3 at the device's bytes) |
| (a) the log v2 read is the device's and the pinned tables are the registered files | `Binding.logged_tables_le`, `Binding.binds_of_prints` | SHA512CR (the read finder), LmsUnforgeable (the forger; remote only), and `hP`, `hS` below |
| (a') v2 samples at the strategy's `afterAt` | `prob_not_sampledDigest_le` at `windowBindFinder` | SHA512CR (a two-continuation finder) |
| the coin map's premise at the sample-coupled map | `windowSampleMap_couples_at` | nothing further |
| the stops' term within the hiding term | `windowErr_sampleMap_le_at` (`avg_pathCharge_game_le`: the class errors along a play of the period's game are within the potential from no tree open, however many sessions it runs) | nothing further |
| each member's trace keeps the play's start and `StopsAt` | `MemberReach.inv`, `MemberReach.stopsAt` | `hR` below |
| the bound | `Window.ks` (`JointShareV5`) | SHA512CR per finder (`JointFinderCR`), in `WindowDeployed.finderCR` |

Left as hypotheses:

| Premise | What it is | Class |
|---|---|---|
| `hP` | the device printed each play's log (`Lines.PrintsLog`) | honest verifier server's log premise (the one-member leg takes the same) |
| `hS` | the device's history at each play signed what the play's signatures are on (`Binding.SignsPlay`) | honest verifier server's log premise (`Device.signsPlay_dev` at the deployed device) |
| `hL` | the window reads `nc` frames' bytes | a fact of the deployment's windows `S` (the one-member `hL`) |
| `hR` | each round's reading picks only openings | a fact of the readings `rd`; holds at the layout's readings once they are built for a window (open item 2) |
| `hD : WindowDeployed` | the vocabulary reads these constructions: finder, forger, no loss, `finderCR` (the per-finder SHA-512 premises follow from `V.finderCR` and `V.premises`), `delta` (the published `δ` is at least the window's), and `claim` | `finder`, `forger`, `loss`, `finderCR`, `delta` are the instance's bookkeeping (`Instance.lean` does the one-member ones); **`claim` is open item 3** |
| `hCov : WindowCovers` | where `S` holds no window, the one-stage form is covered at the published `δ` | the instance's: the one-member leg at the windows of one member (`spotSound` of `Sampled/Leg.lean`) |
| `hm : MemberSteps` | each member's step: **open items 1 and 2** | not a trust premise; to be proved |

Trust base when the open items close: SHA512CR, uniform randomness, the honest verifier server's log premises;
LmsUnforgeable remote only. Nothing else is introduced.

## What is left

1. **A member's proof files against the next member's coins** (a ruling, asked on PRF-490 at 15:41Z, unanswered).
   `JointAuditV5.sessions` (`seqG`) sends member `i`'s proof files before member `i+1`'s coins. The period's game sends
   every blob with the evidence, after every coin. Decoding member `i`'s files from the play on the tape its walk ends
   on is wrong at some strategy (two members whose files are both valid iff one coin bit of member 2's session is 0:
   accepted with 1/2, projected with 1/4), so the earlier `MemberDecodes.step` could not be met as stated.
   What the branch does now: a member's decoders (`PosDecoders`) give its proof files as a function of the position
   its walk ends at (`files`), so the statement no longer fixes where they come from. `Decoders.atEnd` is the old
   reading. `MemberSteps.step` can hold only at a `files` that the play fixes by that position, which needs one of:
   - **(A) bind it**: v2's per-member form holds statement `i+1` (its `Hello` nonce, or its first opening) to SHA-512
     of statement `i`'s evidence entry. Then `files` reads the cut's most likely message at the end position
     (`q.toStop.modes`), with one more two-continuation term per member. A cut at a `Hello` that is not an opening
     also needs a handoff step between members in `seqP`, not written.
   - **(B) a member at a time**: couple each member's session alone, every other tree's coins in its law, then
     `MemberV5.accepted_drawnWrong_le` and a union. No protocol change; it drops the `seqG` walk and is not
     `JointAuditV5`'s game, so the bound would not be `JointShareV5`'s as it stands.
2. **`hm`, each member's step** (`MemberSteps.step`): from member `i`'s trace (`MemberReach`), the invariant, and a
   tape past its end whose play v2 accepts at the cut, conclude `AcceptsZKP m.I m.own (m.wr … out)`.
   Already at hand for it: `MemberReach.logged` (the outcome is the root at the first stop, the link points, the
   answered rounds in order and `files`; the play releases their bits at the stops' openings; its `hML`, `hBL`, `hM`,
   `hB` at the sample maps are `rfl`/`id`, at the count-aware events they need the count match), and v2's side as m
   verdicts (`run_verdicts_at`, `Verdicts.getElem`: member `i`'s `SessionVerdict`).
   Missing: the one-session record and moves chain at tree `i` and start `p`. `MovesReach*`, `Moves`, `Record/*`,
   `Stops.firstStop_*`, `readingAt`/`decodersAt`, `accepts_sent`, `wr_accepts_of_verdict`, `CountsReach` are stated
   at `A : Live`, `ParamsAt`, one record, `firstStop` and one tree (`trees.size = 1`); each needs the member's tree
   index and its start position in place of the root. `Walk.lean` (`walk_relK`, `StopOf`, `TreesOf`) and the
   `RoundsReach.*` half of `Stops.lean` are already stated at any position and any number of trees
   (`RoundsEnd.reach`, `RoundsReach.ends` bridge to them). This is the largest piece.
3. **The claim's transfer** (`WindowDeployed.claim`): `JointAuditV5`'s bound concludes in `wrongSet` form (the
   members' incorrect proof blocks at V5's own openings). The one-member leg goes on to the deployed commitment
   (`Live.ks`: `AssemblyTargetV5`, `RootFixesDomainL`, `OpensUnder`); Flock has no joint analog, and `Vocab.read` reads
   one circuit file and one public file.
4. **The window instance**: `WindowDeployed`'s other fields and `WindowCovers` at the deployed vocabulary, and a
   theorem in the shape of `remote_soundness_deployedReady` with `windowSpotSound_assembled` as its sampled leg.
5. **The verifier's gate (done-when 2) is not liftable on this base.** `MULTI_REFUSED`, `runMembers` and
   `membersHeld` are in the staged merge (`875cc642dc`, `sampled_proofs/SampledProofs/OneStageZk.lean`, the gate near
   line 971) and on `origin/remote`; this base's gate is `Spot.OneStageZk.oneSession`. After the restack: lift the
   gate for the per-member form only, once items 1 to 4 close (the verifier gains no form ahead of its proof), keep
   every other several-member form refused, and rewrite `Window/Members.lean` (a stand-in over this tree's `step`:
   `PerMember.statementOf`, `membersHeld`, `runMembers`, `runMembers_iff`, `accepts_iff`) against the verifier's own
   `runMembers`. On this base the Python reference
   (`verity/core/protocols/composition/tests/reference.py`) mirrors `oneSession`; it and its difftest change with
   the gate.
6. **Sequential sessions only** (the coordinator's ruling). `SeqReach` walks member `i+1`'s stops after member
   `i`'s; the readings must miss a play that opens two members' coin trees at once.
7. **`Window.hcls`** stays a field: `JointAuditV5.hcls_of_bounds` discharges it from bounds.

The forms off the empty classes themselves (`windowOpenMap_couples`, `windowSampleMap_couples`,
`windowCouplingTarget_of_sample`), which needed a bound on the stops' term, are deleted: the count-aware forms
replace them.

## Runs

`verity lean try FILE --on auto`, each `lake build +MODULE:deps` then `lake env lean FILE` with the axiom probe.

| Run | File | Tree | rc | Probe |
|---|---|---|---|---|
| `r20261110-152439-446b` | `Window/Leg.lean` | `8078441923` | 0 | standard axioms |
| `r20261110-153404-936b` | `Window/Logged.lean` | `8078441923` | 0 | standard axioms |
| `r20261110-153404-ef4c` | `Window/Draw.lean` | `8078441923` | 0 | standard axioms |
| `r20261110-154050-b5dd` | `Window/Members.lean` | `46dfb5e5d3` | 0 | 28 theorems, standard axioms |
| `r20261110-155915-3c61` | `Window/Draw.lean` | `PosDecoders` in the working tree | 0 | see the run |
| `r20261110-155916-d15d` | `Window/Leg.lean` | `PosDecoders` and the first `windowSpotSound_of_sample` | 0 | 3 theorems, standard axioms |
| `r20261110-162404-7405` | `Window/Logged.lean` | `388586f624` | 0 | 6 theorems, standard axioms |
| `r20261110-162404-e1be` | `Window/Leg.lean` | `388586f624` | 0 | 3 theorems, standard axioms |
| `r20261110-162404-5413` | `Window/Assembled.lean` | `388586f624` (the form with the stops' term as a premise) | 0 | 3 theorems, standard axioms |

## Branch name

The cloud instructions name `cursor/verity-soundness-8fb2`; the brief names `cursor/soundness-multileg-8fb2`. This
work is on the brief's.
