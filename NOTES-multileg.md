# The sampled leg over a window of m members: skeleton and pieces

Scratch notes for PRF-490, on `cursor/soundness-multileg-8fb2` only. This file never goes in the PR.

Base: `cursor/soundness-remote-deployed-8fb2` (#8569). Everything is in namespace `Verity.Sampled`, package
`security_proofs`, registered in `Security/Proofs/lakefile.toml`.

## Skeleton: where each part is

| File (under `Security/Proofs/Properties/Verity/Sampled/`) | Holds | Owner |
|---|---|---|
| `Window.lean` | `Member`, `Member.toV5`, `Member.part`, `Window`, `Window.joint` (the `JointAuditV5`), `Window.extend`, `Window.L`, the bound (`drawMiss`, `sessionDelta`, `delta`), `JointFinderCR`, `Window.ks`, `WindowConstructions`, `WindowFinderCR`, `WindowCouplingTarget`, `WindowCovers`, `WindowDeployed`, **`spotSound_of_window`** | LEGLEAD |
| `Window/Walk.lean` | the run of record walking m sessions: `sessionK` (a session, then what follows it), `WindowCoinMap`, `seqP`/`seqW`/`seqOff`/`SeqReach`, `windowP`, `windowG`, `WindowReaches`, `windowOff`, `WindowCouples`, `WindowDecodes`, `windowLogged`, `windowCouplingTarget_of_logged` | LEGLEAD |
| `Window/Accepts.lean` | v2's acceptance as m verdicts in order: `StepAt`, `SessionVerdict`, `ranOn`, `Verdicts`, `verdicts_of_steps`, `run_verdicts` | LEGACC |
| `Window/Draw.lean` | the one draw's per-member parts, the marginals, the coin map's premise over the window: `Member.mem_part`, `Window.sum_part_card`, `Window.avg_design`, `Window.eps_extend`, `Window.sessionDelta_extend`, `seqErr`, `windowErr`, `windowCouples_of_close` | LEGCOIN |

Scratch `sorry` marks each open proof. No definition is `sorry`.

## The leg (what the three pieces compose to)

```lean
theorem spotSound_of_window {K : WindowConstructions V} {S : ByteArray → Array ByteArray → Option Window}
    (hD : WindowDeployed V K S) (hCov : WindowCovers V k S)
    (hC : ∀ θ F W, S θ F = some W → WindowCouplingTarget V k K W θ F) : V.SpotSound k
```

Proved in the skeleton (no `sorry`), from `Window.ks`, which is `Flock.SecurityProofs.JointShareV5` applied at
`Window.joint`, not proved again:

```lean
theorem Window.ks (W : Window) (t : ℕ) (τ : FlockSoundness.Game.Strategy (W.joint t).game)
    (hF : JointFinderCR (W.joint t) τ) (hI : W.ε * W.nJ < (((W.joint t).wrongSet τ).card : ℝ)) :
    Verity.Game.prob (fun o => (W.joint t).Accepts o) (W.joint t).game τ ≤ W.sessionDelta t
```

The bound is the sample's own miss plus the members' terms summed:
`W.sessionDelta t = W.drawMiss + (W.joint t).eps`, and `JointAuditV5.eps = Σ_k (ε_ks,k + ε_bind,k)` at member `k`'s
part of the one sample (`JointAuditV5.design k`). `W.delta t cr` adds one two-continuation term per round of every
member's session (`W.rounds = Σ_m (2·2^roundsLog_m + 1)`), the record's, the bound digests', the hiding term and `cr`.

`WindowCouplingTarget V k K W θ F` is what the walk, the coin map and the acceptance prove together:

```lean
∀ c σ, prob (V.Verdict k .sampled) (V.game θ F c) σ ≤
  prob (fun o => ((K.live W θ F c σ).joint (V.cost θ F c σ)).Accepts o)
      ((K.live W θ F c σ).joint (V.cost θ F c σ)).game (K.project W θ F c σ) +
    (Σ_j √msgFinder_j + √recordFinder + √bindFinder + W.hidingTerm + readFinder + forger)
```

## Piece: LEGACC, v2's acceptance as m verdicts in order (`Window/Accepts.lean`)

```lean
theorem step_all {a : StepAt} {bs} {r r' : Spot.OneStageZk.Run} {s : Spot.OneStageZk.RunFiles}
    (h : a.step bs r s = .ok r') (hall : r'.all = true) : r.all = true

theorem verdict_of_step {a : StepAt} {bs} {r r'} {s} (h : a.step bs r s = .ok r') (hall : r'.all = true) :
    ∃ record spec tabs rest proofs units, SessionVerdict a bs r.coinKeys s record spec tabs rest proofs units ∧
      r' = ranOn a r s record spec tabs rest units

theorem verdicts_of_steps {bs} {l : List (StepAt × Spot.OneStageZk.RunFiles)} {r r' : Spot.OneStageZk.Run}
    (h : l.foldlM (fun r x => x.1.step bs r x.2) r = .ok r') (hall : r'.all = true) : Verdicts bs r l r'

theorem run_verdicts {p : Spot.OneStageZk.Own} {sessions : Array Spot.OneStageZk.RunFiles} {pu : Option (Array ℕ)}
    {bs} {r} (h : Spot.OneStageZk.run p sessions pu bs = .ok r) (hall : r.all = true) :
    ∃ (a : StepAt) (tags0 : Flock.Tags) (circuitB pubB : ByteArray),
      Flock.Tags.find? Spot.OneStageZk.STATEMENT = some tags0 ∧ tags0.plainLeaves = false ∧
      bs.get p.circuit "circuit" = .ok circuitB ∧ bs.get p.pub "public file" = .ok pubB ∧
      a.I = Spot.OneStageZk.inputsOf tags0 p circuitB pubB bs ∧
      Spot.OneStageZk.readsOf p.registered = .ok a.rd ∧ a.scope = Record.scopeOf a.I a.rd ∧
      a.population = Spot.FilePart.check p.program (Spot.getD p.partition "query") pubB ∧
      Verdicts bs ⟨#[], #[], #[], true⟩ (sessions.toList.map fun s => (a, s)) r ∧
      ∀ want, pu = some want → Spot.OneStageZk.provedHeld r.proved want = .ok ()
```

`Verdicts bs r l r'` is the sessions' verdicts in order: each session's `SessionVerdict` (its record a blob as UTF-8,
`setupAt` at the record's own draw, `filesOf`, its coin-tree key fresh among the earlier sessions' keys,
`Flock.Zk.verify`, `verdictOf = .ok ()`, the program's digest, `provedUnits`) after the run before it, and the run
after it (`ranOn`) the run before the next. `StepAt` is per session, so the same statement serves the per-member form
(one `StepAt` per member) once that form is in the tree; `run_verdicts` is this tree's `run`, at one statement.
The existing one-session lemmas to reuse: `Moves.push_ok`, `Moves.verdictOf_ok`, `Moves.step_verified`,
`Moves.run_verified`, `Record.verdict_of_accepts`.

## Piece: LEGCOIN, the coin map and the marginals (`Window/Draw.lean`)

The one draw mapped to the members' parts: member `m`'s part of the window's sample `S` is `m.part S`
(`MemberV5.part`: its proof blocks whose place `m.emb u` is in `S`); the window's projection hands member `i`'s
session exactly that part (`seqP`), and its sample file is `printedPart (m.part S)` (`Member.toV5`'s `dj`).

```lean
theorem Member.mem_part {nJ : ℕ} (m : Member nJ) (S : Finset (Fin nJ)) (u : Fin m.n) : u ∈ m.part S ↔ m.emb u ∈ S

theorem Window.sum_part_card (W : Window) (h : W.Partitions) (S : Finset (Fin W.nJ)) :
    (W.ms.map fun m => (m.part S).card).sum = S.card
```

The marginals lemma (each member's term reads only its part of the one sample, so the bound is the same at the window
widened by the projection's coins; `Live.sessionDelta_extend`'s analog, from `avg_fst` and Flock's `avg_memberLaw`,
`epsKs_memberLaw`, `link_memberLaw`):

```lean
theorem Window.avg_design (W : Window) (t : ℕ) (k : Fin (W.joint t).ms.length)
    (G : Finset (Fin ((W.joint t).ms.get k).n) → ℝ≥0∞) :
    avg (fun ω : ((W.joint t).design k).Ω => G (((W.joint t).design k).draw ω)) =
      avg fun ω : W.L.Ω => G (((W.joint t).ms.get k).part (W.L.draw ω))

theorem Window.drawMiss_extend (W : Window) (S : Widening) : (W.extend S).drawMiss = W.drawMiss
theorem Window.eps_extend (W : Window) (S : Widening) (t : ℕ) : ((W.extend S).joint t).eps = (W.joint t).eps
theorem Window.sessionDelta_extend (W : Window) (S : Widening) (t : ℕ) :   -- proved from the two above
    (W.extend S).sessionDelta t = W.sessionDelta t
```

The coin map over the window (`CoinMap.lean`'s `couples_of_close`, with the continuation; `seqErr` sums each member's
session's `roundsErr` along the walk):

```lean
theorem windowCouples_of_close (W : Window) (t : ℕ) (rd : Reading W.rounds) (σ : Strategy (V.game θ F c))
    (D : ℕ → Decoders W.rounds ByteArray) (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds)
    {γ : ℕ → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → (j : ℕ) → SubmitJ j → ℝ≥0∞}
    {γL : ℕ → (mPts : ℕ) → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℝ≥0∞} {ε₀ : ℝ≥0∞}
    (BL : WindowLinkHold) (B : WindowRoundHold) (hR : ∀ i mPts, (M.member i mPts).RoundClose (γ i))
    (hL : ∀ i mPts, (M.member i mPts).LinkClose (γL i mPts)) (hF : WindowFirstStopFresh V θ F c W rd σ M)
    (hS : WindowStartWithin V θ F c W σ M ε₀) :
    WindowCouplesWithin V θ F c W t rd σ D M BL B (ε₀ + windowErr V θ F c W t rd σ D M
      (fun i q j s => γ i q j s + prCoin (B i q.g' j s)) (fun i mPts q => γL i mPts q + prCoin (BL i mPts q.g')))
```

Then the deployed window coin map: `sampleMap`'s analog as a `WindowCoinMap` (one `start` from the bytes the sample
reads; member `i`'s `link`/`round` written into its own openings), with `WindowFirstStopFresh` and
`WindowStartWithin … 0`, as `sampleMap_fresh` and `sampleMap_start` give at one session. If the induction wants the
error's recursion in another shape, change `seqErr`/`windowErr` and say so here: `WindowCouplesWithin`'s conclusion
and the per-member summing are what the leg reads.

## Piece: LEGLEAD, the walk through m sessions (`Window/Walk.lean`)

```lean
theorem prob_sessionK … :   -- a session's projection, then what follows it, under its joint law
    prob E (((liveSession dg J mPts N).map wr).bind K) (sessionK D M dg J wr K nextP N p) =
      prob (fun o => E o.1) (sessionWK D M dg J wr nextW N p) w

theorem prob_windowP (W) (t) (rd) (σ) (D) (M) (E : W.Out → Prop) (w : Strategy (windowG V θ F c W t rd σ D M)) :
    FlockSoundness.Game.prob E ((W.extend M.widening).joint t).game (windowP V θ F c W t rd σ D M) =
      prob (fun o => E o.1) (windowG V θ F c W t rd σ D M) w

theorem windowOff_le_prob … (h : ∀ o, WindowReaches V θ F c W t rd σ D M BL B o → Q o.2 → E o.1) (w) :
    windowOff V θ F c W t rd σ D M BL B Q ≤ prob (fun o => E o.1) (windowG V θ F c W t rd σ D M) w

theorem windowLogged … (hC : WindowCouples …) (hD : WindowDecodes … k P …) :
    ENNReal.ofReal (Claims.Pr fun x => V.Verdict k .sampled (run x) ∧ NoMiss V θ F c rd σ x ∧ P (run x)) ≤
      FlockSoundness.Game.prob (fun o => ((W.extend M.widening).joint t).Accepts o)
        ((W.extend M.widening).joint t).game (windowP V θ F c W t rd σ D M) + W.hidingTerm

theorem windowCouplingTarget_of_logged … (ha) (hq) (hC : ∀ c σ, WindowCouples …) (hD : ∀ c σ, WindowDecodes …) :
    WindowCouplingTarget V k (K.withProject V rd D M) W θ F
```

`WindowDecodes` is where LEGACC's verdicts and the walk meet: on the joint law's support (`WindowReaches`), an accepted
play that misses nowhere is beside an outcome with `AcceptsZKP` at every member with a proof block in the sample.

## Open, with no owner yet

1. **The claim's transfer** (`WindowDeployed.claim`). `JointAuditV5`'s proved bounds conclude in `wrongSet` form (the
   members' incorrect proof blocks at V5's own openings). The one-session leg goes on to the deployed commitment
   (`Live.ks`: `AssemblyTargetV5`, `RootFixesDomainL`, `OpensUnder`); Flock has no joint analog of that step, and
   `Vocab.read` reads one circuit file and one public file. `WindowDeployed.claim` states what is needed: where the
   claim fails at a strategy, more than an `ε` share of the population is in `(joint).wrongSet` at the projected
   strategy. It needs a Flock-side lemma (each member's `wrongSet` against its deployed commitment) and a per-member
   reading in the vocabulary.
2. **The per-member verifier form is not in this tree.** `MULTI_REFUSED`, `runMembers` and `membersHeld` are on
   `origin/remote` (`verity/core/service/sampled_proofs/SampledProofs/OneStageZk.lean`) only; neither `origin/main`
   nor #8569's branch has them. This tree's gate is `Spot.OneStageZk.oneSession`, and its v2 runs every session at one
   circuit and one public file, where a member's place in the population (`emb`) isn't fixed by a statement of its
   own. The gate can be lifted, and the `#guard` added, only once the per-member form reaches the base.
3. **A session's proof files before the next session's coins.** `seqG` sends member `i`'s proof files before member
   `i+1`'s coins; the period's game sends every blob at the end. `seqP` sends, at a session's end, the proof files
   decoded from the play's output on the tape as it stands there (later sessions rewrite later coins). `WindowDecodes`
   must then hold with those files: either the member's decoder builds them from the session's own submissions (fixed
   at its stops), or each session's files get a cut of their own (one more two-continuation term per member).
4. **Sequential sessions only.** `SeqReach` walks member `i+1`'s stops after member `i`'s. A play that opens two
   members' coin trees at once is not covered; the readings (`Reading W.rounds`, the rounds in the members' order)
   must refuse or miss it.

## Branch name

The cloud instructions name `cursor/verity-soundness-8fb2`; the brief names `cursor/soundness-multileg-8fb2`. This
work is on the brief's.
