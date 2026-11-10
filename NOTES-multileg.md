# The sampled leg over a window of m members: skeleton and pieces

Scratch notes for PRF-490, on `cursor/soundness-multileg-8fb2` only. This file never goes in the PR.

Base: `cursor/soundness-remote-deployed-8fb2` (#8569). Everything is in namespace `Verity.Sampled`, package
`security_proofs`, registered in `Security/Proofs/lakefile.toml`. LEGACC's and LEGCOIN's branches are merged here
(`…-multileg-acc-8fb2` at `e97f23c967`, `…-multileg-coin-8fb2` at `6977821723`).

## Where each part is

| File (under `Security/Proofs/Properties/Verity/Sampled/`) | Holds | Owner | State |
|---|---|---|---|
| `Window.lean` | `Member`, `Window`, `Window.joint` (the `JointAuditV5`), `Window.extend`, the bound (`drawMiss`, `sessionDelta`, `delta`), `JointFinderCR`, `Window.ks`, `WindowConstructions`, `WindowCouplingTarget`, `WindowCovers`, `WindowDeployed`, **`spotSound_of_window`** | LEGLEAD | proved; `lean try` exit 0, standard axioms |
| `Window/Walk.lean` | the projection through m sessions in order: `sessionK` (a session, then what follows it), `WindowCoinMap`, `seqP`/`seqW`/`seqOff`/`SeqReach`, `windowP`, `windowG`, `WindowReaches`, `windowOff`, `WindowCouples`, `WindowDecodes`, `windowLogged`, `windowCouplingTarget_of_logged` | LEGLEAD | no `sorry`; first try left 5 elaboration errors, fixed, second try running |
| `Window/Reach.lean` | each member's session's trace in the window's: `RoundsEnd`, `MemberReach`, `SeqChain`, `SeqReach.chain`, `SeqChain.forall₂`, `MemberDecodes`, **`windowDecodes_of_members`** | LEGLEAD | no `sorry`; try running |
| `Window/Leg.lean` | the composition: `windowSpotSound_of_logged`, **`windowSpotSound_of_members`** | LEGLEAD | no `sorry`; try running |
| `Window/Accepts.lean` | v2's acceptance as m verdicts in order: `StepAt`, `SessionVerdict`, `ranOn`, `Verdicts`, `step_iff`, `steps_iff`, `verdicts_of_steps`, `Verdicts.getElem`, `Verdicts.coins_own`, `run_verdicts_at`, `run_verdicts` | LEGACC | filled by LEGACC |
| `Window/Members.lean` | a stand-in for the per-member form over this tree's `step` (`PerMember.statementOf`, `membersHeld`, `runMembers`, `runMembers_iff`, `SliceHeld`, `Member.Fixed`, `accepts_iff`); not the verifier's, and the gate is not lifted | LEGACC | filled by LEGACC |
| `Window/Draw.lean` | the marginals and the coin map's premise over the window: `Member.mem_part`, `Window.sum_part_card`, `Window.avg_design`, `Window.drawMiss_extend`, `Window.eps_extend`, `Window.sessionDelta_extend`, `seqErr`, `windowErr`, `windowCouples_of_close` | LEGCOIN | 6 scratch `sorry` |
| `Members.lean`, `Members/{Draw,Widen,Landing,Session,Project}.lean`, `Flock/EndToEnd/Joint/V5/Layout.lean` | LEGCOIN's own vocabulary on `JointAuditV5`: `extend`, `eps_extend`, `pa_extend`, `sampled_extend`, the draw's layout (`Laid`, `joint_draw_eq`, `member_draw_eq`), a landing-based projection (`membersL`, `projectMembers`) | LEGCOIN | written before the skeleton |

Scratch `sorry` is only in `Window/Draw.lean`.

## The leg

```lean
theorem windowSpotSound_of_members (rd : WindowReadings) (K : WindowConstructions V)
    (D : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) →
      ℕ → Decoders W.rounds ByteArray)
    (M : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      WindowCoinMap (V.game θ F c) σ W.Ω W.rounds)
    {S : ByteArray → Array ByteArray → Option Window} (hD : WindowDeployed V (K.withProject V rd D M) S)
    (hCov : WindowCovers V k S) (P : Exchange → Prop)
    (Q : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → Exchange → Prop)
    (BL : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → WindowLinkHold)
    (B : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → WindowRoundHold)
    (I : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      (W.extend (M W θ F c σ).widening).L.Ω → ℕ → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → Prop)
    (ha : ∀ θ F c σ, prob (fun o => V.Verdict k .sampled o ∧ ¬ P o) (V.game θ F c) σ ≤
        ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) + ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify))
    (hq : ∀ θ F W, S θ F = some W → ∀ c σ, prob (fun o => V.Verdict k .sampled o ∧ ¬ Q θ F c σ o) (V.game θ F c) σ ≤
        ENNReal.ofReal (Real.sqrt ((K.bindFinder W θ F c σ).Pr sha512Msg)))
    (hC : ∀ θ F W, S θ F = some W → ∀ c σ,
      WindowCouples V θ F c W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ) (M W θ F c σ) (BL θ F c σ) (B θ F c σ))
    (hm : ∀ θ F W, S θ F = some W → ∀ c σ,
      MemberDecodes V θ F c k (fun o => P o ∧ Q θ F c σ o) W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ)
        (M W θ F c σ) (BL θ F c σ) (B θ F c σ) (I W θ F c σ)) :
    V.SpotSound k
```

It is `spotSound_of_window` (the bound) after `windowCouplingTarget_of_logged` (the projection) after
`windowDecodes_of_members` (the walk a member at a time):

```lean
theorem spotSound_of_window {K : WindowConstructions V} {S : ByteArray → Array ByteArray → Option Window}
    (hD : WindowDeployed V K S) (hCov : WindowCovers V k S)
    (hC : ∀ θ F W, S θ F = some W → WindowCouplingTarget V k K W θ F) : V.SpotSound k
```

The bound is `JointShareV5`'s own, applied at `Window.joint`, not proved again:

```lean
theorem Window.ks (W : Window) (t : ℕ) (τ : FlockSoundness.Game.Strategy (W.joint t).game)
    (hF : JointFinderCR (W.joint t) τ) (hI : W.ε * W.nJ < (((W.joint t).wrongSet τ).card : ℝ)) :
    Verity.Game.prob (fun o => (W.joint t).Accepts o) (W.joint t).game τ ≤ W.sessionDelta t
```

`W.sessionDelta t = W.drawMiss + (W.joint t).eps`, and `JointAuditV5.eps = Σ_k (ε_ks,k + ε_bind,k)` at member `k`'s
part of the one sample (`JointAuditV5.design k`). `W.delta t cr` adds one two-continuation term per round of every
member's session (`W.rounds = Σ_m (2·2^roundsLog_m + 1)`), the record's, the bound digests', the hiding term and `cr`.

## Where each piece plugs in

### LEGCOIN: `hC`, and the marginals

`hC : WindowCouples …` is `windowCouples_of_within (windowCouples_of_close …) (hε : err ≤ W.hidingTerm)`
(`Window/Draw.lean`), at the deployed window coin map (`sampleMap`'s analog as a `WindowCoinMap`: one `start` from the
bytes the sample reads; member `i`'s `link`/`round` written into its own openings).

The marginals in `Window/Draw.lean` are LEGCOIN's `Members/Widen.lean` at `W.joint t`:
`(W.extend S).joint t` and `(W.joint t).extend S` are the same structure field by field, so
`Window.eps_extend W S t` should be `(W.joint t).eps_extend S` after a `show`/`rfl` bridge, and `drawMiss_extend`
is `rfl` as `pa_extend` is (neither reads `Ω` or `src`). `WindowDeployed.delta` is stated at the widened window; once
`sessionDelta_extend` is proved it can be restated at `W.delta`.

`Members/{Landing,Session,Project}.lean` project a second time what `Window/Walk.lean` projects (`membersL` beside
`seqP`, `projectMembers` beside `windowP`). One should stay. `Window/Walk.lean`'s has the held-off events `BL` and
`B` at each member's stops, which the deployed coin map needs (the one-session coupling is off `linkEmptyAt` and
`roundEmptyAt`); the landing-based one is off no event. `Members/Draw.lean` and `Members/Widen.lean` have no
counterpart here and stay.

### LEGACC: `hm`, at each member

```lean
structure MemberDecodes (k) (P : Exchange → Prop) (W : Window) (t : ℕ) (rd : Reading W.rounds)
    (σ : Strategy (V.game θ F c)) (D : ℕ → Decoders W.rounds ByteArray)
    (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : WindowLinkHold) (B : WindowRoundHold)
    (I : (W.extend M.widening).L.Ω → ℕ → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → Prop) : Prop where
  start : ∀ ω, I ω 0 (Pos.root (roundCuts V θ F c rd) (M.start ω))
  step : ∀ ω i m p q r out, ((W.extend M.widening).joint t).ms[i]? = some m → I ω i p →
    MemberReach D M BL B ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) i m p q out → q.Le r →
    r.Inv (roundCuts V θ F c rd) →
    V.Verdict k .sampled (run r.tape) ∧ NoMiss V θ F c rd σ r.tape ∧ P (run r.tape) →
    I ω (i + 1) q ∧ ((m.part ((W.extend M.widening).L.draw ω)).Nonempty →
      AcceptsZKP m.I m.own (m.wr (m.part ((W.extend M.widening).L.draw ω)) (windowReg V θ F c W rd σ D) out))
```

`MemberReach … i m p q out` is member `i`'s session's trace from the position `p` to the position `q` it ends at:
link points off `BL i` at `p.toStop`, then `RoundsEnd` (the one-session `RoundsReach` with its end position:
`RoundsEnd.reach`, `RoundsReach.ends`), so every `RoundsReach.*` lemma of `Stops.lean` applies to it as it stands.
`r` is where the last member's session ends; `r.tape` is the tape v2's verdict is read on. `I` is what the
per-member argument needs of the position a member's session starts at (the device's log there shows `i` `Hello`s,
`StopsAt`, the coin-tree keys so far): it holds at the play's start and each member's trace must keep it.

The verdict side is `Verdicts` / `PerMember.runMembers_iff`: member `i`'s `SessionVerdict` is `Verdicts.getElem` at
`i`; with `Record.verdict_of_accepts`-style lemmas at that session it gives `AcceptsZKP` of the record and proof files
v2 read, and `wr_accepts` (one session: `Accepts.lean`'s `wr_accepts_of_verdict`, `MovesOf`) carries it to
`m.wr … out` once `out`'s moves are those v2's files decode to. That last step is the m-session `MovesReach`: the
one-session proofs in `MovesReach.lean`, `Moves.lean`, `Record/*` are stated at `A : Live`, `firstStop` and one tree
(`KeyAt.eq_zero`, `trees.size = 1`), and need the member's tree index `i` and its start `p` in place of the root.
`Walk.lean` (`walk_relK`, `StopOf`, `TreesOf`, `stopOf_next`) and the `RoundsReach.*` half of `Stops.lean` are already
stated at any position and any number of trees.

### LEGLEAD: the walk (`Window/Walk.lean`, `Window/Reach.lean`, `Window/Leg.lean`)

```lean
theorem prob_sessionK … :   -- a session's projection, then what follows it, under its joint law
    prob E (((liveSession dg J mPts N).map wr).bind K) (sessionK D M dg J wr K nextP N p) =
      prob (fun o => E o.1) (sessionWK D M dg J wr nextW N p) w
theorem sessionOffK_le_prob … (h : ∀ o, SessionReachK D M dg J B wr BL nextR N p o → H o → E o) :
    sessionOffK D M J B BL e N p ≤ prob E (sessionWK D M dg J wr nextW N p) w
theorem prob_seqP : ∀ l E i p w, prob E (seqG (fun m => m.live (m.part S) Rg) l) (seqP D M S Rg i l p) =
    prob (fun o => E o.1) (seqW D M S Rg i l p) w
theorem seqOff_le_prob (Q) : ∀ l E i p w, (∀ o, SeqReach D M BL B S Rg i l p o → Q o.2 → E o.1) →
    seqOff D M BL B S Rg Q i l p ≤ prob (fun o => E o.1) (seqW D M S Rg i l p) w
theorem prob_windowP … : FlockSoundness.Game.prob E ((W.extend M.widening).joint t).game (windowP V θ F c W t rd σ D M) =
    prob (fun o => E o.1) (windowG V θ F c W t rd σ D M) w
theorem windowOff_le_prob … (h : ∀ o, WindowReaches V θ F c W t rd σ D M BL B o → Q o.2 → E o.1) (w) :
    windowOff V θ F c W t rd σ D M BL B Q ≤ prob (fun o => E o.1) (windowG V θ F c W t rd σ D M) w
theorem windowLogged … (hC : WindowCouples …) (hD : WindowDecodes … k P …) :
    ENNReal.ofReal (Claims.Pr fun x => V.Verdict k .sampled (run x) ∧ NoMiss V θ F c rd σ x ∧ P (run x)) ≤
      FlockSoundness.Game.prob (fun o => ((W.extend M.widening).joint t).Accepts o)
        ((W.extend M.widening).joint t).game (windowP V θ F c W t rd σ D M) + W.hidingTerm
theorem windowCouplingTarget_of_logged … (ha) (hq) (hC : ∀ c σ, WindowCouples …) (hD : ∀ c σ, WindowDecodes …) :
    WindowCouplingTarget V k (K.withProject V rd D M) W θ F
theorem SeqReach.chain : ∀ l i p o, SeqReach D M BL B S Rg i l p o →
    ∃ r, SeqChain D M BL B S Rg i l p o.1 r ∧ o.2 = r.tape
theorem SeqChain.forall₂ (I : ℕ → Pos … → Prop) (hc : SeqChain D M BL B S Rg i l p ts r) : I i p →
    (∀ j k m p' q out, j = i + k → l[k]? = some m → I j p' → MemberReach D M BL B S Rg j m p' q out → q.Le r →
      I (j + 1) q ∧ Acc m (m.wr (m.part S) Rg out)) → List.Forall₂ Acc l ts
theorem windowDecodes_of_members … (I) (h : MemberDecodes V θ F c k P W t rd σ D M BL B I) :
    WindowDecodes V θ F c k P W t rd σ D M BL B
```

## Open, with no owner yet

1. **A member's proof files before the next member's coins** (decides whether `MemberDecodes.step` can hold).
   `seqG` sends member `i`'s proof files before member `i+1`'s coins. The period's game (`relK`) binds each round's
   message by the digest its `.opening` frame answers, and sends every blob at the end; nothing binds a session's
   final proof files before the next session's `Hello`. `seqP` sends, at member `i`'s session's end `q`, the proof
   files decoded from the play on `q.tape`; later sessions then rewrite later coins, and v2's verdict is on the play
   of `r.tape`. In `MemberDecodes.step` that is `out.proofs = (D i).proofs (run q.tape)` against v2's files in
   `run r.tape`. They agree only where the play up to `q` fixes member `i`'s files. A strategy whose member-`i` files
   depend on member `i+1`'s coins makes the two differ, and no single strategy of `seqG` has its acceptance
   probability (two sessions each valid iff one later coin bit is 0: accepted with 1/2, projected with 1/4).
   Two ways out, neither mine to choose:
   - **(A) bind it**: v2 holds each member's proof files' digest to something the device logs before the next
     member's `Hello` (one more cut and one more two-continuation term per member). Then the decoder reads the files
     at that cut and `step` holds.
   - **(B) a member at a time on the Flock side**: `JointSampledV5` is already a union bound over members, each
     played alone with the earlier sessions' coins in its law (`memberLaw`); allowing the later sessions' coins there
     too would cover delayed files, but that is a new joint theorem over a game that is not `seqG`, against "reuse
     it" and "sequential sessions only".
2. **The claim's transfer** (`WindowDeployed.claim`). `JointAuditV5`'s bounds conclude in `wrongSet` form (the
   members' incorrect proof blocks at V5's own openings). The one-session leg goes on to the deployed commitment
   (`Live.ks`: `AssemblyTargetV5`, `RootFixesDomainL`, `OpensUnder`); Flock has no joint analog of that step, and
   `Vocab.read` reads one circuit file and one public file. `WindowDeployed.claim` states what is needed.
3. **The per-member verifier form is not in this tree.** `MULTI_REFUSED`, `runMembers` and `membersHeld` are on
   `origin/remote` (`verity/core/service/sampled_proofs/SampledProofs/OneStageZk.lean`) only; neither `origin/main`
   nor #8569's branch has them. This tree's gate is `Spot.OneStageZk.oneSession`. The gate can be lifted, and the
   `#guard` added, only once the per-member form reaches the base and the leg above is closed (1 and 2); the verifier
   gains no form ahead of its proof. LEGACC's `Window/Members.lean` is a stand-in over this tree's `step`.
4. **Sequential sessions only.** `SeqReach` walks member `i+1`'s stops after member `i`'s. A play that opens two
   members' coin trees at once is not covered; the readings (`Reading W.rounds`, the rounds in the members' order)
   must refuse or miss it.
5. **`Window.hcls`** stays a field (the registration's refusal of a member whose fewest and most sampled proof blocks
   give different link points): `Flock.…JointAuditV5.hcls_of_bounds` discharges it from bounds.

## Branch name

The cloud instructions name `cursor/verity-soundness-8fb2`; the brief names `cursor/soundness-multileg-8fb2`. This
work is on the brief's.
