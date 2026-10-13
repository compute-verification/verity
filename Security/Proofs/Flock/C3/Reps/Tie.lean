import Proofs.Flock.C3.Compose.Decoder

/-!
# The reps' tie: each rep's steps are a rep unit V*'s layers check

`RepTie e u i`: rep `i`'s steps in the run `u` (`u.ts i`) are an accepting run of the executable's holo rep
(`RepUnit`) at the run's statement, schedule, extra claims at its points, level-0 cap and openings: the values, proof,
record and claims V*'s rep-`i` units check on the layers. `SrcRepTie` is n7's tie for the reps beside `SrcOK`: the
source's runs are tied so at every accepted outer proof.

* `reps_prefix`, `repsReadAt_of_tie`: past the cap's and the points' steps, `runSteps u` begins with rep 0's steps in
  the reps' interleaving and then rep 1's, so the two ties are the reps' read fact on the run's own steps
  (`RepsReadAt`). `repsRead_of_tie` and `hReps_of_tie` are `RepsRead` and `HReps` at the fixed decoder.
* `cutOf_append`, `c3Msgs_repPart`, `msgsRead_repPart`: a round that closes in the reps' part of the steps (the cap's
  and the points' steps, then the reps': `repPart`) has the messages of that part. So `MsgsReadC3` at those rounds,
  which are committed ones, is a fact of the reps' parts alone (`RepsMsgsRead`).
* `srcRepTie_of_alg`: the tie from G's items 1–4 at a rep decoder of V*'s layers (`AlgebraDecodes`, `AccChains`, which
  give a unit's `alg` and `acc`) and `RepUnitOfAlg`, the rest of the unit at those values, which is not proved here.
  `repUnitOfAlg_of_parts` assembles it (`unitOfParts`) from four stubs at values `vals` of the layers, each sized in
  its docstring: the proof's part (`ProofOfVals`), the setup's and schedule's (`SetupOfVals`), the record's frames
  (`FramesOfVals`, beside c1) and the source's rep steps (`StepsOfVals`).
-/

namespace FlockC3.Reps

noncomputable section

/-! ## A rep unit's parts -/

section parts

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

open Flock FlockSoundness FlockSoundness.Refine FlockC3.HoloValues
open Flock.RecAlgebra (AnswersL ZcProofOf LcProofOf decoded valOf verifierRows hOfW)
open FlockVBridge.Compiled (RepData AlgebraHolds AccHolds)
open FlockVBridge.Runs (rhoOf tsOf)
open FlockSoundness.Discharge.Exec (kindOf)
open FlockSoundness.Discharge.PlaysCoin (coinArr)

/-- A rep unit's values past its shape and its algebra's values. -/
structure UnitVals where
  st : Flock.Setup
  msId : String
  sch : Flock.Schedule
  p : Flock.Proof
  extras : Array Flock.Claim
  t : Flock.Transcript
  claims : Array Flock.Claim
  rpp : ℕ → Array Flock.F128
  gamma : Array Flock.F128
  t₄ : Flock.Transcript

/-- **The proof's part** of a rep unit at values `d` and `v`: the record's coins are `d`'s, the proof's zerocheck and
lincheck messages and the circuit fold are `verifierRows`', and the ring switches answer the claims. -/
structure UnitProof (sh : Flock.Shape) (d : RepData) (v : UnitVals) : Prop where
  coins : d.rounds = v.t.rounds.map (·.coins)
  zp : ∀ w info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (w, info, kappa) →
    ZcProofOf (decoded sh d.w d.comb d.cons) (info.zc.vc.eval (valOf sh d.w d.comb d.cons)) v.p.zerocheck
  lp : LcProofOf (decoded sh d.w d.comb d.cons) v.p.lincheck
  foldPts : ∀ w info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (w, info, kappa) →
    v.st.fold.folded info.lc.fold.alpha info.lc.fold.beta info.lc.fold.lw info.lc.fold.rhoIn info.lc.fold.ts =
      .ok d.comb
  rs : v.p.ringSwitches.size = v.claims.size
  rsShape : ∀ k < v.claims.size,
    v.p.ringSwitches[k]!.size = 128 ∧ v.claims[k]!.skip.size = 64 ∧ v.claims[k]!.x.size = v.st.m - 6
  rsVal : ∀ k < v.claims.size, dot (wOf v.claims[k]!) v.p.ringSwitches[k]! = v.claims[k]!.value

/-- **The setup's and schedule's part**: the setup's `m`, `k_log` and circuit fold are the shape's and `S`'s, its
schedule is `fast100`'s and the model's `sch₀`, and its extra claims are `extra` claim by claim. -/
structure UnitSetup (S : Model.Statement) (sch₀ : Accounting.Schedule) (extra : List (Model.Claim 𝔽 S.m))
    (sh : Flock.Shape) (v : UnitVals) : Prop where
  m : v.st.m = sh.m
  kL : v.st.fold.kLog = sh.kLog
  mS : v.st.m = S.m
  kS : v.st.kLog = S.kLog
  fold : FoldRealizes v.st.fold S
  fast : Flock.fast100 v.st.m = some v.sch
  sched : schedOf v.sch = sch₀
  merkle : v.st.merkle = Flock.MerkleScheme.sha512 v.msId
  extrasLen : v.extras.size = extra.length
  extrasRel : ∀ (j : ℕ) (hj : j < extra.length), ClaimRel S.m v.extras[j]! extra[j]

/-- **The record's frames**: a record from round 0 that answers the rep's frames (the lincheck's with hidden units),
whose claims are the zerocheck's, the lincheck's and the extras, through ring switching and Ligerito to `t₄`. -/
structure UnitFrames (sh : Flock.Shape) (d : RepData) (v : UnitVals) : Prop where
  k0 : v.t.k = 0
  wf : RoundsWF v.t.rounds
  n₀ : (v.t.rounds[v.t.k]!).n = 6
  n₁ : (v.t.rounds[v.t.k + 1]!).n = v.st.m - 13
  n₂ : (v.t.rounds[v.t.k + 2]!).n = 1
  nr : ∀ i < v.st.m - 6, (v.t.rounds[v.t.k + 3 + i]!).n = 1
  a₀ : AnswersL (v.t.rounds[v.t.k]!) (v.t.pending.data.toList ++ encs [.label "flock-r1cs-v0".toUTF8,
    .bytes v.st.digest, .bytes (v.p.cap.foldl (· ++ ·) ByteArray.empty), .label "flock-zerocheck-v0".toUTF8, .sq 2 6])
  a₁ : AnswersL (v.t.rounds[v.t.k + 1]!) (encs [.sq 2 (v.st.m - 13)])
  a₂ : AnswersL (v.t.rounds[v.t.k + 2]!)
    (encs [.vec v.p.zerocheck.round1ab.toList, .vec v.p.zerocheck.round1c.toList, .sq 1 1])
  ar : ∀ i < v.st.m - 6, AnswersL (v.t.rounds[v.t.k + 3 + i]!)
    (encs [.one (v.p.zerocheck.rounds[i]!).1, .one (v.p.zerocheck.rounds[i]!).2, .sq 1 1])
  lc : HoloLcAnswered ⟨v.t.rounds, v.t.k + 3 + (v.st.m - 6),
    ⟨(encs [.one v.p.zerocheck.finalA, .one v.p.zerocheck.finalB]).toArray⟩⟩ v.st.fold.kLog v.p.lincheck (hOfW sh d.w)
  cl : v.claims = Flock.abcClaims v.st.m v.st.kLog
    (zcOf v.p.zerocheck (pinned ++ (v.t.rounds[v.t.k + 1]!).coins) ((v.t.rounds[v.t.k + 2]!).coins.getD 0 0)
      (rhoOf v.t v.st.m))
    (lcOf v.p.lincheck ((v.t.rounds[v.t.k + 3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6) + 1]!).coins.getD 0 0)
      (tsOf v.t (v.t.k + 3 + (v.st.m - 6)) v.st.fold.kLog)) ++ v.extras
  rsRounds : RsRounds ⟨v.t.rounds, v.t.k + 3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6) + 2, ByteArray.empty⟩
    v.p.ringSwitches v.claims.size v.rpp v.gamma
  lg : Flock.ligerito v.st.merkle v.sch (rsT v.claims v.p.ringSwitches v.rpp v.gamma) v.p.cap v.p.ligerito
    (Flock.basis (rsSws v.claims v.rpp v.gamma))
    ⟨v.t.rounds, v.t.k + 3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6) + 2 + v.claims.size + 1, ByteArray.empty⟩ =
      .ok ((), v.t₄)

/-- **The steps' part**: the steps take the kinds of `holoRepC`'s shape, their messages are the proof's
(`holoRepMsgs`) and their coin steps the record's rounds the program consumes, the proof's level-0 cap is `cap₀` and
the openings at rep `r` are the proof's rows (`openRow`). -/
structure UnitSteps (S : Model.Statement) (sch₀ : Accounting.Schedule) (extra : List (Model.Claim 𝔽 S.m))
    (cap₀ : Model.Cap Dig) (os : Model.Opens 𝔽 Dig) (r : ℕ) (ts : List Game.Step) (sh : Flock.Shape) (d : RepData)
    (v : UnitVals) : Prop where
  kinds : ts.map kindOf <+: holoRepShape 𝔽 𝕂 Dig S.m S.kLog extra.length sch₀
  msgs : ts.filterMap Sum.getLeft? = holoRepMsgs v.st v.sch (2 + extra.length) v.p (hOfW sh d.w) capDig
  coinSteps : ts.filterMap Sum.getRight? =
    (consumed v.t.rounds v.t.k (v.t.k + 3 + (v.st.m - 6) + 2 + (S.kLog - 6)) ++
      consumed v.t.rounds (v.t.k + 3 + (v.st.m - 6) + 2 + (S.kLog - 6) + 1) v.t₄.k).map (coinArr fOf)
  cap : capDig v.p.cap = cap₀
  opens : ∀ ℓ j, os r ℓ j = openRow v.p.ligerito v.sch ℓ j

/-- **A rep unit from its parts** at values `d`, `v` meeting E's preconditions. -/
def unitOfParts {S : Model.Statement} {sch₀ : Accounting.Schedule} {extra : List (Model.Claim 𝔽 S.m)}
    {cap₀ : Model.Cap Dig} {os : Model.Opens 𝔽 Dig} {r : ℕ} {ts : List Game.Step} {sh : Flock.Shape} {d : RepData}
    (v : UnitVals) (alg : AlgebraHolds sh d) (acc : AccHolds sh d) (hp : UnitProof sh d v)
    (hs : UnitSetup S sch₀ extra sh v) (hf : UnitFrames sh d v) (hst : UnitSteps S sch₀ extra cap₀ os r ts sh d v) :
    RepUnit S sch₀ extra cap₀ os r ts :=
  { sh := sh, d := d, st := v.st, msId := v.msId, sch := v.sch, p := v.p, extras := v.extras, t := v.t,
    claims := v.claims, rpp := v.rpp, gamma := v.gamma, t₄ := v.t₄, alg := alg, acc := acc,
    coins := hp.coins, zp := hp.zp, lp := hp.lp, foldPts := hp.foldPts,
    m := hs.m, kL := hs.kL, mS := hs.mS, kS := hs.kS, fold := hs.fold, fast := hs.fast, sched := hs.sched,
    merkle := hs.merkle, extrasLen := hs.extrasLen, extrasRel := hs.extrasRel,
    k0 := hf.k0, wf := hf.wf, n₀ := hf.n₀, n₁ := hf.n₁, n₂ := hf.n₂, nr := hf.nr,
    a₀ := hf.a₀, a₁ := hf.a₁, a₂ := hf.a₂, ar := hf.ar, lc := hf.lc, cl := hf.cl,
    rs := hp.rs, rsShape := hp.rsShape, rsVal := hp.rsVal, rsRounds := hf.rsRounds, lg := hf.lg,
    kinds := hst.kinds, msgs := hst.msgs, coinSteps := hst.coinSteps, cap := hst.cap, opens := hst.opens }

end parts

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.Family (skOf maskOf)
open FlockVBridge.InnerAccepts (firstSteps)
open FlockC3.Compose

/-! ## The tie and the read fact -/

/-- **Rep `i`'s steps are a rep unit**: an accepting run of the executable's holo rep at the run's statement, schedule,
extra claims at the run's points, level-0 cap and openings, on rep `i`'s steps. -/
def RepTie (e : ExecHolo) (u : C3Run e) (i : Fin 2) : Prop :=
  Nonempty (FlockC3.HoloValues.RepUnit (e.cls.pubStatement e.regs) e.sch
    (extraClaims (e.cls.pubStatement e.regs) e.cls.ptLocal e.cls.mPts u.pts) u.cap₀ u.os.1 i (u.ts i))

/-- **`runSteps`' reps' prefix**: past the cap's and the points' steps, rep 0's steps in the reps' interleaving, then
rep 1's. -/
theorem reps_prefix {e : ExecHolo} (u : C3Run e) : firstSteps (u.ts 0) ++ u.ts 1 <+: (runSteps u).drop 2 :=
  List.prefix_append_of_prefix (List.prefix_append_of_prefix (List.prefix_append_of_prefix
    (List.prefix_append_of_prefix (List.prefix_append _ _))))

/-- **The reps' read fact on the run's own steps, from the two ties.** -/
theorem repsReadAt_of_tie {e : ExecHolo} {u : C3Run e} (h₀ : RepTie e u 0) (h₁ : RepTie e u 1) :
    FlockC3.HoloValues.RepsReadAt e (runSteps u) u :=
  ⟨u.ts 0, u.ts 1, reps_prefix u, h₀, h₁⟩

/-! ## The messages of the reps' part -/

/-- A list of steps' messages cut at its coins (`PlaysCoin.cut`). -/
abbrev cutOf (l : List Step) : List (List Game.Msg) := PlaysCoin.cut (l.map Exec.kindOf) (l.filterMap Sum.getLeft?)

theorem cutOf_ne_nil (l : List Step) : cutOf l ≠ [] := FlockVBridge.Family.cut_ne_nil _ _

/-- **The rounds that close in a prefix are the prefix's**: steps `a ++ b` have `a`'s messages at each round before
`a`'s last coin. -/
theorem cutOf_append (b : List Step) : ∀ (a : List Step) (i : ℕ), i < (a.filterMap Sum.getRight?).length →
    (cutOf (a ++ b))[i]? = (cutOf a)[i]?
  | [], _, hi => absurd hi (Nat.not_lt_zero _)
  | .inl m :: a, i, hi => by
      have hl : ((Sum.inl m :: a : List Step).filterMap Sum.getRight?).length =
          (a.filterMap Sum.getRight?).length := rfl
      have ih := cutOf_append b a i (by omega)
      change (PlaysCoin.consHead m (cutOf (a ++ b)))[i]? = (PlaysCoin.consHead m (cutOf a))[i]?
      obtain ⟨g, gs, hg⟩ := List.exists_cons_of_ne_nil (cutOf_ne_nil (a ++ b))
      obtain ⟨g', gs', hg'⟩ := List.exists_cons_of_ne_nil (cutOf_ne_nil a)
      rw [hg, hg'] at ih ⊢
      cases i with
      | zero =>
          simp only [List.getElem?_cons_zero, Option.some.injEq] at ih
          simp only [PlaysCoin.consHead, List.getElem?_cons_zero, ih]
      | succ j => simpa only [PlaysCoin.consHead, List.getElem?_cons_succ] using ih
  | .inr c :: a, i, hi => by
      have hl : ((Sum.inr c :: a : List Step).filterMap Sum.getRight?).length =
          (a.filterMap Sum.getRight?).length + 1 := rfl
      change ([] :: cutOf (a ++ b))[i]? = ([] :: cutOf a)[i]?
      cases i with
      | zero => rfl
      | succ j =>
          rw [List.getElem?_cons_succ, List.getElem?_cons_succ]
          exact cutOf_append b a j (by omega)

/-- **The reps' part of a run's steps**: the cap's and the points' steps, then the reps' in their interleaving. -/
def repPart {e : ExecHolo} (u : C3Run e) : List Step := (runSteps u).take 2 ++ (firstSteps (u.ts 0) ++ u.ts 1)

theorem runSteps_split {e : ExecHolo} (u : C3Run e) : ∃ rest, runSteps u = repPart u ++ rest := by
  obtain ⟨rest, h⟩ := reps_prefix u
  exact ⟨rest, by rw [repPart, List.append_assoc ((runSteps u).take 2) (firstSteps (u.ts 0) ++ u.ts 1) rest, h,
    List.take_append_drop]⟩

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

section source

variable {S : ℕ} {Vs : Fin S → VStmt}

/-- **The source's round-`i` messages, at a round that closes in the reps' part, are that part's.** -/
theorem c3Msgs_repPart (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (Lb : (j : Fin S) → (Vs j).Bits) (i : ℕ)
    (hi : i < ((repPart (src R ω t Lb)).filterMap Sum.getRight?).length) :
    (c3Msgs src R ω t Lb)[i]? = (cutOf (repPart (src R ω t Lb)))[i]? := by
  obtain ⟨rest, h⟩ := runSteps_split (src R ω t Lb)
  show (cutOf (runSteps (src R ω t Lb)))[i]? = _
  rw [h]
  exact cutOf_append rest _ i hi

/-- **`MsgsReadC3`'s reps' part**: two accepted outer proofs whose layers agree on the read wires of stage `i`'s
commit strings have the same messages at round `i` of the source's reps' parts, where it closes in both. -/
def RepsMsgsRead (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) : Prop :=
  ∀ i R ω t R₂ τ o R' ω' t' R₂' τ' o', FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
    FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R' ω' t' R₂' τ' o' →
    maskOf Vs (skOf Vs own sel) i (layers zs dg N k Rw R ω t R₂ τ) =
      maskOf Vs (skOf Vs own sel) i (layers zs dg N k Rw R' ω' t' R₂' τ') →
    i < ((repPart (src R ω t (layers zs dg N k Rw R ω t R₂ τ))).filterMap Sum.getRight?).length →
    i < ((repPart (src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ'))).filterMap Sum.getRight?).length →
    (cutOf (repPart (src R ω t (layers zs dg N k Rw R ω t R₂ τ))))[i]? =
      (cutOf (repPart (src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ'))))[i]?

/-- **`MsgsReadC3` at the rounds that close in the reps' parts**, from those parts' read fact. -/
theorem msgsRead_repPart {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    (h : RepsMsgsRead zs dg N k Rw src own sel) {i : ℕ}
    {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    {R' : Reg} {ω' : L.Ω} {t' : List (Cm × (ℐ).Coin)} {R₂' : Reg₂} {τ' o'}
    (hA : FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o)
    (hA' : FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R' ω' t' R₂' τ' o')
    (hX : maskOf Vs (skOf Vs own sel) i (layers zs dg N k Rw R ω t R₂ τ) =
      maskOf Vs (skOf Vs own sel) i (layers zs dg N k Rw R' ω' t' R₂' τ'))
    (hi : i < ((repPart (src R ω t (layers zs dg N k Rw R ω t R₂ τ))).filterMap Sum.getRight?).length)
    (hi' : i < ((repPart (src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ'))).filterMap Sum.getRight?).length) :
    (c3Msgs src R ω t (layers zs dg N k Rw R ω t R₂ τ))[i]? =
      (c3Msgs src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ'))[i]? := by
  rw [c3Msgs_repPart src R ω t _ i hi, c3Msgs_repPart src R' ω' t' _ i hi']
  exact h i R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX hi hi'

/-! ## The source tied to V*'s rep units -/

/-- **The source tied to V*'s rep units** (n7's tie for the reps): at every accepted outer proof, each rep's steps in
the source's run are a rep unit (`RepTie`). -/
def SrcRepTie (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)) : Prop :=
  AtSrc zs dg N k Rw src fun _ u => ∀ i : Fin 2, RepTie (holoAt st hp) u i

/-- **The reps' read fact on the source's own steps**, at a source tied to V*'s rep units: `repsRead_c3DecOf`'s
`hrp`. -/
theorem srcRepsReadAt_of_tie {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (h : SrcRepTie zs dg N k Rw src) :
    AtSrc zs dg N k Rw src fun _ u => FlockC3.HoloValues.RepsReadAt (holoAt st hp) (runSteps u) u :=
  fun R ω t R₂ τ o hacc => repsReadAt_of_tie (h R ω t R₂ τ o hacc 0) (h R ω t R₂ τ o hacc 1)

/-- At every accepted outer proof and every rep `i` whose values `rep R ω t Lb i` at its history and V*'s layers meet
E's preconditions at `sh` (`alg`, `acc`). -/
def AtRep (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Flock.Shape)
    (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (P : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Prop) : Prop :=
  ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ i : Fin 2,
    AlgebraHolds sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) i) →
      AccHolds sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) i) →
    P R ω t (layers zs dg N k Rw R ω t R₂ τ) i

/-- **What is left of a rep unit once its values meet E's preconditions** (not proved here; its parts are
`ProofOfVals`, `SetupOfVals`, `FramesOfVals` and `StepsOfVals`, `repUnitOfAlg_of_parts`): the source's rep-`i` steps
are a rep unit at the values `rep R ω t Lb i`. G's plain analogue builds its record from the table files' values
(`TableSrc.runs_filesAt`) under `VStarFamily.hRuns_of`'s obligations. -/
def RepUnitOfAlg (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (sh : Flock.Shape) (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData) :
    Prop :=
  AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
    ∃ U : FlockC3.HoloValues.RepUnit ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch
        (extraClaims ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).cls.ptLocal
          (holoAt st hp).cls.mPts (src R ω t Lb).pts)
        (src R ω t Lb).cap₀ (src R ω t Lb).os.1 i ((src R ω t Lb).ts i),
      U.sh = sh ∧ U.d = rep R ω t Lb i

/-- **Stub: the proof's part** (80–120 lines) at values `vals R ω t Lb i`: the proof's messages, the circuit fold and
the ring switches from the rep's values, and the record's coins, as G's `RunsFrom` gives them at the table files'
decode. -/
def ProofOfVals (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Flock.Shape)
    (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals) : Prop :=
  AtRep zs dg N k Rw sh rep fun R ω t Lb i => UnitProof sh (rep R ω t Lb i) (vals R ω t Lb i)

/-- **Stub: the setup's and schedule's part** (40–60 lines) at the run's statement, schedule and extra claims, as G's
`SetupRealizes` and `ShapeIs` give them in the plain case. -/
def SetupOfVals (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (sh : Flock.Shape) (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals) : Prop :=
  AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
    UnitSetup ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch
      (extraClaims ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).cls.ptLocal
        (holoAt st hp).cls.mPts (src R ω t Lb).pts) sh (vals R ω t Lb i)

/-- **Stub: the record's frames at the holo shape** (150–250 lines beside c1, the holo generalization of
`run_of_frames` and `hlg_of_frames`, about 600): `framed_frames` and `hlg_of_frames` at holo. -/
def FramesOfVals (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Flock.Shape)
    (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals) : Prop :=
  AtRep zs dg N k Rw sh rep fun R ω t Lb i => UnitFrames sh (rep R ω t Lb i) (vals R ω t Lb i)

/-- **Stub: the source's rep steps** (about 60 lines): rep `i`'s steps in the source's run are `holoRepMsgs` of the
proof woven with the record's coins, and the run's cap and openings are the proof's. -/
def StepsOfVals (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (sh : Flock.Shape) (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals) : Prop :=
  AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
    UnitSteps ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch
      (extraClaims ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).cls.ptLocal
        (holoAt st hp).cls.mPts (src R ω t Lb).pts)
      (src R ω t Lb).cap₀ (src R ω t Lb).os.1 i ((src R ω t Lb).ts i) sh (rep R ω t Lb i)
      (vals R ω t Lb i)

/-- **`RepUnitOfAlg` from its four parts** at values `vals`. -/
theorem repUnitOfAlg_of_parts {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    (hpr : ProofOfVals zs dg N k Rw sh rep vals) (hs : SetupOfVals zs dg N k Rw src sh rep vals)
    (hf : FramesOfVals zs dg N k Rw sh rep vals) (hst : StepsOfVals zs dg N k Rw src sh rep vals) :
    RepUnitOfAlg zs dg N k Rw src sh rep := fun R ω t R₂ τ o hA i halg hacc =>
  ⟨unitOfParts (vals R ω t (layers zs dg N k Rw R ω t R₂ τ) i) halg hacc (hpr R ω t R₂ τ o hA i halg hacc)
    (hs R ω t R₂ τ o hA i halg hacc) (hf R ω t R₂ τ o hA i halg hacc) (hst R ω t R₂ τ o hA i halg hacc), rfl, rfl⟩

/-- **The reps' tie from G's items 1–4** at a rep decoder `rep` of the history and V*'s layers and the rest of each
unit at its values (`RepUnitOfAlg`). -/
theorem srcRepTie_of_alg {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    (halg : AlgebraDecodes (I := ℐ) zs dg N k Rw sh rep) (hacc : AccChains (I := ℐ) zs dg N k Rw sh rep)
    (h : RepUnitOfAlg zs dg N k Rw src sh rep) : SrcRepTie zs dg N k Rw src := by
  intro R ω t R₂ τ o hA i
  obtain ⟨U, -, -⟩ := h R ω t R₂ τ o hA i (halg R ω t R₂ τ o hA i) (hacc R ω t R₂ τ o hA i)
  exact ⟨U⟩

end source

/-! ## At the fixed decoder -/

section honest

variable {ish : RecVStar.Inner} {ss : Array VStar.Stmt} {hss : RecVStar.statements ish = .ok ss}
  {bind : Fin (vstarCount ish) → VStar.Bindings} {Vs : Fin (vstarCount ish) → VStmt}
  {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
    (Vs j).Outer Reg'}
  {own : Fin (vstarCount ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
  {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String} {reg : Lean.Json}
  {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {src : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) →
    C3Run (holoAt st hp)}

local notation "𝒵" => zs
local notation "𝓓" => C3Family.honest (xR := xR) ish hss bind hob 𝒵 own sel bc
  (c3DecOf 𝒵 dg N k Rw src own sel) wr reg

/-- **`RepsRead` at the fixed decoder, from the reps' tie.** -/
theorem repsRead_of_tie (hs : SrcOK 𝒵 dg N k Rw src own sel) (ht : SrcRepTie 𝒵 dg N k Rw src) :
    FlockC3.HoloValues.RepsRead 𝓓 dg N k Rw :=
  repsRead_c3DecOf hs (srcRepsReadAt_of_tie ht)

/-- **(n2) at the fixed decoder, from the reps' tie**: each rep plays `holoRepC` to its decoded output. -/
theorem hReps_of_tie (hs : SrcOK 𝒵 dg N k Rw src own sel) (ht : SrcRepTie 𝒵 dg N k Rw src) :
    FlockC3.HoloValues.HReps 𝓓 dg N k Rw :=
  FlockC3.HoloValues.hReps_honest_of_read (repsRead_of_tie hs ht)

end honest

end

end FlockC3.Reps

#print axioms FlockC3.Reps.UnitVals
#print axioms FlockC3.Reps.UnitProof
#print axioms FlockC3.Reps.UnitSetup
#print axioms FlockC3.Reps.UnitFrames
#print axioms FlockC3.Reps.UnitSteps
#print axioms FlockC3.Reps.unitOfParts
#print axioms FlockC3.Reps.RepTie
#print axioms FlockC3.Reps.reps_prefix
#print axioms FlockC3.Reps.repsReadAt_of_tie
#print axioms FlockC3.Reps.cutOf
#print axioms FlockC3.Reps.cutOf_ne_nil
#print axioms FlockC3.Reps.cutOf_append
#print axioms FlockC3.Reps.repPart
#print axioms FlockC3.Reps.runSteps_split
#print axioms FlockC3.Reps.c3Msgs_repPart
#print axioms FlockC3.Reps.RepsMsgsRead
#print axioms FlockC3.Reps.msgsRead_repPart
#print axioms FlockC3.Reps.SrcRepTie
#print axioms FlockC3.Reps.srcRepsReadAt_of_tie
#print axioms FlockC3.Reps.AtRep
#print axioms FlockC3.Reps.RepUnitOfAlg
#print axioms FlockC3.Reps.ProofOfVals
#print axioms FlockC3.Reps.SetupOfVals
#print axioms FlockC3.Reps.FramesOfVals
#print axioms FlockC3.Reps.StepsOfVals
#print axioms FlockC3.Reps.repUnitOfAlg_of_parts
#print axioms FlockC3.Reps.srcRepTie_of_alg
#print axioms FlockC3.Reps.repsRead_of_tie
#print axioms FlockC3.Reps.hReps_of_tie
