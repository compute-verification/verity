import Proofs.Flock.C3.Sparse.PubLift
import Proofs.Flock.C3.Opening.SessionCoins
import Proofs.Flock.C3.VStarList.Range

/-!
# `SparsePubRead` from check 5's acceptance

* `checkSparse_of_check`: at a unit named `rec-sparse/n<NU>-e<ELL>-s<S_LOG>-t<T_LOG>` (`RecSparse.params`), `RecCoins.check`
  accepting is `checkSparse` at the verifier's coins record and the inner session it loads (the name is neither R's
  level's nor R's opening's, `stackedParams_sparse`, `openingParams_sparse`).
* `rep_of_checkSparse`: what `checkSparse` computed at rep `i`: the algebra's shape and extra claims, the unit range,
  the rep's `verifierRows` at its rounds, the sparse coins and the layout factors at the rep's points.
* `sparsePubRead_of_checkSparse`: at an inner session whose `m`, `k_log` and unit range are the class's, and rep `i`'s
  rounds `d`'s (the custody), the coins record `⟨α, z, ρ_in, r_in, U, sparseCs sc⟩` reads `SparsePubRead`.
* `load_files`, `sparsePubRead_of_files`: an inner session loaded from the inner statement's files has the statement's
  `m`, `k_log` and unit range, which are the class's where its class checks pass, so the custody left is that the files'
  statement is the class's and rep `i`'s rounds `d`'s.
-/

namespace FlockC3.Sparse

open Flock FlockSoundness.Discharge.Recursion
open FlockVBridge.Sparse (SCoins)
open FlockVBridge.Compiled (RepData)

/-- Two strings whose first `n` characters differ differ after any suffixes. -/
theorem lit_ne {a b x y : String} (n : ℕ) (ha : n ≤ a.toList.length) (hb : n ≤ b.toList.length)
    (h : a.toList.take n ≠ b.toList.take n) : a ++ x ≠ b ++ y := by
  intro e
  have := congrArg (fun s : String => s.toList.take n) e
  simp only [String.toList_append] at this
  rw [List.take_append_of_le_length ha, List.take_append_of_le_length hb] at this
  exact h this

/-- A RecSparse unit's name is no R level's (`stackedParams`). -/
theorem stackedParams_sparse (sh : RecSparse.Shape) : RecCoins.stackedParams (RecSparse.unitName sh) = none := by
  unfold RecCoins.stackedParams
  split
  · rfl
  · split
    · split
      · rename_i hq
        exfalso
        revert hq
        simp only [beq_iff_eq, RecSparse.unitName, String.append_assoc]
        exact lit_ne 5 (by decide) (by decide) (by decide)
      · rfl
    · rfl

/-- A RecSparse unit's name is no R opening's (`RecOpening.params`). -/
theorem openingParams_sparse (sh : RecSparse.Shape) : RecOpening.params (RecSparse.unitName sh) = none := by
  cases hp : RecOpening.params (RecSparse.unitName sh) with
  | none => rfl
  | some p =>
    exfalso
    obtain ⟨m, c, j⟩ := p
    have e : RecOpening.unitName m c j = RecSparse.unitName sh := by
      unfold RecOpening.params at hp
      repeat' (first | split at hp | dsimp only at hp)
      all_goals first
        | (simp only [Option.some.injEq, Prod.mk.injEq] at hp; obtain ⟨rfl, rfl, rfl⟩ := hp; assumption)
        | cases hp
    revert e
    simp only [RecSparse.unitName, RecOpening.unitName, String.append_assoc]
    exact lit_ne 5 (by decide) (by decide) (by decide)

/-- **Check 5 at a RecSparse unit is `checkSparse`**: at the verifier's coins record and the inner session it loads. -/
theorem checkSparse_of_check {sj : Flock.Stmt} {own : Array Registered.Port} {src : Option RecCoins.Source}
    {c3 : Option RecCoins.C3} (h : RecCoins.check sj (some own) src c3 = .ok ()) {sh : RecSparse.Shape}
    (hn : RecSparse.params sj.c.unit.name = some sh) :
    ∃ c3' f inn, c3 = some c3' ∧ src = some f ∧ f.load sj.c.unit.name = .ok inn ∧
      RecCoins.checkSparse inn c3' sj.c sj.pub own sh = .ok () := by
  have hu := FlockVBridge.Sparse.unitName_params hn
  have hso : RecCoins.stackedParams sj.c.unit.name = none := hu ▸ stackedParams_sparse sh
  have hop : RecOpening.params sj.c.unit.name = none := hu ▸ openingParams_sparse sh
  unfold RecCoins.check at h
  simp only [hn, hso, hop, Option.isNone_some, Bool.and_false, Bool.false_and, Bool.false_eq_true, ↓reduceIte] at h
  cases src with
  | none => cases h
  | some f =>
    obtain ⟨inn, hl, h⟩ := FlockSoundness.Refine.ebind_ok h
    obtain ⟨c, hc, h⟩ := FlockSoundness.Refine.ebind_ok h
    cases c3 with
    | none => cases hc
    | some c3' =>
      obtain rfl : c3' = c := Except.ok.inj hc
      exact ⟨c3', f, inn, rfl, rfl, hl, h⟩

/-- The verifier's comparison of RecSparse shapes (`RecVStar.instBEqShape`) is equality. -/
theorem sparseShape_eq {a b : RecSparse.Shape} (h : ¬ (a != b) = true) : a = b := by
  obtain ⟨_, _, _, _⟩ := a; obtain ⟨_, _, _, _⟩ := b
  simp only [bne, BEq.beq, Bool.not_eq_true', Bool.not_eq_false, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨⟨rfl, rfl⟩, rfl⟩, rfl⟩ := h
  rfl

theorem mapM_two_get {A : Type} {f : Nat → Except String A} {cs : Array A} (h : #[0, 1].mapM f = .ok cs)
    (i : Fin 2) : ∃ a, f i.val = .ok a := by
  obtain ⟨a, b, h0, h1, -⟩ := FlockC3.Opening.mapM_two h
  fin_cases i
  exacts [⟨a, h0⟩, ⟨b, h1⟩]

/-- **What `checkSparse` computed at rep `i`**: the algebra's shape and extra claims, the unit range, the rep's values at
its rounds, its sparse coins and the layout factors at its points. -/
theorem rep_of_checkSparse {inn : RecCoins.Inner} {c3 : RecCoins.C3} {cc : Circuit} {pub : Public}
    {own : Array Registered.Port} {sh : RecSparse.Shape} (h : RecCoins.checkSparse inn c3 cc pub own sh = .ok ())
    (i : Fin 2) :
    sh = c3.sparse ∧ ∃ ash extras rng rep sc U, inn.algebra = .ok (ash, extras) ∧ inn.unitRange = .ok rng ∧
      inn.rep ash extras i.val = .ok rep ∧ RecCoins.sparseCoins c3 cc.unit.name i.val = .ok sc ∧
      RecCoins.layoutFactors sh rep.info.lc.fold.rhoIn rep.info.lc.rIn rng = .ok U := by
  unfold RecCoins.checkSparse at h
  by_cases hs : (sh != c3.sparse) = true
  · rw [ite_eq_left hs] at h; cases h
  rw [ite_eq_right hs] at h
  refine ⟨sparseShape_eq hs, ?_⟩
  obtain ⟨⟨ash, extras⟩, ha, h⟩ := FlockSoundness.Refine.ebind_ok h
  dsimp only at h
  obtain ⟨rng, hr, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨rows, hrows, -⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨row, hrow⟩ := mapM_two_get hrows i
  obtain ⟨rep, hrep, hrow⟩ := FlockSoundness.Refine.ebind_ok hrow
  obtain ⟨sc, hsc, hrow⟩ := FlockSoundness.Refine.ebind_ok hrow
  obtain ⟨v, hv, -⟩ := FlockSoundness.Refine.ebind_ok hrow
  obtain ⟨U, hU, -⟩ := FlockSoundness.Refine.ebind_ok (FlockC3.Opening.mapError_ok hv)
  exact ⟨ash, extras, rng, rep, sc, U, ha, (FlockC3.Opening.mapError_ok hr), hrep, hsc, hU⟩

/-- `Inner.algebra`'s shape is at the session's `m` and `k_log` (`RecResiduals.shape?`). -/
theorem algebra_shape {inn : RecCoins.Inner} {ash : Shape} {extras : Array Claim}
    (h : inn.algebra = .ok (ash, extras)) : ash.m = inn.m ∧ ash.kLog = inn.kLog := by
  unfold RecCoins.Inner.algebra at h
  obtain ⟨xs, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  cases hs : RecResiduals.shape? inn.m inn.kLog (2 + xs.size) with
  | none => rw [hs] at h; cases h
  | some s =>
    rw [hs] at h
    cases h
    unfold RecResiduals.shape? at hs
    cases hf : fast100 inn.m with
    | none => simp [hf] at hs
    | some sch =>
      rw [hf] at hs
      dsimp only [bind, Option.bind] at hs
      split at hs
      · cases hs
      · cases hs; exact ⟨rfl, rfl⟩

/-- `Inner.rep`'s values are `verifierRows` at the session's rounds. -/
theorem rep_rows {inn : RecCoins.Inner} {ash : Shape} {extras : Array Claim} {r : ℕ} {rep : RecCoins.Rep}
    (h : inn.rep ash extras r = .ok rep) : inn.rounds r = .ok rep.rounds ∧
      RecAlgebra.verifierRows ash rep.rounds extras RecCoins.ZERO_COMB = .ok (rep.v, rep.info, rep.kappa) := by
  unfold RecCoins.Inner.rep at h
  obtain ⟨rs, hrs, h⟩ := FlockSoundness.Refine.ebind_ok h
  split at h
  · rename_i v info kappa hv
    cases h
    exact ⟨hrs, hv⟩
  · cases h

/-- **`SparsePubRead` from `checkSparse`**: at an inner session at the class's `m`, `k_log` and unit range, where rep `i`'s
rounds are `d`'s (`hd`, the custody), the coins record of rep `i`'s points, layout factors and sparse coins reads it. -/
theorem sparsePubRead_of_checkSparse {inn : RecCoins.Inner} {c3 : RecCoins.C3} {cc : Circuit} {pub : Public}
    {own : Array Registered.Port} {sh : RecSparse.Shape} (h : RecCoins.checkSparse inn c3 cc pub own sh = .ok ())
    {e : ExecHolo} {rng : Range} (hsh : c3.sparse = unitShape e) (hm : inn.m = e.cls.m) (hk : inn.kLog = e.cls.kLog)
    (hrng : inn.unitRange = .ok rng) (i : Fin 2) {d : RepData} (hd : inn.rounds i.val = .ok d.rounds) :
    ∃ sc c, RecCoins.sparseCoins c3 cc.unit.name i.val = .ok sc ∧ c.cs = sparseCs sc ∧ SparsePubRead e rng d c := by
  obtain ⟨rfl, ash, extras, rng', rep, sc, U, ha, hr, hrep, hsc, hU⟩ := rep_of_checkSparse h i
  rw [hrng] at hr
  cases hr
  obtain ⟨ham, hak⟩ := algebra_shape ha
  obtain ⟨hrs, hrows⟩ := rep_rows hrep
  rw [Except.ok.inj (hrs.symm.trans hd)] at hrows
  rw [hsh] at hU
  exact ⟨sc, ⟨rep.info.lc.fold.alpha, rep.info.zc.z, rep.info.lc.fold.rhoIn, rep.info.lc.rIn, U, sparseCs sc⟩, hsc,
    rfl, ash, extras, _, rep.v, rep.info, rep.kappa, hrows, ham.trans hm, hak.trans hk, rfl, rfl, rfl, rfl, hU⟩

/-- **An inner session loaded from the inner statement's files** has the statement's `m`, `k_log` and unit range. -/
theorem load_files {unit : String} {f : RecCoins.Files} {inn : RecCoins.Inner}
    (h : RecCoins.Source.load unit (.files f) = .ok inn) :
    ∃ spec st, f.stmt unit = .ok (spec, st) ∧ inn.m = st.m ∧ inn.kLog = st.c.kLog ∧
      inn.unitRange = RecCoins.unitRangeOf st := by
  unfold RecCoins.Source.load RecCoins.Inner.load at h
  obtain ⟨⟨spec, st⟩, hs, h⟩ := FlockSoundness.Refine.ebind_ok h
  refine ⟨spec, st, hs, ?_⟩
  dsimp only at h
  repeat' split at h
  all_goals cases h
  all_goals exact ⟨rfl, rfl, rfl⟩

/-- **`SparsePubRead` at the inner statement's files**: where check 5 loads the inner session from the files of the
statement `st` (`hs`), whose class checks pass (`hc`), its `m`, `k_log` and unit range are the class's, so
`sparsePubRead_of_checkSparse` leaves only rep `i`'s rounds (`hd`). -/
theorem sparsePubRead_of_files {unit : String} {f : RecCoins.Files} {inn : RecCoins.Inner}
    (hl : RecCoins.Source.load unit (.files f) = .ok inn) {spec : SessionSpec} {st : Flock.Stmt}
    (hs : f.stmt unit = .ok (spec, st)) {hp : HoloParams} (hc : ClassOk (shapeOf st)) {c3 : RecCoins.C3} {cc : Circuit}
    {pub : Public} {own : Array Registered.Port} {sh : RecSparse.Shape}
    (h : RecCoins.checkSparse inn c3 cc pub own sh = .ok ()) (hsh : c3.sparse = unitShape (holoAt st hp)) (i : Fin 2)
    {d : RepData} (hd : inn.rounds i.val = .ok d.rounds) :
    ∃ rng sc c, unitRangeOf st = .ok rng ∧ RecCoins.sparseCoins c3 cc.unit.name i.val = .ok sc ∧ c.cs = sparseCs sc ∧
      SparsePubRead (holoAt st hp) rng d c := by
  obtain ⟨spec', st', hs', hm, hk, hr⟩ := load_files hl
  obtain ⟨-, rfl⟩ := Prod.mk.inj (Except.ok.inj (hs'.symm.trans hs))
  obtain ⟨-, -, rng, -, -, -, -, hrng, -⟩ := (rep_of_checkSparse h i).2
  have hm' : (holoAt st' hp).cls.m = st'.m := by
    show (execClassAt st').cls.m = st'.m
    rw [execClassAt, dite_eq_left hc]
    rfl
  have hk' : (holoAt st' hp).cls.kLog = st'.c.kLog := FlockC3.VStarList.execClassAt_kLog hc
  obtain ⟨sc, c, hsc, hcs, hr'⟩ :=
    sparsePubRead_of_checkSparse h hsh (hm.trans hm'.symm) (hk.trans hk'.symm) hrng i hd
  exact ⟨rng, sc, c, hr.symm.trans hrng, hsc, hcs, hr'⟩

section src

open FlockSoundness FlockSoundness.Discharge
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose FlockC3.Reps

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
  {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {stp : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) →
    C3Run (holoAt st hp)}
  {sh : Flock.Shape}
  {rep : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
  {c : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → SCoins}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **`SrcSparseTie` at `tiedSrc (sparseOf stp w c)` from its two halves**: V*'s sessions (`hs`), the pub conjunct from
check 5's values at the reps' rounds (`srcSparsePub_of_alg`, `hr` from `sparsePubRead_of_files`), and the `h` conjunct
at the copy each session names (`hH`, which `FlockC3.HPort.h_of_sessionAt` gives from `SrcHTie`). -/
theorem srcSparseTie_of_pubH {rng : Flock.Range} (hu : UnitsOf st hp) (hrng : unitRangeOf st = .ok rng)
    (halg : AlgebraDecodes (I := ℐ) zs dg N k Rw sh rep) (hacc : AccChains (I := ℐ) zs dg N k Rw sh rep)
    (hU : RepUnitOfAlg zs dg N k Rw (tiedSrc stp) sh rep)
    (hr : AtRep zs dg N k Rw sh rep fun R ω t Lb i => SparsePubRead (holoAt st hp) rng (rep R ω t Lb i) (c R ω t Lb i))
    (c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Flock.RecCoins.C3) (unit : String)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array Flock.F128)
    (hs : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ i : Fin 2,
      SparseSessionAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
        (c3 R ω t (layers zs dg N k Rw R ω t R₂ τ)) unit i (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i)
        (c R ω t (layers zs dg N k Rw R ω t R₂ τ) i))
    (hH : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ i : Fin 2,
      SparseSessionAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
        (c3 R ω t (layers zs dg N k Rw R ω t R₂ τ)) unit i (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i)
        (c R ω t (layers zs dg N k Rw R ω t R₂ τ) i) →
      ∀ l : Fin 64, fOf (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i)[l.val]! =
        ((tiedSrc stp R ω t (layers zs dg N k Rw R ω t R₂ τ)).rep i).2.h l) :
    SrcSparseTie zs dg N k Rw (tiedSrc (sparseOf stp w c)) :=
  srcSparseTie_sparseOf c3 unit w c fun R ω t R₂ τ o hA i =>
    ⟨hs R ω t R₂ τ o hA i, srcSparsePub_of_alg hu hrng halg hacc hU hr R ω t R₂ τ o hA i,
      hH R ω t R₂ τ o hA i (hs R ω t R₂ τ o hA i)⟩

end src

end FlockC3.Sparse

#print axioms FlockC3.Sparse.checkSparse_of_check
#print axioms FlockC3.Sparse.rep_of_checkSparse
#print axioms FlockC3.Sparse.sparsePubRead_of_checkSparse
#print axioms FlockC3.Sparse.sparsePubRead_of_files
#print axioms FlockC3.Sparse.srcSparseTie_of_pubH
