import Proofs.Flock.VBridge.Compiled
import Proofs.Flock.VBridge.Inputs

/-!
# VBridge G item 4: the running sums `rec-acc` chain into E's `hacc`

V*'s running sums are one chain (`rec_outer.Chain`, rec_outer.py:28-35): the openings of both reps, rep-major, then by
level and query, are positions `0 .. N - 1`, and the prover registers the chain's `N + 1` rows as the value `rec-acc` of
the outer proof's registration `R₂` (`outerV`). The `rec-open` unit of the opening at position `p` reads `acc` at row
`p` and `acc_out` at row `p + 1`, and its coefficients `coef` are the verifier's own (`rec_outer.coefficients`,
rec_outer.py:159-161: `κ[level][r][l] · eq(α)[j]`, `coefAt`); V*'s algebra reads a rep's first and last rows as its
running sums before and after its openings (`Chain.rep_ends`), the message elements `acc0` and `acc1`.

* `Openings`: one rep's openings in V*'s level sessions: where its chain starts, each level's number of openings, their
  rows, the level's `α` and the `coef` each opening's unit reads. `Openings.start o ℓ` is the chain position of level
  `ℓ`'s first opening.
* `UnitSums`: each opening's `rec-open` unit's conclusion on its running sums (`sound_recOpen`'s `acc_out`, decoded):
  the first two elements of row `p + 1` are `consTerm` of its row, row `p` and its `coef`. One hypothesis per unit, all
  at the one function `acc`, the rows of `R₂`'s `rec-acc`, which every level session reads.
* `CoefsAre`, `ConsAre`, `ReadsEnds`: the verifier's registered `coef` (`coefAt` of the `kappa` `verifierRows` gives),
  the rep's consistency sums (`consAt` of the opened rows), and the algebra's `acc0`, `acc1` (the chain's rows at the
  rep's ends).
* `verifierRows_kappa`: `kappa` has a row per Ligerito level and `2 ^ lanes` entries per component, as `build` starts it.
* `accHolds_of_units`: from those, E's `hacc` at the rep (`AccHolds`). The chain telescopes: each unit adds its
  opening's consistency terms (`fOf_consTerm_coefAt`), so each level adds its component of the final check's
  consistency sums, and in characteristic 2 the sums before and after add to their total, `kapSum`. Row 0 being zero is
  not needed.
* `accChains_of_units`: `AccChains`, from the same facts at every accepted outer proof.
-/

namespace FlockVBridge.AccChain

open Flock
open FlockSoundness.Refine (fOf fOf_add fOf_mul fOf_injective)
open Flock.RecAlgebra
open FlockVBridge.Compiled (RepData AccHolds)

/-! ## `kappa`'s shape -/

/-- `K'` has `K`'s shape. -/
def SameShape (K K' : Array (Array (Array F128))) : Prop :=
  K'.size = K.size ∧ ∀ ℓ r : ℕ, ((K'[ℓ]!)[r]!).size = ((K[ℓ]!)[r]!).size

theorem SameShape.refl (K : Array (Array (Array F128))) : SameShape K K := ⟨rfl, fun _ _ => rfl⟩

theorem SameShape.trans {a b c : Array (Array (Array F128))} (h1 : SameShape a b) (h2 : SameShape b c) :
    SameShape a c :=
  ⟨h2.1.trans h1.1, fun ℓ r => (h2.2 ℓ r).trans (h1.2 ℓ r)⟩

theorem sameShape_modify (K : Array (Array (Array F128))) {level r l : ℕ} (x : F128) (h1 : level < K.size)
    (h2 : r < K[level]!.size) : SameShape K (K.modify level (·.modify r (·.modify l (· + x)))) := by
  refine ⟨Array.size_modify .., fun ℓ r' => ?_⟩
  by_cases hl : level = ℓ
  · subst hl
    rw [getElem!_modify_eq _ _ _ h1]
    by_cases hr : r = r'
    · subst hr
      rw [getElem!_modify_eq _ _ _ h2, Array.size_modify]
    · rw [getElem!_modify_ne _ _ hr]
  · rw [getElem!_modify_ne _ _ hl]

theorem kapAdd_shape (r : ℕ) : ∀ (l : List (Key × F128)) (st st' : BSt), kapAdd r l st = .ok ((), st') →
    SameShape st.kappa st'.kappa
  | [], st, st', hr => by
    cases hr
    exact SameShape.refl _
  | (k, x) :: rest, st, st', hr => by
    simp only [kapAdd] at hr
    cases k
    case cons level l =>
      obtain ⟨st0, st2, hg, hr⟩ := bm_bind hr
      cases hg
      split at hr
      · rename_i hb
        obtain ⟨_, st1, hs, hr⟩ := bm_bind hr
        cases hs
        exact (sameShape_modify _ x hb.1 hb.2.1).trans (kapAdd_shape r rest _ st' hr)
      · obtain ⟨_, _, hs, _⟩ := bm_bind hr
        cases hs
    all_goals exact kapAdd_shape r rest st st' hr

section Shape

variable {sh : Shape} {comb0 : ℕ}

theorem accTerms_shape {fin ri : ℕ} {l : List (Key × F128)} {st st' : BSt} {acc : Array RecAlgebra.Term}
    (hr : accTerms sh fin ri l st = .ok (acc, st')) : SameShape st.kappa st'.kappa := by
  unfold accTerms at hr
  split at hr
  · cases hr
    exact SameShape.refl _
  · split at hr
    · cases hr
    · obtain ⟨_, st1, hk, hr⟩ := bm_bind hr
      have hs := kapAdd_shape _ l st st1 hk
      cases h1 : wi sh (.acc1 (ri - fin)) with
      | none => simp only [h1] at hr; cases hr
      | some i1 =>
        cases h0 : wi sh (.acc0 (ri - fin)) with
        | none => simp only [h1, h0] at hr; cases hr
        | some i0 =>
          simp only [h1, h0] at hr
          cases hr
          exact hs

theorem buildRes_shape {fin ri : ℕ} {f : Lin} {st st' : BSt} {res : RecAlgebra.Res} (hi : OpsOk sh comb0 st)
    (hr : buildRes sh fin comb0 ri f st = .ok (res, st')) : SameShape st.kappa st'.kappa := by
  unfold buildRes at hr
  obtain ⟨acc, st1, h1, hr⟩ := bm_bind hr
  obtain ⟨a1, a2, a3, a4, -, -, -⟩ := accTerms_spec #[] #[] (fun _ _ => 0) h1
  obtain ⟨l1, st1', hs1, hr⟩ := bm_bind hr
  obtain ⟨-, h⟩ := liftB_ok hs1
  subst st1'
  obtain ⟨cm, st2, h2, hr⟩ := bm_bind hr
  obtain ⟨l2, st2', hs2, hr⟩ := bm_bind hr
  obtain ⟨-, h⟩ := liftB_ok hs2
  subst st2'
  obtain ⟨pm, st3, h3, hr⟩ := bm_bind hr
  obtain ⟨cp, st4, h4, hr⟩ := bm_bind hr
  cases hr
  have hi1 : OpsOk sh comb0 st1 := hi.of_eq a1 a2 a3 a4
  obtain ⟨h2i, -, h2k, -⟩ := coefOps_spec #[] #[] (fun _ _ => 0) ri l1 st1 st2 cm hi1 h2
  obtain ⟨h3i, -, h3k, -⟩ := prodOps_spec #[] #[] (fun _ _ => 0) ri l2 st2 st3 pm h2i h3
  obtain ⟨-, -, h4k, -⟩ := constTerm_spec #[] #[] h3i h4
  rw [h4k, h3k, h2k]
  exact accTerms_shape h1

theorem buildAll_shape {fin : ℕ} : ∀ (fs : List Lin) (ri : ℕ) (st st' : BSt) (rs : Array RecAlgebra.Res),
    OpsOk sh comb0 st → buildAll sh fin comb0 fs ri st = .ok (rs, st') → SameShape st.kappa st'.kappa
  | [], ri, st, st', rs, hi, hr => by
    cases hr
    exact SameShape.refl _
  | f :: rest, ri, st, st', rs, hi, hr => by
    obtain ⟨r, st1, h1, hr⟩ := bm_bind hr
    obtain ⟨rs', st2, h2, hr⟩ := bm_bind hr
    cases hr
    exact (buildRes_shape hi h1).trans
      (buildAll_shape rest (ri + 1) st1 st' rs' (buildRes_spec #[] #[] (fun _ _ => 0) hi h1).1 h2)

end Shape

/-- Lanes of level `ℓ` of the shape's Ligerito schedule: `2 ^ lanes`, the size of `kappa`'s rows `build` starts. -/
def levelLanes (sh : Shape) (ℓ : ℕ) : ℕ := 2 ^ (sh.sched.levels[ℓ]!).lanes

theorem build_kappa {sh : Shape} {lins : Array Lin} {b : Built} (hb : build sh lins = .ok b) :
    SameShape (sh.sched.levels.map fun lv => Array.replicate 2 (Array.replicate (2 ^ lv.lanes) (0 : F128))) b.kappa := by
  unfold build at hb
  generalize hK : sh.sched.levels.map (fun lv => Array.replicate 2 (Array.replicate (2 ^ lv.lanes) (0 : F128))) = K
    at hb
  generalize hc0 : (lins.toList.map slotsOf).sum = comb0 at hb
  simp only at hb
  cases hB : (buildAll sh (2 + sh.claims) comb0 lins.toList 0).run { kappa := K } with
  | error e => rw [hB] at hb; cases hb
  | ok p =>
    obtain ⟨res, st⟩ := p
    rw [hB] at hb
    simp only [bind, Except.bind] at hb
    split at hb
    · cases hb
    · cases hb
      have hB' : buildAll sh (2 + sh.claims) comb0 lins.toList 0 { kappa := K } = .ok (res, st) := hB
      have hinit : OpsOk sh comb0 { kappa := K } := ⟨rfl, fun _ _ hk => by cases hk⟩
      subst hK
      exact buildAll_shape lins.toList 0 _ _ _ hinit hB'

/-- **`kappa`'s shape**: the `kappa` `verifierRows` gives has a row per level of the shape's Ligerito schedule, and
each of its two components has the level's `2 ^ lanes` entries (rec_algebra.py:837-841). -/
theorem verifierRows_kappa {sh : Shape} {rounds : Array (Array F128)} {extras : Array Claim} {comb v : Array F128}
    {info : Info Lin} {kappa : Array (Array (Array F128))}
    (hrows : verifierRows sh rounds extras comb = .ok (v, info, kappa)) :
    kappa.size = sh.sched.levels.size ∧
      ∀ ℓ < sh.sched.levels.size, ∀ r < 2, ((kappa[ℓ]!)[r]!).size = levelLanes sh ℓ := by
  unfold verifierRows at hrows
  cases hl : residualLins sh (Coins.ofRounds rounds) extras with
  | error e => rw [hl] at hrows; cases hrows
  | ok q =>
    obtain ⟨lins, info'⟩ := q
    rw [hl] at hrows
    simp only [bind, Except.bind] at hrows
    cases hb : build sh lins with
    | error e => rw [hb] at hrows; cases hrows
    | ok b =>
      rw [hb] at hrows
      simp only at hrows
      split at hrows
      · cases hrows
      · cases hrows
        obtain ⟨hs, hr⟩ := build_kappa hb
        refine ⟨by rw [hs, Array.size_map], fun ℓ hℓ r hr2 => ?_⟩
        rw [hr ℓ r, levelLanes]
        simp [getElem!_def, Array.getElem?_map, Array.getElem?_eq_getElem hℓ, hr2]

/-! ## One rep's openings and their units -/

/-- **One rep's openings in V*'s level sessions.** The chain position of the rep's first opening `p0`; at level `ℓ`,
the number of openings `q ℓ`, their rows `rows ℓ` (opening `j`'s at `j`), the level's `α` (`alpha ℓ`), and the `coef`
the `rec-open` unit of opening `j` reads (`coef ℓ j`). -/
structure Openings where
  p0 : ℕ
  q : ℕ → ℕ
  rows : ℕ → Array (Array F128)
  alpha : ℕ → Array F128
  coef : ℕ → ℕ → Array F128

/-- The chain position of level `ℓ`'s first opening (`rec_outer.chain_order`: by level, then query). -/
def Openings.start (o : Openings) (ℓ : ℕ) : ℕ := o.p0 + ((List.range ℓ).map o.q).sum

theorem Openings.start_zero (o : Openings) : o.start 0 = o.p0 := by
  simp [Openings.start]

theorem Openings.start_succ (o : Openings) (ℓ : ℕ) : o.start (ℓ + 1) = o.start ℓ + o.q ℓ := by
  simp [Openings.start, List.range_succ, List.sum_append, Nat.add_assoc]

/-- **The `rec-open` units' conclusions on the running sums.** For each opening `j` of each level `ℓ` of the rep, its
unit (`recOpen`) reads its row, `acc` at its chain position `p` and its `coef`, and its `acc_out` is the chain's row
`p + 1`; the first two elements of that row are the opening's consistency terms (`sound_recOpen`'s second conclusion,
read as elements: `rec_open.acc_ref`). `acc` is the rows of `R₂`'s `rec-acc`, the same in every level session. -/
def UnitSums (sh : Shape) (acc : ℕ → Array F128) (o : Openings) : Prop :=
  ∀ ℓ < sh.sched.levels.size, ∀ j < o.q ℓ, ∀ k < 2,
    (acc (o.start ℓ + j + 1))[k]! = consTerm (levelLanes sh ℓ) (o.rows ℓ)[j]! (acc (o.start ℓ + j)) (o.coef ℓ j) k

/-- **The verifier's coefficients** (`rec_outer.coefficients`, rec_outer.py:159-161): the `coef` it registers for
opening `j` of level `ℓ` is `κ[ℓ][r][l] · eq(α)[j]`, at the `kappa` `verifierRows` gives at the rep's coins. -/
def CoefsAre (sh : Shape) (d : RepData) (o : Openings) : Prop :=
  ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
    ∀ ℓ < sh.sched.levels.size, ∀ j < o.q ℓ,
      o.coef ℓ j = coefAt (kappa[ℓ]!) (levelLanes sh ℓ) (eqTable (o.alpha ℓ))[j]!

/-- **The rep's consistency sums** are the verifier's, `Σ_j eq(α)[j]·row_j[l]` over the opened rows (`consAt`). -/
def ConsAre (sh : Shape) (d : RepData) (o : Openings) : Prop :=
  ∀ ℓ < sh.sched.levels.size, ∀ l < levelLanes sh ℓ, d.cons ℓ l = consAt (o.rows ℓ) (o.q ℓ) (o.alpha ℓ) l

/-- **V*'s algebra reads the chain's rows at the rep's ends** (`Chain.rep_ends`): `acc0` is the row before the rep's
first opening, `acc1` the row after its last. -/
def ReadsEnds (sh : Shape) (d : RepData) (acc : ℕ → Array F128) (o : Openings) : Prop :=
  ∀ r < 2, valOf sh d.w d.comb d.cons (.acc0 r) = (acc o.p0)[r]! ∧
    valOf sh d.w d.comb d.cons (.acc1 r) = (acc (o.start sh.sched.levels.size))[r]!

/-! ## The chain telescopes -/

/-- Over a level's first `j` openings, from chain position `s`, each unit adding its opening's consistency terms, the
running sum's component `k` gains `Σ_l κ[k][l] · Σ_{i < j} eq(α)[i]·rows[i][l]`. -/
theorem level_sums {LANES : ℕ} {kap rows : Array (Array F128)} {alpha : Array F128} {A : ℕ → Array F128}
    {coef : ℕ → Array F128} {s q : ℕ}
    (hu : ∀ j < q, ∀ k < 2, (A (s + j + 1))[k]! = consTerm LANES rows[j]! (A (s + j)) (coef j) k)
    (hc : ∀ j < q, coef j = coefAt kap LANES (eqTable alpha)[j]!) {k : ℕ} (hk : k < 2) :
    ∀ j ≤ q, fOf (A (s + j))[k]! = fOf (A s)[k]! +
      ∑ l ∈ Finset.range LANES, fOf (kap[k]!)[l]! * ∑ i ∈ Finset.range j, fOf (eqTable alpha)[i]! * fOf (rows[i]!)[l]!
  | 0, _ => by simp
  | j + 1, hj => by
    rw [← Nat.add_assoc, hu j (by omega) k hk, hc j (by omega), fOf_consTerm_coefAt _ _ _ _ _ hk,
      level_sums hu hc hk j (by omega), add_assoc, ← Finset.sum_add_distrib]
    refine congrArg (_ + ·) (Finset.sum_congr rfl fun l _ => ?_)
    rw [Finset.sum_range_succ, mul_add]

/-- Over the rep's first `L` levels, the running sum's component `k` gains each level's component `k` of the final
check's consistency sums. -/
theorem chain_sums {sh : Shape} {acc : ℕ → Array F128} {o : Openings} {kappa : Array (Array (Array F128))}
    (hu : UnitSums sh acc o)
    (hc : ∀ ℓ < sh.sched.levels.size, ∀ j < o.q ℓ,
      o.coef ℓ j = coefAt (kappa[ℓ]!) (levelLanes sh ℓ) (eqTable (o.alpha ℓ))[j]!) {k : ℕ} (hk : k < 2) :
    ∀ L ≤ sh.sched.levels.size, fOf (acc (o.start L))[k]! = fOf (acc o.p0)[k]! +
      ∑ ℓ ∈ Finset.range L, ∑ l ∈ Finset.range (levelLanes sh ℓ), fOf ((kappa[ℓ]!)[k]!)[l]! *
        ∑ i ∈ Finset.range (o.q ℓ), fOf (eqTable (o.alpha ℓ))[i]! * fOf ((o.rows ℓ)[i]!)[l]!
  | 0, _ => by simp [Openings.start_zero]
  | L + 1, hL => by
    rw [Openings.start_succ, level_sums (fun j hj => hu L (by omega) j hj) (hc L (by omega)) hk _ le_rfl,
      chain_sums hu hc hk L (by omega), Finset.sum_range_succ, add_assoc]

theorem fOf_kapSum {sh : Shape} {kappa : Array (Array (Array F128))} {cons : ℕ → ℕ → F128}
    (hs : kappa.size = sh.sched.levels.size)
    (hl : ∀ ℓ < sh.sched.levels.size, ∀ r < 2, ((kappa[ℓ]!)[r]!).size = levelLanes sh ℓ) {r : ℕ} (hr : r < 2) :
    fOf (kapSum kappa r cons) = ∑ ℓ ∈ Finset.range sh.sched.levels.size,
      ∑ l ∈ Finset.range (levelLanes sh ℓ), fOf ((kappa[ℓ]!)[r]!)[l]! * fOf (cons ℓ l) := by
  rw [kapSum, fOf_rangeSum, hs]
  refine Finset.sum_congr rfl fun ℓ hℓ => ?_
  rw [fOf_rangeSum, hl ℓ (Finset.mem_range.mp hℓ) r hr]
  exact Finset.sum_congr rfl fun l _ => fOf_mul _ _

/-- **E's `hacc` at a rep, from V*'s units.** If each `rec-open` unit of the rep's openings adds its opening's
consistency terms to the chain `acc` (`UnitSums`, at the `coef` the verifier registers: `CoefsAre`), the rep's
consistency sums are the opened rows' (`ConsAre`) and V*'s algebra reads `acc0`, `acc1` at the rep's ends of the chain
(`ReadsEnds`), then the rep's running sums meet E's `hacc` (`AccHolds`). -/
theorem accHolds_of_units {sh : Shape} {d : RepData} {acc : ℕ → Array F128} {o : Openings}
    (hu : UnitSums sh acc o) (hc : CoefsAre sh d o) (hcons : ConsAre sh d o) (hrd : ReadsEnds sh d acc o) :
    AccHolds sh d := by
  intro v info kappa hrows r hr
  obtain ⟨hs, hl⟩ := verifierRows_kappa hrows
  apply fOf_injective
  rw [fOf_add, (hrd r hr).1, (hrd r hr).2, chain_sums hu (hc v info kappa hrows) hr _ le_rfl,
    fOf_kapSum hs hl hr, add_comm (fOf (acc o.p0)[r]!) _, add_assoc, CharTwo.add_self_eq_zero, add_zero]
  refine Finset.sum_congr rfl fun ℓ hℓ => Finset.sum_congr rfl fun l hl' => ?_
  rw [hcons ℓ (Finset.mem_range.mp hℓ) l (Finset.mem_range.mp hl'), consAt, fOf_rangeSum]
  exact congrArg _ (Finset.sum_congr rfl fun i _ => (fOf_mul _ _).symm)

/-! ## At every accepted outer proof -/

section outer

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open FlockVBridge.Compiled (Accepted layers AccChains)

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}

/-- **G item 4: the running sums chain into E's `hacc`.** At every accepted outer proof, with `acc` the rows of the
registration's `rec-acc` and `ops` each rep's openings, both read off V*'s layers: if every `rec-open` unit of the
rep's openings adds its opening's consistency terms (`UnitSums`, discharged unit by unit by `sound_recOpen` on the
unit's `Good` layer), the units read the verifier's `coef` (`CoefsAre`), the rep's consistency sums are the opened
rows' (`ConsAre`) and V*'s algebra reads the chain at the rep's ends (`ReadsEnds`), then `AccChains` holds. -/
theorem accChains_of_units (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Shape) (rep : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (acc : ((j : Fin S) → (Vs j).Bits) → ℕ → Array F128) (ops : ((j : Fin S) → (Vs j).Bits) → Fin 2 → Openings)
    (hu : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r,
      UnitSums sh (acc (layers zs dg N k Rw R ω t R₂ τ)) (ops (layers zs dg N k Rw R ω t R₂ τ) r))
    (hc : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r,
      CoefsAre sh (rep (layers zs dg N k Rw R ω t R₂ τ) r) (ops (layers zs dg N k Rw R ω t R₂ τ) r))
    (hcons : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r,
      ConsAre sh (rep (layers zs dg N k Rw R ω t R₂ τ) r) (ops (layers zs dg N k Rw R ω t R₂ τ) r))
    (hrd : ∀ R ω t R₂ τ o, Accepted zs dg N k Rw R ω t R₂ τ o → ∀ r,
      ReadsEnds sh (rep (layers zs dg N k Rw R ω t R₂ τ) r) (acc (layers zs dg N k Rw R ω t R₂ τ))
        (ops (layers zs dg N k Rw R ω t R₂ τ) r)) :
    AccChains zs dg N k Rw sh (fun _ _ _ => rep) := fun R ω t R₂ τ o h r =>
  accHolds_of_units (hu R ω t R₂ τ o h r) (hc R ω t R₂ τ o h r) (hcons R ω t R₂ τ o h r) (hrd R ω t R₂ τ o h r)

end outer

end FlockVBridge.AccChain
