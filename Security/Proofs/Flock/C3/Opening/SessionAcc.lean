import Proofs.Flock.C3.Opening.Rows
import Proofs.Flock.VBridge.AccUnits

/-!
# n4's session (c): the running sums from the `rec-open-stacked` units

R's opening's running sums `acc0`, `acc1` are the chain's rows at R's ends, and each `rec-open-stacked` unit adds its
opening's consistency terms at the `coef` the verifier registers (`RecCoins.checkStackedLevel`:
`coefAt (κ[ℓ]) LANES eq(α_ℓ)[j]`, at the `kappa` of R's opening). As in the full bridge (`accHolds_of_units`), the chain
telescopes: in characteristic 2 the sums before and after add to the levels' consistency sums weighted by `kappa`,
`openingRead_of_rows`' `hacc`.

* `kappa_shapeO`: the passes' `kappa` has a row per Ligerito level and `2 ^ lanes` entries per component.
* `consO`: the levels' consistency sums of the opened rows (`consAt`).
* `hacc_of_units`: `hacc` at `consO`, from `UnitSums`, the registered `coef` and the chain's ends.
-/

namespace FlockC3.Opening

open Flock Flock.RecAlgebra
open FlockSoundness.Refine (fOf fOf_add fOf_mul fOf_injective)
open FlockVBridge.AccChain

/-- `RecCoins`' `coefAt` (what V* registers) is the proofs' (`FlockVBridge.coefAt`). -/
theorem coefAt_eq (kap : Array (Array F128)) (LANES : ℕ) (e : F128) :
    RecCoins.coefAt kap LANES e = FlockVBridge.coefAt kap LANES e := by
  have h : ∀ k : ℕ, (Array.range LANES).map (fun l => (kap[k]!)[l]! * e) =
      Array.ofFn (n := LANES) (fun l => (kap[k]!)[l.1]! * e) := fun k => by
    apply Array.ext
    · simp
    · intro i h1 h2
      simp
  unfold RecCoins.coefAt FlockVBridge.coefAt
  rw [h 0, h 1]

section Shape

variable {sh : Shape}

theorem accTerms_shapeO {ri : ℕ} {l : List (Key × F128)} {st st' : BSt} {acc : Array Term}
    (hr : RecOpening.accTerms sh ri l st = .ok (acc, st')) : SameShape st.kappa st'.kappa := by
  simp only [RecOpening.accTerms] at hr
  split at hr
  · cases hr
    exact SameShape.refl _
  · split at hr
    · cases hr
    · obtain ⟨_, st1, hk, hr⟩ := bm_bind hr
      have hs := kapAdd_shape _ l st st1 hk
      cases h1 : RecOpening.wi sh (.acc1 (ri - sh.claims)) with
      | none => simp only [h1] at hr; cases hr
      | some i1 =>
        cases h0 : RecOpening.wi sh (.acc0 (ri - sh.claims)) with
        | none => simp only [h1, h0] at hr; cases hr
        | some i0 =>
          simp only [h1, h0] at hr
          cases hr
          exact hs

theorem buildRes_shapeO {ri : ℕ} {f : Lin} {st st' : BSt} {res : Res} (hi : OpsOkO sh st)
    (hr : RecOpening.buildRes sh ri f st = .ok (res, st')) : SameShape st.kappa st'.kappa := by
  unfold RecOpening.buildRes at hr
  obtain ⟨acc, st1, h1, hr⟩ := bm_bind hr
  obtain ⟨a1, a2, a3, a4, -, -, -⟩ := accTerms_specO #[] (fun _ _ => 0) h1
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
  have hi1 : OpsOkO sh st1 := hi.of_eq a1 a2 a3 a4
  obtain ⟨h2i, -, h2k, -⟩ := coefOps_specO #[] (fun _ _ => 0) ri l1 st1 st2 cm hi1 h2
  obtain ⟨h3i, -, h3k, -⟩ := prodOps_specO #[] (fun _ _ => 0) ri l2 st2 st3 pm h2i h3
  obtain ⟨-, -, h4k, -⟩ := constTerm_specO #[] h3i h4
  rw [h4k, h3k, h2k]
  exact accTerms_shapeO h1

theorem buildAll_shapeO : ∀ (fs : List Lin) (ri : ℕ) (st st' : BSt) (rs : Array Res),
    OpsOkO sh st → RecOpening.buildAll sh fs ri st = .ok (rs, st') → SameShape st.kappa st'.kappa
  | [], ri, st, st', rs, hi, hr => by
    cases hr
    exact SameShape.refl _
  | f :: rest, ri, st, st', rs, hi, hr => by
    obtain ⟨r, st1, h1, hr⟩ := bm_bind hr
    obtain ⟨rs', st2, h2, hr⟩ := bm_bind hr
    cases hr
    exact (buildRes_shapeO hi h1).trans
      (buildAll_shapeO rest (ri + 1) st1 st' rs' (buildRes_specO #[] (fun _ _ => 0) hi h1).1 h2)

/-- **The passes' `kappa` has the schedule's shape**: a row per Ligerito level, each of its two components the
level's `2 ^ lanes` entries, as `kappa0` starts it. -/
theorem kappa_shapeO {lins : Array Lin} {res : Array Res} {bs : BSt}
    (hb : (RecOpening.buildAll sh lins.toList 0).run { kappa := kappa0 sh } = .ok (res, bs)) :
    bs.kappa.size = sh.sched.levels.size ∧
      ∀ ℓ < sh.sched.levels.size, ∀ r < 2, ((bs.kappa[ℓ]!)[r]!).size = levelLanes sh ℓ := by
  have hB : RecOpening.buildAll sh lins.toList 0 { kappa := kappa0 sh } = .ok (res, bs) := hb
  have hinit : OpsOkO sh { kappa := kappa0 sh } := ⟨rfl, fun _ _ hk => by cases hk⟩
  obtain ⟨hs, hr⟩ := buildAll_shapeO lins.toList 0 _ _ _ hinit hB
  refine ⟨by rw [hs]; simp [kappa0], fun ℓ hℓ r hr2 => ?_⟩
  rw [hr ℓ r, levelLanes]
  simp [kappa0, getElem!_def, Array.getElem?_map, Array.getElem?_eq_getElem hℓ, hr2]

end Shape

/-- The levels' consistency sums of the opened rows: lane `l` of level `ℓ` is `Σ_j eq(α_ℓ)[j]·row_j[l]` (`consAt`). -/
def consO (o : Openings) (ℓ l : ℕ) : F128 := consAt (o.rows ℓ) (o.q ℓ) (o.alpha ℓ) l

/-- **(n4 (c)) `hacc` from the `rec-open-stacked` units.** At the passes' `kappa`, if each unit of R's openings adds
its opening's consistency terms to the chain `acc` (`UnitSums`) at the `coef` the verifier registers
(`RecCoins.coefAt` of `kappa`), and w's `acc0`, `acc1` are the chain's rows at R's ends, then w's running sums differ
by the opened rows' consistency sums (`consO`) weighted by `kappa`. -/
theorem hacc_of_units {sh : Shape} {lins : Array Lin} {res : Array Res} {bs : BSt}
    (hb : (RecOpening.buildAll sh lins.toList 0).run { kappa := kappa0 sh } = .ok (res, bs))
    {w : Array F128} {acc : ℕ → Array F128} {o : Openings} (hu : UnitSums sh acc o)
    (hc : ∀ ℓ < sh.sched.levels.size, ∀ j < o.q ℓ,
      o.coef ℓ j = RecCoins.coefAt (bs.kappa[ℓ]!) (levelLanes sh ℓ) (eqTable (o.alpha ℓ))[j]!)
    (hrd : ∀ r < 2, valO sh w (consO o) (.acc0 r) = (acc o.p0)[r]! ∧
      valO sh w (consO o) (.acc1 r) = (acc (o.start sh.sched.levels.size))[r]!) :
    ∀ r < 2, valO sh w (consO o) (.acc1 r) + valO sh w (consO o) (.acc0 r) = kapSum bs.kappa r (consO o) := by
  intro r hr
  obtain ⟨hs, hl⟩ := kappa_shapeO hb
  apply fOf_injective
  rw [fOf_add, (hrd r hr).1, (hrd r hr).2,
    chain_sums hu (fun ℓ hℓ j hj => (hc ℓ hℓ j hj).trans (coefAt_eq _ _ _)) hr _ le_rfl,
    fOf_kapSum hs hl hr, add_comm (fOf (acc o.p0)[r]!) _, add_assoc, CharTwo.add_self_eq_zero, add_zero]
  refine Finset.sum_congr rfl fun ℓ _ => Finset.sum_congr rfl fun l _ => ?_
  rw [consO, consAt, fOf_rangeSum]
  exact congrArg _ (Finset.sum_congr rfl fun i _ => (fOf_mul _ _).symm)

end FlockC3.Opening
