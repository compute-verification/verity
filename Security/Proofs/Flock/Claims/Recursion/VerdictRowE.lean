import Proofs.Flock.Claims.Recursion.VerdictRows
import Proofs.Flock.VBridge.AccChain
import Proofs.Flock.VBridge.Family.Opened
import Proofs.Flock.VBridge.Family.Structure

/-!
# Row E: the running sums at a rep of the decision

`RowE` asks `AccHolds` at rep `r` of table `i` (`DecAt.rep`). VBridge's `accHolds_of_units` gives it from four facts
about one rep's openings (`Openings`): the `rec-open` units add their openings' consistency terms to the running-sum
chain (`UnitSums`), they read the verifier's coefficients (`CoefsAre`), the rep's consistency sums are the opened rows'
(`ConsAre`), and V*'s algebra reads the chain at the rep's ends (`ReadsEnds`).

* `opens`: rep `r`'s openings at the decision, from chain position `p0`: level `ℓ`'s `q` queries of the schedule, the
  rows its Ligerito proof opens there (`openAt`, as `consOf` reads them), the `α` of the verifier rows' consistency
  context for `ℓ` and the verifier's `coef` at the rows' `κ`.
* `coefsAre`: by definition.
* `consAre`: `DecAt.rep`'s `cons` is `consOf`, the opened rows' `consAt` at the context whose start is `ℓ`. That context
  exists below the schedule's levels once `verifierRows` returns (`rows_conss`, at `fast100`'s level seams: row G's
  `fast`).
* `rowE_of_units`: `UnitSums` and `ReadsEnds` at `opens` give `RowE`. Those two are what the accumulator pieces
  establish: the `rec-open` copies' `acc_out` columns (`AccUnits.unitSum_of_cols`) and the algebra unit's `acc0`,
  `acc1` ports read at the chain's ends (`AccUnits.readsEnds_of_regBits`).
-/

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin FinalP)
open FlockVBridge.AccChain (Openings UnitSums CoefsAre ConsAre ReadsEnds levelLanes accHolds_of_units)
open Flock.RecAlgebra (verifierRows consAt)
open FlockVBridge.Family (combOf)
open FlockSoundness.Refine (openAt)
open FlockSoundness.Discharge.Recursion (HClass)

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass) (acc : ℕ → Array F128) (i : ℕ) (r : Fin 2)

/-- The `α` of the verifier rows' consistency context for level `ℓ` at rep `r`'s coins, as `consOf` reads it. -/
noncomputable def alphaAt (ℓ : ℕ) : Array F128 :=
  match verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb with
  | .ok (_, info, _) =>
    match info.lg.conss.find? (·.start == ℓ) with
    | some c => c.alpha
    | none => #[]
  | .error _ => #[]

/-- The verifier's `coef` for opening `j` of level `ℓ` at rep `r`: `κ[ℓ] · eq(α)[j]` at the verifier rows' `κ`. -/
noncomputable def coefAt (ℓ j : ℕ) : Array F128 :=
  match verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb with
  | .ok (_, _, kappa) =>
    FlockVBridge.coefAt (kappa[ℓ]!) (levelLanes (D.sh cls i) ℓ) (eqTable (D.alphaAt cls acc i r ℓ))[j]!
  | .error _ => #[]

/-- **Rep `r`'s openings at the decision**, from chain position `p0`: level `ℓ`'s queries of the schedule, the rows its
Ligerito proof opens at `ℓ`, the verifier's `α` and `coef`. -/
noncomputable def opens (p0 : ℕ) : Openings where
  p0 := p0
  q ℓ := ((D.files cls i).sch.levels.getD ℓ default).q
  rows ℓ := (openAt ((D.files cls i).proof r).ligerito ((D.files cls i).sch.levels.size - 1) ℓ).1
  alpha := D.alphaAt cls acc i r
  coef := D.coefAt cls acc i r

/-- **The openings read the verifier's coefficients**, by definition of `coefAt`. -/
theorem coefsAre (p0 : ℕ) : CoefsAre (D.sh cls i) (D.rep cls acc i r) (D.opens cls acc i r p0) := by
  intro v info kappa h ℓ _ j _
  show D.coefAt cls acc i r ℓ j =
    FlockVBridge.coefAt (kappa[ℓ]!) (levelLanes (D.sh cls i) ℓ) (eqTable (D.alphaAt cls acc i r ℓ))[j]!
  unfold DecAt.coefAt
  rw [h]

/-- **The rep's consistency sums are the opened rows'** wherever `verifierRows` returns at the rep's coins: below the
schedule's levels the verifier rows hold a consistency context starting at each level (`rows_conss`), and `consOf` and
`alphaAt` both read the first one. -/
theorem consAre (hG : D.RowG cls acc i) (p0 : ℕ)
    (hfit : ∃ y, verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras
      (D.rep cls acc i r).comb = .ok y) :
    ConsAre (D.sh cls i) (D.rep cls acc i r) (D.opens cls acc i r p0) := by
  obtain ⟨⟨v, info, kappa⟩, hv⟩ := hfit
  intro ℓ hℓ l _
  have hf : Flock.fast100 (D.files cls i).st.m = some (D.sh cls i).sched := hG.fast
  have h0 := (FlockVBridge.Family.fast100_q0 hf).1
  have hseam := FlockSoundness.Refine.fast100_level _ _ hf
  have hq : ∀ k < (D.sh cls i).sched.levels.size, ∀ c,
      (Flock.positions ((D.sh cls i).sched.levels[k]!) c).size = ((D.sh cls i).sched.levels[k]!).q := fun k hk c =>
    FlockSoundness.Refine.positions_size _ (hseam _ (by rw [getElem!_pos _ k hk]; exact Array.getElem_mem hk)) c
  have E := FlockVBridge.Family.rows_conss hv h0 hq
  have hmem : (ℓ, ((D.sh cls i).sched.levels[ℓ]!).q) ∈ info.lg.conss.toList.map (fun c => (c.start, c.pos.size)) := by
    rw [E]
    exact List.mem_map.2 ⟨ℓ, List.mem_range.2 hℓ, rfl⟩
  obtain ⟨c₀, hc₀, hs₀⟩ := List.mem_map.1 hmem
  have hsome : (info.lg.conss.find? (·.start == ℓ)).isSome := by
    rw [Array.find?_isSome]
    exact ⟨c₀, Array.mem_toList_iff.1 hc₀, by simp [(Prod.mk.inj hs₀).1]⟩
  obtain ⟨c, hc⟩ := Option.isSome_iff_exists.1 hsome
  have hv' : verifierRows (D.sh cls i) (FlockVBridge.Family.roundsOf (D.files cls i) r) (D.files cls i).extras
      (combOf (D.sh cls i) (D.files cls i) (FlockVBridge.Family.roundsOf (D.files cls i) r)) =
        .ok (v, info, kappa) := hv
  show consOf (D.sh cls i) (D.files cls i).sch ((D.files cls i).proof r)
      (FlockVBridge.Family.roundsOf (D.files cls i) r) (D.files cls i).extras
      (combOf (D.sh cls i) (D.files cls i) (FlockVBridge.Family.roundsOf (D.files cls i) r)) ℓ l =
    consAt (openAt ((D.files cls i).proof r).ligerito ((D.files cls i).sch.levels.size - 1) ℓ).1
      ((D.files cls i).sch.levels.getD ℓ default).q (D.alphaAt cls acc i r ℓ) l
  have ha : D.alphaAt cls acc i r ℓ = c.alpha := by
    unfold alphaAt
    simp only [hv, hc]
  rw [ha]
  simp only [consOf, hv', hc]

/-- **Row E from the openings' units and ends**: at row G's setup, wherever `verifierRows` returns at rep `r`'s coins,
the `rec-open` units adding their openings' consistency terms to a chain `accC` (`UnitSums`) and V*'s algebra reading
`accC` at the rep's ends (`ReadsEnds`), both at `opens`, give `RowE`. -/
theorem rowE_of_units (hG : D.RowG cls acc i) {accC : ℕ → Array F128} {p0 : ℕ}
    (hfit : ∃ y, verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras
      (D.rep cls acc i r).comb = .ok y)
    (hu : UnitSums (D.sh cls i) accC (D.opens cls acc i r p0))
    (hrd : ReadsEnds (D.sh cls i) (D.rep cls acc i r) accC (D.opens cls acc i r p0)) :
    D.RowE cls acc i r :=
  accHolds_of_units hu (D.coefsAre cls acc i r p0) (D.consAre cls acc i r hG p0 hfit) hrd

end DecAt

end RecDeploy.Rows
