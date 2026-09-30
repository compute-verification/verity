import FlockSoundness.E2E
import FlockSoundness.ExecStratified
import FlockSoundness.Binding.Weights

/-!
# The end-to-end theorem with the value binding of `hm96-sha512/row/v1` rows

`flock_e2e_count` and `flock_e2e_drawn` (`E2E.lean`), and their forms at the executable's draw law
(`ExecStratified.lean`), with `vb : ValueBinding` built from the serving rows (`HmRows.binding`). The binding's hash is
SHA-512 (`Refine.H512`), so A2 (`hCR`) is about SHA-512 itself, and what `vb` carried becomes:
* `HmRowComputes` (`hHm`), the circuit's row hashing (phase 2i), the one gap of the binding;
* the layout's facts, in `lay : Layout` (`decode_row`: the decoder reads the rows the witness holds) and `rs : HmRows`
  (`bc_registered`: the `b ‖ c` the witness holds hashes to the registered leaf; `other_registered`: the returned
  outputs' leaves).

`collide_spec` is proved (`ExplicitLeaf.hm96`): the link finder's collisions are explicit SHA-512 collisions. The other
hypotheses (`hExec`, `dp`, `hL1`, `hOne`) are the skeleton's, unchanged (`assumptions/e2e-checklist.md`).

The bounds are at the registered leaves (`Layout.regLeaf`). An audit bound read from registered roots adds the roots'
Merkle binding: `δ_tree`, under SHA-512 collision resistance (`cr/sha-512`; `DESIGN.md` §1).
-/

namespace FlockSoundness.Binding

open Game Model Accounting Extract LinkFinder Refine Audit Audit.Partition
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : Law n}

/-- **The end-to-end count curve, with the rows' value binding**: `flock_e2e_count` at `vb := HmRows.binding rs hHm`.
If `execAccept` holds, the probability that at least `K` units of the committed circuit are wrong at the committed values
is at most `miss_L(K) + ε_ks(σ) + δ_link`. Beyond the skeleton's `hExec`, `dp`, `hL1` and `hOne`, it takes the rows'
layout (`lay`, `rs`) and `HmRowComputes` (`hHm`); A2 (`hCR`, for SHA-512) is the assumption of record. -/
theorem flock_e2e_count_hm96
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    {CB : Circuit} (PB : Partition CB n) (proj : (Fin C.N → Bool) → Fin CB.N → Bool) (ones : Finset (Fin C.N))
    (hL1 : P.RowsL1 PB proj ones)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (hCR : ∀ (R : Reg) (τ : Cont L Reg (batchedSession H E plan) R) (c : lay.Pos)
      (_ : Nonempty (P.Ωc plan tab (rs.binding hHm) L c)), Assumptions.SHA512ExpectedTimeCR H512
        (finderG (P.trialG H E plan tab (rs.binding hHm) R τ c k Rw) M)
        (finderStrat _ (P.trialStrat H E plan tab (rs.binding hHm) R τ c k Rw) M)
        (P.outC H E plan tab (rs.binding hHm) R τ c)
        (finderCost (P.rdT H E plan tab (rs.binding hHm) R τ c) (cT plan R t')))
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hOne : ∀ g ∈ ones, P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw g = true) :
    prob (fun o => execAccept o ∧
        K ≤ (PB.wrong (proj (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) +
        P.linkBoundE (L := L) plan tab (rs.binding hHm) k Rw M ρ t' :=
  P.flock_e2e_count H E plan tab hExec dp PB proj ones hL1 (rs.binding hHm) hk hRw hM hρ hr ht hCR K σ hOne

/-- **The end-to-end drawn-unit bound, with the rows' value binding**: `flock_e2e_drawn` at
`vb := HmRows.binding rs hHm`. -/
theorem flock_e2e_drawn_hm96
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    {CB : Circuit} (PB : Partition CB n) (proj : (Fin C.N → Bool) → Fin CB.N → Bool) (ones : Finset (Fin C.N))
    (hL1 : P.RowsL1 PB proj ones)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (hCR : ∀ (R : Reg) (τ : Cont L Reg (batchedSession H E plan) R) (c : lay.Pos)
      (_ : Nonempty (P.Ωc plan tab (rs.binding hHm) L c)), Assumptions.SHA512ExpectedTimeCR H512
        (finderG (P.trialG H E plan tab (rs.binding hHm) R τ c k Rw) M)
        (finderStrat _ (P.trialStrat H E plan tab (rs.binding hHm) R τ c k Rw) M)
        (P.outC H E plan tab (rs.binding hHm) R τ c)
        (finderCost (P.rdT H E plan tab (rs.binding hHm) R τ c) (cT plan R t')))
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hOne : ∀ g ∈ ones, P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw g = true) :
    prob (fun o => execAccept o ∧
        ¬ Disjoint (PB.wrong (proj (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab (rs.binding hHm) k Rw M ρ t' :=
  P.flock_e2e_drawn H E plan tab hExec dp PB proj ones hL1 (rs.binding hHm) hk hRw hM hρ hr ht hCR σ hOne

variable {m : ℕ} (St : Fin m → Flock.Draw.Stratum) (Ls : Fin m → ℕ)

/-- **The e2e drawn-unit bound at the executable's law, with the rows' value binding**: `flock_e2e_drawn_exec` at
`vb := HmRows.binding rs hHm`. -/
theorem flock_e2e_drawn_exec_hm96
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    {CB : Circuit} (PB : Partition CB n) (proj : (Fin C.N → Bool) → Fin CB.N → Bool) (ones : Finset (Fin C.N))
    (hL1 : P.RowsL1 PB proj ones)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (hCR : ∀ (R : Reg) (τ : Cont (Law.execStratified St Ls) Reg (batchedSession H E plan) R) (c : lay.Pos)
      (_ : Nonempty (P.Ωc plan tab (rs.binding hHm) (Law.execStratified St Ls) c)),
      Assumptions.SHA512ExpectedTimeCR H512
        (finderG (P.trialG H E plan tab (rs.binding hHm) R τ c k Rw) M)
        (finderStrat _ (P.trialStrat H E plan tab (rs.binding hHm) R τ c k Rw) M)
        (P.outC H E plan tab (rs.binding hHm) R τ c)
        (finderCost (P.rdT H E plan tab (rs.binding hHm) R τ c) (cT plan R t')))
    (σ : Strategy (audit (Law.execStratified St Ls) Reg (batchedSession H E plan)))
    (hOne : ∀ g ∈ ones, P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw g = true) :
    prob (fun o => execAccept o ∧
        ¬ Disjoint (PB.wrong (proj (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))) o.2)
        (audit (Law.execStratified St Ls) Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) +
        P.linkBoundE (L := Law.execStratified St Ls) plan tab (rs.binding hHm) k Rw M ρ t' :=
  P.flock_e2e_drawn_exec H E plan tab St Ls hExec dp PB proj ones hL1 (rs.binding hHm) hk hRw hM hρ hr ht hCR σ hOne

/-- **The e2e count curve at the executable's law, with the rows' value binding**: `flock_e2e_count_exec` at
`vb := HmRows.binding rs hHm`, its count curve bounded by `Law.stratified`'s `miss`. -/
theorem flock_e2e_count_exec_hm96 (σs : Fin n → Fin m) (ks : Fin m → ℕ)
    (hks : ∀ s, ks s ≤ (Law.stratum σs s).card) (hS : Law.ExecStrata σs ks St)
    {execAccept : Bool × Finset (Fin n) → Prop} (hExec : ∀ o, execAccept o → o.1 = true)
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    {CB : Circuit} (PB : Partition CB n) (proj : (Fin C.N → Bool) → Fin CB.N → Bool) (ones : Finset (Fin C.N))
    (hL1 : P.RowsL1 PB proj ones)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (hCR : ∀ (R : Reg) (τ : Cont (Law.execStratified St Ls) Reg (batchedSession H E plan) R) (c : lay.Pos)
      (_ : Nonempty (P.Ωc plan tab (rs.binding hHm) (Law.execStratified St Ls) c)),
      Assumptions.SHA512ExpectedTimeCR H512
        (finderG (P.trialG H E plan tab (rs.binding hHm) R τ c k Rw) M)
        (finderStrat _ (P.trialStrat H E plan tab (rs.binding hHm) R τ c k Rw) M)
        (P.outC H E plan tab (rs.binding hHm) R τ c)
        (finderCost (P.rdT H E plan tab (rs.binding hHm) R τ c) (cT plan R t')))
    (K : ℕ) (σ : Strategy (audit (Law.execStratified St Ls) Reg (batchedSession H E plan)))
    (hOne : ∀ g ∈ ones, P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw g = true) :
    prob (fun o => execAccept o ∧
        K ≤ (PB.wrong (proj (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card)
        (audit (Law.execStratified St Ls) Reg (batchedSession H E plan)) σ ≤
      (Law.stratified σs ks hks).miss K + ksAvgBE H E plan k (reg σ) (cont σ) +
        P.linkBoundE (L := Law.execStratified St Ls) plan tab (rs.binding hHm) k Rw M ρ t' :=
  P.flock_e2e_count_exec H E plan tab St Ls σs ks hks hS hExec dp PB proj ones hL1 (rs.binding hHm) hk hRw hM hρ hr
    ht hCR K σ hOne

end FlockSoundness.Binding
