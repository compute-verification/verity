import Proofs.Flock.Soundness.Binding.E2E
import Proofs.Flock.Soundness.Registered.Binding

/-!
# The end-to-end theorem with registered reads

`flock_e2e_drawn_hm96` and `flock_e2e_count_hm96` (`Binding.E2E`) take the layout's registered leaves (`Layout.regLeaf`) as
what the value binding commits to. At a commit string whose row the session reads from a registered value, that leaf is
the session's own, checked against the registered root the verifier holds (`Flock.Registered.check_ok`), and the value
behind it should be the registrant's. These forms add the event that some registered read's leaf, under the prover's
registration, is not the registrant's (`Reads.Off`), at `qR²/2^513` under `SHA512CRStrict` (`cr/sha-512`) for the
registered-path finder (`hR`). Off that event, the binding's commitment at every registered read is the registrant's
leaf, so the committed values there are the rows the registrant committed (`Reads.registered_weights`).

They take, beyond the forms they extend, the layout's registered reads (`rd : Reads`) and `hchk`: on every outcome the
executable accepts, the registered check held at the prover's registration, through that outcome's paths. `hchk` is
`check_ok`'s conclusion read at the model's leaves, the same link from the executable's openings to the model's (R11) that
`bc_registered` and `tr` name.
-/

namespace FlockSoundness.Registered

open Game Model Accounting Extract LinkFinder Refine Audit Audit.Partition Binding
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : Law n}

/-- An event on the audit's outcome, or `Off`, splits into the two. -/
theorem prob_or_off_le {β : Type} (G : Game β) (S : Strategy G) (A B : β → Prop) (Off : Prop) :
    prob (fun o => A o ∧ (B o ∨ Off)) G S ≤ prob (fun o => A o ∧ B o) G S + prob (fun o => A o ∧ Off) G S :=
  (prob_mono (q := fun o => (A o ∧ B o) ∨ (A o ∧ Off)) (fun _ ⟨ha, hb⟩ => hb.imp (⟨ha, ·⟩) (⟨ha, ·⟩)) G S).trans
    (prob_or_le _ _ G S)

/-- **The end-to-end drawn-unit bound with registered reads**: `flock_e2e_drawn_hm96`, with the event that some
registered read's leaf under the prover's registration is not the registrant's, at most
`ε_ks + δ_link + δ_tree + qR²/2^513`: the last under `SHA512CRStrict` of SHA-512 (`cr/sha-512`) for the registered-path
finder (`hR`). -/
theorem flock_e2e_drawn_hm96_reads
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRows lay) (hHm : rs.Computes)
    (rd : Reads lay rs)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H E (plan (L.draw ω) (reg σ)) j
      (batchStrat H E plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit L Reg (batchedSession H E plan)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT)
    (path : Bool × Finset (Fin n) → lay.Pos → ByteArray) (hchk : ∀ o, execAccept o → rd.Checked (reg σ) (path o))
    {qR : ℝ} (hR : (StrictCR.oneRun (audit L Reg (batchedSession H E plan)) σ fun o => rd.pick (reg σ) (path o)).CR
      H512 qR) :
    prob (fun o => execAccept o ∧
        (¬ Disjoint (P.wrong (Xpub ones zeros (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))) o.2 ∨
          rd.Off (reg σ)))
        (audit L Reg (batchedSession H E plan)) σ ≤
      StrictCR.ksAvgStrict plan L k qF qS (reg σ) + P.linkBoundE (L := L) plan tab (rs.binding hHm) k Rw M ρ t' +
        ENNReal.ofReal (tr.bound qT) + ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) :=
  (prob_or_off_le _ σ execAccept _ _).trans (add_le_add
    (flock_e2e_drawn_hm96 P H E plan tab dp ones hConst zeros hZero lay rs hHm hk hRw hM hρ hr ht σ hCR hKS tr hT)
    (rd.prob_off_le _ σ (reg σ) execAccept path hchk hR))

/-- **The end-to-end count curve with registered reads**: `flock_e2e_count_hm96`, with the event that some registered
read's leaf under the prover's registration is not the registrant's, at most
`miss_L(K) + ε_ks + δ_link + δ_tree + qR²/2^513`, as in `flock_e2e_drawn_hm96_reads`. -/
theorem flock_e2e_count_hm96_reads
    {execAccept : Bool × Finset (Fin n) → Prop}
    {unitRows : Fin n → FlockSoundness.Rows} (dp : P.DerivedPlaces plan tab unitRows)
    (ones : Finset (Fin C.N)) (hConst : P.ConstCols plan tab dp.place ones)
    (zeros : Finset (Fin C.N)) (hZero : P.ZeroCols plan tab dp.place zeros)
    (lay : Layout P plan tab (P.placeDecoder plan tab dp.place)) (rs : HmRows lay) (hHm : rs.Computes)
    (rd : Reads lay rs)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, StrictCR.TableCR execArith H E (plan (L.draw ω) (reg σ)) j
      (batchStrat H E plan (cont σ ω)) qF qS)
    (tr : StrictCR.TreeRewind H (audit L Reg (batchedSession H E plan)) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR H qT)
    (path : Bool × Finset (Fin n) → lay.Pos → ByteArray) (hchk : ∀ o, execAccept o → rd.Checked (reg σ) (path o))
    {qR : ℝ} (hR : (StrictCR.oneRun (audit L Reg (batchedSession H E plan)) σ fun o => rd.pick (reg σ) (path o)).CR
      H512 qR) :
    prob (fun o => execAccept o ∧
        (K ≤ (P.wrong (Xpub ones zeros (P.Xplur H E plan tab (rs.binding hHm) (reg σ) (cont σ) k Rw))).card ∨
          rd.Off (reg σ)))
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + StrictCR.ksAvgStrict plan L k qF qS (reg σ) +
        P.linkBoundE (L := L) plan tab (rs.binding hHm) k Rw M ρ t' + ENNReal.ofReal (tr.bound qT) +
        ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) :=
  (prob_or_off_le _ σ execAccept _ _).trans (add_le_add
    (flock_e2e_count_hm96 P H E plan tab dp ones hConst zeros hZero lay rs hHm hk hRw hM hρ hr ht K σ hCR hKS tr hT)
    (rd.prob_off_le _ σ (reg σ) execAccept path hchk hR))

end FlockSoundness.Registered
