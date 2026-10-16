import Proofs.Flock.Claims.KS.Rounds
import Proofs.Flock.Claims.Bridge.LivePlain

/-!
# One table's salted session after `Commit`, under a live strategy

The old salted transfer (`live_le_sessionBLJ`) bounds a live verdict by the salted session `sessionBL` on the tables'
model specs; the old session theorems (`session_sound_of_table_accL`, `epsSL`) run after `Commit`, on `sessAfterL`. At
one table they meet: `sessionBL [t]` is `Commit`'s cap, then `sessAfterL` with no other game (`sessAfterBL_one`), and
its acceptance is the table's (`AccOne`, `prob_sessionBL_one`, at the strategy after `Commit`, `afterOne`). So a live
verdict is no likelier than the table's acceptance after `Commit` against the simulated strategy (`live_le_afterOne`),
and at a table `T` the model does not commit to, no likelier than that strategy's `ε_c⁻` and its probability of
accepting while opening a level-0 position off `T` (`live_le_epsSL`). Averaged over a mixture of live strategies
(`mean_live_le`), each term is the mean over the mixture's strategies after `Commit`.
-/

namespace FlockDeploy.KS

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables

/-- **One table's salted session after `Commit`** is `sessAfterL` with no other game. -/
theorem sessAfterBL_one {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
    (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List FlockLevel3.GF128) D)
    (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts) (caps : Fin [t].length → Model.Cap D) :
    sessAfterBL (gamesBL A Hc E Lf [t] caps) =
      sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] (caps ⟨0, Nat.one_pos⟩) := by
  unfold sessAfterBL gamesBL sessAfterL
  congr 1

/-- The strategy after `Commit` of a strategy of one table's salted session. -/
noncomputable def afterOne {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
    (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List FlockLevel3.GF128) D)
    (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts) (τ : Strategy (sessionBL A Hc E Lf [t])) :
    Strategy (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] (τ.1 ⟨0, Nat.one_pos⟩)) :=
  cast (congrArg Strategy (sessAfterBL_one A Hc E Lf t τ.1)) τ.2

/-- One table's acceptance: the output at its position is accepted. -/
def AccOne {D Sl : Type} [Inhabited D] [Inhabited Sl] (outs : List (Model.OutCL FlockLevel3.GF128 D Sl)) : Prop :=
  (outs.getD 0 dfltOutL).accepted = true

theorem expect_cast {β : Type} {g g' : Game β} (h : g = g') (f : β → ℝ) (s : Strategy g) :
    expect f g' (cast (congrArg Strategy h) s) = expect f g s := by
  subst h
  rfl

open Classical in
/-- **One table's salted session accepts** as the table does after `Commit`. -/
theorem prob_sessionBL_one {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List FlockLevel3.GF128) D)
    (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts) (τ : Strategy (sessionBL A Hc E Lf [t])) :
    (prob (acceptedBL [t]) (sessionBL A Hc E Lf [t]) τ).toReal =
      expect (ind AccOne) (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] (τ.1 ⟨0, Nat.one_pos⟩))
        (afterOne A Hc E Lf t τ) := by
  have hacc : acceptedBL (F := FlockLevel3.GF128) (D := D) (Sl := Sl) [t] = AccOne := by
    funext outs
    apply propext
    unfold acceptedBL AccOne
    refine ⟨fun h => h ⟨0, Nat.one_pos⟩, fun h i => ?_⟩
    have hi : i.val = 0 := by
      have h := i.isLt
      simp only [List.length_cons, List.length_nil] at h
      omega
    rwa [show i = ⟨0, Nat.one_pos⟩ from Fin.ext hi]
  rw [hacc, prob_toReal]
  exact (expect_cast (sessAfterBL_one A Hc E Lf t τ.1) (ind AccOne) τ.2).symm

open Classical in
/-- **A live verdict is no likelier than one table's acceptance after `Commit`** against the simulated strategy, when
the table's decoder is right on the outcomes the verdict accepts (`live_le_sessionBLJ`). -/
theorem live_le_afterOne {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D)
    (E : Merkle.Enc (List FlockLevel3.GF128) D) (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts)
    (pL : LiveOutJ [t].length mPts → Prop)
    (opn : Array ByteArray → Fin [t].length → Model.Opens FlockLevel3.GF128 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin [t].length → ℕ → ByteArray → List Msg) (one : Fin [t].length → ℕ → Bool)
    (fin : Fin [t].length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin [t].length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => o.accepted = true) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A ([t].get j)) j)
    (Pr : Strategy (liveSession dg [t].length mPts N)) :
    (prob pL (liveSession dg [t].length mPts N) Pr).toReal ≤
      expect (ind AccOne) (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] []
          ((simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec Pr).1 ⟨0, Nat.one_pos⟩))
        (afterOne A Hc E Lf t (simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec Pr)) := by
  rw [← prob_sessionBL_one]
  exact ENNReal.toReal_mono (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))
    (live_le_sessionBLJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec Pr)

open Classical in
/-- **A live verdict at one table the model does not commit to** is no likelier than the simulated strategy's `ε_c⁻`
after `Commit` and its probability of accepting while opening a level-0 position off the table
(`session_sound_of_table_accL`). -/
theorem live_le_epsSL {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (hA : A.Correct) (Hc : List UInt8 → D)
    (E : Merkle.Enc (List FlockLevel3.GF128) D) (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts)
    (pL : LiveOutJ [t].length mPts → Prop)
    (opn : Array ByteArray → Fin [t].length → Model.Opens FlockLevel3.GF128 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin [t].length → ℕ → ByteArray → List Msg) (one : Fin [t].length → ℕ → Bool)
    (fin : Fin [t].length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin [t].length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => o.accepted = true) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A ([t].get j)) j)
    (T : Model.Oracle FlockLevel3.GF128) (hT : ¬ Committed A t.S t.sch T)
    (Pr : Strategy (liveSession dg [t].length mPts N)) :
    (prob pL (liveSession dg [t].length mPts N) Pr).toReal ≤
      epsSL A Hc E Lf t.S t.m t.k0 t.l₀ t.l₁ t.rest t.ext (FlockSoundness.fast100_shape t.S.m _ t.hsch).2.1 t.hm
          t.ptLocal mPts [] [] ((simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec Pr).1 ⟨0, Nat.one_pos⟩)
          (afterOne A Hc E Lf t (simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec Pr)) +
        expect (ind fun outs => AccOne outs ∧ Off₀L Hc E Lf t.sch
            ((simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec Pr).1 ⟨0, Nat.one_pos⟩) T
            (outs.getD 0 dfltOutL))
          (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] []
            ((simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec Pr).1 ⟨0, Nat.one_pos⟩))
          (afterOne A Hc E Lf t (simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec Pr)) := by
  refine le_trans (live_le_afterOne A Hc E Lf t pL opn dg dec one fin capOf N hdec Pr) ?_
  have h := session_sound_of_table_accL A hA Hc E Lf t.S t.m t.k0 t.l₀ t.l₁ t.rest t.ext t.hsch t.hm t.ptLocal mPts
    t.hlay [] [] ((simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec Pr).1 ⟨0, Nat.one_pos⟩) AccOne
    (fun _ h => h) T (afterOne A Hc E Lf t (simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec Pr))
  have e : (ind fun outs : List (Model.OutCL FlockLevel3.GF128 D Sl) => AccOne outs ∧
      ¬ Committed A t.S t.sch T) = ind AccOne := by
    funext outs
    simp only [ind, hT, not_false_eq_true, and_true]
  rw [e] at h
  exact h

open Classical in
/-- **Averaged over a mixture of live strategies**, at one table the model does not commit to: the mean live verdict is
at most the mean `ε_c⁻` and the mean probability of accepting off the table, over the simulated strategies after
`Commit`. -/
theorem mean_live_le {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (hA : A.Correct) (Hc : List UInt8 → D)
    (E : Merkle.Enc (List FlockLevel3.GF128) D) (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts)
    (pL : LiveOutJ [t].length mPts → Prop)
    (opn : Array ByteArray → Fin [t].length → Model.Opens FlockLevel3.GF128 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin [t].length → ℕ → ByteArray → List Msg) (one : Fin [t].length → ℕ → Bool)
    (fin : Fin [t].length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin [t].length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => o.accepted = true) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A ([t].get j)) j)
    (T : Model.Oracle FlockLevel3.GF128) (hT : ¬ Committed A t.S t.sch T) {τ : Type} [Fintype τ]
    (σL : τ → Strategy (liveSession dg [t].length mPts N)) :
    (∑ u, (prob pL (liveSession dg [t].length mPts N) (σL u)).toReal) / Fintype.card τ ≤
      (∑ u, epsSL A Hc E Lf t.S t.m t.k0 t.l₀ t.l₁ t.rest t.ext (FlockSoundness.fast100_shape t.S.m _ t.hsch).2.1 t.hm
          t.ptLocal mPts [] []
          ((simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec (σL u)).1 ⟨0, Nat.one_pos⟩)
          (afterOne A Hc E Lf t (simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec (σL u)))) /
          Fintype.card τ +
        (∑ u, expect (ind fun outs => AccOne outs ∧ Off₀L Hc E Lf t.sch
            ((simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec (σL u)).1 ⟨0, Nat.one_pos⟩) T
            (outs.getD 0 dfltOutL))
          (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] []
            ((simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec (σL u)).1 ⟨0, Nat.one_pos⟩))
          (afterOne A Hc E Lf t (simSessJ A Hc E Lf [t] pL opn dg dec one fin capOf N hdec (σL u)))) /
          Fintype.card τ := by
  rw [← add_div, ← Finset.sum_add_distrib]
  exact div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ =>
    live_le_epsSL A hA Hc E Lf t pL opn dg dec one fin capOf N hdec T hT (σL u)) (Nat.cast_nonneg _)

end FlockDeploy.KS
