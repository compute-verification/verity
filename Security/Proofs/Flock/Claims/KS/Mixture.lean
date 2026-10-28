import Proofs.Flock.Claims.KS.Rounds
import Proofs.Flock.Claims.Bridge.Dead
import Proofs.Flock.Claims.Bridge.Live

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

The same holds at any verdict `ok` on the table's output that the decoder is right for (`simSessG`,
`live_le_sessionBLG`, `prob_sessionBL_oneG`), with the simulation's dead branches, past which no live run is accepted,
sending empty openings (`nulOp`, `Claims.Bridge.Sim.mapDead`). The simulated `Commit` sends the cap the live roots
decode to (`simSessG_fst`), so over a mixture of live strategies whose roots agree, the strategies after `Commit` share
one cap (`simAt`, `live_le_simAt`, `mean_live_le_simAt`).
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

theorem fst_rec_send {β M : Type} {k k' : M → Game β} (h : Game.send M k = Game.send M k')
    (s : Strategy (Game.send M k)) : (h ▸ s : Strategy (Game.send M k')).1 = s.1 := by
  obtain rfl : k = k' := eq_of_heq (Game.send.inj h).2
  rfl

/-- **Empty openings**: every slot's row and path empty, at the default salts. -/
def nulOp {D Sl : Type} [Inhabited Sl] : Model.Opens FlockLevel3.GF128 D × Model.Salts Sl :=
  (fun _ _ _ => ([], []), fun _ _ _ => default)

/-- The simulated strategy of the salted session (`simSessJ`) at any verdict `ok` on each table's output, its dead
branches sending empty openings (`nulOp`). -/
noncomputable def simSessG {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D)
    (E : Merkle.Enc (List FlockLevel3.GF128) D) (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc)
    (ts : List (TabSpec mPts)) (ok : Model.OutCL FlockLevel3.GF128 D Sl → Prop)
    (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → Model.Opens FlockLevel3.GF128 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A (ts.get j)) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) : Strategy (sessionBL A Hc E Lf ts) :=
  sessionBL_eq_modelSession A Hc E Lf ts ▸
    (Claims.Bridge.Sim.mapDead (Claims.Bridge.answerWith (nulOp (D := D) (Sl := Sl)))
      (simSession (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) dfltOutL pL opn dg
        dec one fin capOf N (fun j => repsOf (D := D) A (ts.get j))
        (fun j _ _ => Model.allInh_repC A _ (ts.get j).S (ts.get j).sch (ts.get j).hm _) hdec)).strategy Pr

/-- The simulated strategy's `Commit` sends the caps the live prover's roots decode to. -/
theorem simSessG_fst {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D)
    (E : Merkle.Enc (List FlockLevel3.GF128) D) (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc)
    (ts : List (TabSpec mPts)) (ok : Model.OutCL FlockLevel3.GF128 D Sl → Prop)
    (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → Model.Opens FlockLevel3.GF128 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A (ts.get j)) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) :
    (simSessG A Hc E Lf ts ok pL opn dg dec one fin capOf N hdec Pr).1 = capOf Pr.1 := by
  unfold simSessG
  exact fst_rec_send (sessionBL_eq_modelSession A Hc E Lf ts) _

/-- **The salted transfer at any verdict** (`live_le_sessionBLJ`): the live verdict is no likelier than every table's
output passing `ok` against the simulated strategy, when every table's decoder is right for `ok`. -/
theorem live_le_sessionBLG {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D)
    (E : Merkle.Enc (List FlockLevel3.GF128) D) (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc)
    (ts : List (TabSpec mPts)) (ok : Model.OutCL FlockLevel3.GF128 D Sl → Prop)
    (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → Model.Opens FlockLevel3.GF128 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A (ts.get j)) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) :
    prob pL (liveSession dg ts.length mPts N) Pr ≤
      prob (fun outs => ∀ i : Fin ts.length, ok (outs.getD i dfltOutL)) (sessionBL A Hc E Lf ts)
        (simSessG A Hc E Lf ts ok pL opn dg dec one fin capOf N hdec Pr) := by
  unfold simSessG
  rw [prob_cast]
  exact Claims.Bridge.Sim.prob_le_mapDead _ _ Pr

open Classical in
/-- One table's salted session passes `ok` as its output does after `Commit` (`prob_sessionBL_one`). -/
theorem prob_sessionBL_oneG {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List FlockLevel3.GF128) D)
    (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts)
    (ok : Model.OutCL FlockLevel3.GF128 D Sl → Prop) (τ : Strategy (sessionBL A Hc E Lf [t])) :
    (prob (fun outs => ∀ i : Fin [t].length, ok (outs.getD i dfltOutL)) (sessionBL A Hc E Lf [t]) τ).toReal =
      expect (ind fun outs => ok (outs.getD 0 dfltOutL))
        (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] (τ.1 ⟨0, Nat.one_pos⟩)) (afterOne A Hc E Lf t τ) := by
  have hacc : (fun outs : List (Model.OutCL FlockLevel3.GF128 D Sl) => ∀ i : Fin [t].length,
      ok (outs.getD i dfltOutL)) = fun outs => ok (outs.getD 0 dfltOutL) := by
    funext outs
    apply propext
    refine ⟨fun h => h ⟨0, Nat.one_pos⟩, fun h i => ?_⟩
    have hi : i.val = 0 := by
      have h := i.isLt
      simp only [List.length_cons, List.length_nil] at h
      omega
    rwa [hi]
  rw [hacc, prob_toReal]
  exact (expect_cast (sessAfterBL_one A Hc E Lf t τ.1) _ τ.2).symm

/-- The strategy after `Commit`, at a cap `cap₀` its `Commit` sends. -/
noncomputable def afterAt {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
    (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List FlockLevel3.GF128) D)
    (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts) (τ : Strategy (sessionBL A Hc E Lf [t]))
    {cap₀ : Model.Cap D} (h : τ.1 ⟨0, Nat.one_pos⟩ = cap₀) :
    Strategy (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] cap₀) :=
  cast (congrArg (fun c => Strategy (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] c)) h)
    (afterOne A Hc E Lf t τ)

theorem expect_afterAt {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
    (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List FlockLevel3.GF128) D)
    (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts) (τ : Strategy (sessionBL A Hc E Lf [t]))
    {cap₀ : Model.Cap D} (h : τ.1 ⟨0, Nat.one_pos⟩ = cap₀) (f : List (Model.OutCL FlockLevel3.GF128 D Sl) → ℝ) :
    expect f (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] cap₀) (afterAt A Hc E Lf t τ h) =
      expect f (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] (τ.1 ⟨0, Nat.one_pos⟩))
        (afterOne A Hc E Lf t τ) := by
  subst h
  rfl

/-- **The simulated strategy after `Commit`** of a live prover whose roots are `rootB₀`, at the cap they decode to. -/
noncomputable def simAt {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D)
    (E : Merkle.Enc (List FlockLevel3.GF128) D) (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts)
    (ok : Model.OutCL FlockLevel3.GF128 D Sl → Prop) (pL : LiveOutJ [t].length mPts → Prop)
    (opn : Array ByteArray → Fin [t].length → Model.Opens FlockLevel3.GF128 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin [t].length → ℕ → ByteArray → List Msg) (one : Fin [t].length → ℕ → Bool)
    (fin : Fin [t].length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin [t].length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A ([t].get j)) j)
    (rootB₀ : Array ByteArray) (Pr : Strategy (liveSession dg [t].length mPts N)) (hP : Pr.1 = rootB₀) :
    Strategy (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩)) :=
  afterAt A Hc E Lf t (simSessG A Hc E Lf [t] ok pL opn dg dec one fin capOf N hdec Pr)
    (by rw [simSessG_fst, hP])

open Classical in
/-- **A live verdict at one table is no likelier than `ok` after `Commit`** at the cap the prover's roots decode to. -/
theorem live_le_simAt {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D)
    (E : Merkle.Enc (List FlockLevel3.GF128) D) (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts)
    (ok : Model.OutCL FlockLevel3.GF128 D Sl → Prop) (pL : LiveOutJ [t].length mPts → Prop)
    (opn : Array ByteArray → Fin [t].length → Model.Opens FlockLevel3.GF128 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin [t].length → ℕ → ByteArray → List Msg) (one : Fin [t].length → ℕ → Bool)
    (fin : Fin [t].length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin [t].length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A ([t].get j)) j)
    (rootB₀ : Array ByteArray) (Pr : Strategy (liveSession dg [t].length mPts N)) (hP : Pr.1 = rootB₀) :
    (prob pL (liveSession dg [t].length mPts N) Pr).toReal ≤
      expect (ind fun outs => ok (outs.getD 0 dfltOutL))
        (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩))
        (simAt A Hc E Lf t ok pL opn dg dec one fin capOf N hdec rootB₀ Pr hP) := by
  unfold simAt
  rw [expect_afterAt, ← prob_sessionBL_oneG]
  exact ENNReal.toReal_mono (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))
    (live_le_sessionBLG A Hc E Lf [t] ok pL opn dg dec one fin capOf N hdec Pr)

open Classical in
/-- **Over a mixture of live strategies with common roots `rootB₀`**: the mean live verdict is at most the mean
probability of `ok` after `Commit`, over the simulated strategies at the one cap the roots decode to. -/
theorem mean_live_le_simAt {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith FlockLevel3.GF128 K) (Hc : List UInt8 → D)
    (E : Merkle.Enc (List FlockLevel3.GF128) D) (Lf : Merkle.Leaf (List FlockLevel3.GF128) Sl D Hc) (t : TabSpec mPts)
    (ok : Model.OutCL FlockLevel3.GF128 D Sl → Prop) (pL : LiveOutJ [t].length mPts → Prop)
    (opn : Array ByteArray → Fin [t].length → Model.Opens FlockLevel3.GF128 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin [t].length → ℕ → ByteArray → List Msg) (one : Fin [t].length → ℕ → Bool)
    (fin : Fin [t].length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin [t].length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A ([t].get j)) j)
    (rootB₀ : Array ByteArray) {τ : Type} [Fintype τ] (σL : τ → Strategy (liveSession dg [t].length mPts N))
    (hσ : ∀ u, (σL u).1 = rootB₀) :
    (∑ u, (prob pL (liveSession dg [t].length mPts N) (σL u)).toReal) / Fintype.card τ ≤
      (∑ u, expect (ind fun outs => ok (outs.getD 0 dfltOutL))
        (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩))
        (simAt A Hc E Lf t ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u))) / Fintype.card τ :=
  div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ =>
    live_le_simAt A Hc E Lf t ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u)) (Nat.cast_nonneg _)

end FlockDeploy.KS
