import Proofs.Flock.Claims.KS.Mixture
import Proofs.Flock.Soundness.Discharge.Composed.DefsJ

/-!
# One table's masked session after `Commit`, under a live strategy

`Mixture` at `--zk`. The old masked transfer (`live_le_sessionZJ`) bounds a live verdict by the compiled session
`sessionZC` on the tables; the old masked session theorems (`session_sound_of_table_accZ`, `epsSZ`) run after `Commit`,
on `sessAtZ`. At one table they meet: `sessionZC [T]` is `Commit`'s cap, then `sessAtZ` with no other game
(`sessAfterZC_one`), and at any verdict `ok` on the table's output that the decoder is right for, a live verdict is no
likelier than `ok` after `Commit` against the simulated strategy (`simSessZG`, `live_le_sessionZG`,
`prob_sessionZC_oneG`). The simulated `Commit` sends the cap the live roots decode to (`simSessZG_fst`), so over a
mixture of live strategies whose roots agree, the strategies after `Commit` share one cap (`simAtZ`, `live_le_simAtZ`,
`mean_live_le_simAtZ`).
-/

namespace FlockDeploy.KS

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable simSession live_le_session)
open FlockSoundness.Discharge.Composed (outJ repJ sessionZC_eq_modelSession)

local notation "𝔽" => FlockLevel3.GF128

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

omit [Fintype K] in
/-- **One table's masked session after `Commit`** is `sessAtZ` with no other game. -/
theorem sessAfterZC_one (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (caps : Fin [T].length → Model.Cap D) :
    sessAfterZC (gamesZC A Hs [T] caps) = sessAtZ [] [] (tabAfterZC A Hs T (caps ⟨0, Nat.one_pos⟩)) := by
  unfold sessAfterZC gamesZC sessAtZ
  rfl

/-- The strategy after `Commit` of a strategy of one table's masked session. -/
noncomputable def afterOneZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (τ : Strategy (sessionZC A Hs [T])) :
    Strategy (sessAtZ [] [] (tabAfterZC A Hs T (τ.1 ⟨0, Nat.one_pos⟩))) :=
  cast (congrArg Strategy (sessAfterZC_one A Hs T τ.1)) τ.2

variable [Inhabited D] [Inhabited Sl]

/-- The simulated strategy of the masked session (`simSessZJ`) at any verdict `ok` on each table's output. -/
noncomputable def simSessZG (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (ok : OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ) (hdec : ∀ j, DecodesTable (outJ Hs ts) (fun _ o => ok o) pL opn dec one fin capOf (repJ A Hs ts) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) : Strategy (sessionZC A Hs ts) :=
  sessionZC_eq_modelSession A Hs ts ▸
    (simSession (outJ Hs ts) (fun _ o => ok o) dfltOutZC pL opn dg dec one fin capOf N (repJ A Hs ts)
      (fun j pts r => allInh_repZCAt A Hs (ts.get j) pts r) hdec).strategy Pr

/-- The simulated strategy's `Commit` sends the caps the live prover's roots decode to. -/
theorem simSessZG_fst (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (ok : OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ) (hdec : ∀ j, DecodesTable (outJ Hs ts) (fun _ o => ok o) pL opn dec one fin capOf (repJ A Hs ts) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) :
    (simSessZG A Hs ts ok pL opn dg dec one fin capOf N hdec Pr).1 = capOf Pr.1 := by
  unfold simSessZG
  exact fst_rec_send (sessionZC_eq_modelSession A Hs ts) _

/-- **The masked transfer at any verdict** (`live_le_sessionZJ`): the live verdict is no likelier than every table's
output passing `ok` against the simulated strategy, when every table's decoder is right for `ok`. -/
theorem live_le_sessionZG (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (ok : OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ ts.length mPts → Prop)
    (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ) (hdec : ∀ j, DecodesTable (outJ Hs ts) (fun _ o => ok o) pL opn dec one fin capOf (repJ A Hs ts) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) :
    prob pL (liveSession dg ts.length mPts N) Pr ≤
      prob (fun outs => ∀ i : Fin ts.length, ok (outs.getD i dfltOutZC)) (sessionZC A Hs ts)
        (simSessZG A Hs ts ok pL opn dg dec one fin capOf N hdec Pr) := by
  unfold simSessZG
  rw [FlockSoundness.Discharge.Exec.prob_cast]
  exact live_le_session _ _ _ pL opn dg dec one fin capOf N _ _ hdec Pr

omit [Fintype K] in
open Classical in
/-- One table's masked session passes `ok` as its output does after `Commit`. -/
theorem prob_sessionZC_oneG (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (ok : OutZC 𝔽 D Sl → Prop) (τ : Strategy (sessionZC A Hs [T])) :
    (prob (fun outs => ∀ i : Fin [T].length, ok (outs.getD i dfltOutZC)) (sessionZC A Hs [T]) τ).toReal =
      expect (ind fun outs => ok (outs.getD 0 dfltOutZC))
        (sessAtZ [] [] (tabAfterZC A Hs T (τ.1 ⟨0, Nat.one_pos⟩))) (afterOneZ A Hs T τ) := by
  have hacc : (fun outs : List (OutZC 𝔽 D Sl) => ∀ i : Fin [T].length,
      ok (outs.getD i dfltOutZC)) = fun outs => ok (outs.getD 0 dfltOutZC) := by
    funext outs
    apply propext
    refine ⟨fun h => h ⟨0, Nat.one_pos⟩, fun h i => ?_⟩
    have hi : i.val = 0 := by
      have h := i.isLt
      simp only [List.length_cons, List.length_nil] at h
      omega
    rwa [hi]
  rw [hacc, prob_toReal]
  exact (expect_cast (sessAfterZC_one A Hs T τ.1) _ τ.2).symm

/-- The strategy after `Commit`, at a cap `cap₀` its `Commit` sends. -/
noncomputable def afterAtZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (τ : Strategy (sessionZC A Hs [T])) {cap₀ : Model.Cap D} (h : τ.1 ⟨0, Nat.one_pos⟩ = cap₀) :
    Strategy (sessAtZ [] [] (tabAfterZC A Hs T cap₀)) :=
  cast (congrArg (fun c => Strategy (sessAtZ [] [] (tabAfterZC A Hs T c))) h) (afterOneZ A Hs T τ)

omit [Fintype K] [Inhabited D] [Inhabited Sl] in
theorem expect_afterAtZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (τ : Strategy (sessionZC A Hs [T])) {cap₀ : Model.Cap D} (h : τ.1 ⟨0, Nat.one_pos⟩ = cap₀)
    (f : List (OutZC 𝔽 D Sl) → ℝ) :
    expect f (sessAtZ [] [] (tabAfterZC A Hs T cap₀)) (afterAtZ A Hs T τ h) =
      expect f (sessAtZ [] [] (tabAfterZC A Hs T (τ.1 ⟨0, Nat.one_pos⟩))) (afterOneZ A Hs T τ) := by
  subst h
  rfl

/-- **The simulated masked strategy after `Commit`** of a live prover whose roots are `rootB₀`, at the cap they decode
to. -/
noncomputable def simAtZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (ok : OutZC 𝔽 D Sl → Prop)
    (pL : LiveOutJ [T].length mPts → Prop) (opn : Array ByteArray → Fin [T].length → OpensZ 𝔽 D Sl)
    (dg : ByteArray → ByteArray) (dec : Fin [T].length → ℕ → ByteArray → List Msg) (one : Fin [T].length → ℕ → Bool)
    (fin : Fin [T].length → Fin 2 → Array ByteArray → List Msg)
    (capOf : Array ByteArray → Fin [T].length → Model.Cap D) (N : ℕ)
    (hdec : ∀ j, DecodesTable (outJ Hs [T]) (fun _ o => ok o) pL opn dec one fin capOf (repJ A Hs [T]) j)
    (rootB₀ : Array ByteArray) (Pr : Strategy (liveSession dg [T].length mPts N)) (hP : Pr.1 = rootB₀) :
    Strategy (sessAtZ [] [] (tabAfterZC A Hs T (capOf rootB₀ ⟨0, Nat.one_pos⟩))) :=
  afterAtZ A Hs T (simSessZG A Hs [T] ok pL opn dg dec one fin capOf N hdec Pr) (by rw [simSessZG_fst, hP])

open Classical in
/-- **A live verdict at one masked table is no likelier than `ok` after `Commit`** at the cap the prover's roots decode
to. -/
theorem live_le_simAtZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (ok : OutZC 𝔽 D Sl → Prop)
    (pL : LiveOutJ [T].length mPts → Prop) (opn : Array ByteArray → Fin [T].length → OpensZ 𝔽 D Sl)
    (dg : ByteArray → ByteArray) (dec : Fin [T].length → ℕ → ByteArray → List Msg) (one : Fin [T].length → ℕ → Bool)
    (fin : Fin [T].length → Fin 2 → Array ByteArray → List Msg)
    (capOf : Array ByteArray → Fin [T].length → Model.Cap D) (N : ℕ)
    (hdec : ∀ j, DecodesTable (outJ Hs [T]) (fun _ o => ok o) pL opn dec one fin capOf (repJ A Hs [T]) j)
    (rootB₀ : Array ByteArray) (Pr : Strategy (liveSession dg [T].length mPts N)) (hP : Pr.1 = rootB₀) :
    (prob pL (liveSession dg [T].length mPts N) Pr).toReal ≤
      expect (ind fun outs => ok (outs.getD 0 dfltOutZC))
        (sessAtZ [] [] (tabAfterZC A Hs T (capOf rootB₀ ⟨0, Nat.one_pos⟩)))
        (simAtZ A Hs T ok pL opn dg dec one fin capOf N hdec rootB₀ Pr hP) := by
  unfold simAtZ
  rw [expect_afterAtZ, ← prob_sessionZC_oneG]
  exact ENNReal.toReal_mono (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))
    (live_le_sessionZG A Hs [T] ok pL opn dg dec one fin capOf N hdec Pr)

open Classical in
/-- **Over a mixture of live strategies with common roots `rootB₀`**, at one masked table: the mean live verdict is at
most the mean probability of `ok` after `Commit`, over the simulated strategies at the one cap the roots decode to. -/
theorem mean_live_le_simAtZ (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (ok : OutZC 𝔽 D Sl → Prop) (pL : LiveOutJ [T].length mPts → Prop)
    (opn : Array ByteArray → Fin [T].length → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Fin [T].length → ℕ → ByteArray → List Msg) (one : Fin [T].length → ℕ → Bool)
    (fin : Fin [T].length → Fin 2 → Array ByteArray → List Msg)
    (capOf : Array ByteArray → Fin [T].length → Model.Cap D) (N : ℕ)
    (hdec : ∀ j, DecodesTable (outJ Hs [T]) (fun _ o => ok o) pL opn dec one fin capOf (repJ A Hs [T]) j)
    (rootB₀ : Array ByteArray) {τ : Type} [Fintype τ] (σL : τ → Strategy (liveSession dg [T].length mPts N))
    (hσ : ∀ u, (σL u).1 = rootB₀) :
    (∑ u, (prob pL (liveSession dg [T].length mPts N) (σL u)).toReal) / Fintype.card τ ≤
      (∑ u, expect (ind fun outs => ok (outs.getD 0 dfltOutZC))
        (sessAtZ [] [] (tabAfterZC A Hs T (capOf rootB₀ ⟨0, Nat.one_pos⟩)))
        (simAtZ A Hs T ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u))) / Fintype.card τ :=
  div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ =>
    live_le_simAtZ A Hs T ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u)) (Nat.cast_nonneg _)

end FlockDeploy.KS
