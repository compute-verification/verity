import Proofs.Flock.Claims.KS.PlainLeafFork
import Proofs.Flock.Claims.Recursion.PlainK

/-!
# Knowledge soundness at the deployed plain verifier, with no premise

The simulator's chain (`PlainSimWD`) covers the statements whose outputs are public or hidden under hm96 leaves. The
deployed scope admits exactly one other leaf, `flock-leaf/sha512-unsalted` (`Claims.Flock.InScope`, `plain_leafU`),
which the deployed decoders read at the model's plain leaf `leafPlain` (`plain_hdecHUAt`), not at the hm96 model leaf
`leaf512`, so the simulator's chain runs there as it does at hm96 leaves:

* `PlainSimWrPAtDU`, `plain_le_sessionsEU`, `plain_tableBoundθAt_of_simDU`, `plainSimWrPAtD_of_partsU`: the
  simulator's table bound at `leafPlain`, copies of `PlainSimWrPAtD`, `plain_le_sessionsE`,
  `plain_tableBoundθAt_of_simD` and `plainSimWrPAtD_of_parts`, whose session steps are the leaf's own.
* `plainSimWrPAtDU_at`: the simulator at every plain-leaf statement, from its clash (`plainSimClashHU`) and its fork
  bound (`plainAdv_leU`).
* `plainSimWD_all`: E14's simulator where the outputs are public or hidden under hm96 leaves.
* `plain_tableBoundθAt_all`, `plain_KS`: so KS at SHA-512 at the deployed plain verifier's trimmed record at the
  deployed relation holds with no hypothesis.
-/

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

section Bound

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Exec

open Classical in
/-- **The plain bound at one tape and table, from salted sessions after `Commit`, at any escapes, at `leafPlain`**: `plain_le_sessionsE`'s proof, by `session_sound_of_table_accL_sc` at `leafPlain`. -/
theorem plain_le_sessionsEUK (keep : Bool → Bool) {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m) (ptLocal : ℕ)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout ptLocal spec.mPts) {Ω : Type} (P : Claims.Prover Ω)
    (e : Claims.Escapes sha512 (VK keep) P x) (ω : Ω)
    (C : Fin (wordLen false x) → List 𝔽)
    (hT : ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (Claims.Flock.oracleOf C))
    (τ : Type) [Fintype τ] [Nonempty τ] (cap₀ : Model.Cap Refine.Dig)
    (σ : τ → Strategy (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leafPlain
      (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀))
    (hacc : (Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (VK keep) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (VK keep) x ω r).seen 0) p.val = some u ∧ u ≠ C p) ≤
      (∑ t, expect (ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leafPlain
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL)))
        (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leafPlain (L.model (Refine.regionsOf st))
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀) (σ t)) / Fintype.card τ)
    (hmean : (∑ t, epsSL execArith Refine.H512 Refine.enc512 Refine.leafPlain (L.model (Refine.regionsOf st)) m k0 l₀
        l₁ rest ext hk0 hm ptLocal spec.mPts [] [] cap₀ (σ t)) / Fintype.card τ ≤ εtable false x + e.term ω) :
    Claims.accPr sha512 (VK keep) P x ω ≤ εtable false x +
      e.term ω +
      Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (VK keep) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (VK keep) x ω r).seen 0) p.val = some u ∧ u ≠ C p := by
  have hper : ∀ t, expect (ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leafPlain
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL)))
      (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leafPlain (L.model (Refine.regionsOf st))
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀) (σ t) ≤
      epsSL execArith Refine.H512 Refine.enc512 Refine.leafPlain (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext
          hk0 hm ptLocal spec.mPts [] [] cap₀ (σ t) := by
    intro t
    have hs := session_sound_of_table_accL_sc execArith execArith_correct Refine.H512 Refine.enc512 Refine.leafPlain
      (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hsch hm ptLocal spec.mPts hlay [] [] cap₀
      (fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
        cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL)))
      (fun _ h => h.1) (Claims.Flock.oracleOf C) (σ t)
    have e : (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
        (AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
          (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (outs.getD 0 dfltOutL))) ∧
        ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (Claims.Flock.oracleOf C)) = ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512
          Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (outs.getD 0 dfltOutL)) :=
      congrArg ind (funext fun _ => propext (and_iff_left hT))
    have e₀ : (ind fun outs : List (Model.OutCL 𝔽 Refine.Dig (List UInt8)) =>
        (AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
          (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (outs.getD 0 dfltOutL))) ∧
        ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (outs.getD 0 dfltOutL) ∧
        Off₀L Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀
          (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL)) = fun _ => 0 := by
      funext outs
      unfold ind
      split_ifs with h0
      · exact absurd ⟨h0.2.2, h0.2.1⟩ h0.1.2
      · rfl
    rw [← e]
    refine le_trans hs (le_of_eq ?_)
    simp only [List.length_nil]
    rw [e₀, expect_const, add_zero]
  have hsum := Finset.sum_le_sum fun t (_ : t ∈ Finset.univ) => hper t
  have hcard : (0 : ℝ) ≤ Fintype.card τ := Nat.cast_nonneg _
  have hdiv := div_le_div_of_nonneg_right hsum hcard
  have hsplit : Claims.accPr sha512 (VK keep) P x ω ≤
      (Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (VK keep) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (VK keep) x ω r).seen 0) p.val = some u ∧ u ≠ C p) +
      Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (VK keep) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (VK keep) x ω r).seen 0) p.val = some u ∧ u ≠ C p := by
    unfold Claims.accPr
    refine le_trans (Claims.Pr_mono fun r hr => ?_) (Claims.Pr_or_le _ _)
    by_cases hq : ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
        Claims.Flock.lookCol (cols false (Claims.run sha512 P (VK keep) x ω r).seen 0) p.val = some u ∧ u ≠ C p
    · exact Or.inr ⟨hr, hq⟩
    · exact Or.inl ⟨hr, hq⟩
  have key : Claims.accPr sha512 (VK keep) P x ω ≤ εtable false x +
      e.term ω +
      Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (VK keep) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (VK keep) x ω r).seen 0) p.val = some u ∧ u ≠ C p := by
    linarith [hsplit, hacc, hmean, hdiv]
  exact key

/-- `plain_le_sessionsEUK` at the deployed plain verifier (`keep := id`). -/
theorem plain_le_sessionsEU {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m) (ptLocal : ℕ)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout ptLocal spec.mPts) {Ω : Type} (P : Claims.Prover Ω)
    (e : Claims.Escapes sha512 (V false) P x) (ω : Ω)
    (C : Fin (wordLen false x) → List 𝔽)
    (hT : ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
      (Claims.Flock.oracleOf C))
    (τ : Type) [Fintype τ] [Nonempty τ] (cap₀ : Model.Cap Refine.Dig)
    (σ : τ → Strategy (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leafPlain
      (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀))
    (hacc : (Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧ u ≠ C p) ≤
      (∑ t, expect (ind fun outs => AccOne outs ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leafPlain
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ (Claims.Flock.oracleOf C) (outs.getD 0 dfltOutL) ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD 0 dfltOutL)))
        (sessAfterL execArith Refine.H512 Refine.enc512 Refine.leafPlain (L.model (Refine.regionsOf st))
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts [] [] cap₀) (σ t)) / Fintype.card τ)
    (hmean : (∑ t, epsSL execArith Refine.H512 Refine.enc512 Refine.leafPlain (L.model (Refine.regionsOf st)) m k0 l₀
        l₁ rest ext hk0 hm ptLocal spec.mPts [] [] cap₀ (σ t)) / Fintype.card τ ≤ εtable false x + e.term ω) :
    Claims.accPr sha512 (V false) P x ω ≤ εtable false x +
      e.term ω +
      Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some u ∧ u ≠ C p :=
  plain_le_sessionsEUK id (x := x) (spec := spec) (st := st) (L := L) (m := m) (k0 := k0) (l₀ := l₀) (l₁ := l₁) (rest := rest) (ext := ext) (hsch := hsch) (hk0 := hk0) (hm := hm) (ptLocal := ptLocal) (hlay := hlay) (Ω := Ω) (P := P) (e := e) (ω := ω) (C := C) (hT := hT) (τ := τ) (cap₀ := cap₀) (σ := σ) (hacc := hacc) (hmean := hmean)

end Bound

section Chain

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Tables Claims.Bridge.Live
open Lean (Json)

open Classical in
/-- **A simulator of each prover at the verdict at the draw, at `leafPlain`**: `PlainSimWrPAtD` with its decoders, live self-clash and fork advantage at `leafPlain`. -/
def PlainSimWrPAtDU {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)])
    (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leafPlain
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAt I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
    ∃ (δ : ℕ → ℝ) (_ : ∀ t, 0 ≤ δ t)
      (_ : ∑ i, Claims.Flock.slackC ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (fun r ℓ (i : Fin ((V false).k x)) =>
            Claims.Flock.slackW ((V false).k x) δ (plainJ (tables_head_one h) hsch r ℓ).val i.val) i ≤
        2 * ((2 : ℝ) ^ l₀.logLen * l₀.queries)),
    ∀ ω : Ω, ∃ (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 spec.mPts) (τ : Type)
      (_ : Fintype τ) (_ : Nonempty τ) (P' : Claims.Prover (Ω × τ))
      (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin)
      (_ : ∀ E : (Fin ((V false).k x) → Coin) → Prop, Claims.Pr (fun p => E (Φ p)) = Claims.Pr E)
      (hσ : ∀ u, (liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)).1 = plainRoot P x ω),
      PlainSimRun P x ω spec.mPts N dg w P' Φ ∧
      (∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) →
        PlainClashRun P x ω spec.mPts N dg w P' Φ
          (LiveSelfClash Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (plainVerdictAt I d C (plainRoot P x ω)) opn
            (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
            (fun rs _ => capOfRoot rs[0]!) fun j => repsOf (D := Refine.Dig) execArith
              ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))) ∧
      (∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) → ∀ r ℓ,
        (∑ u, advRSL execArith Refine.H512 Refine.enc512 Refine.leafPlain (L.model (Refine.regionsOf st)) m
            k0 l₀ l₁ rest ext hk0 hm PT_LOCAL spec.mPts [] [] (capOfRoot (plainRoot P x ω)[0]!) r.val (ℓ.val + 1)
            (simAt execArith Refine.H512 Refine.enc512 Refine.leafPlain
              (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAt I d C (plainRoot P x ω)) opn
              dg (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
              (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
              (liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) (hσ u))) /
            Fintype.card τ ≤
          (Claims.Pr fun s : (Fin ((V false).k x) → Coin) × (Fin ((V false).k x) → Coin) =>
            plainForkEv h hsch P r ℓ ω s.1 s.2) +
            Claims.Flock.forkSlack sha512 (V false) P x δ (plainJ (tables_head_one h) hsch r ℓ).val ω)

open Classical in
/-- **A simulator at the verdict at the draw gives the plain table bound at a statement, at `leafPlain`**: `plain_tableBoundθAt_of_simD`'s proof, at `plain_le_sessionsEU`. -/
theorem plain_tableBoundθAt_of_simDU {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt}
    {h : tables false x = [(spec, st)]} {L : Layout st} {m k0 : ℕ} {l₀ l₁ : Accounting.Level}
    {rest : List Accounting.Level} {ext : ℕ} {hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩}
    {hk0 : l₀.lanes = k0} {hm : 13 ≤ (L.model (Refine.regionsOf st)).m}
    {hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts} {d : Json}
    {opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)}
    {hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leafPlain
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAt ins.read.I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j}
    (hset : setupOf false x = some (ins, spec, sts)) (hd : ins.read.draw = some d)
    (hnc : ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) →
      ¬ Claims.Flock.GoodTable (deployPlain Readable.relation) (wordLen false) x C →
      ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (Claims.Flock.oracleOf C))
    (hs : PlainSimWrPAtDU h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d opn hdec) :
    Claims.Flock.TableBoundθAt sha512 (deployPlain Readable.relation) (wordLen false) (wide false) x := by
  intro Ω _ _ P
  obtain ⟨δ, hδ, hfit, hω⟩ := hs Ω P
  refine ⟨Claims.Flock.slackEsc sha512 (V false) P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (plainJ (tables_head_one h) hsch)
      (fun r ℓ ω s s' => plainForkEv h hsch P r ℓ ω s s') (plainEv₁ P x) (fun _ _ _ _ _ he => he.2.2)
      (fun _ _ he => he.2)
      (fun r ℓ i => Claims.Flock.slackW ((V false).k x) δ (plainJ (tables_head_one h) hsch r ℓ).val i.val)
      (Claims.Flock.slackG sha512 (V false) P x) (Claims.Flock.slackG_tied sha512 (V false) P x),
    slackEsc_fitsFork_plain Readable.relation P x (tables_head_one h) hsch _ _ _ _ _ _ _ hfit, fun ω C hok hg => ?_⟩
  obtain ⟨N, dg, w, τ, _, _, P', Φ, hΦ, hσ, hrun, hsc, hfork⟩ := hω ω
  have htr := plain_first_of_simD P x ω C spec.mPts hset hd N dg w P' Φ ⟨hΦ, hrun⟩
  have hsc' := liveMix_le_Pr (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x ω _ (plainEv₁ P x ω) Φ (hΦ _)
    (hsc C hok)
  exact plain_le_sessionsEU L m k0 l₀ l₁ rest ext hsch hk0 hm PT_LOCAL hlay P _ ω C (hnc C hok hg) τ
    (capOfRoot (plainRoot P x ω)[0]!) (fun u => simAt execArith Refine.H512 Refine.enc512 Refine.leafPlain
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAt ins.read.I d C (plainRoot P x ω)) opn
      dg (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
      (liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) (hσ u))
    (le_trans htr (mean_live_le_simAt execArith Refine.H512 Refine.enc512 Refine.leafPlain
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAt ins.read.I d C (plainRoot P x ω)) opn
      dg (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
      (fun u => liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) hσ))
    (mean_epsSL_le_slack execArith Refine.H512 Refine.enc512 Refine.leafPlain h L m k0 l₀ l₁ rest ext hsch hk0 hm
      PT_LOCAL [] [] _ _ sha512 (V false) P (plainJ (tables_head_one h) hsch) _ _ _ _ _
      (fun _ _ _ => Claims.Flock.slackW_nonneg hδ _ _ _) _ _ ω
      (fun r ℓ => (hfork C hok r ℓ).trans (add_le_add le_rfl
        (Claims.Flock.forkSlack_le sha512 (V false) P x hδ _ ω)))
      (le_trans (mean_selfClash_le_live execArith Refine.H512 Refine.enc512 Refine.leafPlain
        (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAt ins.read.I d C (plainRoot P x ω)) opn
        dg (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
        (fun u => liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) hσ) hsc'))

open Classical in
/-- **The concrete simulator at the verdict at the draw, from its clash and its fork bound, at `leafPlain`**: `plainSimWrPAtD_of_parts`'s proof. -/
theorem plainSimWrPAtD_of_partsU {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt}
    (h : tables false x = [(spec, st)]) {d : Json} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leafPlain
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAt ins.read.I d C rootB₀) opn
        (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j)
    (hc : ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) (ω : Ω),
      PlainClashQU P x ω st (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) (plain_mPts hs hin hset).1
        ins.read.I d opn (SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
    (hf : ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽),
      (∀ p, wide false x (C p)) → ∀ r ℓ,
        plainAdvU L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d x opn hdec (plain_mPts hs hin hset).1 P ω C r
            ℓ ≤
          Claims.Flock.simForkPr sha512 (V false) P x (plainΦ x spec st) (fun t => Claims.Flock.FnFork.forkN t) ω
            (plainJ (tables_head_one h) hsch r ℓ).val) :
    PlainSimWrPAtDU h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d opn hdec := by
  intro Ω _ _ P
  obtain ⟨hw, hm1, hp⟩ := plain_mPts hs hin hset
  have hL := plainN_add hset
  refine ⟨Claims.Flock.FnFork.δN FlockLevel3.GF128 65536 (plainNOf x spec st),
    fun t => Claims.Flock.FnFork.δN_nonneg _ t,
    plain_slack_fits (tables_head_one h) hsch _ fun t _ => plainNOf_pos x spec st t, fun ω => ?_⟩
  refine ⟨plainN, plainDg spec, plainW spec hw, plainTape x, inferInstance, inferInstance, plainSim P x spec st,
    plainΦ x spec st, Pr_plainΦ hL, plainSim_root P x spec st ω hw, plainSimRun P ω hset h hw hm1 hp,
    fun C _ => ?_, fun C hC r ℓ => ?_⟩
  · refine plainClashRun_of_rawL (plainRawClashRunL_of_abort (fun u r' hna hq => ?_) (hc Ω P ω C))
    obtain ⟨a, ha, -⟩ := hq
    exact plainSimAbortAt x s hs hin ins spec sts hset st h Ω P ω ins.read.I d C (plainRoot P x ω) u r' hna ha.1
  · exact plain_fork_le_fnAt h hsch P ω hL (plainNOf x spec st) (Claims.Flock.FnFork.prefixKey_histKey _) r ℓ
      ((hf Ω P ω C hC r ℓ).trans_eq (by rw [plainΦ_eq hL]; rfl))

/-- **An in-scope plain statement's hidden outputs not under hm96 leaves are under plain SHA-512 leaves**: the
deployed scope's leaf rule (`Claims.Flock.InScope`, `ProvedScope.leafOk`). -/
theorem plain_leafU {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) (hl : ins.read.I.tags.merkleLeaf ≠ "hm96-sha512/v1") :
    ins.read.I.tags.merkleLeaf = "flock-leaf/sha512-unsalted" := by
  obtain ⟨p, hp, rfl, -, -, -⟩ := parsed_of_parse hs
  obtain ⟨-, hpr, -⟩ := parseBytes_setup hp hset
  have hzk : ins.read.zk = false := readOf_zk (x := x) (by unfold readOf; rw [(setupOf_some hset).1]; rfl)
  have htags : ins.read.tags = ins.read.I.tags := by simp [Readable.Read.tags, hzk]
  have hform : (Readable.toReadable Readable.relation p).form = ins.read.form := by rw [← hpr]; rfl
  obtain ⟨-, -, -, -, -, -, -, hlf, -⟩ := hin
  rw [hform] at hlf
  simp only [Readable.Read.form, htags] at hlf
  rcases hlf with e | ⟨-, -, e⟩
  · exact absurd e hl
  · exact e

/-- **The concrete simulator at a plain-leaf statement**: at a plain statement in scope with hidden outputs under
`flock-leaf/sha512-unsalted` leaves, `PlainSimWrPAtDU` at the hidden openings and decoders (`plainOpnH`,
`plain_hdecHUAt`), from the live self-clash's clash (`plainSimClashHU`) and the fork bound (`plainAdv_leU`). -/
theorem plainSimWrPAtDU_at {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse false Readable.relation x = some s)
    (hin : Claims.Flock.InScope false s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf false x = some (ins, spec, sts)) {st : Stmt} (h : tables false x = [(spec, st)])
    (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "flock-leaf/sha512-unsalted")
    {d : Json} (hd : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) :
    PlainSimWrPAtDU h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d
      (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
      (plain_hdecHUAt hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) :=
  plainSimWrPAtD_of_partsU hs hin hset h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay _ _
    (fun _ _ _ P ω => plainSimClashHU hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay P ω)
    (fun _ _ _ P ω C _ r ℓ => plainAdv_leU hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay P ω C r ℓ)

/-- **The concrete simulator where the outputs are public or hidden under hm96 leaves**, at every statement: E14's
`plainSimWD_of_parts` at the clash (`plainSimClashAt_of_later`, `plainSimLaterClashAt`) and the fork bound
(`plainSimForkLe_of` at `plainSchedAlign` and `plainForkClash`). -/
theorem plainSimWD_all (x : Claims.Msg) : PlainSimWD x :=
  plainSimWD_of_parts (plainSimClashAt_of_later (plainSimLaterClashAt x))
    (plainSimForkLe_of (plainForkAlign_of_slot (plainSlotAlign_of_sched (plainSchedAlign x))) (plainForkClash x))

/-- **The plain table bound at every statement some run accepts**: such a statement is in scope with one table
(`acc_inScope`, `plain_one`), whose model statement no table off it is committed to (`plain_not_committed`), so
`plain_tableBoundθAt_of_simD` at `plainSimWD_all` where the outputs are public or hidden under hm96 leaves, and
`plain_tableBoundθAt_of_simDU` at `plainSimWrPAtDU_at` where they are hidden under the one other leaf the scope admits
(`plain_leafU`). -/
theorem plain_tableBoundθAt_all {x : Claims.Msg} (hno : ¬ Claims.Flock.NoAcc sha512 (V false) x) :
    Claims.Flock.TableBoundθAt sha512 (deployPlain Readable.relation) (wordLen false) (wide false) x := by
  obtain ⟨s, hs, hin⟩ := acc_inScope Readable.relation hno
  obtain ⟨ins, spec, sts, d, hset, hd⟩ := acc_draw hno
  have h1 := plain_one hs hin hset
  have h0 : 0 < sts.size := by omega
  have hm := parse_m hs hset h0
  have h22 : 22 ≤ (sts[0]'h0).m := by have := hin.1; simp only [Bool.false_eq_true, ↓reduceIte] at this; omega
  have h35 : (sts[0]'h0).m ≤ 35 := by have := hin.2.1; omega
  obtain ⟨L, hlay⟩ := plain_table0 hs hin hset h0
  obtain ⟨k0, l₀, l₁, rest, ext, hsch, hk0, e⟩ := plain_sched h22 h35
  have ht := tables_of_one hset h1
  have hm13 : 13 ≤ (L.model (Refine.regionsOf (sts[0]'h0))).m := by show 13 ≤ (sts[0]'h0).m; omega
  have hnc : ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) →
      ¬ Claims.Flock.GoodTable (deployPlain Readable.relation) (wordLen false) x C →
      ¬ Committed execArith (L.model (Refine.regionsOf (sts[0]'h0))) ⟨_, k0, l₀ :: l₁ :: rest, ext⟩
        (Claims.Flock.oracleOf C) := fun C hw hg => by
    have := plain_not_committed hs hin hset h0 L C hw hg
    rwa [e] at this
  obtain ⟨hw, hhid⟩ := plainSimWD_all x s hs hin ins spec sts hset _ ht d hd L _ k0 l₀ l₁ rest ext hsch hk0 hm13 hlay
  cases hh : ins.read.I.tags.hiddenOutputs
  · exact plain_tableBoundθAt_of_simD hset hd hnc (hw hh)
  · by_cases hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1"
    · exact plain_tableBoundθAt_of_simD hset hd hnc (hhid hh hl)
    · exact plain_tableBoundθAt_of_simDU hset hd hnc (plainSimWrPAtDU_at hs hin hset ht hh
        (plain_leafU hs hin hset hl) hd L _ k0 l₀ l₁ rest ext hsch hk0 hm13 hlay)

/-- **The deployed plain verifier: KS**, at SHA-512 at the trimmed record at the deployed relation, with no
hypothesis: `plain_KS_trim` at `plain_tableBoundθAt_all`. -/
theorem plain_KS : Claims.Flock.KS sha512 (deployPlainTrim Readable.relation) :=
  plain_KS_trim Readable.relation (plain_tableBoundθ_of_acc Readable.relation (wide false)
    fun _ _ _ _ _ hno _ _ _ => plain_tableBoundθAt_all hno)

end Chain

end FlockDeploy.KS
