import Proofs.Flock.Claims.KS.Mixture
import Proofs.Flock.Claims.Bridge.Returns

/-!
# The self-clash after `Commit` arises only on accepted live runs

A self-clash (`SelfClashL`) is two verifying openings of one position, and an opening verifies only with a row of
`2^lanes` elements (`VerifiedL`). So an output whose openings are all empty never self-clashes (`not_selfClashL_nil`),
and a table's output at the empty openings (`nulOp`, which the simulation's dead branches send) is one
(`tableOutL_nulOp_opens`). At the salted session, an event no outputs at the empty openings satisfy is no likelier
against the simulated strategy than on accepted live runs whose decoded outputs satisfy it (`sessionBL_le_decoded`, from
`Claims.Bridge.prob_le_decoded`). So the self-clash after `Commit` against the simulated strategy (`simAt`) is no
likelier than the live event it decodes from: the live verdict holds, and table 0's
output, at the reps' pendings on their runs of their whole decoded tapes, the cap the roots decode to and the openings
the proof files decode to, self-clashes (`LiveSelfClash`, `selfClash_le_live`). Averaged over a mixture of live
strategies, so is the mean (`mean_selfClash_le_live`).
-/

namespace FlockDeploy.KS

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables

local notation "𝔽" => FlockLevel3.GF128

/-- **An output whose openings are all empty opens every slot with an empty row.** -/
theorem slotOpensL_row_nil {F D Sl : Type} [Field F] [Fintype F] [DecidableEq F] {out : Model.OutCL F D Sl}
    (h : out.opens = fun _ _ _ => ([], [])) {r ℓ Q : ℕ} {x : ℕ × List F × Sl × List D}
    (hx : x ∈ slotOpensL out r ℓ Q) : x.2.1 = [] := by
  unfold slotOpensL at hx
  split at hx
  · simp at hx
  · obtain ⟨pj, -, rfl⟩ := List.mem_map.1 hx
    rw [h]

/-- **An empty row never verifies**: a verifying opening has `2^lanes` elements. -/
theorem not_verifiedL_nil {F D Sl : Type} [Field F] [Fintype F] [DecidableEq F] (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (cap : Model.Cap D) (lanes depth : ℕ)
    {x : ℕ × List F × Sl × List D} (hx : x.2.1 = []) : ¬ VerifiedL H E Lf cap lanes depth x := by
  rintro ⟨hlen, -⟩
  rw [hx, List.length_nil] at hlen
  exact Nat.pos_iff_ne_zero.1 (Nat.two_pow_pos lanes) hlen.symm

theorem not_clashL_nil {F D Sl : Type} [Field F] [Fintype F] [DecidableEq F] (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (cap : Model.Cap D) (lanes depth : ℕ)
    {xs : List (ℕ × List F × Sl × List D)} (hxs : ∀ x ∈ xs, x.2.1 = []) : ¬ ClashL H E Lf cap lanes depth xs := by
  rintro ⟨x, hx, -, -, -, hv, -⟩
  exact not_verifiedL_nil H E Lf cap lanes depth (hxs x hx) hv

/-- **An output whose openings are all empty never self-clashes.** -/
theorem not_selfClashL_nil {F D Sl : Type} [Field F] [Fintype F] [DecidableEq F] (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (sch : FlockSoundness.Accounting.Schedule)
    {out : Model.OutCL F D Sl} (h : out.opens = fun _ _ _ => ([], [])) : ¬ SelfClashL H E Lf sch out := by
  rintro (hc | ⟨r, -, ℓ, -, cap, -, hc⟩)
  · refine not_clashL_nil H E Lf _ _ _ (fun x hx => ?_) hc
    unfold opens₀L at hx
    rcases List.mem_append.1 hx with hx | hx
    · exact slotOpensL_row_nil h hx
    · exact slotOpensL_row_nil h hx
  · exact not_clashL_nil H E Lf _ _ _ (fun x hx => slotOpensL_row_nil h hx) hc

/-- A table's output at the empty openings has every opening empty. -/
theorem tableOutL_nulOp_opens {D Sl : Type} [Inhabited Sl] (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (cap : Model.Cap D) (pend : Model.PendL 𝔽 D × Model.PendL 𝔽 D) :
    (tableOutL Hc E Lf cap pend (nulOp (D := D) (Sl := Sl))).opens = fun _ _ _ => ([], []) :=
  rfl

/-- **A decoded self-clash on an accepted live run**: the live verdict `pL` holds at the outcome, and table 0's output,
at the reps' pendings on their runs of their whole decoded tapes, the cap the roots decode to and the openings the proof
files decode to (`Claims.Bridge.RetOK`), self-clashes. -/
def LiveSelfClash {D Sl : Type} [Inhabited D] [Inhabited Sl] {J mPts : ℕ} (Hc : List UInt8 → D)
    (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (sch : FlockSoundness.Accounting.Schedule)
    (pL : LiveOutJ J mPts → Prop) (opn : Array ByteArray → Fin J → Model.Opens 𝔽 D × Model.Salts Sl)
    (dec : Fin J → ℕ → ByteArray → List Msg) (one : Fin J → ℕ → Bool)
    (fin : Fin J → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin J → Model.Cap D)
    (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game (Model.PendL 𝔽 D)) (b : LiveOutJ J mPts) : Prop :=
  ∃ a, Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Hc E Lf cap pend os) pL opn dec one fin capOf G b a ∧
    SelfClashL Hc E Lf sch (a.getD 0 dfltOutL)

/-- **The salted session's escapes arise only on accepted live runs**: an event `Q` of the tables' outputs that no
outputs at the empty openings satisfy is, against the simulated strategy (`simSessG`), no likelier than a live outcome
the live verdict accepts whose decoded outputs are in `Q` (`Claims.Bridge.prob_le_decoded`). -/
theorem sessionBL_le_decoded {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith 𝔽 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (ts : List (TabSpec mPts)) (ok : Model.OutCL 𝔽 D Sl → Prop)
    (pL : LiveOutJ ts.length mPts → Prop) (opn : Array ByteArray → Fin ts.length → Model.Opens 𝔽 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A (ts.get j)) j)
    {Q : List (Model.OutCL 𝔽 D Sl) → Prop}
    (hQ : ∀ (caps : Fin ts.length → Model.Cap D) outs, (∀ j : Fin ts.length, ∃ pend,
      outs.getD j dfltOutL = tableOutL Hc E Lf (caps j) pend (nulOp (D := D) (Sl := Sl))) → ¬ Q outs)
    (Pr : Strategy (liveSession dg ts.length mPts N)) :
    prob Q (sessionBL A Hc E Lf ts) (simSessG A Hc E Lf ts ok pL opn dg dec one fin capOf N hdec Pr) ≤
      prob (fun b => ∃ a, Claims.Bridge.RetOK (fun _ cap pend os => tableOutL Hc E Lf cap pend os) pL opn dec one fin
        capOf (fun j => repsOf (D := D) A (ts.get j)) b a ∧ Q a) (liveSession dg ts.length mPts N) Pr := by
  unfold simSessG
  rw [prob_cast]
  exact Claims.Bridge.prob_le_decoded _ _ dfltOutL pL opn dg dec one fin capOf N _ _ hdec _ hQ Pr

open Classical in
/-- **The self-clash after `Commit` arises only on accepted live runs**: against the simulated strategy at the cap the
live prover's roots decode to, the self-clash is no likelier than a decoded self-clash on a live run the live verdict
accepts (`LiveSelfClash`). -/
theorem selfClash_le_live {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith 𝔽 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (t : TabSpec mPts) (ok : Model.OutCL 𝔽 D Sl → Prop)
    (pL : LiveOutJ [t].length mPts → Prop) (opn : Array ByteArray → Fin [t].length → Model.Opens 𝔽 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin [t].length → ℕ → ByteArray → List Msg) (one : Fin [t].length → ℕ → Bool)
    (fin : Fin [t].length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin [t].length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A ([t].get j)) j)
    (rootB₀ : Array ByteArray) (Pr : Strategy (liveSession dg [t].length mPts N)) (hP : Pr.1 = rootB₀) :
    expect (ind fun outs => SelfClashL Hc E Lf t.sch (outs.getD 0 dfltOutL))
        (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩))
        (simAt A Hc E Lf t ok pL opn dg dec one fin capOf N hdec rootB₀ Pr hP) ≤
      (prob (LiveSelfClash Hc E Lf t.sch pL opn dec one fin capOf fun j => repsOf (D := D) A ([t].get j))
        (liveSession dg [t].length mPts N) Pr).toReal := by
  unfold simAt
  rw [expect_afterAt, ← prob_sessionBL_oneG A Hc E Lf t (SelfClashL Hc E Lf t.sch)]
  refine ENNReal.toReal_mono (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)) ?_
  refine le_trans (sessionBL_le_decoded A Hc E Lf [t] ok pL opn dg dec one fin capOf N hdec
    (fun caps outs h hq => ?_) Pr) (prob_mono (fun _ ⟨a, hr, hq⟩ => ⟨a, hr, hq ⟨0, Nat.one_pos⟩⟩) _ _)
  obtain ⟨pend, hp⟩ := h ⟨0, Nat.one_pos⟩
  have hc := hq ⟨0, Nat.one_pos⟩
  rw [hp] at hc
  exact not_selfClashL_nil Hc E Lf t.sch (tableOutL_nulOp_opens Hc E Lf _ pend) hc

open Classical in
/-- **Over a mixture of live strategies with common roots `rootB₀`**: the mean self-clash after `Commit`, over the
simulated strategies, is at most the mean probability of a decoded self-clash on an accepted live run. -/
theorem mean_selfClash_le_live {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Nonempty D] [Inhabited D]
    [Inhabited Sl] {mPts : ℕ} (A : Model.Arith 𝔽 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (t : TabSpec mPts) (ok : Model.OutCL 𝔽 D Sl → Prop)
    (pL : LiveOutJ [t].length mPts → Prop) (opn : Array ByteArray → Fin [t].length → Model.Opens 𝔽 D × Model.Salts Sl)
    (dg : ByteArray → ByteArray) (dec : Fin [t].length → ℕ → ByteArray → List Msg) (one : Fin [t].length → ℕ → Bool)
    (fin : Fin [t].length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin [t].length → Model.Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (fun _ cap pend os => tableOutL Hc E Lf cap pend os) (fun _ o => ok o) pL
      opn dec one fin capOf (fun j => repsOf (D := D) A ([t].get j)) j)
    (rootB₀ : Array ByteArray) {τ : Type} [Fintype τ] (σL : τ → Strategy (liveSession dg [t].length mPts N))
    (hσ : ∀ u, (σL u).1 = rootB₀) :
    (∑ u, expect (ind fun outs => SelfClashL Hc E Lf t.sch (outs.getD 0 dfltOutL))
        (sessAfterL A Hc E Lf t.S t.sch t.hm t.ptLocal mPts [] [] (capOf rootB₀ ⟨0, Nat.one_pos⟩))
        (simAt A Hc E Lf t ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u))) / Fintype.card τ ≤
      (∑ u, (prob (LiveSelfClash Hc E Lf t.sch pL opn dec one fin capOf fun j => repsOf (D := D) A ([t].get j))
        (liveSession dg [t].length mPts N) (σL u)).toReal) / Fintype.card τ :=
  div_le_div_of_nonneg_right (Finset.sum_le_sum fun u _ =>
    selfClash_le_live A Hc E Lf t ok pL opn dg dec one fin capOf N hdec rootB₀ (σL u) (hσ u)) (Nat.cast_nonneg _)

end FlockDeploy.KS
