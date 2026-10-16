import Proofs.Flock.Recursive.Registrants

/-!
# Recursion: a session's registered read off the table's leaf, priced by a finder of the recursive audit

At an accepted session the verifier's registered check opens each registered read's root at the read's position at the
table's leaf (`ReadsX.Checked`, `checked_of_acceptsHJ`), and the registrant opened it there at its own leaf
(`ReadsX.committed`): the two leaves are one, or the two openings conflict (`TwoOpenings.Conflict`), a SHA-512
collision. The session's own finder outputs that pair behind its `boundIf`, at the session's outcome. At V*'s bridge,
which reads every session's outcome at one history, the pair is a history's, so the finder is the whole audit's:

* `LeafColl`, `LeafColl.collision`: some session's registered read whose openings under some registration and paths
  conflict;
* `leafFinder`, `prob_leafColl_le`: the finder that plays the recursive audit once and outputs the two openings' pair
  (both preimages); under `cr/sha-512` (`Finder.CR` at `qL`), the audit's outcome has such a history with probability
  at most `qL²/2^513`, once per audit. At an accepted session the table's leaf at a row read is `treeLeaf` of the
  public file's digest (`FlockC3.RegLevels.regLeaf_inl`) and the path the public file header's (`regPathsHJ`), so an
  honest budget counts the audit run and, at each registered read, the two openings' recomputation.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg
open scoped ENNReal

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type}

/-- **A session's registered read is off the table's leaf, by a collision**: at the history `(R, ω, t)` and
registration `R₂`, some session `j`'s registered read `p` whose opening under a registration `R'` through `path` and
the registrant's conflict. -/
def LeafColl (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (R : Reg) (ω : L.Ω)
    (t : List (Cm × I.Coin)) (R₂ : Reg₂) : Prop :=
  ∃ j, ∃ R' : Reg', ∃ path p, (zs R ω t R₂ j).rd.isRead p ∧ ((zs R ω t R₂ j).rd.openings R' path p).Conflict

theorem LeafColl.collision {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {R : Reg}
    {ω : L.Ω} {t : List (Cm × I.Coin)} {R₂ : Reg₂} (h : LeafColl zs R ω t R₂) : Merkle.Collision H512 := by
  obtain ⟨_, _, _, _, -, hc⟩ := h
  obtain ⟨h1, h2⟩ := FlockSoundness.Registered.TwoOpenings.collides _ hc
  exact ⟨_, _, h1, h2⟩

/-- **The table leaves' finder**: one run of the recursive audit against the prover `σ`, output the pair of a session's
two conflicting openings at the outcome's history and registration (`LeafColl`), else nothing. -/
noncomputable def leafFinder (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ) (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N)) :
    Finder Pair :=
  oneRun _ σ (onePick H512 (fun o : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) =>
    LeafColl zs o.1 o.2.1 o.2.2.1 o.2.2.2.1) fun _ h => LeafColl.collision h)

/-- **A session's read is off the table's leaf at most `qL²/2^513` likely** under `cr/sha-512` for `leafFinder` at the
budget `qL`: once per audit, whatever the number of sessions and reads. -/
theorem prob_leafColl_le (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ) (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    {qL : ℝ} (hL : (leafFinder zs dg N σ).CR H512 qL) :
    prob (fun o => LeafColl zs o.1 o.2.1 o.2.2.1 o.2.2.2.1) (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      ENNReal.ofReal (qL ^ 2 / 2 ^ (513 : ℝ)) := by
  have hev : (prob (fun o => LeafColl zs o.1 o.2.1 o.2.2.1 o.2.2.2.1)
      (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ).toReal ≤ qL ^ 2 / 2 ^ (513 : ℝ) := by
    rw [prob_toReal]
    exact (expect_one_le H512 _ σ (fun o : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) =>
      LeafColl zs o.1 o.2.1 o.2.2.1 o.2.2.2.1) fun _ h => LeafColl.collision h).trans hL.le
  exact (ENNReal.le_ofReal_iff_toReal_le (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))
    (by positivity)).2 hev

end FlockSoundness.Discharge.Recursion
