import Proofs.Flock.Claims.KS.PlainAlign
import Proofs.Flock.Claims.KS.OffAlign

/-!
# The plain slot's off mass, after the slot's coins

`advRSL_le_forkC` moves the session's fork at rep `r`'s level-`ℓ + 1` table node to the fork after `j` coins where the
strategy reaches that node after `j` coins (`SlotRSL`). The same alignment moves the slot's raw term of
`session_sound_of_table_accL_nodeOff`, the per-position off mass at the node at `plurSL`, to the session's plurality
off mass after `j` coins at the cap table 0's output records (`opCapL`):

* `opCapL`: rep `r`'s level-`ℓ + 1` openings at the recorded cap;
* `slotOffRSL_le_offC`: at any level-0 table, by `Lock.nodeE_congr` to `SlotRSL`'s lockstep and
  `Lock.nodeE_nodeOff_le_offC`, whose openings agree with `opCapL`'s on the outcomes the session's leaves relate
  (`capAtL_of_sessRelL`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Exec

local notation "𝔽" => FlockLevel3.GF128

/-- **Rep `r`'s level-`ℓ + 1` openings at the cap table 0's output records**: `opRL` at the recorded cap, and nothing
where none is recorded. -/
noncomputable def opCapL {D Sl : Type} [Inhabited D] [Inhabited Sl] (Hc : List UInt8 → D)
    (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (sch : FlockSoundness.Accounting.Schedule) (r ℓ : ℕ)
    (outs : List (Model.OutCL 𝔽 D Sl)) :
    Fin (2 ^ (lvOf sch (ℓ + 1)).logLen) → Option (Fin (2 ^ (lvOf sch (ℓ + 1)).lanes) → 𝔽) := fun p =>
  ((if r = 0 then (outs.getD 0 dfltOutL).pend.1 else (outs.getD 0 dfltOutL).pend.2).caps[ℓ]?).bind fun d =>
    opRL Hc E Lf sch r (ℓ + 1) d (outs.getD 0 dfltOutL) p

/-- **The slot's off mass at its node is at most the session's plurality off mass after `j` coins** at the recorded
cap, at any level-0 table, where the strategy reaches rep `r`'s level-`ℓ + 1` table node after `j` coins. -/
theorem slotOffRSL_le_offC {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    (A : Model.Arith 𝔽 K) (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D Hc)
    (S : Model.Statement) (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (cap₀ : Model.Cap D) (T : Model.Oracle 𝔽)
    (r : Fin 2) (ℓ j : ℕ)
    (σ : Strategy (sessAfterL A Hc E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts [] [] cap₀))
    (h : SlotRSL A Hc E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts cap₀ r.val ℓ j σ) :
    Lock.nodeE (plurSL Hc E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0) (slotOf r.val (ℓ + 1))
        (Lock.nodeOff (fun d (outs : List (Model.OutCL 𝔽 D Sl)) =>
            opRL Hc E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val (ℓ + 1) d (outs.getD 0 dfltOutL))
          (rowT (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ + 1)).lanes
            (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ + 1)).logLen)
          (plurSL Hc E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0) (slotOf r.val (ℓ + 1)))
        (lockSessL A Hc E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts [] [] T cap₀) σ ≤
      offC (opCapL Hc E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val ℓ) j _ σ := by
  rw [Lock.nodeE_congr _ _ _ (aligned_lockSessL A Hc E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts [] [] T default
    cap₀) σ]
  exact Lock.nodeE_nodeOff_le_offC _ _ _ _
    (fun d g'' σ'' => plurS_pluralityL Hc E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0 r.val (ℓ + 1) r.isLt d g'' σ'')
    _ (fun t d i a b hR hg p v hv => by
      have hcap := capAtL_of_sessRelL hR hg (p := p) (by rw [hv]; rfl)
      unfold CapAtL at hcap
      show (((if r.val = 0 then (b.getD 0 dfltOutL).pend.1 else (b.getD 0 dfltOutL).pend.2).caps[ℓ]?).bind
        fun d => opRL Hc E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r.val (ℓ + 1) d (b.getD 0 dfltOutL) p) = some v
      rw [hcap, Option.bind_some]
      exact hv) h

end FlockDeploy.KS
