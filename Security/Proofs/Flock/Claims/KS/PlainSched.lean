import Proofs.Flock.Claims.KS.SlotLead
import Proofs.Flock.Claims.KS.RepDistDeploy
import Proofs.Flock.Claims.KS.PlainAlign

/-!
# The simulated fork's alignment, from the reps' schedule

`SlotRSL` asks that the strategy after `Commit` reach rep `r`'s level-`ℓ + 1` table node after `forkRound` coins. The
link coin is the first of them; below it the session is one batch position, the table's lockstep after the link points
(`lockAfterL`), whose reps run interleaved and meet the slot after `dist_lockRep_plain`'s count. A strategy that
advances rep 0 at every choice until it ends after `repRounds` coins, or plays a part every reached outcome of which
opens only empty rows, meets rep 0's slots after that count and rep 1's after `repRounds` more:

* `afterOf`: the table's strategy after the link points, read off the session's;
* `NulOut`, `LeadsAfter`: an outcome opening only empty rows, and the strategy leading with rep 0 (`Lock.Lead`);
* `slotRSL_lead`: so `SlotRSL` holds at `forkRound spec st k0 r ℓ`;
* `PlainSchedAlign`, `plainSlotAlign_of_sched`: so `PlainSlotAlign` holds where the simulated strategy leads with
  rep 0 for `repRounds` coins.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **The table's strategy after the link points `pts`**, read off a strategy of the session after `Commit` through
its one batch position. -/
noncomputable def afterOf {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] (A : Model.Arith 𝔽 K)
    (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (S : Model.Statement)
    (sch : FlockSoundness.Accounting.Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (cap₀ : Model.Cap D)
    (σ : Strategy (sessAfterL A Hc E Lf S sch hm ptLocal mPts [] [] cap₀)) (pts : Fin 2 → Fin mPts → 𝔽) :
    Strategy (tableAfterL A Hc E Lf S sch hm ptLocal mPts cap₀ pts) :=
  Strategy.ofMap (·, []) _ (cast (congrArg Strategy (Lock.interleave_ret [] _))
    (Strategy.ofMap (fun x : Model.OutCL 𝔽 D Sl × List (Model.OutCL 𝔽 D Sl) => x.1 :: x.2)
      ((tableAfterL A Hc E Lf S sch hm ptLocal mPts cap₀ pts).interleave (.ret [])) (σ pts)))

/-- **An outcome opening only empty rows.** -/
def NulOut {D Sl : Type} (o : Model.OutCL 𝔽 D Sl) : Prop :=
  ∀ r ℓ i, (o.opens r ℓ i).1 = []

/-- **The strategy after `Commit` leads with rep 0 for `n` coins**: at every link point, at every choice of the reps'
interleaving it advances rep 0, which ends after exactly `n` coins, until rep 1 has ended, except below a part of the
run every reached outcome of which opens only empty rows (`Lock.Lead`). -/
def LeadsAfter {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] (A : Model.Arith 𝔽 K)
    (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (S : Model.Statement)
    (sch : FlockSoundness.Accounting.Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (cap₀ : Model.Cap D) (n : ℕ)
    (σ : Strategy (sessAfterL A Hc E Lf S sch hm ptLocal mPts [] [] cap₀)) : Prop :=
  ∀ pts, Lock.Lead _ NulOut n (Model.repC A 0 S sch hm (Model.extraClaims S ptLocal mPts pts))
    (Model.repC A 1 S sch hm (Model.extraClaims S ptLocal mPts pts)) (afterOf A Hc E Lf S sch hm ptLocal mPts cap₀ σ pts)

section Lead

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
  {spec : SessionSpec} {st : Stmt} (L : Layout st) (hp : spec.points = 2) (A : Model.Arith 𝔽 K)
  (Hc : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D Hc) (m k0 : ℕ)
  (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
  (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
  (hm : 13 ≤ (L.model (Refine.regionsOf st)).m) (ptLocal mPts : ℕ) (cap₀ : Model.Cap D)
  {σ : Strategy (sessAfterL A Hc E Lf (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts
    [] [] cap₀)}

include hp hsch

/-- **Rep 0's level-`ℓ + 1` table node is met after `forkRound spec st k0 0 ℓ` coins** by a strategy leading with
rep 0: the link coin, then `dist_lockRep_plain`'s count in rep 0, with no node of the slot in rep 1. -/
theorem slotRSL_lead_zero {n : ℕ}
    (hS : LeadsAfter A Hc E Lf (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts cap₀ n σ)
    (ℓ : ℕ) :
    SlotRSL A Hc E Lf (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts cap₀ 0 ℓ
      (forkRound spec st k0 0 ℓ) σ := by
  have e : forkRound spec st k0 0 ℓ = preLig spec st + capRound k0 (ℓ + 1) + 1 := by
    rw [forkRound_eq]
    omega
  rw [e]
  refine Lock.SlotAt.coin _ fun pts => ?_
  refine Lock.SlotAt.mono _ ?_
  refine Lock.SlotAt.batchLock_one _ _ _ ?_
  refine Lock.SlotAt.mapL _ _ ?_
  refine Lock.SlotAt.lead_fst disjoint_reps ?_ (fun _ _ _ _ => .send _ fun _ => .ret _) (hS pts) _ _
    (dist_lockRep_plain L hp A 0 (by norm_num) m k0 l₀ l₁ rest ext hsch hk0 hm default ptLocal mPts pts ℓ)
    (Lock.noSlot_of_get_none (fun i => ?_) _)
  · intro d b b' hb hb'
    exact ite_eq_right fun ⟨p, hp⟩ => not_conflictAt_nul Hc E Lf _ 0 (ℓ + 1) d (b := [b]) (b' := [b']) hb hb' p hp
  · unfold getRep slotOf
    exact ite_eq_right (by omega)

/-- **Rep 1's level-`ℓ + 1` table node is met after `forkRound spec st k0 1 ℓ` coins** by a strategy leading with
rep 0 for `repRounds` coins: the link coin, rep 0's coins, then `dist_lockRep_plain`'s count in rep 1, with no node of
the slot in rep 0. -/
theorem slotRSL_lead_one
    (hS : LeadsAfter A Hc E Lf (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts cap₀
      (repRounds false spec st) σ)
    (ℓ : ℕ) :
    SlotRSL A Hc E Lf (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts cap₀ 1 ℓ
      (forkRound spec st k0 1 ℓ) σ := by
  have e : forkRound spec st k0 1 ℓ = repRounds false spec st + (preLig spec st + capRound k0 (ℓ + 1)) + 1 := by
    rw [forkRound_eq]
    omega
  rw [e]
  refine Lock.SlotAt.coin _ fun pts => ?_
  refine Lock.SlotAt.mono _ ?_
  refine Lock.SlotAt.batchLock_one _ _ _ ?_
  refine Lock.SlotAt.mapL _ _ ?_
  refine Lock.SlotAt.lead_snd disjoint_reps ?_ (fun _ _ _ _ => .send _ fun _ => .ret _) (hS pts) _ _
    (Lock.noSlot_of_get_none (fun i => ?_) _)
    (dist_lockRep_plain L hp A 1 (by norm_num) m k0 l₀ l₁ rest ext hsch hk0 hm default ptLocal mPts pts ℓ)
  · intro d b b' hb hb'
    exact ite_eq_right fun ⟨p, hp⟩ => not_conflictAt_nul Hc E Lf _ 1 (ℓ + 1) d (b := [b]) (b' := [b']) hb hb' p hp
  · unfold getRep slotOf
    exact ite_eq_right (by omega)

/-- **Each rep's level-`ℓ + 1` table node is met after `forkRound` coins** by a strategy leading with rep 0 for
`repRounds` coins. -/
theorem slotRSL_lead
    (hS : LeadsAfter A Hc E Lf (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts cap₀
      (repRounds false spec st) σ) :
    ∀ (r : Fin 2) (ℓ : ℕ), SlotRSL A Hc E Lf (L.model (Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts
      cap₀ r.val ℓ (forkRound spec st k0 r.val ℓ) σ
  | ⟨0, _⟩, ℓ => slotRSL_lead_zero L hp A Hc E Lf m k0 l₀ l₁ rest ext hsch hk0 hm ptLocal mPts cap₀ hS ℓ
  | ⟨1, _⟩, ℓ => slotRSL_lead_one L hp A Hc E Lf m k0 l₀ l₁ rest ext hsch hk0 hm ptLocal mPts cap₀ hS ℓ

end Lead

/-- **The simulated strategy leads with rep 0 for `repRounds` coins**: `LeadsAfter` against `plainSimAt`, at every
plain statement in scope, its one table's `fast100` schedule and the plain decoders where the outputs are public and
where they are hidden under hm96 leaves. -/
def PlainSchedAlign (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf false x = some (ins, spec, sts)) (st : Stmt)
    (h : tables false x = [(spec, st)]) (d : Json) (hd : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω),
    (∀ hh : ins.read.I.tags.hiddenOutputs = false, ∀ C u,
      LeadsAfter execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm PT_LOCAL spec.mPts (capOfRoot (plainRoot P x ω)[0]!) (repRounds false spec st)
        (plainSimAt L m k0 l₀ l₁ rest ext hsch hm hlay ins.read.I d x
          (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
          (plain_hdecAt hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω C u)) ∧
    (∀ (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1"), ∀ C u,
      LeadsAfter execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm PT_LOCAL spec.mPts (capOfRoot (plainRoot P x ω)[0]!) (repRounds false spec st)
        (plainSimAt L m k0 l₀ l₁ rest ext hsch hm hlay ins.read.I d x
          (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
          (plain_hdecHAt hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω C u))

theorem plainSlotAlign_of_sched {x : Claims.Msg} (h : PlainSchedAlign x) : PlainSlotAlign x :=
  fun s hs hin ins spec sts hset st ht d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay Ω P ω =>
    ⟨fun hh C r ℓ u => slotRSL_lead L (plain_mPts hs hin hset).2.2 _ _ _ _ m k0 l₀ l₁ rest ext hsch hk0 hm _ _ _
        ((h s hs hin ins spec sts hset st ht d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω P ω).1 hh C u) r ℓ.val,
      fun hh hl C r ℓ u => slotRSL_lead L (plain_mPts hs hin hset).2.2 _ _ _ _ m k0 l₀ l₁ rest ext hsch hk0 hm _ _ _
        ((h s hs hin ins spec sts hset st ht d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω P ω).2 hh hl C u) r ℓ.val⟩

end FlockDeploy.KS
