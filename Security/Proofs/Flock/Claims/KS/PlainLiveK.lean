import Proofs.Flock.Claims.KS.PlainLive

/-!
# The plain simulator's schedule at `VK keep`

`PlainSched`'s `PlainSchedAlign` and `plainSlotAlign_of_sched`, `PlainLead`'s `leadsAfter_plainSimAt` and `PlainLive`'s
`plainSchedAlign`, over the verifier `VK keep`:

* `PlainSchedAlignK`, `plainSlotAlign_of_schedK`: `PlainSlotAlignK` holds where the simulated strategy leads.
* `leadsAfter_plainSimAtK`, `plainSchedAlignK`: it leads at every statement.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Claims.Bridge.Live
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- `PlainSchedAlign` at `VK keep`. -/
def PlainSchedAlignK (keep : Bool → Bool) (x : Claims.Msg) : Prop :=
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
        (plainSimAtK keep L m k0 l₀ l₁ rest ext hsch hm hlay ins.read.I d x
          (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
          (plain_hdecAtK keep hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω C u)) ∧
    (∀ (hh : ins.read.I.tags.hiddenOutputs = true) (hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1"), ∀ C u,
      LeadsAfter execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
        ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm PT_LOCAL spec.mPts (capOfRoot (plainRoot P x ω)[0]!) (repRounds false spec st)
        (plainSimAtK keep L m k0 l₀ l₁ rest ext hsch hm hlay ins.read.I d x
          (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
          (plain_hdecHAtK keep hs hin hset h hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω C u))

/-- `plainSlotAlign_of_sched` at `VK keep`. -/
theorem plainSlotAlign_of_schedK (keep : Bool → Bool) {x : Claims.Msg} (h : PlainSchedAlignK keep x) : PlainSlotAlignK keep x :=
  fun s hs hin ins spec sts hset st ht d hd L m k0 l₀ l₁ rest ext hsch hk0 hm hlay Ω P ω =>
    ⟨fun hh C r ℓ u => slotRSL_lead L (plain_mPts hs hin hset).2.2 _ _ _ _ m k0 l₀ l₁ rest ext hsch hk0 hm _ _ _
        ((h s hs hin ins spec sts hset st ht d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω P ω).1 hh C u) r ℓ.val,
      fun hh hl C r ℓ u => slotRSL_lead L (plain_mPts hs hin hset).2.2 _ _ _ _ m k0 l₀ l₁ rest ext hsch hk0 hm _ _ _
        ((h s hs hin ins spec sts hset st ht d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω P ω).2 hh hl C u) r ℓ.val⟩

/-- `leadsAfter_plainSimAt` at `VK keep`. -/
theorem leadsAfter_plainSimAtK (keep : Bool → Bool) {spec : SessionSpec} {st : Stmt} (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level)
    (rest : List Accounting.Level) (ext : ℕ) (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (x : Claims.Msg) (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : PlainHDecK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn) (hw : 2 * spec.mPts ≤ 65536) {Ω : Type}
    (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽) (u : plainTape x) {n : ℕ}
    (hS : ∀ pts, SchedL (plainDg spec) (plainLiveK keep x spec st hw P ω u).1 pts (plainN x) (fun _ _ => []) n
      ((plainLiveK keep x spec st hw P ω u).2 pts)) :
    LeadsAfter execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st))
      ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm PT_LOCAL spec.mPts (capOfRoot (plainRoot P x ω)[0]!) n
      (plainSimAtK keep L m k0 l₀ l₁ rest ext hsch hm hlay I d x opn hdec hw P ω C u) :=
  leadsAfter_simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
    (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ _ opn (plainDg spec)
    (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
    (fun rs _ => capOfRoot rs[0]!) (plainN x) (hdec C (plainRoot P x ω)) (fun _ => rfl) (plainRoot P x ω)
    (plainLiveK keep x spec st hw P ω u) (plainSim_rootK keep P x spec st ω hw u) hS

/-- `plainSchedAlign` at `VK keep`. -/
theorem plainSchedAlignK (keep : Bool → Bool) (x : Claims.Msg) : PlainSchedAlignK keep x := by
  intro s hs hin ins spec sts hset st ht d hd L m k0 l₀ l₁ rest ext hsch hm hlay Ω P ω
  have hS := fun u => plainLive_schedLK keep hset ht hsch (by simp) (plain_mPts hs hin hset).1 P ω u
  refine ⟨fun hh C u => ?_, fun hh hl C u => ?_⟩
  · exact leadsAfter_plainSimAtK keep L m k0 l₀ l₁ rest ext hsch hm hlay ins.read.I d x
      (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
      (plain_hdecAtK keep hs hin hset ht hh hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω C u (hS u)
  · exact leadsAfter_plainSimAtK keep L m k0 l₀ l₁ rest ext hsch hm hlay ins.read.I d x
      (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
      (plain_hdecHAtK keep hs hin hset ht hh hl hd L m k0 l₀ l₁ rest ext hsch hm hlay) (plain_mPts hs hin hset).1 P ω C u
      (hS u)

end FlockDeploy.KS
