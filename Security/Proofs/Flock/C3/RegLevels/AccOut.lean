import Proofs.Flock.C3.RegLevels.OutRegs
import Proofs.Flock.C3.RegLevels.AccReads

/-!
# n11: `AccOutL` (`HAccRows`' second half), from each opening's registered `acc_out` read

`AccOutL` (`Family/Ends.lean`) puts `V`'s next row in the layer's `acc_out` at each opening. At the verifier's unit,
`accOutRegs_of_check` registers copy 0's `acc_out` columns as the bits of the row its `acc_out` read holds; where that
read is of `rec-acc`'s registration `s` at the opening's next chain position (`OutKeysS`), `AccRowsAreS` makes the row
`V`'s there, and `accOut_of_regOut` reads it off the layer (`accOutL_of_keysS`). `OutsLaid`, V*'s output layout, is a
binder.

`outRead_drawn` is `OutKeysS`' registration and position at a drawn unit (`accRead_drawn` at `acc_out`, V*'s port
`nIn + 1`): the read's port has `acc0`'s registration and reads `accReads`' row of its index, one on.
-/

namespace FlockC3.RegLevels

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open FlockVBridge.Family
open FlockVBridge.AccChain (levelLanes)
open FlockSoundness.Discharge.Hm (bitL)
open FlockSoundness.Discharge.ZkReg (LiveAcceptsZKR)

section Drawn

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)

/-- **A drawn unit's `acc_out` read, from the verifier's checks** (`accRead_drawn` at `acc_out`, V*'s port `nIn + 1`
by `OutsLaid`): the read's port has `acc0`'s registration in the verifier's shared rows, and the read is one past
`accReads`' row at some index. -/
theorem outRead_drawn {ω : Lv.Ω} {R' : Reg'} {o}
    (hacc : LiveAcceptsZKR zo.I zo.own (R', Lv.draw ω, zo.wr (Lv.draw ω) R' o)) {u : Fin nv} (hu : u ∈ Lv.draw ω)
    {j : ℕ} (hlaid : OutsLaid c j) (hq : HmOut.nIn c + 1 < c.ports.size) (hr : zo.rd.isRead (.inl (u, ⟨_, hq⟩)))
    {i : Flock.RecVStar.Inner} {s : VStar.Stmt} {rows : Option (Array Flock.Registered.Port)}
    (hck : Flock.RecVStar.checkAcc s zo.own rows = .ok ()) (hrd : Flock.RecVStar.checkAccReads i s zo.own = .ok ())
    (hl : Flock.RecVStar.accPorts s = ["acc", "acc_out"]) {want : Array (Option ℕ)}
    (hw : Flock.RecVStar.accReads i s = .ok (some want))
    (hnd : ∀ p ∈ zo.own, ∀ p' ∈ zo.own, p.name = p'.name → p = p') :
    (∃ rws w, rows = some rws ∧ rws.find? (·.name == "acc0") = some w ∧
        site (zo.rd.port (.inl (u, ⟨_, hq⟩))) = site w) ∧
      ∃ x p, want[x]? = some (some p) ∧ zo.rd.readAt (.inl (u, ⟨_, hq⟩)) = p + 1 := by
  obtain ⟨-, -, -, -, -, hn1, -⟩ := hlaid
  obtain ⟨-, hsite, x, hx⟩ := accRead_drawn zo hacc hu hr hck hrd hl hw hnd (nm := "acc_out") (d := 1) (by simp) hn1
  refine ⟨hsite, x, ?_⟩
  rw [Array.getElem?_map] at hx
  cases hw' : want[x]? with
  | none => simp [hw'] at hx
  | some a =>
    rw [hw'] at hx
    cases a with
    | none => simp at hx
    | some p => exact ⟨p, hw', (Option.some.inj (Option.some.inj hx)).symm⟩

end Drawn

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)

/-- **Each opening's `acc_out` row is read at registration `s`, at its next chain position**: at every accepted outer
proof, unit `unO ℓ r j`'s row `nIn + 1` (V*'s `acc_out`, `OutsLaid`) is a registered read of a port with registration
`s`, at the row after the opening's (rec_outer.py:238, 242: `acc_out` reads `rec-acc`'s row `p + 1`). Its registration
and position are `outRead_drawn`'s where `s` is `acc0`'s and `accReads`' rows are the chain's. -/
def OutKeysS (s : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Site) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r : Fin 2, ∀ ℓ < F.sh.sched.levels.size,
    ∀ j < F.q ℓ, ∃ hq : HmOut.nIn (F.Vs (F.sessO ℓ)).c + 1 < (F.Vs (F.sessO ℓ)).c.ports.size,
      (F.zs R ω t R₂ (F.sessO ℓ)).rd.isRead (.inl (F.unO ℓ r j, ⟨_, hq⟩)) ∧
        site ((F.zs R ω t R₂ (F.sessO ℓ)).rd.port (.inl (F.unO ℓ r j, ⟨_, hq⟩))) = s R ω t R₂ ∧
        (F.zs R ω t R₂ (F.sessO ℓ)).rd.readAt (.inl (F.unO ℓ r j, ⟨_, hq⟩)) =
          r.val * F.nOpen + ((List.range ℓ).map F.q).sum + j + 1

/-- **`AccOutL` from the openings' `acc_out` reads** (`accOutRegs_of_check`, `AccRowsAreS`, `accOut_of_regOut`): where
each level's circuit has V*'s output layout at copy 0 (`hlaid`) and each opening's `acc_out` row is read at
registration `s` at its next chain position (`OutKeysS`), the layer's `acc_out` is the first two elements of `V`'s row
there, wherever the registrant's rows of `s` are `V`'s (`AccRowsAreS`). -/
theorem accOutL_of_keysS (hU : F.HOpenUnits) (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128)
    (s : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Site) (hacc : F.AccRowsAreS dg N k Rw V s)
    (hlaid : ∀ ℓ < F.sh.sched.levels.size, OutsLaid (F.Vs (F.sessO ℓ)).c 0) (hk : OutKeysS F dg N k Rw s) :
    F.AccOutL dg N k Rw V := by
  intro R ω t R₂ τ o h r ℓ hℓ j hj i hi
  obtain ⟨hup, -, H, hp⟩ := hU ℓ hℓ
  have hp' : Flock.RecOpen.params (F.Vs (F.sessO ℓ)).c.unit.name = some (levelLanes F.sh ℓ, F.depth ℓ) := by
    unfold VStarFamily.depth
    rw [hp]
    rfl
  obtain ⟨hq, hrd, hsite, hat⟩ := hk R ω t R₂ τ o h r ℓ hℓ j hj
  refine accOut_of_regOut (F.zs R ω t R₂ (F.sessO ℓ)) (h.1 (F.sessO ℓ)).2 hup hp'
    (A := V R ω t R₂ (r.val * F.nOpen + ((List.range ℓ).map F.q).sum + j + 1))
    (accOutRegs_of_check (F.zs R ω t R₂ (F.sessO ℓ)) (parse_facts (F.zs R ω t R₂ (F.sessO ℓ)).hc).recNet hp' hup
      (hlaid ℓ hℓ) hq hrd fun b hb hb' => ?_) i hi
  rw [← hat]
  exact hacc R ω t R₂ τ o h (F.sessO ℓ) (F.unO ℓ r j) ⟨_, hq⟩ hrd hsite b hb hb'

end family

end FlockC3.RegLevels
