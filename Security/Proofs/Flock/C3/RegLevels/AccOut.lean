import Proofs.Flock.C3.RegLevels.OutRegs
import Proofs.Flock.C3.RegLevels.AccReads
import Proofs.Flock.Recursive.HPortTie

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
    (hw : Flock.RecVStar.accReads i s = .ok (some want)) :
    (∃ rws w, rows = some rws ∧ rws.find? (·.name == "acc0") = some w ∧
        site (zo.rd.port (.inl (u, ⟨_, hq⟩))) = site w) ∧
      ∃ x p : ℕ, want[x]? = some (some p) ∧ zo.rd.readAt (.inl (u, ⟨_, hq⟩)) = p + 1 := by
  obtain ⟨-, -, hq1, -, -, hn1, -⟩ := hlaid
  obtain ⟨-, hsite, x, hx⟩ := accRead_drawn zo hacc hu hr hck hrd hl hw (nm := "acc_out") (d := 1) (by simp) hn1
  refine ⟨hsite, x, ?_⟩
  rw [Array.getElem?_map] at hx
  cases hw' : want[x]? with
  | none => simp [hw'] at hx
  | some a =>
    rw [hw'] at hx
    cases a with
    | none => simp at hx
    | some p => exact ⟨p, rfl, (Option.some.inj (Option.some.inj hx)).symm⟩

end Drawn

/-- **Rep `r`'s opening `j` in `accReads`' rows** (`chain_order`, rep-major): unit `r · q + j` (`j < q`) reads row
`r · n + b + j`. -/
theorem accRow_opening {q n b r j : ℕ} (hj : j < q) : (r * q + j) / q * n + b + (r * q + j) % q = r * n + b + j := by
  have hq : 0 < q := by omega
  have h1 : (r * q + j) / q = r := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hq, Nat.div_eq_of_lt hj, Nat.zero_add]
  have h2 : (r * q + j) % q = j := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj]
  rw [h1, h2]

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

/-- Obligation: discharged where the verifier checked the level's statement (`checkAcc`, `checkAccReads`:
`outRead_drawn`, `accPos_of_checked`). **Each opening's `acc_out` row is read at registration `s`, one past its unit's
row of `accReads`** (`u / q · n + Σ_{ℓ' < ℓ} q + u % q`). -/
def OutReadsS (s : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Site) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r : Fin 2, ∀ ℓ < F.sh.sched.levels.size,
    ∀ j < F.q ℓ, ∃ hq : HmOut.nIn (F.Vs (F.sessO ℓ)).c + 1 < (F.Vs (F.sessO ℓ)).c.ports.size,
      (F.zs R ω t R₂ (F.sessO ℓ)).rd.isRead (.inl (F.unO ℓ r j, ⟨_, hq⟩)) ∧
        site ((F.zs R ω t R₂ (F.sessO ℓ)).rd.port (.inl (F.unO ℓ r j, ⟨_, hq⟩))) = s R ω t R₂ ∧
        (F.zs R ω t R₂ (F.sessO ℓ)).rd.readAt (.inl (F.unO ℓ r j, ⟨_, hq⟩)) =
          (F.unO ℓ r j).val / F.q ℓ * F.nOpen + ((List.range ℓ).map F.q).sum + (F.unO ℓ r j).val % F.q ℓ + 1

/-- **`OutKeysS` from the units' rows** (the unit-to-`(r, j)` map): rep `r`'s opening `j` of level `ℓ` is unit
`r · q + j` (`hunO`, within the level's `2 q` instances, `hnv`), whose row is the opening's chain position
(`accRow_opening`). -/
theorem outKeysS_of_reads (hnv : ∀ ℓ < F.sh.sched.levels.size, 2 * F.q ℓ ≤ (F.Vs (F.sessO ℓ)).nv)
    {s : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Site} (h : OutReadsS F dg N k Rw s) : OutKeysS F dg N k Rw s := by
  intro R ω t R₂ τ o ha r ℓ hℓ j hj
  obtain ⟨hq, hrd, hsite, hat⟩ := h R ω t R₂ τ o ha r ℓ hℓ j hj
  refine ⟨hq, hrd, hsite, ?_⟩
  have hlt : r.val * F.q ℓ + j < (F.Vs (F.sessO ℓ)).nv := by
    have := hnv ℓ hℓ
    rcases (show r.val = 0 ∨ r.val = 1 by omega) with h0 | h0 <;> rw [h0] <;> omega
  have e := F.hunO ℓ r j hlt
  rw [show (F.sh.sched.levels[ℓ]!).q = F.q ℓ from rfl] at e
  rw [hat, e, accRow_opening hj]

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

/-- **Level `ℓ`'s statement is a `.recOpen` of `2 q` instances** (`statements_recOpen` at the verifier's schedule,
`sched_eq`), so its session has `2 q` units (`hnv`). -/
theorem open_stmt {ℓ : ℕ} (hℓ : ℓ < F.sh.sched.levels.size) :
    (∃ LANES H, F.stmts[F.sessO ℓ] = .recOpen ℓ LANES H (2 * F.q ℓ)) ∧ (F.Vs (F.sessO ℓ)).nv = 2 * F.q ℓ := by
  obtain ⟨s, -, -, hs, -, -, -⟩ := statements_eq F.hstmts
  have hss := F.sched_eq hs
  subst hss
  obtain ⟨hj, hst⟩ := statements_recOpen F.hstmts hs hℓ
  rw [F.hsessO ℓ ⟨ℓ, hj⟩ ⟨_, _, _, hst⟩]
  exact ⟨⟨_, _, hst⟩, (F.hnv _).trans (congrArg VStar.Stmt.instances hst)⟩

/-- Obligation: discharged at C3's family by `readsOwnAt_of` (`ReachedChecked`: `readsOwn_of_hown` and the label
check). **Each level's registered reads are of `own`'s entry of the port's name, at the unit's own row of its
reads** (`ReadsOwnAt` at the openings' sessions). -/
def ReadsOwnO : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ ℓ < F.sh.sched.levels.size,
    ∀ (u : Fin (F.Vs (F.sessO ℓ)).nv) (q : Fin (F.Vs (F.sessO ℓ)).c.ports.size),
      (F.zs R ω t R₂ (F.sessO ℓ)).rd.isRead (.inl (u, q)) →
      (F.zs R ω t R₂ (F.sessO ℓ)).rd.port (.inl (u, q)) ∈ (F.zs R ω t R₂ (F.sessO ℓ)).own ∧
        ((F.zs R ω t R₂ (F.sessO ℓ)).rd.port (.inl (u, q))).name = (F.Vs (F.sessO ℓ)).c.ports[q].name ∧
        ((F.zs R ω t R₂ (F.sessO ℓ)).rd.port (.inl (u, q))).reads[u.val]? =
          some (some ((F.zs R ω t R₂ (F.sessO ℓ)).rd.readAt (.inl (u, q))))

/-- Obligation: discharged by the verifier's check of each level's statement (`checkAcc_of_checkStmt`, from
`ChecksAccepted`). **`checkAcc` and `checkAccReads` pass at each level's statement**, at its session's reads file and
the shared rows `rows`. -/
def LevelsAccChecked (i : Flock.RecVStar.Inner) (rows : Option (Array Flock.Registered.Port)) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ ℓ < F.sh.sched.levels.size,
    Flock.RecVStar.checkAcc F.stmts[F.sessO ℓ] (F.zs R ω t R₂ (F.sessO ℓ)).own rows = .ok () ∧
      Flock.RecVStar.checkAccReads i F.stmts[F.sessO ℓ] (F.zs R ω t R₂ (F.sessO ℓ)).own = .ok ()

/-- **`OutReadsS` at `acc0`'s registration, from the verifier's checks of the levels** (`accPort_of_checks` at
`acc_out`): V*'s `acc_out` is port `nIn + 1` (`OutsLaid`), a port `own` names, so its read is registered
(`RowsRegistered`) and is of `own`'s `acc_out` entry at the unit's row (`ReadsOwnO`), which has `acc0`'s registration
and reads `accReads`' rows (`accReads_recOpen` at the inner schedule, `hsch`), one on. -/
theorem outReadsS_of_checks (hreg : F.RowsRegistered dg N k Rw) (hro : ReadsOwnO F dg N k Rw)
    {i : Flock.RecVStar.Inner} (hsch : Flock.fast100 i.m = some F.sh.sched)
    {rows : Option (Array Flock.Registered.Port)} (hck : LevelsAccChecked F dg N k Rw i rows)
    {rws : Array Flock.Registered.Port} {w : Flock.Registered.Port} (hrows : rows = some rws)
    (hw : rws.find? (·.name == "acc0") = some w)
    (hlaid : ∀ ℓ < F.sh.sched.levels.size, OutsLaid (F.Vs (F.sessO ℓ)).c 0) :
    OutReadsS F dg N k Rw fun _ _ _ _ => site w := by
  intro R ω t R₂ τ o ha r ℓ hℓ j hj
  obtain ⟨⟨LANES, H, hst⟩, hnv⟩ := open_stmt F hℓ
  obtain ⟨-, -, hq, -, -, hn1, -⟩ := hlaid ℓ hℓ
  obtain ⟨hca, hcr⟩ := hck R ω t R₂ τ o ha ℓ hℓ
  have hl : Flock.RecVStar.accPorts F.stmts[F.sessO ℓ] = ["acc", "acc_out"] :=
    (congrArg Flock.RecVStar.accPorts hst).trans rfl
  have hwnt := (congrArg (Flock.RecVStar.accReads i) hst).trans (accReads_recOpen hsch ℓ LANES H (2 * F.q ℓ))
  obtain ⟨p, hf, -⟩ := value_of_checkAcc hca "acc_out" (by rw [hl]; simp)
  have hpn : p.name = "acc_out" := by simpa using Array.find?_some hf
  have hrd : (F.zs R ω t R₂ (F.sessO ℓ)).rd.isRead (.inl (F.unO ℓ r j, ⟨_, hq⟩)) :=
    hreg R ω t R₂ τ o ha _ _ _ ⟨p, Array.mem_of_find?_eq_some hf, hpn.trans hn1.symm⟩
  obtain ⟨hmem, hname, hpos⟩ := hro R ω t R₂ τ o ha ℓ hℓ _ _ hrd
  obtain ⟨⟨rws', w', hrows', hw', hsite⟩, hreads⟩ :=
    accPort_of_checks hca hcr hl hwnt (nm := "acc_out") (d := 1) (by simp) hmem (hname.trans hn1)
  obtain rfl : rws' = rws := Option.some.inj (hrows'.symm.trans hrows)
  obtain rfl : w' = w := Option.some.inj (hw'.symm.trans hw)
  refine ⟨hq, hrd, hsite, ?_⟩
  have hlt : (F.unO ℓ r j).val < 2 * (F.sh.sched.levels[ℓ]!).q := lt_of_lt_of_eq (F.unO ℓ r j).isLt hnv
  rw [hreads, getElem?_accRow _ hlt] at hpos
  exact (Option.some.inj (Option.some.inj hpos)).symm

/-- **`OutKeysS` from the verifier's checks of the levels** (`outReadsS_of_checks`, `outKeysS_of_reads` at the level's
`2 q` units, `open_stmt`). -/
theorem outKeysS_of_checks (hreg : F.RowsRegistered dg N k Rw) (hro : ReadsOwnO F dg N k Rw)
    {i : Flock.RecVStar.Inner} (hsch : Flock.fast100 i.m = some F.sh.sched)
    {rows : Option (Array Flock.Registered.Port)} (hck : LevelsAccChecked F dg N k Rw i rows)
    {rws : Array Flock.Registered.Port} {w : Flock.Registered.Port} (hrows : rows = some rws)
    (hw : rws.find? (·.name == "acc0") = some w)
    (hlaid : ∀ ℓ < F.sh.sched.levels.size, OutsLaid (F.Vs (F.sessO ℓ)).c 0) :
    OutKeysS F dg N k Rw fun _ _ _ _ => site w :=
  outKeysS_of_reads F dg N k Rw (fun _ hℓ => (open_stmt F hℓ).2.ge)
    (outReadsS_of_checks F dg N k Rw hreg hro hsch hck hrows hw hlaid)

/-- **`AccOutL` from the verifier's checks of the levels** (`accOutL_of_keysS` at `outKeysS_of_checks`), wherever the
registrant's rows of `acc0`'s registration are `V`'s. -/
theorem accOutL_of_checks (hU : F.HOpenUnits) (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128)
    (hreg : F.RowsRegistered dg N k Rw) (hro : ReadsOwnO F dg N k Rw)
    {i : Flock.RecVStar.Inner} (hsch : Flock.fast100 i.m = some F.sh.sched)
    {rows : Option (Array Flock.Registered.Port)} (hck : LevelsAccChecked F dg N k Rw i rows)
    {rws : Array Flock.Registered.Port} {w : Flock.Registered.Port} (hrows : rows = some rws)
    (hw : rws.find? (·.name == "acc0") = some w) (hacc : F.AccRowsAreS dg N k Rw V fun _ _ _ _ => site w)
    (hlaid : ∀ ℓ < F.sh.sched.levels.size, OutsLaid (F.Vs (F.sessO ℓ)).c 0) :
    F.AccOutL dg N k Rw V :=
  accOutL_of_keysS F dg N k Rw hU V _ hacc hlaid (outKeysS_of_checks F dg N k Rw hreg hro hsch hck hrows hw hlaid)

end family

section C3

open FlockSoundness.Game FlockSoundness.Discharge.ZkSession FlockSoundness.Discharge.ZkExec FlockC3.Compose

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **An opening's `acc` (`acc_out`) read is at its chain position (the next one)**, at C3's family: where the verifier
checked the session's statements (`ChecksAccepted`), the table's level `l` (`.recOpen`) at the inner schedule `sch`,
and each accepted outcome is reached and passes candidate 3's checks at its own transcript (`ReachedChecked`), the read
of unit `r · q + j`, rep `r`'s opening `j`, is at row `r · n + opensBelow l + j` (plus `d`): `accRead_at_unit` at
`accReads_recOpen`'s rows, `accRow_opening`. -/
theorem accPos_of_checked {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂}
    {rows : Option (Array Flock.Registered.Port)} {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    (hchk : FlockC3.PubRows.ChecksAccepted F rows wr reg) {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    (hrc : FlockC3.HPort.ReachedChecked F dg N k Rw) {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) (s : Fin (vstarCount F.ish)) {l LANES H m : ℕ}
    (hs : F.stmts[(s : ℕ)]? = some (.recOpen l LANES H m)) {sch : Flock.Schedule}
    (hsch : Flock.fast100 F.ish.m = some sch) {r j : ℕ} (hr : r < 2) (hj : j < (sch.levels[l]!).q)
    {u : Fin (F.Vs s).nv} (hu : u.val = r * (sch.levels[l]!).q + j) {nm : String} {d : ℕ}
    (hnm : (nm, d) ∈ [("acc", 0), ("acc_out", 1)]) {q : Fin (F.Vs s).c.ports.size}
    (hq : (F.Vs s).c.ports[q].name = nm) (hrd : (F.zs R ω t R₂ s).rd.isRead (.inl (u, q))) :
    (F.zs R ω t R₂ s).rd.readAt (.inl (u, q)) =
      r * Flock.RecVStar.opensBelow sch sch.levels.size + Flock.RecVStar.opensBelow sch l + j + d := by
  obtain ⟨hre, _, _, _, _, _, hc3⟩ := hrc R ω t R₂ τ o h s
  obtain ⟨-, -, hpos⟩ := accRead_at_unit hchk h s hre hc3 hs rfl (accReads_recOpen hsch l LANES H m) hnm hq hrd
  have hlt : u.val < 2 * (sch.levels[l]!).q := by
    rw [hu]
    rcases (show r = 0 ∨ r = 1 by omega) with rfl | rfl <;> omega
  rw [getElem?_accRow _ hlt] at hpos
  have e := Option.some.inj (Option.some.inj hpos)
  have e2 := accRow_opening (n := Flock.RecVStar.opensBelow sch sch.levels.size)
    (b := Flock.RecVStar.opensBelow sch l) (r := r) hj
  rw [← hu] at e2
  rw [← e]
  exact congrArg (· + d) e2

end C3

end FlockC3.RegLevels
