import Proofs.Flock.C3.RegLevels.AccReads

/-!
# A registered read is its reads file's entry, at an unsegmented row

The one form of these facts, which `TopLeaf`, `PartTies`, `AccOut` and `HPortTie` read (K round 10's, moved here):

* `ReadsOwnAt`, `readsOwnAt_of`: a registered read is the reads file's entry of its name at the unit's own row, from
  `readsOwn_of_hown` and the label check at reached, checked outcomes (`ReachedChecked`, named: `accRead_at_unit`'s
  premises at every session).
* `readsUnsegC3`: no read is of a segmented row (`VStmt.hscope`'s `v1Ports`, `segs_of_scopeOk`).
-/

namespace FlockC3.RegLevels

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose

/-- **A V* circuit's rows are unsegmented** (`scopeOk`'s `v1Ports`). -/
theorem segs_of_scopeOk {c : Flock.Circuit} (h : Layout.scopeOk c = true) (q : Fin c.ports.size) :
    c.ports[q.val]!.segs = none := by
  unfold Layout.scopeOk Layout.v1Ports at h
  simp only [Bool.and_eq_true, Array.all_eq_true', Bool.not_eq_true', beq_iff_eq, decide_eq_true_eq] at h
  rw [getElem!_pos c.ports q.val q.isLt]
  obtain ⟨⟨⟨-, hs⟩, -⟩, -⟩ := h.2 _ (Array.getElem_mem q.isLt)
  simpa using hs

section family

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

/-- Obligation (`accRead_at_unit`'s premises, at every session): each accepted outcome is one its session's game
reaches (`Reached`) and passes candidate 3's checks at its own transcript (`ChecksC3`, which the event's `ChecksVR`
states there). -/
def ReachedChecked (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ j : Fin (vstarCount F.ish),
    Reached (F.zs R ω t R₂ j) (o j) ∧ ∃ (ish : Flock.RecVStar.Inner) (cs : Option Flock.RecVStar.InnerCommits)
      (coins : Option Flock.RecCoins.Source) (f : VStar.Bindings) (rows' : Option (Array Flock.Registered.Port)),
      Flock.Guarantees.ChecksC3 ish cs coins j f rows' (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own (o j).2.2

/-- **A registered read's port is the reads file's entry of its name, read at the unit's own row.** -/
def ReadsOwnAt (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ j (u : Fin (F.Vs j).nv)
    (q : Fin (F.Vs j).c.ports.size), (F.zs R ω t R₂ j).rd.isRead (.inl (u, q)) →
      (F.zs R ω t R₂ j).rd.port (.inl (u, q)) ∈ (F.zs R ω t R₂ j).own ∧
      ((F.zs R ω t R₂ j).rd.port (.inl (u, q))).name = (F.Vs j).c.ports[q].name ∧
      ((F.zs R ω t R₂ j).rd.port (.inl (u, q))).reads[u.val]? =
        some (some ((F.zs R ω t R₂ j).rd.readAt (.inl (u, q))))

/-- **`ReadsOwnAt` from `ReachedChecked`**: `readsOwn_of_hown` puts the read at the unit's label, which the label check
makes the unit (`label_of_checked`), as `accRead_at_unit` does at a level's `acc`. -/
theorem readsOwnAt_of {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    (hrc : ReachedChecked F dg N k Rw) : ReadsOwnAt F dg N k Rw := by
  intro R ω t R₂ τ o h j u q hr
  obtain ⟨hre, _, _, _, _, _, hc3⟩ := hrc R ω t R₂ τ o h j
  obtain ⟨hmem, hname, d, D, dp, idx, hd, hD, hdp, hidx, hu, hpos⟩ := readsOwn_of_hown h j hre u q hr
  obtain ⟨spec, sts, hst, -, -, -, hlb⟩ := hc3
  rw [FlockC3.PubRows.label_of_checked (F.zs R ω t R₂ j) (h.1 j).1 ⟨spec, sts, hst, hlb⟩ hd hD hdp hidx hu] at hpos
  exact ⟨hmem, hname, hpos⟩

/-- `ReadsUnsegmented`'s, at candidate 3's family: no registered read is of a segmented row. -/
def ReadsUnsegC3 (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ j (u : Fin (F.Vs j).nv)
    (q : Fin (F.Vs j).c.ports.size), (F.zs R ω t R₂ j).rd.isRead (.inl (u, q)) → (F.Vs j).c.ports[q.val]!.segs = none

/-- **`ReadsUnsegC3` holds**: every V* circuit's rows are unsegmented (`VStmt.hscope`, `segs_of_scopeOk`), read or not. -/
theorem readsUnsegC3 (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) :
    ReadsUnsegC3 F dg N k Rw :=
  fun _ _ _ _ _ _ _ j _ q _ => segs_of_scopeOk (F.Vs j).hscope q

end family

end FlockC3.RegLevels
