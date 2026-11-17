module

public import Proofs.Flock.C3.Compose.Decoder
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.C3.Opening.SrcOpening

/-! `RegTape` at the source, from R's Ligerito tied to an accepting run. -/

@[expose] public section

namespace FlockC3.RegLevels

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose
open FlockSoundness.Discharge.PlaysCoin (coinArr)

theorem tapeR_runSteps {e : ExecHolo} (u : C3Run e) :
    tapeR (runSteps u) u = u.lg ++ [.inl ⟨Opens GF128 Dig × Salts (List UInt8), u.os⟩] := by
  change (FlockC3.Opening.afterSparse u (runSteps u)).drop u.op.length = _
  rw [FlockC3.Opening.afterSparse_runSteps, List.drop_left]

def LevelTie (cap₀ : Cap Dig) (e : ExecHolo) (u : C3Run e) : Prop :=
  ∃ (msId : String) (m : ℕ) (sch : Flock.Schedule) (Tw : Flock.F128) (sws : Array Flock.Switched) (nc : ℕ)
    (cap : Array ByteArray) (p : Flock.LigeritoProof) (rt rt' : Flock.Transcript),
    Flock.fast100 m = some sch ∧ Refine.schedOf sch = e.rsch.sch ∧ capDig cap = cap₀ ∧
      OpenRel sch.m nc Tw sws u.opened ∧ RoundsWF rt.rounds ∧
      Flock.ligerito (Flock.MerkleScheme.sha512 msId) sch Tw cap p (Flock.basis sws) rt = .ok ((), rt') ∧
      u.lg.map kindOf <+: ligShape GF128 GF256 Dig e.rsch.sch ∧
      u.lg.filterMap Sum.getLeft? = ligMsgs p capDig sch ∧
      u.lg.filterMap Sum.getRight? = (consumed rt.rounds rt.k rt'.k).map (coinArr fOf) ∧
      ∀ ℓ j, u.os.1 2 ℓ j = openRow p sch ℓ j

theorem regTape_of_levelTie {cap₀ : Cap Dig} {e : ExecHolo} {u : C3Run e} (h : LevelTie cap₀ e u) :
    RegTape cap₀ e.rsch.sch u.opened (tapeR (runSteps u) u) := by
  obtain ⟨msId, m, sch, Tw, sws, nc, cap, p, rt, rt', hf, hsch, hcap, hop, hwf, hlig, hk, hl, hr, ho⟩ := h
  rw [tapeR_runSteps]
  exact ⟨msId, m, sch, Tw, sws, nc, cap, p, rt, rt', u.lg, u.os, hf, hsch, hcap, hop, hwf, hlig,
    List.prefix_append _ _, hk, hl, hr, by rw [List.drop_left]; rfl, ho⟩

noncomputable section

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

def SrcLevelTie {S : ℕ} {Vs : Fin S → VStmt}
    (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (cap₀ : Cap Dig) : Prop :=
  AtSrc zs dg N k Rw src fun _ u => LevelTie cap₀ (holoAt st hp) u

theorem regTapeAt_src {S : ℕ} {Vs : Fin S → VStmt}
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (h : SrcLevelTie zs dg N k Rw src xR.cap) :
    AtSrc zs dg N k Rw src fun _ u => RegTape xR.cap (holoAt st hp).rsch.sch u.opened (tapeR (runSteps u) u) :=
  fun R ω t R₂ τ o hacc => regTape_of_levelTie (h R ω t R₂ τ o hacc)

variable {ish : RecVStar.Inner} {ss : Array VStar.Stmt} {hss : RecVStar.statements ish = .ok ss}
  {bind : Fin (vstarCount ish) → VStar.Bindings} {Vs : Fin (vstarCount ish) → VStmt}
  {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
    (Vs j).Outer Reg'}
  {own : Fin (vstarCount ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
  {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String} {reg : Lean.Json}
  {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {src : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) →
    C3Run (holoAt st hp)}

theorem regLevelsRead_of_tie (hs : SrcOK zs dg N k Rw src own sel) (h : SrcLevelTie zs dg N k Rw src xR.cap) :
    RegLevelsRead (C3Family.honest (xR := xR) ish hss bind hob zs own sel bc (c3DecOf zs dg N k Rw src own sel) wr
      reg) dg N k Rw :=
  regLevelsRead_c3DecOf hs (regTapeAt_src h)

end

end FlockC3.RegLevels
