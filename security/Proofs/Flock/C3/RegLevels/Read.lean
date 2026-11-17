module

public import Proofs.Flock.C3.RegLevels.Chain
public import Proofs.Flock.C3.Compose.Honest
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
import Proofs.Flock.C3.Compose.Plays
import Proofs.Flock.Soundness.Discharge.PlaysCoin.Coins
import Proofs.Flock.Soundness.Refine.Basis
import Proofs.Flock.Soundness.Refine.Seam

/-! R's Ligerito and the openings from the tape (`RegLevelsRead`), and `HRegLevels` from it. -/

@[expose] public section

namespace FlockC3.RegLevels

noncomputable section

namespace Read

open FlockSoundness FlockSoundness.Game FlockSoundness.Model FlockSoundness.Refine FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.PlaysCoin (coinArr toT_exact)
open FlockC3.Compose (stageOf proverAt stageOf_of_playsT)

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

theorem proverAt_of_eq {β : Type} {s : Option Step} {d b : β} (hs : s = some (.inl ⟨β, b⟩)) :
    proverAt β s d = b := by
  have hex : ∃ a : β, s = some (.inl ⟨β, a⟩) := ⟨b, hs⟩
  unfold proverAt
  rw [dite_eq_left hex]
  exact eq_of_heq (Sigma.mk.inj (Sum.inl.inj (Option.some.inj (hex.choose_spec.symm.trans hs)))).2

theorem playsT_lig_of_tape {sch : FlockSoundness.Accounting.Schedule} {T₀ : 𝔽} {b : ℕ → 𝔽} {ms : List Game.Msg}
    {cs : List (Array Flock.F128)} {pend : PendL 𝔽 Dig} {ts : List Step}
    (h : Plays fOf (ligeritoC (D := Dig) execArith 2 sch T₀ b) ms cs pend)
    (hk : ts.map kindOf <+: ligShape 𝔽 𝕂 Dig sch) (hl : ts.filterMap Sum.getLeft? = ms)
    (hr : ts.filterMap Sum.getRight? = cs.map (coinArr fOf)) :
    PlaysT (ligeritoC (D := Dig) execArith 2 sch T₀ b) ts pend := by
  obtain ⟨ts', hp, h1, h2⟩ := toT_exact fOf h
  have hlen : ts.length = ts'.length := by
    rw [FlockVBridge.InnerAccepts.length_eq_lefts_rights ts, FlockVBridge.InnerAccepts.length_eq_lefts_rights ts',
      hl, hr, h1, h2]
  have hw : ts = weave ((ligShape 𝔽 𝕂 Dig sch).take ts.length) (ts.filterMap Sum.getLeft?)
      (ts.filterMap Sum.getRight?) := by
    have k := List.prefix_iff_eq_take.mp hk
    rw [List.length_map] at k
    calc ts = weave (ts.map kindOf) (ts.filterMap Sum.getLeft?) (ts.filterMap Sum.getRight?) := (weave_self ts).symm
      _ = _ := by rw [k]
  have e : ts = ts' := by
    rw [hw, Spine.tape hp (ligeritoC_spine (D := Dig) execArith 2 sch T₀ b), hlen, hl, hr, h1, h2]
  rw [e]
  exact hp

end Read

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose
open FlockSoundness.Discharge.PlaysCoin (coinArr)

abbrev ligAt (e : ExecHolo) (u : C3Run e) : Game (PendL GF128 Dig) :=
  ligeritoC (D := Dig) execArith 2 e.rsch.sch u.opened.T u.opened.b

def tapeR {e : ExecHolo} (l : List Step) (u : C3Run e) : List Step :=
  ((((l.drop 2).drop (2 * (u.ts 0).length + (u.ts 1).length)).drop (u.sp 0).length).drop (u.sp 1).length).drop
    u.op.length

def RegTape (cap₀ : Cap Dig) (rsch : FlockSoundness.Accounting.Schedule) (op : OpenOut GF128) (l : List Step) :
    Prop :=
  ∃ (msId : String) (m : ℕ) (sch : Flock.Schedule) (Tw : Flock.F128) (sws : Array Flock.Switched) (nc : ℕ)
    (cap : Array ByteArray) (p : Flock.LigeritoProof) (rt rt' : Flock.Transcript) (ts : List Step)
    (os : Opens GF128 Dig × Salts (List UInt8)),
    Flock.fast100 m = some sch ∧ Refine.schedOf sch = rsch ∧ capDig cap = cap₀ ∧ OpenRel sch.m nc Tw sws op ∧
      RoundsWF rt.rounds ∧
      Flock.ligerito (Flock.MerkleScheme.sha512 msId) sch Tw cap p (Flock.basis sws) rt = .ok ((), rt') ∧
      ts <+: l ∧ ts.map kindOf <+: ligShape GF128 GF256 Dig rsch ∧
      ts.filterMap Sum.getLeft? = ligMsgs p capDig sch ∧
      ts.filterMap Sum.getRight? = (consumed rt.rounds rt.k rt'.k).map (coinArr fOf) ∧
      (l.drop ts.length)[0]? = some (.inl ⟨Opens GF128 Dig × Salts (List UInt8), os⟩) ∧
      ∀ ℓ j, os.1 2 ℓ j = openRow p sch ℓ j

def LgStages (e : ExecHolo) (l : List Step) (u : C3Run e) : Prop :=
  u.lg = (stageOf (ligAt e u) (tapeR l u) PendL.reject).1 ∧
    u.pendR = (stageOf (ligAt e u) (tapeR l u) PendL.reject).2 ∧
    ∀ os, ((tapeR l u).drop u.lg.length)[0]? = some (.inl ⟨Opens GF128 Dig × Salts (List UInt8), os⟩) → u.os = os

theorem lgStages_decodeRun (e : ExecHolo) (l : List Step) : LgStages e l (decodeRun e l) :=
  ⟨rfl, rfl, fun _ hos => Read.proverAt_of_eq hos⟩

theorem regLevels_of_readAt {e : ExecHolo} {l : List Step} {u : C3Run e} {cap₀ : Cap Dig} (hs : LgStages e l u)
    (hR : RegTape cap₀ e.rsch.sch u.opened (tapeR l u)) :
    PlaysT (ligAt e u) u.lg u.pendR ∧ u.pendR.verdict u.os.1 = true ∧
      OpensOKL H512 enc512 leafPlain cap₀ 2 u.pendR u.os.1 u.os.2 := by
  obtain ⟨hlg, hpR, hos⟩ := hs
  obtain ⟨msId, m, sch, Tw, sws, nc, cap, p, rt, rt', ts, os, hf, hsch, hcap, hop, hwf, hlig, hpre, hk, hl, hr, hnext,
    ho⟩ := hR
  subst hcap
  have hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default) := fun i hi =>
    fast100_level m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  obtain ⟨pend, -, -, hpl, hcaps, hreqs, hver⟩ := ligerito_accepts_opening (Flock.MerkleScheme.sha512 msId) sch Tw sws
    nc u.opened hop cap p 2 capDig os.1 (fast100_dims m sch hf).1 hlv
    (fun i hi => fast100_alpha m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi))
    (fun ℓ _ j _ => show (os.1 2 ℓ j).1 = (openRow p sch ℓ j).1 from congrArg Prod.fst (ho ℓ j)) hwf hlig
  rw [hsch] at hpl
  have hT : PlaysT (ligAt e u) ts pend := Read.playsT_lig_of_tape hpl hk hl hr
  rw [stageOf_of_playsT PendL.reject hT hpre] at hlg hpR
  dsimp only at hlg hpR
  have hos' : u.os = os := hos os (by rw [hlg]; exact hnext)
  rw [hlg, hpR, hos']
  exact ⟨hT, hver, opensOK_of msId sch hlv cap p 2 pend os.1 hcaps hreqs ho⟩

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type}

/-- It reads the tape, not the decoder of V*'s messages. -/
def RegLevelsRead (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) :
    Prop :=
  F.AtAccepted dg N k Rw fun R ω t Lb =>
    let u := F.run R ω t Lb
    RegTape xR.cap (holoAt st hp).rsch.sch u.opened (tapeR (schedSteps T CT 0 ((F.wmsg Lb).zip (t.map Prod.snd))) u)

def HLgStages (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  F.AtAccepted dg N k Rw fun R ω t Lb =>
    LgStages (holoAt st hp) (schedSteps T CT 0 ((F.wmsg Lb).zip (t.map Prod.snd))) (F.run R ω t Lb)

theorem hRegLevels_of_read {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    (hst : HLgStages F dg N k Rw) (hread : RegLevelsRead F dg N k Rw) : HRegLevels F dg N k Rw := by
  intro R ω t R₂ τ o hacc
  exact regLevels_of_readAt (hst R ω t R₂ τ o hacc) (hread R ω t R₂ τ o hacc)

variable {ish : RecVStar.Inner} {ss : Array VStar.Stmt} {hss : RecVStar.statements ish = .ok ss}
  {bind : Fin (vstarCount ish) → VStar.Bindings} {Vs : Fin (vstarCount ish) → VStmt}
  {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
    (Vs j).Outer Reg'}
  {own : Fin (vstarCount ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
  {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg}
  {wr : List (Cm × (holoInnerAt T CT st hp).Coin) → Lean.Json × String} {reg : Lean.Json}

local notation "𝓗" => C3Family.honest (xR := xR) ish hss bind hob zs own sel bc dec wr reg

theorem hLgStages_honest {dg : ByteArray → ByteArray} {N k Rw : ℕ} : HLgStages 𝓗 dg N k Rw := by
  intro _ _ _ _ _ _ _
  exact lgStages_decodeRun (holoAt st hp) _

theorem hRegLevels_honest_of_read {dg : ByteArray → ByteArray} {N k Rw : ℕ} (hread : RegLevelsRead 𝓗 dg N k Rw) :
    HRegLevels 𝓗 dg N k Rw :=
  hRegLevels_of_read hLgStages_honest hread

end

end FlockC3.RegLevels
