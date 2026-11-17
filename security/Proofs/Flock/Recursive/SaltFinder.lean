module

public import Flock.RecCoins
public import Proofs.Flock.Recursive.Flock
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
import Proofs.Flock.Soundness.Game

/-! The held salts' finder (`saltFinder`): a registrant's read of R's level-0 top against the salts its nodes were committed under, which the prover's workers hold. -/

@[expose] public section

namespace FlockC3.RegLevels

open Flock

def wordBytes (ws : List ℕ) : List UInt8 := ws.flatMap fun w => [(w % 256).toUInt8, (w / 256 % 256).toUInt8]

def rowInV (row : Array ℕ) : List UInt8 :=
  let ws := RecCoins.padTo row (Registered.rowWords row.size 16)
  (HmRow.rowPrefix ws.size).data.toList ++ wordBytes ws.toList

def capRow (top : ByteArray) : Array ℕ := (ZkServer.topRows top).headD #[]

def capRows (lv : FlockSoundness.Accounting.Level) (cap : FlockSoundness.Model.Cap FlockSoundness.Refine.Dig) :
    Array (Array ℕ) :=
  (Array.range (2 ^ FlockSoundness.Model.capDepth lv)).map fun i => capRow (cap i).1

end FlockC3.RegLevels

namespace FlockSoundness.Registered

open Game Refine Binding Audit

variable {C : Verity.Boolean.Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}
  {lay : Binding.Layout P plan tab decode} {rs : HmRowsX lay}

def ReadsX.SaltClash (rd : ReadsX lay rs) (rows : Array (Array ℕ)) (salt : ℕ → ByteArray) : Prop :=
  ∃ p, rd.isRead p ∧ Collides (hm96Pair (rs.enc p (rd.wv p)) (rd.ws p)
    (FlockC3.RegLevels.rowInV rows[rd.readAt p]!) (salt (rd.readAt p)).data.toList)

noncomputable def ReadsX.saltCands (rd : ReadsX lay rs) (rows : Array (Array ℕ)) (salt : ℕ → ByteArray)
    (ps : List lay.Pos) : List (Prop × (List UInt8 × List UInt8)) :=
  ps.map fun p =>
    (rd.isRead p ∧ Collides (hm96Pair (rs.enc p (rd.wv p)) (rd.ws p)
        (FlockC3.RegLevels.rowInV rows[rd.readAt p]!) (salt (rd.readAt p)).data.toList),
      hm96Pair (rs.enc p (rd.wv p)) (rd.ws p) (FlockC3.RegLevels.rowInV rows[rd.readAt p]!)
        (salt (rd.readAt p)).data.toList)

theorem ReadsX.saltCands_spec {rd : ReadsX lay rs} {rows : Array (Array ℕ)} {salt : ℕ → ByteArray}
    {ps : List lay.Pos} {c : Prop × (List UInt8 × List UInt8)} (hc : c ∈ rd.saltCands rows salt ps) (h : c.1) :
    Hash.Collision H512 c.2.1 c.2.2 := by
  obtain ⟨p, -, rfl⟩ := List.mem_map.1 hc
  have hw : rd.isRead p ∧ Collides (hm96Pair (rs.enc p (rd.wv p)) (rd.ws p)
      (FlockC3.RegLevels.rowInV rows[rd.readAt p]!) (salt (rd.readAt p)).data.toList) := h
  exact hw.2

theorem ReadsX.SaltClash.cand {rd : ReadsX lay rs} {rows : Array (Array ℕ)} {salt : ℕ → ByteArray}
    {ps : List lay.Pos} (h : rd.SaltClash rows salt) (hps : ∀ p, p ∈ ps) :
    ∃ c ∈ rd.saltCands rows salt ps, c.1 := by
  obtain ⟨p, hp, h⟩ := h
  exact ⟨_, List.mem_map.2 ⟨p, hps p, rfl⟩, hp, h⟩

end FlockSoundness.Registered

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting StrictCR
open scoped ENNReal

variable {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg₂ : Type} {I : Inner}
variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type} {dg : ByteArray → ByteArray} {N : ℕ}

def SaltColl (lv : FlockSoundness.Accounting.Level) (cap : Cap Refine.Dig) (salt : ℕ → ByteArray)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (R : Reg) (ω : L.Ω)
    (t : List (Cm × I.Coin)) (R₂ : Reg₂) : Prop :=
  ∃ j, (zs R ω t R₂ j).rd.SaltClash (FlockC3.RegLevels.capRows lv cap) salt

noncomputable def saltCands (lv : FlockSoundness.Accounting.Level) (cap : Cap Refine.Dig) (salt : ℕ → ByteArray)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (R : Reg) (ω : L.Ω)
    (t : List (Cm × I.Coin)) (R₂ : Reg₂) : List (Prop × (List UInt8 × List UInt8)) :=
  (List.finRange S).flatMap fun j =>
    (zs R ω t R₂ j).rd.saltCands (FlockC3.RegLevels.capRows lv cap) salt (zs R ω t R₂ j).posList

theorem saltCands_spec {lv : FlockSoundness.Accounting.Level} {cap : Cap Refine.Dig} {salt : ℕ → ByteArray}
    {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {R : Reg} {ω : L.Ω}
    {t : List (Cm × I.Coin)} {R₂ : Reg₂} {c : Prop × (List UInt8 × List UInt8)}
    (hc : c ∈ saltCands lv cap salt zs R ω t R₂) (h : c.1) : Hash.Collision H512 c.2.1 c.2.2 := by
  obtain ⟨j, -, hc⟩ := List.mem_flatMap.1 hc
  exact Registered.ReadsX.saltCands_spec hc h

theorem SaltColl.cand {lv : FlockSoundness.Accounting.Level} {cap : Cap Refine.Dig} {salt : ℕ → ByteArray}
    {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {R : Reg} {ω : L.Ω}
    {t : List (Cm × I.Coin)} {R₂ : Reg₂} (h : SaltColl lv cap salt zs R ω t R₂) :
    ∃ c ∈ saltCands lv cap salt zs R ω t R₂, c.1 := by
  obtain ⟨j, h⟩ := h
  obtain ⟨c, hc, hp⟩ := h.cand (zs R ω t R₂ j).mem_posList
  exact ⟨c, List.mem_flatMap.2 ⟨j, List.mem_finRange j, hc⟩, hp⟩

/-- An honest `Finder.CR` budget counts the audit run and, at each registered read of each session, `hm96Pair`'s four
SHA-512 evaluations over both salts. -/
noncomputable def saltFinder (lv : FlockSoundness.Accounting.Level) (cap : Cap Refine.Dig) (salt : ℕ → ByteArray)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N)) : Finder Pair :=
  oneRun _ σ (onePick fun o : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) =>
    saltCands lv cap salt zs o.1 o.2.1 o.2.2.1 o.2.2.2.1)

theorem prob_saltColl_le (lv : FlockSoundness.Accounting.Level) (cap : Cap Refine.Dig) (salt : ℕ → ByteArray)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    {qS : ℝ} (hS : (saltFinder lv cap salt zs σ).CR H512 qS) :
    prob (fun o => SaltColl lv cap salt zs o.1 o.2.1 o.2.2.1 o.2.2.2.1)
      (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤ ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) := by
  have hev : (prob (fun o => SaltColl lv cap salt zs o.1 o.2.1 o.2.2.1 o.2.2.2.1)
      (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ).toReal ≤ qS ^ 2 / 2 ^ (513 : ℝ) := by
    rw [prob_toReal]
    exact (expect_one_le H512 _ σ (fun o : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) =>
      SaltColl lv cap salt zs o.1 o.2.1 o.2.2.1 o.2.2.2.1)
      (fun o => saltCands lv cap salt zs o.1 o.2.1 o.2.2.1 o.2.2.2.1)
      (fun _ h => h.cand) fun _ _ hc hp => saltCands_spec hc hp).trans hS.le
  exact (ENNReal.le_ofReal_iff_toReal_le (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))
    (by positivity)).2 hev

end FlockSoundness.Discharge.Recursion
