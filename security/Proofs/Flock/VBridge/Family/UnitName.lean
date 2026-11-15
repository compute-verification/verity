module

public import Proofs.Flock.VBridge.Family.Names
public import Proofs.Flock.VBridge.Family.Structure
import all Init.Data.String.Legacy
import all Init.Data.String.Defs
import all Init.Data.Repr
import all Init.Data.Int.Repr
import Std.Data.String.ToNat
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
import Mathlib.Tactic.ReduceModChar
import Proofs.Flock.Soundness.Refine.Setup

/-!
# A statement's circuit file names its unit

`HmRow.parse` of a statement's `circuitFile` (`Family.lean`) names the circuit's unit as the statement's
(`file_unitName`), from the text facts of `Names.lean`: `HUnitName` and `HOpenName` at any history and registration.
-/

@[expose] public section

namespace FlockVBridge.Names

open Flock FlockSoundness

/-- **The parse of a statement's circuit file names its unit as the statement's unit**: `circuitFile` writes
`flock-circuit <unit>`, the one-line `META`, `CIRCUIT unit <n>` and the unit's header `flock-ir-unit/v2 <unit> …`
(`Family.lean:51`, `Compose.lean:187-244`, `Unit.lean:80-100`), and `HmRow.parse` takes the first section, `unit`, as
the circuit's unit (`parse_unit`), named by its header's second token (`parse_name`). The name is one word. -/
theorem file_unitName {s : VStar.Stmt} {b : VStar.Bindings} {file : ByteArray}
    (hf : FlockVBridge.Family.circuitFile s b = .ok file) {tags : Tags} {tables : List (String × ByteArray)}
    {c : Circuit} (hp : HmRow.parse tags file tables = .ok c)
    (hw : ∀ u, s.unit = .ok u → NoWs u.name ∧ u.name ≠ "") : ∃ u, s.unit = .ok u ∧ c.unit.name = u.name := by
  unfold FlockVBridge.Family.circuitFile at hf
  obtain ⟨u, hu, hf⟩ := except_bind_ok hf
  obtain ⟨l, -, hf⟩ := except_bind_ok hf
  obtain ⟨e, he, hf⟩ := except_bind_ok hf
  cases hf
  refine ⟨u, hu, ?_⟩
  obtain ⟨hws, hne⟩ := hw u hu
  obtain ⟨sha, head, hh, he'⟩ := emitCircuit_ok he
  obtain ⟨j, d, hd, rfl⟩ := metaHead_ok hh
  have hpre : Pre ((s!"{VStar.FORMAT_CIRCUIT} {u.name}\nMETA " ++ d ++ "\n") ++
      (("CIRCUIT unit " ++ toString (VStar.netLines l) ++ "\n") ++ (VStar.header u.name l ++ "\n"))) e := by
    rw [he']
    exact pre_put _ _ (fun _ => sections_text u l)
  obtain ⟨tl, hX⟩ := hpre ""
  have hp' : HmRow.parse tags (StateT.run e "").2.toUTF8 tables = .ok c := hp
  rw [hX] at hp'
  have key : ∀ text : String,
      text = "" ++ ((s!"{VStar.FORMAT_CIRCUIT} {u.name}\nMETA " ++ d ++ "\n") ++
        (("CIRCUIT unit " ++ toString (VStar.netLines l) ++ "\n") ++ (VStar.header u.name l ++ "\n"))) ++ tl →
      HmRow.parse tags text.toUTF8 tables = .ok c → c.unit.name = u.name := by
    intro text ht hp
    have e1 : text = ("flock-circuit " ++ u.name) ++ "\n" ++ (("META " ++ d) ++ "\n" ++
        (("CIRCUIT unit " ++ toString (VStar.netLines l)) ++ "\n" ++ (VStar.header u.name l ++ "\n" ++ tl))) := by
      rw [ht, show s!"{VStar.FORMAT_CIRCUIT} {u.name}\nMETA " = "flock-circuit " ++ u.name ++ "\n" ++ "META " from by
        show "flock-circuit" ++ " " ++ u.name ++ "\nMETA " = _
        rw [show ("flock-circuit" : String) ++ " " = "flock-circuit " by decide,
          show ("\nMETA " : String) = "\n" ++ "META " by decide]
        simp only [String.append_assoc]]
      simp only [String.empty_append, String.append_assoc]
    have hsplit : text.splitOn "\n" = ("flock-circuit " ++ u.name) :: ("META " ++ d) ::
        ("CIRCUIT unit " ++ toString (VStar.netLines l)) :: VStar.header u.name l :: tl.splitOn "\n" := by
      rw [e1, splitOn_line _ _ (noNl_append.2 ⟨by simp [NoNl], noNl_of_noWs hws⟩),
        splitOn_line _ _ (noNl_append.2 ⟨by simp [NoNl], pyDumps_noNl hd⟩),
        splitOn_line _ _ (noNl_append.2 ⟨by simp [NoNl], nat_noNl _⟩),
        splitOn_line _ _ (header_noNl _ _ (noNl_of_noWs hws))]
    obtain ⟨h2, h3, -⟩ := linesOf_four hsplit (splitOn_ne_nil tl)
    have hu2 : (tokens ((linesOf text).getD 2 "")).getD 1 "" = "unit" := by
      rw [h2, show ("CIRCUIT unit " : String) = "CIRCUIT" ++ " " ++ "unit" ++ " " by decide]
      exact tokens_second _ _ _ (by decide) (by decide) (by simp [NoWs]) (by simp [NoWs])
    obtain ⟨n, hn, hpn⟩ := parse_unit hp (fromUTF8?_toUTF8 text) hu2
    obtain ⟨hfmt, hname⟩ := parse_name hpn
    rw [hname]
    cases n with
    | zero =>
      have : ((linesOf text).extract 3 (3 + 0)).getD 0 "" = "" := by
        rw [Array.getD_eq_getD_getElem?, Array.getElem?_extract, ite_eq_right (by omega)]; rfl
      rw [this, tokens_eq] at hfmt
      simp at hfmt
    | succ n =>
      have : ((linesOf text).extract 3 (3 + (n + 1))).getD 0 "" = (linesOf text).getD 3 "" := by
        rw [Array.getD_eq_getD_getElem?, Array.getElem?_extract, ite_eq_left (by omega), ← Array.getD_eq_getD_getElem?]
      rw [this, h3]
      obtain ⟨rest, hr⟩ := header_eq u.name l
      rw [hr]
      exact tokens_second _ _ _ (by decide) hne (by simp [NoWs, VStar.FORMAT]) hws
  exact key _ rfl hp'

end FlockVBridge.Names

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open FlockVBridge.Names

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂)

/-- **`HUnitName` from the circuit file's round trip** (`file_unitName`): a part statement `.inner p nm pt` has unit
`innerUnit pt.S nm` (`Stage.lean:44-47`), named `nm` (`InnerRepCheck.lean:38-52`), which is
`RecResiduals.unitName …`, one word (`statements_name`). At any history and registration the session parses its circuit
from that file (`parse_circuitFile`). -/
theorem hUnitName_of (hne : Nonempty (Reg × L.Ω × Reg₂)) : F.HUnitName := by
  obtain ⟨⟨R, ω, R₂⟩⟩ := hne
  intro j p nm pt hj
  have hp := F.parse_circuitFile R ω [] R₂ j (F.hfile R ω [] R₂ j)
  have hnm := statements_name F.hstmts j hj
  obtain ⟨u, hu, hn⟩ := file_unitName (F.hfile R ω [] R₂ j) hp fun u hu => by
    rw [hj] at hu
    rw [innerUnit_name hu, hnm]
    exact residuals_name _ _ _ _
  rw [hj] at hu
  rw [hn, innerUnit_name hu]

/-- **`HOpenName` from the circuit file's round trip** (`file_unitName`): an opening statement `.recOpen ℓ LANES H m`
has unit `recOpenUnit LANES H` (`Stage.lean:44-46`), named `RecOpen.unitName LANES H` (`Unit.lean:36-46`), one word. -/
theorem hOpenName_of (hne : Nonempty (Reg × L.Ω × Reg₂)) : F.HOpenName := by
  obtain ⟨⟨R, ω, R₂⟩⟩ := hne
  intro j ℓ LANES H m hj
  have hp := F.parse_circuitFile R ω [] R₂ j (F.hfile R ω [] R₂ j)
  obtain ⟨u, hu, hn⟩ := file_unitName (F.hfile R ω [] R₂ j) hp fun u hu => by
    rw [hj] at hu
    rw [recOpenUnit_name hu]
    exact recOpen_name _ _
  rw [hj] at hu
  rw [hn, recOpenUnit_name hu]

end VStarFamily

end family

end FlockVBridge.Family
