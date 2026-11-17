module

import all Init.Data.Repr
public import Proofs.Flock.Recursive.HoloMap
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
import Proofs.Flock.C3.RegLevels.StackedCoins
import Proofs.Flock.C3.RegLevels.TopHold

/-! R's rounds in the history's order are the ones META's shape gives R's held stream. -/

@[expose] public section

deriving instance ReflBEq, LawfulBEq for Flock.RecVStar.StackedShape

namespace FlockC3.RegLevels

open Flock.RecVStar
open Flock.Guarantees (HoloStream holoStreams holoRounds sparseRounds)

theorem covers_ok {i : Inner} {cs : InnerCommits} (h : (checkCovers i cs).isOk = true) :
    checkCovers i cs = .ok () := by
  revert h
  cases checkCovers i cs with
  | error e => intro h; cases h
  | ok u => intro _; rfl

theorem stacked_of_covers {i : Inner} {cs : InnerCommits} (h : checkCovers i cs = .ok ()) :
    i.stacked = cs.stacked := by
  unfold checkCovers at h
  extract_lets sparse stacked showR jp3 jp2 jp1 jp0 at h
  split at h
  · cases h
  simp (config := { zeta := false }) only [jp0] at h
  split at h
  · cases h
  simp (config := { zeta := false }) only [jp1] at h
  split at h
  · cases h
  simp (config := { zeta := false }) only [jp2] at h
  split at h
  · cases h
  rename_i hne
  simpa using hne

theorem repName_ne_stacked {tbl : String} (htbl : tbl ≠ s!"{STACKED}/") (x : Fin 2) :
    (HoloStream.rep x).name tbl ≠ HoloStream.stacked.name tbl := by
  intro h
  have h := congrArg String.toList h
  simp only [HoloStream.name, String.toList_append, toString, Nat.toList_repr, List.append_assoc] at h
  have e : STACKED.toList ++ "/rep0".toList = (s!"{STACKED}/").toList ++ ("rep".toList ++ ['0']) := by decide
  rw [e] at h
  obtain ⟨v, hv⟩ := x
  simp only at h
  rcases v with _ | _ | n
  · rw [show Nat.toDigits 10 0 = ['0'] by decide +kernel] at h
    exact htbl (String.toList_inj.1 (List.append_inj' h rfl).1)
  · rw [show Nat.toDigits 10 (0 + 1) = ['1'] by decide +kernel] at h
    exact absurd (List.append_inj' h rfl).2 (by decide)
  · omega

theorem holoStreams_stacked {ish : Inner} {rep rs : Array SchedRound} {sh : StackedShape}
    (hrep : holoRounds ish.m ish.kLog ish.claims = some rep) (hsh : ish.stacked = some sh)
    (hrs : stackedRounds sh = some rs) :
    ∃ L, holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some (L ++ [(.stacked, rs)]) ∧
      (∀ e ∈ L, e.1.idx ≠ HoloStream.idx .stacked) ∧
      ∀ tbl : String, tbl ≠ s!"{STACKED}/" →
        (L ++ [(HoloStream.stacked, rs)]).find? (fun e => e.1.name tbl == HoloStream.stacked.name tbl) =
          some (.stacked, rs) := by
  have hR : ∀ tbl : String, tbl ≠ s!"{STACKED}/" → ∀ L : List (HoloStream × Array SchedRound),
      (∀ e ∈ L, e.1.name tbl ≠ HoloStream.stacked.name tbl) →
      (L ++ [(HoloStream.stacked, rs)]).find? (fun e => e.1.name tbl == HoloStream.stacked.name tbl) =
        some (.stacked, rs) := by
    intro tbl _ L hL
    rw [List.find?_append, List.find?_eq_none.2 fun e he => by simpa using hL e he]
    simp
  cases hsp : ish.sparse with
  | none =>
    refine ⟨[(.rep 0, rep), (.rep 1, rep)], ?_, ?_, fun tbl htbl => hR tbl htbl _ ?_⟩
    · unfold holoStreams
      simp [hrep, hsh, hrs]
    · intro e he
      simp only [List.mem_cons, List.not_mem_nil, or_false] at he
      rcases he with rfl | rfl <;> simp [HoloStream.idx]
    · intro e he
      simp only [List.mem_cons, List.not_mem_nil, or_false] at he
      rcases he with rfl | rfl
      · exact repName_ne_stacked htbl 0
      · exact repName_ne_stacked htbl 1
  | some sp =>
    refine ⟨[(.rep 0, rep), (.rep 1, rep), (.sparse 0, sparseRounds sp), (.sparse 1, sparseRounds sp)], ?_, ?_,
      fun tbl htbl => hR tbl htbl _ ?_⟩
    · unfold holoStreams
      simp [hrep, hsh, hrs]
    · intro e he
      simp only [List.mem_cons, List.not_mem_nil, or_false] at he
      rcases he with rfl | rfl | rfl | rfl <;> simp [HoloStream.idx]
    · intro e he
      simp only [List.mem_cons, List.not_mem_nil, or_false] at he
      rcases he with rfl | rfl | rfl | rfl
      · exact repName_ne_stacked htbl 0
      · exact repName_ne_stacked htbl 1
      · simp only [HoloStream.name]; decide
      · simp only [HoloStream.name]; decide

theorem stackedStreams_of_covers {tr : Lean.Json} {record : String} {st : Flock.Stmt} {reg : Lean.Json}
    {ish : Inner} {cs : InnerCommits} {rep : Array SchedRound}
    (h : FlockSoundness.Discharge.Recursion.innerCommitsAt tr record st reg = .ok cs)
    (hcov : (checkCovers ish cs).isOk = true) (hrep : holoRounds ish.m ish.kLog ish.claims = some rep)
    {x : InnerStream} (hx : cs.stream? "stacked/rep0" = some x) :
    ∃ sh rs L, ish.stacked = some sh ∧ stackedShapeOf st.c.metaJ = .ok (some sh) ∧ sh.claims = STACKED_CLAIMS ∧
      stackedRounds sh = some rs ∧ x.coins.size = rs.size ∧
      holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some (L ++ [(.stacked, rs)]) ∧
      (∀ e ∈ L, e.1.idx ≠ HoloStream.idx .stacked) ∧
      ∀ tbl : String, tbl ≠ s!"{STACKED}/" →
        (L ++ [(HoloStream.stacked, rs)]).find? (fun e => e.1.name tbl == HoloStream.stacked.name tbl) =
          some (.stacked, rs) := by
  unfold FlockSoundness.Discharge.Recursion.innerCommitsAt at h
  obtain ⟨shape, hsh, h⟩ := TopHold.eok_bind h
  obtain ⟨⟨sh, rfl, hc, rs, hrs, hsz⟩, hstk⟩ := StackedCoins.innerCommits_held h hx
  have hi : ish.stacked = some sh := (stacked_of_covers (covers_ok hcov)).trans hstk
  obtain ⟨L, hS, hL, hf⟩ := holoStreams_stacked hrep hi hrs
  exact ⟨sh, rs, L, hi, hsh, hc, hrs, hsz, hS, hL, hf⟩

end FlockC3.RegLevels
