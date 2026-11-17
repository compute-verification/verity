module

public import Proofs.Flock.C3.Compose.HoloEncoded
public import Proofs.Flock.C3.RegLevels.SrcMaps
public import Proofs.Flock.C3.RegLevels.StackedCoins
import all Init.Data.Array.Basic
import all Init.Data.Range.Basic
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
import Proofs.Flock.C3.RegLevels.TopHold

/-! R's Ligerito stage's coins, as the verifier holds R's stream. -/

@[expose] public section

namespace FlockC3.RegLevels

open FlockSoundness.Assumptions (heldCoins)
open FlockSoundness.Discharge.PlaysCoin (coinArr)

theorem commits_c3Of {ish : Flock.RecVStar.Inner} {cs : Flock.RecVStar.InnerCommits} {c3 : Flock.RecCoins.C3}
    (hc3 : Flock.Guarantees.c3Of ish (some cs) = some c3) : c3.commits = cs := by
  unfold Flock.Guarantees.c3Of at hc3
  cases hsp : ish.sparse with
  | none => rw [hsp] at hc3; cases hc3
  | some sp => rw [hsp] at hc3; cases hc3; rfl

theorem heldCoins_rep0 {ish : Flock.RecVStar.Inner} {cs : Flock.RecVStar.InnerCommits} {c3 : Flock.RecCoins.C3}
    (hc3 : Flock.Guarantees.c3Of ish (some cs) = some c3) (unit : String) (q : ℕ) :
    heldCoins (.ok cs) s!"{Flock.RecVStar.STACKED}/rep0" q = (FlockC3.Opening.rep0 c3 unit)[q]? := by
  have hc := commits_c3Of hc3
  cases hst : cs.stream? s!"{Flock.RecVStar.STACKED}/rep0" with
  | none =>
    have hr : FlockC3.Opening.rep0 c3 unit = #[] := by
      unfold FlockC3.Opening.rep0 Flock.RecCoins.C3.coins
      rw [hc, hst]
      rfl
    rw [hr]
    show (cs.stream? s!"{Flock.RecVStar.STACKED}/rep0").bind (·.coins[q]?) = _
    rw [hst]
    rfl
  | some st =>
    rw [FlockC3.Opening.rep0_eq (c := st.coins) (by unfold Flock.RecCoins.C3.coins; rw [hc, hst]; rfl)]
    show (cs.stream? s!"{Flock.RecVStar.STACKED}/rep0").bind (·.coins[q]?) = _
    rw [hst]
    rfl

theorem heldStage_of_lgCoins {ish : Flock.RecVStar.Inner} {cs : Flock.RecVStar.InnerCommits}
    {c3 : Flock.RecCoins.C3} (hc3 : Flock.Guarantees.c3Of ish (some cs) = some c3) {unit : String}
    {ts : List FlockSoundness.Game.Step} {k₂ n : ℕ}
    (hts : ts.filterMap Sum.getRight? =
      ((List.range' k₂ n).map fun i => (FlockC3.Opening.rep0 c3 unit)[i]!).map (coinArr FlockSoundness.Refine.fOf)) :
    ts.filterMap Sum.getRight? = (List.range' k₂ n).map
        (coinArr FlockSoundness.Refine.fOf ∘ fun q => (FlockC3.Opening.rep0 c3 unit)[q]!) ∧
      ∀ q c, heldCoins (.ok cs) s!"{Flock.RecVStar.STACKED}/rep0" q = some c →
        (FlockC3.Opening.rep0 c3 unit)[q]! = c := by
  refine ⟨by rw [hts, List.map_map], fun q c hq => ?_⟩
  rw [heldCoins_rep0 hc3 unit q] at hq
  rw [getElem!_def, hq]

open Flock.Guarantees (HoloStream heldBefore roundIn holoStreams)
open Flock.RecVStar (SchedRound)

theorem heldBefore_succ (s : HoloStream) (rs : Array SchedRound) {k : ℕ} (hk : k < rs.size) :
    heldBefore s rs (k + 1) = heldBefore s rs k + s.group (rs.getD k default).1 := by
  unfold heldBefore
  rw [Nat.min_eq_left (show k + 1 ≤ rs.size from hk), Nat.min_eq_left hk.le, List.range_succ, List.map_append,
    List.sum_append]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]

theorem heldBefore_stacked {rs : Array SchedRound} {k₂ : ℕ} (p : ℕ)
    (h : ∀ p' < p, 0 < (rs.getD (k₂ + p') default).1) :
    heldBefore .stacked rs (k₂ + p) = heldBefore .stacked rs k₂ + p := by
  induction p with
  | zero => rfl
  | succ p ih =>
    have hp := h p (Nat.lt_succ_self p)
    have hk : k₂ + p < rs.size := by
      by_contra hn
      unfold Array.getD at hp
      simp only [hn, ↓reduceDIte] at hp
      exact absurd hp (Nat.lt_irrefl 0)
    rw [← Nat.add_assoc, heldBefore_succ _ _ hk, ih fun p' hp' => h p' (by omega)]
    show _ + min _ 1 = _
    omega

theorem roundIn_stacked (rs : Array SchedRound) (q : ℕ) :
    ∀ L : List (HoloStream × Array SchedRound), (∀ e ∈ L, e.1.idx ≠ HoloStream.idx .stacked) →
      roundIn (L ++ [(.stacked, rs)]) .stacked q =
        (L.map fun e => heldBefore e.1 e.2 e.2.size).sum + heldBefore .stacked rs q
  | [], _ => by
    show roundIn [(.stacked, rs)] .stacked q = 0 + _
    rw [FlockC3.Compose.roundIn_cons_self, Nat.zero_add]
  | (s', rs') :: L, h => by
    simp only [List.cons_append, List.map_cons, List.sum_cons]
    rw [FlockC3.Compose.roundIn_cons_ne (s' := s') (s := .stacked) (h (s', rs') (by simp)),
      roundIn_stacked rs q L fun e he => h e (by simp [he]),
      Nat.add_assoc]

theorem holoSelOne_stacked {ish : Flock.RecVStar.Inner} {tbl s : String} {rs : Array SchedRound}
    {L : List (HoloStream × Array SchedRound)} {q : ℕ}
    (hS : holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some (L ++ [(.stacked, rs)]))
    (hL : ∀ e ∈ L, e.1.idx ≠ HoloStream.idx .stacked)
    (hf : (L ++ [(HoloStream.stacked, rs)]).find? (fun e => e.1.name tbl == s) = some (.stacked, rs))
    (hq : 0 < (rs.getD q default).1) :
    FlockC3.Compose.holoSelOne ish tbl
      (1 + (L.map fun e => heldBefore e.1 e.2 e.2.size).sum + heldBefore .stacked rs q) s q :=
  ⟨_, _, hS, hf, Nat.min_eq_right hq, by rw [Nat.add_assoc, ← roundIn_stacked rs q L hL]; rfl⟩

theorem sampled_stacked {ish : Flock.RecVStar.Inner} {tbl s : String} {rs : Array SchedRound}
    {L : List (HoloStream × Array SchedRound)} {k₂ n : ℕ}
    (hS : holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some (L ++ [(.stacked, rs)]))
    (hL : ∀ e ∈ L, e.1.idx ≠ HoloStream.idx .stacked)
    (hf : (L ++ [(HoloStream.stacked, rs)]).find? (fun e => e.1.name tbl == s) = some (.stacked, rs))
    (hpos : ∀ p < n, 0 < (rs.getD (k₂ + p) default).1) :
    ∀ p q, (List.range' k₂ n)[p]? = some q → FlockC3.Compose.holoSelOne ish tbl
      (1 + (L.map fun e => heldBefore e.1 e.2 e.2.size).sum + heldBefore .stacked rs k₂ + p) s q := by
  intro p q hp
  obtain ⟨hlt, rfl⟩ := List.getElem?_eq_some_iff.1 hp
  rw [List.length_range'] at hlt
  simp only [List.getElem_range', Nat.one_mul]
  have h := holoSelOne_stacked (q := k₂ + p) hS hL hf (hpos p hlt)
  rwa [heldBefore_stacked p fun p' hp' => hpos p' (by omega), ← Nat.add_assoc] at h

theorem stackedRounds_coins {sh : Flock.RecVStar.StackedShape} (hc : sh.claims = Flock.RecVStar.STACKED_CLAIMS)
    {rs : Array SchedRound} (h : Flock.RecVStar.stackedRounds sh = some rs) :
    ∀ q < rs.size, 0 < (rs.getD q default).1 := by
  obtain ⟨m, claims⟩ := sh
  subst hc
  by_cases hm : m < 36
  · have hall : ∀ m < 36, ∀ rs, Flock.RecVStar.stackedRounds { m, claims := Flock.RecVStar.STACKED_CLAIMS } = some rs →
        ∀ q < rs.size, 0 < (rs.getD q default).1 := by decide +kernel
    exact hall m hm rs h
  · have hf : Flock.fast100 m = none := by
      unfold Flock.fast100
      simp [show m > 35 by omega]
    simp [Flock.RecVStar.stackedRounds, hf] at h

theorem stackedCoins_of_innerCommitsAt {tr : Lean.Json} {record : String} {st : Flock.Stmt} {reg : Lean.Json}
    {cs : Flock.RecVStar.InnerCommits}
    (h : FlockSoundness.Discharge.Recursion.innerCommitsAt tr record st reg = .ok cs)
    {x : Flock.RecVStar.InnerStream} (hx : cs.stream? "stacked/rep0" = some x) :
    ∃ sh rs, Flock.RecVStar.stackedShapeOf st.c.metaJ = .ok (some sh) ∧ sh.claims = Flock.RecVStar.STACKED_CLAIMS ∧
      Flock.RecVStar.stackedRounds sh = some rs ∧ x.coins.size = rs.size := by
  unfold FlockSoundness.Discharge.Recursion.innerCommitsAt at h
  obtain ⟨shape, hsh, h⟩ := TopHold.eok_bind h
  obtain ⟨⟨sh, rfl, hc, rs, hrs, hsz⟩, -⟩ := StackedCoins.innerCommits_held h hx
  exact ⟨sh, rs, hsh, hc, hrs, hsz⟩

end FlockC3.RegLevels
