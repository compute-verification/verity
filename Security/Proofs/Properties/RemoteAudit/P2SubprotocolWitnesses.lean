import Proofs.Accounting.RemoteHeld.P2Subprotocol
import Proofs.Properties.RemoteAudit.ComponentWitnesses

/-!
# PoUB-P2's line of record at the subprotocol fails at a verifier that accepts everything

The guarantee is `LineHeldAtSubprotocol` at the deployed verifier. Its honest witness is `AcceptIsVerified`'s
(`acceptIsVerified_honest`: the auditor's session's request reads as its parameters on the honest store, and the check
leaves the session undecided until its windows verify); the premises it takes are the memory model's named
assumptions. Its nontrivial witness: at a verifier that accepts everything and records one summary for every record,
the bound fails on a node whose two credited passes read its one unit once (`lineHeldAtSubprotocol_nontrivial`).

The toy instance (`Toy`) is a node whose session has no window (`Audited.windows` 0), so no draw, on a store of one
unit, whose recorded summary has two windows of one pass each, each pass reading the one unit and drawing it. The
credit's pooled bound then places no wrong receipt (`fill_nonpos`), so it credits two units, `2·UNIT_BYTES` bytes
(`toy_qBytes`), where the trace (v0's toy trace, `Properties.RemoteAudit.Component.Toy.trace`: the unit in node memory
until tick 10 only, its first pass over `[0, 10)` and the rest over no tick) reads the unit once.
-/

namespace Properties.RemoteAudit.P2Subprotocol

open Finset Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Draw
  Accounting.Remote.Draw.Assumptions Properties.RemoteAudit.P2Line

namespace Toy

/-- A store of one unit, four blocks. -/
def reg : RemoteAudit.P2.Registration where
  node := .empty
  blocksPerSegment := 4
  segments := 1
  commitment := .empty
  salt := .empty
  root := .empty
  rows := 0

/-- A session with no window on the one-unit store. -/
def node : Audited where
  params := { deadlineNs := 1, windows := 0, samples := 1, minPasses := 1, windowNs := 1, lifetime := 1 }
  held := ⟨reg, RemoteAudit.Plan.whole 1⟩

def rec : Replay := ⟨0, .null, #[]⟩

/-- A window of one pass reading one unit, drawing it, its root arriving at tick 10. -/
def win : RemoteAudit.Session.WindowSummary where
  lengths := #[1]
  steps := #[]
  post := #[]
  heads := #[]
  stale := #[]
  arrivedNs := #[10]
  releasedNs := 0
  deadlineNs := 0
  openedNs := 0
  drawn := 1

def sm : RemoteAudit.Session.Summary where
  windows := #[win, win]
  blocks := 1
  appended := 0
  released := 0
  lifetime := 1
  pace := none
  appends := none

abbrev sum : Audited → Replay → Option RemoteAudit.Session.Summary := fun _ _ => some sm

theorem draws_zero : node.draws = 0 := rfl

instance instIsEmptyFinDrawsNode : IsEmpty (Fin node.draws) := by
  rw [draws_zero]
  infer_instance

def receipts : Receipts node.draws 0 Unit :=
  ⟨fun _ _ => 0, fun t => isEmptyElim t, fun t => isEmptyElim t, fun t => isEmptyElim t, fun t => isEmptyElim t⟩

end Toy

/-- Where every window draws all its receipts, the pooled bound's greedy places no wrong receipt: each window's share
is at most `1 − s/U = 0`. -/
theorem fill_nonpos (ws : Array (ℕ × ℕ)) (h : ∀ i : ℕ, ws[i]!.2 = ws[i]!.1) :
    ∀ (l : List ℕ) (b : ℚ), RemoteAudit.Pooled.fill ws l b ≤ 0
  | [], _ => le_refl _
  | i :: l, b => by
    have hs : RemoteAudit.Pooled.share ws[i]! b ≤ 0 := by
      unfold RemoteAudit.Pooled.share
      rw [h i]
      rcases Nat.eq_zero_or_pos ws[i]!.1 with h0 | h0
      · rw [h0]
        simp
      · refine (min_le_left _ _).trans ?_
        rw [div_self (by exact_mod_cast h0.ne')]
        norm_num
    show RemoteAudit.Pooled.share ws[i]! b * (ws[i]!.1 : ℚ) +
      RemoteAudit.Pooled.fill ws l (max 0 (b - RemoteAudit.Pooled.share ws[i]! b * (ws[i]!.2 : ℚ))) ≤ 0
    exact add_nonpos (mul_nonpos_of_nonpos_of_nonneg hs (Nat.cast_nonneg _)) (fill_nonpos ws h l _)

/-- The toy summary's windows, as the pooled bound reads them: two of one receipt, each drawn. -/
theorem toy_pw : Toy.sm.windows.map (fun w : RemoteAudit.Session.WindowSummary => (w.lengths.toList.sum, w.drawn)) =
    #[(1, 1), (1, 1)] := by
  simp [Toy.sm, Toy.win]

theorem toy_ws : ∀ i : ℕ, (#[(1, 1), (1, 1)] : Array (ℕ × ℕ))[i]!.2 = (#[(1, 1), (1, 1)] : Array (ℕ × ℕ))[i]!.1
  | 0 => rfl
  | 1 => rfl
  | _ + 2 => rfl

theorem toy_pool : ∃ b, RemoteAudit.Pooled.pooled (pooledWindows Toy.sm) RemoteAudit.P2Session.DELTA = .ok b ∧
    b.wrong ≤ 0 := by
  rw [pooledWindows, toy_pw]
  have h2 : ¬ (RemoteAudit.P2Session.DELTA ≤ 0 ∨ 1 ≤ RemoteAudit.P2Session.DELTA) := by
    unfold RemoteAudit.P2Session.DELTA
    norm_num
  cases hp : RemoteAudit.Pooled.pooled #[(1, 1), (1, 1)] RemoteAudit.P2Session.DELTA with
  | error why =>
    unfold RemoteAudit.Pooled.pooled at hp
    rw [ite_eq_right (by decide), ite_eq_right h2] at hp
    cases hp
  | ok b =>
    refine ⟨b, rfl, ?_⟩
    rw [(Properties.RemoteAudit.Greedy.ok_facts hp).2.2.2.2]
    exact fill_nonpos _ toy_ws _ _

/-- The toy summary's credit: its two units, no wrong receipt. -/
theorem toy_qBytes : (RemoteAudit.P2Session.credit Toy.sm).qBytes = 2 * RemoteAudit.P2.UNIT_BYTES := by
  obtain ⟨b, hp, hw⟩ := toy_pool
  rw [pooledWindows, toy_pw] at hp
  have hn : b.wrong.num.toNat = 0 := Int.toNat_eq_zero.2 (Rat.num_nonpos.2 hw)
  have hd : (b.wrong.num.toNat + b.wrong.den - 1) / b.wrong.den = 0 := by
    rw [hn, zero_add]
    exact Nat.div_eq_of_lt (Nat.sub_lt b.wrong.den_pos one_pos)
  unfold RemoteAudit.P2Session.credit
  dsimp only
  rw [toy_pw, hp]
  dsimp only
  rw [hd]
  simp

theorem toy_passes : P2Line.passes Toy.sm = 2 := rfl

/-- Where the passes read the one unit at most once, the toy summary's credit of two units fails. -/
theorem toy_not_read {Ω : Type} (T : Trace Ω) (ω : Ω) (hr : T.reads ω 2 ≤ 1) :
    ¬ ReadBy Toy.sum T ω Toy.node Toy.rec := fun h => by
  have h1 := h Toy.sm rfl
  rw [toy_qBytes, toy_passes] at h1
  have h2 : (T.reads ω 2 : ℝ) ≤ 1 := by exact_mod_cast hr
  have hu : (0 : ℝ) < RemoteAudit.P2.UNIT_BYTES := by exact_mod_cast (by decide : 0 < RemoteAudit.P2.UNIT_BYTES)
  have h3 := mul_le_mul_of_nonneg_left h2 (by positivity : (0 : ℝ) ≤ 8 * RemoteAudit.P2.UNIT_BYTES)
  push_cast at h1
  linarith

/-- The toy trace is over the toy node's one unit, so at any verdict its trace of record is it with that verdict. -/
theorem traceBy_toy {Ω : Type} (ver : Audited → Replay → Prop) :
    traceBy (Component.Toy.trace (Ω := Ω)) Toy.node ver (fun _ => Toy.rec) =
      { Component.Toy.trace (Ω := Ω) with accepts := fun _ => ver Toy.node Toy.rec } := by
  show { Line.overBlocks (Component.Toy.trace (Ω := Ω)) (Component.Toy.trace (Ω := Ω)).blocks with
      accepts := fun _ => ver Toy.node Toy.rec } = _
  rw [Component.overBlocks_self]

theorem toy_covers {Ω : Type} (ver : Audited → Replay → Prop) :
    Covers (traceBy (Component.Toy.trace (Ω := Ω)) Toy.node ver fun _ => Toy.rec) := by
  rw [traceBy_toy]
  exact Component.covers_univ _ fun _ _ => rfl

/-- The toy trace's first two passes read its one unit once: v0's toy trace's first four do. -/
theorem toy_reads_by {Ω : Type} (ver : Audited → Replay → Prop) (ω : Ω) :
    (traceBy (Component.Toy.trace (Ω := Ω)) Toy.node ver fun _ => Toy.rec).reads ω 2 ≤ 1 := by
  rw [traceBy_toy]
  calc _ ≤ (Component.Toy.trace (Ω := Ω)).reads ω 4 :=
        Finset.sum_le_sum_of_subset (Finset.range_subset_range.2 (by norm_num))
    _ = 1 := Component.toy_reads ω

theorem toy_windows : windowsBy Toy.sum Toy.node Toy.rec = #[Toy.win, Toy.win] := rfl

theorem toy_paced {Ω : Type} (ver : Audited → Replay → Prop) :
    Paced (traceBy (Component.Toy.trace (Ω := Ω)) Toy.node ver fun _ => Toy.rec)
      (paceBy Toy.sum Toy.node fun _ => Toy.rec) := by
  rw [traceBy_toy]
  intro ω _
  refine ⟨fun a b hab => ?_, ?_, ?_, fun g hg => ?_, fun a b _ => ?_, fun g hg => ?_, fun g hg p h1 h2 => ?_⟩
  · show (if a = 0 then 0 else 10) ≤ (if b = 0 then 0 else 10)
    split_ifs <;> omega
  · show P2Line.passesBefore (windowsBy Toy.sum Toy.node Toy.rec) 0 = 0
    rfl
  · show P2Line.passesBefore (windowsBy Toy.sum Toy.node Toy.rec) (Toy.node.windows - 1 + 1) ≤ 1
    rw [toy_windows]
    decide
  · change g ≤ 0 at hg
    obtain rfl := Nat.le_zero.1 hg
    show P2Line.passesBefore (windowsBy Toy.sum Toy.node Toy.rec) 0 <
      P2Line.passesBefore (windowsBy Toy.sum Toy.node Toy.rec) (0 + 1)
    rw [toy_windows]
    decide
  · show (windowsBy Toy.sum Toy.node Toy.rec)[min a (Toy.node.windows - 1)]!.releasedNs ≤
      (windowsBy Toy.sum Toy.node Toy.rec)[min b (Toy.node.windows - 1)]!.releasedNs
    rw [show Toy.node.windows - 1 = 0 from rfl, Nat.min_zero, Nat.min_zero]
  · change g ≤ 0 at hg
    obtain rfl := Nat.le_zero.1 hg
    show (windowsBy Toy.sum Toy.node Toy.rec)[min (0 + 1) (Toy.node.windows - 1)]!.releasedNs ≤
      P2Line.ended (windowsBy Toy.sum Toy.node Toy.rec) 0
    rw [toy_windows]
    decide
  · change g ≤ 0 at hg
    obtain rfl := Nat.le_zero.1 hg
    change p < P2Line.passesBefore (windowsBy Toy.sum Toy.node Toy.rec) (0 + 1) at h2
    rw [toy_windows, show P2Line.passesBefore #[Toy.win, Toy.win] (0 + 1) = 1 by decide] at h2
    obtain rfl : p = 0 := by omega
    show (windowsBy Toy.sum Toy.node Toy.rec)[min 0 (Toy.node.windows - 1)]!.releasedNs ≤
        (if 0 = 0 then 0 else 10) ∧
      (if 0 + 1 = 0 then 0 else 10) ≤ P2Line.ended (windowsBy Toy.sum Toy.node Toy.rec) 0
    rw [toy_windows]
    decide

theorem toy_sketch {Ω : Type} [Fintype Ω] (ver : Audited → Replay → Prop)
    (O : Outside (traceBy (Component.Toy.trace (Ω := Ω)) Toy.node ver fun _ => Toy.rec)) :
    SketchBoundAt (traceBy (Component.Toy.trace (Ω := Ω)) Toy.node ver fun _ => Toy.rec) O 0 := by
  revert O
  rw [traceBy_toy]
  exact fun O => Component.sketch_zero _ O fun _ p hp => by
    change p < 1 at hp
    obtain rfl : p = 0 := by omega
    refine ⟨0, ?_, ?_, ?_⟩
    · show (if 0 = 0 then 0 else 10) ≤ 0
      simp
    · show 0 < (if 0 + 1 = 0 then 0 else 10)
      simp
    · refine Finset.eq_univ_of_forall fun j => Finset.mem_union_left _ ?_
      show j ∈ (if (0 : ℕ) < 10 then Finset.univ else ∅)
      rw [ite_eq_left (by norm_num)]
      exact Finset.mem_univ j

/-- **`LineHeldAtSubprotocol` fails at a verifier that accepts everything.** With an acceptance and a verdict that hold
of every record and one summary recorded for every record, the toy node (no draw, one unit, held in node memory until
tick 10 only, its two passes reading it once) meets every premise at `δs = δnet = 0`, while the summary's credit claims
two units read at every outcome where the certifiers count nothing else entering, which is every outcome: the event has
probability 1 and the bound is `δ = 2^-40`. -/
theorem lineHeldAtSubprotocol_nontrivial : ∃ sm : RemoteAudit.Session.Summary,
    ¬ LineHeldAtSubprotocol (fun _ _ _ _ _ => True) (fun _ _ => True) (fun _ _ => some sm) := by
  refine ⟨Toy.sm, fun h => ?_⟩
  have hb := h Toy.node 0 Unit Unit (fun t => isEmptyElim t) (fun _ _ _ => 0) (fun _ _ => Toy.rec)
    (fun _ _ _ => ∅) (fun _ => ()) Component.Toy.trace Toy.receipts 0 0 (fun _ => .empty) (fun _ => .empty)
    (fun _ => .empty) (Component.noOutside _) (Component.quietIngress _) (toy_paced _) (toy_covers _)
    (toy_sketch _ _) (Component.covered_noOutside _ _) (Component.ingressCounted_quiet _)
    (fun _ t => isEmptyElim t) (fun _ _ t => isEmptyElim t)
    (Component.counted_of _ _ fun _ _ => Finset.Subset.refl _)
  refine absurd hb (not_le.2 ?_)
  rw [Pouw.pr_eq_one]
  · norm_num [Component.binding_empty, RemoteAudit.P2Session.DELTA]
  · exact fun ω => ⟨trivial, fun hq => toy_not_read _ ω (toy_reads_by _ ω) (hq fun _ _ _ _ _ => rfl)⟩

end Properties.RemoteAudit.P2Subprotocol
