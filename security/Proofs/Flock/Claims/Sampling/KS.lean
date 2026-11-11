module

public import Proofs.Flock.Claims.Aggregate
public import Proofs.Flock.Claims.Sampling.Bext
public import Proofs.Flock.Claims.Sampling.Escape
public import Proofs.Flock.Claims.Sampling.Wait
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Ring.Star
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.RingTheory.WittVector.IsPoly
import Mathlib.Tactic.ENatToNat
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
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.ERep
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Binding
import Proofs.Flock.Claims.Sampling.Cost
import Std.Tactic.BVDecide.Normalize.BitVec

/-!
# The deployed sampled verifier's hypotheses, and the sampled extractor's lemmas

The compiled sampled verifier's knowledge soundness (`SamplingGeneric.Compiled.KS`) is proved at every compiled
verifier, and at a frozen record `D` through `SamplingGeneric.ofDeploy`, where it reads named facts about the deployed
sampled verifier, each a hypothesis to discharge there. This file states them, and holds the lemmas its extractor's
proof reads: the opening checks' rerun programs, kept openings and the assembled witness, the waits' kept lists, the
refused sets and the unions' sizes against the waits' slack.

* `DrawLaw`, `Refuses`, `InteriorFixed`, `SessionOpens`: **the hypotheses**. The sampling design over `Design.Pa`
  (`drawLaw_of_slots` proves it of the deployed sample, `Sampling.Select.DrawLaw`); the scope refusal; the interior
  commitments fixed by the first stage; and that a checked unit's Flock witness reveals openings making its proof block
  correct.
* `SlotsCover`, `SessionOpensSlots`: **sessions that cover several checked units**. Every checked unit has the rounds
  and session of the unit at a slot below `S x` (one session per unit at `S = nPU D`, `slotsCover_units`; one session
  over every checked unit at `S = 1`, `slotsCover_one`), and each slot's session has one Flock witness revealing what
  makes every unit it covers correct; that is `SessionOpens` (`sessionOpensSlots_iff`).
* `interiorFixed_of_prefix`: **`InteriorFixed` from the interiors' place in the run**: a run's first `m` messages read
  only its first `m` rounds' coins (`run_msgs_take`), so interior roots read off the messages before the second stage
  are kept by fresh second-stage coins.
* `agree_of_passes`, `val_assemble`, `ιOf_eq`: **the kept openings**. Two kept openings at one position agree unless
  the calls that checked them collide, and the assembled witness opens what they kept.
* `S₁`, `rpow_le_slack`, `Pa_le_miss`: **the unions**. The refused sets are at most `2^{n_PU}`, which `k ≥ 2^j·n_PU`
  accepted runs spread to at most the slack `c` (`j = slackExp`), and a set of `t` proof blocks is missed at most
  as the design misses `t`.
-/

@[expose] public section

namespace Claims.Sampling

open Claims Finset

/-! ## Rerun programs, evaluated -/

theorem flatMapRP_eval {R β γ : Type} (H : Msg → Msg) (run : R → Transcript) (f : β → RP R (List γ)) :
    ∀ l : List β, ((flatMapRP f l).eval H run).1 = l.flatMap fun b => ((f b).eval H run).1
  | [] => rfl
  | b :: bs => by
    simp only [flatMapRP, List.flatMap_cons]
    rw [(RP.eval_bind H run _ _).1, (RP.eval_bind H run _ _).1, flatMapRP_eval H run f bs]
    rfl

theorem mem_zip_finRange {S : Type} {k : ℕ} {l : List S} {j : Fin k} {a : S} (h : l[j.val]? = some a) :
    (j, a) ∈ (List.finRange k).zip l := by
  rw [List.mem_iff_getElem?]
  exact ⟨j.val, List.getElem?_zip_eq_some.2 ⟨by simp, h⟩⟩

theorem mem_checkRevealed {R : Type} (H : Msg → Msg) (run : R → Transcript) (Cm : CommitmentScheme Bool)
    {rs : List (List Msg × Opening Bool)} {p : List Msg × Opening Bool} (hp : p ∈ rs)
    (hok : Cm.Opens H p.1 p.2 = true) : p ∈ ((checkRevealed (R := R) Cm rs).eval H run).1 := by
  unfold checkRevealed
  rw [flatMapRP_eval]
  refine List.mem_flatMap.2 ⟨p, hp, ?_⟩
  rw [(RP.eval_bind H run _ _).1, eval_ofOC]
  unfold CommitmentScheme.Opens at hok
  rw [hok]
  simp [RP.eval]

/-! ## Kept openings pass their checks -/

/-- Everything `p` returns passes its opening check, with that check's calls among `p`'s own. -/
def Passes {R : Type} (H : Msg → Msg) (run : R → Transcript) (Cm : CommitmentScheme Bool)
    (p : RP R (List (List Msg × Opening Bool))) : Prop :=
  ∀ q ∈ (p.eval H run).1, Cm.Opens H q.1 q.2 = true ∧ ∀ t ∈ ((Cm.opens q.1 q.2).eval H).2, t ∈ ownLog H run p

theorem Passes.ret_nil {R : Type} (H : Msg → Msg) (run : R → Transcript) (Cm : CommitmentScheme Bool) :
    Passes H run Cm (.ret []) := by
  intro q hq
  simp [RP.eval] at hq

theorem Passes.bind {R α : Type} {H : Msg → Msg} {run : R → Transcript} {Cm : CommitmentScheme Bool} (p : RP R α)
    {f : α → RP R (List (List Msg × Opening Bool))} (hf : ∀ a, Passes H run Cm (f a)) :
    Passes H run Cm (p.bind f) := by
  intro q hq
  rw [(RP.eval_bind H run _ _).1] at hq
  obtain ⟨h1, h2⟩ := hf _ q hq
  refine ⟨h1, fun t ht => ?_⟩
  rw [ownLog_bind]
  exact List.mem_append_right _ (h2 t ht)

theorem Passes.flatMapRP {R β : Type} {H : Msg → Msg} {run : R → Transcript} {Cm : CommitmentScheme Bool}
    {f : β → RP R (List (List Msg × Opening Bool))} {l : List β} (hf : ∀ b ∈ l, Passes H run Cm (f b)) :
    Passes H run Cm (Sampling.flatMapRP f l) := by
  intro q hq
  obtain ⟨b, hb, hq', hlog⟩ := flatMapRP_mem H run f l q hq
  obtain ⟨h1, h2⟩ := hf b hb q hq'
  exact ⟨h1, fun t ht => hlog t (h2 t ht)⟩

theorem Passes.checkRevealed {R : Type} (H : Msg → Msg) (run : R → Transcript) (Cm : CommitmentScheme Bool)
    (rs : List (List Msg × Opening Bool)) : Passes H run Cm (Sampling.checkRevealed (R := R) Cm rs) := by
  intro q hq
  obtain ⟨-, h1, h2⟩ := checkRevealed_mem H run Cm rs q hq
  exact ⟨h1, h2⟩

/-- Two kept openings under one roots at one position agree, unless the calls that checked them collide. -/
theorem agree_of_passes {H : Msg → Msg} {Cm : CommitmentScheme Bool} (hb : Cm.Binds H) {l : List (List Msg × Opening Bool)}
    {L : List (Msg × Msg)}
    (hp : ∀ q ∈ l, Cm.Opens H q.1 q.2 = true ∧ ∀ t ∈ ((Cm.opens q.1 q.2).eval H).2, t ∈ L) (hL : ¬ LogColl L)
    {q q' : List Msg × Opening Bool} (hq : q ∈ l) (hq' : q' ∈ l) (h1 : q.1 = q'.1) (h2 : q.2.pos = q'.2.pos) :
    q.2.val = q'.2.val := by
  obtain ⟨rt, o⟩ := q
  obtain ⟨rt', o'⟩ := q'
  dsimp only at h1 h2 ⊢
  subst h1
  by_contra hne
  exact hL (disagree_collision hb h2 hne (hp _ hq).1 (hp _ hq').1 (hp _ hq).2 (hp _ hq').2)

/-! ## The assembled witness and the interior values -/

/-- A kept opening under the statement's roots is what the assembled witness opens there, when the kept openings
agree. -/
theorem val_assemble {H : Msg → Msg} {Cm : CommitmentScheme Bool} {s : AuditStatement} {found : List (List Msg × Opening Bool)}
    (hagree : ∀ q ∈ found, ∀ q' ∈ found, q.1 = q'.1 → q.2.pos = q'.2.pos → q.2.val = q'.2.val)
    {q : List Msg × Opening Bool} (hq : q ∈ found) (hr : q.1 = s.roots) (hok : Cm.Opens H s.roots q.2 = true) :
    (assemble s found).val H Cm s q.2.pos = some q.2.val := by
  have hmem : q.2 ∈ (assemble s found).vals :=
    List.mem_filterMap.2 ⟨q, hq, by simp [hr]⟩
  unfold Wit.val
  obtain ⟨o, ho⟩ : ∃ o, (assemble s found).vals.find? (fun o => o.pos == q.2.pos && Cm.Opens H s.roots o) = some o :=
    Option.isSome_iff_exists.1 (List.find?_isSome.2 ⟨q.2, hmem, by simp [hok]⟩)
  rw [ho]
  have hpo := List.find?_some ho
  simp only [Bool.and_eq_true, beq_iff_eq] at hpo
  obtain ⟨a, ha, hao⟩ := List.mem_filterMap.1 (List.mem_of_find?_eq_some ho)
  by_cases har : a.1 = s.roots
  · simp only [har, ite_true, Option.some.injEq] at hao
    subst hao
    simp only [Option.map_some, Option.some.injEq]
    exact hagree a ha q hq (har.trans hr.symm) hpo.1
  · simp [har] at hao

/-- The interior values the kept openings under `roots` give: at each position, the first kept one's value there. -/
def ιOf (found : List (List Msg × Opening Bool)) (roots : List Msg) (g : ℕ) : Bool :=
  ((found.find? fun p => decide (p.1 = roots) && p.2.pos == g).map (·.2.val)).getD false

theorem ιOf_eq {found : List (List Msg × Opening Bool)}
    (hagree : ∀ q ∈ found, ∀ q' ∈ found, q.1 = q'.1 → q.2.pos = q'.2.pos → q.2.val = q'.2.val)
    {q : List Msg × Opening Bool} (hq : q ∈ found) : ιOf found q.1 q.2.pos = q.2.val := by
  unfold ιOf
  obtain ⟨p, hp⟩ : ∃ p, found.find? (fun p => decide (p.1 = q.1) && p.2.pos == q.2.pos) = some p :=
    Option.isSome_iff_exists.1 (List.find?_isSome.2 ⟨q, hq, by simp⟩)
  rw [hp]
  have hpp := List.find?_some hp
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hpp
  exact hagree p (List.mem_of_find?_eq_some hp) q hq hpp.1 hpp.2

/-! ## Waits, as kept lists -/

theorem accList_map {R R' : Type} (acc : R → Prop) [DecidablePred acc] (k : ℕ) (g : R' → R) (l : List R') :
    accList acc k (l.map g) = (accList (fun d => acc (g d)) k l).map g := by
  unfold accList
  rw [List.filter_map, List.map_take]
  rfl

theorem acc_of_accList {R : Type} {acc : R → Prop} [DecidablePred acc] {k : ℕ} {l : List R} {j : ℕ} {r : R}
    (h : (accList acc k l)[j]? = some r) : acc r := by
  have hm := List.mem_of_mem_take (List.mem_of_getElem? h)
  simpa using (List.mem_filter.1 hm).2

theorem accList_length_eq {R : Type} {acc : R → Prop} [DecidablePred acc] {k : ℕ} {l : List R}
    (h : ¬ (accList acc k l).length < k) : (accList acc k l).length = k :=
  le_antisymm (accList_length_le acc k l) (not_lt.1 h)

theorem getElem?_map_pair {R : Type} (run : R → Transcript) {l : List R} {j : ℕ} {r : R} (h : l[j]? = some r) :
    (l.map fun r => (r, (run r).seen))[j]? = some (r, (run r).seen) := by
  rw [List.getElem?_map, h]
  rfl

/-! ## The incorrect set's proof blocks -/

/-- **An incorrect proof block is wrong at any interior values**: at slot `v` of the incorrect set, whatever interior values `ι`, its
proof block `b` is not correct on `ref_ι`. -/
theorem faulty_wrong {H : Msg → Msg} {Cm : CommitmentScheme Bool} {s : AuditStatement} {P : Partition} {w : Wit} {v : ℕ}
    (hv : v ∈ faulty H Cm s P w) (ι : ℕ → Bool) :
    ∃ b, P.blocks[v]? = some b ∧ ¬ s.prog.Correct (s.prog.refIn b (w.val H Cm s) ι) b := by
  classical
  unfold faulty at hv
  obtain ⟨hlt, hf⟩ := mem_filter.1 hv
  have hlt' : v < P.blocks.length := mem_range.1 hlt
  have hget : P.blocks[v]? = some P.blocks[v] := List.getElem?_eq_getElem hlt'
  exact ⟨P.blocks[v], hget, fun hc => hf ⟨_, hget, ι, hc⟩⟩

/-! ## What the deployed sampled verifier is taken to do -/

section Hyps

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- Flock's `γ` at the sampled verifier's rounds, `16·(√(2·k(x)·C) + k(x))²`; the block's `γ` is `c⁴·γF`. -/
noncomputable def γF (D : Deploy Coin Col T) (x : Msg) : ℝ :=
  16 * (Real.sqrt (2 * D.V.k x * D.weight x) + D.V.k x) ^ 2

/-- **The sampling design**: on a statement in scope, for each set `B` of proof blocks, a weight `esc` of the first-stage coins
bounds, at each, the probability over the second stage that the checked units (the first `n_PU`) miss every slot
`v ∈ B`, and averages at most `P_a(B)`. -/
def DrawLaw (D : Deploy Coin Col T) : Prop :=
  ∀ x s Pt, D.parse x = some s → D.partition s = some Pt → InScope Pt s →
    ∀ B ⊆ range Pt.blocks.length, ∃ esc : (Fin (D.V.k x) → Coin) → ℝ, (∀ r, 0 ≤ esc r) ∧
      (∀ r, Pr (fun d : Fin (D.V.k x) → Coin => ∀ v ∈ B, ∀ p : Fin (nPU D x),
          (D.checked x (forkOn (D.stage2 x) r d))[p.val]? ≠ some v) ≤ esc r) ∧
      Ex esc ≤ s.design.Pa Pt B

/-- **The verifier refuses what is out of scope**: an accepted run's statement parses, its query evaluates, and the
pair is in scope. -/
def Refuses (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ x (Ω : Type) (P : Prover Ω) ω r, (run H P D.V x ω r).out = .acc →
    ∃ s Pt, D.parse x = some s ∧ D.partition s = some Pt ∧ InScope Pt s

/-- **The interior commitments precede the second stage**: the roots `iroots` a run shows for proof block `v`'s
interior values are the same at fresh second-stage coins. -/
def InteriorFixed (H : Msg → Msg) (D : Deploy Coin Col T) (iroots : Msg → Seen → ℕ → List Msg) : Prop :=
  ∀ x (Ω : Type) (P : Prover Ω) ω r d v,
    iroots x (run H P D.V x ω (forkOn (D.stage2 x) r d)).seen v = iroots x (run H P D.V x ω r).seen v

/-- **A checked unit's Flock witness makes it correct**: on a statement in scope, at an accepted run that checks slot
`u` and shows its session, a Flock witness in the session's relation reveals openings such that any values `val` and
interior values `ι` agreeing with those that pass (under the statement's roots and the interior roots the run showed for
`u`) make the proof block at `u` correct on `ref_ι`. -/
def SessionOpens (H : Msg → Msg) (D : Deploy Coin Col T) (iroots : Msg → Seen → ℕ → List Msg) : Prop :=
  ∀ x s Pt (Ω : Type) (P : Prover Ω) ω r u xS vS wF,
    D.parse x = some s → D.partition s = some Pt → InScope Pt s →
    (run H P D.V x ω r).out = .acc → (∃ p : Fin (nPU D x), (D.checked x r)[p.val]? = some u) →
    D.session x u (run H P D.V x ω r).seen = some (xS, vS) → Flock.check D.flock.parse xS wF = true →
    ∃ b, Pt.blocks[u]? = some b ∧
      ∀ (val : ℕ → Option Bool) (ι : ℕ → Bool),
        (∀ q ∈ D.reveal s u (run H P D.V x ω r).seen wF, D.commit.Opens H q.1 q.2 = true →
          (q.1 = s.roots → val q.2.pos = some q.2.val) ∧
          (q.1 = iroots x (run H P D.V x ω r).seen u → ι q.2.pos = q.2.val)) →
        s.prog.Correct (s.prog.refIn b val ι) b

/-! ### Sessions that cover several checked units -/

/-- **The slots' sessions are every checked unit's**: at coins `r`, every checked unit has the rounds and the session of
the unit checked at some slot below `S x`. One session per unit is `S = nPU D` (`slotsCover_units`); one session over
every checked unit, as the one-stage audit runs, is `S = 1` (`slotsCover_one`). -/
def SlotsCover (D : Deploy Coin Col T) (S : Msg → ℕ) : Prop :=
  ∀ x r u, (∃ p : Fin (nPU D x), (D.checked x r)[p.val]? = some u) →
    ∃ p < S x, ∃ u', (D.checked x r)[p]? = some u' ∧ D.rounds x u = D.rounds x u' ∧ D.session x u = D.session x u'

/-- **`SessionOpens` a session at a time**: on a statement in scope, at an accepted run, the session of the unit checked
at a slot below `S x`, shown with statement `xS`, and one Flock witness in its relation reveal, for every checked unit
that session covers, openings that make that unit correct, as `SessionOpens` says of each unit. -/
def SessionOpensSlots (H : Msg → Msg) (D : Deploy Coin Col T) (iroots : Msg → Seen → ℕ → List Msg)
    (S : Msg → ℕ) : Prop :=
  ∀ x s Pt (Ω : Type) (P : Prover Ω) ω r (p : ℕ) u' xS vS wF,
    D.parse x = some s → D.partition s = some Pt → InScope Pt s →
    (run H P D.V x ω r).out = .acc → p < S x → (D.checked x r)[p]? = some u' →
    D.session x u' (run H P D.V x ω r).seen = some (xS, vS) → Flock.check D.flock.parse xS wF = true →
    ∀ u, (∃ q : Fin (nPU D x), (D.checked x r)[q.val]? = some u) → D.session x u = D.session x u' →
    ∃ b, Pt.blocks[u]? = some b ∧
      ∀ (val : ℕ → Option Bool) (ι : ℕ → Bool),
        (∀ q ∈ D.reveal s u (run H P D.V x ω r).seen wF, D.commit.Opens H q.1 q.2 = true →
          (q.1 = s.roots → val q.2.pos = some q.2.val) ∧
          (q.1 = iroots x (run H P D.V x ω r).seen u → ι q.2.pos = q.2.val)) →
        s.prog.Correct (s.prog.refIn b val ι) b

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- **One session per unit covers every record.** -/
theorem slotsCover_units (D : Deploy Coin Col T) : SlotsCover D (nPU D) := by
  rintro x r u ⟨p, hp⟩
  exact ⟨p.val, p.isLt, u, hp, rfl, rfl⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- **One session over every checked unit**: where every unit has the same rounds and session, the first checked unit's
covers them all. -/
theorem slotsCover_one (D : Deploy Coin Col T) (hr : ∀ x u u', D.rounds x u = D.rounds x u')
    (hs : ∀ x u u', D.session x u = D.session x u') : SlotsCover D (fun _ => 1) := by
  rintro x r u ⟨p, hp⟩
  cases hc : D.checked x r with
  | nil => rw [hc] at hp; simp at hp
  | cons u' l => exact ⟨0, Nat.one_pos, u', by simp, hr x u u', hs x u u'⟩

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
/-- **`SessionOpens` from its slot form**: each checked unit's session is a slot's, whose witness covers it. -/
theorem sessionOpens_of_slots {H : Msg → Msg} {D : Deploy Coin Col T} {iroots : Msg → Seen → ℕ → List Msg}
    {S : Msg → ℕ} (hc : SlotsCover D S) (h : SessionOpensSlots H D iroots S) : SessionOpens H D iroots := by
  intro x s Pt Ω P ω r u xS vS wF hs hP hin hacc hu hsess hck
  obtain ⟨p, hp, u', hu', -, hss⟩ := hc x r u hu
  exact h x s Pt Ω P ω r p u' xS vS wF hs hP hin hacc hp hu' (hss ▸ hsess) hck u hu hss

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
/-- **The slot form is `SessionOpens`** wherever the slots cover the checked units, one session per unit among them. -/
theorem sessionOpensSlots_iff {H : Msg → Msg} {D : Deploy Coin Col T} {iroots : Msg → Seen → ℕ → List Msg}
    {S : Msg → ℕ} (hc : SlotsCover D S) : SessionOpensSlots H D iroots S ↔ SessionOpens H D iroots :=
  ⟨sessionOpens_of_slots hc, fun h x s Pt Ω P ω r _ _ xS vS wF hs hP hin hacc _ _ hsess hck u hu hss =>
    h x s Pt Ω P ω r u xS vS wF hs hP hin hacc hu (hss ▸ hsess) hck⟩

end Hyps

/-! ## The interior roots from the run's first rounds -/

section Interior

theorem stepWith_msgs_prefix {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (vnext : List Msg → OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript),
      ∃ ext, (stepWith H P x ω vnext n hist as t).msgs = t.msgs ++ ext := by
  intro n
  induction n with
  | zero => intro hist as t; exact ⟨[], by simp [stepWith]⟩
  | succ n ih =>
    intro hist as t
    simp only [stepWith]
    split
    · obtain ⟨ext, he⟩ := ih (hist ++ [(((P.next x hist ω).eval H).1, _)]) (as ++ [((P.next x hist ω).eval H).1])
        { t with msgs := t.msgs ++
                   [(((P.next x hist ω).eval H).1, ((vnext (as ++ [((P.next x hist ω).eval H).1])).eval H).1)],
                 pLog := t.pLog ++ ((P.next x hist ω).eval H).2,
                 vLog := t.vLog ++ ((vnext (as ++ [((P.next x hist ω).eval H).1])).eval H).2 }
      rw [he]
      exact ⟨_, List.append_assoc _ _ _⟩
    · exact ⟨_, rfl⟩
    · exact ⟨_, rfl⟩

/-- **The first `m` rounds read only the verifier's first `m` answers**: two verifiers that answer alike on at most
`m` prover messages give runs whose first `m` messages agree. -/
theorem stepWith_take {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vnext vnext' : List Msg → OC Out)
    (m : ℕ) (hv : ∀ as : List Msg, as.length ≤ m → vnext as = vnext' as) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript), as.length = t.msgs.length →
      t.msgs.length ≤ m →
      (stepWith H P x ω vnext n hist as t).msgs.take m = (stepWith H P x ω vnext' n hist as t).msgs.take m := by
  intro n
  induction n with
  | zero => intro hist as t _ _; rfl
  | succ n ih =>
    intro hist as t hlen hm
    rcases Nat.lt_or_ge t.msgs.length m with hlt | hge
    · have hv' := hv (as ++ [((P.next x hist ω).eval H).1]) (by simp; omega)
      simp only [stepWith]
      rw [hv']
      split
      · exact ih _ _ _ (by simp [hlen]) (by simp; omega)
      · rfl
      · rfl
    · obtain ⟨e, he⟩ := stepWith_msgs_prefix H P x ω vnext (n + 1) hist as t
      obtain ⟨e', he'⟩ := stepWith_msgs_prefix H P x ω vnext' (n + 1) hist as t
      have hm' : t.msgs.length = m := le_antisymm hm hge
      rw [he, he', List.take_append_of_le_length hm'.ge, List.take_append_of_le_length hm'.ge]

/-- **Acceptance is the verifier's answer**: rounds that end accepting had the verifier answer `acc` to some prover
messages. -/
theorem stepWith_acc_next {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vnext : List Msg → OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript),
      (stepWith H P x ω vnext n hist as t).out = .acc → ∃ as', ((vnext as').eval H).1 = .acc := by
  intro n
  induction n with
  | zero => intro hist as t h; simp [stepWith] at h
  | succ n ih =>
    intro hist as t h
    simp only [stepWith] at h
    split at h
    · exact ih _ _ _ h
    · exact ⟨_, by assumption⟩
    · simp at h

variable {Coin : Type} [Inhabited Coin]

/-- **An accepted run's verifier answered `acc`** to some prover messages and coins. -/
theorem run_acc_next {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg) (ω : Ω)
    (r : Fin (V.k x) → Coin) (h : (run H P V x ω r).out = .acc) : ∃ as cs, ((V.next x as cs).eval H).1 = .acc := by
  obtain ⟨as, has⟩ := stepWith_acc_next H P x ω _ _ _ _ _ h
  exact ⟨as, _, has⟩

/-- **A run's first `m` messages read only its first `m` rounds' coins.** -/
theorem run_msgs_take {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg) (ω : Ω) (m : ℕ)
    (r r' : Fin (V.k x) → Coin) (h : ∀ t : Fin (V.k x), t.val < m → r t = r' t) :
    (run H P V x ω r).msgs.take m = (run H P V x ω r').msgs.take m := by
  unfold run
  refine stepWith_take H P x ω _ _ m (fun as has => ?_) _ _ _ _ rfl (Nat.zero_le _)
  congr 1
  refine List.map_congr_left fun i hi => ?_
  have hi' : i < m := lt_of_lt_of_le (List.mem_range.1 hi) has
  unfold coinSeq
  split
  · exact h _ hi'
  · rfl

/-- **`InteriorFixed` from the interiors' place in the run**: where the second stage starts at round `t₀ x` and the
interior roots a run shows are read off its first `t₀ x` messages (the interiors are registered after the sample and
before any proof), fresh second-stage coins keep them. -/
theorem interiorFixed_of_prefix {Col T : Type} (H : Msg → Msg) (D : Deploy Coin Col T)
    (iroots : Msg → Seen → ℕ → List Msg) (t₀ : Msg → ℕ) (hst : ∀ x, ∀ t ∈ D.stage2 x, t₀ x ≤ t.val)
    (hread : ∀ x (s s' : Seen) v, s.msgs.take (t₀ x) = s'.msgs.take (t₀ x) → iroots x s v = iroots x s' v) :
    InteriorFixed H D iroots := by
  intro x Ω P ω r d v
  refine hread x _ _ v (run_msgs_take H P D.V x ω (t₀ x) _ _ fun t ht => ?_)
  have hn : t ∉ D.stage2 x := fun hmem => absurd (hst x t hmem) (not_le.2 ht)
  simp [forkOn, hn]

end Interior

/-! ## The refused sets -/

/-- The incorrect sets the relation at `δ` refuses: those of acceptance probability at most `δ`. -/
noncomputable def S₁ (d : Design) (Pt : Partition) (δ : ℝ) : Finset (Finset ℕ) :=
  (range Pt.blocks.length).powerset.filter fun B => d.Pa Pt B ≤ δ

/-! ## The unions' sizes, against the waits' slack -/

theorem card_S₁_le (d : Design) (Pt : Partition) (δ : ℝ) : (S₁ d Pt δ).card ≤ 2 ^ Pt.blocks.length :=
  (card_filter_le _ _).trans (by rw [card_powerset, card_range])

theorem one_le_slack : 1 ≤ slack := by
  unfold slack
  exact Real.one_le_rpow (by norm_num) (by positivity)

/-- At most `2^m` sets spread over `k ≥ 2^j·m` accepted runs (`j = slackExp`) cost at most the slack. -/
theorem rpow_le_slack {N m k : ℕ} (hN : N ≤ 2 ^ m) (hk : 2 ^ slackExp * m ≤ k) :
    (N : ℝ) ^ (1 / (k : ℝ)) ≤ slack := by
  rcases Nat.eq_zero_or_pos k with rfl | hk0
  · rw [Nat.cast_zero, div_zero, Real.rpow_zero]
    exact one_le_slack
  have hkR : (0 : ℝ) < k := by exact_mod_cast hk0
  have hj : (0 : ℝ) < 2 ^ slackExp := by positivity
  calc (N : ℝ) ^ (1 / (k : ℝ)) ≤ ((2 : ℝ) ^ m) ^ (1 / (k : ℝ)) :=
        Real.rpow_le_rpow (Nat.cast_nonneg _) (by exact_mod_cast hN) (by positivity)
    _ = (2 : ℝ) ^ ((m : ℝ) * (1 / (k : ℝ))) := by
        rw [← Real.rpow_natCast, ← Real.rpow_mul (by norm_num)]
    _ ≤ slack := by
        unfold slack
        refine Real.rpow_le_rpow_of_exponent_le (by norm_num) ?_
        rw [Real.rpow_neg (by norm_num), Real.rpow_natCast, mul_one_div, div_le_iff₀ hkR, ← div_eq_inv_mul,
          le_div_iff₀ hj, mul_comm]
        exact_mod_cast hk

/-! ## The waits' numbers -/

section Bound

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem nPU_eq {D : Deploy Coin Col T} {x : Msg} {s : AuditStatement} {Pt : Partition} (hs : D.parse x = some s)
    (hPt : D.partition s = some Pt) : nPU D x = Pt.blocks.length := by
  simp only [nPU, hs, Deploy.part, hPt, Option.getD_some]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem k_pos (D : Deploy Coin Col T) (x : Msg) : 0 < D.k x := by
  unfold Deploy.k; positivity

theorem Pr_or_le {α : Type} [Fintype α] {p q r : α → Prop} {a b : ℝ} (h : ∀ s, p s → q s ∨ r s)
    (hq : Pr q ≤ a) (hr : Pr r ≤ b) : Pr p ≤ a + b :=
  (Pr_mono h).trans ((Pr_union_le q r).trans (add_le_add hq hr))

end Bound

/-! ## The design's miss -/

/-- A set of at least `t` proof blocks has acceptance probability at most the design's miss at `t`. -/
theorem Pa_le_miss (d : Design) (P : Partition) {t : ℕ} {B : Finset ℕ} (hB : B ⊆ range P.blocks.length)
    (ht : t ≤ B.card) : d.Pa P B ≤ d.miss P t := by
  have : Finite {B : Finset ℕ // B ⊆ range P.blocks.length ∧ t ≤ B.card} :=
    Finite.of_injective (fun B => (⟨B.1, mem_powerset.2 B.2.1⟩ : (range P.blocks.length).powerset))
      fun a b h => Subtype.ext (congrArg (fun z : (range P.blocks.length).powerset => z.1) h)
  exact le_ciSup (f := fun B : {B : Finset ℕ // B ⊆ range P.blocks.length ∧ t ≤ B.card} => d.Pa P B.1)
    (Set.finite_range _).bddAbove ⟨B, hB, ht⟩

end Claims.Sampling
