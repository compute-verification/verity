module

public import Definitions.Claims.SamplingTwo
public import Proofs.Flock.SamplingGeneric.Assemble
public import Proofs.Flock.SamplingGeneric.Session
import Proofs.Flock.Claims.Aggregate
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Extract

/-!
# The two-stage compiler is knowledge sound at the footprint

`CompilerKSTwo` on `CompilerKS`'s machinery: the compiled verifier's extractor, waits, errors and numbers are
`Compiled`'s, the sessions' bound is `Compiled.sessionKS_of` (it reads only the contract's `fixed` and `runs`,
`Compiled.RunsFixed`), and the extraction's deterministic half is `Compiled.split`'s with the interior roots read at a
replay block. Only the failure event changes: its candidate sets are sets `Y` of proof blocks whose counts per cell
(`counts`) are a footprint of acceptance probability at most `δ` (`S₁Two`), and a kept inner run is bad when it finds no
block of any set at least as large as `Y` in every cell (`above`), which every kept outer run's incorrect set at its own
interior values is.

* `escCell_nonneg`, `escCell_le_one`, `escCell_zero`, `escCell_anti`: a cell's escape is a probability, `1` at no
  incorrect proof block and antitone in their count.
* `prod_filter_eq_sum`, `Ex_esc_le`: **the first stage**. Catching each replay block independently, `∏_{ρ∈S} e_ρ` is
  the sum over sets `T` of the indicator that `S` misses `T` times `T`'s weight, so its average over the replay sample
  is at most `P̄_a` (`Sessions.replayDraws`).
* `inner_badTwo_le`, `outer_badTwo_le`, `pr_failsTwo_le_of`: **the failure event's probability**, at `Compiled`'s
  `κ_δ` and `γ`: the inner wait's union bound over `above Y` (at most `2^{n_PU}` sets) at the second stage's miss
  (`Sessions.secondDraws`, at most `∏_{ρ∈S} e_ρ` at `Y`'s counts by `escCell_anti`), and the outer wait's over `S₁Two`
  (at most `2^{n_PU}`), the two within the waits' slack.
* `exists_counts`, `splitTwo`: **the deterministic half**. At an accepted run whose extraction fails, without an own-call
  collision the kept openings assemble a witness whose footprint is refused at `δ`; a `Y` with its counts is in
  `S₁Two`, and each kept outer run's incorrect set at the interior values its interior roots open is in `above Y`
  (each cell's at least the footprint, the infimum); a block of it found at a kept inner run would be correct
  (`Sessions.opens`, with the interior roots fixed by `Sessions.interior`).
* `compilerKSTwo`: **`CompilerKSTwo`**.
-/

@[expose] public section

namespace Claims.Sampling

open Finset

theorem escCell_nonneg (n k m : ℕ) : 0 ≤ escCell n k m := by
  unfold escCell
  positivity

theorem escCell_le_one (n k m : ℕ) : escCell n k m ≤ 1 := by
  unfold escCell
  rw [div_le_one (by exact_mod_cast Nat.choose_pos (min_le_right k n))]
  exact_mod_cast Nat.choose_le_choose _ (Nat.sub_le n m)

/-- A cell with no incorrect proof block always escapes. -/
theorem escCell_zero (n k : ℕ) : escCell n k 0 = 1 := by
  unfold escCell
  rw [Nat.sub_zero]
  exact div_self (by exact_mod_cast (Nat.choose_pos (min_le_right k n)).ne')

/-- **A cell's escape is antitone** in its incorrect count. -/
theorem escCell_anti (n k : ℕ) {m m' : ℕ} (h : m ≤ m') : escCell n k m' ≤ escCell n k m := by
  unfold escCell
  exact div_le_div_of_nonneg_right (by exact_mod_cast Nat.choose_le_choose _ (Nat.sub_le_sub_left h n))
    (Nat.cast_nonneg _)

theorem Design.Two.esc_nonneg (d : Design.Two) (Pt : Partition) (fp : ℕ → ℕ → ℕ) (ρ : ℕ) : 0 ≤ d.esc Pt fp ρ :=
  prod_nonneg fun _ _ => escCell_nonneg _ _ _

theorem Design.Two.esc_le_one (d : Design.Two) (Pt : Partition) (fp : ℕ → ℕ → ℕ) (ρ : ℕ) : d.esc Pt fp ρ ≤ 1 :=
  prod_le_one₀ (fun _ _ => escCell_nonneg _ _ _) fun _ _ => escCell_le_one _ _ _

/-- `P̄_a` reads a footprint only on the grid of replay blocks and strata. -/
theorem Design.Two.Pa_congr (d : Design.Two) (Pt : Partition) {f g : ℕ → ℕ → ℕ}
    (h : ∀ ρ < d.part.blocks.length, ∀ t < Pt.strata.length, f ρ t = g ρ t) : d.Pa Pt f = d.Pa Pt g := by
  have he : ∀ ρ ∈ range d.part.blocks.length, d.esc Pt f ρ = d.esc Pt g ρ := fun ρ hρ =>
    prod_congr rfl fun t ht => by rw [h ρ (mem_range.1 hρ) t (mem_range.1 ht)]
  unfold Design.Two.Pa
  refine sum_congr rfl fun T hT => ?_
  have hT' := mem_powerset.1 hT
  rw [prod_congr rfl fun ρ hρ => by rw [he ρ (hT' hρ)], prod_congr rfl fun ρ hρ => he ρ (mem_sdiff.1 hρ).1]

/-- **Catching each replay block independently**: the product of the escapes over the sampled replay blocks is the sum,
over sets `T`, of the indicator that the sample misses `T` times `T`'s weight `∏_{ρ∈T}(1 − e_ρ)·∏_{ρ∉T} e_ρ`. -/
theorem prod_filter_eq_sum (N : Finset ℕ) (S : ℕ → Prop) [DecidablePred S] (e : ℕ → ℝ) :
    ∏ ρ ∈ N.filter S, e ρ =
      ∑ T ∈ N.powerset, (if ∀ ρ ∈ T, ¬ S ρ then (1 : ℝ) else 0) * ((∏ ρ ∈ T, (1 - e ρ)) * ∏ ρ ∈ N \ T, e ρ) := by
  rw [prod_filter]
  have h : ∀ ρ, (if S ρ then e ρ else 1) = (if S ρ then 0 else 1 - e ρ) + e ρ := by
    intro ρ
    split_ifs <;> ring
  rw [prod_congr rfl fun ρ _ => h ρ, prod_add]
  refine sum_congr rfl fun T _ => ?_
  by_cases hT : ∀ ρ ∈ T, ¬ S ρ
  · rw [if_pos hT, one_mul]
    congr 1
    exact prod_congr rfl fun ρ hρ => if_neg (hT ρ hρ)
  · rw [if_neg hT, zero_mul]
    push_neg at hT
    obtain ⟨ρ, hρ, hSρ⟩ := hT
    exact mul_eq_zero_of_left (prod_eq_zero hρ (by rw [if_pos hSρ])) _

/-- **The first stage's average**: over coins whose replay sample misses each set `T` of replay blocks with probability
at most the first stage's `P_a(T)`, the product of the escapes over the sampled replay blocks averages at most
`P̄_a`. -/
theorem Ex_esc_le {R : Type} [Fintype R] (d : Design.Two) (Pt : Partition) (fp : ℕ → ℕ → ℕ) (S : R → List ℕ)
    (hreplay : ∀ T ⊆ range d.part.blocks.length, Pr (fun r => ∀ ρ ∈ S r, ρ ∉ T) ≤ d.replay.Pa d.part T) :
    Ex (fun r => ∏ ρ ∈ (range d.part.blocks.length).filter (· ∈ S r), d.esc Pt fp ρ) ≤ d.Pa Pt fp := by
  classical
  have hw0 : ∀ T ∈ (range d.part.blocks.length).powerset,
      0 ≤ (∏ ρ ∈ T, (1 - d.esc Pt fp ρ)) * ∏ ρ ∈ range d.part.blocks.length \ T, d.esc Pt fp ρ := fun T _ =>
    mul_nonneg (prod_nonneg fun ρ _ => sub_nonneg.2 (d.esc_le_one Pt fp ρ))
      (prod_nonneg fun ρ _ => d.esc_nonneg Pt fp ρ)
  have hind : ∀ T : Finset ℕ,
      Ex (fun r => (if ∀ ρ ∈ T, ¬ (ρ ∈ S r) then (1 : ℝ) else 0) *
        ((∏ ρ ∈ T, (1 - d.esc Pt fp ρ)) * ∏ ρ ∈ range d.part.blocks.length \ T, d.esc Pt fp ρ)) =
      Pr (fun r => ∀ ρ ∈ T, ¬ (ρ ∈ S r)) *
        ((∏ ρ ∈ T, (1 - d.esc Pt fp ρ)) * ∏ ρ ∈ range d.part.blocks.length \ T, d.esc Pt fp ρ) := by
    intro T
    rw [Ex_mul_right, Ex_ind]
  calc Ex (fun r => ∏ ρ ∈ (range d.part.blocks.length).filter (· ∈ S r), d.esc Pt fp ρ)
      = Ex (fun r => ∑ T ∈ (range d.part.blocks.length).powerset,
          (if ∀ ρ ∈ T, ¬ (ρ ∈ S r) then (1 : ℝ) else 0) *
            ((∏ ρ ∈ T, (1 - d.esc Pt fp ρ)) * ∏ ρ ∈ range d.part.blocks.length \ T, d.esc Pt fp ρ)) := by
        congr 1
        funext r
        exact prod_filter_eq_sum _ (· ∈ S r) _
    _ = ∑ T ∈ (range d.part.blocks.length).powerset, Pr (fun r => ∀ ρ ∈ T, ¬ (ρ ∈ S r)) *
          ((∏ ρ ∈ T, (1 - d.esc Pt fp ρ)) * ∏ ρ ∈ range d.part.blocks.length \ T, d.esc Pt fp ρ) := by
        rw [Ex_finsum]
        exact sum_congr rfl fun T _ => hind T
    _ ≤ d.Pa Pt fp := by
        unfold Design.Two.Pa
        refine sum_le_sum fun T hT => mul_le_mul_of_nonneg_right ?_ (hw0 T hT)
        exact (Pr_congr fun r => ⟨fun h ρ hρ hρT => h ρ hρT hρ, fun h ρ hρT hρ => h ρ hρ hρT⟩).trans_le
          (hreplay T (mem_powerset.1 hT))

theorem cell_subset (Pt Pρ : Partition) (ρ t : ℕ) : cell Pt Pρ ρ t ⊆ range Pt.blocks.length :=
  filter_subset _ _

theorem replayOf_of_mem_cell {Pt Pρ : Partition} {ρ t v : ℕ} (h : v ∈ cell Pt Pρ ρ t) : replayOf Pt Pρ v = some ρ :=
  (mem_filter.1 h).2.1

theorem cell_disj {Pt Pρ : Partition} {ρ t ρ' t' v : ℕ} (h : v ∈ cell Pt Pρ ρ t) (h' : v ∈ cell Pt Pρ ρ' t') :
    ρ = ρ' ∧ t = t' :=
  ⟨Option.some.inj ((replayOf_of_mem_cell h).symm.trans (replayOf_of_mem_cell h')),
    (mem_filter.1 h).2.2.symm.trans (mem_filter.1 h').2.2⟩

theorem footprint_le (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (Pt Pρ : Partition) (w : Wit)
    (ρ t : ℕ) (ι : ℕ → Bool) : footprint H Cm s Pt Pρ w ρ t ≤ (wrongIn H Cm s Pt Pρ w ρ t ι).card :=
  ciInf_le (OrderBot.bddBelow _) ι

open Classical in
theorem footprint_le_cell (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (Pt Pρ : Partition)
    (w : Wit) (ρ t : ℕ) : footprint H Cm s Pt Pρ w ρ t ≤ (cell Pt Pρ ρ t).card :=
  (footprint_le H Cm s Pt Pρ w ρ t fun _ => false).trans (card_le_card (filter_subset _ _))

end Claims.Sampling

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScopeTwo RTwo cell escCell replayOf footprint wrongIn forkOn
  forkOn_forkOn' accList AllBad wait_bad_le Pr_or_le slack slack_pos Pr_acc_forkOn Opening OwnColl agree_of_passes
  val_assemble ιOf ιOf_eq acc_of_accList accList_length_eq assemble escCell_anti Ex_esc_le cell_subset cell_disj
  replayOf_of_mem_cell footprint_le footprint_le_cell)
open Claims.Recursion (serve viewing)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## The failure event -/

/-- `Y`'s proof blocks in cell `(ρ, t)`. -/
def counts (Pt Pρ : Partition) (Y : Finset ℕ) (ρ t : ℕ) : ℕ := (Y ∩ cell Pt Pρ ρ t).card

open Classical in
/-- **The candidate sets**: sets of proof blocks whose counts per cell are a footprint of acceptance probability at
most `δ`. -/
noncomputable def S₁Two (Pt : Partition) (d : Claims.Sampling.Design.Two) (δ : ℝ) : Finset (Finset ℕ) :=
  (range Pt.blocks.length).powerset.filter fun Y => d.Pa Pt (counts Pt d.part Y) ≤ δ

open Classical in
/-- The sets of proof blocks at least as large as `Y` in every cell of the grid. -/
noncomputable def above (Pt : Partition) (d : Claims.Sampling.Design.Two) (Y : Finset ℕ) : Finset (Finset ℕ) :=
  (range Pt.blocks.length).powerset.filter fun X =>
    ∀ ρ < d.part.blocks.length, ∀ t < Pt.strata.length, counts Pt d.part Y ρ t ≤ counts Pt d.part X ρ t

/-- **An outer run bad for the sets `Xs`**, its second stage `g`: the inner wait comes up short, or for some `X ∈ Xs`
no kept inner run has a proof block of `X` good. -/
def BadOuterTwo (L : CompiledTwo Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Trace)
    (found : (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop) (Xs : Finset (Finset ℕ))
    (g : (Fin (L.V.k x) → Coin) → Fin (L.V.k x) → Coin) (e : L.E1 x) : Prop :=
  (accList (fun d => (run' (g d)).out = .acc) (L.k x) ((List.finRange (L.cap x)).map e.1)).length < L.k x ∨
    ∃ X ∈ Xs, AllBad (fun d => (run' (g d)).out = .acc) (L.k x)
      (fun d uc => ∀ v ∈ X, ¬ L.GoodAt x found (g d) v uc) ((List.finRange (L.cap x)).map e.1) e.2

/-- **The extraction's failure event**: the statement is in scope, and the outer wait comes up short or, for some
candidate `Y`, every kept outer run is bad for the sets at least as large as `Y`. -/
def FailsTwo (L : CompiledTwo Coin A) (δ : ℝ) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Trace)
    (found : (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop) (c : L.ECoins x) : Prop :=
  ∃ s Pt d, L.parse x = some s ∧ L.partition s = some Pt ∧ L.two s = some d ∧ InScopeTwo Pt d s ∧
    ((accList (fun r => (run' r).out = .acc) (L.k x) ((List.finRange (L.cap x)).map c.1)).length < L.k x ∨
      ∃ Y ∈ S₁Two Pt d δ, AllBad (fun r => (run' r).out = .acc) (L.k x)
        (fun r e => L.BadOuterTwo x run' found (above Pt d Y) (forkOn (L.stage2 x) r) e)
        ((List.finRange (L.cap x)).map c.1) (L.toE1 x c))

/-! ## The failure event's probability -/

omit [Inhabited Coin] in
/-- **One outer run's inner wait**, at sets `Xs` of proof blocks below `n_PU`, at most `2^{n_PU}` of them, each missed by
the second stage at `r` with probability at most `esc`: `Compiled.inner_bad_le`'s union bound at the picked slot for each
`X`, and the wait's over `Xs`. -/
theorem inner_badTwo_le (L : CompiledTwo Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Trace)
    (found : (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop) {Xs : Finset (Finset ℕ)}
    (hXs : ∀ X ∈ Xs, ∀ v ∈ X, v < L.nPU x) (hcard : Xs.card ≤ 2 ^ L.nPU x)
    (hslots : ∀ r', (L.checked x r').length ≤ L.nPU x) (r : Fin (L.V.k x) → Coin)
    {esc κr : ℝ} (hesc0 : 0 ≤ esc) (hκr0 : 0 ≤ κr)
    (hesc : ∀ X ∈ Xs, Pr (fun d : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x (forkOn (L.stage2 x) r d), u ∉ X) ≤ esc)
    (hsess : ∀ sel : List ℕ → Option (Fin (L.nPU x)),
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        (run' (forkOn (L.stage2 x) r p.1)).out = .acc ∧
          ∃ i, sel (L.checked x (forkOn (L.stage2 x) r p.1)) = some i ∧
            ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.stage2 x) r p.1))[i.val]? = some u.val ∧
              ¬ found u (forkOn (L.stage2 x) r p.1) (p.2 i u)) ≤ κr) :
    Pr (fun d : Fin (L.V.k x) → Coin => (run' (forkOn (L.stage2 x) r d)).out = .acc) *
        Pr (fun e : L.E1 x => L.BadOuterTwo x run' found Xs (forkOn (L.stage2 x) r) e) ≤
      2 * (L.k x : ℝ) / L.cap x + Real.exp (-(L.k x : ℝ) / 8) + slack * (esc + κr) := by
  have hy : ∀ X ∈ Xs,
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        (run' (forkOn (L.stage2 x) r p.1)).out = .acc ∧
          ∀ v ∈ X, ¬ L.GoodAt x found (forkOn (L.stage2 x) r p.1) v p.2) ≤ esc + κr := by
    intro X hX
    let sel : List ℕ → Option (Fin (L.nPU x)) := fun l =>
      (List.finRange (L.nPU x)).find? fun i => decide (∃ v ∈ X, l[i.val]? = some v)
    have hsel : ∀ l : List ℕ, l.length ≤ L.nPU x → ∀ v ∈ l, v ∈ X →
        ∃ i, sel l = some i ∧ ∃ v ∈ X, l[i.val]? = some v := by
      intro l hl v hvl hvX
      obtain ⟨j, hj, hjv⟩ := List.getElem_of_mem hvl
      have hjv' : l[(⟨j, lt_of_lt_of_le hj hl⟩ : Fin (L.nPU x)).val]? = some v := by
        show l[j]? = some v
        rw [List.getElem?_eq_getElem hj, hjv]
      obtain ⟨i, hi⟩ := Option.isSome_iff_exists.1
        ((List.find?_isSome (p := fun i : Fin (L.nPU x) => decide (∃ v ∈ X, l[i.val]? = some v))).2
          ⟨⟨j, lt_of_lt_of_le hj hl⟩, List.mem_finRange _, decide_eq_true ⟨v, hvX, hjv'⟩⟩)
      exact ⟨i, hi, of_decide_eq_true
        (List.find?_some (p := fun i : Fin (L.nPU x) => decide (∃ v ∈ X, l[i.val]? = some v)) hi)⟩
    have hmiss := (Pr_fst (β := L.SlotEC x) (fun d : Fin (L.V.k x) → Coin =>
      ∀ u ∈ L.checked x (forkOn (L.stage2 x) r d), u ∉ X)).trans_le (hesc X hX)
    refine Claims.Sampling.Pr_or_le (fun p hp => ?_) hmiss (hsess sel)
    by_cases hm : ∃ v ∈ L.checked x (forkOn (L.stage2 x) r p.1), v ∈ X
    · obtain ⟨v, hvl, hvX⟩ := hm
      obtain ⟨i, hi, v', hv'X, hv'⟩ := hsel _ (hslots _) v hvl hvX
      exact Or.inr ⟨hp.1, i, hi, ⟨v', hXs X hX v' hv'X⟩, hv',
        fun hf => hp.2 v' hv'X ⟨i, hXs X hX v' hv'X, hv', hf⟩⟩
    · exact Or.inl fun v hvl hvX => hm ⟨v, hvl, hvX⟩
  have hw := wait_bad_le (fun d => (run' (forkOn (L.stage2 x) r d)).out = .acc) (L.k x) (L.cap x) (L.cap_pos x)
    (L.k_pos x) Xs (fun X d uc => ∀ v ∈ X, ¬ L.GoodAt x found (forkOn (L.stage2 x) r d) v uc)
    (add_nonneg hesc0 hκr0) hy
  have hc : ((Xs.card : ℕ) : ℝ) ^ (1 / (L.k x : ℝ)) ≤ slack := L.rpow_k_le_slack x hcard (by omega)
  exact hw.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_right hc (add_nonneg hesc0 hκr0)))

set_option synthInstance.maxSize 1024 in
omit [Inhabited Coin] in
/-- **One tape's outer wait**: on a statement in scope at `(s, Pt, d)`, where the first stage misses each set of replay
blocks with probability at most its `P_a` and the second stage misses `X` with at most the product of its cells'
escapes, the outer wait (times its acceptance) comes up short or, for some candidate `Y`, keeps only runs bad for the
sets at least as large as `Y`, with probability at most `T + c·(T + c·(δ + E[κr]))`, `T = 2k/K + e^{−k/8}`. -/
theorem outer_badTwo_le (L : CompiledTwo Coin A) {δ : ℝ} (hδ : 0 ≤ δ) (x : Msg)
    (run' : (Fin (L.V.k x) → Coin) → Trace)
    (found : (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop)
    (Pt : Partition) (d : Claims.Sampling.Design.Two) (hPU : L.nPU x = Pt.blocks.length)
    (κr : (Fin (L.V.k x) → Coin) → ℝ) (hκr0 : ∀ r, 0 ≤ κr r)
    (hreplay : ∀ T ⊆ range d.part.blocks.length,
      Pr (fun r : Fin (L.V.k x) → Coin => ∀ ρ ∈ L.replaySample x r, ρ ∉ T) ≤ d.replay.Pa d.part T)
    (hsecond : ∀ (r : Fin (L.V.k x) → Coin) (X : Finset ℕ),
      Pr (fun e : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x (forkOn (L.stage2 x) r e), u ∉ X) ≤
        ∏ ρ ∈ (range d.part.blocks.length).filter (· ∈ L.replaySample x r),
          ∏ t ∈ range Pt.strata.length,
            escCell (cell Pt d.part ρ t).card (d.k.getD t 0) (X ∩ cell Pt d.part ρ t).card)
    (hslots : ∀ r', (L.checked x r').length ≤ L.nPU x)
    (hsess : ∀ r (sel : List ℕ → Option (Fin (L.nPU x))),
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        (run' (forkOn (L.stage2 x) r p.1)).out = .acc ∧
          ∃ i, sel (L.checked x (forkOn (L.stage2 x) r p.1)) = some i ∧
            ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.stage2 x) r p.1))[i.val]? = some u.val ∧
              ¬ found u (forkOn (L.stage2 x) r p.1) (p.2 i u)) ≤
        κr r) :
    Pr (fun r : Fin (L.V.k x) → Coin => (run' r).out = .acc) *
        Pr (fun c : (Fin (L.cap x) → Fin (L.V.k x) → Coin) × (Fin (L.k x) → L.E1 x) =>
          (accList (fun r => (run' r).out = .acc) (L.k x) ((List.finRange (L.cap x)).map c.1)).length < L.k x ∨
            ∃ Y ∈ S₁Two Pt d δ, AllBad (fun r => (run' r).out = .acc) (L.k x)
              (fun r e => L.BadOuterTwo x run' found (above Pt d Y) (forkOn (L.stage2 x) r) e)
              ((List.finRange (L.cap x)).map c.1) c.2) ≤
      2 * (L.k x : ℝ) / L.cap x + Real.exp (-(L.k x : ℝ) / 8) +
        slack * (2 * (L.k x : ℝ) / L.cap x + Real.exp (-(L.k x : ℝ) / 8) + slack * (δ + Ex κr)) := by
  have hc := slack_pos
  have hEκ : 0 ≤ Ex κr := Ex_nonneg hκr0
  have hy : ∀ Y ∈ S₁Two Pt d δ,
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.E1 x =>
        (run' p.1).out = .acc ∧ L.BadOuterTwo x run' found (above Pt d Y) (forkOn (L.stage2 x) p.1) p.2) ≤
      2 * (L.k x : ℝ) / L.cap x + Real.exp (-(L.k x : ℝ) / 8) + slack * (δ + Ex κr) := by
    intro Y hY
    obtain ⟨_, hYδ⟩ := mem_filter.1 hY
    let esc : (Fin (L.V.k x) → Coin) → ℝ := fun r =>
      ∏ ρ ∈ (range d.part.blocks.length).filter (· ∈ L.replaySample x r), d.esc Pt (counts Pt d.part Y) ρ
    have hesc0 : ∀ r, 0 ≤ esc r := fun r => prod_nonneg fun ρ _ => d.esc_nonneg Pt _ ρ
    have hesc : ∀ r, ∀ X ∈ above Pt d Y,
        Pr (fun e : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x (forkOn (L.stage2 x) r e), u ∉ X) ≤ esc r := by
      intro r X hX
      have hXY := (mem_filter.1 hX).2
      refine (hsecond r X).trans (prod_le_prod₀ (fun ρ _ => prod_nonneg fun t _ => Claims.Sampling.escCell_nonneg _ _ _)
        fun ρ hρ => prod_le_prod₀ (fun t _ => Claims.Sampling.escCell_nonneg _ _ _) fun t ht => ?_)
      exact escCell_anti _ _ (hXY ρ (mem_range.1 (mem_filter.1 hρ).1) t (mem_range.1 ht))
    have hXs : ∀ X ∈ above Pt d Y, ∀ v ∈ X, v < L.nPU x := fun X hX v hv => by
      rw [hPU]
      exact mem_range.1 (mem_powerset.1 (mem_filter.1 hX).1 hv)
    have hcardXs : (above Pt d Y).card ≤ 2 ^ L.nPU x :=
      (card_filter_le _ _).trans (by rw [card_powerset, card_range, hPU])
    have hEesc : Ex esc ≤ δ :=
      (Ex_esc_le d Pt (counts Pt d.part Y) (L.replaySample x) hreplay).trans hYδ
    refine (Pr_acc_forkOn (L.stage2 x) (fun r => (run' r).out = .acc)
      (fun r e => L.BadOuterTwo x run' found (above Pt d Y) (forkOn (L.stage2 x) r) e)
      fun r e' e => by rw [forkOn_forkOn']).trans_le ?_
    calc Ex (fun r => Pr (fun e' => (run' (forkOn (L.stage2 x) r e')).out = .acc) *
          Pr (fun e : L.E1 x => L.BadOuterTwo x run' found (above Pt d Y) (forkOn (L.stage2 x) r) e))
        ≤ Ex (fun r => 2 * (L.k x : ℝ) / L.cap x + Real.exp (-(L.k x : ℝ) / 8) + slack * (esc r + κr r)) :=
          Ex_mono fun r => L.inner_badTwo_le x run' found hXs hcardXs hslots r (hesc0 r) (hκr0 r) (hesc r)
            (hsess r)
      _ = 2 * (L.k x : ℝ) / L.cap x + Real.exp (-(L.k x : ℝ) / 8) + slack * (Ex esc + Ex κr) := by
          rw [Ex_add, Ex_const, Ex_mul_left, Ex_add]
      _ ≤ 2 * (L.k x : ℝ) / L.cap x + Real.exp (-(L.k x : ℝ) / 8) + slack * (δ + Ex κr) := by
          gcongr
  have hy0 : 0 ≤ 2 * (L.k x : ℝ) / L.cap x + Real.exp (-(L.k x : ℝ) / 8) + slack * (δ + Ex κr) := by
    positivity
  have hw := wait_bad_le (fun r => (run' r).out = .acc) (L.k x) (L.cap x) (L.cap_pos x) (L.k_pos x)
    (S₁Two Pt d δ) (fun Y r e => L.BadOuterTwo x run' found (above Pt d Y) (forkOn (L.stage2 x) r) e) hy0 hy
  have hcard : ((S₁Two Pt d δ).card : ℝ) ^ (1 / (L.k x : ℝ)) ≤ slack :=
    L.rpow_k_le_slack x (m := L.nPU x) ((card_filter_le _ _).trans (by rw [card_powerset, card_range, hPU]))
      (by omega)
  exact hw.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_right hcard hy0))

set_option synthInstance.maxSize 1024 in
/-- **The failure event's probability, at any found event**: `Compiled.pr_fails_le_of` at the two-stage failure event,
`κ(δ) + √(γ·Pr[B])`, with the outer wait's bound `outer_badTwo_le` from `Sessions.replayDraws` and
`Sessions.secondDraws`. -/
theorem pr_failsTwo_le_of (H : Msg → Msg) (L : CompiledTwo Coin A) {δ : ℝ} (hδ : 0 ≤ δ) (hS : L.Sessions H)
    (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (found : Ω → (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop)
    (β : Ω → (Fin (L.V.k x) → Coin) → ℝ) (hβ0 : ∀ ω r, 0 ≤ β ω r)
    (hβ : Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => β p.1 p.2) ≤ PrB H L.V P x)
    (hsess : ∀ ω r (sel : List ℕ → Option (Fin (L.nPU x))),
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        (run H P L.V x ω (forkOn (L.stage2 x) r p.1)).out = .acc ∧
          ∃ i, sel (L.checked x (forkOn (L.stage2 x) r p.1)) = some i ∧
            ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.stage2 x) r p.1))[i.val]? = some u.val ∧
              ¬ found ω u (forkOn (L.stage2 x) r p.1) (p.2 i u)) ≤
        L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r)) :
    Pr (fun s : Ω × (Fin (L.V.k x) → Coin) × L.ECoins x =>
      (run H P L.V x s.1 s.2.1).out = .acc ∧ L.FailsTwo δ x (run H P L.V x s.1) (found s.1) s.2.2) ≤
      L.κ δ x + Real.sqrt (L.γ x * PrB H L.V P x) := by
  refine (Pr_indep (fun ω r => (run H P L.V x ω r).out = .acc)
    (fun ω c => L.FailsTwo δ x (run H P L.V x ω) (found ω) c)).trans_le ?_
  have hc := slack_pos
  have hκ0 := L.κ_nonneg hδ x
  have hγ0 : 0 ≤ (L.V.k x : ℝ) * L.γs x := mul_nonneg (Nat.cast_nonneg _) (L.γs_nonneg x)
  by_cases hsc : ∃ s Pt d, L.parse x = some s ∧ L.partition s = some Pt ∧ L.two s = some d ∧ InScopeTwo Pt d s
  swap
  · have h0 : ∀ ω, Pr (fun c : L.ECoins x => L.FailsTwo δ x (run H P L.V x ω) (found ω) c) = 0 := fun ω =>
      (Pr_congr fun c => ⟨fun ⟨s, Pt, d, h1, h2, h3, h4, _⟩ => hsc ⟨s, Pt, d, h1, h2, h3, h4⟩, False.elim⟩).trans
        Pr_false
    simp only [h0, mul_zero, Ex_const]
    exact add_nonneg hκ0 (Real.sqrt_nonneg _)
  obtain ⟨s, Pt, d, hs, hPt, hd, hin⟩ := hsc
  have hper : ∀ ω, Pr (fun r => (run H P L.V x ω r).out = .acc) *
      Pr (fun c : L.ECoins x => L.FailsTwo δ x (run H P L.V x ω) (found ω) c) ≤
      L.κ δ x + slack ^ 2 * Ex (fun r => Real.sqrt (L.V.k x * L.γs x * β ω r)) := by
    intro ω
    have hF : Pr (fun c : L.ECoins x => L.FailsTwo δ x (run H P L.V x ω) (found ω) c) =
        Pr (fun c : (Fin (L.cap x) → Fin (L.V.k x) → Coin) × (Fin (L.k x) → L.E1 x) =>
          (accList (fun r => (run H P L.V x ω r).out = .acc) (L.k x)
            ((List.finRange (L.cap x)).map c.1)).length < L.k x ∨
            ∃ Y ∈ S₁Two Pt d δ, AllBad (fun r => (run H P L.V x ω r).out = .acc) (L.k x)
              (fun r e => L.BadOuterTwo x (run H P L.V x ω) (found ω) (above Pt d Y) (forkOn (L.stage2 x) r) e)
              ((List.finRange (L.cap x)).map c.1) c.2) := by
      refine Eq.trans ?_ (Claims.Sampling.Pr_equiv (L.ecoinsEquiv x) _)
      refine Pr_congr fun c => ⟨fun ⟨s', Pt', d', hs', hPt', hd', _, h⟩ => ?_,
        fun h => ⟨s, Pt, d, hs, hPt, hd, hin, h⟩⟩
      rw [hs, Option.some.injEq] at hs'
      subst hs'
      rw [hPt, Option.some.injEq] at hPt'
      subst hPt'
      rw [hd, Option.some.injEq] at hd'
      subst hd'
      exact h
    rw [hF]
    refine (L.outer_badTwo_le hδ x (run H P L.V x ω) (found ω) Pt d (Compiled.nPU_eq hs hPt)
      (fun r => L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r))
      (fun r => add_nonneg (L.κs_nonneg x) (Real.sqrt_nonneg _))
      (hS.replayDraws x s Pt d hs hPt hd hin) (hS.secondDraws x s Pt d hs hPt hd hin) (hS.slots x)
      (hsess ω)).trans (le_of_eq ?_)
    rw [Ex_add, Ex_const]
    unfold Compiled.κ
    ring
  calc Ex (fun ω => Pr (fun r => (run H P L.V x ω r).out = .acc) *
        Pr (fun c : L.ECoins x => L.FailsTwo δ x (run H P L.V x ω) (found ω) c))
      ≤ Ex (fun ω => L.κ δ x + slack ^ 2 * Ex (fun r => Real.sqrt (L.V.k x * L.γs x * β ω r))) := Ex_mono hper
    _ = L.κ δ x + slack ^ 2 *
          Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => Real.sqrt (L.V.k x * L.γs x * β p.1 p.2)) := by
        rw [Ex_add, Ex_const, Ex_mul_left,
          Ex_prod (fun p : Ω × (Fin (L.V.k x) → Coin) => Real.sqrt (L.V.k x * L.γs x * β p.1 p.2))]
    _ ≤ L.κ δ x + slack ^ 2 * Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) := by
        refine add_le_add le_rfl (mul_le_mul_of_nonneg_left ?_ (by positivity))
        refine (Ex_sqrt_le _ fun p => mul_nonneg hγ0 (hβ0 _ _)).trans (Real.sqrt_le_sqrt ?_)
        rw [Ex_mul_left]
        exact mul_le_mul_of_nonneg_left hβ hγ0
    _ = L.κ δ x + Real.sqrt (L.γ x * PrB H L.V P x) := by
        rw [show L.γ x * PrB H L.V P x = (slack ^ 2) ^ 2 * (L.V.k x * L.γs x * PrB H L.V P x) by
            unfold Compiled.γ; ring,
          Real.sqrt_mul (pow_nonneg (pow_nonneg hc.le 2) 2) (L.V.k x * L.γs x * PrB H L.V P x),
          Real.sqrt_sq (pow_nonneg hc.le 2)]

set_option synthInstance.maxSize 1024 in
/-- **`CompiledTwo.KS` from the split and the sessions' bound** (`Compiled.KS_of_split` at the two-stage failure
event and `RTwo`). -/
theorem KS_of_splitTwo {H : Msg → Msg} {L : CompiledTwo Coin A} (hS : L.Sessions H)
    (found : ∀ (x : Msg) (Ω : Type), Prover Ω → Ω → (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) →
      A.EC (L.sh x u) → Prop)
    (hsplit : ∀ δ : ℝ, 0 ≤ δ → ∀ x (Ω : Type) (P : Prover Ω) (s : Ω × (Fin (L.V.k x) → Coin) × L.ECoins x),
      (run H P L.V x s.1 s.2.1).out = .acc →
        ¬ (RTwo H L.commit L.parse L.partition L.two δ).ext x (L.E.eval H P x s.1 s.2.1 s.2.2).1 →
        L.FailsTwo δ x (run H P L.V x s.1) (found x Ω P s.1) s.2.2 ∨ OwnColl H L.E P x s.1 s.2.1 s.2.2)
    (hsess : ∀ x (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      ∃ β : Ω → (Fin (L.V.k x) → Coin) → ℝ, (∀ ω r, 0 ≤ β ω r) ∧
        Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => β p.1 p.2) ≤ PrB H L.V P x ∧
        ∀ ω r (sel : List ℕ → Option (Fin (L.nPU x))),
          Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
            (run H P L.V x ω (forkOn (L.stage2 x) r p.1)).out = .acc ∧
              ∃ i, sel (L.checked x (forkOn (L.stage2 x) r p.1)) = some i ∧
                ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.stage2 x) r p.1))[i.val]? = some u.val ∧
                  ¬ found x Ω P ω u (forkOn (L.stage2 x) r p.1) (p.2 i u)) ≤
            L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r)) :
    L.KS H := by
  intro δ hδ x _
  exact Sampling.KSLin_of_ownColl H L.V (RTwo H L.commit L.parse L.partition L.two δ) L.E (L.κ δ) L.γ
    (fun x Ω P s => (run H P L.V x s.1 s.2.1).out = .acc ∧ L.FailsTwo δ x (run H P L.V x s.1) (found x Ω P s.1) s.2.2)
    (fun x Ω P s hacc hfail => (hsplit δ hδ x Ω P s hacc hfail).imp_left fun h => ⟨hacc, h⟩)
    (fun x Ω _ _ P => by
      obtain ⟨β, hβ0, hβ, hκ⟩ := hsess x Ω P
      exact L.pr_failsTwo_le_of H hδ hS x Ω P (found x Ω P) β hβ0 hβ hκ) x

/-! ## The split -/

/-- **A set of proof blocks with given counts per cell**, at counts within the cells' sizes: one subset of each cell of
the grid, of the count's size, their union (the cells are disjoint). -/
theorem exists_counts (Pt Pρ : Partition) (nR nS : ℕ) (f : ℕ → ℕ → ℕ)
    (hf : ∀ ρ < nR, ∀ t < nS, f ρ t ≤ (cell Pt Pρ ρ t).card) :
    ∃ Y ⊆ range Pt.blocks.length, ∀ ρ < nR, ∀ t < nS, counts Pt Pρ Y ρ t = f ρ t := by
  classical
  have hZ : ∀ p : ℕ × ℕ, ∃ Z ⊆ cell Pt Pρ p.1 p.2, (p.1 < nR ∧ p.2 < nS → Z.card = f p.1 p.2) := by
    intro p
    by_cases hp : p.1 < nR ∧ p.2 < nS
    · obtain ⟨Z, hZ, hc⟩ := exists_subset_card_eq (hf p.1 hp.1 p.2 hp.2)
      exact ⟨Z, hZ, fun _ => hc⟩
    · exact ⟨∅, empty_subset _, fun h => absurd h hp⟩
  choose Z hZsub hZcard using hZ
  refine ⟨(range nR ×ˢ range nS).biUnion Z, fun v hv => ?_, fun ρ hρ t ht => ?_⟩
  · obtain ⟨p, _, hvp⟩ := mem_biUnion.1 hv
    exact cell_subset Pt Pρ p.1 p.2 (hZsub p hvp)
  · unfold counts
    have e : (range nR ×ˢ range nS).biUnion Z ∩ cell Pt Pρ ρ t = Z (ρ, t) := by
      ext v
      rw [mem_inter, mem_biUnion]
      constructor
      · rintro ⟨⟨p, _, hvp⟩, hvc⟩
        obtain ⟨h1, h2⟩ := cell_disj (hZsub p hvp) hvc
        have hp : p = (ρ, t) := Prod.ext h1 h2
        rw [← hp]
        exact hvp
      · intro hv
        exact ⟨⟨(ρ, t), mem_product.2 ⟨mem_range.2 hρ, mem_range.2 ht⟩, hv⟩, hZsub _ hv⟩
    rw [e, hZcard _ ⟨hρ, ht⟩]

/-- **The split**: at an accepted run whose extraction fails, the failure event holds or the extractor's own calls
collide. Otherwise the kept openings agree (`agree_of_passes`) and assemble a witness `w` whose footprint is refused at
`δ`; take `Y` with its counts (`exists_counts`) and a kept outer run `r`. In each cell the proof blocks incorrect at the
interior values the kept openings under `r`'s interior roots of their replay block give are at least the footprint
(`footprint_le`), so their union is at least as large as `Y`; a kept inner run at which one is found would make it
correct (`Sessions.opens`, its revealed openings kept by `Compiled.mem_found`, the interior roots `r`'s by
`Sessions.interior`), so `r` is bad for it. -/
theorem splitTwo {H : Msg → Msg} {L : CompiledTwo Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H) (δ : ℝ)
    (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω) (r₀ : Fin (L.V.k x) → Coin) (c : L.ECoins x)
    (hacc : (run H P L.V x ω r₀).out = .acc)
    (hfail : ¬ (RTwo H L.commit L.parse L.partition L.two δ).ext x (L.E.eval H P x ω r₀ c).1) :
    L.FailsTwo δ x (run H P L.V x ω) (L.Found H x (run H P L.V x ω)) c ∨ OwnColl H L.E P x ω r₀ c := by
  classical
  obtain ⟨s, Pt, d, hs, hPt, hd, hin⟩ := hS.refuses x Ω P ω r₀ hacc
  by_cases hcoll : OwnColl H L.E P x ω r₀ c
  · exact Or.inr hcoll
  refine Or.inl ⟨s, Pt, d, hs, hPt, hd, hin, ?_⟩
  set run' := run H P L.V x ω with hrun'
  set fd := L.found H x s run' c (L.outerList H x run' c) with hfd
  have hpass : ∀ q ∈ fd, L.commit.Opens H q.1 q.2 = true ∧
      ∀ t ∈ ((L.commit.opens q.1 q.2).eval H).2, t ∈ Claims.Sampling.ownLog H run' (L.E.ext x r₀ c) := by
    intro q hq
    rw [hfd, Compiled.found_eq] at hq
    obtain ⟨h1, h2⟩ := L.passes_found H x run' s c _ q hq
    exact ⟨h1, fun t ht => L.ownLog_E H P x ω r₀ c hs hacc t (h2 t ht)⟩
  have hagree : ∀ q ∈ fd, ∀ q' ∈ fd, q.1 = q'.1 → q.2.pos = q'.2.pos → q.2.val = q'.2.val :=
    fun q hq q' hq' h1 h2 => agree_of_passes hbind hpass hcoll hq hq' h1 h2
  rw [L.E_eval H P x ω r₀ c hs hacc] at hfail
  set w := assemble s fd with hw
  have hPa : d.Pa Pt (footprint H L.commit s Pt d.part w) ≤ δ := by
    by_contra hlt
    exact hfail ⟨s, Pt, d, hs, hPt, hd, lt_of_not_ge hlt⟩
  obtain ⟨Y, hYsub, hY⟩ := exists_counts Pt d.part d.part.blocks.length Pt.strata.length
    (footprint H L.commit s Pt d.part w) fun ρ _ t _ => footprint_le_cell H L.commit s Pt d.part w ρ t
  have hYS : Y ∈ S₁Two Pt d δ := by
    refine mem_filter.2 ⟨mem_powerset.2 hYsub, ?_⟩
    rw [d.Pa_congr Pt hY]
    exact hPa
  by_cases hshort : (accList (fun r => (run' r).out = .acc) (L.k x) ((List.finRange (L.cap x)).map c.1)).length <
    L.k x
  · exact Or.inl hshort
  refine Or.inr ⟨Y, hYS, accList_length_eq hshort, fun j r hj => ?_⟩
  have hjZ := L.mem_outerList H x run' c hj
  by_cases hshort2 : (accList (fun d => (run' (forkOn (L.stage2 x) r d)).out = .acc) (L.k x)
    ((List.finRange (L.cap x)).map (c.2.1 j))).length < L.k x
  · exact Or.inl hshort2
  let ιr : ℕ → ℕ → Bool := fun ρ => ιOf fd (L.iroots x (run' r).seen ρ)
  let Xr : Finset ℕ := (range d.part.blocks.length ×ˢ range Pt.strata.length).biUnion fun p =>
    wrongIn H L.commit s Pt d.part w p.1 p.2 (ιr p.1)
  have hXr : Xr ∈ above Pt d Y := by
    refine mem_filter.2 ⟨mem_powerset.2 fun v hv => ?_, fun ρ hρ t ht => ?_⟩
    · obtain ⟨p, _, hvp⟩ := mem_biUnion.1 hv
      exact cell_subset Pt d.part p.1 p.2 ((filter_subset _ _) hvp)
    · rw [hY ρ hρ t ht]
      refine (footprint_le H L.commit s Pt d.part w ρ t (ιr ρ)).trans (card_le_card fun v hv => ?_)
      exact mem_inter.2 ⟨mem_biUnion.2 ⟨(ρ, t), mem_product.2 ⟨mem_range.2 hρ, mem_range.2 ht⟩, hv⟩,
        (filter_subset _ _) hv⟩
  refine Or.inr ⟨Xr, hXr, accList_length_eq hshort2, fun l e hl v hv hgood => ?_⟩
  obtain ⟨⟨ρ', t'⟩, _, hvp⟩ := mem_biUnion.1 hv
  obtain ⟨hvcell, hvwrong⟩ := mem_filter.1 hvp
  obtain ⟨i, hvlt, hvi, xS, vS, hsess, hsh, wS, hwS, hRw⟩ := hgood
  have hvc := List.mem_of_getElem? hvi
  have hacc' : (run' (forkOn (L.stage2 x) r e)).out = .acc := by
    have h := acc_of_accList hl
    exact h
  obtain ⟨b, ρ, rb, hb, hρ, hrb, hcor⟩ :=
    hS.opens x s Pt d Ω P ω (forkOn (L.stage2 x) r e) v xS vS wS hs hPt hd hin hacc' hvc hsess hRw
  have hρp : ρ = ρ' := Option.some.inj (hρ.symm.trans (replayOf_of_mem_cell hvcell))
  subst hρp
  have hcorr := hcor (w.val H L.commit s) (ιr ρ) fun q hq hok => by
    have hqf : q ∈ fd := L.mem_found H x s run' c hjZ hl hvi hvlt
      (L.unitExt_mem H x run' s v _ _ hsess hsh hwS hq hok)
    refine ⟨fun hr => ?_, fun hi => ?_⟩
    · rw [hr] at hok
      exact val_assemble hagree hqf hr hok
    · rw [hS.interior x Ω P ω r e ρ] at hi
      show ιOf fd (L.iroots x (run' r).seen ρ) q.2.pos = q.2.val
      rw [← hi]
      exact ιOf_eq hagree hqf
  exact hvwrong ⟨b, rb, hb, hrb, hcorr⟩

end CompiledTwo

/-- **`CompilerKSTwo`**: the sessions' bound at `Compiled.Found` is `Compiled.sessionKS_of` at the contract's `fixed`
and `runs`, and the split is `splitTwo`. -/
theorem compilerKSTwo : CompilerKSTwo := by
  intro Coin _ _ _ A L H hA hbind hS
  have hSK : L.toCompiled.SessionKS H := Compiled.sessionKS_of hA ⟨hS.fixed, hS.runs⟩
  exact CompiledTwo.KS_of_splitTwo hS (fun x _ P ω => L.Found H x (run H P L.V x ω))
    (fun δ _ x _ P s hacc hfail => CompiledTwo.splitTwo hbind hS δ x P s.1 s.2.1 s.2.2 hacc hfail)
    (fun x Ω _ _ P => hSK x Ω P)

end Claims.SamplingGeneric
