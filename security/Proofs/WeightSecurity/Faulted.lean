module

public import Definitions.WeightSecurity.Guarantees
public import Proofs.Spot.Necessity
public import Proofs.WeightSecurity.Served

/-! `WeightSecurity.SecurityProofs.TheftFaulted : WeightSecurity.Guarantees.TheftFaulted`, `TheftServed` with faults as
inputs to the served circuit. On a run whose accepted releases are faulted runs and whose audit leaves fewer than `M`
incorrect proof blocks undetected, the outsider's view is its view against the datacenter that deviates through the
fault channel with the run's own covert string (`Served.view_eq_faultResp`): the patterns of the windows accepted before
the first rejection, or the cut's hint in place of a pattern it cuts, whose sizes sum to at most `lead < M`; each
window's advice and bucket; the values the windows from the first rejection to the halt released, at most
`(L + 1) · |O|` (`Served.card_tailOverwrite_le`); and the halt. That deviation is honest use of the faulted function by
the outsider with the empty pattern beside each query (`Outsider.withFaults_model`), and that party with each covert
string written in sets at most `M` faults (`withString_mem_faultBudget`), all at patterns the cut doesn't cut
(`Cut.Interiors`). So the outsider steals a typical weight only where the run leaves `M` undetected, a release isn't
faithful, or some faulted-use party steals it, which `FaultedUseHard` at the interior patterns bounds by
`N^(λ - 1) · p` each, `N` counting only them.

The fault model's facts about populations (`Population.Dataflow`). In a checked population the run with some units'
writes fixed is unique (`runOf_unique`, from `determinate_of_checked`), and any assignment is the run with its own
incorrect units' writes fixed (`runOf_faultsOf`), whose pattern names exactly its incorrect units (`faultSet_faultsOf`).
So a release the network certifier ties to a trace is a faulted run of the population's faulted function
(`Served.faithful_of_tie`). When every unit reads each slot through an idempotent clamp fixed in advance, the run with a
pattern differs from the run with the pattern clamped only on the faulted slots, each by its clamp (`runOf_clamp`). With
the clamp the identity on the egress, every pattern releases what an in-range one does (`runOf_clamp_egress`): the
faulted function at a pattern is the faulted function at its clamp, which names the same units (`faultedOut_clamp`), so
`FaultedUseHard` need cover only in-range faults.

The witnesses are proved at the toy deployment with faults. -/

@[expose] public section

namespace Population.Dataflow

open Finset

variable {U S : Type} (P : Dataflow U S) {V : Type}

/-- **A run with faults**: `σ` holds the inputs `a`, each unit the pattern `φ` names writes `φ`'s values, and every
other unit is correct. -/
def RunOf (f : P.Fns V) (a : S → V) (φ : (u : U) → Option (P.writes u → V)) (σ : S → V) : Prop :=
  (∀ s ∈ P.input, σ s = a s) ∧ (∀ u w, φ u = some w → ∀ t : P.writes u, σ t = w t) ∧
    ∀ u, φ u = none → P.Correct f σ u

/-- The unit functions with each unit the pattern `φ` names writing `φ`'s values whatever it reads. -/
def fixFns (f : P.Fns V) (φ : (u : U) → Option (P.writes u → V)) : P.Fns V :=
  fun u x => (φ u).getD (f u x)

theorem correct_fixFns_iff (f : P.Fns V) (φ : (u : U) → Option (P.writes u → V)) (σ : S → V) (u : U) :
    P.Correct (P.fixFns f φ) σ u ↔
      (∀ w, φ u = some w → ∀ t : P.writes u, σ t = w t) ∧ (φ u = none → P.Correct f σ u) := by
  unfold Correct fixFns
  cases φ u with
  | none => simp
  | some w => simp

theorem runOf_iff (f : P.Fns V) (a : S → V) (φ : (u : U) → Option (P.writes u → V)) (σ : S → V) :
    P.RunOf f a φ σ ↔ (∀ s ∈ P.input, σ s = a s) ∧ ∀ u, P.Correct (P.fixFns f φ) σ u := by
  simp only [RunOf, correct_fixFns_iff, forall_and]

/-- **A checked population has one run per inputs and fault pattern**: `determinate_of_checked`, with each unit the
pattern names made to write what the pattern says. -/
theorem runOf_unique [Finite U] (h : P.Checked) {f : P.Fns V} {a : S → V} {φ : (u : U) → Option (P.writes u → V)}
    {σ τ : S → V} (hσ : P.RunOf f a φ σ) (hτ : P.RunOf f a φ τ) : σ = τ :=
  (P.determinate_of_checked h V (P.fixFns f φ) a).unique ((P.runOf_iff f a φ σ).1 hσ) ((P.runOf_iff f a φ τ).1 hτ)

theorem runOf_exists [Finite U] (h : P.Checked) (f : P.Fns V) (a : S → V) (φ : (u : U) → Option (P.writes u → V)) :
    ∃ σ, P.RunOf f a φ σ := by
  obtain ⟨σ, hσ, -⟩ := P.determinate_of_checked h V (P.fixFns f φ) a
  exact ⟨σ, (P.runOf_iff f a φ σ).2 hσ⟩

open Classical in
/-- **An assignment's fault pattern**: what each incorrect unit wrote, and nothing at a correct one. -/
noncomputable def faultsOf (f : P.Fns V) (σ : S → V) : (u : U) → Option (P.writes u → V) :=
  fun u => if P.Correct f σ u then none else some fun t => σ t

/-- **Every assignment is the run with its own incorrect units' writes fixed**, at its own inputs. -/
theorem runOf_faultsOf (f : P.Fns V) (σ : S → V) : P.RunOf f σ (P.faultsOf f σ) σ := by
  refine ⟨fun _ _ => rfl, fun u w hw t => ?_, fun u hu => ?_⟩
  · unfold faultsOf at hw
    split_ifs at hw <;> cases hw
    rfl
  · unfold faultsOf at hu
    split_ifs at hu with hc
    exact hc

open Classical in
/-- The units a fault pattern names. -/
noncomputable def faultSet [Fintype U] (φ : (u : U) → Option (P.writes u → V)) : Finset U :=
  univ.filter fun u => (φ u).isSome

/-- **An assignment's fault pattern names exactly its incorrect units.** -/
theorem faultSet_faultsOf [Fintype U] (f : P.Fns V) (σ : S → V) : P.faultSet (P.faultsOf f σ) = P.incorrect f σ := by
  ext u
  have key : (P.faultsOf f σ u).isSome = true ↔ ¬ P.Correct f σ u := by
    simp only [faultsOf]
    split_ifs with hc <;> simp [hc]
  simpa [faultSet, incorrect] using key

/-- **A checked population's faulted function**: the values on the slots `eg` of its run at the inputs `a` with the
fault pattern `φ`. -/
noncomputable def faultedOut [Finite U] (h : P.Checked) (f : P.Fns V) {O : Type} (eg : O → S) (a : S → V)
    (φ : (u : U) → Option (P.writes u → V)) : O → V :=
  fun o => Classical.choose (P.runOf_exists h f a φ) (eg o)

theorem faultedOut_eq [Finite U] (h : P.Checked) (f : P.Fns V) {O : Type} (eg : O → S) {a : S → V}
    {φ : (u : U) → Option (P.writes u → V)} {σ : S → V} (hσ : P.RunOf f a φ σ) :
    P.faultedOut h f eg a φ = fun o => σ (eg o) := by
  funext o
  unfold faultedOut
  rw [P.runOf_unique h (Classical.choose_spec (P.runOf_exists h f a φ)) hσ]

/-- **Readers that clamp**: the unit functions `f` with every read slot `t` read through `κ t`. -/
def clampFns (f : P.Fns V) (κ : S → V → V) : P.Fns V := fun u x => f u fun t => κ t (x t)

/-- A fault pattern clamped: each faulted unit's value `v` at its slot `t` replaced by `κ t v`. -/
def clampFaults (κ : S → V → V) (φ : (u : U) → Option (P.writes u → V)) : (u : U) → Option (P.writes u → V) :=
  fun u => (φ u).map fun w t => κ t (w t)

theorem faultSet_clampFaults [Fintype U] (κ : S → V → V) (φ : (u : U) → Option (P.writes u → V)) :
    P.faultSet (P.clampFaults κ φ) = P.faultSet φ := by
  ext u
  simp [faultSet, clampFaults]

open Classical in
/-- **Clamped readers make every fault pattern an in-range one.** In a checked population whose units read every slot
`t` through an idempotent clamp `κ t` fixed before the run, the run with the pattern `φ` and the run with `φ` clamped
agree on every slot no faulted unit writes, and on a faulted unit's slot the second holds the first's value clamped. -/
theorem runOf_clamp [Finite U] (h : P.Checked) (f : P.Fns V) (κ : S → V → V) (hκ : ∀ t v, κ t (κ t v) = κ t v)
    {a : S → V} {φ : (u : U) → Option (P.writes u → V)} {σ τ : S → V}
    (hσ : P.RunOf (P.clampFns f κ) a φ σ) (hτ : P.RunOf (P.clampFns f κ) a (P.clampFaults κ φ) τ) :
    τ = fun t => if ∃ u, (φ u).isSome ∧ t ∈ P.writes u then κ t (σ t) else σ t := by
  obtain ⟨ha, hφ, hc⟩ := hσ
  have hone : ∀ u, ∀ t ∈ P.writes u, φ u = none → ¬ ∃ u', (φ u').isSome ∧ t ∈ P.writes u' := by
    rintro u t ht hu ⟨u', hu', ht'⟩
    obtain ⟨v, -, hv⟩ := (h.1 t).2 fun hi => (h.1 t).1 hi u ht
    rw [hv u ht, ← hv u' ht'] at hu
    rw [hu] at hu'
    simp at hu'
  refine P.runOf_unique h hτ ⟨fun s hs => ?_, fun u w hw t => ?_, fun u hu t => ?_⟩
  · have hno : ¬ ∃ u, (φ u).isSome ∧ s ∈ P.writes u := fun ⟨u, _, hsu⟩ => (h.1 s).1 hs u hsu
    simp only [hno, ↓reduceIte]
    exact ha s hs
  · obtain ⟨w₀, hw₀, rfl⟩ := Option.map_eq_some_iff.1 hw
    have hex : ∃ u', (φ u').isSome ∧ (t : S) ∈ P.writes u' := ⟨u, by simp [hw₀], t.2⟩
    simp only [hex, ↓reduceIte]
    rw [hφ u w₀ hw₀ t]
  · have hu' : φ u = none := by simpa [clampFaults] using hu
    have hno := hone u t t.2 hu'
    simp only [hno, ↓reduceIte]
    refine (hc u hu' t).trans (congrArg (fun x => f u x t) (funext fun r => ?_))
    by_cases hr : ∃ u', (φ u').isSome ∧ (r : S) ∈ P.writes u'
    · simp only [hr, ↓reduceIte, hκ]
    · simp only [hr, ↓reduceIte]

/-- **With the clamp the identity on a slot, every fault pattern releases there what its clamp does.** -/
theorem runOf_clamp_egress [Finite U] (h : P.Checked) (f : P.Fns V) (κ : S → V → V) (hκ : ∀ t v, κ t (κ t v) = κ t v)
    {a : S → V} {φ : (u : U) → Option (P.writes u → V)} {σ τ : S → V}
    (hσ : P.RunOf (P.clampFns f κ) a φ σ) (hτ : P.RunOf (P.clampFns f κ) a (P.clampFaults κ φ) τ) {t : S}
    (ht : ∀ v, κ t v = v) : τ t = σ t := by
  classical
  rw [P.runOf_clamp h f κ hκ hσ hτ]
  dsimp only
  split_ifs <;> simp [ht]

/-- **The reduction to in-range faults.** In a checked population whose units read through the clamp `κ`, with `κ` the
identity on the egress slots, the faulted function at any fault pattern is the faulted function at its clamp, which
names the same units (`faultSet_clampFaults`) and writes only clamped values: a model claim about the faulted circuit
need only cover in-range faults. -/
theorem faultedOut_clamp [Finite U] (h : P.Checked) (f : P.Fns V) (κ : S → V → V) (hκ : ∀ t v, κ t (κ t v) = κ t v)
    {O : Type} (eg : O → S) (heg : ∀ o v, κ (eg o) v = v) (a : S → V) (φ : (u : U) → Option (P.writes u → V)) :
    P.faultedOut h (P.clampFns f κ) eg a φ = P.faultedOut h (P.clampFns f κ) eg a (P.clampFaults κ φ) := by
  obtain ⟨σ, hσ⟩ := P.runOf_exists h (P.clampFns f κ) a φ
  obtain ⟨τ, hτ⟩ := P.runOf_exists h (P.clampFns f κ) a (P.clampFaults κ φ)
  rw [P.faultedOut_eq h _ eg hσ, P.faultedOut_eq h _ eg hτ]
  funext o
  exact (P.runOf_clamp_egress h f κ hκ hσ hτ (heg o)).symm

end Population.Dataflow

namespace WeightSecurity

open Finset

section Lift

variable {R X M Θ' F : Type}

/-- A view with the pattern `φ₀` beside each query. -/
def liftView (φ₀ : F) (h : List (X × M)) : List ((X × F) × M) := h.map fun e => ((e.1, φ₀), e.2)

theorem unlift_liftView (φ₀ : F) (h : List (X × M)) : (liftView φ₀ h).map (fun e => (e.1.1, e.2)) = h := by
  simp [liftView, Function.comp_def]

/-- **The outsider asking the faulted function sees the outsider's view, lifted**, against the responder that answers
each query with the empty pattern beside it. -/
theorem Outsider.withFaults_view (O : Outsider R X M Θ') (φ₀ : F) (r : R) (resp : List ((X × F) × M) → X × F → M) :
    ∀ n, (O.withFaults φ₀).view r resp n = liftView φ₀ (O.view r (fun h x => resp (liftView φ₀ h) (x, φ₀)) n)
  | 0 => rfl
  | n + 1 => by
    have ih := Outsider.withFaults_view O φ₀ r resp n
    have hq : ∀ V, (O.withFaults φ₀).query r (liftView φ₀ V) = (O.query r V, φ₀) := fun V => by
      show (O.query r ((liftView φ₀ V).map fun e => (e.1.1, e.2)), φ₀) = _
      rw [unlift_liftView]
    show ((O.withFaults φ₀).query r ((O.withFaults φ₀).view r resp n),
        resp ((O.withFaults φ₀).view r resp n) ((O.withFaults φ₀).query r ((O.withFaults φ₀).view r resp n))) ::
        (O.withFaults φ₀).view r resp n = _
    rw [ih, hq]
    rfl

/-- **The outsider asking the faulted function outputs the outsider's model.** -/
theorem Outsider.withFaults_model (O : Outsider R X M Θ') (φ₀ : F) (r : R) (resp : List ((X × F) × M) → X × F → M)
    (q : ℕ) : (O.withFaults φ₀).model r resp q = O.model r (fun h x => resp (liftView φ₀ h) (x, φ₀)) q := by
  show O.out r (((O.withFaults φ₀).view r resp q).map fun e => (e.1.1, e.2)) = O.out r (O.view r _ q)
  rw [Outsider.withFaults_view, unlift_liftView]

theorem Outsider.rebuild_length {Y : Type} (O : Outsider R X M Θ') (ch : Channel R X Y M) (s : ch.S) (r : R) :
    ∀ h : List (X × Y), (O.rebuild ch s r h).length = h.length
  | [] => rfl
  | (_, _) :: h => by simp [Outsider.rebuild, Outsider.rebuild_length O ch s r h]

end Lift

section Channel

variable {R Q Adv O V Tick F Θ' : Type} [Fintype Adv] [Fintype O] [DecidableEq O] [Fintype V] [Fintype Tick]
  [Fintype F]

open WeightSecurity.Assumptions

open Classical in
/-- **The fault channel's covert strings**: each window's pattern or hint, then the served channel's strings. -/
theorem card_faultChannel (κ : Cut (Q × Adv) (O → V)) (size : F → ℕ) (φ₀ : F) (T M K : ℕ) :
    Fintype.card (faultChannel R Q Adv O V Tick F κ size φ₀ T M K).S =
      Fintype.card {w : Fin T → F ⊕ κ.H // ∑ t, Sum.elim size κ.size (w t) ≤ M} *
        Fintype.card (servedChannel R Q Adv O V Tick T K).S := by
  classical
  calc Fintype.card (faultChannel R Q Adv O V Tick F κ size φ₀ T M K).S
      = Fintype.card ({w : Fin T → F ⊕ κ.H // ∑ t, Sum.elim size κ.size (w t) ≤ M} ×
          (servedChannel R Q Adv O V Tick T K).S) :=
        Fintype.card_congr' rfl
    _ = _ := Fintype.card_prod _ _

/-- The fault channel's query at a history of length `n < T` carries its string's `n`-th pattern, or the empty pattern
at a hint. -/
theorem faultChannel_ask_snd (κ : Cut (Q × Adv) (O → V)) (size : F → ℕ) (φ₀ : F) (T M K : ℕ)
    (s : (faultChannel R Q Adv O V Tick F κ size φ₀ T M K).S) (r : R)
    (hist : List (((Q × Adv) × F) × Option ((O → V) × Tick))) (x : (Q × Adv) × F) :
    ((faultChannel R Q Adv O V Tick F κ size φ₀ T M K).ask s r hist x).2 =
      if hn : hist.length < T then Sum.elim id (fun _ => φ₀) (s.1.val ⟨hist.length, hn⟩) else x.2 := by
  simp only [faultChannel]
  split_ifs with hn
  · cases s.1.val ⟨hist.length, hn⟩ <;> rfl
  · rfl

/-- **The outsider with a fault channel string written in sets at most `M` faults**: its `n`-th query carries the
string's `n`-th pattern, or the empty pattern at a hint, and their sizes sum to at most the string's. -/
theorem withString_mem_faultBudget (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ')
    (κ : Cut (Q × Adv) (O → V)) (size : F → ℕ) (φ₀ : F) (hφ₀ : size φ₀ = 0) (T M K : ℕ)
    (s : (faultChannel R Q Adv O V Tick F κ size φ₀ T M K).S) :
    (Ou.withFaults φ₀).withString (faultChannel R Q Adv O V Tick F κ size φ₀ T M K) s ∈ FaultBudget size T M := by
  have hq : ∀ r h, (((Ou.withFaults φ₀).withString (faultChannel R Q Adv O V Tick F κ size φ₀ T M K) s).query r
      h).2 = if hn : h.length < T then Sum.elim id (fun _ => φ₀) (s.1.val ⟨h.length, hn⟩) else φ₀ := by
    intro r h
    simp only [Outsider.withString]
    rw [faultChannel_ask_snd, Outsider.rebuild_length]
    rfl
  refine fun r resp => ?_
  have key : ∀ n, ((((Ou.withFaults φ₀).withString (faultChannel R Q Adv O V Tick F κ size φ₀ T M K) s).view r
      resp n).map fun e => size e.1.2).sum =
        ∑ i ∈ range n, if hi : i < T then size (Sum.elim id (fun _ => φ₀) (s.1.val ⟨i, hi⟩)) else 0 := by
    intro n
    induction n with
    | zero => rfl
    | succ n ih =>
      rw [sum_range_succ, ← ih]
      simp only [Outsider.view, List.map_cons, List.sum_cons, hq, Outsider.view_length]
      split_ifs
      · exact Nat.add_comm _ _
      · simp [hφ₀]
  show _ ≤ M
  rw [key T]
  calc ∑ i ∈ range T, (if hi : i < T then size (Sum.elim id (fun _ => φ₀) (s.1.val ⟨i, hi⟩)) else 0)
      = ∑ t : Fin T, size (Sum.elim id (fun _ => φ₀) (s.1.val t)) := by
        rw [← Fin.sum_univ_eq_sum_range]
        exact sum_congr rfl fun t _ => dif_pos t.2
    _ ≤ ∑ t : Fin T, Sum.elim size κ.size (s.1.val t) := sum_le_sum fun t _ => by
        cases s.1.val t <;> simp [hφ₀]
    _ ≤ M := s.1.2

end Channel

namespace Served

variable {O V Tick Adv R Q Θ Θ' F : Type}

/-- The values a run released from the first rejection through the halt, by window and slot: the fault channel's
overwrite. -/
noncomputable def tailOverwrite (L : ℕ) (l : List (Served O V Tick Adv)) (T : ℕ) : Fin T × O → Option V :=
  fun p => if accLen l ≤ p.1.val ∧ p.1.val < count L l then l[p.1.val]?.map (·.egress p.2) else none

theorem card_tailOverwrite_le [Fintype O] (L : ℕ) (l : List (Served O V Tick Adv)) (T : ℕ) :
    (univ.filter fun p => tailOverwrite L l T p ≠ none).card ≤ (L + 1) * Fintype.card O := by
  classical
  calc (univ.filter fun p => tailOverwrite L l T p ≠ none).card
      ≤ ((Ico (accLen l) (count L l)) ×ˢ (univ : Finset O)).card := by
        refine card_le_card_of_injOn (fun p => (p.1.val, p.2)) (fun p hp => ?_) (fun p _ p' _ h => ?_)
        · have hp' : tailOverwrite L l T p ≠ none := (mem_filter.1 hp).2
          have hc : accLen l ≤ p.1.val ∧ p.1.val < count L l := by
            by_contra hc
            exact hp' (if_neg hc)
          exact mem_product.2 ⟨mem_Ico.2 hc, mem_univ _⟩
        · simp only [Prod.mk.injEq] at h
          exact Prod.ext (Fin.ext h.1) h.2
    _ = (count L l - accLen l) * Fintype.card O := by rw [card_product, Nat.card_Ico, card_univ]
    _ ≤ (L + 1) * Fintype.card O := Nat.mul_le_mul_right _ (by have := count_le_accLen L l; omega)

variable [Fintype O] [DecidableEq O] [Fintype V] [Fintype Adv] [Inhabited Adv] [Fintype Tick] [Inhabited Tick]
  [Fintype F]

/-- **A faithful run that leaves fewer than `M` incorrect proof blocks undetected is the fault channel's deviation with
its own covert string**, at the patterns `κ` doesn't cut: for each window accepted before the first rejection its
pattern, or `κ`'s hint where `κ` cuts it; each window's advice and bucket; the values released from the first rejection
through the halt; and the halt. -/
theorem view_eq_faultResp (Cf : Θ → (Q × Adv) × F → O → V) (size : F → ℕ) (φ₀ : F) (hφ₀ : size φ₀ = 0)
    (κ : Cut (Q × Adv) (O → V)) (M : ℕ) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ') (T L : ℕ) (θ : Θ)
    (r : R) (l : List (Served O V Tick Adv)) (hfa : Faithful Cf size Ou L θ r l) (hesc : ¬ Undetected M l) :
    ∃ s : (faultChannel R Q Adv O V Tick (κ.Interiors Cf size φ₀) κ (fun φ => size φ.1) ⟨φ₀, Or.inl rfl⟩ T M
        ((L + 1) * Fintype.card O)).S,
      view Ou L r l T = Ou.view r (fun h x => (faultChannel R Q Adv O V Tick (κ.Interiors Cf size φ₀) κ
        (fun φ => size φ.1) ⟨φ₀, Or.inl rfl⟩ T M ((L + 1) * Fintype.card O)).resp
          (fun θ (x : (Q × Adv) × κ.Interiors Cf size φ₀) => Cf θ (x.1, x.2.1)) s θ r
          (liftView ⟨φ₀, Or.inl rfl⟩ h) (x, ⟨φ₀, Or.inl rfl⟩)) T := by
  classical
  have hcl := count_le_length L l
  have hac := accLen_le_count L l
  have hacc : ¬ ∃ w ∈ l, w.accepted ∧ w.sampledIncorrect := fun h => hesc (Or.inl h)
  have hlead : lead l < M := lt_of_not_ge fun h => hesc (Or.inr h)
  have hpat : ∀ n, ∃ e : κ.Interiors Cf size φ₀ ⊕ κ.H,
      Sum.elim (fun φ => size φ.1) κ.size e ≤ (if n < accLen l then incorrectAt l n else 0) ∧
      (n < accLen l → ∀ w, l[n]? = some w → w.egress =
        Sum.elim (fun φ => Cf θ (((Ou.query r (view Ou L r l n)).1, w.adv), φ.1))
          (fun c => κ.sim ((Ou.query r (view Ou L r l n)).1, w.adv) c
            (Cf θ (κ.ask ((Ou.query r (view Ou L r l n)).1, w.adv) c, φ₀))) e) := by
    intro n
    by_cases hn : n < accLen l
    · have hw : l[n]? = some l[n] := List.getElem?_eq_getElem (lt_of_lt_of_le hn (hac.trans hcl))
      obtain ⟨φ, hφs, hφe⟩ := hfa n (lt_of_lt_of_le hn hac) l[n] hw (accepted_of_lt_accLen l n l[n] hn hw)
      have hinc : size φ ≤ (if n < accLen l then incorrectAt l n else 0) := by
        rw [if_pos hn]
        simpa [incorrectAt, hw] using hφs
      by_cases hi : κ.Interior Cf size φ₀ φ
      · refine ⟨.inl ⟨φ, hi⟩, hinc, fun _ w hw' => ?_⟩
        rw [hw, Option.some.injEq] at hw'
        subst hw'
        exact hφe
      · have hcut : κ.Cuts (fun θ x => Cf θ (x, φ₀)) Cf size (· = φ) := by
          by_contra hc
          exact hi (Or.inr hc)
        obtain ⟨c, hcs, hce⟩ := hcut θ ((Ou.query r (view Ou L r l n)).1, l[n].adv) φ rfl
        refine ⟨.inr c, le_trans hcs hinc, fun _ w hw' => ?_⟩
        rw [hw, Option.some.injEq] at hw'
        subst hw'
        exact hφe.trans hce
    · exact ⟨.inl ⟨φ₀, Or.inl rfl⟩, by simp [hn, hφ₀], fun h => absurd h hn⟩
  choose es hes using hpat
  have hsum : ∑ t : Fin T, Sum.elim (fun φ => size φ.1) κ.size (es t) ≤ M := by
    calc ∑ t : Fin T, Sum.elim (fun φ => size φ.1) κ.size (es t)
        ≤ ∑ t : Fin T, (if (t : ℕ) < accLen l then incorrectAt l t else 0) :=
          sum_le_sum fun t _ => (hes t).1
      _ = ∑ n ∈ range T, (if n < accLen l then incorrectAt l n else 0) :=
          Fin.sum_univ_eq_sum_range (fun n => if n < accLen l then incorrectAt l n else 0) T
      _ = ∑ n ∈ (range T).filter (· < accLen l), incorrectAt l n := (sum_filter _ _).symm
      _ ≤ ∑ n ∈ range (accLen l), incorrectAt l n :=
          sum_le_sum_of_subset_of_nonneg (fun n hn => mem_range.2 (mem_filter.1 hn).2) fun _ _ _ => Nat.zero_le _
      _ ≤ lead l := sum_incorrectAt_le_lead l hacc
      _ ≤ M := hlead.le
  refine ⟨(⟨fun t => es t, hsum⟩, fun t => (l[t.val]?.map (·.adv)).getD default,
    fun t => (l[t.val]?.map (·.tick)).getD default, ⟨tailOverwrite L l T, card_tailOverwrite_le L l T⟩,
    ⟨min (count L l) T, Nat.lt_succ_of_le (min_le_right _ _)⟩), ?_⟩
  refine Outsider.view_congr Ou r _ _ T fun n hn => ?_
  have hlen := Outsider.view_length Ou r (fun h _ => reply L l h.length) n
  have hv : view Ou L r l n = Ou.view r (fun h _ => reply L l h.length) n := rfl
  generalize Ou.view r (fun h _ => reply L l h.length) n = V at hlen hv ⊢
  subst hlen
  simp only [Channel.resp, faultChannel, liftView, List.length_map]
  rw [reply]
  by_cases hnc : V.length < count L l
  · have hw : l[V.length]? = some l[V.length] := List.getElem?_eq_getElem (lt_of_lt_of_le hnc hcl)
    rw [if_pos hnc, dif_pos ⟨lt_min hnc hn, hn⟩, dif_pos hn]
    simp only [hw, Option.map, Option.getD_some]
    refine congrArg some (Prod.ext (funext fun o => ?_) rfl)
    dsimp only
    by_cases ha : V.length < accLen l
    · have hno : tailOverwrite L l T (⟨V.length, hn⟩, o) = none :=
        if_neg fun h => absurd h.1 (not_le.2 ha)
      rw [hno, Option.getD_none, (hes V.length).2 ha l[V.length] hw, hv]
      rcases es V.length with φ | c <;> rfl
    · have hso : tailOverwrite L l T (⟨V.length, hn⟩, o) = some (l[V.length].egress o) := by
        unfold tailOverwrite
        rw [if_pos ⟨not_lt.1 ha, hnc⟩, hw]
        rfl
      rw [hso, Option.getD_some]
  · rw [if_neg hnc, dif_neg fun h => hnc (lt_of_lt_of_le h.1 (min_le_left _ _))]

omit [Fintype O] in
/-- **`Faithful` from the dataflow.** If each accepted window that releases released, on the egress slots `eg`, a trace
`σ` of a checked population at the inputs `inp θ x` of the outsider's query `x` served with the window's advice, with at
most the window's count of incorrect units (the network certifier's tie), then its releases are faulted runs of the
population's faulted function, each at its trace's own fault pattern. -/
theorem faithful_of_tie {U S : Type} [Fintype U] (P : Population.Dataflow U S) (h : P.Checked) (f : P.Fns V)
    (inp : Θ → Q × Adv → S → V) (eg : O → S) (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ') (L : ℕ)
    (θ : Θ) (r : R) (l : List (Served O V Tick Adv))
    (htie : ∀ n < count L l, ∀ w, l[n]? = some w → w.accepted → ∃ σ : S → V,
      (∀ s ∈ P.input, σ s = inp θ ((Ou.query r (view Ou L r l n)).1, w.adv) s) ∧
        (P.incorrect f σ).card ≤ w.incorrect ∧ w.egress = fun o => σ (eg o)) :
    Faithful (fun θ x => P.faultedOut h f eg (inp θ x.1) x.2) (fun φ => (P.faultSet φ).card) Ou L θ r l := by
  intro n hn w hw ha
  obtain ⟨σ, hin, hcard, heg⟩ := htie n hn w hw ha
  refine ⟨P.faultsOf f σ, ?_, ?_⟩
  · show (P.faultSet (P.faultsOf f σ)).card ≤ w.incorrect
    rw [P.faultSet_faultsOf]
    exact hcard
  · rw [heg]
    exact (P.faultedOut_eq h f eg ⟨hin, (P.runOf_faultsOf f σ).2⟩).symm

end Served

namespace SecurityProofs

open Verity.Game WeightSecurity.Assumptions
open scoped ENNReal

theorem TheftFaulted : Guarantees.TheftFaulted := by
  intro Ω R Θ Q Adv O V Tick F Θ' _ _ _ _ _ _ _ _ _ _ C Cf size φ₀ κ gen Typ Stolen A Af p lam Ou T L M δ run hφ₀ hhon
    hfh hsim hesc
  classical
  have hcov : ∀ (z : Ω × R) (l : List (Served O V Tick Adv)),
      Stolen (gen z.1) (Ou.out z.2 (Served.view Ou L z.2 l T)) →
        (gen z.1 ∉ Typ ∨ ¬ Served.Faithful Cf size Ou L (gen z.1) z.2 l ∨ Served.Undetected M l) ∨
          ∃ s : (faultChannel R Q Adv O V Tick (κ.Interiors Cf size φ₀) κ (fun φ => size φ.1) ⟨φ₀, Or.inl rfl⟩ T M
              ((L + 1) * Fintype.card O)).S,
            (fun θ θ' => θ ∈ Typ ∧ Stolen θ θ') (gen z.1)
              (((Ou.withFaults (⟨φ₀, Or.inl rfl⟩ : κ.Interiors Cf size φ₀)).withString
                (faultChannel R Q Adv O V Tick (κ.Interiors Cf size φ₀) κ (fun φ => size φ.1) ⟨φ₀, Or.inl rfl⟩ T M
                  ((L + 1) * Fintype.card O)) s).model z.2
                (honest (fun θ (x : (Q × Adv) × κ.Interiors Cf size φ₀) => Cf θ (x.1, x.2.1)) (gen z.1)) T) := by
    intro z l hst
    by_cases ht : gen z.1 ∈ Typ
    · by_cases hfa : Served.Faithful Cf size Ou L (gen z.1) z.2 l
      · by_cases he : Served.Undetected M l
        · exact Or.inl (Or.inr (Or.inr he))
        · obtain ⟨s, hs⟩ := Served.view_eq_faultResp Cf size φ₀ hφ₀ κ M Ou T L (gen z.1) z.2 l hfa he
          refine Or.inr ⟨s, ht, ?_⟩
          rw [withString_model, Outsider.withFaults_model, Outsider.model, ← hs]
          exact hst
      · exact Or.inl (Or.inr (Or.inl hfa))
    · exact Or.inl (Or.inl ht)
  refine (value_leak (fun θ (x : (Q × Adv) × κ.Interiors Cf size φ₀) => Cf θ (x.1, x.2.1)) gen
    (fun θ θ' => θ ∈ Typ ∧ Stolen θ θ') T (Af ∩ FaultBudget (fun φ : κ.Interiors Cf size φ₀ => size φ.1) T M)
    ((patternCount (fun φ : κ.Interiors Cf size φ₀ => size φ.1) T M : ℝ) ^ (lam - 1) * p)
    (Ou.withFaults (⟨φ₀, Or.inl rfl⟩ : κ.Interiors Cf size φ₀))
    (faultChannel R Q Adv O V Tick (κ.Interiors Cf size φ₀) κ (fun φ => size φ.1) ⟨φ₀, Or.inl rfl⟩ T M
      ((L + 1) * Fintype.card O)) (hfh p hhon)
    (fun s => ⟨hsim s, withString_mem_faultBudget Ou κ (fun φ : κ.Interiors Cf size φ₀ => size φ.1)
      (⟨φ₀, Or.inl rfl⟩ : κ.Interiors Cf size φ₀) hφ₀ T M _ s⟩)
    (fun z => run (gen z.1) z.2)
    (fun z l => Stolen (gen z.1) (Ou.out z.2 (Served.view Ou L z.2 l T)))
    (fun z l => gen z.1 ∉ Typ ∨ ¬ Served.Faithful Cf size Ou L (gen z.1) z.2 l ∨ Served.Undetected M l)
    hcov).trans ?_
  have hz : ∀ z : Ω × R, value (fun l => gen z.1 ∉ Typ ∨ ¬ Served.Faithful Cf size Ou L (gen z.1) z.2 l ∨
      Served.Undetected M l) (run (gen z.1) z.2) ≤ (if gen z.1 ∉ Typ then 1 else 0) +
        (value (fun l => ¬ Served.Faithful Cf size Ou L (gen z.1) z.2 l) (run (gen z.1) z.2) + δ) := fun z =>
    (value_or_le _ _ _).trans
      (add_le_add (value_const_le _ _) ((value_or_le _ _ _).trans (add_le_add le_rfl (hesc _ _))))
  have hind : avg (fun z : Ω × R => if gen z.1 ∉ Typ then (1 : ℝ≥0∞) else 0) =
      ENNReal.ofReal (Claims.Pr fun z : Ω × R => gen z.1 ∉ Typ) := by
    convert avg_indicator (fun z : Ω × R => gen z.1 ∉ Typ)
  generalize ENNReal.ofReal (Fintype.card (faultChannel R Q Adv O V Tick (κ.Interiors Cf size φ₀) κ
    (fun φ => size φ.1) ⟨φ₀, Or.inl rfl⟩ T M ((L + 1) * Fintype.card O)).S *
      ((patternCount (fun φ : κ.Interiors Cf size φ₀ => size φ.1) T M : ℝ) ^ (lam - 1) * p)) = B
  calc avg (fun z : Ω × R => value (fun l => gen z.1 ∉ Typ ∨ ¬ Served.Faithful Cf size Ou L (gen z.1) z.2 l ∨
          Served.Undetected M l) (run (gen z.1) z.2)) + B
      ≤ avg (fun z : Ω × R => (if gen z.1 ∉ Typ then (1 : ℝ≥0∞) else 0) +
          (value (fun l => ¬ Served.Faithful Cf size Ou L (gen z.1) z.2 l) (run (gen z.1) z.2) + δ)) + B :=
        add_le_add (avg_mono hz) le_rfl
    _ ≤ (ENNReal.ofReal (Claims.Pr fun z : Ω × R => gen z.1 ∉ Typ) +
          (avg (fun z : Ω × R => value (fun l => ¬ Served.Faithful Cf size Ou L (gen z.1) z.2 l)
            (run (gen z.1) z.2)) + δ)) + B := by
        rw [avg_add, avg_add, hind]
        gcongr
        exact avg_le_const δ
    _ = _ := by ring

namespace ToyFaulted

open WeightSecurity.ToyServed

/-- The toy's faulted function at the patterns no cut covers. -/
abbrev CfI : W → (Unit × Unit) × FaultI → Unit → Fin 16 := fun θ x => Cf θ (x.1, x.2.1)

theorem honest_Cf (θ : W) : honest CfI θ = honest CfI (0, 0) := rfl

/-- Faulted use of the toy reveals nothing: the faulted function reads no weight, so every party's output is one fixed
guess. -/
theorem guess_le (P : Outsider Unit ((Unit × Unit) × FaultI) (Unit → Fin 16) W) :
    Claims.Pr (fun z : W × Unit => gen z.1 ∈ Set.univ ∧ Stolen (gen z.1)
      (P.model z.2 (honest CfI (gen z.1)) 2)) ≤ 1 / 256 := by
  calc Claims.Pr (fun z : W × Unit => gen z.1 ∈ Set.univ ∧ Stolen (gen z.1)
        (P.model z.2 (honest CfI (gen z.1)) 2))
      ≤ Pouw.pr (fun z : W × Unit => z.1 = P.model () (honest CfI (0, 0)) 2) :=
        Pouw.pr_mono fun z h => by
          obtain ⟨θ, ⟨⟩⟩ := z
          have h' : P.model () (honest CfI θ) 2 = θ := h.2
          rw [honest_Cf] at h'
          exact h'.symm
    _ = 1 / 256 := by
        rw [Pouw.pr_eq_card _ {(P.model () (honest CfI (0, 0)) 2, ())} fun z => by simp [Prod.ext_iff],
          card_singleton, ToyServed.card_W]
        norm_num

/-- The party that asks twice and outputs `(0, 0)`. -/
def guesser : Outsider Unit (Unit × Unit) (Unit → Fin 16) W where
  query _ _ := ((), ())
  out _ _ := (0, 0)

/-- **Every honest-use bound at the toy is at least `1/256`**: the party that outputs `(0, 0)` steals the weights on
one coin in `256`. -/
theorem le_of_hard {p : ℝ}
    (hp : HonestUseHard C gen (fun θ θ' => θ ∈ Set.univ ∧ Stolen θ θ') 2 anyone p) : 1 / 256 ≤ p := by
  have he : Pouw.pr (fun z : W × Unit => (fun θ θ' => θ ∈ Set.univ ∧ Stolen θ θ') (gen z.1)
      (guesser.model z.2 (honest C (gen z.1)) 2)) = 1 / 256 := by
    rw [Pouw.pr_eq_card _ {((0, 0), ())} fun z => ?_, card_singleton, ToyServed.card_W]
    · norm_num
    · obtain ⟨θ, ⟨⟩⟩ := z
      constructor
      · intro hz
        have h1 : θ = (0, 0) := congrArg Prod.fst (mem_singleton.1 hz)
        exact ⟨Set.mem_univ _, h1.symm⟩
      · intro hz
        have h1 : (0, 0) = θ := hz.2
        exact mem_singleton.2 (by rw [← h1])
  rw [← he]
  exact hp guesser (Set.mem_univ _)

/-- At the budget `0` the toy's fault channel has one string of patterns and hints: the empty pattern in each window,
since every other pattern has size `1` and `Cut.none` has no hints. -/
theorem strings_subsingleton :
    Subsingleton {w : Fin 2 → FaultI ⊕ (Cut.none (Unit × Unit) (Unit → Fin 16)).H //
      ∑ t, Sum.elim (fun φ : FaultI => size φ.1) (Cut.none (Unit × Unit) (Unit → Fin 16)).size (w t) ≤ 0} := by
  have hz : ∀ w : Fin 2 → FaultI ⊕ (Cut.none (Unit × Unit) (Unit → Fin 16)).H,
      ∑ t, Sum.elim (fun φ : FaultI => size φ.1) (Cut.none (Unit × Unit) (Unit → Fin 16)).size (w t) ≤ 0 →
        ∀ t, w t = .inl ⟨none, Or.inl rfl⟩ := by
    intro w hw t
    have h0 := (Finset.sum_eq_zero_iff.1 (Nat.le_zero.1 hw)) t (Finset.mem_univ t)
    rcases hc : w t with ⟨φ, hφ⟩ | c
    · rw [hc] at h0
      cases φ with
      | none => rfl
      | some v => exact absurd h0 (by simp [size])
    · exact c.elim
  exact ⟨fun a b => Subtype.ext (funext fun t => (hz a.1 a.2 t).trans (hz b.1 b.2 t).symm)⟩

end ToyFaulted

theorem TheftFaultedHonest : Guarantees.TheftFaultedHonest := by
  classical
  obtain ⟨-, -, hund⟩ := TheftServedHonest
  refine ⟨rfl, fun P _ => ?_, fun p hp P _ => ?_, fun _ => Set.mem_univ _, hund⟩
  · calc Claims.Pr (fun z : WeightSecurity.ToyServed.W × Unit => (fun θ θ' => θ ∈ Set.univ ∧
          WeightSecurity.ToyServed.Stolen θ θ') (WeightSecurity.ToyServed.gen z.1)
          (P.model z.2 (honest WeightSecurity.ToyServed.C (WeightSecurity.ToyServed.gen z.1)) 2))
        ≤ Pouw.pr (fun z : WeightSecurity.ToyServed.W × Unit => WeightSecurity.ToyServed.Stolen
          (WeightSecurity.ToyServed.gen z.1)
          (P.model z.2 (honest WeightSecurity.ToyServed.C (WeightSecurity.ToyServed.gen z.1)) 2)) :=
          Pouw.pr_mono fun _ h => h.2
      _ ≤ 1 / 256 := ToyServed.guess_le P
  · rw [sub_self, Real.rpow_zero, one_mul]
    exact (ToyFaulted.guess_le P).trans (ToyFaulted.le_of_hard hp)

theorem TheftFaultedNontrivial : Guarantees.TheftFaultedNontrivial := by
  classical
  intro h
  have hsteal : ∀ z : WeightSecurity.ToyServed.W × Unit, value (fun l => WeightSecurity.ToyServed.Stolen
      (WeightSecurity.ToyServed.gen z.1) (WeightSecurity.ToyServed.thief.out z.2
        (Served.view WeightSecurity.ToyServed.thief 0 z.2 l 2)))
      (WeightSecurity.ToyServed.leakRun (WeightSecurity.ToyServed.gen z.1) z.2) = 1 := by
    intro z
    unfold WeightSecurity.ToyServed.leakRun
    rw [value_ret, ite_eq_left]
    simp [WeightSecurity.ToyServed.Stolen, WeightSecurity.ToyServed.gen, WeightSecurity.ToyServed.thief, Served.view,
      Served.reply, Served.count, Outsider.view]
  have hfa : ∀ z : WeightSecurity.ToyServed.W × Unit, value (fun l => ¬ Served.Faithful WeightSecurity.ToyServed.Cf
      WeightSecurity.ToyServed.size WeightSecurity.ToyServed.thief 0 (WeightSecurity.ToyServed.gen z.1) z.2 l)
      (WeightSecurity.ToyServed.leakRun (WeightSecurity.ToyServed.gen z.1) z.2) = 0 := by
    intro z
    unfold WeightSecurity.ToyServed.leakRun
    rw [value_ret, ite_eq_right (not_not.2 ?_)]
    intro n _ w hw _
    have hwr : w.incorrect = 1 := by
      rcases n with _ | _ | n
      · simp only [List.getElem?_cons_zero, Option.some.injEq] at hw
        rw [← hw]
      · simp only [List.getElem?_cons_succ, List.getElem?_cons_zero, Option.some.injEq] at hw
        rw [← hw]
      · simp at hw
    refine ⟨some (w.egress ()), ?_, funext fun _ => rfl⟩
    rw [hwr]
    exact le_rfl
  have htyp : Claims.Pr (fun z : WeightSecurity.ToyServed.W × Unit =>
      WeightSecurity.ToyServed.gen z.1 ∉ (Set.univ : Set WeightSecurity.ToyServed.W)) = 0 :=
    Pouw.pr_eq_zero fun z hz => hz (Set.mem_univ _)
  have hN : (patternCount (fun φ : WeightSecurity.ToyServed.FaultI => WeightSecurity.ToyServed.size φ.1) 2 0 : ℝ) ^
      ((1 : ℝ) - 1) = 1 := by
    rw [sub_self, Real.rpow_zero]
  have hc : Fintype.card (faultChannel Unit Unit Unit Unit (Fin 16) Unit WeightSecurity.ToyServed.FaultI (Cut.none _ _)
      (fun φ : WeightSecurity.ToyServed.FaultI => WeightSecurity.ToyServed.size φ.1) ⟨none, Or.inl rfl⟩ 2 0
      ((0 + 1) * Fintype.card Unit)).S ≤ 144 := by
    calc Fintype.card (faultChannel Unit Unit Unit Unit (Fin 16) Unit WeightSecurity.ToyServed.FaultI (Cut.none _ _)
      (fun φ : WeightSecurity.ToyServed.FaultI => WeightSecurity.ToyServed.size φ.1) ⟨none, Or.inl rfl⟩ 2 0
      ((0 + 1) * Fintype.card Unit)).S
        ≤ Fintype.card (servedChannel Unit Unit Unit Unit (Fin 16) Unit 2 ((0 + 1) * Fintype.card Unit)).S :=
          Fintype.card_le_of_injective (fun s => s.2) fun a b h =>
            Prod.ext (@Subsingleton.elim _ ToyFaulted.strings_subsingleton _ _) h
      _ ≤ 144 := (card_servedChannel_le Unit Unit Unit Unit (Fin 16) Unit 2 _).trans (by simp [Finset.sum_range_succ])
  simp only [hsteal, hfa, avg_const, add_zero, htyp, ENNReal.ofReal_zero, zero_add, hN, one_mul] at h
  have hlt : ENNReal.ofReal ((Fintype.card (faultChannel Unit Unit Unit Unit (Fin 16) Unit
      WeightSecurity.ToyServed.FaultI (Cut.none _ _)
      (fun φ : WeightSecurity.ToyServed.FaultI => WeightSecurity.ToyServed.size φ.1) ⟨none, Or.inl rfl⟩ 2 0
      ((0 + 1) * Fintype.card Unit)).S : ℝ) * (1 / 256)) < 1 := by
    rw [ENNReal.ofReal_lt_one]
    have : (Fintype.card (faultChannel Unit Unit Unit Unit (Fin 16) Unit WeightSecurity.ToyServed.FaultI (Cut.none _ _)
      (fun φ : WeightSecurity.ToyServed.FaultI => WeightSecurity.ToyServed.size φ.1) ⟨none, Or.inl rfl⟩ 2 0
      ((0 + 1) * Fintype.card Unit)).S : ℝ) ≤ 144 := by exact_mod_cast hc
    linarith
  exact absurd h (not_le.2 hlt)

end SecurityProofs

end WeightSecurity
