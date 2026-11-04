module

public import Definitions.Claims.SamplingGeneric
public import Proofs.Flock.Claims.Sampling.KS
public import Proofs.Flock.SamplingGeneric.Split
public import Proofs.Flock.SamplingGeneric.Bits

/-!
# The compiled extractor's waits

`Compiled.E` waits twice: for `k` accepted reruns among `K` fresh draws of the whole run (the outer wait), and at each
kept one for `k` accepted reruns among `K` fresh draws of the second stage (the inner wait), and at each of those
extracts every sampled proof block on coins of its own. This file bounds, at any compiled sampled verifier `L`, the
probability that a wait comes up short or that every run it keeps is bad, by `waitAccC`'s truncation and union bound
(`Sampling.wait_bad_le`, `2k/K + e^{−k/8}` plus the union's size against the slack `c`), generic in what a proof block's
extraction finding a witness means (`ok`). The extraction's failure event (`Fails`) is the split's target, and its
probability is `κ(δ) + √(γ·Pr[B])` from a bound on the sessions' extraction (`Pr_fails_le`), so the compiled verifier's
`KS` needs only the split and that bound (`KS_of_split`).

* `UCoins`, `E1`, `ecoinsEquiv`: **the coins** of `ECoins` as the outer wait's and, per outer run, its inner wait's and
  every proof block's extractor coins at each inner run.
* `Good`, `BadOuter`, `Fails`: **the events**. A sampled proof block whose extraction succeeds; an outer run whose inner
  wait comes up short or keeps no run that samples a block of `B` and extracts it; and, in scope, the outer wait short
  or, for some faulty set `B` the relation at `δ` refuses, every kept outer run bad for `B`.
* `inner_bad_le`, `outer_bad_le`: **one wait each**. At first-stage coins where the sample misses `B` with probability at
  most `esc` (`Sessions.draws`) and the sampled block a selection picks fails its extraction with probability at most
  `κs`, the inner wait is bad with probability `2k/K + e^{−k/8} + c·(esc + κs)`; averaged over the first stage, the outer
  wait at `T + c·(T + c·(δ + 𝔼[κs]))`, `T = 2k/K + e^{−k/8}`, its union over at most `2^{n_PU}` sets costing `c` at
  `k = 2^40·(2·n_PU + 1)`.
* `fails_tape_le`, `Pr_fails_le`: **the failure event's probability**, at one tape and over the tapes, the sessions'
  error `κ_s + √(k_V·γ_s·β)` averaging by Jensen to `κ(δ) + √(γ·Pr[B])`.
* `KS_of_split`, `KS_of_sessionKS`: **`Compiled.KS`** from the split into `Fails` or an own-call collision and the
  sessions' bound, which at the extraction success `UnitOk` is `SessionKS` (`SamplingGeneric/Split.lean`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScope forkOn slack accList AllBad wait_bad_le S₁ card_S₁_le
  card_rpow_le_slack Pr_or_le Pr_acc_forkOn forkOn_forkOn' slack_pos OwnColl KSLin_of_ownColl)

theorem le_foldr_max {a : ℝ} : ∀ {l : List ℝ}, a ∈ l → a ≤ l.foldr max 0
  | [], h => by simp at h
  | b :: l, h => by
    rw [List.foldr_cons]
    rcases List.mem_cons.1 h with rfl | h
    · exact le_max_left _ _
    · exact le_max_of_le_right (le_foldr_max h)

theorem le_maxBelow {n : ℕ} (f : ℕ → ℝ) {u : ℕ} (hu : u < n) : f u ≤ maxBelow n f :=
  le_foldr_max (List.mem_map.2 ⟨u, List.mem_range.2 hu, rfl⟩)

namespace Compiled

/-! ## The waits' numbers -/

section Numbers

variable {Coin : Type} {A : Arg Coin}

theorem k_pos (L : Compiled Coin A) (x : Msg) : 0 < L.k x := by
  unfold Compiled.k; positivity

theorem cap_pos (L : Compiled Coin A) (x : Msg) : 0 < L.cap x := by
  unfold Compiled.cap; exact Nat.mul_pos (by positivity) (L.k_pos x)

theorem nPU_eq {L : Compiled Coin A} {x : Msg} {s : AuditStatement} {Pt : Partition} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) : L.nPU x = Pt.blocks.length := by
  simp only [Compiled.nPU, hs, Compiled.part, hPt, Option.getD_some]

/-- At most `2^m` sets, `m ≤ 2·n_PU + 1`, spread over the waits' `k` accepted runs cost at most the slack. -/
theorem rpow_k_le_slack (L : Compiled Coin A) (x : Msg) {N m : ℕ} (hN : N ≤ 2 ^ m) (hm : m ≤ 2 * L.nPU x + 1) :
    (N : ℝ) ^ (1 / (L.k x : ℝ)) ≤ slack := by
  unfold Compiled.k
  exact card_rpow_le_slack hN hm

theorem κs_nonneg (L : Compiled Coin A) (x : Msg) : 0 ≤ L.κs x := maxBelow_nonneg _ _

theorem γs_nonneg (L : Compiled Coin A) (x : Msg) : 0 ≤ L.γs x := maxBelow_nonneg _ _

/-- Each proof block's `κ` is at most the sessions' (`κs`). -/
theorem κ_le_κs (L : Compiled Coin A) (x : Msg) {u : ℕ} (hu : u < L.nPU x) : A.κ (L.sh x u) ≤ L.κs x :=
  le_maxBelow (fun u => A.κ (L.sh x u)) hu

/-- Each proof block's `γ` per round is at most the sessions' (`γs`). -/
theorem γ_le_γs (L : Compiled Coin A) (x : Msg) {u : ℕ} (hu : u < L.nPU x) :
    A.γ (L.sh x u) / A.rounds (L.sh x u) ≤ L.γs x :=
  le_maxBelow (fun u => A.γ (L.sh x u) / A.rounds (L.sh x u)) hu

end Numbers

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## The coins -/

/-- The extractor's coins at one inner run: each proof block's, `A`'s extractor's at the block's shape. -/
abbrev UCoins (L : Compiled Coin A) (x : Msg) := (u : Fin (L.nPU x)) → A.EC (L.sh x u)

/-- The extractor's coins for one outer run: its inner wait's, and at each inner run every proof block's. -/
abbrev E1 (L : Compiled Coin A) (x : Msg) :=
  (Fin (L.cap x) → Fin (L.V.k x) → Coin) × (Fin (L.k x) → UCoins L x)

def toE1 (L : Compiled Coin A) (x : Msg) (c : ECoins L x) (j : Fin (L.k x)) : E1 L x :=
  (c.2.1 j, fun l u => c.2.2 (j, l, u))

/-- The extractor's coins as the outer wait's and one `E1` per outer run. -/
def ecoinsEquiv (L : Compiled Coin A) (x : Msg) :
    ECoins L x ≃ (Fin (L.cap x) → Fin (L.V.k x) → Coin) × (Fin (L.k x) → E1 L x) where
  toFun c := (c.1, toE1 L x c)
  invFun p := (p.1, fun j => (p.2 j).1, fun t => (p.2 t.1).2 t.2.1 t.2.2)
  left_inv _ := rfl
  right_inv _ := rfl

/-! ## The events -/

/-- The proof block `v` is one the run on coins `r'` samples, and its extraction on its coins in `e` succeeds (`ok`). -/
def Good (L : Compiled Coin A) (x : Msg)
    (ok : (Fin (L.V.k x) → Coin) → (u : Fin (L.nPU x)) → A.EC (L.sh x u) → Prop) (r' : Fin (L.V.k x) → Coin)
    (v : ℕ) (e : UCoins L x) : Prop :=
  ∃ u : Fin (L.nPU x), u.val = v ∧ v ∈ L.checked x r' ∧ ok r' u (e u)

/-- **An outer run bad for `B`**, its second stage `g`: the inner wait comes up short, or no kept inner run samples a
proof block of `B` and extracts it. -/
def BadOuter (L : Compiled Coin A) (x : Msg)
    (ok : (Fin (L.V.k x) → Coin) → (u : Fin (L.nPU x)) → A.EC (L.sh x u) → Prop)
    (run' : (Fin (L.V.k x) → Coin) → Trace) (B : Finset ℕ) (g : (Fin (L.V.k x) → Coin) → Fin (L.V.k x) → Coin)
    (e : E1 L x) : Prop :=
  (accList (fun d => (run' (g d)).out = .acc) (L.k x) ((List.finRange (L.cap x)).map e.1)).length < L.k x ∨
    AllBad (fun d => (run' (g d)).out = .acc) (L.k x)
      (fun d u => ∀ v ∈ B, ¬ Good L x ok (g d) v u) ((List.finRange (L.cap x)).map e.1) e.2

/-- **The extraction's failure event**: the statement is in scope, and the outer wait comes up short or, for some
faulty set `B` the relation at `δ` refuses, every kept outer run is bad for `B`. -/
def Fails (L : Compiled Coin A) (x : Msg)
    (ok : (Fin (L.V.k x) → Coin) → (u : Fin (L.nPU x)) → A.EC (L.sh x u) → Prop) (δ : ℝ)
    (run' : (Fin (L.V.k x) → Coin) → Trace) (c : ECoins L x) : Prop :=
  ∃ s Pt, L.parse x = some s ∧ L.partition s = some Pt ∧ InScope Pt s ∧
    ((accList (fun r => (run' r).out = .acc) (L.k x) ((List.finRange (L.cap x)).map c.1)).length < L.k x ∨
      ∃ B ∈ S₁ s.design Pt δ, AllBad (fun r => (run' r).out = .acc) (L.k x)
        (fun r e => BadOuter L x ok run' B (forkOn (L.stage2 x) r) e) ((List.finRange (L.cap x)).map c.1)
        (toE1 L x c))

/-! ## One wait each -/

omit [Inhabited Coin] in
/-- **One outer run's inner wait**: at first-stage coins `r` where the sample misses every block of `B` with
probability at most `esc`, and the sampled proof block any selection picks fails its extraction with probability at
most `κs`, the inner wait (times its acceptance) comes up short or keeps only runs that miss or fail with probability at
most `2k/K + e^{−k/8} + c·(esc + κs)`. -/
theorem inner_bad_le (L : Compiled Coin A) (x : Msg)
    (ok : (Fin (L.V.k x) → Coin) → (u : Fin (L.nPU x)) → A.EC (L.sh x u) → Prop)
    (run' : (Fin (L.V.k x) → Coin) → Trace) {B : Finset ℕ} (hB : ∀ v ∈ B, v < L.nPU x) (r : Fin (L.V.k x) → Coin)
    {esc κs : ℝ} (hesc0 : 0 ≤ esc) (hκs0 : 0 ≤ κs)
    (hesc : Pr (fun d : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x (forkOn (L.stage2 x) r d), u ∉ B) ≤ esc)
    (hsess : ∀ sel : List ℕ → Option (Fin (L.nPU x)),
      Pr (fun p : (Fin (L.V.k x) → Coin) × UCoins L x =>
        (run' (forkOn (L.stage2 x) r p.1)).out = .acc ∧
          ∃ u, sel (L.checked x (forkOn (L.stage2 x) r p.1)) = some u ∧
            u.val ∈ L.checked x (forkOn (L.stage2 x) r p.1) ∧ ¬ ok (forkOn (L.stage2 x) r p.1) u (p.2 u)) ≤ κs) :
    Pr (fun d : Fin (L.V.k x) → Coin => (run' (forkOn (L.stage2 x) r d)).out = .acc) *
        Pr (fun e : E1 L x => BadOuter L x ok run' B (forkOn (L.stage2 x) r) e) ≤
      2 * (L.k x : ℝ) / L.cap x + Real.exp (-(L.k x : ℝ) / 8) + slack * (esc + κs) := by
  have hy : ∀ _u ∈ ({()} : Finset Unit),
      Pr (fun p : (Fin (L.V.k x) → Coin) × UCoins L x =>
        (run' (forkOn (L.stage2 x) r p.1)).out = .acc ∧
          ∀ v ∈ B, ¬ Good L x ok (forkOn (L.stage2 x) r p.1) v p.2) ≤ esc + κs := by
    intro _ _
    obtain ⟨sel, hsel⟩ : ∃ sel : List ℕ → Option (Fin (L.nPU x)), ∀ l,
        (∃ q : Fin (L.nPU x), q.val ∈ B ∧ q.val ∈ l) → ∃ i, sel l = some i ∧ i.val ∈ B ∧ i.val ∈ l := by
      classical
      refine ⟨fun l => (List.finRange (L.nPU x)).find? fun q => decide (q.val ∈ B ∧ q.val ∈ l),
        fun l ⟨q, hq⟩ => ?_⟩
      obtain ⟨i, hi⟩ := Option.isSome_iff_exists.1
        ((List.find?_isSome (p := fun q : Fin (L.nPU x) => decide (q.val ∈ B ∧ q.val ∈ l))).2
          ⟨q, List.mem_finRange q, decide_eq_true hq⟩)
      have hpi := List.find?_some hi
      exact ⟨i, hi, of_decide_eq_true hpi⟩
    have hmiss := (Pr_fst (β := UCoins L x) (fun d : Fin (L.V.k x) → Coin =>
      ∀ u ∈ L.checked x (forkOn (L.stage2 x) r d), u ∉ B)).trans_le hesc
    refine Pr_or_le (fun p hp => ?_) hmiss (hsess sel)
    by_cases hm : ∀ u ∈ L.checked x (forkOn (L.stage2 x) r p.1), u ∉ B
    · exact Or.inl hm
    right
    push Not at hm
    obtain ⟨v, hvc, hvB⟩ := hm
    obtain ⟨i, hi, hiB, hic⟩ := hsel _ ⟨⟨v, hB v hvB⟩, hvB, hvc⟩
    exact ⟨hp.1, i, hi, hic, fun hok => hp.2 i.val hiB ⟨i, rfl, hic, hok⟩⟩
  have hw := wait_bad_le (fun d => (run' (forkOn (L.stage2 x) r d)).out = .acc) (L.k x) (L.cap x) (L.cap_pos x)
    (L.k_pos x) ({()} : Finset Unit)
    (fun _ d u => ∀ v ∈ B, ¬ Good L x ok (forkOn (L.stage2 x) r d) v u)
    (add_nonneg hesc0 hκs0) hy
  have hcard : (({()} : Finset Unit).card : ℝ) ^ (1 / (L.k x : ℝ)) ≤ slack :=
    L.rpow_k_le_slack x (m := 0) (by simp) (by omega)
  have hE : Pr (fun e : E1 L x => BadOuter L x ok run' B (forkOn (L.stage2 x) r) e) =
      Pr (fun c : E1 L x =>
        (accList (fun d => (run' (forkOn (L.stage2 x) r d)).out = .acc) (L.k x)
          ((List.finRange (L.cap x)).map c.1)).length < L.k x ∨
        ∃ _u ∈ ({()} : Finset Unit), AllBad (fun d => (run' (forkOn (L.stage2 x) r d)).out = .acc) (L.k x)
          (fun d u => ∀ v ∈ B, ¬ Good L x ok (forkOn (L.stage2 x) r d) v u)
          ((List.finRange (L.cap x)).map c.1) c.2) :=
    Pr_congr fun e => by simp [BadOuter]
  rw [hE]
  exact hw.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_right hcard (add_nonneg hesc0 hκs0)))

set_option synthInstance.maxSize 1024 in
omit [Inhabited Coin] in
/-- **One tape's outer wait**: on a statement in scope at `(s, Pt)`, where a weight `esc` of the first-stage coins bounds
the sample's miss of each set of proof blocks and averages at most its `P_a` (`Sessions.draws`), and the sampled proof
block any selection picks at first-stage coins `r` fails its extraction with probability at most `κs(r)`, the outer
wait (times its acceptance) comes up short or, for some refused `B`, keeps only runs bad for `B`, with probability at
most `T + c·(T + c·(δ + 𝔼[κs]))`, `T = 2k/K + e^{−k/8}`. -/
theorem outer_bad_le (L : Compiled Coin A) (x : Msg)
    (ok : (Fin (L.V.k x) → Coin) → (u : Fin (L.nPU x)) → A.EC (L.sh x u) → Prop) {δ : ℝ} (hδ : 0 ≤ δ)
    (run' : (Fin (L.V.k x) → Coin) → Trace) (s : AuditStatement) (Pt : Partition)
    (hPU : L.nPU x = Pt.blocks.length) (κs : (Fin (L.V.k x) → Coin) → ℝ) (hκs0 : ∀ r, 0 ≤ κs r)
    (hdraw : ∀ B ⊆ range Pt.blocks.length, ∃ esc : (Fin (L.V.k x) → Coin) → ℝ, (∀ r, 0 ≤ esc r) ∧
      (∀ r, Pr (fun d : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x (forkOn (L.stage2 x) r d), u ∉ B) ≤ esc r) ∧
      Ex esc ≤ s.design.Pa Pt B)
    (hsess : ∀ r (sel : List ℕ → Option (Fin (L.nPU x))),
      Pr (fun p : (Fin (L.V.k x) → Coin) × UCoins L x =>
        (run' (forkOn (L.stage2 x) r p.1)).out = .acc ∧
          ∃ u, sel (L.checked x (forkOn (L.stage2 x) r p.1)) = some u ∧
            u.val ∈ L.checked x (forkOn (L.stage2 x) r p.1) ∧ ¬ ok (forkOn (L.stage2 x) r p.1) u (p.2 u)) ≤ κs r) :
    Pr (fun r : Fin (L.V.k x) → Coin => (run' r).out = .acc) *
        Pr (fun c : (Fin (L.cap x) → Fin (L.V.k x) → Coin) × (Fin (L.k x) → E1 L x) =>
          (accList (fun r => (run' r).out = .acc) (L.k x) ((List.finRange (L.cap x)).map c.1)).length < L.k x ∨
            ∃ B ∈ S₁ s.design Pt δ, AllBad (fun r => (run' r).out = .acc) (L.k x)
              (fun r e => BadOuter L x ok run' B (forkOn (L.stage2 x) r) e)
              ((List.finRange (L.cap x)).map c.1) c.2) ≤
      2 * (L.k x : ℝ) / L.cap x + Real.exp (-(L.k x : ℝ) / 8) +
        slack * (2 * (L.k x : ℝ) / L.cap x + Real.exp (-(L.k x : ℝ) / 8) + slack * (δ + Ex κs)) := by
  have hc := slack_pos
  have hEκ : 0 ≤ Ex κs := Ex_nonneg hκs0
  have hy : ∀ B ∈ S₁ s.design Pt δ,
      Pr (fun p : (Fin (L.V.k x) → Coin) × E1 L x =>
        (run' p.1).out = .acc ∧ BadOuter L x ok run' B (forkOn (L.stage2 x) p.1) p.2) ≤
      2 * (L.k x : ℝ) / L.cap x + Real.exp (-(L.k x : ℝ) / 8) + slack * (δ + Ex κs) := by
    intro B hB
    obtain ⟨hBsub, hBδ⟩ := mem_filter.1 hB
    have hBsub' := mem_powerset.1 hBsub
    obtain ⟨esc, hesc0, hesc, hEesc⟩ := hdraw B hBsub'
    have hBn : ∀ v ∈ B, v < L.nPU x := fun v hv => by
      rw [hPU]
      exact mem_range.1 (hBsub' hv)
    refine (Pr_acc_forkOn (L.stage2 x) (fun r => (run' r).out = .acc)
      (fun r e => BadOuter L x ok run' B (forkOn (L.stage2 x) r) e)
      fun r d e => by rw [forkOn_forkOn']).trans_le ?_
    calc Ex (fun r => Pr (fun d => (run' (forkOn (L.stage2 x) r d)).out = .acc) *
          Pr (fun e : E1 L x => BadOuter L x ok run' B (forkOn (L.stage2 x) r) e))
        ≤ Ex (fun r => 2 * (L.k x : ℝ) / L.cap x + Real.exp (-(L.k x : ℝ) / 8) + slack * (esc r + κs r)) :=
          Ex_mono fun r => L.inner_bad_le x ok run' hBn r (hesc0 r) (hκs0 r) (hesc r) (hsess r)
      _ = 2 * (L.k x : ℝ) / L.cap x + Real.exp (-(L.k x : ℝ) / 8) + slack * (Ex esc + Ex κs) := by
          rw [Ex_add, Ex_const, Ex_mul_left, Ex_add]
      _ ≤ 2 * (L.k x : ℝ) / L.cap x + Real.exp (-(L.k x : ℝ) / 8) + slack * (δ + Ex κs) := by
          have := hEesc.trans hBδ
          gcongr
  have hy0 : 0 ≤ 2 * (L.k x : ℝ) / L.cap x + Real.exp (-(L.k x : ℝ) / 8) + slack * (δ + Ex κs) := by
    positivity
  have hw := wait_bad_le (fun r => (run' r).out = .acc) (L.k x) (L.cap x) (L.cap_pos x) (L.k_pos x)
    (S₁ s.design Pt δ) (fun B r e => BadOuter L x ok run' B (forkOn (L.stage2 x) r) e) hy0 hy
  have hcard : ((S₁ s.design Pt δ).card : ℝ) ^ (1 / (L.k x : ℝ)) ≤ slack :=
    L.rpow_k_le_slack x (card_S₁_le _ _ _) (by rw [← hPU]; omega)
  exact hw.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_right hcard hy0))

/-! ## The failure event's probability -/

set_option synthInstance.maxSize 1024 in
/-- **At one tape**: the judged run accepts and the extraction's failure event holds with probability at most
`c²·(δ + 𝔼[κs]) + (1 + c)·(2k/K + e^{−k/8})`, `κs(r)` bounding at first-stage coins `r` the sampled proof block any
selection picks failing its extraction. Out of scope the event is empty; in scope it is `outer_bad_le` at the
contract's draw law. -/
theorem fails_tape_le (H : Msg → Msg) (L : Compiled Coin A) (hS : L.Sessions H) (x : Msg)
    (ok : (Fin (L.V.k x) → Coin) → (u : Fin (L.nPU x)) → A.EC (L.sh x u) → Prop) {δ : ℝ} (hδ : 0 ≤ δ)
    (run' : (Fin (L.V.k x) → Coin) → Trace) (κs : (Fin (L.V.k x) → Coin) → ℝ) (hκs0 : ∀ r, 0 ≤ κs r)
    (hsess : ∀ r (sel : List ℕ → Option (Fin (L.nPU x))),
      Pr (fun p : (Fin (L.V.k x) → Coin) × UCoins L x =>
        (run' (forkOn (L.stage2 x) r p.1)).out = .acc ∧
          ∃ u, sel (L.checked x (forkOn (L.stage2 x) r p.1)) = some u ∧
            u.val ∈ L.checked x (forkOn (L.stage2 x) r p.1) ∧ ¬ ok (forkOn (L.stage2 x) r p.1) u (p.2 u)) ≤ κs r) :
    Pr (fun r : Fin (L.V.k x) → Coin => (run' r).out = .acc) * Pr (fun c : ECoins L x => Fails L x ok δ run' c) ≤
      slack ^ 2 * (δ + Ex κs) + (1 + slack) * (2 * (L.k x : ℝ) / L.cap x + Real.exp (-(L.k x : ℝ) / 8)) := by
  have hc := slack_pos
  have hEκ : 0 ≤ Ex κs := Ex_nonneg hκs0
  have hR : 0 ≤ slack ^ 2 * (δ + Ex κs) +
      (1 + slack) * (2 * (L.k x : ℝ) / L.cap x + Real.exp (-(L.k x : ℝ) / 8)) := by
    positivity
  by_cases hsc : L.Scope x
  swap
  · have h0 : Pr (fun c : ECoins L x => Fails L x ok δ run' c) = 0 :=
      (Pr_congr fun c => ⟨fun ⟨s, Pt, h1, h2, h3, _⟩ => hsc ⟨s, Pt, h1, h2, h3⟩, False.elim⟩).trans Pr_false
    rw [h0, mul_zero]
    exact hR
  obtain ⟨s, Pt, hs, hPt, hin⟩ := hsc
  have hF : Pr (fun c : ECoins L x => Fails L x ok δ run' c) =
      Pr (fun c : (Fin (L.cap x) → Fin (L.V.k x) → Coin) × (Fin (L.k x) → E1 L x) =>
        (accList (fun r => (run' r).out = .acc) (L.k x) ((List.finRange (L.cap x)).map c.1)).length < L.k x ∨
          ∃ B ∈ S₁ s.design Pt δ, AllBad (fun r => (run' r).out = .acc) (L.k x)
            (fun r e => BadOuter L x ok run' B (forkOn (L.stage2 x) r) e)
            ((List.finRange (L.cap x)).map c.1) c.2) := by
    refine Eq.trans ?_ (Sampling.Pr_equiv (L.ecoinsEquiv x) _)
    refine Pr_congr fun c => ⟨fun ⟨s', Pt', hs', hPt', _, h⟩ => ?_, fun h => ⟨s, Pt, hs, hPt, hin, h⟩⟩
    rw [hs, Option.some.injEq] at hs'
    subst hs'
    rw [hPt, Option.some.injEq] at hPt'
    subst hPt'
    exact h
  rw [hF]
  refine (L.outer_bad_le x ok hδ run' s Pt (nPU_eq hs hPt) κs hκs0 (hS.draws x s Pt hs hPt hin) hsess).trans
    (le_of_eq ?_)
  ring

set_option synthInstance.maxSize 1024 in
/-- **Over the tapes**: at a weight `β` of each tape and first-stage coins averaging at most `Pr[B]`, under which the
sampled proof block any selection picks fails its extraction with probability at most `κ_s + √(k_V·γ_s·β(ω, r))`, the
judged run accepts and the extraction's failure event holds with probability at most `κ(δ) + √(γ·Pr[B])`: at each
tape `fails_tape_le`, then Jensen, `c²·√(k_V·γ_s·𝔼[β]) ≤ √(c⁴·k_V·γ_s·Pr[B])`. -/
theorem Pr_fails_le (H : Msg → Msg) (L : Compiled Coin A) (hS : L.Sessions H) {δ : ℝ} (hδ : 0 ≤ δ) (x : Msg)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (ok : Ω → (Fin (L.V.k x) → Coin) → (u : Fin (L.nPU x)) → A.EC (L.sh x u) → Prop)
    (β : Ω → (Fin (L.V.k x) → Coin) → ℝ) (hβ0 : ∀ ω r, 0 ≤ β ω r)
    (hβ : Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => β p.1 p.2) ≤ PrB H L.V P x)
    (hsess : ∀ ω r (sel : List ℕ → Option (Fin (L.nPU x))),
      Pr (fun p : (Fin (L.V.k x) → Coin) × UCoins L x =>
        (run H P L.V x ω (forkOn (L.stage2 x) r p.1)).out = .acc ∧
          ∃ u, sel (L.checked x (forkOn (L.stage2 x) r p.1)) = some u ∧
            u.val ∈ L.checked x (forkOn (L.stage2 x) r p.1) ∧ ¬ ok ω (forkOn (L.stage2 x) r p.1) u (p.2 u)) ≤
        L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r)) :
    Pr (fun s : Ω × (Fin (L.V.k x) → Coin) × ECoins L x =>
      (run H P L.V x s.1 s.2.1).out = .acc ∧ Fails L x (ok s.1) δ (run H P L.V x s.1) s.2.2) ≤
      L.κ δ x + Real.sqrt (L.γ x * PrB H L.V P x) := by
  refine (Pr_indep (fun ω r => (run H P L.V x ω r).out = .acc)
    (fun ω c => Fails L x (ok ω) δ (run H P L.V x ω) c)).trans_le ?_
  have hc := slack_pos
  have hκs := L.κs_nonneg x
  have hγs : 0 ≤ (L.V.k x : ℝ) * L.γs x := mul_nonneg (Nat.cast_nonneg _) (L.γs_nonneg x)
  have hper : ∀ ω, Pr (fun r => (run H P L.V x ω r).out = .acc) *
      Pr (fun c : ECoins L x => Fails L x (ok ω) δ (run H P L.V x ω) c) ≤
      L.κ δ x + slack ^ 2 * Ex (fun r => Real.sqrt (L.V.k x * L.γs x * β ω r)) := by
    intro ω
    refine (L.fails_tape_le H hS x (ok ω) hδ (run H P L.V x ω)
      (fun r => L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r)) (fun r => add_nonneg hκs (Real.sqrt_nonneg _))
      (hsess ω)).trans (le_of_eq ?_)
    rw [Ex_add, Ex_const]
    unfold Compiled.κ
    ring
  calc Ex (fun ω => Pr (fun r => (run H P L.V x ω r).out = .acc) *
        Pr (fun c : ECoins L x => Fails L x (ok ω) δ (run H P L.V x ω) c))
      ≤ Ex (fun ω => L.κ δ x + slack ^ 2 * Ex (fun r => Real.sqrt (L.V.k x * L.γs x * β ω r))) := Ex_mono hper
    _ = L.κ δ x + slack ^ 2 *
          Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => Real.sqrt (L.V.k x * L.γs x * β p.1 p.2)) := by
        rw [Ex_add, Ex_const, Ex_mul_left,
          Ex_prod (fun p : Ω × (Fin (L.V.k x) → Coin) => Real.sqrt (L.V.k x * L.γs x * β p.1 p.2))]
    _ ≤ L.κ δ x + slack ^ 2 * Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) := by
        refine add_le_add le_rfl (mul_le_mul_of_nonneg_left ?_ (pow_nonneg hc.le 2))
        refine (Ex_sqrt_le _ fun p => mul_nonneg hγs (hβ0 _ _)).trans (Real.sqrt_le_sqrt ?_)
        rw [Ex_mul_left]
        exact mul_le_mul_of_nonneg_left hβ hγs
    _ = L.κ δ x + Real.sqrt (L.γ x * PrB H L.V P x) := by
        congr 1
        rw [show L.γ x * PrB H L.V P x = (slack ^ 2) ^ 2 * (L.V.k x * L.γs x * PrB H L.V P x) by
            unfold Compiled.γ; ring,
          Real.sqrt_mul (pow_nonneg (pow_nonneg hc.le 2) 2), Real.sqrt_sq (pow_nonneg hc.le 2)]

/-! ## KS -/

set_option synthInstance.maxSize 1024 in
/-- **`Compiled.KS` from the split and the sessions' bound**: if, at every `δ ≥ 0`, an accepted run whose extraction
fails is in the failure event (at each prover's and tape's extraction success `ok`) or its extraction's own calls
collide, and on every prover a weight `β` averaging at most `Pr[B]` bounds the sampled proof block any selection picks
failing its extraction by `κ_s + √(k_V·γ_s·β)`, then the compiled verifier is knowledge sound
(`Sampling.KSLin_of_ownColl` at `Pr_fails_le`). -/
theorem KS_of_split {H : Msg → Msg} {L : Compiled Coin A} (hS : L.Sessions H)
    (ok : ∀ (x : Msg) (Ω : Type), Prover Ω → Ω → (Fin (L.V.k x) → Coin) → (u : Fin (L.nPU x)) →
      A.EC (L.sh x u) → Prop)
    (hsplit : ∀ δ : ℝ, 0 ≤ δ → ∀ x (Ω : Type) (P : Prover Ω) (s : Ω × (Fin (L.V.k x) → Coin) × ECoins L x),
      (run H P L.V x s.1 s.2.1).out = .acc →
        ¬ (Sampling.R H L.commit L.parse L.partition δ).ext x (L.E.eval H P x s.1 s.2.1 s.2.2).1 →
        Fails L x (ok x Ω P s.1) δ (run H P L.V x s.1) s.2.2 ∨ OwnColl H L.E P x s.1 s.2.1 s.2.2)
    (hsess : ∀ x (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      ∃ β : Ω → (Fin (L.V.k x) → Coin) → ℝ, (∀ ω r, 0 ≤ β ω r) ∧
        Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => β p.1 p.2) ≤ PrB H L.V P x ∧
        ∀ ω r (sel : List ℕ → Option (Fin (L.nPU x))),
          Pr (fun p : (Fin (L.V.k x) → Coin) × UCoins L x =>
            (run H P L.V x ω (forkOn (L.stage2 x) r p.1)).out = .acc ∧
              ∃ u, sel (L.checked x (forkOn (L.stage2 x) r p.1)) = some u ∧
                u.val ∈ L.checked x (forkOn (L.stage2 x) r p.1) ∧
                  ¬ ok x Ω P ω (forkOn (L.stage2 x) r p.1) u (p.2 u)) ≤
            L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r)) :
    L.KS H := by
  intro δ hδ x _
  exact KSLin_of_ownColl H L.V (Sampling.R H L.commit L.parse L.partition δ) L.E (L.κ δ) L.γ
    (fun x Ω P s => (run H P L.V x s.1 s.2.1).out = .acc ∧ Fails L x (ok x Ω P s.1) δ (run H P L.V x s.1) s.2.2)
    (fun x Ω P s hacc hfail => (hsplit δ hδ x Ω P s hacc hfail).imp_left fun h => ⟨hacc, h⟩)
    (fun x Ω _ _ P => by
      obtain ⟨β, hβ0, hβ, hκ⟩ := hsess x Ω P
      exact L.Pr_fails_le H hS hδ x Ω P (ok x Ω P) β hβ0 hβ hκ) x

set_option synthInstance.maxSize 1024 in
/-- **`Compiled.KS` from the sessions' KS and the split**: `SessionKS` is `KS_of_split`'s bound at the extraction
success `UnitOk`, so `KSOf` at a compiled verifier needs only the split into `Fails` at `UnitOk` or an own-call
collision. -/
theorem KS_of_sessionKS {H : Msg → Msg} {L : Compiled Coin A} (hS : L.Sessions H) (hSK : L.SessionKS H)
    (hsplit : ∀ δ : ℝ, 0 ≤ δ → ∀ x (Ω : Type) (P : Prover Ω) (s : Ω × (Fin (L.V.k x) → Coin) × ECoins L x),
      (run H P L.V x s.1 s.2.1).out = .acc →
        ¬ (Sampling.R H L.commit L.parse L.partition δ).ext x (L.E.eval H P x s.1 s.2.1 s.2.2).1 →
        Fails L x (fun r u c => L.UnitOk H x (run H P L.V x s.1) u r c) δ (run H P L.V x s.1) s.2.2 ∨
          OwnColl H L.E P x s.1 s.2.1 s.2.2) :
    L.KS H :=
  KS_of_split hS (fun x _ P ω r u c => L.UnitOk H x (run H P L.V x ω) u r c) hsplit
    (fun x Ω _ _ P => hSK x Ω P)

end Compiled

end Claims.SamplingGeneric
