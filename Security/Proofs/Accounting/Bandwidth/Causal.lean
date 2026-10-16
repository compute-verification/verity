import Mathlib
import Proofs.Accounting.Bandwidth.Ticked
import Proofs.Accounting.Actual.Transducer
import Definitions.Accounting.Actual.Source
import Models.Accounting.Guarantees.Bandwidth.Causal

/-!
# B3 on a strategy causal from every tick: the chip as a `Ticked.Chip`

What remained of B3 in v2.4 of `note:console/20261012T0645Z-draft-actual-circuit-model`: the chip of a run causal from
every tick as the tick transducer `Ticked.Chip`, so that `Ticked.audit_reads` bounds it. A strategy here is a run for each
store and oracle, `S : Store × Oracle → Actual.Run L`; the chip is the location `X`, and the window is `[t₀, t₀ + T)`.

What the model doesn't state yet is one hypothesis, `Splits`: at each tick of the window, what crosses into the chip is a
fixed function (`Interface.cross`) of what the chip has done since `t₀` and of three replies, each read through its link:
- HBM's reply, of the width the chip's requests fix (`Interface.width`), its bits a function of the store and of what the
  chip did (passive memory: given the chip's requests, a read carries only its values);
- the rest's signal, `βs` bits (its index in the side link's alphabet), a function of the store, the oracle and what the
  chip did;
- the oracle's answers to the queries the chip's acts fix (the oracle as a location the chip queries).
With the chip's state at `t₀` fixed by the store (`hpre`: laid out before the seed) and at most `M` bits, `run_eq` builds
the `Ticked.Chip` of state `M + 1` bits (the content's length, `transducer_bits`) whose tags are any function `τ` of what
the chip did in the window and whose HBM reads are the widths summed (`reads`). HBM's requests carry the chip's
observations so far, and so do its sends to the rest, as `Ticked` allows: neither is counted.

`SecurityProofs.Causal.audit_reads` is then `AuditInteractive`'s bound at `S = M + 1`, `c = βs T`, for a strategy causal
from every tick: the probability that the chip's committed tags pass while `reads + M + 1 + βs T + m b + λ < k b` is at most
`((k − 1)/N)^s + C(N, m) 2^{βs T}/2^{h m} + C(T q, k − m)/2^λ`. `audit_reads_ro` is `AuditInteractiveRO`'s when the rest's
signal ignores the oracle, and `audit_ingress` reads HBM's widths as ingress from HBM's leaves (`ingressFrom`).

`splits_of_links` derives `Splits` when every wire into the chip runs on a port of HBM, of the rest or of the oracle:
the rest's ports from the model's `Alphabet` (their signals at a tick are one of `2^{βs}`, `βs` their capacities summed),
and the other two from the chip's ports' constraints (`Security/Definitions/Accounting/Bandwidth/Causal.lean`):
- `PassiveMemory`: HBM's ports deliver a function of the store and of what the chip did, at most `2^{wd}` signals for
  each such `a`, `wd` the width the chip's requests fix;
- `OracleLocation`: the oracle's ports deliver a function of what the chip did and of the oracle's answers to the
  queries that fixes.
`audit_links` and `audit_links_ro` are B3 from these. `audit_capacity` and `audit_capacity_ro` are the capacity form,
with only `HearsChip` (HBM's ports deliver a function of the store and of what the chip did) in place of
`PassiveMemory`: with HBM's links alphabets it gives passive memory at `βm` bits a tick (`passiveMemory_of_hearsChip`),
and the bound holds at `βm T` in place of `∑ wd`. The four are the guarantees `Bandwidth.Guarantees.B3`, `B3RO`,
`B3Capacity` and `B3CapacityRO` (`Security/Models/Accounting/Guarantees/Bandwidth/Causal.lean`).
-/

namespace Bandwidth

open Pous (Bits pr)
open Accounting Accounting.SecurityProofs.Actual
open scoped ENNReal

namespace Causal

/-- **The chip's links in the window**, at tick `u` after the chip did `a` since the window's start: HBM's reply has
`width u a` bits, bit `j` being `hbm E u a j`; the rest's signal has bit `j` `side E O u a j`; the oracle answers the
queries `ask u a`; and `cross u a` of the three is what crosses into the chip at `u`. -/
structure Interface (L : Type*) (N b h : ℕ) where
  width : ℕ → Acts L → ℕ
  hbm : Store N b → ℕ → Acts L → ℕ → Bool
  side : Store N b → Oracle N b h → ℕ → Acts L → ℕ → Bool
  ask : ℕ → Acts L → List (Fin N × Bits b)
  cross : ℕ → Acts L → Ticked.Obs h → Cross L

namespace Interface

variable {L : Type*} {N b h : ℕ} (I : Interface L N b h)

/-- What the chip observes at `u` after doing `a`: HBM's reply, the rest's `βs`-bit signal and the oracle's answers. -/
def obs (βs : ℕ) (E : Store N b) (O : Oracle N b h) (u : ℕ) (a : Acts L) : Ticked.Obs h :=
  ((List.range (I.width u a)).reverse.map (I.hbm E u a), (List.range βs).reverse.map (I.side E O u a),
    (I.ask u a).map O)

/-- The crossings that the chip's observations `o` (newest first) fix, oldest first, when `F n st cs` is what the chip
does in its first `n` ticks from the state `st` with the crossings `cs`. -/
def crosses {σ : Type*} (t₀ : ℕ) (F : ℕ → σ → List (Cross L) → Acts L) (st : σ) :
    List (Ticked.Obs h) → List (Cross L)
  | [] => []
  | x :: o => crosses t₀ F st o ++ [I.cross (t₀ + o.length) (F o.length st (crosses t₀ F st o)) x]

/-- What the chip has done since the window's start, having observed `o`. -/
def done {σ : Type*} (t₀ : ℕ) (F : ℕ → σ → List (Cross L) → Acts L) (st : σ) (o : List (Ticked.Obs h)) : Acts L :=
  F o.length st (I.crosses t₀ F st o)

/-- **The chip as a `Ticked.Chip`**: each tick it fixes HBM's width and its queries from what it has done, sends HBM its
observations as the request and the rest the same in unary, and commits to `τ` of what it did. -/
noncomputable def chip {S : ℕ} (t₀ : ℕ) (F : ℕ → Bits S → List (Cross L) → Acts L) (τ : Acts L → Fin N → Bits h) :
    Ticked.Chip N b h S where
  key _ o := Encodable.encode o
  width st o := I.width (t₀ + o.length) (I.done t₀ F st o)
  out _ o := false :: List.replicate (Encodable.encode o) true
  ask st o := I.ask (t₀ + o.length) (I.done t₀ F st o)
  tags st o := τ (I.done t₀ F st o)

/-- HBM as `Ticked.Hbm`: the newest request names the chip's observations, from which, and the store, it replies. -/
noncomputable def hbmOf {S : ℕ} (t₀ : ℕ) (F : ℕ → Bits S → List (Cross L) → Acts L) (state : Store N b → Bits S) :
    Ticked.Hbm N b := fun E ks j =>
  match ks with
  | k :: _ => (Encodable.decode k : Option (List (Ticked.Obs h))).elim false fun o =>
    I.hbm E (t₀ + o.length) (I.done t₀ F (state E) o) j
  | [] => false

/-- The rest as `Ticked.Rest`: the newest send names the chip's observations in unary. -/
noncomputable def restOf {S : ℕ} (t₀ : ℕ) (F : ℕ → Bits S → List (Cross L) → Acts L) (state : Store N b → Bits S) :
    Ticked.Rest N b h := fun E O v _ j =>
  (Encodable.decode (v.takeWhile id).length : Option (List (Ticked.Obs h))).elim false fun o =>
    I.side E O (t₀ + o.length) (I.done t₀ F (state E) o) j

theorem hbmOf_key {S : ℕ} (t₀ : ℕ) (F : ℕ → Bits S → List (Cross L) → Acts L) (state : Store N b → Bits S)
    (E : Store N b) (o : List (Ticked.Obs h)) (ks : List ℕ) :
    I.hbmOf t₀ F state E (Encodable.encode o :: ks) = I.hbm E (t₀ + o.length) (I.done t₀ F (state E) o) := by
  funext j
  simp [hbmOf, Encodable.encodek]

theorem restOf_out {S : ℕ} (t₀ : ℕ) (F : ℕ → Bits S → List (Cross L) → Acts L) (state : Store N b → Bits S)
    (E : Store N b) (O : Oracle N b h) (o : List (Ticked.Obs h)) (v : List Bool) (u : ℕ) :
    I.restOf t₀ F state E O ((false :: List.replicate (Encodable.encode o) true).reverse ++ v) u =
      I.side E O (t₀ + o.length) (I.done t₀ F (state E) o) := by
  funext j
  have : ((false :: List.replicate (Encodable.encode o) true).reverse ++ v).takeWhile id =
      List.replicate (Encodable.encode o) true := by
    simp [List.reverse_replicate]
  simp only [restOf, this, List.length_replicate, Encodable.encodek, Option.elim]

/-- **The chip's observations replay the run**: when `F` from the store's state and the run's crossings gives what the
chip did (`hac`), and each crossing is `cross` of what the chip did and observed (`hcr`), the `Ticked` chip's observations
fix the run's crossings tick by tick, and its HBM reads are the widths summed. -/
theorem observe_eq {S : ℕ} (t₀ T βs : ℕ) (F : ℕ → Bits S → List (Cross L) → Acts L) (τ : Acts L → Fin N → Bits h)
    (state : Store N b → Bits S) (E : Store N b) (O : Oracle N b h) (cr : ℕ → Cross L) (ac : ℕ → Acts L)
    (hac : ∀ n ≤ T, F n (state E) ((List.range n).map cr) = ac n)
    (hcr : ∀ i < T, cr i = I.cross (t₀ + i) (ac i) (I.obs βs E O (t₀ + i) (ac i))) :
    ∀ n (o : List (Ticked.Obs h)) (ks : List ℕ) (v : List Bool), o.length + n ≤ T →
      I.crosses t₀ F (state E) o = (List.range o.length).map cr →
      I.crosses t₀ F (state E)
          (Ticked.observe (I.chip t₀ F τ) βs (state E) (I.hbmOf t₀ F state) (I.restOf t₀ F state) E O n o ks v) =
        (List.range (Ticked.observe (I.chip t₀ F τ) βs (state E) (I.hbmOf t₀ F state) (I.restOf t₀ F state) E O n o
          ks v).length).map cr ∧
      (Ticked.observe (I.chip t₀ F τ) βs (state E) (I.hbmOf t₀ F state) (I.restOf t₀ F state) E O n o ks v).length =
        o.length + n ∧
      Ticked.hbmBits
          (Ticked.observe (I.chip t₀ F τ) βs (state E) (I.hbmOf t₀ F state) (I.restOf t₀ F state) E O n o ks v) =
        Ticked.hbmBits o + ∑ i ∈ Finset.range n, I.width (t₀ + (o.length + i)) (ac (o.length + i)) := by
  intro n
  induction n with
  | zero =>
    intro o ks v _ h
    simp [Ticked.observe, h]
  | succ n ih =>
    intro o ks v hn h
    have hd : I.done t₀ F (state E) o = ac o.length := by
      rw [done, h]
      exact hac _ (by omega)
    have hx : ((List.range ((I.chip t₀ F τ).width (state E) o)).reverse.map
          (I.hbmOf t₀ F state E ((I.chip t₀ F τ).key (state E) o :: ks)),
        (List.range βs).reverse.map
          (I.restOf t₀ F state E O (((I.chip t₀ F τ).out (state E) o).reverse ++ v) o.length),
        ((I.chip t₀ F τ).ask (state E) o).map O) = I.obs βs E O (t₀ + o.length) (ac o.length) := by
      simp only [chip, hbmOf_key, restOf_out, obs, hd]
    have h' : I.crosses t₀ F (state E) (I.obs βs E O (t₀ + o.length) (ac o.length) :: o) =
        (List.range (I.obs βs E O (t₀ + o.length) (ac o.length) :: o).length).map cr := by
      have hd' : F o.length (state E) (I.crosses t₀ F (state E) o) = ac o.length := hd
      rw [crosses, hd', h, List.length_cons, List.range_succ, List.map_append, List.map_singleton,
        hcr _ (by omega)]
    simp only [Ticked.observe]
    rw [hx]
    obtain ⟨h1, h2, h3⟩ := ih _ _ _ (by simp only [List.length_cons]; omega) h'
    refine ⟨h1, by rw [h2, List.length_cons]; ring, ?_⟩
    rw [h3, Finset.sum_range_succ']
    simp only [Ticked.hbmBits, List.map_cons, List.sum_cons, obs, List.length_map, List.length_reverse,
      List.length_range, List.length_cons, add_zero]
    have e : ∀ i, o.length + 1 + i = o.length + (i + 1) := fun i => by ring
    simp only [e]
    ring

end Interface

/-- **The chip's links split what crosses into it** over the window `[t₀, t₀ + T)`: at each tick, what crosses into `X`
is `I.cross` of what `X` has done since `t₀` and of what it observes through `I`. -/
def Splits {L : Type*} {N b h : ℕ} (S : Store N b × Oracle N b h → Actual.Run L) (X : Finset L) (t₀ T βs : ℕ)
    (I : Interface L N b h) : Prop :=
  ∀ E O, ∀ i < T, (S (E, O)).crossing (↑X)ᶜ ↑X {t₀ + i} =
    I.cross (t₀ + i) ((S (E, O)).acts ↑X (Set.Ico t₀ (t₀ + i)))
      (I.obs βs E O (t₀ + i) ((S (E, O)).acts ↑X (Set.Ico t₀ (t₀ + i))))

/-- The bits HBM delivers to the chip in the window: the widths its requests fix, tick by tick. -/
def reads {L : Type*} {N b h : ℕ} (I : Interface L N b h) (S : Store N b × Oracle N b h → Actual.Run L) (X : Finset L)
    (t₀ T : ℕ) (r : Store N b × Oracle N b h) : ℕ :=
  ∑ i ∈ Finset.range T, I.width (t₀ + i) ((S r).acts ↑X (Set.Ico t₀ (t₀ + i)))

end Causal

namespace SecurityProofs.Causal

open Bandwidth.Causal

variable {L : Type*} [LinearOrder L] {N b h : ℕ}

/-- **The chip of a strategy causal from every tick is a `Ticked.Chip`**: with its state at `t₀` at most `M` bits and
fixed by the store, and its links splitting what crosses into it, there are a chip of state `M + 1` bits, its state from
the store, HBM and the rest such that, for every store and oracle, the chip's tags are `τ` of what `X` did in the window
and its HBM reads are `reads`. Its queries are the interface's, and the rest ignores the oracle when the side does. -/
theorem run_eq (S : Store N b × Oracle N b h → Actual.Run L) (hc : CausalFrom S) (X : Finset L) (t₀ T M βs : ℕ)
    (hM : ∀ r, (S r).mem X t₀ ≤ M) (hpre : ∀ E O O', (S (E, O)).content X t₀ = (S (E, O')).content X t₀)
    (I : Interface L N b h) (hsplit : Splits S X t₀ T βs I) (τ : Acts L → Fin N → Bits h) :
    ∃ (C : Ticked.Chip N b h (M + 1)) (state : Store N b → Bits (M + 1)) (hbm : Ticked.Hbm N b)
      (rest : Ticked.Rest N b h),
      (∀ st o, ∃ u a, C.ask st o = I.ask u a) ∧
      (∀ E O, C.tags (state E) (Ticked.observe C βs (state E) hbm rest E O T [] [] []) =
          τ ((S (E, O)).acts ↑X (Set.Ico t₀ (t₀ + T))) ∧
        Ticked.hbmBits (Ticked.observe C βs (state E) hbm rest E O T [] [] []) = reads I S X t₀ T (E, O)) ∧
      ((∀ E O O' u a, I.side E O u a = I.side E O' u a) → ∀ E O O', rest E O = rest E O') := by
  obtain ⟨F, hF⟩ := transducer_bits hc X t₀ (fun r u => (S r).crossing (↑X)ᶜ ↑X {u}) (fun _ _ _ h => h) M hM
  let O₀ : Oracle N b h := fun _ _ => false
  let state : Store N b → Bits (M + 1) := fun E => pad M ((S (E, O₀)).content X t₀)
  refine ⟨I.chip t₀ F τ, state, I.hbmOf t₀ F state, I.restOf t₀ F state, fun st o => ⟨_, _, rfl⟩, fun E O => ?_,
    fun hside E O O' => ?_⟩
  · have hac : ∀ n ≤ T, F n (state E) ((List.range n).map fun i => (S (E, O)).crossing (↑X)ᶜ ↑X {t₀ + i}) =
        (S (E, O)).acts ↑X (Set.Ico t₀ (t₀ + n)) := by
      intro n _
      rw [hF (E, O) n]
      simp only [state, hpre E O₀ O]
    obtain ⟨h1, h2, h3⟩ := I.observe_eq t₀ T βs F τ state E O _ _ hac (hsplit E O) T [] [] [] (by simp) rfl
    simp only [List.length_nil, zero_add] at h2 h3
    refine ⟨?_, ?_⟩
    · change τ (F _ (state E) (I.crosses t₀ F (state E) _)) = _
      rw [h1, h2, hac T le_rfl]
    · rw [h3]
      simp [Ticked.hbmBits, reads]
  · funext v u j
    simp only [Interface.restOf, hside E O O']

/-- **B3 on a strategy causal from every tick, in the reads form**: with the chip's state at `t₀` at most `M` bits and
fixed by the store, its links splitting what crosses into it (`Splits`: HBM's reply at the width the chip's acts fix, the
rest's signal `βs` bits, at most `qt` oracle queries a tick), and the chip committing to `τ` of what it did in the window,
the probability over the store, the oracle and the sample that the tags pass while `reads + M + 1 + βs T + m b + λ < k b`
is at most `AuditInteractive`'s bound at `c = βs T`. -/
theorem audit_reads (S : Store N b × Oracle N b h → Actual.Run L) (hc : CausalFrom S) (X : Finset L) (t₀ T M βs qt : ℕ)
    (hM : ∀ r, (S r).mem X t₀ ≤ M) (hpre : ∀ E O O', (S (E, O)).content X t₀ = (S (E, O')).content X t₀)
    (I : Interface L N b h) (hq : ∀ u a, (I.ask u a).length ≤ qt) (hsplit : Splits S X t₀ T βs I)
    (τ : Acts L → Fin N → Bits h) (s k m lam : ℕ) :
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, τ ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + T))) (x.2.2 i) = x.2.1 (x.2.2 i, x.1 (x.2.2 i))) ∧
          reads I S X t₀ T (x.1, x.2.1) + (M + 1) + βs * T + m * b + lam < k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (N.choose m : ℝ≥0∞) * 2 ^ (T * βs) / 2 ^ (h * m)
        + (((T * qt).choose (k - m) : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  obtain ⟨C, state, hbm, rest, hask, heq, -⟩ := run_eq S hc X t₀ T M βs hM hpre I hsplit τ
  have hqC : ∀ st o, (C.ask st o).length ≤ qt := fun st o => by
    obtain ⟨u, a, e⟩ := hask st o
    rw [e]
    exact hq u a
  refine (pr_mono fun x hx => ?_).trans (Bandwidth.SecurityProofs.Ticked.audit_reads C βs qt T hqC state hbm rest s k m lam)
  obtain ⟨ht, hr⟩ := heq x.1 x.2.1
  exact ⟨fun i => by rw [ht]; exact hx.1 i, by rw [hr]; exact hx.2⟩

/-- **The reads form when the rest's signal ignores the oracle**: `AuditInteractiveRO`'s bound, with no relayed-tag
term. -/
theorem audit_reads_ro (S : Store N b × Oracle N b h → Actual.Run L) (hc : CausalFrom S) (X : Finset L)
    (t₀ T M βs qt : ℕ) (hM : ∀ r, (S r).mem X t₀ ≤ M)
    (hpre : ∀ E O O', (S (E, O)).content X t₀ = (S (E, O')).content X t₀) (I : Interface L N b h)
    (hq : ∀ u a, (I.ask u a).length ≤ qt) (hside : ∀ E O O' u a, I.side E O u a = I.side E O' u a)
    (hsplit : Splits S X t₀ T βs I) (τ : Acts L → Fin N → Bits h) (s k lam : ℕ) :
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, τ ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + T))) (x.2.2 i) = x.2.1 (x.2.2 i, x.1 (x.2.2 i))) ∧
          reads I S X t₀ T (x.1, x.2.1) + (M + 1) + βs * T + lam < k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (((T * qt).choose k : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  obtain ⟨C, state, hbm, rest, hask, heq, hrest⟩ := run_eq S hc X t₀ T M βs hM hpre I hsplit τ
  have hqC : ∀ st o, (C.ask st o).length ≤ qt := fun st o => by
    obtain ⟨u, a, e⟩ := hask st o
    rw [e]
    exact hq u a
  refine (pr_mono fun x hx => ?_).trans
    (Bandwidth.SecurityProofs.Ticked.audit_reads_ro C βs qt T hqC state hbm rest (hrest hside) s k lam)
  obtain ⟨ht, hr⟩ := heq x.1 x.2.1
  exact ⟨fun i => by rw [ht]; exact hx.1 i, by rw [hr]; exact hx.2⟩

/-- **The reads form with HBM's widths as ingress**: when each tick's width is at most the chip's ingress from HBM's
leaves `H` at that tick (`ingressFrom`), the bound holds with that ingress summed over the window in place of `reads`. -/
theorem audit_ingress (S : Store N b × Oracle N b h → Actual.Run L) (hc : CausalFrom S) (X H : Finset L)
    (t₀ T M βs qt : ℕ) (hM : ∀ r, (S r).mem X t₀ ≤ M)
    (hpre : ∀ E O O', (S (E, O)).content X t₀ = (S (E, O')).content X t₀) (I : Interface L N b h)
    (hq : ∀ u a, (I.ask u a).length ≤ qt) (hsplit : Splits S X t₀ T βs I)
    (hw : ∀ r, ∀ i < T, I.width (t₀ + i) ((S r).acts ↑X (Set.Ico t₀ (t₀ + i))) ≤
      (S r).A.ingressFrom X H {t₀ + i})
    (τ : Acts L → Fin N → Bits h) (s k m lam : ℕ) :
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, τ ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + T))) (x.2.2 i) = x.2.1 (x.2.2 i, x.1 (x.2.2 i))) ∧
          (∑ i ∈ Finset.range T, (S (x.1, x.2.1)).A.ingressFrom X H {t₀ + i}) + (M + 1) + βs * T + m * b + lam <
            k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (N.choose m : ℝ≥0∞) * 2 ^ (T * βs) / 2 ^ (h * m)
        + (((T * qt).choose (k - m) : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  refine (pr_mono fun x hx => ⟨hx.1, ?_⟩).trans (audit_reads S hc X t₀ T M βs qt hM hpre I hq hsplit τ s k m lam)
  have : reads I S X t₀ T (x.1, x.2.1) ≤ ∑ i ∈ Finset.range T, (S (x.1, x.2.1)).A.ingressFrom X H {t₀ + i} :=
    Finset.sum_le_sum fun i hi => hw _ i (Finset.mem_range.1 hi)
  have := hx.2
  omega

/-- A set of at most `2^w` elements injects into `Bits w`. -/
theorem exists_bits {α : Type*} (s : Set α) (w : ℕ) (hs : s.encard ≤ ((2 ^ w : ℕ) : ℕ∞)) :
    ∃ f : α → Bits w, Set.InjOn f s := by
  obtain ⟨f, -, hf⟩ := (Set.finite_of_encard_le_coe hs).exists_injOn_of_encard_le (t := (Set.univ : Set (Bits w)))
    (by rw [Set.encard_univ, ENat.card_eq_coe_fintype_card]; simpa [Pous.Bits] using hs)
  exact ⟨f, hf⟩

/-- `w` bits read out as a list, highest index first, are the bits. -/
theorem bits_inj {w : ℕ} {v v' : Bits w}
    (h : (List.range w).reverse.map (fun j => if hj : j < w then v ⟨j, hj⟩ else false) =
      (List.range w).reverse.map (fun j => if hj : j < w then v' ⟨j, hj⟩ else false)) : v = v' := by
  funext ⟨j, hj⟩
  have := List.map_inj_left.1 h j (by simp [hj])
  simpa [hj] using this

omit [LinearOrder L] in
/-- **The links split what crosses into the chip**: when every wire into `X` runs on a port of HBM, the rest or the
oracle, the links are alphabets, HBM's ports are passive memory of widths `wd` and the oracle's ports a location asked
`Q`, then `Splits` holds for an interface of widths `wd`, queries `Q` and `βs` the rest's ports' capacities summed, whose
rest's signal ignores the oracle when the rest's ports' signals do. -/
theorem splits_of_links (S : Store N b × Oracle N b h → Actual.Run L) {K : Type*} (β : K → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K) (hα : Alphabet S β link) (X : Finset L) (t₀ T : ℕ)
    (hbmPorts sidePorts orPorts : List (K × L))
    (hports : ∀ r g, ∀ a ∈ (S r).A.reads g, (S r).A.x a ∉ X → (S r).A.x g ∈ X →
      (link r g a, (S r).A.x g) ∈ hbmPorts ++ sidePorts ++ orPorts)
    (wd : ℕ → Acts L → ℕ) (hpm : PassiveMemory S link X t₀ T hbmPorts wd)
    (Q : ℕ → Acts L → List (Fin N × Bits b)) (hor : OracleLocation S link X t₀ T orPorts Q) :
    ∃ I : Interface L N b h, I.width = wd ∧ I.ask = Q ∧
      Splits S X t₀ T (sidePorts.map fun p => β p.1).sum I ∧
      ((∀ E O O' u, portSignals S link sidePorts (E, O) u = portSignals S link sidePorts (E, O') u) →
        ∀ E O O' u a, I.side E O u a = I.side E O' u a) := by
  obtain ⟨G, hG, hGc⟩ := hpm
  obtain ⟨R, hR⟩ := hor
  set βs := (sidePorts.map fun p => β p.1).sum
  choose fH hfH using fun u a => exists_bits _ _ (hGc u a)
  choose fS hfS using fun u => exists_bits _ βs (portSignals_encard_le S β link hα u sidePorts)
  let I₀ : Interface L N b h :=
    { width := wd
      hbm := fun E u a j => if hj : j < wd u a then fH u a (G E u a) ⟨j, hj⟩ else false
      side := fun E O u _ j => if hj : j < βs then fS u (portSignals S link sidePorts (E, O) u) ⟨j, hj⟩ else false
      ask := Q
      cross := fun _ _ _ => ∅ }
  let key : ℕ → Store N b × Oracle N b h → Acts L × Ticked.Obs h := fun u r =>
    ((S r).acts ↑X (Set.Ico t₀ u), I₀.obs βs r.1 r.2 u ((S r).acts ↑X (Set.Ico t₀ u)))
  let cross : ℕ → Acts L → Ticked.Obs h → Cross L := fun u a x =>
    Function.extend (key u) (fun r => (S r).crossing (↑X)ᶜ ↑X {u}) (fun _ => ∅) (a, x)
  refine ⟨{ I₀ with cross := cross }, rfl, rfl, fun E O i hi => ?_, fun h0 E O O' u a => ?_⟩
  swap
  · funext j
    simp only [I₀, h0 E O O' u]
  have hf : Function.FactorsThrough (fun r => (S r).crossing (↑X)ᶜ ↑X {t₀ + i}) (key (t₀ + i)) := by
    rintro ⟨E₁, O₁⟩ ⟨E₂, O₂⟩ hk
    have hG1 := hG E₁ O₁ i hi
    have hG2 := hG E₂ O₂ i hi
    have hR1 := hR E₁ O₁ i hi
    have hR2 := hR E₂ O₂ i hi
    simp only [key] at hk
    generalize (S (E₁, O₁)).acts ↑X (Set.Ico t₀ (t₀ + i)) = a at hk hG1 hR1
    generalize (S (E₂, O₂)).acts ↑X (Set.Ico t₀ (t₀ + i)) = a₂ at hk hG2 hR2
    obtain ⟨rfl, hx⟩ := Prod.mk.inj hk
    simp only [Interface.obs, Prod.mk.injEq, I₀] at hx
    obtain ⟨h1, h2, h3⟩ := hx
    have hH : G E₁ (t₀ + i) a = G E₂ (t₀ + i) a :=
      hfH _ _ ⟨E₁, rfl⟩ ⟨E₂, rfl⟩ (bits_inj h1)
    have hS : portSignals S link sidePorts (E₁, O₁) (t₀ + i) = portSignals S link sidePorts (E₂, O₂) (t₀ + i) :=
      hfS _ ⟨_, rfl⟩ ⟨_, rfl⟩ (bits_inj h2)
    have hall : portSignals S link (hbmPorts ++ sidePorts ++ orPorts) (E₁, O₁) (t₀ + i) =
        portSignals S link (hbmPorts ++ sidePorts ++ orPorts) (E₂, O₂) (t₀ + i) := by
      have e : ∀ r, portSignals S link (hbmPorts ++ sidePorts ++ orPorts) r (t₀ + i) =
          portSignals S link hbmPorts r (t₀ + i) ++ portSignals S link sidePorts r (t₀ + i) ++
            portSignals S link orPorts r (t₀ + i) := fun r => by
        simp only [portSignals, List.map_append]
      rw [e, e, hG1, hG2, hR1, hR2, hH, hS, h3]
    exact crossing_of_ports S link X _ hports (t₀ + i) _ _ hall
  show (S (E, O)).crossing (↑X)ᶜ ↑X {t₀ + i} =
    Function.extend (key (t₀ + i)) (fun r => (S r).crossing (↑X)ᶜ ↑X {t₀ + i}) (fun _ => ∅) (key (t₀ + i) (E, O))
  exact (hf.extend_apply (fun _ => ∅) (E, O)).symm

/-- **B3 on a strategy causal from every tick, from the links**: with the chip's state at `t₀` at most `M` bits and
fixed by the store, every wire into it on a port of HBM, the rest or the oracle, the links alphabets, HBM passive memory
of widths `wd` and the oracle a location asked at most `qt` queries a tick, the probability that the chip's tags pass
while `∑ wd + M + 1 + βs T + m b + λ < k b` is at most `AuditInteractive`'s bound at `c = βs T`, `βs` the rest's ports'
capacities summed. -/
theorem audit_links (S : Store N b × Oracle N b h → Actual.Run L) (hc : CausalFrom S) {K : Type*} (β : K → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K) (hα : Alphabet S β link) (X : Finset L) (t₀ T M qt : ℕ)
    (hM : ∀ r, (S r).mem X t₀ ≤ M) (hpre : ∀ E O O', (S (E, O)).content X t₀ = (S (E, O')).content X t₀)
    (hbmPorts sidePorts orPorts : List (K × L))
    (hports : ∀ r g, ∀ a ∈ (S r).A.reads g, (S r).A.x a ∉ X → (S r).A.x g ∈ X →
      (link r g a, (S r).A.x g) ∈ hbmPorts ++ sidePorts ++ orPorts)
    (wd : ℕ → Acts L → ℕ) (hpm : PassiveMemory S link X t₀ T hbmPorts wd)
    (Q : ℕ → Acts L → List (Fin N × Bits b)) (hq : ∀ u a, (Q u a).length ≤ qt)
    (hor : OracleLocation S link X t₀ T orPorts Q) (τ : Acts L → Fin N → Bits h) (s k m lam : ℕ) :
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, τ ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + T))) (x.2.2 i) = x.2.1 (x.2.2 i, x.1 (x.2.2 i))) ∧
          (∑ i ∈ Finset.range T, wd (t₀ + i) ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + i)))) + (M + 1) +
            (sidePorts.map fun p => β p.1).sum * T + m * b + lam < k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s +
          (N.choose m : ℝ≥0∞) * 2 ^ (T * (sidePorts.map fun p => β p.1).sum) / 2 ^ (h * m)
        + (((T * qt).choose (k - m) : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  obtain ⟨I, hw, hQ, hs, -⟩ := splits_of_links S β link hα X t₀ T hbmPorts sidePorts orPorts hports wd hpm Q hor
  subst hw hQ
  exact audit_reads S hc X t₀ T M _ qt hM hpre I hq hs τ s k m lam

/-- **B3 from the links at `e = 0`**: when the rest's ports' signals ignore the oracle too, `AuditInteractiveRO`'s bound,
with no relayed-tag term. -/
theorem audit_links_ro (S : Store N b × Oracle N b h → Actual.Run L) (hc : CausalFrom S) {K : Type*} (β : K → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K) (hα : Alphabet S β link) (X : Finset L) (t₀ T M qt : ℕ)
    (hM : ∀ r, (S r).mem X t₀ ≤ M) (hpre : ∀ E O O', (S (E, O)).content X t₀ = (S (E, O')).content X t₀)
    (hbmPorts sidePorts orPorts : List (K × L))
    (hports : ∀ r g, ∀ a ∈ (S r).A.reads g, (S r).A.x a ∉ X → (S r).A.x g ∈ X →
      (link r g a, (S r).A.x g) ∈ hbmPorts ++ sidePorts ++ orPorts)
    (wd : ℕ → Acts L → ℕ) (hpm : PassiveMemory S link X t₀ T hbmPorts wd)
    (Q : ℕ → Acts L → List (Fin N × Bits b)) (hq : ∀ u a, (Q u a).length ≤ qt)
    (hor : OracleLocation S link X t₀ T orPorts Q)
    (h0 : ∀ E O O' u, portSignals S link sidePorts (E, O) u = portSignals S link sidePorts (E, O') u)
    (τ : Acts L → Fin N → Bits h) (s k lam : ℕ) :
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, τ ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + T))) (x.2.2 i) = x.2.1 (x.2.2 i, x.1 (x.2.2 i))) ∧
          (∑ i ∈ Finset.range T, wd (t₀ + i) ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + i)))) + (M + 1) +
            (sidePorts.map fun p => β p.1).sum * T + lam < k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (((T * qt).choose k : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  obtain ⟨I, hw, hQ, hs, hside⟩ :=
    splits_of_links S β link hα X t₀ T hbmPorts sidePorts orPorts hports wd hpm Q hor
  subst hw hQ
  exact audit_reads_ro S hc X t₀ T M _ qt hM hpre I hq (hside h0) hs τ s k lam

omit [LinearOrder L] in
/-- **The capacity form's passive memory**: when HBM hears only the chip and its links are alphabets, HBM's ports are
passive memory of width `βm` a tick, `βm` their capacities summed. -/
theorem passiveMemory_of_hearsChip {ε : Type*} (S : ε × Oracle N b h → Actual.Run L) {K : Type*} (β : K → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K) (hα : Alphabet S β link) (X : Finset L) (t₀ T : ℕ)
    (hbmPorts : List (K × L)) (hhc : HearsChip S link X t₀ T hbmPorts) :
    PassiveMemory S link X t₀ T hbmPorts fun _ _ => (hbmPorts.map fun p => β p.1).sum := by
  classical
  obtain ⟨G, hG⟩ := hhc
  let O₀ : Oracle N b h := fun _ _ => false
  refine ⟨fun E u a => if h : ∃ O, (S (E, O)).acts ↑X (Set.Ico t₀ u) = a then
      portSignals S link hbmPorts (E, h.choose) u else portSignals S link hbmPorts (E, O₀) u,
    fun E O i hi => ?_, fun u a => ?_⟩
  · dsimp only
    split_ifs with h
    · rw [hG E O i hi, hG E h.choose i hi, h.choose_spec]
    · exact absurd ⟨O, rfl⟩ h
  · refine (Set.encard_le_encard ?_).trans (portSignals_encard_le S β link hα u hbmPorts)
    rintro _ ⟨E, rfl⟩
    dsimp only
    split_ifs
    · exact ⟨_, rfl⟩
    · exact ⟨_, rfl⟩

/-- **B3 from the links, in the capacity form**: as `audit_links`, with HBM hearing only the chip in place of passive
memory, and every tick's HBM signal counted at `βm` bits, its ports' capacities summed. -/
theorem audit_capacity (S : Store N b × Oracle N b h → Actual.Run L) (hc : CausalFrom S) {K : Type*} (β : K → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K) (hα : Alphabet S β link) (X : Finset L) (t₀ T M qt : ℕ)
    (hM : ∀ r, (S r).mem X t₀ ≤ M) (hpre : ∀ E O O', (S (E, O)).content X t₀ = (S (E, O')).content X t₀)
    (hbmPorts sidePorts orPorts : List (K × L))
    (hports : ∀ r g, ∀ a ∈ (S r).A.reads g, (S r).A.x a ∉ X → (S r).A.x g ∈ X →
      (link r g a, (S r).A.x g) ∈ hbmPorts ++ sidePorts ++ orPorts)
    (hhc : HearsChip S link X t₀ T hbmPorts)
    (Q : ℕ → Acts L → List (Fin N × Bits b)) (hq : ∀ u a, (Q u a).length ≤ qt)
    (hor : OracleLocation S link X t₀ T orPorts Q) (τ : Acts L → Fin N → Bits h) (s k m lam : ℕ) :
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, τ ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + T))) (x.2.2 i) = x.2.1 (x.2.2 i, x.1 (x.2.2 i))) ∧
          (hbmPorts.map fun p => β p.1).sum * T + (M + 1) + (sidePorts.map fun p => β p.1).sum * T + m * b + lam <
            k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s +
          (N.choose m : ℝ≥0∞) * 2 ^ (T * (sidePorts.map fun p => β p.1).sum) / 2 ^ (h * m)
        + (((T * qt).choose (k - m) : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  refine (pr_mono fun x hx => ⟨hx.1, ?_⟩).trans (audit_links S hc β link hα X t₀ T M qt hM hpre hbmPorts sidePorts
    orPorts hports _ (passiveMemory_of_hearsChip S β link hα X t₀ T hbmPorts hhc) Q hq hor τ s k m lam)
  simp only [Finset.sum_const, Finset.card_range, smul_eq_mul]
  have := hx.2
  nlinarith

/-- **B3 from the links, in the capacity form at `e = 0`**: as `audit_links_ro`, with HBM hearing only the chip in place
of passive memory, and every tick's HBM signal counted at `βm` bits, its ports' capacities summed. -/
theorem audit_capacity_ro (S : Store N b × Oracle N b h → Actual.Run L) (hc : CausalFrom S) {K : Type*} (β : K → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → K) (hα : Alphabet S β link) (X : Finset L) (t₀ T M qt : ℕ)
    (hM : ∀ r, (S r).mem X t₀ ≤ M) (hpre : ∀ E O O', (S (E, O)).content X t₀ = (S (E, O')).content X t₀)
    (hbmPorts sidePorts orPorts : List (K × L))
    (hports : ∀ r g, ∀ a ∈ (S r).A.reads g, (S r).A.x a ∉ X → (S r).A.x g ∈ X →
      (link r g a, (S r).A.x g) ∈ hbmPorts ++ sidePorts ++ orPorts)
    (hhc : HearsChip S link X t₀ T hbmPorts)
    (Q : ℕ → Acts L → List (Fin N × Bits b)) (hq : ∀ u a, (Q u a).length ≤ qt)
    (hor : OracleLocation S link X t₀ T orPorts Q)
    (h0 : ∀ E O O' u, portSignals S link sidePorts (E, O) u = portSignals S link sidePorts (E, O') u)
    (τ : Acts L → Fin N → Bits h) (s k lam : ℕ) :
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, τ ((S (x.1, x.2.1)).acts ↑X (Set.Ico t₀ (t₀ + T))) (x.2.2 i) = x.2.1 (x.2.2 i, x.1 (x.2.2 i))) ∧
          (hbmPorts.map fun p => β p.1).sum * T + (M + 1) + (sidePorts.map fun p => β p.1).sum * T + lam < k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (((T * qt).choose k : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  refine (pr_mono fun x hx => ⟨hx.1, ?_⟩).trans (audit_links_ro S hc β link hα X t₀ T M qt hM hpre hbmPorts
    sidePorts orPorts hports _ (passiveMemory_of_hearsChip S β link hα X t₀ T hbmPorts hhc) Q hq hor h0 τ s k lam)
  simp only [Finset.sum_const, Finset.card_range, smul_eq_mul]
  have := hx.2
  nlinarith

end SecurityProofs.Causal

namespace SecurityProofs

/-- **B3, the bandwidth audit on a strategy's runs, from the links** (`Bandwidth.Guarantees.B3`). -/
theorem B3 : Guarantees.B3 := by
  intro L K _ N b h S hc β link hα X t₀ T M qt hM hpre hbmPorts sidePorts orPorts hports wd hpm Q hq hor τ s k m lam
  exact Causal.audit_links S hc β link hα X t₀ T M qt hM hpre hbmPorts sidePorts orPorts hports wd hpm Q hq hor τ s k m
    lam

/-- **B3 at `e = 0`** (`Bandwidth.Guarantees.B3RO`). -/
theorem B3RO : Guarantees.B3RO := by
  intro L K _ N b h S hc β link hα X t₀ T M qt hM hpre hbmPorts sidePorts orPorts hports wd hpm Q hq hor h0 τ s k lam
  exact Causal.audit_links_ro S hc β link hα X t₀ T M qt hM hpre hbmPorts sidePorts orPorts hports wd hpm Q hq hor h0
    τ s k lam

/-- **B3 in the capacity form** (`Bandwidth.Guarantees.B3Capacity`). -/
theorem B3Capacity : Guarantees.B3Capacity := by
  intro L K _ N b h S hc β link hα X t₀ T M qt hM hpre hbmPorts sidePorts orPorts hports hhc Q hq hor τ s k m lam
  exact Causal.audit_capacity S hc β link hα X t₀ T M qt hM hpre hbmPorts sidePorts orPorts hports hhc Q hq hor τ s k
    m lam

/-- **B3 in the capacity form at `e = 0`** (`Bandwidth.Guarantees.B3CapacityRO`). -/
theorem B3CapacityRO : Guarantees.B3CapacityRO := by
  intro L K _ N b h S hc β link hα X t₀ T M qt hM hpre hbmPorts sidePorts orPorts hports hhc Q hq hor h0 τ s k lam
  exact Causal.audit_capacity_ro S hc β link hα X t₀ T M qt hM hpre hbmPorts sidePorts orPorts hports hhc Q hq hor h0
    τ s k lam

end SecurityProofs

end Bandwidth
