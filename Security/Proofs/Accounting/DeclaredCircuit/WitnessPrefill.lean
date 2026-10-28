import Mathlib
import Proofs.Accounting.DeclaredCircuit.Witness

/-!
# An honest run with a prefill meets every hypothesis of `DeclaredCircuitRequirement`

`Witness.lean`'s run decodes one token a step, so at most two pages of a sequence, layer and plane are incomplete when
read, and its page-parity groups meet `OneBlockEach`. A served session prefills: a chunk completes several pages at once,
and the next step reads them while their coins are out, two of one parity among them, which page-parity groups don't
cover. This run is a chunked prefill followed by the first decode step, and its groups are pages (`kvGrp`: request,
layer `0`, plane `0`, page), which meet `OneBlockEach` on any values, at every step of any schedule (`oneBlockEach_any`).
It meets every hypothesis of `G` but the world's (`meets`), so `G` bounds it (`applies`).

* **The KV pool** (`wiring`, `placement`): request `r = j + P·w` (one a pass) has `2N + 7` tokens of 16,384 bits, two to
  a 32,768-bit page. Its prefill's first chunk produces tokens `0 … 2N−1` (pages `0 … N−1`) at step `4r`, the second
  chunk tokens `2N … 2N+6` (pages `N`, `N+1`, `N+2`, and the first token of page `N+3`) at step `4r + 1`, and the first
  decode step, `4r + 2`, reads every token (its read gate). Pass `j` of window `w` is bound to that step (`passBind`).
* **Deferred residency** (`kvLayout`): page `k` of request `r` is block `r·(N+4) + k`, and its coin is back two steps
  after the step that completes it (`lag`): the first chunk's pages are first read encoded at `4r + 2`, the second
  chunk's three complete pages at `4r + 3`, and page `N+3`, which the decode step's token fills, at `4r + 4`. So the
  decode step reads pages `0 … N−1` complete and credited, three complete pages with their coins out, and a page with
  one token.
* **Page groups** (`kvGrp`): every page is its own group, so each costs `G`'s credit a page while it is incomplete. The
  run's credit is still exactly its `N` tiles' bits a pass (`credit_eq`): the held term, `N·32768`, is the larger, and
  the other term is `N·32768 − 16384` (`other_eq`), where page-parity groups would have made it `N·32768 + 49152`.
* **The store** (`kvStore`): the memory map writes each page in the record of the step before its first encoded read and
  frees a request's pages in the record of step `4r + 3`. One append a page (`T = 1 + n·P·(N+4)`, `M = 1`): page block
  `b` is append `1 + b` (`kvAddr`), the pass's layout puts its request's pages `0 … N−1` at its `N` tiles, and a page
  joins the window of its request (`joined`), so the lifetime cap holds at every `L ≥ 1`.
* **The devices and the gate model** are `Witness.lean`'s: the honest pass device, one grinding candidate, a `0`-bit
  summary, no carrier, and the empty `Wc`.
* **At the 30 s life** (`L = max(2, ⌊30/T⌋)` windows of `T` seconds, `L = 30` at 1 s, `L = 2` for a prover that defers
  residency at windows of 15 to 60 s) both of `G`'s thresholds are positive for 8B and for 70B TP8 (`life30_8b`,
  `life30_70b`).
-/

namespace Accounting.SecurityProofs.DeclaredCircuitWitnessPrefill

open Pous (pr)
open scoped ENNReal
open Accounting.SecurityProofs.DeclaredCircuitWitness (dec_div dec_mod dec_le dec_inj blk_lt tw tw_eq req req_lt
  disjoint_ticks honestSession hashBudget circuitCarriers restricted property sum_ge gamma_bound)

/-! ## The KV pool -/

/-- A value of the pool: token `i` of request `r`'s KV (`kv r i`), or request `r`'s read gate (`att r`). -/
inductive Val : Type where
  | kv : ℕ → ℕ → Val
  | att : ℕ → Val
  deriving DecidableEq

/-- The values the run produces: `2N + 7` tokens and a read gate for each of the `n·P` requests. -/
def ran (n P N : ℕ) : Finset Val :=
  ((Finset.range (n * P) ×ˢ Finset.range (2 * N + 7)).image fun p => Val.kv p.1 p.2) ∪
    (Finset.range (n * P)).image Val.att

theorem kv_mem_ran {n P N r i : ℕ} : Val.kv r i ∈ ran n P N ↔ r < n * P ∧ i < 2 * N + 7 := by
  simp [ran]

theorem att_mem_ran {n P N r : ℕ} : Val.att r ∈ ran n P N ↔ r < n * P := by
  simp [ran]

/-- A gate's operands: the read gate of request `r` reads its `2N + 7` tokens. -/
def ops (N : ℕ) : Val → Finset Val
  | .kv _ _ => ∅
  | .att r => (Finset.range (2 * N + 7)).image (Val.kv r)

theorem mem_ops {N : ℕ} {v g : Val} :
    v ∈ ops N g ↔ ∃ r i, g = Val.att r ∧ v = Val.kv r i ∧ i < 2 * N + 7 := by
  cases g with
  | kv a b =>
    constructor
    · intro h
      simp [ops] at h
    · rintro ⟨r, i, h, -, -⟩
      cases h
  | att r =>
    simp only [ops, Finset.mem_image, Finset.mem_range]
    constructor
    · rintro ⟨i, hi, rfl⟩
      exact ⟨r, i, rfl, rfl, hi⟩
    · rintro ⟨r', i, h, rfl, hi⟩
      cases h
      exact ⟨i, hi, rfl⟩

/-- The pool's wiring: 16,384-bit values, nothing registered. -/
def wiring (N : ℕ) : Requirement.Wiring Val where
  width _ := tw
  registered := ∅
  inputs := ∅
  ops := ops N
  uses _ := ops N

/-- The prefill's chunk of token `i`: tokens `0 … 2N−1` in the first, the rest in the second. -/
def chunk (N i : ℕ) : ℕ := if i < 2 * N then 0 else 1

theorem chunk_le (N i : ℕ) : chunk N i ≤ 1 := by
  unfold chunk
  split <;> omega

/-- Request `r` produces token `i` in its chunk's step `4r` or `4r + 1`, and reads its tokens at step `4r + 2`. -/
def step (N : ℕ) : Val → ℕ
  | .kv r i => r * 4 + chunk N i
  | .att r => r * 4 + 2

/-- The tokens' bits `val r i`. -/
def value (val : ℕ → ℕ → ℕ) : Val → ℕ
  | .kv r i => val r i
  | .att _ => 0

/-- The run on one device whose clock begins step `s` at tick `s·τ`. -/
def placement (n P N τ : ℕ) (val : ℕ → ℕ → ℕ) : Requirement.Placement Unit Val where
  ran := ran n P N
  step := step N
  value := value val
  leaf _ := ()
  ticks _ s := s * τ

/-- A page's block: page `i/2` of request `r` is block `r·(N+4) + i/2`; read gates sit past every page. -/
def block (n P N : ℕ) : Val → ℕ
  | .kv r i => r * (N + 4) + i / 2
  | .att r => n * P * (N + 4) + r

/-- A token's offset in its page: the even one first. -/
def offset : Val → ℕ
  | .kv _ i => i % 2 * tw
  | .att _ => 0

/-- When page `k` is first read encoded, counted from its request's first step: two steps after the step that completes
it, step `0` for the first chunk's pages, step `1` for the second chunk's three complete ones, and step `2`, whose
decode token fills it, for page `N+3`. -/
def lag (N k : ℕ) : ℕ := if k < N then 2 else if k < N + 3 then 3 else 4

theorem lag_lt {N k : ℕ} (hk : k < N) : lag N k = 2 := by
  simp [lag, hk]

theorem lag_ge {N k : ℕ} (hk : N ≤ k) : 3 ≤ lag N k := by
  unfold lag
  split_ifs <;> omega

/-- Deferred residency: page `k` of request `r` is first read encoded at step `4r + lag N k`. -/
def encoded (N b : ℕ) : ℕ := b / (N + 4) * 4 + lag N (b % (N + 4))

def kvLayout (n P N : ℕ) : Requirement.Layout Val := ⟨block n P N, offset, encoded N⟩

/-- A value's group: its request, layer `0`, plane `0` and page. -/
def kvGrp : Val → ℕ × ℕ × ℕ × ℕ
  | .kv r i => (r, 0, 0, i / 2)
  | .att r => (r, 1, 0, 0)

/-- The memory map: a page is written in the record of the step before its first encoded read, and a request's pages are
freed in the record of step `4r + 3`; a page's payload is its two tokens. -/
def kvStore (n P N : ℕ) (val : ℕ → ℕ → ℕ) : Requirement.Store where
  registered := 0
  written s := (Finset.range (n * P * (N + 4))).filter fun b => encoded N b = s + 1
  freed s := (Finset.range (n * P * (N + 4))).filter fun b => b / (N + 4) * 4 + 3 = s
  payload b := val (b / (N + 4)) (2 * (b % (N + 4))) + val (b / (N + 4)) (2 * (b % (N + 4)) + 1) * 2 ^ tw

/-- Each pass is bound to its request's decode step. -/
def passBind (n P N : ℕ) (w : Fin n) (j : Fin P) : ℕ := req w j * 4 + 2

/-- Each pass plans its request's first chunk's pages `0 … N−1`. -/
def kvPlan (n P N : ℕ) (w : Fin n) (j : Fin P) : Finset ℕ :=
  (Finset.range N).image fun k => req w j * (N + 4) + k

/-- One append a page: page block `b` is append `1 + b`. -/
def kvAddr (n P N b : ℕ) : Fin (1 + n * P * (N + 4)) × Fin 1 :=
  if h : b < n * P * (N + 4) then (⟨1 + b, by omega⟩, 0) else (⟨0, by omega⟩, 0)

theorem kvAddr_page {n P N b : ℕ} (hb : b < n * P * (N + 4)) : kvAddr n P N b = (⟨1 + b, by omega⟩, 0) := by
  simp [kvAddr, hb]

/-- Pass `j` of window `w` certifies its request's pages `0 … N−1` at its `N` tiles. -/
def slot (n P N : ℕ) (w : Fin n) (j : Fin P) : Fin N ↪ Fin (1 + n * P * (N + 4)) × Fin 1 where
  toFun i := (⟨1 + (req w j * (N + 4) + i), by
    have hi := i.isLt
    have := blk_lt (req_lt w j) (show (i : ℕ) < N + 4 by omega)
    omega⟩, 0)
  inj' a b h := by
    have h1 := congrArg (fun p => (p.1 : ℕ)) h
    try simp only at h1
    exact Fin.ext (by omega)

/-- The audit's layout: the pass's slots, read from the prefix's window and pass only. -/
def passLayout (n P N ρ : ℕ) (Ω Ca : Type) :
    RateSketch.SketchPrefix n P N 128 ρ 512 16 1 2773 → Ω × (Fin (1 + n * P * (N + 4)) → Ca) →
      Fin N ↪ Fin (1 + n * P * (N + 4)) × Fin 1 :=
  fun c _ => slot n P N c.w c.j

/-- The `0`-bit summary. -/
def steer0 (n P N ρ : ℕ) (Ω Ca : Type) :
    RateSketch.SketchPrefix n P N 128 ρ 512 16 1 2773 → Ω × (Fin (1 + n * P * (N + 4)) → Ca) → Pous.Bits 0 :=
  fun _ _ i => i.elim0

/-- Append `t = 1 + b` joins the window of page block `b`'s request. -/
def joined (n P N ρ : ℕ) (Ω Ca : Type) :
    RateSketch.StackedCoins Ω Ca (1 + n * P * (N + 4)) n P N ρ 1 → Fin (1 + n * P * (N + 4)) → ℕ :=
  fun _ t => ((t : ℕ) - 1) / (N + 4) / P

/-- Each window's audit: the window of step `t` is its request's, and each window certifies its passes' stores. -/
def aud (n P N : ℕ) : Gates.Audit := ⟨n, fun t => t / 4 / P, fun _ => P * N * 4096⟩

/-! ## The requirement side -/

section Requirement

variable {n P N τ : ℕ} (val : ℕ → ℕ → ℕ)

theorem reads_eq {r : ℕ} (hr : r < n * P) :
    Requirement.reads (wiring N) (placement n P N τ val).toExecution (r * 4 + 2) =
      (Finset.range (2 * N + 7)).image (Val.kv r) := by
  ext v
  simp only [Requirement.reads, Finset.mem_filter, Finset.mem_image, Finset.mem_range]
  constructor
  · rintro ⟨-, -, g, -, hgs, hv⟩
    change v ∈ ops N g at hv
    obtain ⟨r', i, rfl, rfl, hi⟩ := mem_ops.1 hv
    change r' * 4 + 2 = r * 4 + 2 at hgs
    have hrr : r' = r := by omega
    subst hrr
    exact ⟨i, hi, rfl⟩
  · rintro ⟨i, hi, rfl⟩
    refine ⟨kv_mem_ran.2 ⟨hr, hi⟩, Or.inr ?_, Val.att r, att_mem_ran.2 hr, rfl, mem_ops.2 ⟨r, i, rfl, rfl, hi⟩⟩
    show r * 4 + chunk N i < r * 4 + 2
    have := chunk_le N i
    omega

theorem live_eq {r : ℕ} (hr : r < n * P) :
    Requirement.live (wiring N) (placement n P N τ val).toExecution (r * 4 + 2) =
      (Finset.range (2 * N + 7)).image (Val.kv r) := by
  ext v
  simp only [Requirement.live, Finset.mem_filter, Finset.mem_image, Finset.mem_range]
  constructor
  · rintro ⟨-, hlt, g, -, hgs, hv⟩
    change v ∈ ops N g at hv
    obtain ⟨r', i, rfl, rfl, hi⟩ := mem_ops.1 hv
    change r * 4 + 2 ≤ r' * 4 + 2 at hgs
    have hlt' : r' * 4 + chunk N i < r * 4 + 2 := by
      rcases hlt with h | h
      · exact absurd h (Finset.notMem_empty _)
      · exact h
    have hrr : r' = r := by omega
    subst hrr
    exact ⟨i, hi, rfl⟩
  · rintro ⟨i, hi, rfl⟩
    refine ⟨kv_mem_ran.2 ⟨hr, hi⟩, Or.inr ?_, Val.att r, att_mem_ran.2 hr, le_refl _, mem_ops.2 ⟨r, i, rfl, rfl, hi⟩⟩
    show r * 4 + chunk N i < r * 4 + 2
    have := chunk_le N i
    omega

theorem encoded_page {r k : ℕ} (hk : k < N + 4) : encoded N (r * (N + 4) + k) = r * 4 + lag N k := by
  unfold encoded
  rw [dec_div hk, dec_mod hk]

/-- At its request's decode step, page `i/2` is incomplete exactly from page `N` on: the second chunk's pages have their
coins out, and page `N+3` is still filling. -/
theorem block_mem_incomplete {r i : ℕ} (hr : r < n * P) (hi : i < 2 * N + 7) :
    r * (N + 4) + i / 2 ∈ Requirement.incomplete (wiring N) (placement n P N τ val).toExecution (kvLayout n P N)
      (r * 4 + 2) ↔ 2 * N ≤ i := by
  simp only [Requirement.incomplete, Finset.mem_image, Finset.mem_filter]
  constructor
  · rintro ⟨v, ⟨hv, -, hs⟩, hb⟩
    cases v with
    | kv r' i' =>
      obtain ⟨hr', hi'⟩ := kv_mem_ran.1 hv
      change r' * (N + 4) + i' / 2 = r * (N + 4) + i / 2 at hb
      change r * 4 + 2 ≤ r' * 4 + chunk N i' ∨ r * 4 + 2 < encoded N (r' * (N + 4) + i' / 2) at hs
      rw [encoded_page (show i' / 2 < N + 4 by omega)] at hs
      obtain ⟨hrr, hk⟩ := dec_inj (show i' / 2 < N + 4 by omega) (show i / 2 < N + 4 by omega) hb
      subst hrr
      by_contra h
      have hl : lag N (i' / 2) = 2 := lag_lt (by omega)
      have hc : chunk N i' = 0 := by
        unfold chunk
        rw [if_pos (by omega)]
      omega
    | att r' =>
      have hr' := att_mem_ran.1 hv
      change n * P * (N + 4) + r' = r * (N + 4) + i / 2 at hb
      have := blk_lt hr (show i / 2 < N + 4 by omega)
      omega
  · intro h
    refine ⟨Val.kv r i, ⟨kv_mem_ran.2 ⟨hr, hi⟩, Finset.notMem_empty _, Or.inr ?_⟩, rfl⟩
    change r * 4 + 2 < encoded N (r * (N + 4) + i / 2)
    rw [encoded_page (show i / 2 < N + 4 by omega)]
    have := lag_ge (N := N) (k := i / 2) (by omega)
    omega

theorem filter_pages {r : ℕ} (hr : r < n * P) :
    ((Finset.range (2 * N + 7)).image (Val.kv r)).filter (fun v => (kvLayout n P N).block v ∉
      Requirement.incomplete (wiring N) (placement n P N τ val).toExecution (kvLayout n P N)
        (r * 4 + 2)) = (Finset.range (2 * N)).image (Val.kv r) := by
  ext v
  simp only [Finset.mem_filter, Finset.mem_image, Finset.mem_range]
  constructor
  · rintro ⟨⟨i, hi, rfl⟩, hb⟩
    have hb' : ¬ (r * (N + 4) + i / 2 ∈ Requirement.incomplete (wiring N) (placement n P N τ val).toExecution
        (kvLayout n P N) (r * 4 + 2)) := hb
    rw [block_mem_incomplete val hr hi] at hb'
    exact ⟨i, by omega, rfl⟩
  · rintro ⟨i, hi, rfl⟩
    refine ⟨⟨i, by omega, rfl⟩, ?_⟩
    show ¬ (r * (N + 4) + i / 2 ∈ Requirement.incomplete (wiring N) (placement n P N τ val).toExecution
        (kvLayout n P N) (r * 4 + 2))
    rw [block_mem_incomplete val hr (show i < 2 * N + 7 by omega)]
    omega

/-- The decode step reads the first chunk's tokens `0 … 2N−1` from complete blocks. -/
theorem creditedReads_eq {r : ℕ} (hr : r < n * P) :
    Requirement.creditedReads (wiring N) (placement n P N τ val).toExecution (kvLayout n P N)
      (r * 4 + 2) = (Finset.range (2 * N)).image (Val.kv r) := by
  rw [Requirement.creditedReads, reads_eq val hr]
  exact filter_pages val hr

theorem credited_eq {r : ℕ} (hr : r < n * P) :
    Requirement.credited (wiring N) (placement n P N τ val).toExecution (kvLayout n P N)
      (r * 4 + 2) = (Finset.range (2 * N)).image (Val.kv r) := by
  rw [Requirement.credited, live_eq val hr]
  exact filter_pages val hr

end Requirement

/-- The layout packs the run's values into 32,768-bit blocks. -/
theorem packs {n P N : ℕ} (S : Finset Val) (hS : S ⊆ ran n P N) :
    Requirement.Packs (wiring N) (kvLayout n P N) 32768 S := by
  intro u hu
  have hu' := hS hu
  refine ⟨?_, fun v hv huv hb => ?_⟩
  · cases u with
    | kv r i =>
      show i % 2 * tw + tw ≤ 32768
      rw [tw_eq]
      omega
    | att r =>
      show 0 + tw ≤ 32768
      rw [tw_eq]
      omega
  · have hv' := hS hv
    cases u with
    | kv r i =>
      obtain ⟨hr, hi⟩ := kv_mem_ran.1 hu'
      cases v with
      | kv r' i' =>
        obtain ⟨hr', hi'⟩ := kv_mem_ran.1 hv'
        change r * (N + 4) + i / 2 = r' * (N + 4) + i' / 2 at hb
        obtain ⟨hrr, hk⟩ := dec_inj (show i / 2 < N + 4 by omega) (show i' / 2 < N + 4 by omega) hb
        subst hrr
        have hii : i ≠ i' := by
          rintro rfl
          exact huv rfl
        show i % 2 * tw + tw ≤ i' % 2 * tw ∨ i' % 2 * tw + tw ≤ i % 2 * tw
        rw [tw_eq]
        omega
      | att r' =>
        change r * (N + 4) + i / 2 = n * P * (N + 4) + r' at hb
        have := blk_lt hr (show i / 2 < N + 4 by omega)
        omega
    | att r =>
      have hr := att_mem_ran.1 hu'
      cases v with
      | kv r' i' =>
        obtain ⟨hr', hi'⟩ := kv_mem_ran.1 hv'
        change n * P * (N + 4) + r = r' * (N + 4) + i' / 2 at hb
        have := blk_lt hr' (show i' / 2 < N + 4 by omega)
        omega
      | att r' =>
        change n * P * (N + 4) + r = n * P * (N + 4) + r' at hb
        have hrr : r = r' := by omega
        subst hrr
        exact absurd rfl huv

/-- Each credited token is in its page's payload. -/
theorem carries {n P N τ r : ℕ} (val : ℕ → ℕ → ℕ) (hval : ∀ r i, val r i < 2 ^ tw) :
    Requirement.Carries (wiring N) (placement n P N τ val).toExecution (kvLayout n P N) (kvStore n P N val)
      ((Finset.range (2 * N)).image (Val.kv r)) := by
  intro v hv
  obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hv
  rw [Finset.mem_range] at hi
  have hk : i / 2 < N + 4 := by omega
  show (val ((r * (N + 4) + i / 2) / (N + 4)) (2 * ((r * (N + 4) + i / 2) % (N + 4))) +
      val ((r * (N + 4) + i / 2) / (N + 4)) (2 * ((r * (N + 4) + i / 2) % (N + 4)) + 1) * 2 ^ tw) /
      2 ^ (i % 2 * tw) % 2 ^ tw = val r i
  rw [dec_div hk, dec_mod hk]
  have hp : 0 < 2 ^ tw := Nat.two_pow_pos tw
  rcases Nat.mod_two_eq_zero_or_one i with h | h
  · have hi2 : 2 * (i / 2) = i := by omega
    rw [h, hi2, Nat.zero_mul, Nat.pow_zero, Nat.div_one, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (hval r i)]
  · have hi2 : 2 * (i / 2) + 1 = i := by omega
    rw [h, hi2, Nat.one_mul, Nat.add_mul_div_right _ _ hp, Nat.div_eq_of_lt (hval _ _), Nat.zero_add,
      Nat.mod_eq_of_lt (hval r i)]

/-- A first-chunk page is resident at its request's decode step: written in the record of step `4r + 1`, freed only in
that of `4r + 3`. -/
theorem resident_page {n P N r i : ℕ} (val : ℕ → ℕ → ℕ) (hr : r < n * P) (hi : i < 2 * N) :
    r * (N + 4) + i / 2 ∈ (kvStore n P N val).resident (r * 4 + 2) := by
  have hk : i / 2 < N + 4 := by omega
  have hb : r * (N + 4) + i / 2 < n * P * (N + 4) := blk_lt hr hk
  simp only [Requirement.Store.resident, Finset.mem_sdiff, Finset.mem_union, Finset.mem_biUnion, Finset.mem_range]
  refine ⟨Or.inr ⟨r * 4 + 1, by omega, ?_⟩, ?_⟩
  · show r * (N + 4) + i / 2 ∈ (Finset.range (n * P * (N + 4))).filter fun b => encoded N b = _ + 1
    rw [Finset.mem_filter, Finset.mem_range, encoded_page hk, lag_lt (by omega)]
    exact ⟨hb, by omega⟩
  · rintro ⟨s', hs', hf⟩
    change r * (N + 4) + i / 2 ∈ (Finset.range (n * P * (N + 4))).filter
      (fun b => b / (N + 4) * 4 + 3 = s') at hf
    rw [Finset.mem_filter, dec_div hk] at hf
    omega

theorem storeMatches {n P N τ r : ℕ} (val : ℕ → ℕ → ℕ) (hval : ∀ r i, val r i < 2 ^ tw) (hr : r < n * P) :
    Requirement.StoreMatches (wiring N) (placement n P N τ val).toExecution (kvLayout n P N) (kvStore n P N val)
      32768 (r * 4 + 2) := by
  refine ⟨?_, packs _ (Finset.filter_subset _ _), ?_⟩
  · rw [credited_eq val hr]
    intro b hb
    obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hb
    obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hv
    rw [Finset.mem_range] at hi
    exact resident_page val hr hi
  · rw [credited_eq val hr]
    exact carries val hval

theorem planMatches {n P N τ : ℕ} (val : ℕ → ℕ → ℕ) (w : Fin n) (j : Fin P) :
    Requirement.PlanMatches (wiring N) (placement n P N τ val).toExecution (kvLayout n P N) (kvPlan n P N w j)
      32768 (passBind n P N w j) := by
  have hr := req_lt w j
  refine ⟨?_, packs _ (Finset.filter_subset _ _)⟩
  show Requirement.blocks (kvLayout n P N) (Requirement.creditedReads (wiring N)
    (placement n P N τ val).toExecution (kvLayout n P N) (req w j * 4 + 2)) ⊆ kvPlan n P N w j
  rw [creditedReads_eq val hr]
  intro b hb
  obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hb
  obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hv
  rw [Finset.mem_range] at hi
  exact Finset.mem_image.2 ⟨i / 2, Finset.mem_range.2 (by omega), rfl⟩

/-- **Page groups meet `OneBlockEach` on any values**, so at every step of any schedule: a group is one page. -/
theorem oneBlockEach_any {n P N : ℕ} (S : Finset Val) : Requirement.OneBlockEach (kvLayout n P N) kvGrp S := by
  intro k
  rw [Finset.card_le_one]
  intro a ha b hb
  obtain ⟨u, hu, rfl⟩ := Finset.mem_image.1 ha
  obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hb
  rw [Finset.mem_filter] at hu hv
  have hg : kvGrp u = kvGrp v := hu.2.trans hv.2.symm
  cases u with
  | kv r i =>
    cases v with
    | kv r' i' =>
      have h1 : r = r' := congrArg Prod.fst hg
      have h4 : i / 2 = i' / 2 := congrArg (fun p : ℕ × ℕ × ℕ × ℕ => p.2.2.2) hg
      show r * (N + 4) + i / 2 = r' * (N + 4) + i' / 2
      rw [h1, h4]
    | att r' =>
      have h : (0 : ℕ) = 1 := congrArg (fun p : ℕ × ℕ × ℕ × ℕ => p.2.1) hg
      exact absurd h (by decide)
  | att r =>
    cases v with
    | kv r' i' =>
      have h : (1 : ℕ) = 0 := congrArg (fun p : ℕ × ℕ × ℕ × ℕ => p.2.1) hg
      exact absurd h (by decide)
    | att r' =>
      have h1 : r = r' := congrArg Prod.fst hg
      show n * P * (N + 4) + r = n * P * (N + 4) + r'
      rw [h1]

theorem plan_lt {n P N : ℕ} {w : Fin n} {j : Fin P} {b : ℕ} (hb : b ∈ kvPlan n P N w j) :
    ∃ k < N, b = req w j * (N + 4) + k ∧ b < n * P * (N + 4) := by
  obtain ⟨k, hk, rfl⟩ := Finset.mem_image.1 hb
  rw [Finset.mem_range] at hk
  exact ⟨k, hk, rfl, blk_lt (req_lt w j) (by omega)⟩

theorem injOn_addr {n P N : ℕ} (w : Fin n) (j : Fin P) : Set.InjOn (kvAddr n P N) ↑(kvPlan n P N w j) := by
  intro a ha b hb hab
  obtain ⟨k, -, rfl, ha'⟩ := plan_lt (Finset.mem_coe.1 ha)
  obtain ⟨k', -, rfl, hb'⟩ := plan_lt (Finset.mem_coe.1 hb)
  rw [kvAddr_page ha', kvAddr_page hb'] at hab
  have h1 := congrArg (fun p => (p.1 : ℕ)) hab
  try simp only at h1
  omega

theorem addr_mem_range {n P N : ℕ} (w : Fin n) (j : Fin P) {b : ℕ} (hb : b ∈ kvPlan n P N w j) :
    kvAddr n P N b ∈ Set.range (slot n P N w j) := by
  obtain ⟨k, hk, rfl, hb'⟩ := plan_lt hb
  rw [kvAddr_page hb']
  exact ⟨⟨k, hk⟩, rfl⟩

theorem passBind_inj {n P N : ℕ} {w w' : Fin n} {j j' : Fin P} (h : passBind n P N w j = passBind n P N w' j') :
    w = w' ∧ j = j' := by
  unfold passBind at h
  have h1 : req w j = req w' j' := by omega
  unfold req at h1
  have h2 := dec_inj j.isLt j'.isLt
    (by rw [Nat.mul_comm (w : ℕ) P, Nat.mul_comm (w' : ℕ) P]; omega : (w : ℕ) * P + j = (w' : ℕ) * P + j')
  exact ⟨Fin.ext h2.1, Fin.ext h2.2⟩

theorem passBind_lt {n P N : ℕ} {w w' : Fin n} {j j' : Fin P} (h : w < w' ∨ w = w' ∧ j < j') :
    passBind n P N w j < passBind n P N w' j' := by
  have hr : req w j < req w' j' := by
    unfold req
    rcases h with h | ⟨rfl, h⟩
    · rw [Fin.lt_def] at h
      have h1 : P * ((w : ℕ) + 1) ≤ P * w' := Nat.mul_le_mul_left P h
      rw [Nat.mul_add_one] at h1
      have := j.isLt
      omega
    · rw [Fin.lt_def] at h
      omega
  unfold passBind
  omega

/-- No device tick is in two passes: the passes are bound to distinct steps, a step's `τ` ticks each. -/
theorem deadlines {n P N τ : ℕ} (val : ℕ → ℕ → ℕ) :
    (placement n P N τ val).Deadlines (passBind n P N) τ := by
  intro d
  refine ⟨fun w w' hw => ?_, fun w j j' hj => ?_⟩
  · simp only [Requirement.Placement.windowTicks]
    rw [Finset.disjoint_biUnion_left]
    intro j _
    rw [Finset.disjoint_biUnion_right]
    intro j' _
    show Disjoint (Finset.Ico (passBind n P N w j * τ) (passBind n P N w j * τ + τ))
      (Finset.Ico (passBind n P N w' j' * τ) (passBind n P N w' j' * τ + τ))
    exact disjoint_ticks fun h => hw (passBind_inj h).1
  · show Disjoint (Finset.Ico (passBind n P N w j * τ) (passBind n P N w j * τ + τ))
      (Finset.Ico (passBind n P N w j' * τ) (passBind n P N w j' * τ + τ))
    exact disjoint_ticks fun h => hj (passBind_inj h).2

/-- Each pass's requirement links: the store holds the credited pages, the plan covers the reads, the incomplete
pages' tokens lie in one page a group, and the plan's pages are at distinct positions of the pass's tiles. -/
theorem requirement {n P N τ : ℕ} (val : ℕ → ℕ → ℕ) (hval : ∀ r i, val r i < 2 ^ tw) (w : Fin n) (j : Fin P) :
    let e := (placement n P N τ val).toExecution
    Requirement.StoreMatches (wiring N) e (kvLayout n P N) (kvStore n P N val) 32768 (passBind n P N w j) ∧
      Requirement.PlanMatches (wiring N) e (kvLayout n P N) (kvPlan n P N w j) 32768 (passBind n P N w j) ∧
      Requirement.OneBlockEach (kvLayout n P N) kvGrp
        (Requirement.reads (wiring N) e (passBind n P N w j) \
          Requirement.creditedReads (wiring N) e (kvLayout n P N) (passBind n P N w j)) ∧
      Set.InjOn (kvAddr n P N) ↑(kvPlan n P N w j) ∧
      ∀ b ∈ kvPlan n P N w j, kvAddr n P N b ∈ Set.range (slot n P N w j) := by
  intro e
  have hr := req_lt w j
  exact ⟨storeMatches val hval hr, planMatches val w j, oneBlockEach_any _, injOn_addr w j,
    fun b hb => addr_mem_range w j hb⟩

/-- A page joins its request's window and is read only there. -/
theorem lifetimeCapped {n P N ρ L : ℕ} {Ω Ca : Type} (hL : 0 < L) :
    RateSketch.LifetimeCapped (passLayout n P N ρ Ω Ca) (joined n P N ρ Ω Ca) L := by
  intro x w j i _
  have hj := j.isLt
  have hi := i.isLt
  show (1 + (req w j * (N + 4) + (i : ℕ)) - 1) / (N + 4) / P ≤ (w : ℕ) ∧
    (w : ℕ) < (1 + (req w j * (N + 4) + (i : ℕ)) - 1) / (N + 4) / P + L
  have h1 : (1 + (req w j * (N + 4) + (i : ℕ)) - 1) / (N + 4) / P = w := by
    rw [show 1 + (req w j * (N + 4) + (i : ℕ)) - 1 = req w j * (N + 4) + (i : ℕ) by omega,
      dec_div (show (i : ℕ) < N + 4 by omega)]
    unfold req
    rw [show (j : ℕ) + P * (w : ℕ) = (w : ℕ) * P + j by rw [Nat.mul_comm]; omega, dec_div hj]
  rw [h1]
  omega

/-! ## The witness -/

/-- **An honest run with a prefill meets every hypothesis of `DeclaredCircuitRequirement`** but the world's
(`SketchRank`, `KeyBinding`, `SquaringFloor`, `ChainSum` and their floor), at every value of the session's parameters
with the link carrying the stream (`ρ·256 ≤ L_r·τ`), `N` queries within the caps and a carrier life `L ≥ 1`, at any nonnegative
caps, codec and append choices, and any token bits. -/
theorem meets (n P N ρ S q Xr Lr qr τ L B εnum εden : ℕ) {Ω Ca Ch : Type}
    (enc : Ω → Fin (1 + n * P * (N + 4)) → Ch → (Fin (1 + n * P * (N + 4)) → Ca) → Fin 1 →
      (Fin 128 → RateSketch.Sector))
    (choose : RateSketch.SketchPrefix n P N 128 ρ 512 16 1 2773 → Ω → (a : Fin (1 + n * P * (N + 4))) →
      ({b : Fin (1 + n * P * (N + 4)) // b < a} → Ca) → Ch)
    (caps : Gates.Caps)
    (hcaps : 0 ≤ caps.rounding ∧ 0 ≤ caps.compareSelect ∧ 0 ≤ caps.gather ∧ 0 ≤ caps.fold ∧ 0 ≤ caps.products)
    (val : ℕ → ℕ → ℕ) (hval : ∀ r i, val r i < 2 ^ tw)
    (hL : 0 < L) (hlink : ρ * 256 ≤ Lr * τ) (hq : N ≤ q) (hqr : N ≤ qr * τ) :
    (∀ s w η j pre, Assumptions.SketchHashBudget
      ((((honestSession n P N ρ S (Lr * τ) q hlink hq).session s).window w η).device j pre) (qr * τ)) ∧
    Assumptions.CircuitCarriers (fun c y => (honestSession n P N ρ S (Lr * τ) q hlink hq).session
        (steer0 n P N ρ Ω Ca c y))
      (RateSketch.steeredStore enc choose (passLayout n P N ρ Ω Ca)) (passLayout n P N ρ Ω Ca)
      (joined n P N ρ Ω Ca) (Xr * τ) L B εnum εden ([] : Gates.Dag Unit) (aud n P N) ∧
    Assumptions.SteeredLayout (passLayout n P N ρ Ω Ca) (steer0 n P N ρ Ω Ca) ∧
    RateSketch.LifetimeCapped (passLayout n P N ρ Ω Ca) (joined n P N ρ Ω Ca) L ∧
    Gates.Restricted B ([] : Gates.Dag Unit) ∧ Gates.Property caps (aud n P N) ([] : Gates.Dag Unit) ∧
    (∀ v < (aud n P N).windows, (aud n P N).reads v ≤ P * N * 4096) ∧
    (∀ x : RateSketch.StackedCoins Ω Ca (1 + n * P * (N + 4)) n P N ρ 1,
      (placement n P N τ val).Deadlines (passBind n P N) τ ∧
      (∀ w j w' j', w < w' ∨ w = w' ∧ j < j' → passBind n P N w j < passBind n P N w' j') ∧
      ∀ w j,
        let e := (placement n P N τ val).toExecution
        Requirement.StoreMatches (wiring N) e (kvLayout n P N) (kvStore n P N val) 32768 (passBind n P N w j) ∧
        Requirement.PlanMatches (wiring N) e (kvLayout n P N) (kvPlan n P N w j) 32768 (passBind n P N w j) ∧
        Requirement.OneBlockEach (kvLayout n P N) kvGrp
          (Requirement.reads (wiring N) e (passBind n P N w j) \
            Requirement.creditedReads (wiring N) e (kvLayout n P N) (passBind n P N w j)) ∧
        Set.InjOn (kvAddr n P N) ↑(kvPlan n P N w j) ∧
        ∀ b ∈ kvPlan n P N w j,
          kvAddr n P N b ∈ Set.range (passLayout n P N ρ Ω Ca (RateSketch.SketchPrefix.of x.2 w j) x.1)) :=
  ⟨hashBudget hlink hq hqr, circuitCarriers hlink hq _ _ _ _ _ _, ⟨fun c _ => slot n P N c.w c.j, fun _ _ => rfl⟩,
    lifetimeCapped hL, restricted B, property caps hcaps _, fun _ _ => le_refl _,
    fun _ => ⟨deadlines val, fun _ _ _ _ h => passBind_lt h, fun w j => requirement val hval w j⟩⟩

/-- **`G` at the honest prefill run**: given `G` and the world's assumptions, its conclusion holds at the run of
`meets`. -/
theorem applies (hG : Guarantees.DeclaredCircuitRequirement) (n P N ρ S q Xr Lr qr τ k L E es B εnum εden : ℕ)
    (ε δ : ℝ≥0∞) {Ω Ca Ch : Type} [Fintype Ω] [Fintype Ca]
    (enc : Ω → Fin (1 + n * P * (N + 4)) → Ch → (Fin (1 + n * P * (N + 4)) → Ca) → Fin 1 →
      (Fin 128 → RateSketch.Sector))
    (choose : RateSketch.SketchPrefix n P N 128 ρ 512 16 1 2773 → Ω → (a : Fin (1 + n * P * (N + 4))) →
      ({b : Fin (1 + n * P * (N + 4)) // b < a} → Ca) → Ch)
    (caps : Gates.Caps)
    (hcaps : 0 ≤ caps.rounding ∧ 0 ≤ caps.compareSelect ∧ 0 ≤ caps.gather ∧ 0 ≤ caps.fold ∧ 0 ≤ caps.products)
    (val : ℕ → ℕ → ℕ) (hval : ∀ r i, val r i < 2 ^ tw) (hn : 7 ≤ n) (hP : 0 < P) (hN : 0 < N) (hE : 0 < E)
    (hL : 0 < L) (hlink : ρ * 256 ≤ Lr * τ) (hq : N ≤ q) (hqr : N ≤ qr * τ)
    (hrank : Assumptions.SketchRank 16 128 12 ε) (hkey : Assumptions.KeyBinding (1 + n * P * (N + 4)) 1 δ enc)
    (hsq : Assumptions.SquaringFloor Gates.p1024 es εnum εden) (hcs : Assumptions.ChainSum Gates.p1024 3 B es εnum εden)
    (hEle : E ≤ 3 * Gates.p1024.squarings * es) (hfloor : εden ≤ 2 ^ 64 * εnum) :
    let T := 1 + n * P * (N + 4)
    let G := 1
    let a := 0
    let Aud := aud n P N
    let layout := passLayout n P N ρ Ω Ca
    let steer := steer0 n P N ρ Ω Ca
    let A := honestSession n P N ρ S (Lr * τ) q hlink hq
    let W := wiring N
    let p := fun _ : RateSketch.StackedCoins Ω Ca T n P N ρ G => placement n P N τ val
    let ℓ := fun _ : RateSketch.StackedCoins Ω Ca T n P N ρ G => kvLayout n P N
    let grp := fun _ : RateSketch.StackedCoins Ω Ca T n P N ρ G => kvGrp
    let bind := fun _ : RateSketch.StackedCoins Ω Ca T n P N ρ G => passBind n P N
    let addr := fun _ : RateSketch.StackedCoins Ω Ca T n P N ρ G => kvAddr n P N
    let RL := ((Finset.range Aud.windows).powerset.filter fun I => I.card ≤ L).sup
      fun I => ∑ v ∈ I, caps.runBytes v
    let C := 8 * RL + ⌈(1024 * L * (P * N * 4096) : ℚ) * caps.total / E⌉₊
    let γ := RateSketch.gammaSketch N 128 ρ S (Lr * τ) (Xr * τ + C + k * 32768)
    let owned := fun (x : RateSketch.StackedCoins Ω Ca T n P N ρ G) (t : ℕ) =>
      (Requirement.blocks (ℓ x) (Requirement.creditedReads W (p x).toExecution (ℓ x) t)).filter
        fun b => ((addr x b).1 : ℕ) ≠ 0
    let credit := fun (x : RateSketch.StackedCoins Ω Ca T n P N ρ G) (t : ℕ) =>
      let e := (p x).toExecution
      max (Requirement.bitsAt W (ℓ x) (Requirement.credited W e (ℓ x) t) (owned x t))
        (Requirement.bits W ((Requirement.reads W e t).filter fun v => ((addr x ((ℓ x).block v)).1 : ℕ) ≠ 0) -
          ((Requirement.reads W e t \ Requirement.creditedReads W e (ℓ x) t).image (grp x)).card * 32768)
    let tiles := fun (x : RateSketch.StackedCoins Ω Ca T n P N ρ G) (w : Fin n) (j : Fin P) =>
      Finset.univ.filter fun i : Fin N => layout (RateSketch.SketchPrefix.of x.2 w j) x.1 i ∈
        (owned x (bind x w j)).image (addr x)
    let store := RateSketch.programmed (RateSketch.steeredStore enc choose layout)
    let win := fun (x : RateSketch.StackedCoins Ω Ca T n P N ρ G) (w : Fin n) =>
      A.win w x.2 fun j => steer (RateSketch.SketchPrefix.of x.2 w j) x.1
    pr (fun x : RateSketch.StackedCoins Ω Ca T n P N ρ G =>
      (∀ w t, (x.2 w).2.2 t ∈ (win x w).correctOn (store x.1 x.2 w) (x.2 w).1 (x.2 w).2.1) ∧
        ((((∑ w, (win x w).readsIn (store x.1 x.2 w) (x.2 w).1 (x.2 w).2.1 (tiles x w) : ℕ)) : ℝ) <
            ((∑ w, ∑ j, credit x (bind x w j) : ℕ) : ℝ) - (0.0015 + γ) * (n * P * N * 32768) ∨
          ∃ w, (((win x w).readsIn (store x.1 x.2 w) (x.2 w).1 (x.2 w).2.1 (tiles x w) : ℕ) : ℝ) <
            ((∑ j, credit x (bind x w j) : ℕ) : ℝ) - (0.01 + γ) * (P * N * 32768)))
      ≤ 2 / 2 ^ 40 + (n : ℝ≥0∞) * ε + (n * P * G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ 512) +
        (n * P : ℝ≥0∞) * (2 ^ a * (((qr * τ : ℕ) : ℝ≥0∞) / 2 ^ 96) ^ (k + 1) + δ) := by
  intro T G a Aud layout steer A W p ℓ grp bind addr
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ :=
    meets n P N ρ S q Xr Lr qr τ L B εnum εden enc choose caps hcaps val hval hL hlink hq hqr
  exact hG n P N ρ S q G Xr Lr qr τ k a L E es B εnum εden T 1 ε δ ([] : Gates.Dag Unit) caps Aud enc choose layout
    steer (joined n P N ρ Ω Ca) A W p ℓ grp bind (fun _ => kvStore n P N val) (fun _ => kvPlan n P N) addr
    hn hP hN hE hrank hkey hsq hcs hEle hfloor h1 h2 h3 h4 h5 h6 h7 h8

/-! ## The credit, exactly

`G`'s credit at a bound step is the larger of two terms: the held one, `W`'s credited bits in the pass's owned pages,
and the other, every bit read less a page a group of the reads from incomplete pages. Page groups cost the other term a
page for each incomplete page read (four here: the second chunk's three and page `N+3`), where page-parity groups would
charge two, so it is `N·32768 − 16384`; the held term is the first chunk's `N` pages, `N·32768`, the larger. -/

theorem bits_image_kv {N r K : ℕ} : Requirement.bits (wiring N) ((Finset.range K).image (Val.kv r)) = K * tw := by
  unfold Requirement.bits
  show ∑ _v ∈ (Finset.range K).image (Val.kv r), tw = K * tw
  rw [Finset.sum_const, Finset.card_image_of_injective _ (fun a b h => (Val.kv.inj h).2), Finset.card_range,
    smul_eq_mul]

/-- The held term of the run's credit at step `t`, as `G` reads it. -/
def held (n P N τ : ℕ) (val : ℕ → ℕ → ℕ) (t : ℕ) : ℕ :=
  Requirement.bitsAt (wiring N) (kvLayout n P N)
    (Requirement.credited (wiring N) (placement n P N τ val).toExecution (kvLayout n P N) t)
    ((Requirement.blocks (kvLayout n P N)
      (Requirement.creditedReads (wiring N) (placement n P N τ val).toExecution (kvLayout n P N) t)).filter
        fun b => ((kvAddr n P N b).1 : ℕ) ≠ 0)

/-- The other term of the run's credit at step `t`, as `G` reads it. -/
def other (n P N τ : ℕ) (val : ℕ → ℕ → ℕ) (t : ℕ) : ℕ :=
  Requirement.bits (wiring N) ((Requirement.reads (wiring N) (placement n P N τ val).toExecution t).filter
      fun v => ((kvAddr n P N ((kvLayout n P N).block v)).1 : ℕ) ≠ 0) -
    ((Requirement.reads (wiring N) (placement n P N τ val).toExecution t \
      Requirement.creditedReads (wiring N) (placement n P N τ val).toExecution (kvLayout n P N) t).image
        kvGrp).card * 32768

/-- The run's credit at step `t`, as `G` reads it. -/
def credit (n P N τ : ℕ) (val : ℕ → ℕ → ℕ) (t : ℕ) : ℕ := max (held n P N τ val t) (other n P N τ val t)

/-- The held term is the first chunk's `N` pages. -/
theorem held_eq {n P N τ : ℕ} (val : ℕ → ℕ → ℕ) (w : Fin n) (j : Fin P) :
    held n P N τ val (passBind n P N w j) = N * 32768 := by
  have hr := req_lt w j
  have hc : Requirement.credited (wiring N) (placement n P N τ val).toExecution (kvLayout n P N)
      (passBind n P N w j) = (Finset.range (2 * N)).image (Val.kv (req w j)) := credited_eq val hr
  have hcr : Requirement.creditedReads (wiring N) (placement n P N τ val).toExecution (kvLayout n P N)
      (passBind n P N w j) = (Finset.range (2 * N)).image (Val.kv (req w j)) := creditedReads_eq val hr
  have hown : ∀ v ∈ (Finset.range (2 * N)).image (Val.kv (req w j)), (kvLayout n P N).block v ∈
      (Requirement.blocks (kvLayout n P N) (Requirement.creditedReads (wiring N)
        (placement n P N τ val).toExecution (kvLayout n P N) (passBind n P N w j))).filter
          fun b => ((kvAddr n P N b).1 : ℕ) ≠ 0 := by
    intro v hv
    obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hv
    rw [Finset.mem_range] at hi
    have hmem : req w j * (N + 4) + i / 2 ∈ Requirement.blocks (kvLayout n P N)
        (Requirement.creditedReads (wiring N) (placement n P N τ val).toExecution (kvLayout n P N)
          (passBind n P N w j)) := by
      rw [hcr]
      exact Finset.mem_image.2 ⟨_, hv, rfl⟩
    refine Finset.mem_filter.2 ⟨hmem, ?_⟩
    show ((kvAddr n P N (req w j * (N + 4) + i / 2)).1 : ℕ) ≠ 0
    rw [kvAddr_page (blk_lt hr (show i / 2 < N + 4 by omega))]
    show 1 + (req w j * (N + 4) + i / 2) ≠ 0
    omega
  unfold held Requirement.bitsAt
  rw [hc, Finset.filter_true_of_mem hown, bits_image_kv, tw_eq]
  omega

/-- **The other term, with page groups**: the `2N + 7` tokens read, less four pages, one for each incomplete page read. -/
theorem other_eq {n P N τ : ℕ} (val : ℕ → ℕ → ℕ) (w : Fin n) (j : Fin P) :
    other n P N τ val (passBind n P N w j) = N * 32768 - 16384 := by
  have hr := req_lt w j
  have h1 : Requirement.reads (wiring N) (placement n P N τ val).toExecution (passBind n P N w j) =
      (Finset.range (2 * N + 7)).image (Val.kv (req w j)) := reads_eq val hr
  have h2 : Requirement.creditedReads (wiring N) (placement n P N τ val).toExecution (kvLayout n P N)
      (passBind n P N w j) = (Finset.range (2 * N)).image (Val.kv (req w j)) := creditedReads_eq val hr
  have hf : ((Finset.range (2 * N + 7)).image (Val.kv (req w j))).filter
      (fun v => ((kvAddr n P N ((kvLayout n P N).block v)).1 : ℕ) ≠ 0) =
      (Finset.range (2 * N + 7)).image (Val.kv (req w j)) := by
    refine Finset.filter_true_of_mem fun v hv => ?_
    obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hv
    rw [Finset.mem_range] at hi
    show ((kvAddr n P N (req w j * (N + 4) + i / 2)).1 : ℕ) ≠ 0
    rw [kvAddr_page (blk_lt hr (show i / 2 < N + 4 by omega))]
    show 1 + (req w j * (N + 4) + i / 2) ≠ 0
    omega
  have hg : ((Finset.range (2 * N + 7)).image (Val.kv (req w j)) \
      (Finset.range (2 * N)).image (Val.kv (req w j))).image kvGrp =
      (Finset.range 4).image fun k => (req w j, (0 : ℕ), (0 : ℕ), N + k) := by
    ext p
    rw [Finset.mem_image, Finset.mem_image]
    constructor
    · rintro ⟨v, hv, rfl⟩
      rw [Finset.mem_sdiff, Finset.mem_image, Finset.mem_image] at hv
      obtain ⟨⟨i, hi, rfl⟩, hn⟩ := hv
      rw [Finset.mem_range] at hi
      have hi2 : 2 * N ≤ i := by
        by_contra h
        exact hn ⟨i, Finset.mem_range.2 (by omega), rfl⟩
      refine ⟨i / 2 - N, Finset.mem_range.2 (by omega), ?_⟩
      show (req w j, 0, 0, N + (i / 2 - N)) = (req w j, 0, 0, i / 2)
      rw [show N + (i / 2 - N) = i / 2 by omega]
    · rintro ⟨k, hk, rfl⟩
      rw [Finset.mem_range] at hk
      refine ⟨Val.kv (req w j) (2 * (N + k)), ?_, ?_⟩
      · rw [Finset.mem_sdiff, Finset.mem_image, Finset.mem_image]
        refine ⟨⟨2 * (N + k), Finset.mem_range.2 (by omega), rfl⟩, ?_⟩
        rintro ⟨i, hi, hik⟩
        rw [Finset.mem_range] at hi
        have := (Val.kv.inj hik).2
        omega
      · show (req w j, 0, 0, 2 * (N + k) / 2) = (req w j, 0, 0, N + k)
        rw [show 2 * (N + k) / 2 = N + k by omega]
  have hinj : Function.Injective fun k : ℕ => (req w j, (0 : ℕ), (0 : ℕ), N + k) := by
    intro a b h
    have h' : N + a = N + b := congrArg (fun p : ℕ × ℕ × ℕ × ℕ => p.2.2.2) h
    omega
  unfold other
  rw [h1, h2, hf, hg, bits_image_kv, Finset.card_image_of_injective _ hinj, Finset.card_range, tw_eq]
  omega

/-- **The run's credit is its `N` tiles' bits a pass**, page groups and all. -/
theorem credit_eq {n P N τ : ℕ} (val : ℕ → ℕ → ℕ) (w : Fin n) (j : Fin P) :
    credit n P N τ val (passBind n P N w j) = N * 32768 := by
  unfold credit
  rw [held_eq val w j, other_eq val w j]
  exact max_eq_left (Nat.sub_le _ _)

theorem credit_session {n P N τ : ℕ} (val : ℕ → ℕ → ℕ) :
    n * P * N * 32768 ≤ ∑ w : Fin n, ∑ j : Fin P, credit n P N τ val (passBind n P N w j) := by
  rw [Nat.mul_assoc (n * P)]
  exact sum_ge _ fun w j => (credit_eq val w j).ge

theorem credit_window {n P N τ : ℕ} (val : ℕ → ℕ → ℕ) (w : Fin n) :
    P * N * 32768 ≤ ∑ j : Fin P, credit n P N τ val (passBind n P N w j) := by
  calc P * N * 32768 = ∑ _j : Fin P, N * 32768 := by
        simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
        ring
    _ ≤ _ := Finset.sum_le_sum fun j _ => (credit_eq val w j).ge

/-- **`G`'s session threshold is positive at the honest prefill run** while `0.0015 + γ < 1`. -/
theorem threshold_pos {n P N τ : ℕ} (val : ℕ → ℕ → ℕ) {γ : ℝ} (hn : 0 < n) (hP : 0 < P) (hN : 0 < N)
    (hγ : 0.0015 + γ < 1) :
    (0.0015 + γ) * (n * P * N * 32768) <
      ((∑ w : Fin n, ∑ j : Fin P, credit n P N τ val (passBind n P N w j) : ℕ) : ℝ) := by
  have h1 : ((n * P * N * 32768 : ℕ) : ℝ) ≤
      ((∑ w : Fin n, ∑ j : Fin P, credit n P N τ val (passBind n P N w j) : ℕ) : ℝ) :=
    Nat.cast_le.2 (credit_session val)
  rw [show ((n * P * N * 32768 : ℕ) : ℝ) = (n : ℝ) * P * N * 32768 by norm_cast] at h1
  have h2 : (0 : ℝ) < n * P * N * 32768 := by positivity
  calc (0.0015 + γ) * (n * P * N * 32768) < 1 * (n * P * N * 32768) := mul_lt_mul_of_pos_right hγ h2
    _ = (n * P * N * 32768 : ℝ) := one_mul _
    _ ≤ _ := h1

/-- **Each window's threshold is positive at the honest prefill run** while `0.01 + γ < 1`. -/
theorem window_pos {n P N τ : ℕ} (val : ℕ → ℕ → ℕ) {γ : ℝ} (w : Fin n) (hP : 0 < P) (hN : 0 < N)
    (hγ : 0.01 + γ < 1) :
    (0.01 + γ) * (P * N * 32768) < ((∑ j : Fin P, credit n P N τ val (passBind n P N w j) : ℕ) : ℝ) := by
  have h1 : ((P * N * 32768 : ℕ) : ℝ) ≤ ((∑ j : Fin P, credit n P N τ val (passBind n P N w j) : ℕ) : ℝ) :=
    Nat.cast_le.2 (credit_window val w)
  rw [show ((P * N * 32768 : ℕ) : ℝ) = (P : ℝ) * N * 32768 by norm_cast] at h1
  have h2 : (0 : ℝ) < P * N * 32768 := by positivity
  calc (0.01 + γ) * (P * N * 32768) < 1 * (P * N * 32768) := mul_lt_mul_of_pos_right hγ h2
    _ = (P * N * 32768 : ℝ) := one_mul _
    _ ≤ _ := h1

/-! ## At the 30 s life

The carrier life is 30 s, in `L = max(2, ⌊30/T⌋)` windows of `T` seconds. `γ` reads the life through the passes and
run-input bytes it spans, `L·P` and `L·R` (`gamma_prod`), at 61 ms passes (`P ≤ ⌈T/0.061⌉`) and 100 KB/s of run input
(`R = T·100,000`). At every window of 1 to 15 s, `L = 30` one-second windows among them, the life spans at most 510
passes and 3,000,000 bytes (`L·P` peaks at `30·17`); at windows of 15 to 60 s, which only a prover that defers residency
takes, `L = 2` and at most 1,968 passes and 12,000,000 bytes. The run's credit is its tiles' bits, so `G` bounds
something exactly when its `ε` is below 1. -/

/-- `gamma_bound` with a life of `L` windows spanning at most `LP` passes and `LR` run-input bytes. -/
theorem gamma_prod {N ρ S Lr τ Xr k L P E E₀ N₀ R LP LR : ℕ} (caps : Gates.Caps) (Aud : Gates.Audit) (γ₀ c₀ : ℝ)
    (hN₀ : 0 < N₀) (hN : N₀ ≤ N) (hE₀ : 0 < E₀) (hE : E₀ ≤ E) (hc : 0 ≤ caps.total)
    (hc₀ : (caps.total : ℝ) ≤ c₀) (hR : ∀ v, caps.runBytes v ≤ R) (hLP : L * P ≤ LP) (hLR : L * R ≤ LR)
    (h₀ : ((S + (Lr * τ - ρ * 256) + Xr * τ + k * 32768 : ℕ) : ℝ) ≤ γ₀ * (N * 32768)) :
    let RL := ((Finset.range Aud.windows).powerset.filter fun I => I.card ≤ L).sup
      fun I => ∑ v ∈ I, caps.runBytes v
    let C := 8 * RL + ⌈(1024 * L * (P * N * 4096) : ℚ) * caps.total / E⌉₊
    RateSketch.gammaSketch N 128 ρ S (Lr * τ) (Xr * τ + C + k * 32768) ≤
      γ₀ + (8 * LR + 1) / (N₀ * 32768) + 128 * LP * c₀ / E₀ := by
  intro RL C
  have hγ : RateSketch.gammaSketch N 128 ρ S (Lr * τ) (Xr * τ + C + k * 32768) ≤
      γ₀ + (8 * L * R + 1) / (N₀ * 32768) + 128 * L * P * c₀ / E₀ :=
    gamma_bound (L₀ := L) (P₀ := P) (R₀ := R) caps Aud γ₀ c₀ hN₀ hN le_rfl le_rfl hE₀ hE hc hc₀ hR h₀
  have hc0 : 0 ≤ c₀ := le_trans (Rat.cast_nonneg.2 hc) hc₀
  have h1 : (L : ℝ) * R ≤ LR := by exact_mod_cast hLR
  have h2 : (L : ℝ) * P ≤ LP := by exact_mod_cast hLP
  have e1 : (8 * L * R + 1 : ℝ) / (N₀ * 32768) ≤ (8 * LR + 1) / (N₀ * 32768) :=
    div_le_div_of_nonneg_right (by linarith) (by positivity)
  have e2 : (128 * L * P * c₀ : ℝ) / E₀ ≤ 128 * LP * c₀ / E₀ :=
    div_le_div_of_nonneg_right (by linarith [mul_le_mul_of_nonneg_right h2 hc0]) (by positivity)
  linarith

/-- **8B** (`c ≤ 0.442`, `E ≥ 364,497`, `γ₀ ≤ 0.835%`, `N ≥ 2^24`): with windows of at most 15 s (`L = 30` one-second
windows among them) the session's `0.0015 + γ` is under 9% and each window's `0.01 + γ` under 10%; with windows up to
60 s under deferred residency, under 32% and 33%. -/
theorem life30_8b {N ρ S Lr τ Xr k L P E R : ℕ} (caps : Gates.Caps) (Aud : Gates.Audit)
    (hN : 16777216 ≤ N) (hE : 364497 ≤ E) (hc : 0 ≤ caps.total) (hc₀ : caps.total ≤ 221 / 500)
    (hR : ∀ v, caps.runBytes v ≤ R)
    (h₀ : ((S + (Lr * τ - ρ * 256) + Xr * τ + k * 32768 : ℕ) : ℝ) ≤ 0.00835 * (N * 32768)) :
    let RL := ((Finset.range Aud.windows).powerset.filter fun I => I.card ≤ L).sup
      fun I => ∑ v ∈ I, caps.runBytes v
    let C := 8 * RL + ⌈(1024 * L * (P * N * 4096) : ℚ) * caps.total / E⌉₊
    let γ := RateSketch.gammaSketch N 128 ρ S (Lr * τ) (Xr * τ + C + k * 32768)
    (L * P ≤ 510 → L * R ≤ 3000000 → 0.0015 + γ < 0.09 ∧ 0.01 + γ < 0.1) ∧
      (L * P ≤ 1968 → L * R ≤ 12000000 → 0.0015 + γ < 0.32 ∧ 0.01 + γ < 0.33) := by
  intro RL C γ
  have hc₀' : (caps.total : ℝ) ≤ 221 / 500 := by
    have h : ((caps.total : ℚ) : ℝ) ≤ ((221 / 500 : ℚ) : ℝ) := Rat.cast_le.2 hc₀
    rwa [show ((221 / 500 : ℚ) : ℝ) = 221 / 500 by norm_num] at h
  refine ⟨fun hLP hLR => ?_, fun hLP hLR => ?_⟩
  · have hγ : γ ≤ _ := gamma_prod (E₀ := 364497) (N₀ := 16777216) caps Aud 0.00835 (221 / 500) (by norm_num) hN
      (by norm_num) hE hc hc₀' hR hLP hLR h₀
    norm_num at hγ
    constructor <;> linarith
  · have hγ : γ ≤ _ := gamma_prod (E₀ := 364497) (N₀ := 16777216) caps Aud 0.00835 (221 / 500) (by norm_num) hN
      (by norm_num) hE hc hc₀' hR hLP hLR h₀
    norm_num at hγ
    constructor <;> linarith

/-- **70B TP8** (`c ≤ 0.0918`): with windows of at most 15 s the session's `0.0015 + γ` is under 3% and each
window's under 4%; with windows up to 60 s under deferred residency, under 8% and 9%. -/
theorem life30_70b {N ρ S Lr τ Xr k L P E R : ℕ} (caps : Gates.Caps) (Aud : Gates.Audit)
    (hN : 16777216 ≤ N) (hE : 364497 ≤ E) (hc : 0 ≤ caps.total) (hc₀ : caps.total ≤ 459 / 5000)
    (hR : ∀ v, caps.runBytes v ≤ R)
    (h₀ : ((S + (Lr * τ - ρ * 256) + Xr * τ + k * 32768 : ℕ) : ℝ) ≤ 0.00835 * (N * 32768)) :
    let RL := ((Finset.range Aud.windows).powerset.filter fun I => I.card ≤ L).sup
      fun I => ∑ v ∈ I, caps.runBytes v
    let C := 8 * RL + ⌈(1024 * L * (P * N * 4096) : ℚ) * caps.total / E⌉₊
    let γ := RateSketch.gammaSketch N 128 ρ S (Lr * τ) (Xr * τ + C + k * 32768)
    (L * P ≤ 510 → L * R ≤ 3000000 → 0.0015 + γ < 0.03 ∧ 0.01 + γ < 0.04) ∧
      (L * P ≤ 1968 → L * R ≤ 12000000 → 0.0015 + γ < 0.08 ∧ 0.01 + γ < 0.09) := by
  intro RL C γ
  have hc₀' : (caps.total : ℝ) ≤ 459 / 5000 := by
    have h : ((caps.total : ℚ) : ℝ) ≤ ((459 / 5000 : ℚ) : ℝ) := Rat.cast_le.2 hc₀
    rwa [show ((459 / 5000 : ℚ) : ℝ) = 459 / 5000 by norm_num] at h
  refine ⟨fun hLP hLR => ?_, fun hLP hLR => ?_⟩
  · have hγ : γ ≤ _ := gamma_prod (E₀ := 364497) (N₀ := 16777216) caps Aud 0.00835 (459 / 5000) (by norm_num) hN
      (by norm_num) hE hc hc₀' hR hLP hLR h₀
    norm_num at hγ
    constructor <;> linarith
  · have hγ : γ ≤ _ := gamma_prod (E₀ := 364497) (N₀ := 16777216) caps Aud 0.00835 (459 / 5000) (by norm_num) hN
      (by norm_num) hE hc hc₀' hR hLP hLR h₀
    norm_num at hγ
    constructor <;> linarith

end Accounting.SecurityProofs.DeclaredCircuitWitnessPrefill
