import Mathlib
import Definitions.Accounting.Guarantees.DeclaredCircuit
import Proofs.Accounting.RateAudit.SketchHonest

/-!
# An honest run meets every hypothesis of `DeclaredCircuitRequirement`

The non-vacuity witness of `Accounting.Guarantees.DeclaredCircuitRequirement` (`G`): one honest run, generic in the
session's parameters (`n` windows of `P` passes over `N` tiles of 128 sectors at 16 sketch rows, the link, the hash
budget, the carrier life `L`, the caps), meets every hypothesis of `G` but the world's: `SketchRank`, `KeyBinding`,
`SquaringFloor`, `ChainSum` and their floor `εden ≤ 2^64·εnum` (`meets`). So `G` bounds the run (`applies`), every pass's credit
is at least its `N` tiles' bits (`credit_ge`), and the session's credit is at least the store's `n·P·N·32768` bits
(`sum_ge`): the bound says something whenever `0.0015 + γ < 1`.

* **The KV pool** (`wiring`, `placement`): request `r = j + P·w` (one a pass) decodes `2N + 3` tokens of 16,384 bits,
  one a step from step `r·(2N+4)`, two to a 32,768-bit page, and one read gate at step `r·(2N+4) + 2N+3` reads every
  token. Pass `j` of window `w` is bound to that step (`passBind`).
* **Deferred residency** (`kvLayout`): page `k` of request `r` is block `r·(N+2) + k`, and its first encoded read is
  `r·(2N+4) + 2k + 4`, two steps after the step that completes it. At the read step pages `0 … N−1` are complete and
  credited, page `N` is complete with its coin out, and page `N+1` holds the last token: two incomplete pages.
* **Page-parity groups** (`kvGrp`): a token's request, layer `0`, plane `0` and page parity. The honest schedule this
  assumes: one token a step per sequence (no prefill completing two pages before their coins return) and each page's
  coin back two steps after it completes, so at most two pages of a sequence, layer and plane are incomplete when read,
  and they have opposite parity.
* **The store** (`kvStore`): the memory map writes each page in the record of the step before its first encoded read
  and frees a request's pages at its read step; a page's payload is its two tokens. One append a page
  (`T = 1 + n·P·(N+2)`, `M = 1`): page block `b` is append `1 + b` (`kvAddr`), and the pass's layout (`passLayout`)
  puts its request's pages `0 … N−1` at its `N` tiles. A page joins the window of its request (`joined`) and is read
  only there, so the lifetime cap holds at every `L ≥ 1`.
* **The devices** (`honestSession`): every pass device is the honest one of `RateAudit/SketchHonest.lean` (nothing on
  chip, the stream over the link, every sector read once, every tile tagged, nothing rebuilt), with one grinding
  candidate and a `0`-bit summary. So the hash budget holds at `N ≤ q_r·τ`, `CircuitCarriers` with no carrier, and
  `SteeredLayout` because the layout reads no coin.
* **The gate model's circuit** `Wc` is the empty list, restricted and within any nonnegative caps. That `Wc` and `W`
  are one circuit is the code's to hold, not `G`'s, and the run's carriers are empty whatever `Wc` is.
-/

namespace Accounting.SecurityProofs.DeclaredCircuitWitness

open Pous (pr)
open scoped ENNReal

/-! ## Decoding `r·D + i` -/

theorem dec_div {D r i : ℕ} (hi : i < D) : (r * D + i) / D = r := by
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ (Nat.lt_of_le_of_lt (Nat.zero_le i) hi), Nat.div_eq_of_lt hi,
    Nat.zero_add]

theorem dec_mod {D r i : ℕ} (hi : i < D) : (r * D + i) % D = i := by
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hi]

theorem dec_le {D r r' i i' : ℕ} (hi : i < D) (hi' : i' < D) (h : r * D + i ≤ r' * D + i') : r ≤ r' := by
  have h1 := Nat.div_le_div_right (c := D) h
  rwa [dec_div hi, dec_div hi'] at h1

theorem dec_inj {D r r' i i' : ℕ} (hi : i < D) (hi' : i' < D) (h : r * D + i = r' * D + i') :
    r = r' ∧ i = i' := by
  have h1 : (r * D + i) / D = (r' * D + i') / D := by rw [h]
  have h2 : (r * D + i) % D = (r' * D + i') % D := by rw [h]
  rw [dec_div hi, dec_div hi'] at h1
  rw [dec_mod hi, dec_mod hi'] at h2
  exact ⟨h1, h2⟩

theorem blk_lt {R M r k : ℕ} (hr : r < R) (hk : k < M) : r * M + k < R * M := by
  have h1 : (r + 1) * M ≤ R * M := Nat.mul_le_mul_right M hr
  rw [Nat.add_one_mul] at h1
  omega

/-! ## The KV pool -/

/-- A value of the pool: token `i` of request `r`'s KV (`kv r i`), or request `r`'s read gate (`att r`). -/
inductive Val : Type where
  | kv : ℕ → ℕ → Val
  | att : ℕ → Val
  deriving DecidableEq

/-- A token's bits: two to a 32,768-bit page. -/
def tw : ℕ := 16384

theorem tw_eq : tw = 16384 := rfl

/-- The values the run produces: `2N + 3` tokens and a read gate for each of the `n·P` requests. -/
def ran (n P N : ℕ) : Finset Val :=
  ((Finset.range (n * P) ×ˢ Finset.range (2 * N + 3)).image fun p => Val.kv p.1 p.2) ∪
    (Finset.range (n * P)).image Val.att

theorem kv_mem_ran {n P N r i : ℕ} : Val.kv r i ∈ ran n P N ↔ r < n * P ∧ i < 2 * N + 3 := by
  simp [ran]

theorem att_mem_ran {n P N r : ℕ} : Val.att r ∈ ran n P N ↔ r < n * P := by
  simp [ran]

/-- A gate's operands: the read gate of request `r` reads its `2N + 3` tokens. -/
def ops (N : ℕ) : Val → Finset Val
  | .kv _ _ => ∅
  | .att r => (Finset.range (2 * N + 3)).image (Val.kv r)

theorem mem_ops {N : ℕ} {v g : Val} :
    v ∈ ops N g ↔ ∃ r i, g = Val.att r ∧ v = Val.kv r i ∧ i < 2 * N + 3 := by
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

/-- Request `r` produces token `i` at step `r·(2N+4) + i` and reads its tokens at step `r·(2N+4) + 2N+3`. -/
def step (N : ℕ) : Val → ℕ
  | .kv r i => r * (2 * N + 4) + i
  | .att r => r * (2 * N + 4) + (2 * N + 3)

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

/-- A page's block: page `i/2` of request `r` is block `r·(N+2) + i/2`; read gates sit past every page. -/
def block (n P N : ℕ) : Val → ℕ
  | .kv r i => r * (N + 2) + i / 2
  | .att r => n * P * (N + 2) + r

/-- A token's offset in its page: the even one first. -/
def offset : Val → ℕ
  | .kv _ i => i % 2 * tw
  | .att _ => 0

/-- Deferred residency: page `k` of request `r` is first read encoded at step `r·(2N+4) + 2k + 4`, two steps after
the step that completes it. -/
def encoded (N b : ℕ) : ℕ := b / (N + 2) * (2 * N + 4) + 2 * (b % (N + 2)) + 4

def kvLayout (n P N : ℕ) : Requirement.Store.Placement Val := ⟨block n P N, offset, encoded N⟩

/-- A value's group: its request, layer `0`, plane `0` and page parity. -/
def kvGrp : Val → ℕ × ℕ × ℕ × ℕ
  | .kv r i => (r, 0, 0, i / 2 % 2)
  | .att r => (r, 0, 0, 2)

/-- The memory map: a page is written in the record of the step before its first encoded read and freed at its
request's read step; its payload is its two tokens. -/
def kvStore (n P N : ℕ) (val : ℕ → ℕ → ℕ) : Requirement.Store where
  registered := 0
  written s := (Finset.range (n * P * (N + 2))).filter fun b => encoded N b = s + 1
  freed s := (Finset.range (n * P * (N + 2))).filter fun b => b / (N + 2) * (2 * N + 4) + (2 * N + 3) = s
  payload b := val (b / (N + 2)) (2 * (b % (N + 2))) + val (b / (N + 2)) (2 * (b % (N + 2)) + 1) * 2 ^ tw

/-- Pass `j` of window `w` serves request `j + P·w`. -/
def req {n P : ℕ} (w : Fin n) (j : Fin P) : ℕ := (j : ℕ) + P * (w : ℕ)

theorem req_lt {n P : ℕ} (w : Fin n) (j : Fin P) : req w j < n * P := by
  unfold req
  have hj := j.isLt
  have hw := w.isLt
  have h1 : P * ((w : ℕ) + 1) ≤ P * n := Nat.mul_le_mul_left P hw
  rw [Nat.mul_add_one] at h1
  rw [Nat.mul_comm n P]
  omega

/-- Each pass is bound to its request's read step. -/
def passBind (n P N : ℕ) (w : Fin n) (j : Fin P) : ℕ := req w j * (2 * N + 4) + (2 * N + 3)

/-- Each pass plans its request's complete pages `0 … N−1`. -/
def kvPlan (n P N : ℕ) (w : Fin n) (j : Fin P) : Finset ℕ :=
  (Finset.range N).image fun k => req w j * (N + 2) + k

/-- One append a page: page block `b` is append `1 + b`. -/
def kvAddr (n P N b : ℕ) : Fin (1 + n * P * (N + 2)) × Fin 1 :=
  if h : b < n * P * (N + 2) then (⟨1 + b, by omega⟩, 0) else (⟨0, by omega⟩, 0)

theorem kvAddr_page {n P N b : ℕ} (hb : b < n * P * (N + 2)) : kvAddr n P N b = (⟨1 + b, by omega⟩, 0) := by
  simp [kvAddr, hb]

/-- Pass `j` of window `w` certifies its request's pages `0 … N−1` at its `N` tiles. -/
def slot (n P N : ℕ) (w : Fin n) (j : Fin P) : Fin N ↪ Fin (1 + n * P * (N + 2)) × Fin 1 where
  toFun i := (⟨1 + (req w j * (N + 2) + i), by
    have hi := i.isLt
    have := blk_lt (req_lt w j) (show (i : ℕ) < N + 2 by omega)
    omega⟩, 0)
  inj' a b h := by
    have h1 := congrArg (fun p => (p.1 : ℕ)) h
    try simp only at h1
    exact Fin.ext (by omega)

/-- The audit's layout: the pass's slots, read from the prefix's window and pass only. -/
def passLayout (n P N ρ : ℕ) (Ω Ca : Type) :
    RateSketch.SketchPrefix n P N 128 ρ 512 16 1 2773 → Ω × (Fin (1 + n * P * (N + 2)) → Ca) →
      Fin N ↪ Fin (1 + n * P * (N + 2)) × Fin 1 :=
  fun c _ => slot n P N c.w c.j

/-- The `0`-bit summary. -/
def steer0 (n P N ρ : ℕ) (Ω Ca : Type) :
    RateSketch.SketchPrefix n P N 128 ρ 512 16 1 2773 → Ω × (Fin (1 + n * P * (N + 2)) → Ca) → Pous.Bits 0 :=
  fun _ _ i => i.elim0

/-- Append `t = 1 + b` joins the window of page block `b`'s request. -/
def joined (n P N ρ : ℕ) (Ω Ca : Type) :
    RateSketch.StackedCoins Ω Ca (1 + n * P * (N + 2)) n P N ρ 1 → Fin (1 + n * P * (N + 2)) → ℕ :=
  fun _ t => ((t : ℕ) - 1) / (N + 2) / P

/-- Each window's audit: the window of step `t` is its request's, and each window certifies its passes' stores. -/
def aud (n P N : ℕ) : Gates.Audit := ⟨n, fun t => t / (2 * N + 4) / P, fun _ => P * N * 4096⟩

/-! ## The requirement side -/

section Requirement

variable {n P N τ : ℕ} (val : ℕ → ℕ → ℕ)

theorem reads_eq {r : ℕ} (hr : r < n * P) :
    Requirement.reads (wiring N) (placement n P N τ val).toExecution (r * (2 * N + 4) + (2 * N + 3)) =
      (Finset.range (2 * N + 3)).image (Val.kv r) := by
  ext v
  simp only [Requirement.reads, Finset.mem_filter, Finset.mem_image, Finset.mem_range]
  constructor
  · rintro ⟨-, -, g, -, hgs, hv⟩
    change v ∈ ops N g at hv
    obtain ⟨r', i, rfl, rfl, hi⟩ := mem_ops.1 hv
    change r' * (2 * N + 4) + (2 * N + 3) = r * (2 * N + 4) + (2 * N + 3) at hgs
    have hrr := (dec_inj (show 2 * N + 3 < 2 * N + 4 by omega) (show 2 * N + 3 < 2 * N + 4 by omega) hgs).1
    subst hrr
    exact ⟨i, hi, rfl⟩
  · rintro ⟨i, hi, rfl⟩
    refine ⟨kv_mem_ran.2 ⟨hr, hi⟩, Or.inr ?_, Val.att r, att_mem_ran.2 hr, rfl, mem_ops.2 ⟨r, i, rfl, rfl, hi⟩⟩
    show r * (2 * N + 4) + i < r * (2 * N + 4) + (2 * N + 3)
    omega

theorem live_eq {r : ℕ} (hr : r < n * P) :
    Requirement.live (wiring N) (placement n P N τ val).toExecution (r * (2 * N + 4) + (2 * N + 3)) =
      (Finset.range (2 * N + 3)).image (Val.kv r) := by
  ext v
  simp only [Requirement.live, Finset.mem_filter, Finset.mem_image, Finset.mem_range]
  constructor
  · rintro ⟨-, hlt, g, -, hgs, hv⟩
    change v ∈ ops N g at hv
    obtain ⟨r', i, rfl, rfl, hi⟩ := mem_ops.1 hv
    change r * (2 * N + 4) + (2 * N + 3) ≤ r' * (2 * N + 4) + (2 * N + 3) at hgs
    have hlt' : r' * (2 * N + 4) + i < r * (2 * N + 4) + (2 * N + 3) := by
      rcases hlt with h | h
      · exact absurd h (Finset.notMem_empty _)
      · exact h
    have h1 := dec_le (show 2 * N + 3 < 2 * N + 4 by omega) (show 2 * N + 3 < 2 * N + 4 by omega) hgs
    have h2 := dec_le (show i < 2 * N + 4 by omega) (show 2 * N + 3 < 2 * N + 4 by omega) hlt'.le
    have hrr : r' = r := le_antisymm h2 h1
    subst hrr
    exact ⟨i, hi, rfl⟩
  · rintro ⟨i, hi, rfl⟩
    refine ⟨kv_mem_ran.2 ⟨hr, hi⟩, Or.inr ?_, Val.att r, att_mem_ran.2 hr, le_refl _, mem_ops.2 ⟨r, i, rfl, rfl, hi⟩⟩
    show r * (2 * N + 4) + i < r * (2 * N + 4) + (2 * N + 3)
    omega

theorem encoded_page {r k : ℕ} (hk : k < N + 2) : encoded N (r * (N + 2) + k) = r * (2 * N + 4) + 2 * k + 4 := by
  unfold encoded
  rw [dec_div hk, dec_mod hk]

/-- At its request's read step, page `i/2` is incomplete exactly from page `N` on. -/
theorem block_mem_incomplete {r i : ℕ} (hr : r < n * P) (hi : i < 2 * N + 3) :
    r * (N + 2) + i / 2 ∈ Requirement.incomplete (wiring N) (placement n P N τ val).toExecution (kvLayout n P N)
      (r * (2 * N + 4) + (2 * N + 3)) ↔ 2 * N ≤ i := by
  simp only [Requirement.incomplete, Finset.mem_image, Finset.mem_filter]
  constructor
  · rintro ⟨v, ⟨hv, -, hs⟩, hb⟩
    cases v with
    | kv r' i' =>
      obtain ⟨hr', hi'⟩ := kv_mem_ran.1 hv
      change r' * (N + 2) + i' / 2 = r * (N + 2) + i / 2 at hb
      change r * (2 * N + 4) + (2 * N + 3) ≤ r' * (2 * N + 4) + i' ∨
        r * (2 * N + 4) + (2 * N + 3) < encoded N (r' * (N + 2) + i' / 2) at hs
      rw [encoded_page (show i' / 2 < N + 2 by omega)] at hs
      obtain ⟨hrr, hk⟩ := dec_inj (show i' / 2 < N + 2 by omega) (show i / 2 < N + 2 by omega) hb
      subst hrr
      omega
    | att r' =>
      have hr' := att_mem_ran.1 hv
      change n * P * (N + 2) + r' = r * (N + 2) + i / 2 at hb
      have := blk_lt hr (show i / 2 < N + 2 by omega)
      omega
  · intro h
    refine ⟨Val.kv r i, ⟨kv_mem_ran.2 ⟨hr, hi⟩, Finset.notMem_empty _, Or.inr ?_⟩, rfl⟩
    change r * (2 * N + 4) + (2 * N + 3) < encoded N (r * (N + 2) + i / 2)
    rw [encoded_page (show i / 2 < N + 2 by omega)]
    omega

theorem filter_pages {r : ℕ} (hr : r < n * P) :
    ((Finset.range (2 * N + 3)).image (Val.kv r)).filter (fun v => (kvLayout n P N).block v ∉
      Requirement.incomplete (wiring N) (placement n P N τ val).toExecution (kvLayout n P N)
        (r * (2 * N + 4) + (2 * N + 3))) = (Finset.range (2 * N)).image (Val.kv r) := by
  ext v
  simp only [Finset.mem_filter, Finset.mem_image, Finset.mem_range]
  constructor
  · rintro ⟨⟨i, hi, rfl⟩, hb⟩
    have hb' : ¬ (r * (N + 2) + i / 2 ∈ Requirement.incomplete (wiring N) (placement n P N τ val).toExecution
        (kvLayout n P N) (r * (2 * N + 4) + (2 * N + 3))) := hb
    rw [block_mem_incomplete val hr hi] at hb'
    exact ⟨i, by omega, rfl⟩
  · rintro ⟨i, hi, rfl⟩
    refine ⟨⟨i, by omega, rfl⟩, ?_⟩
    show ¬ (r * (N + 2) + i / 2 ∈ Requirement.incomplete (wiring N) (placement n P N τ val).toExecution
        (kvLayout n P N) (r * (2 * N + 4) + (2 * N + 3)))
    rw [block_mem_incomplete val hr (show i < 2 * N + 3 by omega)]
    omega

/-- The read step reads its complete pages' tokens `0 … 2N−1` from complete blocks. -/
theorem creditedReads_eq {r : ℕ} (hr : r < n * P) :
    Requirement.creditedReads (wiring N) (placement n P N τ val).toExecution (kvLayout n P N)
      (r * (2 * N + 4) + (2 * N + 3)) = (Finset.range (2 * N)).image (Val.kv r) := by
  rw [Requirement.creditedReads, reads_eq val hr]
  exact filter_pages val hr

theorem credited_eq {r : ℕ} (hr : r < n * P) :
    Requirement.credited (wiring N) (placement n P N τ val).toExecution (kvLayout n P N)
      (r * (2 * N + 4) + (2 * N + 3)) = (Finset.range (2 * N)).image (Val.kv r) := by
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
        change r * (N + 2) + i / 2 = r' * (N + 2) + i' / 2 at hb
        obtain ⟨hrr, hk⟩ := dec_inj (show i / 2 < N + 2 by omega) (show i' / 2 < N + 2 by omega) hb
        subst hrr
        have hii : i ≠ i' := by
          rintro rfl
          exact huv rfl
        show i % 2 * tw + tw ≤ i' % 2 * tw ∨ i' % 2 * tw + tw ≤ i % 2 * tw
        rw [tw_eq]
        omega
      | att r' =>
        change r * (N + 2) + i / 2 = n * P * (N + 2) + r' at hb
        have := blk_lt hr (show i / 2 < N + 2 by omega)
        omega
    | att r =>
      have hr := att_mem_ran.1 hu'
      cases v with
      | kv r' i' =>
        obtain ⟨hr', hi'⟩ := kv_mem_ran.1 hv'
        change n * P * (N + 2) + r = r' * (N + 2) + i' / 2 at hb
        have := blk_lt hr' (show i' / 2 < N + 2 by omega)
        omega
      | att r' =>
        change n * P * (N + 2) + r = n * P * (N + 2) + r' at hb
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
  have hk : i / 2 < N + 2 := by omega
  show (val ((r * (N + 2) + i / 2) / (N + 2)) (2 * ((r * (N + 2) + i / 2) % (N + 2))) +
      val ((r * (N + 2) + i / 2) / (N + 2)) (2 * ((r * (N + 2) + i / 2) % (N + 2)) + 1) * 2 ^ tw) /
      2 ^ (i % 2 * tw) % 2 ^ tw = val r i
  rw [dec_div hk, dec_mod hk]
  have hp : 0 < 2 ^ tw := Nat.two_pow_pos tw
  rcases Nat.mod_two_eq_zero_or_one i with h | h
  · have hi2 : 2 * (i / 2) = i := by omega
    rw [h, hi2, Nat.zero_mul, Nat.pow_zero, Nat.div_one, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (hval r i)]
  · have hi2 : 2 * (i / 2) + 1 = i := by omega
    rw [h, hi2, Nat.one_mul, Nat.add_mul_div_right _ _ hp, Nat.div_eq_of_lt (hval _ _), Nat.zero_add,
      Nat.mod_eq_of_lt (hval r i)]

/-- A complete page is resident at its request's read step: written before, freed only then. -/
theorem resident_page {n P N r i : ℕ} (val : ℕ → ℕ → ℕ) (hr : r < n * P) (hi : i < 2 * N) :
    r * (N + 2) + i / 2 ∈ (kvStore n P N val).resident (r * (2 * N + 4) + (2 * N + 3)) := by
  have hk : i / 2 < N + 2 := by omega
  have hb : r * (N + 2) + i / 2 < n * P * (N + 2) := blk_lt hr hk
  simp only [Requirement.Store.resident, Finset.mem_sdiff, Finset.mem_union, Finset.mem_biUnion, Finset.mem_range]
  refine ⟨Or.inr ⟨r * (2 * N + 4) + 2 * (i / 2) + 3, by omega, ?_⟩, ?_⟩
  · show r * (N + 2) + i / 2 ∈ (Finset.range (n * P * (N + 2))).filter fun b => encoded N b = _ + 1
    rw [Finset.mem_filter, Finset.mem_range, encoded_page hk]
    exact ⟨hb, by omega⟩
  · rintro ⟨s', hs', hf⟩
    change r * (N + 2) + i / 2 ∈ (Finset.range (n * P * (N + 2))).filter
      (fun b => b / (N + 2) * (2 * N + 4) + (2 * N + 3) = s') at hf
    rw [Finset.mem_filter, dec_div hk] at hf
    omega

theorem storeMatches {n P N τ r : ℕ} (val : ℕ → ℕ → ℕ) (hval : ∀ r i, val r i < 2 ^ tw) (hr : r < n * P) :
    Requirement.StoreMatches (wiring N) (placement n P N τ val).toExecution (kvLayout n P N) (kvStore n P N val)
      32768 (r * (2 * N + 4) + (2 * N + 3)) := by
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
    (placement n P N τ val).toExecution (kvLayout n P N) (req w j * (2 * N + 4) + (2 * N + 3))) ⊆ kvPlan n P N w j
  rw [creditedReads_eq val hr]
  intro b hb
  obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hb
  obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hv
  rw [Finset.mem_range] at hi
  exact Finset.mem_image.2 ⟨i / 2, Finset.mem_range.2 (by omega), rfl⟩

/-- The tokens read from incomplete pages (pages `N` and `N+1`) lie in one page a group. -/
theorem oneBlockEach {n P N τ r : ℕ} (val : ℕ → ℕ → ℕ) (hr : r < n * P) :
    Requirement.OneBlockEach (kvLayout n P N) kvGrp
      (Requirement.reads (wiring N) (placement n P N τ val).toExecution (r * (2 * N + 4) + (2 * N + 3)) \
        Requirement.creditedReads (wiring N) (placement n P N τ val).toExecution (kvLayout n P N)
          (r * (2 * N + 4) + (2 * N + 3))) := by
  rw [reads_eq val hr, creditedReads_eq val hr]
  intro k
  rw [Finset.card_le_one]
  intro a ha b hb
  obtain ⟨u, hu, rfl⟩ := Finset.mem_image.1 ha
  obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hb
  rw [Finset.mem_filter, Finset.mem_sdiff] at hu hv
  obtain ⟨⟨hu1, hu2⟩, hu3⟩ := hu
  obtain ⟨⟨hv1, hv2⟩, hv3⟩ := hv
  obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hu1
  obtain ⟨i', hi', rfl⟩ := Finset.mem_image.1 hv1
  rw [Finset.mem_range] at hi hi'
  have hi2 : 2 * N ≤ i := by
    by_contra h
    exact hu2 (Finset.mem_image.2 ⟨i, Finset.mem_range.2 (by omega), rfl⟩)
  have hi2' : 2 * N ≤ i' := by
    by_contra h
    exact hv2 (Finset.mem_image.2 ⟨i', Finset.mem_range.2 (by omega), rfl⟩)
  have hg : (kvGrp (Val.kv r i)).2.2.2 = (kvGrp (Val.kv r i')).2.2.2 := by rw [hu3, hv3]
  change i / 2 % 2 = i' / 2 % 2 at hg
  show r * (N + 2) + i / 2 = r * (N + 2) + i' / 2
  omega

theorem plan_lt {n P N : ℕ} {w : Fin n} {j : Fin P} {b : ℕ} (hb : b ∈ kvPlan n P N w j) :
    ∃ k < N, b = req w j * (N + 2) + k ∧ b < n * P * (N + 2) := by
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

theorem disjoint_ticks {τ t t' : ℕ} (h : t ≠ t') :
    Disjoint (Finset.Ico (t * τ) (t * τ + τ)) (Finset.Ico (t' * τ) (t' * τ + τ)) := by
  rw [Finset.disjoint_left]
  intro x hx hx'
  rw [Finset.mem_Ico] at hx hx'
  rcases Nat.lt_or_gt_of_ne h with h | h
  · have h1 : (t + 1) * τ ≤ t' * τ := Nat.mul_le_mul_right τ h
    rw [Nat.add_one_mul] at h1
    omega
  · have h1 : (t' + 1) * τ ≤ t * τ := Nat.mul_le_mul_right τ h
    rw [Nat.add_one_mul] at h1
    omega

theorem passBind_inj {n P N : ℕ} {w w' : Fin n} {j j' : Fin P} (h : passBind n P N w j = passBind n P N w' j') :
    w = w' ∧ j = j' := by
  unfold passBind at h
  have h1 := (dec_inj (show 2 * N + 3 < 2 * N + 4 by omega) (show 2 * N + 3 < 2 * N + 4 by omega) h).1
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
  have := Nat.mul_lt_mul_of_pos_right hr (show 0 < 2 * N + 4 by omega)
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
  exact ⟨storeMatches val hval hr, planMatches val w j, oneBlockEach val hr, injOn_addr w j,
    fun b hb => addr_mem_range w j hb⟩

/-! ## The audit side -/

/-- The honest pass device (`RateSketch.Completeness.honest`) at the session's on-chip state, link and query caps. -/
def honestDev (N ρ S Lk q : ℕ) (hL : ρ * 256 ≤ Lk) (hq : N ≤ q) :
    RateSketch.Device N 128 ρ 512 S Lk q 256 RateSketch.Sector (RateSketch.SketchSigns 16 128)
      (Fin 16 → ZMod (2 ^ 32)) where
  held := ∅
  held_le := by simp
  chip σ _ := RateSketch.Completeness.chip RateSketch.sketch σ
  bounded σ _ O D := by
    obtain ⟨-, h2, -, h4, -⟩ := RateSketch.Completeness.chip_eq (h := 512) RateSketch.sketch σ O D
    exact ⟨by rw [h4]; exact hL, h2.le.trans hq⟩

/-- Every pass runs the honest device, with one grinding candidate and a `0`-bit summary. -/
def honestSession (n P N ρ S Lk q : ℕ) (hL : ρ * 256 ≤ Lk) (hq : N ≤ q) :
    RateSketch.SketchSummarySession 0 n P N 128 ρ 512 S Lk q 16 1 2773 :=
  ⟨fun _ => ⟨fun _ _ => ⟨fun _ _ _ => honestDev N ρ S Lk q hL hq, fun _ _ _ _ _ => 0⟩⟩⟩

theorem hashBudget {n P N ρ S Lk q qt : ℕ} (hL : ρ * 256 ≤ Lk) (hq : N ≤ q) (hqt : N ≤ qt) :
    ∀ s w η j pre, Assumptions.SketchHashBudget
      ((((honestSession n P N ρ S Lk q hL hq).session s).window w η).device j pre) qt := by
  intro s w η j pre σ Os D
  rw [Fin.sum_univ_one]
  exact (RateSketch.Completeness.chip_eq (h := 512) RateSketch.sketch σ (Os 0) D).2.1.le.trans hqt

theorem rebuildsIn_le {N m ρ h : ℕ} {X Dg α : Type} (U : Finset (Fin N)) (O : RateSketch.Oracle N ρ h X Dg)
    (D : RateSketch.Data N m ρ X) (c : RateSketch.Chip N m ρ h X Dg α) : c.rebuildsIn U O D ≤ c.rebuilds O D := by
  induction c with
  | done _ => simp [RateSketch.Chip.rebuildsIn, RateSketch.Chip.rebuilds]
  | tag x k ih =>
    simp only [RateSketch.Chip.rebuildsIn, RateSketch.Chip.rebuilds]
    exact ih _
  | mem p k ih =>
    simp only [RateSketch.Chip.rebuildsIn, RateSketch.Chip.rebuilds]
    exact ih _
  | link p k ih =>
    simp only [RateSketch.Chip.rebuildsIn, RateSketch.Chip.rebuilds]
    exact ih _
  | rebuild p k ih =>
    simp only [RateSketch.Chip.rebuildsIn, RateSketch.Chip.rebuilds]
    have h1 := ih (D (.inl p))
    have h2 : (if p.1 ∈ U then 1 else 0) ≤ 1 := by split <;> omega
    omega

/-- The empty carrier. -/
def noCarrier (N : ℕ) : RateSketch.Carrier N 128 where
  carried := ∅
  provenance := .done fun p => absurd p.2 (Finset.notMem_empty _)
  holds _ := funext fun p => absurd p.2 (Finset.notMem_empty _)

/-- The honest devices rebuild nothing, so every pass meets `CircuitCarriers` with no carrier, at every store,
layout, life and circuit. -/
theorem circuitCarriers {n P N ρ S Lk q X L B εnum εden T M : ℕ} {ι Y : Type} (hL : ρ * 256 ≤ Lk) (hq : N ≤ q)
    (steer : RateSketch.SketchPrefix n P N 128 ρ 512 16 1 2773 → Y → Pous.Bits 0)
    (store : RateSketch.SketchPrefix n P N 128 ρ 512 16 1 2773 → Y → RateSketch.SketchStore N 128)
    (layout : RateSketch.SketchPrefix n P N 128 ρ 512 16 1 2773 → Y → Fin N ↪ Fin T × Fin M)
    (joined : Y × (Fin n → RateSketch.SketchWindowCoins P N 128 ρ 512 16 1 2773) → Fin T → ℕ)
    (Wc : Gates.Dag ι) (Aud : Gates.Audit) :
    Assumptions.CircuitCarriers (fun c y => (honestSession n P N ρ S Lk q hL hq).session (steer c y)) store layout
      joined X L B εnum εden Wc Aud := by
  intro _ c g y
  refine ⟨noCarrier N, ∅, fun R O => ?_, Finset.empty_subset _, by simp,
    fun E => ⟨by simp [noCarrier, RateSketch.Provenance.inputs], by simp [noCarrier, RateSketch.Provenance.gates]⟩⟩
  intro dv D
  have h5 : (dv.prog c.signs D).rebuilds O D = 0 :=
    (RateSketch.Completeness.chip_eq (h := 512) RateSketch.sketch c.signs O D).2.2.2.2
  have h0 := rebuildsIn_le (RateSketch.appendedTiles layout c y) O D (dv.prog c.signs D)
  rw [h5] at h0
  rw [Nat.le_zero.1 h0, Nat.zero_mul]
  exact Nat.zero_le _

/-- A page joins its request's window and is read only there. -/
theorem lifetimeCapped {n P N ρ L : ℕ} {Ω Ca : Type} (hL : 0 < L) :
    RateSketch.LifetimeCapped (passLayout n P N ρ Ω Ca) (joined n P N ρ Ω Ca) L := by
  intro x w j i _
  have hj := j.isLt
  have hi := i.isLt
  show (1 + (req w j * (N + 2) + (i : ℕ)) - 1) / (N + 2) / P ≤ (w : ℕ) ∧
    (w : ℕ) < (1 + (req w j * (N + 2) + (i : ℕ)) - 1) / (N + 2) / P + L
  have h1 : (1 + (req w j * (N + 2) + (i : ℕ)) - 1) / (N + 2) / P = w := by
    rw [show 1 + (req w j * (N + 2) + (i : ℕ)) - 1 = req w j * (N + 2) + (i : ℕ) by omega,
      dec_div (show (i : ℕ) < N + 2 by omega)]
    unfold req
    rw [show (j : ℕ) + P * (w : ℕ) = (w : ℕ) * P + j by rw [Nat.mul_comm]; omega, dec_div hj]
  rw [h1]
  omega

theorem restricted (B : ℕ) : Gates.Restricted B ([] : Gates.Dag Unit) :=
  ⟨fun i h => absurd h (Nat.not_lt_zero i), fun g hg => by cases hg⟩

theorem property (caps : Gates.Caps)
    (hc : 0 ≤ caps.rounding ∧ 0 ≤ caps.compareSelect ∧ 0 ≤ caps.gather ∧ 0 ≤ caps.fold ∧ 0 ≤ caps.products)
    (Aud : Gates.Audit) :
    Gates.Property caps Aud ([] : Gates.Dag Unit) := by
  obtain ⟨h1, h2, h3, h4, h7⟩ := hc
  refine ⟨fun g hg => (by cases hg), fun w _ => ?_, fun i h => absurd h (Nat.not_lt_zero i)⟩
  have hu : ∀ b, Gates.unitsAt ([] : Gates.Dag Unit) (fun t => Aud.win t = w) b = 0 := fun b => by
    simp [Gates.unitsAt]
  have hc0 : ∀ f, Gates.countAt ([] : Gates.Dag Unit) (fun t => Aud.win t = w) f = 0 := fun f => by
    simp [Gates.countAt]
  have ht : Gates.productsAt ([] : Gates.Dag Unit) (fun t => Aud.win t = w) = 0 := by
    simp [Gates.productsAt]
  intro c
  have e1 : c.rounding = 0 := hu .rounding
  have e2 : c.compareSelect = 0 := hu .compareSelect
  have e3 : c.gather = 0 := hu .gather
  have e4 : c.fold = 0 := hu .fold
  have e5 : c.neutral = 0 := hu .neutral
  have e6 : c.runBits = 0 := hc0 (Gates.Gate.inputBits .run)
  rw [e1, e2, e3, e4, e5, e6, ht, Nat.cast_zero]
  exact ⟨mul_nonneg h1 (Nat.cast_nonneg _), mul_nonneg h2 (Nat.cast_nonneg _), mul_nonneg h3 (Nat.cast_nonneg _),
    mul_nonneg h4 (Nat.cast_nonneg _), rfl, Nat.zero_le _, mul_nonneg h7 (Nat.cast_nonneg _)⟩

/-! ## The witness -/

/-- **An honest run meets every hypothesis of `DeclaredCircuitRequirement`** but the world's (`SketchRank`,
`KeyBinding`, `SquaringFloor`, `ChainSum` and their floor), at every value of the session's parameters with the link
carrying the stream (`ρ·256 ≤ L_r·τ`), `N` queries within the caps and a carrier life `L ≥ 1`, at any nonnegative
caps, codec and append choices, and any token bits. -/
theorem meets (n P N ρ S q Xr Lr qr τ L B εnum εden : ℕ) {Ω Ca Ch : Type}
    (enc : Ω → Fin (1 + n * P * (N + 2)) → Ch → (Fin (1 + n * P * (N + 2)) → Ca) → Fin 1 →
      (Fin 128 → RateSketch.Sector))
    (choose : RateSketch.SketchPrefix n P N 128 ρ 512 16 1 2773 → Ω → (a : Fin (1 + n * P * (N + 2))) →
      ({b : Fin (1 + n * P * (N + 2)) // b < a} → Ca) → Ch)
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
    (∀ x : RateSketch.StackedCoins Ω Ca (1 + n * P * (N + 2)) n P N ρ 1,
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

/-- **`G` at the honest run**: given `G` and the world's assumptions, its conclusion holds at the run of `meets`. -/
theorem applies (hG : Guarantees.DeclaredCircuitRequirement) (n P N ρ S q Xr Lr qr τ k L E es B εnum εden : ℕ)
    (ε δ : ℝ≥0∞) {Ω Ca Ch : Type} [Fintype Ω] [Fintype Ca]
    (enc : Ω → Fin (1 + n * P * (N + 2)) → Ch → (Fin (1 + n * P * (N + 2)) → Ca) → Fin 1 →
      (Fin 128 → RateSketch.Sector))
    (choose : RateSketch.SketchPrefix n P N 128 ρ 512 16 1 2773 → Ω → (a : Fin (1 + n * P * (N + 2))) →
      ({b : Fin (1 + n * P * (N + 2)) // b < a} → Ca) → Ch)
    (caps : Gates.Caps)
    (hcaps : 0 ≤ caps.rounding ∧ 0 ≤ caps.compareSelect ∧ 0 ≤ caps.gather ∧ 0 ≤ caps.fold ∧ 0 ≤ caps.products)
    (val : ℕ → ℕ → ℕ) (hval : ∀ r i, val r i < 2 ^ tw) (hn : 7 ≤ n) (hP : 0 < P) (hN : 0 < N) (hE : 0 < E)
    (hL : 0 < L) (hlink : ρ * 256 ≤ Lr * τ) (hq : N ≤ q) (hqr : N ≤ qr * τ)
    (hrank : Assumptions.SketchRank 16 128 12 ε) (hkey : Assumptions.KeyBinding (1 + n * P * (N + 2)) 1 δ enc)
    (hsq : Assumptions.SquaringFloor Gates.p1024 es εnum εden) (hcs : Assumptions.ChainSum Gates.p1024 3 B es εnum εden)
    (hEle : E ≤ 3 * Gates.p1024.squarings * es) (hfloor : εden ≤ 2 ^ 64 * εnum) :
    let T := 1 + n * P * (N + 2)
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

/-! ## The credit fills the store -/

theorem bits_image_kv {N r K : ℕ} : Requirement.bits (wiring N) ((Finset.range K).image (Val.kv r)) = K * tw := by
  unfold Requirement.bits
  show ∑ _v ∈ (Finset.range K).image (Val.kv r), tw = K * tw
  rw [Finset.sum_const, Finset.card_image_of_injective _ (fun a b h => (Val.kv.inj h).2), Finset.card_range,
    smul_eq_mul]

/-- **Each pass's credit is at least its `N` tiles' bits**: the held part alone, `W`'s credited bits in its owned
pages, is the request's `N` complete pages. -/
theorem credit_ge {n P N τ : ℕ} (val : ℕ → ℕ → ℕ) (w : Fin n) (j : Fin P) :
    let e := (placement n P N τ val).toExecution
    let owned := (Requirement.blocks (kvLayout n P N)
      (Requirement.creditedReads (wiring N) e (kvLayout n P N) (passBind n P N w j))).filter
        fun b => ((kvAddr n P N b).1 : ℕ) ≠ 0
    N * 32768 ≤ max (Requirement.bitsAt (wiring N) (kvLayout n P N)
        (Requirement.credited (wiring N) e (kvLayout n P N) (passBind n P N w j)) owned)
      (Requirement.bits (wiring N) ((Requirement.reads (wiring N) e (passBind n P N w j)).filter
          fun v => ((kvAddr n P N ((kvLayout n P N).block v)).1 : ℕ) ≠ 0) -
        ((Requirement.reads (wiring N) e (passBind n P N w j) \
          Requirement.creditedReads (wiring N) e (kvLayout n P N) (passBind n P N w j)).image kvGrp).card * 32768) := by
  intro e owned
  refine le_trans (le_of_eq ?_) (le_max_left _ _)
  have hr := req_lt w j
  have hc : Requirement.credited (wiring N) e (kvLayout n P N) (passBind n P N w j) =
      (Finset.range (2 * N)).image (Val.kv (req w j)) := credited_eq val hr
  have hcr : Requirement.creditedReads (wiring N) e (kvLayout n P N) (passBind n P N w j) =
      (Finset.range (2 * N)).image (Val.kv (req w j)) := creditedReads_eq val hr
  have hown : ∀ v ∈ (Finset.range (2 * N)).image (Val.kv (req w j)), (kvLayout n P N).block v ∈ owned := by
    intro v hv
    obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hv
    rw [Finset.mem_range] at hi
    have hmem : req w j * (N + 2) + i / 2 ∈ Requirement.blocks (kvLayout n P N)
        (Requirement.creditedReads (wiring N) e (kvLayout n P N) (passBind n P N w j)) := by
      rw [hcr]
      exact Finset.mem_image.2 ⟨_, hv, rfl⟩
    refine Finset.mem_filter.2 ⟨hmem, ?_⟩
    show ((kvAddr n P N (req w j * (N + 2) + i / 2)).1 : ℕ) ≠ 0
    rw [kvAddr_page (blk_lt hr (show i / 2 < N + 2 by omega))]
    show 1 + (req w j * (N + 2) + i / 2) ≠ 0
    omega
  unfold Requirement.bitsAt
  rw [hc, Finset.filter_true_of_mem hown, bits_image_kv, tw_eq]
  omega

/-- A credit of at least `c` a pass sums to at least `n·P·c` over the session. -/
theorem sum_ge {n P c : ℕ} (f : Fin n → Fin P → ℕ) (h : ∀ w j, c ≤ f w j) : n * P * c ≤ ∑ w, ∑ j, f w j := by
  calc n * P * c = ∑ _w : Fin n, ∑ _j : Fin P, c := by
        simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
        ring
    _ ≤ ∑ w, ∑ j, f w j := Finset.sum_le_sum fun w _ => Finset.sum_le_sum fun j _ => h w j

/-! ## Where `G` bounds something

The run credits the whole store, so `G`'s session threshold `Σ credit − (0.0015 + γ)·n·P·N·32768` is positive while
`0.0015 + γ < 1` (`threshold_pos`), and each window's while `0.01 + γ < 1` (`window_pos`). The carriers' share of `γ` is
linear in the life `L`: `C/(N·32768) ≤ (8·L·R + 1)/(N·32768) + 128·L·P·c/E` (`carrier_le`), `R` a window's run-input
bytes and `c` the caps' total. At the reference inputs (`P = 984` passes a 60 s window, `E = 364,497`, 100 KB/s of run
inputs, the pass's other resources `γ₀ ≤ 0.835%`, here `N ≥ 2^24` tiles) both thresholds are positive up to `L = 6` for
8B (`c = 0.442`, `bounds_8b`); for 70B TP8 (`c = 0.0918`) the session's up to `L = 31` and every window's up to `L = 30`
(`bounds_70b`). -/

/-- The run's credit at step `t`, as `G` reads it. -/
def credit (n P N τ : ℕ) (val : ℕ → ℕ → ℕ) (t : ℕ) : ℕ :=
  let e := (placement n P N τ val).toExecution
  let owned := (Requirement.blocks (kvLayout n P N)
    (Requirement.creditedReads (wiring N) e (kvLayout n P N) t)).filter fun b => ((kvAddr n P N b).1 : ℕ) ≠ 0
  max (Requirement.bitsAt (wiring N) (kvLayout n P N) (Requirement.credited (wiring N) e (kvLayout n P N) t) owned)
    (Requirement.bits (wiring N) ((Requirement.reads (wiring N) e t).filter
        fun v => ((kvAddr n P N ((kvLayout n P N).block v)).1 : ℕ) ≠ 0) -
      ((Requirement.reads (wiring N) e t \ Requirement.creditedReads (wiring N) e (kvLayout n P N) t).image
        kvGrp).card * 32768)

theorem credit_session {n P N τ : ℕ} (val : ℕ → ℕ → ℕ) :
    n * P * N * 32768 ≤ ∑ w : Fin n, ∑ j : Fin P, credit n P N τ val (passBind n P N w j) := by
  rw [Nat.mul_assoc (n * P)]
  exact sum_ge _ fun w j => credit_ge val w j

theorem credit_window {n P N τ : ℕ} (val : ℕ → ℕ → ℕ) (w : Fin n) :
    P * N * 32768 ≤ ∑ j : Fin P, credit n P N τ val (passBind n P N w j) := by
  calc P * N * 32768 = ∑ _j : Fin P, N * 32768 := by
        simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
        ring
    _ ≤ _ := Finset.sum_le_sum fun j _ => credit_ge val w j

/-- **`G`'s session threshold is positive at the honest run** while `0.0015 + γ < 1`. -/
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

/-- **Each window's threshold is positive at the honest run** while `0.01 + γ < 1`. -/
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

/-- The run-input bytes the caps allow a carrier's life: at most `L·R` when a window's are at most `R`. -/
theorem runLife_le {L R : ℕ} (caps : Gates.Caps) (Aud : Gates.Audit) (hR : ∀ v, caps.runBytes v ≤ R) :
    ((Finset.range Aud.windows).powerset.filter fun I => I.card ≤ L).sup (fun I => ∑ v ∈ I, caps.runBytes v) ≤
      L * R := by
  refine Finset.sup_le fun I hI => ?_
  have hI' := (Finset.mem_filter.1 hI).2
  calc ∑ v ∈ I, caps.runBytes v ≤ ∑ _v ∈ I, R := Finset.sum_le_sum fun v _ => hR v
    _ = I.card * R := by rw [Finset.sum_const, smul_eq_mul]
    _ ≤ L * R := Nat.mul_le_mul_right R hI'

/-- **The carriers' bits a pass, a share of its store**: `G`'s `C` over `N·32768` bits is at most
`(8·L·R + 1)/(N·32768) + 128·L·P·c/E`. -/
theorem carrier_le {N L P E R : ℕ} (caps : Gates.Caps) (Aud : Gates.Audit) (hN : 0 < N) (hE : 0 < E)
    (hc : 0 ≤ caps.total) (hR : ∀ v, caps.runBytes v ≤ R) :
    ((8 * ((Finset.range Aud.windows).powerset.filter fun I => I.card ≤ L).sup
        (fun I => ∑ v ∈ I, caps.runBytes v) + ⌈(1024 * L * (P * N * 4096) : ℚ) * caps.total / E⌉₊ : ℕ) : ℝ) /
        (N * 32768) ≤
      (8 * L * R + 1) / (N * 32768) + 128 * L * P * (caps.total : ℝ) / E := by
  have hN' : (0 : ℝ) < N * 32768 := by positivity
  have hs : ((((Finset.range Aud.windows).powerset.filter fun I => I.card ≤ L).sup
      (fun I => ∑ v ∈ I, caps.runBytes v) : ℕ) : ℝ) ≤ L * R := by
    exact_mod_cast runLife_le caps Aud hR
  have hx : (0 : ℚ) ≤ (1024 * L * (P * N * 4096) : ℚ) * caps.total / E :=
    div_nonneg (mul_nonneg (by positivity) hc) (by positivity)
  have hq : ((⌈(1024 * L * (P * N * 4096) : ℚ) * caps.total / E⌉₊ : ℕ) : ℝ) ≤
      (((1024 * L * (P * N * 4096) : ℚ) * caps.total / E : ℚ) : ℝ) + 1 := by
    exact_mod_cast (Nat.ceil_lt_add_one hx).le
  have hxr : (((1024 * L * (P * N * 4096) : ℚ) * caps.total / E : ℚ) : ℝ) =
      128 * L * P * (caps.total : ℝ) / E * (N * 32768) := by
    push_cast
    ring
  rw [hxr] at hq
  rw [div_le_iff₀ hN', add_mul, div_mul_cancel₀ _ hN'.ne']
  push_cast at hs hq ⊢
  linarith

/-- `γ` is the pass's other resources' share `γ₀` plus the carriers' bits a stored bit. -/
theorem gamma_le {N ρ S Lr τ Xr k C : ℕ} (γ₀ : ℝ) (hN : 0 < N)
    (h₀ : ((S + (Lr * τ - ρ * 256) + Xr * τ + k * 32768 : ℕ) : ℝ) ≤ γ₀ * (N * 32768)) :
    RateSketch.gammaSketch N 128 ρ S (Lr * τ) (Xr * τ + C + k * 32768) ≤ γ₀ + (C : ℝ) / (N * 32768) := by
  have hN' : (0 : ℝ) < N * 32768 := by positivity
  have hsum : S + (Lr * τ - ρ * 256) + (Xr * τ + C + k * 32768) =
      (S + (Lr * τ - ρ * 256) + Xr * τ + k * 32768) + C := by omega
  have key : ((S + (Lr * τ - ρ * 256) + (Xr * τ + C + k * 32768) : ℕ) : ℝ) ≤
      (γ₀ + (C : ℝ) / (N * 32768)) * (N * 32768) := by
    rw [hsum, add_mul, div_mul_cancel₀ _ hN'.ne', Nat.cast_add]
    linarith
  unfold RateSketch.gammaSketch
  rw [div_le_iff₀ (by positivity)]
  calc _ ≤ (γ₀ + (C : ℝ) / (N * 32768)) * (N * 32768) := key
    _ = _ := by push_cast; ring

/-- **`γ` at a life of at most `L₀` windows**, at most `P₀` passes a window, at least `E₀` for `E`, at least `N₀`
tiles, caps' total at most `c₀` and at most `R₀` run-input bytes a window. -/
theorem gamma_bound {N ρ S Lr τ Xr k L P E L₀ P₀ E₀ N₀ R₀ : ℕ} (caps : Gates.Caps) (Aud : Gates.Audit) (γ₀ c₀ : ℝ)
    (hN₀ : 0 < N₀) (hN : N₀ ≤ N) (hL : L ≤ L₀) (hP : P ≤ P₀) (hE₀ : 0 < E₀) (hE : E₀ ≤ E) (hc : 0 ≤ caps.total)
    (hc₀ : (caps.total : ℝ) ≤ c₀) (hR : ∀ v, caps.runBytes v ≤ R₀)
    (h₀ : ((S + (Lr * τ - ρ * 256) + Xr * τ + k * 32768 : ℕ) : ℝ) ≤ γ₀ * (N * 32768)) :
    let RL := ((Finset.range Aud.windows).powerset.filter fun I => I.card ≤ L).sup
      fun I => ∑ v ∈ I, caps.runBytes v
    let C := 8 * RL + ⌈(1024 * L * (P * N * 4096) : ℚ) * caps.total / E⌉₊
    RateSketch.gammaSketch N 128 ρ S (Lr * τ) (Xr * τ + C + k * 32768) ≤
      γ₀ + (8 * L₀ * R₀ + 1) / (N₀ * 32768) + 128 * L₀ * P₀ * c₀ / E₀ := by
  intro RL C
  have hN0 : 0 < N := lt_of_lt_of_le hN₀ hN
  have hE0 : 0 < E := lt_of_lt_of_le hE₀ hE
  have hg : RateSketch.gammaSketch N 128 ρ S (Lr * τ) (Xr * τ + C + k * 32768) ≤ γ₀ + (C : ℝ) / (N * 32768) :=
    gamma_le γ₀ hN0 h₀
  have hC : (C : ℝ) / (N * 32768) ≤ (8 * L * R₀ + 1) / (N * 32768) + 128 * L * P * (caps.total : ℝ) / E :=
    carrier_le caps Aud hN0 hE0 hc hR
  have hNr : (N₀ : ℝ) ≤ N := by exact_mod_cast hN
  have hLr : (L : ℝ) ≤ L₀ := by exact_mod_cast hL
  have hPr : (P : ℝ) ≤ P₀ := by exact_mod_cast hP
  have hEr : (E₀ : ℝ) ≤ E := by exact_mod_cast hE
  have hc0 : (0 : ℝ) ≤ caps.total := by exact_mod_cast hc
  have h1 : (8 * (L : ℝ) * R₀ + 1) / (N * 32768) ≤ (8 * L₀ * R₀ + 1) / (N₀ * 32768) := by
    have hLR := mul_le_mul_of_nonneg_right hLr (Nat.cast_nonneg (α := ℝ) R₀)
    exact div_le_div₀ (by positivity) (by linarith) (by positivity) (by linarith)
  have h2 : 128 * (L : ℝ) * P * caps.total / E ≤ 128 * L₀ * P₀ * c₀ / E₀ := by
    have s1 : 128 * (L : ℝ) ≤ 128 * L₀ := mul_le_mul_of_nonneg_left hLr (by norm_num)
    have s2 : 128 * (L : ℝ) * P ≤ 128 * L₀ * P₀ := mul_le_mul s1 hPr (Nat.cast_nonneg _) (by positivity)
    have s3 : 128 * (L : ℝ) * P * caps.total ≤ 128 * L₀ * P₀ * c₀ := mul_le_mul s2 hc₀ hc0 (by positivity)
    exact div_le_div₀ (mul_nonneg (by positivity) (hc0.trans hc₀)) s3 (by positivity) hEr
  linarith

/-- **8B**: at the reference inputs (`c ≤ 0.442`, at most 984 passes and 6,000,000 run-input bytes a 60 s window,
`E ≥ 364,497`, `γ₀ ≤ 0.835%`, `N ≥ 2^24`), both of `G`'s thresholds are positive at a carrier life of `L ≤ 6`. -/
theorem bounds_8b {N ρ S Lr τ Xr k L P E : ℕ} (caps : Gates.Caps) (Aud : Gates.Audit)
    (hN : 16777216 ≤ N) (hL : L ≤ 6) (hP : P ≤ 984) (hE : 364497 ≤ E) (hc : 0 ≤ caps.total)
    (hc₀ : caps.total ≤ 221 / 500) (hR : ∀ v, caps.runBytes v ≤ 6000000)
    (h₀ : ((S + (Lr * τ - ρ * 256) + Xr * τ + k * 32768 : ℕ) : ℝ) ≤ 0.00835 * (N * 32768)) :
    let RL := ((Finset.range Aud.windows).powerset.filter fun I => I.card ≤ L).sup
      fun I => ∑ v ∈ I, caps.runBytes v
    let C := 8 * RL + ⌈(1024 * L * (P * N * 4096) : ℚ) * caps.total / E⌉₊
    let γ := RateSketch.gammaSketch N 128 ρ S (Lr * τ) (Xr * τ + C + k * 32768)
    0.0015 + γ < 1 ∧ 0.01 + γ < 1 := by
  intro RL C γ
  have hc₀' : (caps.total : ℝ) ≤ 221 / 500 := by
    have h : ((caps.total : ℚ) : ℝ) ≤ ((221 / 500 : ℚ) : ℝ) := Rat.cast_le.2 hc₀
    rwa [show ((221 / 500 : ℚ) : ℝ) = 221 / 500 by norm_num] at h
  have hγ : γ ≤ _ := gamma_bound (L₀ := 6) (P₀ := 984) (E₀ := 364497) (N₀ := 16777216) (R₀ := 6000000) caps Aud
    0.00835 (221 / 500) (by norm_num) hN hL hP (by norm_num) hE hc hc₀' hR h₀
  norm_num at hγ
  constructor <;> linarith

/-- **70B TP8**: at the reference inputs with `c ≤ 0.0918`, the session threshold is positive at a carrier life of
`L ≤ 31`, and every window's at `L ≤ 30`. -/
theorem bounds_70b {N ρ S Lr τ Xr k L P E : ℕ} (caps : Gates.Caps) (Aud : Gates.Audit)
    (hN : 16777216 ≤ N) (hP : P ≤ 984) (hE : 364497 ≤ E) (hc : 0 ≤ caps.total)
    (hc₀ : caps.total ≤ 459 / 5000) (hR : ∀ v, caps.runBytes v ≤ 6000000)
    (h₀ : ((S + (Lr * τ - ρ * 256) + Xr * τ + k * 32768 : ℕ) : ℝ) ≤ 0.00835 * (N * 32768)) :
    let RL := ((Finset.range Aud.windows).powerset.filter fun I => I.card ≤ L).sup
      fun I => ∑ v ∈ I, caps.runBytes v
    let C := 8 * RL + ⌈(1024 * L * (P * N * 4096) : ℚ) * caps.total / E⌉₊
    let γ := RateSketch.gammaSketch N 128 ρ S (Lr * τ) (Xr * τ + C + k * 32768)
    (L ≤ 31 → 0.0015 + γ < 1) ∧ (L ≤ 30 → 0.01 + γ < 1) := by
  intro RL C γ
  have hc₀' : (caps.total : ℝ) ≤ 459 / 5000 := by
    have h : ((caps.total : ℚ) : ℝ) ≤ ((459 / 5000 : ℚ) : ℝ) := Rat.cast_le.2 hc₀
    rwa [show ((459 / 5000 : ℚ) : ℝ) = 459 / 5000 by norm_num] at h
  refine ⟨fun hL => ?_, fun hL => ?_⟩
  · have hγ : γ ≤ _ := gamma_bound (L₀ := 31) (P₀ := 984) (E₀ := 364497) (N₀ := 16777216) (R₀ := 6000000) caps Aud
      0.00835 (459 / 5000) (by norm_num) hN hL hP (by norm_num) hE hc hc₀' hR h₀
    norm_num at hγ
    linarith
  · have hγ : γ ≤ _ := gamma_bound (L₀ := 30) (P₀ := 984) (E₀ := 364497) (N₀ := 16777216) (R₀ := 6000000) caps Aud
      0.00835 (459 / 5000) (by norm_num) hN hL hP (by norm_num) hE hc hc₀' hR h₀
    norm_num at hγ
    linarith

end Accounting.SecurityProofs.DeclaredCircuitWitness
