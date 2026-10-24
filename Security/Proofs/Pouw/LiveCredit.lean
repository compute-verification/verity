import Mathlib
import Proofs.Accounting.TS.Session.Lanes.Layer

/-!
# Dense credit counts the served rows (`PouwLiveSlots_v1`, rule A)

Rule A: a credited row serves an admitted request, a slot the round's slot table assigns to a request in the
gate-signed request table under the window-open root. Round `s`'s request table is the requests admitted before round
`s`'s root (round 0's under round-0-only admission): each round's session decodes its own, so a call's view is its round's
tables, never round 0's by construction. Both tables are hiding leaves, so credit counts round `s`'s
served rows `m_s` through one proof a round, every round, of `PouwLiveSlots_v1{T}`
(`verity/core/protocols/pouw/circuit/live.py`) at the public `m_s`. `T` is the tables' leaf count, the pouw setting
`rows`; credit's tile grid is over the window's slot `M ≥ T` (`h1window.slot`, a power of two), and `m_s ≤ T`. Its
interface is note:compute-accounting/20261024T0212Z-draft-liveunit-interface.

* `liveOk`: **the unit on words** (`circuit.live.live_ok`). `m ≤ T`; the entries (prompt length ≠ 0) are a prefix, each
  at its own index; every slot below `m` names an entry (`1 ≤ slot ≤ N`, `N` the entries' count), and every slot from
  `m` on is `EMPTY_SLOT`, 0.
* `admitted`: **rule A's count**, the slots that name (as `q + 1`) an entry `q` (`circuit.live.admitted`).
* `liveOk_admitted`: an accepted round's `admitted` is its `m`. `liveOk_empty`: a slot from `m` on is empty and admits
  nothing.
* `LiveSlotsChecks`: **the named premise**, that the unit's circuit computes `liveOk` at its decoded port rows, in
  `Accounting.Assumptions.TS.LeafChecks`' form; `liveOk_of_accept` reads it.
* `dense_credit`, `dense_credit_of_sessions`: **dense credit is `Σ_c m_{s(c)}·n_c·K_c`**. The rows rule A credits, times
  each call's `n·K`, sum to the public rows': over accepted views, then over accepted sessions through the premise.
* `Dead`, `dead_iff_rows`, `population`: a tile is dead iff every one of its rows is at or past `m`. Its admit bits
  `[i < m]` are then all 0, so its receipt is `h1tile`'s canonical dead-tile receipt. The draw's population is the live
  tiles.
* `population_work`, `credit_le_work`, `zero_call`, `population_admits`, `dead_not_drawn`, `live_credit`: over that
  population. Its work, `Σ_c ⌈m_c/R⌉·(n_c/C)·R·C·K_c`, is at least the credit. A call with `m = 0` has no unit and earns
  0. Each unit holds a row that serves an admitted request, no dead tile is a unit, and every `m_c` is at most `T`.

What a decode reads from the port rows (the slot words, each header's index and prompt length, the public `m` and the
`ok` word) is the statement's port layout, read through `dec`. That the decoded `m` is the call's public row, and the
decoded slot and request rows are the round's registered `slots@<s>` and its request table (`requests@0` under
round-0-only admission), is the statement's registration
(flock-verify's complete check and the served path's roots), outside this file.
-/

namespace Pouw.SecurityProofs.LiveCredit

open Finset
open Flock.SoundAtBridge (RowsSession)
open Accounting.TS.Lanes (portRead)

/-- **A round's words as `PouwLiveSlots_v1{T}` reads them**: slot `i`'s word (`q + 1` for a row of request `q`, 0 for
`EMPTY_SLOT`), request leaf `q`'s header index and prompt length (`EMPTY_REQUEST`'s are 0), the public row `m` and the
`ok` word. -/
structure LiveView where
  slot : ℕ → ℕ
  index : ℕ → ℕ
  plen : ℕ → ℕ
  m : ℕ
  ok : Bool

/-- The request table's entries: the leaves below `T` whose prompt length is nonzero (every entry has a prompt token). -/
def entries (T : ℕ) (v : LiveView) : Finset ℕ := (range T).filter fun q => v.plen q ≠ 0

/-- **`PouwLiveSlots_v1{T}`'s `ok`, on words** (`circuit.live.live_ok`): `m ≤ T`; each entry's predecessor is an entry;
each entry's index is its leaf; each slot below `m` names an entry (`1 ≤ slot ≤ N`) and each slot from `m` on is 0. -/
def liveOk (T : ℕ) (v : LiveView) : Prop :=
  v.m ≤ T ∧ (∀ q, q + 1 < T → v.plen (q + 1) ≠ 0 → v.plen q ≠ 0) ∧ (∀ q < T, v.plen q ≠ 0 → v.index q = q) ∧
    ∀ i < T, (i < v.m → 1 ≤ v.slot i ∧ v.slot i ≤ (entries T v).card) ∧ (v.m ≤ i → v.slot i = 0)

/-- **Rule A at slot `i`**: the slot names (as `q + 1`) a leaf `q` of the request table that is an entry. -/
def Admits (T : ℕ) (v : LiveView) (i : ℕ) : Prop := ∃ q < T, v.slot i = q + 1 ∧ v.plen q ≠ 0

/-- **Rule A's count**: the slots below `T` that serve an admitted request (`circuit.live.admitted`). -/
noncomputable def admitted (T : ℕ) (v : LiveView) : ℕ := by
  classical exact ((range T).filter (Admits T v)).card

/-! ## One round -/

theorem entries_down {T : ℕ} {v : LiveView} (h : ∀ q, q + 1 < T → v.plen (q + 1) ≠ 0 → v.plen q ≠ 0) :
    ∀ a b, a ≤ b → b ∈ entries T v → a ∈ entries T v := by
  intro a b hab hb
  induction b, hab using Nat.le_induction with
  | base => exact hb
  | succ b hab ih =>
    simp only [entries, mem_filter, mem_range] at hb ⊢ ih
    exact ih ⟨by omega, h b hb.1 hb.2⟩

/-- A downward-closed set of naturals holds every number below its size. -/
theorem mem_of_lt_card {S : Finset ℕ} (hS : ∀ a b, a ≤ b → b ∈ S → a ∈ S) {q : ℕ} (hq : q < S.card) : q ∈ S := by
  by_contra hn
  have hsub : S ⊆ range q := fun s hs => mem_range.2 (by
    by_contra h'
    exact hn (hS q s (by omega) hs))
  have := card_le_card hsub
  rw [card_range] at this
  omega

/-- **In an accepted round, slot `i` serves an admitted request iff `i < m`.** -/
theorem liveOk_admits {T : ℕ} {v : LiveView} (h : liveOk T v) {i : ℕ} (hi : i < T) : Admits T v i ↔ i < v.m := by
  obtain ⟨-, hpre, -, hs⟩ := h
  constructor
  · rintro ⟨q, -, hq, -⟩
    by_contra hm
    have := (hs i hi).2 (by omega)
    omega
  · intro him
    obtain ⟨h1, hN⟩ := (hs i hi).1 him
    have hmem := mem_of_lt_card (entries_down hpre) (q := v.slot i - 1) (by omega)
    simp only [entries, mem_filter, mem_range] at hmem
    exact ⟨v.slot i - 1, hmem.1, by omega, hmem.2⟩

/-- **An accepted round's rule-A count is its public `m`.** -/
theorem liveOk_admitted {T : ℕ} {v : LiveView} (h : liveOk T v) : admitted T v = v.m := by
  classical
  have hm := h.1
  unfold admitted
  rw [filter_congr fun i hi => liveOk_admits h (mem_range.1 hi)]
  have : (range T).filter (· < v.m) = range v.m := by
    ext i
    simp only [mem_filter, mem_range]
    omega
  rw [this, card_range]

/-- In an accepted round a slot from `m` on is `EMPTY_SLOT` and serves no admitted request. -/
theorem liveOk_empty {T : ℕ} {v : LiveView} (h : liveOk T v) {i : ℕ} (hi : i < T) (hmi : v.m ≤ i) :
    v.slot i = 0 ∧ ¬ Admits T v i :=
  ⟨(h.2.2.2 i hi).2 hmi, fun ha => by have := (liveOk_admits h hi).1 ha; omega⟩

/-! ## The named premise -/

/-- `LiveSlotsChecks` (a fact of `circuit.live.PouwLiveSlots_v1{T}`'s circuit, backed by circuit-check and
`test_circuit_live`'s agreement of its gates with `live_ok`, not proved; owner @compute-accounting; no guarantee takes
it yet, so it has no claim id or lock entry). **The unit's circuit computes its Definition at its port rows**: at round
`s`'s session `sess`, whose circuit is `PouwLiveSlots_v1{T}`'s lowering, on every wire assignment `vals` the key
program's evaluation agrees with on every unit's wires, if the `ok` word `dec` reads from `vals`' port rows
(`portRead`) is `1`, then `dec`'s view of them meets `liveOk T`. It reads every port row, registered or not, so it is a
fact of the circuit alone, as `Accounting.Assumptions.TS.LeafChecks` is of a leaf's 33. -/
def LiveSlotsChecks (T : ℕ) (sess : RowsSession) (dec : RowsSession → (ℕ → ℕ → Option Bool) → LiveView) : Prop :=
  ∀ vals : Fin sess.Circ.N → Bool,
    (∀ u, ∀ g ∈ sess.P.io u, sess.Circ.eval vals g = vals g) →
    (dec sess (portRead sess vals)).ok = true →
    liveOk T (dec sess (portRead sess vals))

/-- **An accepted round's view meets `liveOk`**, at an assignment the circuit's evaluation agrees with whose `ok` reads
`1`. -/
theorem liveOk_of_accept {T : ℕ} {sess : RowsSession} {dec : RowsSession → (ℕ → ℕ → Option Bool) → LiveView}
    (hchk : LiveSlotsChecks T sess dec) (vals : Fin sess.Circ.N → Bool)
    (heval : ∀ u, ∀ g ∈ sess.P.io u, sess.Circ.eval vals g = vals g) (hok : (dec sess (portRead sess vals)).ok = true) :
    liveOk T (dec sess (portRead sess vals)) :=
  hchk vals heval hok

/-! ## The window -/

/-- **Dense credit is `Σ_c m_{s(c)}·n_c·K_c`**: where every call's round is accepted, the rows rule A credits, times each
call's `n·K`, sum to the public rows'. -/
theorem dense_credit {ι σ : Type} (calls : Finset ι) (step : ι → σ) (n K : ι → ℕ) (T : ℕ) (v : σ → LiveView)
    (hok : ∀ c ∈ calls, liveOk T (v (step c))) :
    ∑ c ∈ calls, admitted T (v (step c)) * n c * K c = ∑ c ∈ calls, (v (step c)).m * n c * K c :=
  sum_congr rfl fun c hc => by rw [liveOk_admitted (hok c hc)]

/-- **Dense credit over the rounds' sessions**: where each call's round `step c` has its `PouwLiveSlots_v1{T}` session
`sess (step c)` accepted (its evaluation agrees with `vals`, whose decoded `ok` is `1`) at the call's public row `m c`,
the rows rule A credits, times each call's `n·K`, sum to `Σ_c m_c·n_c·K_c`. -/
theorem dense_credit_of_sessions {ι σ : Type} (calls : Finset ι) (step : ι → σ) (n K m : ι → ℕ) (T : ℕ)
    (sess : σ → RowsSession) (dec : RowsSession → (ℕ → ℕ → Option Bool) → LiveView)
    (hchk : ∀ c ∈ calls, LiveSlotsChecks T (sess (step c)) dec) (vals : (s : σ) → Fin (sess s).Circ.N → Bool)
    (heval : ∀ c ∈ calls, ∀ u, ∀ g ∈ (sess (step c)).P.io u, (sess (step c)).Circ.eval (vals (step c)) g = vals (step c) g)
    (hok : ∀ c ∈ calls, (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))).ok = true)
    (hm : ∀ c ∈ calls, (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))).m = m c) :
    ∑ c ∈ calls, admitted T (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))) * n c * K c =
      ∑ c ∈ calls, m c * n c * K c :=
  sum_congr rfl fun c hc => by
    rw [liveOk_admitted (liveOk_of_accept (hchk c hc) _ (heval c hc) (hok c hc)), hm c hc]

/-! ## Tiles and the draw's population -/

/-- **Row block `ti` of an `R`-row tile is dead at `m`**: its first row, and so every row, is at or past `m`. -/
def Dead (R m ti : ℕ) : Prop := m ≤ R * ti

/-- A tile is dead iff every one of its rows is at or past `m`. -/
theorem dead_iff_rows {R m ti : ℕ} (hR : 0 < R) : Dead R m ti ↔ ∀ r < R, m ≤ R * ti + r := by
  unfold Dead
  constructor
  · intro h r _
    omega
  · intro h
    simpa using h 0 hR

theorem ceil_le {m R M : ℕ} (hR : 0 < R) (hRM : R ∣ M) (hm : m ≤ M) : (m + R - 1) / R ≤ M / R := by
  obtain ⟨k, rfl⟩ := hRM
  rw [Nat.mul_div_cancel_left k hR, Nat.div_le_iff_le_mul_add_pred hR]
  generalize R * k = x at *
  omega

theorem lt_ceil_iff {m R ti : ℕ} (hR : 0 < R) : ti < (m + R - 1) / R ↔ R * ti < m := by
  rw [← Nat.add_one_le_iff, Nat.le_div_iff_mul_le hR, Nat.add_mul, one_mul, Nat.mul_comm ti R]
  generalize R * ti = x
  omega

theorem le_ceil_mul (m R : ℕ) (hR : 0 < R) : m ≤ (m + R - 1) / R * R := by
  have h1 := Nat.div_add_mod (m + R - 1) R
  have h2 := Nat.mod_lt (m + R - 1) hR
  rw [Nat.mul_comm R] at h1
  generalize (m + R - 1) / R * R = x at *
  omega

/-- A call's live row blocks: those of its `M / R` with a row below `m`. -/
def liveBlocks (R M m : ℕ) : Finset ℕ := (range (M / R)).filter fun ti => R * ti < m

/-- A call has `⌈m/R⌉` live row blocks. -/
theorem liveBlocks_card {R M m : ℕ} (hR : 0 < R) (hRM : R ∣ M) (hm : m ≤ M) :
    (liveBlocks R M m).card = (m + R - 1) / R := by
  have : liveBlocks R M m = range ((m + R - 1) / R) := by
    ext ti
    simp only [liveBlocks, mem_filter, mem_range]
    have := ceil_le hR hRM hm
    rw [← lt_ceil_iff hR]
    omega
  rw [this, card_range]

/-- **The dense draw's population**: each call's live tiles `(c, ti, tj)`, row block `ti` live at the call's public
`m c` and column block `tj` one of its `n c / C`, in row-block-major order. -/
def population {ι : Type} (calls : Finset ι) (R C M : ℕ) (m n : ι → ℕ) : Finset (Σ _ : ι, ℕ × ℕ) :=
  calls.sigma fun c => liveBlocks R M (m c) ×ˢ range (n c / C)

/-- **The population's work** (each unit `R·C·K` of its call's `K`) is `Σ_c ⌈m_c/R⌉·(n_c/C)·R·C·K_c`. -/
theorem population_work {ι : Type} (calls : Finset ι) {R C M : ℕ} (m n K : ι → ℕ) (hR : 0 < R) (hRM : R ∣ M)
    (hm : ∀ c ∈ calls, m c ≤ M) :
    ∑ u ∈ population calls R C M m n, R * C * K u.1 = ∑ c ∈ calls, (m c + R - 1) / R * (n c / C) * (R * C * K c) := by
  unfold population
  rw [sum_sigma]
  refine sum_congr rfl fun c hc => ?_
  simp only [sum_const, card_product, liveBlocks_card hR hRM (hm c hc), card_range, smul_eq_mul]

/-- One call's credit, `m·n·K`, is at most its live tiles' work. -/
theorem call_credit_le {m n K R C : ℕ} (hR : 0 < R) (hC : 0 < C) (hCn : C ∣ n) :
    m * n * K ≤ (m + R - 1) / R * (n / C) * (R * C * K) := by
  obtain ⟨d, rfl⟩ := hCn
  rw [Nat.mul_div_cancel_left d hC]
  have h := Nat.mul_le_mul_right (C * d * K) (le_ceil_mul m R hR)
  calc m * (C * d) * K = m * (C * d * K) := by ring
    _ ≤ (m + R - 1) / R * R * (C * d * K) := h
    _ = (m + R - 1) / R * d * (R * C * K) := by ring

/-- **The credit is at most the population's work**: `Σ_c m_c·n_c·K_c ≤ Σ_{u ∈ population} R·C·K_u`. -/
theorem credit_le_work {ι : Type} (calls : Finset ι) {R C M : ℕ} (m n K : ι → ℕ) (hR : 0 < R) (hC : 0 < C)
    (hRM : R ∣ M) (hm : ∀ c ∈ calls, m c ≤ M) (hCn : ∀ c ∈ calls, C ∣ n c) :
    ∑ c ∈ calls, m c * n c * K c ≤ ∑ u ∈ population calls R C M m n, R * C * K u.1 := by
  rw [population_work calls m n K hR hRM hm]
  exact sum_le_sum fun c hc => call_credit_le hR hC (hCn c hc)

/-- **A call with `m = 0` earns 0**: it has no unit in the population, and its credit `m·n·K` is 0. -/
theorem zero_call {ι : Type} (calls : Finset ι) {R C M : ℕ} (m n : ι → ℕ) {c : ι} (h0 : m c = 0) (K : ℕ) :
    (∀ u ∈ population calls R C M m n, u.1 ≠ c) ∧ m c * n c * K = 0 := by
  refine ⟨fun u hu he => ?_, by rw [h0, zero_mul, zero_mul]⟩
  simp only [population, mem_sigma, mem_product, liveBlocks, mem_filter, mem_range] at hu
  rw [he, h0] at hu
  omega

/-- **No dead tile is in the population**: a tile whose rows are all at or past its call's `m` is never drawn. -/
theorem dead_not_drawn {ι : Type} (calls : Finset ι) {R C M : ℕ} (m n : ι → ℕ) (u : Σ _ : ι, ℕ × ℕ)
    (hd : Dead R (m u.1) u.2.1) : u ∉ population calls R C M m n := by
  intro hu
  simp only [population, mem_sigma, mem_product, liveBlocks, mem_filter, mem_range] at hu
  unfold Dead at hd
  omega

/-- **Each unit of the population serves an admitted request**: where its call's round is accepted at the tables' leaf
count `T` with public row `m c`, the tile's first row is a slot that names an entry of the request table. -/
theorem population_admits {ι σ : Type} (calls : Finset ι) (step : ι → σ) {R C M T : ℕ} (m n : ι → ℕ)
    (v : σ → LiveView) (hok : ∀ c ∈ calls, liveOk T (v (step c))) (hm : ∀ c ∈ calls, (v (step c)).m = m c)
    {u : Σ _ : ι, ℕ × ℕ} (hu : u ∈ population calls R C M m n) : Admits T (v (step u.1)) (R * u.2.1) := by
  simp only [population, mem_sigma, mem_product, liveBlocks, mem_filter, mem_range] at hu
  have hc := hu.1
  have hlt : R * u.2.1 < (v (step u.1)).m := by rw [hm _ hc]; exact hu.2.1.2
  exact (liveOk_admits (hok _ hc) (lt_of_lt_of_le hlt (hok _ hc).1)).2 hlt

/-- **Dense credit over the live population.** `T` is the round tables' leaf count (the pouw setting `rows`), and `M`,
at least `T`, the window's slot (`h1window.slot`), whose `M / R` row blocks are credit's tile grid. Where each call's round
has its `PouwLiveSlots_v1{T}` session accepted at the call's public row `m c`, and the draw's population `P` is the live
tiles (a dead tile, every row at or past `m c`, never drawn): every `m c` is at most `T`; the rows rule A credits, times
`n·K`, sum to `C = Σ_c m_c·n_c·K_c`; `C` is at most `P`'s work; every unit of `P` holds a row serving an admitted
request; no dead tile is in `P`; and a call with `m c = 0` has no unit in `P` and earns 0. -/
theorem live_credit {ι σ : Type} (calls : Finset ι) (step : ι → σ) (n K m : ι → ℕ) {R C M T : ℕ} (hR : 0 < R)
    (hC : 0 < C) (hRM : R ∣ M) (hTM : T ≤ M) (hCn : ∀ c ∈ calls, C ∣ n c)
    (sess : σ → RowsSession) (dec : RowsSession → (ℕ → ℕ → Option Bool) → LiveView)
    (hchk : ∀ c ∈ calls, LiveSlotsChecks T (sess (step c)) dec) (vals : (s : σ) → Fin (sess s).Circ.N → Bool)
    (heval : ∀ c ∈ calls, ∀ u, ∀ g ∈ (sess (step c)).P.io u, (sess (step c)).Circ.eval (vals (step c)) g = vals (step c) g)
    (hok : ∀ c ∈ calls, (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))).ok = true)
    (hm : ∀ c ∈ calls, (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))).m = m c)
    (P : Finset (Σ _ : ι, ℕ × ℕ)) (hP : P = population calls R C M m n) :
    (∀ c ∈ calls, m c ≤ T) ∧
      ∑ c ∈ calls, admitted T (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))) * n c * K c =
        ∑ c ∈ calls, m c * n c * K c ∧
      ∑ c ∈ calls, m c * n c * K c ≤ ∑ u ∈ P, R * C * K u.1 ∧
      (∀ u ∈ P, Admits T (dec (sess (step u.1)) (portRead (sess (step u.1)) (vals (step u.1)))) (R * u.2.1)) ∧
      (∀ u : Σ _ : ι, ℕ × ℕ, Dead R (m u.1) u.2.1 → u ∉ P) ∧
      ∀ c ∈ calls, m c = 0 → (∀ u ∈ P, u.1 ≠ c) ∧ m c * n c * K c = 0 := by
  let v : σ → LiveView := fun s => dec (sess s) (portRead (sess s) (vals s))
  have hv : ∀ c ∈ calls, liveOk T (v (step c)) := fun c hc => liveOk_of_accept (hchk c hc) _ (heval c hc) (hok c hc)
  have hmT : ∀ c ∈ calls, m c ≤ T := fun c hc => hm c hc ▸ (hv c hc).1
  have hmM : ∀ c ∈ calls, m c ≤ M := fun c hc => le_trans (hmT c hc) hTM
  subst hP
  exact ⟨hmT, dense_credit_of_sessions calls step n K m T sess dec hchk vals heval hok hm,
    credit_le_work calls m n K hR hC hRM hmM hCn,
    fun u hu => population_admits calls step m n v hv hm hu,
    fun u hd => dead_not_drawn calls m n u hd,
    fun c _ h0 => zero_call calls m n h0 (K c)⟩

end Pouw.SecurityProofs.LiveCredit
