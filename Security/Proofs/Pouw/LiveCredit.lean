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
`rows`; credit's tile grid is over the window's slot `M ≥ T` (`h1window.slot`, a power of two), and `m_s ≤ T`.

* `liveOk`: **the unit on words** (`circuit.live.live_ok`) at round `s`. `m ≤ T`; the entries (prompt length ≠ 0) are a
  prefix, each at its own index; every slot below `m` names an entry (`1 ≤ slot ≤ N`, `N` the entries' count), and every
  slot from `m` on is `EMPTY_SLOT`, 0; and no request is named by more slots than it `takes` in round `s`: its prompt
  length in the round that admitted it (its prefill), one in each of the `cap - 1` rounds after (a decode row), none in
  any other.
* `admitted`: **rule A's count**, the slots that name (as `q + 1`) an entry `q` (`circuit.live.admitted`).
* `liveOk_admitted`: an accepted round's `admitted` is its `m`. `liveOk_empty`: a slot from `m` on is empty and admits
  nothing.
* `liveOk_named`, `liveOk_decode`, `liveOk_one_request`, `liveOk_m_le`, `liveOk_decode_round`: **the per-request
  bound**. A request holds at most what it `takes`, a decoding one at most one slot; one request named in every slot below
  `m` bounds `m` by what it takes, so it can't fill the round with padded rows; `m ≤ Σ_{q ∈ entries} takes q`, and in a
  round where no entry is prefilling, `m` is at most the entries' count. `takesAt` is the bound at a given round, and
  `takes` is it at the decoded `s`; `live_credit` states it at each call's own round through `hs`.
* `LiveSlotsChecks`: **the named premise**, that the unit's circuit computes `liveOk` at its decoded port rows, in
  `Accounting.Assumptions.TS.LeafChecks`' form; `liveOk_of_accept` reads it.
* `dense_credit`, `dense_credit_of_sessions`: **dense credit is `Σ_c m_{s(c)}·n_c·K_c`**. The rows rule A credits, times
  each call's `n·K`, sum to the public rows': over accepted views, then over accepted sessions through the premise.
* `Dead`, `dead_iff_rows`, `population`: a tile is dead iff every one of its rows is at or past `m`. Its admit bits
  `[i < m]` are then all 0, so its receipt is `h1tile`'s canonical dead-tile receipt. The draw's population is the live
  tiles.
* `population_work`, `credit_le_work`, `zero_call`, `population_admits`, `dead_not_drawn`, `live_credit`: over that
  population. Its work, `Σ_c ⌈m_c/R⌉·(n_c/C)·R·C·K_c`, is at least the credit. A call with `m = 0` has no unit and earns
  0. Each unit holds a row that serves an admitted request, no dead tile is a unit, and every `m_c` is at most `T` and at
  most what its round's entries take.
* `LiveView2`, `capsBound`, `liveOk2`, `LiveSlotsChecks2`, `liveSlotsChecks_of_v2`, `live_schedule`:
  **`PouwLiveSlots_v2{T}`**, which credit reads. Its public port `cap` is each leaf's public `max_tokens`, and its `ok`
  adds that a leaf is an entry iff its cap is nonzero, admitted in round 0 with its header cap that cap. Its premise gives
  v1's, so `live_credit` holds of it. With the decoded caps the committed `max_tokens` (`hcap`, as `hm` and `hs`),
  a round `s ≥ 1` serves at most `#{q < T : s < max_tokens q}` rows, so no hidden request takes another's decode rows.

What a decode reads from the port rows (the slot words, each header's index, round, cap and prompt length, the public
`m` and `s`, and the `ok` word) is the statement's port layout, read through `dec`. That the decoded `m` is the call's
public row and the decoded `s` its step's round are `live_credit`'s hypotheses `hm` and `hs`: credit opens the unit's `m`
at the known `m@<s>` row of the manifest's `m_s` and its `s` at the known `round@<s>` row of the round's index, so a
statement decoded at another round proves nothing about this one. That the decoded slot and request rows are the round's
registered `slots@<s>` and its request table (`requests@0` under round-0-only admission) is the statement's registration
(flock-verify's complete check and the served path's roots), outside this file.
-/

namespace Pouw.SecurityProofs.LiveCredit

open Finset
open Flock.SoundAtBridge (RowsSession)
open Accounting.TS.Lanes (portRead)

/-- **A round's words as `PouwLiveSlots_v1{T}` reads them**: slot `i`'s word (`q + 1` for a row of request `q`, 0 for
`EMPTY_SLOT`), request leaf `q`'s header index, round (the round that admitted it), cap and prompt length
(`EMPTY_REQUEST`'s are 0), the public row `m`, the public round `s` and the `ok` word. -/
structure LiveView where
  slot : ℕ → ℕ
  index : ℕ → ℕ
  rnd : ℕ → ℕ
  cap : ℕ → ℕ
  plen : ℕ → ℕ
  m : ℕ
  s : ℕ
  ok : Bool

/-- The request table's entries: the leaves below `T` whose prompt length is nonzero (every entry has a prompt token). -/
def entries (T : ℕ) (v : LiveView) : Finset ℕ := (range T).filter fun q => v.plen q ≠ 0

/-- **How many of round `r`'s slots request `q` can take**: its prompt length when `r` is the round that admitted it
(its prefill), one in each of the `cap - 1` rounds after (a decode row each), none in any other round. -/
def takesAt (v : LiveView) (r q : ℕ) : ℕ :=
  if v.rnd q = r then v.plen q else if v.rnd q < r ∧ r < v.rnd q + v.cap q then 1 else 0

/-- **How many of round `s`'s slots request `q` can take** (`circuit.live.takes`): its prompt length in the round that
admitted it (its prefill, the whole prompt in one step), one in each of the `cap - 1` rounds after (a decode row each), none
in any other round. -/
def takes (v : LiveView) (q : ℕ) : ℕ := takesAt v v.s q

/-- The slots below `T` that name request `q` (as `q + 1`). -/
def named (T : ℕ) (v : LiveView) (q : ℕ) : Finset ℕ := (range T).filter fun i => v.slot i = q + 1

/-- **`PouwLiveSlots_v1{T}`'s `ok`, on words** (`circuit.live.live_ok`): `m ≤ T`; each entry's predecessor is an entry;
each entry's index is its leaf; each slot below `m` names an entry (`1 ≤ slot ≤ N`) and each slot from `m` on is 0; and
each request is named by at most the slots it `takes` in round `s`. -/
def liveOk (T : ℕ) (v : LiveView) : Prop :=
  v.m ≤ T ∧ (∀ q, q + 1 < T → v.plen (q + 1) ≠ 0 → v.plen q ≠ 0) ∧ (∀ q < T, v.plen q ≠ 0 → v.index q = q) ∧
    (∀ i < T, (i < v.m → 1 ≤ v.slot i ∧ v.slot i ≤ (entries T v).card) ∧ (v.m ≤ i → v.slot i = 0)) ∧
    ∀ q < T, (named T v q).card ≤ takes v q

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
  obtain ⟨-, hpre, -, hs, -⟩ := h
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
  ⟨(h.2.2.2.1 i hi).2 hmi, fun ha => by have := (liveOk_admits h hi).1 ha; omega⟩

/-- **In an accepted round no request holds more slots than it `takes`**: its prompt length in its own round, at most one
in a later round within its cap, none in any other. -/
theorem liveOk_named {T : ℕ} {v : LiveView} (h : liveOk T v) {q : ℕ} (hq : q < T) : (named T v q).card ≤ takes v q :=
  h.2.2.2.2 q hq

/-- **A decoding request holds at most one slot**: in an accepted round, a request admitted in another round than `s`. -/
theorem liveOk_decode {T : ℕ} {v : LiveView} (h : liveOk T v) {q : ℕ} (hq : q < T) (hd : v.rnd q ≠ v.s) :
    (named T v q).card ≤ 1 := by
  have := liveOk_named h hq
  unfold takes takesAt at this
  split_ifs at this <;> omega

/-- **One request can't fill a round**: in an accepted round whose every slot below `m` names request `q`, `m` is at most
what `q` takes (at most 1 when it is decoding), so padded rows named for it are refused. -/
theorem liveOk_one_request {T : ℕ} {v : LiveView} (h : liveOk T v) {q : ℕ} (hq : q < T)
    (hall : ∀ i < v.m, v.slot i = q + 1) : v.m ≤ takes v q := by
  have hsub : range v.m ⊆ named T v q := fun i hi => by
    rw [mem_range] at hi
    simp only [named, mem_filter, mem_range]
    exact ⟨lt_of_lt_of_le hi h.1, hall i hi⟩
  simpa using le_trans (card_le_card hsub) (liveOk_named h hq)

/-- **An accepted round's `m` is at most what its entries take**: `m ≤ Σ_{q ∈ entries} takes q`. -/
theorem liveOk_m_le {T : ℕ} {v : LiveView} (h : liveOk T v) : v.m ≤ ∑ q ∈ entries T v, takes v q := by
  classical
  obtain ⟨hmT, hpre, -, hs, hb⟩ := h
  have hmaps : ((range v.m : Finset ℕ) : Set ℕ).MapsTo (fun i => v.slot i - 1) (entries T v) := by
    intro i hi
    rw [mem_coe, mem_range] at hi
    obtain ⟨h1, hN⟩ := (hs i (by omega)).1 hi
    exact mem_of_lt_card (entries_down hpre) (q := v.slot i - 1) (by omega)
  calc v.m = (range v.m).card := (card_range _).symm
    _ = ∑ q ∈ entries T v, ((range v.m).filter fun i => v.slot i - 1 = q).card :=
        card_eq_sum_card_fiberwise hmaps
    _ ≤ ∑ q ∈ entries T v, takes v q := by
        refine sum_le_sum fun q hq => le_trans (card_le_card fun i hi => ?_) (hb q ?_)
        · simp only [mem_filter, mem_range] at hi
          obtain ⟨h1, -⟩ := (hs i (by omega)).1 hi.1
          simp only [named, mem_filter, mem_range]
          exact ⟨by omega, by omega⟩
        · simp only [entries, mem_filter, mem_range] at hq
          exact hq.1

/-- **A decode round serves at most one row a request**: in an accepted round where no entry was admitted in round `s`,
`m` is at most the entries' count. -/
theorem liveOk_decode_round {T : ℕ} {v : LiveView} (h : liveOk T v) (hd : ∀ q ∈ entries T v, v.rnd q ≠ v.s) :
    v.m ≤ (entries T v).card := by
  refine le_trans (liveOk_m_le h) (le_trans (sum_le_sum fun q hq => (?_ : takes v q ≤ 1)) (by simp))
  have := hd q hq
  unfold takes takesAt
  split_ifs <;> omega

/-! ## The named premise -/

/-- `LiveSlotsChecks` (a fact of `circuit.live.PouwLiveSlots_v1{T}`'s circuit, backed by circuit-check and
`test_circuit_live`'s agreement of its gates with `live_ok`, not proved; no guarantee takes
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
has its `PouwLiveSlots_v1{T}` session accepted at the call's public row `m c` and at its step's round `round (step c)`
(`hs`: the decoded `s`, so the per-request bound is the call's own round's, not one the prover picks), and the draw's
population `P` is the live tiles (a dead tile, every row at or past `m c`, never drawn): every `m c` is at most `T`; in
the call's round each request holds at most what it `takesAt` that round (its prompt length in the round that admitted
it, one row in a later round within its cap, none otherwise), so a request decoding there holds at most one slot, and
`m c` is at most what the round's entries take, so no request's padded rows are credited; the rows rule A credits, times
`n·K`, sum to `C = Σ_c m_c·n_c·K_c`; `C` is at most `P`'s work; every unit of `P` holds a row serving an admitted
request; no dead tile is in `P`; and a call with `m c = 0` has no unit in `P` and earns 0. -/
theorem live_credit {ι σ : Type} (calls : Finset ι) (step : ι → σ) (n K m : ι → ℕ) {R C M T : ℕ} (hR : 0 < R)
    (hC : 0 < C) (hRM : R ∣ M) (hTM : T ≤ M) (hCn : ∀ c ∈ calls, C ∣ n c)
    (sess : σ → RowsSession) (dec : RowsSession → (ℕ → ℕ → Option Bool) → LiveView)
    (hchk : ∀ c ∈ calls, LiveSlotsChecks T (sess (step c)) dec) (vals : (s : σ) → Fin (sess s).Circ.N → Bool)
    (heval : ∀ c ∈ calls, ∀ u, ∀ g ∈ (sess (step c)).P.io u, (sess (step c)).Circ.eval (vals (step c)) g = vals (step c) g)
    (hok : ∀ c ∈ calls, (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))).ok = true)
    (hm : ∀ c ∈ calls, (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))).m = m c)
    (round : σ → ℕ) (hs : ∀ c ∈ calls, (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))).s = round (step c))
    (P : Finset (Σ _ : ι, ℕ × ℕ)) (hP : P = population calls R C M m n) :
    (∀ c ∈ calls, m c ≤ T) ∧
      (∀ c ∈ calls, ∀ q < T, (named T (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))) q).card ≤
        takesAt (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))) (round (step c)) q) ∧
      (∀ c ∈ calls, ∀ q < T, (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))).rnd q ≠ round (step c) →
        (named T (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))) q).card ≤ 1) ∧
      (∀ c ∈ calls, m c ≤ ∑ q ∈ entries T (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))),
        takesAt (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))) (round (step c)) q) ∧
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
  exact ⟨hmT, fun c hc q hq => by rw [← hs c hc]; exact liveOk_named (hv c hc) hq,
    fun c hc q hq hd => liveOk_decode (hv c hc) hq fun he => hd (he.trans (hs c hc)),
    fun c hc => by rw [← hm c hc, ← hs c hc]; exact liveOk_m_le (hv c hc),
    dense_credit_of_sessions calls step n K m T sess dec hchk vals heval hok hm,
    credit_le_work calls m n K hR hC hRM hmM hCn,
    fun u hu => population_admits calls step m n v hv hm hu,
    fun u hd => dead_not_drawn calls m n u hd,
    fun c _ h0 => zero_call calls m n h0 (K c)⟩

/-! ## `PouwLiveSlots_v2`: each request's cap is its public `max_tokens` -/

/-- **A round's words as `PouwLiveSlots_v2{T}` reads them**: v1's, and the public port `cap`, leaf `q`'s public
`max_tokens` (0 past the period's requests). -/
structure LiveView2 extends LiveView where
  pcap : ℕ → ℕ

/-- **The caps binding** (`circuit.live.caps_bound`): leaf `q` is an entry iff its public cap is nonzero, and an entry was
admitted in round 0 with its header's cap its public cap. -/
def capsBound (T : ℕ) (v : LiveView2) : Prop :=
  ∀ q < T, (v.plen q ≠ 0 ↔ v.pcap q ≠ 0) ∧ (v.plen q ≠ 0 → v.rnd q = 0 ∧ v.cap q = v.pcap q)

/-- **`PouwLiveSlots_v2{T}`'s `ok`, on words** (`circuit.live.live_ok_v2`): v1's `ok` and the caps binding. -/
def liveOk2 (T : ℕ) (v : LiveView2) : Prop := liveOk T v.toLiveView ∧ capsBound T v

/-- In an accepted v2 round at `s ≥ 1`, an entry takes one slot while `s` is below its public cap and none after. -/
theorem liveOk2_takes {T : ℕ} {v : LiveView2} (h : liveOk2 T v) (hs : 1 ≤ v.s) {q : ℕ}
    (hq : q ∈ entries T v.toLiveView) : takes v.toLiveView q = if v.s < v.pcap q then 1 else 0 := by
  simp only [entries, mem_filter, mem_range] at hq
  obtain ⟨h0, hc⟩ := (h.2 q hq.1).2 hq.2
  unfold takes takesAt
  rw [h0, hc]
  split_ifs <;> omega

/-- **An accepted v2 decode round serves at most the public schedule's rows**: at `s ≥ 1`, `m` is at most the count of
leaves whose public cap exceeds `s`, whatever requests the hidden table holds and whichever slots name them. -/
theorem liveOk2_m_le {T : ℕ} {v : LiveView2} (h : liveOk2 T v) (hs : 1 ≤ v.s) :
    v.m ≤ ((range T).filter fun q => v.s < v.pcap q).card := by
  calc v.m ≤ ∑ q ∈ entries T v.toLiveView, takes v.toLiveView q := liveOk_m_le h.1
    _ = ∑ q ∈ entries T v.toLiveView, if v.s < v.pcap q then 1 else 0 :=
        sum_congr rfl fun q hq => liveOk2_takes h hs hq
    _ = ((entries T v.toLiveView).filter fun q => v.s < v.pcap q).card := (card_filter _ _).symm
    _ ≤ ((range T).filter fun q => v.s < v.pcap q).card := card_le_card fun q hq => by
        simp only [entries, mem_filter, mem_range] at hq ⊢
        exact ⟨hq.1.1, hq.2⟩

/-- `LiveSlotsChecks2` (a fact of `circuit.live.PouwLiveSlots_v2{T}`'s circuit, backed as `LiveSlotsChecks` is, by
circuit-check and `test_circuit_live`'s agreement of its gates with `live_ok_v2`, not proved; no guarantee takes it yet).
**The unit's circuit computes its Definition at its port rows**, in `LiveSlotsChecks`' form. -/
def LiveSlotsChecks2 (T : ℕ) (sess : RowsSession) (dec : RowsSession → (ℕ → ℕ → Option Bool) → LiveView2) : Prop :=
  ∀ vals : Fin sess.Circ.N → Bool,
    (∀ u, ∀ g ∈ sess.P.io u, sess.Circ.eval vals g = vals g) →
    (dec sess (portRead sess vals)).ok = true →
    liveOk2 T (dec sess (portRead sess vals))

/-- v2's premise gives v1's at the decoder's v1 view, so `live_credit` holds of every v2 window. -/
theorem liveSlotsChecks_of_v2 {T : ℕ} {sess : RowsSession} {dec : RowsSession → (ℕ → ℕ → Option Bool) → LiveView2}
    (h : LiveSlotsChecks2 T sess dec) : LiveSlotsChecks T sess fun s r => (dec s r).toLiveView :=
  fun vals heval hok => (h vals heval hok).1

/-- **The public schedule bounds every v2 window** (the fixed length, `credit._length`). Where each call's round has its
`PouwLiveSlots_v2{T}` session accepted at the call's public row `m c` (`hm`) and its step's round (`hs`), with the decoded
caps the period's committed `max_tokens` (`hcap`: credit opens each `cap_<i>` at the known `<prefix>caps` row of the
committed length): in a round `≥ 1`, `m c` is at most `#{q < T : round < max_tokens q}`, so no hidden request takes a
decode row past its own public `max_tokens`; and every entry of the round's request table is a leaf with a nonzero
`max_tokens`, admitted in round 0, whose header cap is that `max_tokens`. -/
theorem live_schedule {ι σ : Type} (calls : Finset ι) (step : ι → σ) (m : ι → ℕ) {T : ℕ}
    (sess : σ → RowsSession) (dec : RowsSession → (ℕ → ℕ → Option Bool) → LiveView2)
    (hchk : ∀ c ∈ calls, LiveSlotsChecks2 T (sess (step c)) dec) (vals : (s : σ) → Fin (sess s).Circ.N → Bool)
    (heval : ∀ c ∈ calls, ∀ u, ∀ g ∈ (sess (step c)).P.io u, (sess (step c)).Circ.eval (vals (step c)) g = vals (step c) g)
    (hok : ∀ c ∈ calls, (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))).ok = true)
    (hm : ∀ c ∈ calls, (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))).m = m c)
    (round : σ → ℕ) (hs : ∀ c ∈ calls, (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))).s = round (step c))
    (maxTokens : ℕ → ℕ)
    (hcap : ∀ c ∈ calls, ∀ q < T, (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))).pcap q = maxTokens q) :
    (∀ c ∈ calls, 1 ≤ round (step c) → m c ≤ ((range T).filter fun q => round (step c) < maxTokens q).card) ∧
      ∀ c ∈ calls, ∀ q < T, (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))).plen q ≠ 0 →
        maxTokens q ≠ 0 ∧ (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))).rnd q = 0 ∧
          (dec (sess (step c)) (portRead (sess (step c)) (vals (step c)))).cap q = maxTokens q := by
  refine ⟨fun c hc h1 => ?_, fun c hc q hq hp => ?_⟩
  · have hv := hchk c hc _ (heval c hc) (hok c hc)
    have hle := liveOk2_m_le hv (by rw [hs c hc]; exact h1)
    rw [hm c hc, hs c hc] at hle
    refine le_trans hle (le_of_eq (congrArg card (filter_congr fun q hq => ?_)))
    rw [hcap c hc q (mem_range.1 hq)]
  · have hb := (hchk c hc _ (heval c hc) (hok c hc)).2 q hq
    obtain ⟨h0, hcp⟩ := hb.2 hp
    refine ⟨?_, h0, hcp.trans (hcap c hc q hq)⟩
    rw [← hcap c hc q hq]
    exact hb.1.1 hp

end Pouw.SecurityProofs.LiveCredit
