module

public import Proofs.Pouw.TS.Session.Lanes.Layer
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
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
import Mathlib.Tactic.Polynomial.Basic
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init

/-! Dense credit counts the served rows (`PouwLiveSlots_v1`, rule A); that the decoded slot and request rows are
the round's registered ones is the statement's registration, outside this file. -/

@[expose] public section

namespace Pouw.SecurityProofs.LiveCredit

open Finset
open Flock.SoundAtBridge (RowsSession)
open ResourceAccounting.TS.Lanes (portRead)

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

def entries (T : ℕ) (v : LiveView) : Finset ℕ := (range T).filter fun q => v.plen q ≠ 0

def takesAt (v : LiveView) (r q : ℕ) : ℕ :=
  if v.rnd q = r then v.plen q else if v.rnd q < r ∧ r < v.rnd q + v.cap q then 1 else 0

def takes (v : LiveView) (q : ℕ) : ℕ := takesAt v v.s q

def named (T : ℕ) (v : LiveView) (q : ℕ) : Finset ℕ := (range T).filter fun i => v.slot i = q + 1

def liveOk (T : ℕ) (v : LiveView) : Prop :=
  v.m ≤ T ∧ (∀ q, q + 1 < T → v.plen (q + 1) ≠ 0 → v.plen q ≠ 0) ∧ (∀ q < T, v.plen q ≠ 0 → v.index q = q) ∧
    (∀ i < T, (i < v.m → 1 ≤ v.slot i ∧ v.slot i ≤ (entries T v).card) ∧ (v.m ≤ i → v.slot i = 0)) ∧
    ∀ q < T, (named T v q).card ≤ takes v q

def Admits (T : ℕ) (v : LiveView) (i : ℕ) : Prop := ∃ q < T, v.slot i = q + 1 ∧ v.plen q ≠ 0

noncomputable def admitted (T : ℕ) (v : LiveView) : ℕ := by
  classical exact ((range T).filter (Admits T v)).card

theorem entries_down {T : ℕ} {v : LiveView} (h : ∀ q, q + 1 < T → v.plen (q + 1) ≠ 0 → v.plen q ≠ 0) :
    ∀ a b, a ≤ b → b ∈ entries T v → a ∈ entries T v := by
  intro a b hab hb
  induction b, hab using Nat.le_induction with
  | base => exact hb
  | succ b hab ih =>
    simp only [entries, mem_filter, mem_range] at hb ⊢ ih
    exact ih ⟨by omega, h b hb.1 hb.2⟩

theorem mem_of_lt_card {S : Finset ℕ} (hS : ∀ a b, a ≤ b → b ∈ S → a ∈ S) {q : ℕ} (hq : q < S.card) : q ∈ S := by
  by_contra hn
  have hsub : S ⊆ range q := fun s hs => mem_range.2 (by
    by_contra h'
    exact hn (hS q s (by omega) hs))
  have := card_le_card hsub
  rw [card_range] at this
  omega

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

theorem liveOk_empty {T : ℕ} {v : LiveView} (h : liveOk T v) {i : ℕ} (hi : i < T) (hmi : v.m ≤ i) :
    v.slot i = 0 ∧ ¬ Admits T v i :=
  ⟨(h.2.2.2.1 i hi).2 hmi, fun ha => by have := (liveOk_admits h hi).1 ha; omega⟩

theorem liveOk_named {T : ℕ} {v : LiveView} (h : liveOk T v) {q : ℕ} (hq : q < T) : (named T v q).card ≤ takes v q :=
  h.2.2.2.2 q hq

theorem liveOk_decode {T : ℕ} {v : LiveView} (h : liveOk T v) {q : ℕ} (hq : q < T) (hd : v.rnd q ≠ v.s) :
    (named T v q).card ≤ 1 := by
  have := liveOk_named h hq
  unfold takes takesAt at this
  split_ifs at this <;> omega

theorem liveOk_one_request {T : ℕ} {v : LiveView} (h : liveOk T v) {q : ℕ} (hq : q < T)
    (hall : ∀ i < v.m, v.slot i = q + 1) : v.m ≤ takes v q := by
  have hsub : range v.m ⊆ named T v q := fun i hi => by
    rw [mem_range] at hi
    simp only [named, mem_filter, mem_range]
    exact ⟨lt_of_lt_of_le hi h.1, hall i hi⟩
  simpa using le_trans (card_le_card hsub) (liveOk_named h hq)

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

theorem liveOk_decode_round {T : ℕ} {v : LiveView} (h : liveOk T v) (hd : ∀ q ∈ entries T v, v.rnd q ≠ v.s) :
    v.m ≤ (entries T v).card := by
  refine le_trans (liveOk_m_le h) (le_trans (sum_le_sum fun q hq => (?_ : takes v q ≤ 1)) (by simp))
  have := hd q hq
  unfold takes takesAt
  split_ifs <;> omega

/-- A fact of `PouwLiveSlots_v1{T}`'s circuit, backed by circuit-check and `test_circuit_live`'s agreement of its
gates with `live_ok`, not proved; no guarantee takes it, so it has no claim id or lock entry. -/
def LiveSlotsChecks (T : ℕ) (sess : RowsSession) (dec : RowsSession → (ℕ → ℕ → Option Bool) → LiveView) : Prop :=
  ∀ vals : Fin sess.Circ.N → Bool,
    (∀ u, ∀ g ∈ sess.P.io u, sess.Circ.eval vals g = vals g) →
    (dec sess (portRead sess vals)).ok = true →
    liveOk T (dec sess (portRead sess vals))

theorem liveOk_of_accept {T : ℕ} {sess : RowsSession} {dec : RowsSession → (ℕ → ℕ → Option Bool) → LiveView}
    (hchk : LiveSlotsChecks T sess dec) (vals : Fin sess.Circ.N → Bool)
    (heval : ∀ u, ∀ g ∈ sess.P.io u, sess.Circ.eval vals g = vals g) (hok : (dec sess (portRead sess vals)).ok = true) :
    liveOk T (dec sess (portRead sess vals)) :=
  hchk vals heval hok

theorem dense_credit {ι σ : Type} (calls : Finset ι) (step : ι → σ) (n K : ι → ℕ) (T : ℕ) (v : σ → LiveView)
    (hok : ∀ c ∈ calls, liveOk T (v (step c))) :
    ∑ c ∈ calls, admitted T (v (step c)) * n c * K c = ∑ c ∈ calls, (v (step c)).m * n c * K c :=
  sum_congr rfl fun c hc => by rw [liveOk_admitted (hok c hc)]

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

def Dead (R m ti : ℕ) : Prop := m ≤ R * ti

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

def liveBlocks (R M m : ℕ) : Finset ℕ := (range (M / R)).filter fun ti => R * ti < m

theorem liveBlocks_card {R M m : ℕ} (hR : 0 < R) (hRM : R ∣ M) (hm : m ≤ M) :
    (liveBlocks R M m).card = (m + R - 1) / R := by
  have : liveBlocks R M m = range ((m + R - 1) / R) := by
    ext ti
    simp only [liveBlocks, mem_filter, mem_range]
    have := ceil_le hR hRM hm
    rw [← lt_ceil_iff hR]
    omega
  rw [this, card_range]

def population {ι : Type} (calls : Finset ι) (R C M : ℕ) (m n : ι → ℕ) : Finset (Σ _ : ι, ℕ × ℕ) :=
  calls.sigma fun c => liveBlocks R M (m c) ×ˢ range (n c / C)

theorem population_work {ι : Type} (calls : Finset ι) {R C M : ℕ} (m n K : ι → ℕ) (hR : 0 < R) (hRM : R ∣ M)
    (hm : ∀ c ∈ calls, m c ≤ M) :
    ∑ u ∈ population calls R C M m n, R * C * K u.1 = ∑ c ∈ calls, (m c + R - 1) / R * (n c / C) * (R * C * K c) := by
  unfold population
  rw [sum_sigma]
  refine sum_congr rfl fun c hc => ?_
  simp only [sum_const, card_product, liveBlocks_card hR hRM (hm c hc), card_range, smul_eq_mul]

theorem call_credit_le {m n K R C : ℕ} (hR : 0 < R) (hC : 0 < C) (hCn : C ∣ n) :
    m * n * K ≤ (m + R - 1) / R * (n / C) * (R * C * K) := by
  obtain ⟨d, rfl⟩ := hCn
  rw [Nat.mul_div_cancel_left d hC]
  have h := Nat.mul_le_mul_right (C * d * K) (le_ceil_mul m R hR)
  calc m * (C * d) * K = m * (C * d * K) := by ring
    _ ≤ (m + R - 1) / R * R * (C * d * K) := h
    _ = (m + R - 1) / R * d * (R * C * K) := by ring

theorem credit_le_work {ι : Type} (calls : Finset ι) {R C M : ℕ} (m n K : ι → ℕ) (hR : 0 < R) (hC : 0 < C)
    (hRM : R ∣ M) (hm : ∀ c ∈ calls, m c ≤ M) (hCn : ∀ c ∈ calls, C ∣ n c) :
    ∑ c ∈ calls, m c * n c * K c ≤ ∑ u ∈ population calls R C M m n, R * C * K u.1 := by
  rw [population_work calls m n K hR hRM hm]
  exact sum_le_sum fun c hc => call_credit_le hR hC (hCn c hc)

theorem zero_call {ι : Type} (calls : Finset ι) {R C M : ℕ} (m n : ι → ℕ) {c : ι} (h0 : m c = 0) (K : ℕ) :
    (∀ u ∈ population calls R C M m n, u.1 ≠ c) ∧ m c * n c * K = 0 := by
  refine ⟨fun u hu he => ?_, by rw [h0, zero_mul, zero_mul]⟩
  simp only [population, mem_sigma, mem_product, liveBlocks, mem_filter, mem_range] at hu
  rw [he, h0] at hu
  omega

theorem dead_not_drawn {ι : Type} (calls : Finset ι) {R C M : ℕ} (m n : ι → ℕ) (u : Σ _ : ι, ℕ × ℕ)
    (hd : Dead R (m u.1) u.2.1) : u ∉ population calls R C M m n := by
  intro hu
  simp only [population, mem_sigma, mem_product, liveBlocks, mem_filter, mem_range] at hu
  unfold Dead at hd
  omega

theorem population_admits {ι σ : Type} (calls : Finset ι) (step : ι → σ) {R C M T : ℕ} (m n : ι → ℕ)
    (v : σ → LiveView) (hok : ∀ c ∈ calls, liveOk T (v (step c))) (hm : ∀ c ∈ calls, (v (step c)).m = m c)
    {u : Σ _ : ι, ℕ × ℕ} (hu : u ∈ population calls R C M m n) : Admits T (v (step u.1)) (R * u.2.1) := by
  simp only [population, mem_sigma, mem_product, liveBlocks, mem_filter, mem_range] at hu
  have hc := hu.1
  have hlt : R * u.2.1 < (v (step u.1)).m := by rw [hm _ hc]; exact hu.2.1.2
  exact (liveOk_admits (hok _ hc) (lt_of_lt_of_le hlt (hok _ hc).1)).2 hlt

/-- `hs` decodes `s` at the call's own round, so the per-request bound is that round's and not one the prover
picks. -/
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

structure LiveView2 extends LiveView where
  pcap : ℕ → ℕ

def capsBound (T : ℕ) (v : LiveView2) : Prop :=
  ∀ q < T, (v.plen q ≠ 0 ↔ v.pcap q ≠ 0) ∧ (v.plen q ≠ 0 → v.rnd q = 0 ∧ v.cap q = v.pcap q)

def liveOk2 (T : ℕ) (v : LiveView2) : Prop := liveOk T v.toLiveView ∧ capsBound T v

theorem liveOk2_takes {T : ℕ} {v : LiveView2} (h : liveOk2 T v) (hs : 1 ≤ v.s) {q : ℕ}
    (hq : q ∈ entries T v.toLiveView) : takes v.toLiveView q = if v.s < v.pcap q then 1 else 0 := by
  simp only [entries, mem_filter, mem_range] at hq
  obtain ⟨h0, hc⟩ := (h.2 q hq.1).2 hq.2
  unfold takes takesAt
  rw [h0, hc]
  split_ifs <;> omega

theorem liveOk2_m_le {T : ℕ} {v : LiveView2} (h : liveOk2 T v) (hs : 1 ≤ v.s) :
    v.m ≤ ((range T).filter fun q => v.s < v.pcap q).card := by
  calc v.m ≤ ∑ q ∈ entries T v.toLiveView, takes v.toLiveView q := liveOk_m_le h.1
    _ = ∑ q ∈ entries T v.toLiveView, if v.s < v.pcap q then 1 else 0 :=
        sum_congr rfl fun q hq => liveOk2_takes h hs hq
    _ = ((entries T v.toLiveView).filter fun q => v.s < v.pcap q).card := (card_filter _ _).symm
    _ ≤ ((range T).filter fun q => v.s < v.pcap q).card := card_le_card fun q hq => by
        simp only [entries, mem_filter, mem_range] at hq ⊢
        exact ⟨hq.1.1, hq.2⟩

/-- A fact of `PouwLiveSlots_v2{T}`'s circuit, backed as `LiveSlotsChecks` is, not proved; no guarantee takes it. -/
def LiveSlotsChecks2 (T : ℕ) (sess : RowsSession) (dec : RowsSession → (ℕ → ℕ → Option Bool) → LiveView2) : Prop :=
  ∀ vals : Fin sess.Circ.N → Bool,
    (∀ u, ∀ g ∈ sess.P.io u, sess.Circ.eval vals g = vals g) →
    (dec sess (portRead sess vals)).ok = true →
    liveOk2 T (dec sess (portRead sess vals))

theorem liveSlotsChecks_of_v2 {T : ℕ} {sess : RowsSession} {dec : RowsSession → (ℕ → ℕ → Option Bool) → LiveView2}
    (h : LiveSlotsChecks2 T sess dec) : LiveSlotsChecks T sess fun s r => (dec s r).toLiveView :=
  fun vals heval hok => (h vals heval hok).1

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
