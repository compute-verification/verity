module

public import Proofs.Flock.Claims.Complete.Json.Goals
public import Proofs.Flock.Claims.Complete.Replay.Zerocheck
public import Proofs.Flock.Claims.Complete.Replay.ZerocheckAlgebra
public import Proofs.Flock.Claims.Complete.Shape
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
import Proofs.Flock.Claims.Complete.Discharge

/-! The skeleton's `Zerocheck` at every honest run, given `TraceR1CS` and what the session walk establishes for each rep
(`ZcAt`). -/

@[expose] public section

namespace FlockComplete.Replay

open Flock FlockDeploy Transcript

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

def _root_.FlockComplete.Run.trace (o : Run H rel) (i : ℕ) : Trace := (tracesOf o.sts o.w)[i]!

def _root_.FlockComplete.Run.zcCoins (o : Run H rel) (i r : ℕ) : Zc.Challenges :=
  Zc.coinsAt o.resp (repStart o.spec o.sts (2 * i + r))

theorem _root_.FlockComplete.Run.tab_m (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) :
    o.tabs[i]!.m = o.sts[i]!.m ∧ 22 ≤ o.tabs[i]!.m ∧ o.tabs[i]!.m ≤ 35 := by
  obtain ⟨st, hsts, -, h22, h35, -⟩ := o.table
  unfold Run.tabs tabsOf at hi ⊢
  rw [hsts] at hi ⊢
  simp only [Array.map_singleton, List.size_toArray, List.length_singleton, Nat.lt_one_iff] at hi
  subst hi
  simp [Setup.ofCircuit, h22, h35]

theorem _root_.FlockComplete.Run.outer_ne_one (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) {r : ℕ} (hr : r < 2)
    {k : ℕ} (hk : k < o.tabs[i]!.m - 13) :
    (Claims.coinSeq o.r (repStart o.spec o.sts (2 * i + r) + 1)).at k ≠ 1 := by
  intro h1
  apply o.good
  refine ⟨(repStart o.spec o.sts (2 * i + r) + 1, k), ?_, h1⟩
  have hsz : o.tabs.size = o.sts.size := by simp [Run.tabs, tabsOf]
  have hq : (2 * i + r) / 2 = i := by omega
  unfold outerSlots
  rw [o.setup]
  simp only [List.mem_flatMap, List.mem_range, List.mem_map, Prod.mk.injEq]
  exact ⟨2 * i + r, by omega, k, by rw [hq, ← (o.tab_m hi).1]; exact hk, rfl, rfl⟩

theorem _root_.FlockComplete.Run.zc_c1 (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) {r : ℕ} (hr : r < 2) :
    ∀ k < o.tabs[i]!.m - 13, (o.zcCoins i r).c1 k ≠ 1 := by
  intro k hk
  show coinOf (Claims.coinSeq o.r (repStart o.spec o.sts (2 * i + r) + 1)).bytes k ≠ 1
  rw [JsonRT.coinOf_bytes]
  exact o.outer_ne_one hi hr hk

theorem zerocheck_replay (hs : Hashes V) {m : ℕ} (hm : 14 ≤ m) (digest cap : ByteArray) (cs : Zc.Challenges) (tr : Trace)
    (R : Array Round) (dom : ByteArray)
    (hR : Agrees R (Zc.tEnd m digest cap cs tr (Transcript.init #[] dom)).rounds 0)
    (hR1CS : ∀ x < 2 ^ m, (tr.az[x]! && tr.bz[x]!) = tr.z[x]!) (hc1 : ∀ k < m - 13, cs.c1 k ≠ 1) :
    ∃ v, (bindAndZerocheckWith hs m digest cap (Zc.proof m cs tr)).run (Transcript.init R dom) = .ok v :=
  ⟨_, zerocheckWith_of hs hm digest cap cs tr (Transcript.init #[] dom) rfl hR (Zc.fold_final hm cs tr hR1CS hc1)⟩

def TraceR1CS (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ x < 2 ^ o.tabs[i]!.m,
    ((o.trace i).az[x]! && (o.trace i).bz[x]!) = (o.trace i).z[x]!

def ZcAt (o : Run H rel) (i r : ℕ) : Prop :=
  (o.proof i r).zerocheck = Zc.proof o.tabs[i]!.m (o.zcCoins i r) (o.trace i) ∧
    Agrees (o.stream i r).rounds (Zc.tEnd o.tabs[i]!.m o.tabs[i]!.digest (flat (o.proof i r).cap) (o.zcCoins i r)
      (o.trace i) (Transcript.init #[] (o.spec.domain o.spec.tables[i]! r).toUTF8)).rounds 0

theorem ok_val {α : Type} [Inhabited α] {x : Except String α} (h : ∃ v, x = .ok v) : x = .ok (val x) := by
  obtain ⟨v, rfl⟩ := h
  rfl

theorem zerocheck_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hR1 : TraceR1CS H rel) (hzc : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, ZcAt o i r) : Zerocheck H rel := by
  intro o i hi r hr
  obtain ⟨hp, hag⟩ := hzc o i hi r hr
  obtain ⟨-, hm, -⟩ := o.tab_m hi
  refine ok_val ?_
  rw [hp]
  exact zerocheck_replay o.hs (by omega) _ _ _ _ _ _ hag (hR1 o i hi) (o.zc_c1 hi hr)

theorem Ret.run_bind {α β : Type} {x : TMW PS α} {f : α → TMW PS β} {t : Transcript} {Q : α × Transcript → Prop}
    {P : β × Transcript → Prop} (hx : Ret (x.run t) Q) (hf : ∀ a t', Q (a, t') → Ret ((f a).run t') P) :
    Ret ((x >>= f).run t) P := by
  rw [StateT.run_bind]
  exact FlockComplete.Ret.bind hx fun ⟨a, t'⟩ h => hf a t' h

theorem repP_zerocheck (resp : ℕ → Claims.Msg) (hsP : Hashes PS) (st : Setup) (sch : Schedule) (tr : Trace)
    (tbl : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (hm : 14 ≤ st.m) (t : Transcript) (l : Array Claims.Msg) {p : Proof} {t' : Transcript} {l' : Array Claims.Msg}
    (h : ((repP (asksS resp) hsP st sch tr tbl cm0 extras salt nonce).run t).go l = (.ok (p, t'), l')) :
    p.zerocheck = Zc.proof st.m (Zc.coinsAt resp l.size) tr ∧ p.cap = cm0.cap := by
  obtain ⟨l₁, -, hz⟩ := zerocheckP_go resp hm st.digest (flat cm0.cap) tr t l
  unfold repP at h
  rw [bind_go hz] at h
  refine (?_ : Ret _ fun r : Proof × Transcript => r.1.zerocheck = Zc.proof st.m (Zc.coinsAt resp l.size) tr ∧
    r.1.cap = cm0.cap) l₁ (p, t') l' h
  refine Ret.run_bind (Ret.true _) fun ⟨lp, lc⟩ _ _ => ?_
  refine Ret.run_bind (Ret.true _) fun ⟨ss, T, sws⟩ _ _ => ?_
  refine Ret.run_bind (Ret.true _) fun lg _ _ => ?_
  rw [StateT.run_pure]
  exact FlockComplete.Ret.pure ⟨rfl, rfl⟩

end FlockComplete.Replay
