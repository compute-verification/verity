module

public import Proofs.Flock.Claims.Recursion.InnerAbort
public import Proofs.Flock.Claims.Recursion.InnerAccepts
public import Proofs.Flock.Claims.Recursion.Registered
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
import Mathlib.Tactic.ENatToNat
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
import Proofs.Flock.Claims.Finder
import Proofs.Flock.Claims.Prob
import Std.Tactic.BVDecide.Normalize.Prop

/-! `VStarAccepts` at the deployed readers, reduced to named facts each about one V\* part's circuit, the staged V\* statements or the inner verifier. -/

@[expose] public section

namespace Claims.Recursion.VStar

open Claims

/-- A byte from up to eight bits, the low bit first. -/
def byte (bs : List Bool) : UInt8 := bs.foldr (fun b a => (if b then 1 else 0) + 2 * a) 0

/-- **Bits as bytes**: eight to a byte, the low bit first, a short last byte padded with zeros. Any other bit order is
this one at permuted positions, which a layout lists. -/
def bytesOf (bs : List Bool) : Msg := (List.range ((bs.length + 7) / 8)).map fun j => byte ((bs.drop (8 * j)).take 8)

def readAt (w : Flock.Wit) (ps : List ℕ) : Msg := bytesOf (ps.map fun p => w.hid.getD p false)

def readSegs (w : Flock.Wit) (segs : List (Msg × List ℕ)) : Msg := segs.flatMap fun p => p.1 ++ readAt w p.2

/-- **Where V\*'s parts hold what the readers read**, fixed by the compiled statement `x`: the inner rounds' count
(`rounds x`); where part `s` holds inner round `i`'s message (`msgAt x s i`, `none` where the part holds none), as
segments, each constant bytes the statement fixes (a round's frame, a length prefix) then the hidden values of the
next bits, in order; each SHA-512 evaluation part `s` computes (`shaAt x s`): the constant bytes its input starts with
(a domain prefix the circuit fixes, such as a row's or the salt's), then the hidden values of the rest of its input's
bits, and those of its output's; the rounds whose message the statement fixes (`fixedAt x i`: a stream round that
absorbs no message, whose framed message is constant), which no part holds; and the public rounds (`pubAt x i`), whose
message is the run's own, sent in the clear, as the link message is. -/
structure Layout where
  rounds : Msg → ℕ
  msgAt : Msg → ℕ → ℕ → Option (List (Msg × List ℕ))
  shaAt : Msg → ℕ → List (Msg × List ℕ × List ℕ)
  fixedAt : Msg → ℕ → Option Msg := fun _ _ => none
  pubAt : Msg → ℕ → Bool := fun _ _ => false

def Layout.held (L : Layout) (x : Msg) (s : ℕ) (w : Flock.Wit) (i : ℕ) : Option Msg :=
  if (L.fixedAt x i).isSome then none else (L.msgAt x s i).map (readSegs w)

def Layout.msg (L : Layout) (x : Msg) (ws : List Flock.Wit) (i : ℕ) : Msg :=
  match L.fixedAt x i with
  | some m => m
  | none => ((List.range ws.length).findSome? fun s => ws[s]?.bind fun w => L.held x s w i).getD []

def Layout.msgs (L : Layout) (x : Msg) (ws : List Flock.Wit) : List Msg := (List.range (L.rounds x)).map (L.msg x ws)

def Layout.calls (L : Layout) (x : Msg) (s : ℕ) (w : Flock.Wit) : List (Msg × Msg) :=
  (L.shaAt x s).map fun p => (p.1 ++ readAt w p.2.1, readAt w p.2.2)

theorem Layout.msgs_getD {L : Layout} {x : Msg} {ws : List Flock.Wit} {i : ℕ} (hi : i < L.rounds x) {s : ℕ}
    (hs : s < ws.length) {m : Msg} (hm : L.held x s ws[s] i = some m)
    (hall : ∀ s' (hs' : s' < ws.length) m', L.held x s' ws[s'] i = some m' → m' = m) :
    (L.msgs x ws).getD i [] = m := by
  have hget : (L.msgs x ws).getD i [] = L.msg x ws i := by
    simp [Layout.msgs, List.getD_eq_getElem?_getD, hi]
  have hfx : L.fixedAt x i = none := by
    cases h : L.fixedAt x i with
    | none => rfl
    | some _ => simp [Layout.held, h] at hm
  simp only [hget, Layout.msg, hfx]
  cases hf : (List.range ws.length).findSome? (fun s => ws[s]?.bind fun w => L.held x s w i) with
  | none =>
    have := List.findSome?_eq_none_iff.1 hf s (List.mem_range.2 hs)
    simp [List.getElem?_eq_getElem hs, hm] at this
  | some m₀ =>
    obtain ⟨s₀, hs₀, h₀⟩ := List.exists_of_findSome?_eq_some hf
    have hs₀' := List.mem_range.1 hs₀
    simp only [List.getElem?_eq_getElem hs₀', Option.bind] at h₀
    exact hall s₀ hs₀' m₀ h₀

theorem Layout.msgs_getD_held {L : Layout} {x : Msg} {ws : List Flock.Wit} {i : ℕ} (hi : i < L.rounds x) {s : ℕ}
    (hs : s < ws.length) (hfx : L.fixedAt x i = none) (hh : (L.msgAt x s i).isSome) :
    ∃ s₀, ∃ hs₀ : s₀ < ws.length, L.held x s₀ ws[s₀] i = some ((L.msgs x ws).getD i []) := by
  have hget : (L.msgs x ws).getD i [] = L.msg x ws i := by
    simp [Layout.msgs, List.getD_eq_getElem?_getD, hi]
  simp only [hget, Layout.msg, hfx]
  cases hf : (List.range ws.length).findSome? (fun s => ws[s]?.bind fun w => L.held x s w i) with
  | none =>
    have := List.findSome?_eq_none_iff.1 hf s (List.mem_range.2 hs)
    obtain ⟨ps, hps⟩ := Option.isSome_iff_exists.1 hh
    simp [List.getElem?_eq_getElem hs, Layout.held, hps, hfx] at this
  | some m₀ =>
    obtain ⟨s₀, hs₀, h₀⟩ := List.exists_of_findSome?_eq_some hf
    have hs₀' := List.mem_range.1 hs₀
    simp only [List.getElem?_eq_getElem hs₀', Option.bind] at h₀
    exact ⟨s₀, hs₀', by simpa using h₀⟩

theorem Layout.msgs_getD_fixed {L : Layout} {x : Msg} {ws : List Flock.Wit} {i : ℕ} (hi : i < L.rounds x) {m : Msg}
    (hm : L.fixedAt x i = some m) : (L.msgs x ws).getD i [] = m := by
  simp [Layout.msgs, List.getD_eq_getElem?_getD, hi, Layout.msg, hm]

theorem stepWith_msg {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out) (n : ℕ)
    (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript) {m : Msg}
    (hv : ((vn (as ++ [((P.next x hist ω).eval H).1])).eval H).1 = .msg m) :
    stepWith H P x ω vn (n + 1) hist as t =
      stepWith H P x ω vn n (hist ++ [(((P.next x hist ω).eval H).1, m)]) (as ++ [((P.next x hist ω).eval H).1])
        { t with
          msgs := t.msgs ++ [(((P.next x hist ω).eval H).1, ((vn (as ++ [((P.next x hist ω).eval H).1])).eval H).1)],
          pLog := t.pLog ++ ((P.next x hist ω).eval H).2,
          vLog := t.vLog ++ ((vn (as ++ [((P.next x hist ω).eval H).1])).eval H).2 } := by
  simp only [stepWith]
  rw [hv]

theorem stepWith_replay_congr (H : Msg → Msg) (x : Msg) (vn : List Msg → OC Out) (ms : List Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript) (ms' : List Msg), hist.length = t.msgs.length →
      (∀ j, t.msgs.length ≤ j → j < (stepWith H (replay ms) x () vn n hist as t).msgs.length →
        ms'.getD j [] = ms.getD j []) →
      stepWith H (replay ms') x () vn n hist as t = stepWith H (replay ms) x () vn n hist as t
  | 0, _, _, _, _, _, _ => rfl
  | n + 1, hist, as, t, ms', hl, h => by
    have hlt : t.msgs.length < (stepWith H (replay ms) x () vn (n + 1) hist as t).msgs.length := by
      have := congrArg List.length (stepWith_succ_fst_take H (replay ms) x () vn n hist as t)
      simp only [List.length_take, List.length_map, List.length_append, List.length_singleton] at this
      omega
    have hm : ms'.getD hist.length [] = ms.getD hist.length [] := h _ hl.ge (hl ▸ hlt)
    have e1 : ((replay ms').next x hist ()).eval H = ((replay ms).next x hist ()).eval H := by
      simp only [replay, OC.eval, hm]
    simp only [stepWith]
    rw [e1]
    split
    · next m hv =>
      rw [stepWith_msg H _ x () vn n hist as t hv] at h
      exact stepWith_replay_congr H x vn ms n _ _ _ ms' (by simp [hl])
        fun j hj1 hj2 => h j (by simp only [List.length_append, List.length_singleton] at hj1; omega) hj2
    · rfl
    · rfl

theorem stepWith_replay_fst (H : Msg → Msg) (x : Msg) (vn : List Msg → OC Out) (ms : List Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript), hist.length = t.msgs.length →
      ∀ j, t.msgs.length ≤ j → j < (stepWith H (replay ms) x () vn n hist as t).msgs.length →
        ((stepWith H (replay ms) x () vn n hist as t).msgs.map Prod.fst).getD j [] = ms.getD j []
  | 0, _, _, t, _, j, hj1, hj => by simp only [stepWith] at hj; omega
  | n + 1, hist, as, t, hl, j, hj1, hj => by
    rcases Nat.lt_or_ge j (t.msgs.length + 1) with hlt | hge
    · obtain rfl : j = t.msgs.length := by omega
      have h2 : ((stepWith H (replay ms) x () vn (n + 1) hist as t).msgs.map Prod.fst)[t.msgs.length]? =
          some (((replay ms).next x hist ()).eval H).1 := by
        have := congrArg (·[t.msgs.length]?) (stepWith_succ_fst_take H (replay ms) x () vn n hist as t)
        simpa [List.getElem?_take] using this
      rw [List.getD_eq_getElem?_getD, h2]
      simp [replay, OC.eval, hl]
    · rcases hv : ((vn (as ++ [(((replay ms).next x hist ()).eval H).1])).eval H).1 with m | _ | _
      · rw [stepWith_msg H _ x () vn n hist as t hv] at hj ⊢
        exact stepWith_replay_fst H x vn ms n _ _ _ (by simp [hl]) j (by simpa using hge) hj
      · simp only [stepWith, hv, List.length_append, List.length_singleton] at hj
        omega
      · simp only [stepWith, hv, List.length_append, List.length_singleton] at hj
        omega

theorem run_replay_self {Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (x : Msg) (ms : List Msg)
    (r : Fin (V.k x) → Coin) :
    run H (replay ((run H (replay ms) V x () r).msgs.map Prod.fst)) V x () r = run H (replay ms) V x () r := by
  exact stepWith_replay_congr H x _ ms (V.k x) [] [] _ _ rfl fun j _ hj =>
    stepWith_replay_fst H x _ ms (V.k x) [] [] _ rfl j (Nat.zero_le _) hj

theorem stepWith_length_le {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript),
      (stepWith H P x ω vn n hist as t).msgs.length ≤ t.msgs.length + n
  | 0, _, _, _ => le_refl _
  | n + 1, hist, as, t => by
    rcases hv : ((vn (as ++ [((P.next x hist ω).eval H).1])).eval H).1 with m | _ | _
    · rw [stepWith_msg H P x ω vn n hist as t hv]
      refine (stepWith_length_le H P x ω vn n _ _ _).trans ?_
      simp only [List.length_append, List.length_singleton]
      omega
    · simp [stepWith, hv]
    · simp [stepWith, hv]

theorem stepWith_length_gt {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out)
    {n : ℕ} (hn : 0 < n) (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript) :
    t.msgs.length < (stepWith H P x ω vn n hist as t).msgs.length := by
  obtain ⟨n, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
  have := congrArg List.length (stepWith_succ_fst_take H P x ω vn n hist as t)
  simp only [List.length_take, List.length_map, List.length_append, List.length_singleton] at this
  omega

theorem stepWith_msg_of_lt {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript) (i : ℕ), t.msgs.length ≤ i →
      i + 1 < (stepWith H P x ω vn n hist as t).msgs.length →
        ∃ m, ((stepWith H P x ω vn n hist as t).msgs[i]?).map Prod.snd = some (.msg m)
  | 0, _, _, t, i, hi, h => by simp only [stepWith] at h; omega
  | n + 1, hist, as, t, i, hi, h => by
    rcases hv : ((vn (as ++ [((P.next x hist ω).eval H).1])).eval H).1 with m | _ | _
    · rw [stepWith_msg H P x ω vn n hist as t hv] at h ⊢
      rcases Nat.lt_or_ge i (t.msgs.length + 1) with hlt | hge
      · obtain rfl : i = t.msgs.length := by omega
        obtain ⟨l, hl⟩ := stepWith_prefix H P x ω vn n _ _ _
        refine ⟨m, ?_⟩
        rw [← hl]
        simp [hv]
      · exact stepWith_msg_of_lt H P x ω vn n _ _ _ i (by simpa using hge) h
    · simp only [stepWith, hv, List.length_append, List.length_singleton] at h
      omega
    · simp only [stepWith, hv, List.length_append, List.length_singleton] at h
      omega

theorem stepWith_last_of_lt {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript),
      (stepWith H P x ω vn n hist as t).msgs.length < t.msgs.length + n →
        ∀ m, ((stepWith H P x ω vn n hist as t).msgs.getLast?).map Prod.snd ≠ some (.msg m)
  | 0, _, _, t, h => by simp only [stepWith] at h; omega
  | n + 1, hist, as, t, h => by
    rcases hv : ((vn (as ++ [((P.next x hist ω).eval H).1])).eval H).1 with m | _ | _
    · rw [stepWith_msg H P x ω vn n hist as t hv] at h ⊢
      exact stepWith_last_of_lt H P x ω vn n _ _ _ (by simp only [List.length_append, List.length_singleton]; omega)
    · simp [stepWith, hv]
    · simp [stepWith, hv]

theorem stepWith_replay_take (H : Msg → Msg) (x : Msg) (vn vn' : List Msg → OC Out) (ms ms' : List Msg) (N : ℕ) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript), hist.length = t.msgs.length →
      as.length = t.msgs.length → (∀ i, t.msgs.length ≤ i → i < N → ms.getD i [] = ms'.getD i []) →
      (∀ bs : List Msg, bs.length ≤ N → vn bs = vn' bs) →
      (stepWith H (replay ms) x () vn n hist as t).msgs.take N =
        (stepWith H (replay ms') x () vn' n hist as t).msgs.take N
  | 0, _, _, _, _, _, _, _ => rfl
  | n + 1, hist, as, t, hl, ha, hm, hv => by
    rcases Nat.lt_or_ge t.msgs.length N with hN | hN
    · have e1 : ((replay ms').next x hist ()).eval H = ((replay ms).next x hist ()).eval H := by
        simp only [replay, OC.eval, hm hist.length hl.ge (hl ▸ hN)]
      have e2 : vn' (as ++ [(((replay ms).next x hist ()).eval H).1]) =
          vn (as ++ [(((replay ms).next x hist ()).eval H).1]) :=
        (hv _ (by simp only [List.length_append, List.length_singleton]; omega)).symm
      simp only [stepWith]
      rw [e1, e2]
      split
      · exact stepWith_replay_take H x vn vn' ms ms' N n _ _ _ (by simp [hl]) (by simp [ha])
          (fun i hi1 hi2 => hm i (by simp only [List.length_append, List.length_singleton] at hi1; omega) hi2) hv
      · rfl
      · rfl
    · have key : ∀ (P : Prover Unit) (v : List Msg → OC Out),
          (stepWith H P x () v (n + 1) hist as t).msgs.take N = t.msgs.take N := fun P v => by
        obtain ⟨l, hl'⟩ := stepWith_prefix H P x () v (n + 1) hist as t
        rw [← hl', List.take_append_of_le_length hN]
      rw [key, key]

theorem run_replay_take {Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (x : Msg)
    (ms ms' : List Msg) (c c' : Fin (V.k x) → Coin) (N : ℕ) (hm : ∀ i < N, ms.getD i [] = ms'.getD i [])
    (hc : ∀ i < N, coinSeq c i = coinSeq c' i) :
    (run H (replay ms) V x () c).msgs.take N = (run H (replay ms') V x () c').msgs.take N :=
  stepWith_replay_take H x _ _ ms ms' N (V.k x) [] [] _ rfl rfl (fun i _ hi => hm i hi) fun bs hb => by
    congr 1
    exact List.map_congr_left fun i hi => hc i (by rw [List.mem_range] at hi; omega)

theorem run_replay_plays {Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (x : Msg)
    (ms ms' : List Msg) (c c' : Fin (V.k x) → Coin) (j : ℕ) (hm : ∀ i < j, ms.getD i [] = ms'.getD i [])
    (hc : ∀ i < j, coinSeq c i = coinSeq c' i) (hj : j < (run H (replay ms) V x () c).msgs.length) :
    j < (run H (replay ms') V x () c').msgs.length := by
  by_contra hn
  push Not at hn
  have ht := run_replay_take H V x ms ms' c c' j hm hc
  rw [List.take_of_length_le hn] at ht
  have hlen : (run H (replay ms') V x () c').msgs.length = j := by
    rw [← ht, List.length_take]
    omega
  have hk : j < V.k x := by
    have := stepWith_length_le H (replay ms) x ()
      (fun as => V.next x as ((List.range as.length).map (coinSeq c))) (V.k x) [] [] ⟨[], [], [], .rej⟩
    simp only [List.length_nil, Nat.zero_add] at this
    exact lt_of_lt_of_le hj this
  have hlast := stepWith_last_of_lt H (replay ms') x ()
    (fun as => V.next x as ((List.range as.length).map (coinSeq c'))) (V.k x) [] [] ⟨[], [], [], .rej⟩
    (by simp only [List.length_nil, Nat.zero_add]; exact hlen ▸ hk)
  rcases j with _ | j
  · have := stepWith_length_gt H (replay ms') x ()
      (fun as => V.next x as ((List.range as.length).map (coinSeq c'))) hk [] [] ⟨[], [], [], .rej⟩
    simp only [List.length_nil] at this
    unfold run at hlen
    omega
  · obtain ⟨m, hm'⟩ := stepWith_msg_of_lt H (replay ms) x ()
      (fun as => V.next x as ((List.range as.length).map (coinSeq c))) (V.k x) [] [] ⟨[], [], [], .rej⟩ j
      (Nat.zero_le _) hj
    have hg : (run H (replay ms') V x () c').msgs.getLast? = (run H (replay ms) V x () c).msgs[j]? := by
      rw [List.getLast?_eq_getElem?, hlen, Nat.add_sub_cancel, ← ht, List.getElem?_take]
      simp
    exact hlast m (by unfold run at hg; rw [hg]; exact hm')

theorem run_length_le {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg)
    (ω : Ω) (r : Fin (V.k x) → Coin) : (run H P V x ω r).msgs.length ≤ V.k x := by
  have := stepWith_length_le H P x ω (fun as => V.next x as ((List.range as.length).map (coinSeq r))) (V.k x) [] []
    ⟨[], [], [], .rej⟩
  simp only [List.length_nil, Nat.zero_add] at this
  exact this

theorem run_replay_fst {Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (x : Msg) (ms : List Msg)
    (r : Fin (V.k x) → Coin) {j : ℕ} (hj : j < (run H (replay ms) V x () r).msgs.length) :
    ((run H (replay ms) V x () r).msgs.map Prod.fst).getD j [] = ms.getD j [] :=
  stepWith_replay_fst H x _ ms (V.k x) [] [] _ rfl j (Nat.zero_le _) hj

section Facts

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

def ReadsThrough (H : Msg → Msg) (D : Deploy Coin) (IV : Verifier Coin) (msgs : Msg → List Flock.Wit → List Msg)
    (calls : Msg → ℕ → Flock.Wit → List (Msg × Msg)) : Prop :=
  (∀ x r ws, D.innerView H x r ws = (run H (replay (msgs x ws)) IV (xI D x) () fun i => coinSeq r i.val).seen) ∧
    ∀ x ws, D.calls x ws = (ws.mapIdx fun s w => calls x s w).flatten

theorem mem_calls_of_part {calls : Msg → ℕ → Flock.Wit → List (Msg × Msg)} {x : Msg} {ws : List Flock.Wit} {s : ℕ}
    (hs : s < ws.length) {e : Msg × Msg} (he : e ∈ calls x s ws[s]) :
    e ∈ (ws.mapIdx fun s w => calls x s w).flatten :=
  List.mem_flatten.2 ⟨_, List.getElem_mem (l := ws.mapIdx fun s w => calls x s w) (by simpa using hs),
    by simpa [List.getElem_mapIdx] using he⟩

theorem part_of_mem_calls {calls : Msg → ℕ → Flock.Wit → List (Msg × Msg)} {x : Msg} {ws : List Flock.Wit}
    {e : Msg × Msg} (he : e ∈ (ws.mapIdx fun s w => calls x s w).flatten) :
    ∃ s, ∃ hs : s < ws.length, e ∈ calls x s ws[s] := by
  obtain ⟨l, hl, he⟩ := List.mem_flatten.1 he
  obtain ⟨s, hs, rfl⟩ := List.getElem_of_mem hl
  exact ⟨s, by simpa using hs, by simpa [List.getElem_mapIdx] using he⟩

theorem innerViewRun_of_readsThrough {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {msgs : Msg → List Flock.Wit → List Msg}
    {calls : Msg → ℕ → Flock.Wit → List (Msg × Msg)} (hR : ReadsThrough H D I.V msgs calls) :
    InnerViewRun H D I DO := by
  intro decO x Ω P ω r c ws _
  unfold innerRun
  rw [hR.1 x r ws]
  exact congrArg Transcript.seen (run_replay_self H I.V (xI D x) (msgs x ws) _).symm

omit [Fintype Coin] [Nonempty Coin] in
theorem callsNil_of_readsThrough {H : Msg → Msg} {D : Deploy Coin} {IV : Verifier Coin}
    {msgs : Msg → List Flock.Wit → List Msg} {calls : Msg → ℕ → Flock.Wit → List (Msg × Msg)}
    (hR : ReadsThrough H D IV msgs calls) : CallsNil H D := fun x e he => by
  rw [hR.2 x []] at he
  simp at he

def RegAt (H : Msg → Msg) (D : Deploy Coin) (x : Msg) (xs : List Msg) (ws : List Flock.Wit) : Prop :=
  (findCollIn (D.calls x ws) (setupCalls H D xs)).isSome = true

def CallsColl (H : Msg → Msg) (D : Deploy Coin) (x : Msg) (xs : List Msg) (ws : List Flock.Wit) : Prop :=
  (∃ u v, findColl (D.calls x ws) = some (u, v) ∧ Collision H u v) ∨ RegAt H D x xs ws

def VstarLen (D : Deploy Coin) : Prop := ∀ x ms, D.nS x ≤ (D.vstar x ms).length

omit [Fintype Coin] [Nonempty Coin] in
theorem regAt_of_part {H : Msg → Msg} {D : Deploy Coin} {IV : Verifier Coin} {L : Layout}
    (hR : ReadsThrough H D IV L.msgs L.calls) {x : Msg} {xs : List Msg} {ws : List Flock.Wit} {s : ℕ}
    (hs : s < ws.length) (hxs : s < xs.length) (h : SetupColl H D (xs.getD s []) (L.calls x s ws[s])) :
    RegAt H D x xs ws := by
  unfold RegAt
  rw [hR.2 x ws]
  exact setupColl_lift (by rw [List.getD_eq_getElem _ _ hxs]; exact List.getElem_mem hxs)
    (fun e he => mem_calls_of_part hs he) h

omit [Fintype Coin] [Nonempty Coin] in
theorem callsColl_of_part {H : Msg → Msg} {D : Deploy Coin} {IV : Verifier Coin} {L : Layout}
    (hR : ReadsThrough H D IV L.msgs L.calls) {x : Msg} {xs : List Msg} {ws : List Flock.Wit} {s : ℕ}
    (hs : s < ws.length) (hxs : s < xs.length) (htrue : ∀ e ∈ D.calls x ws, H e.1 = e.2)
    (h : (findColl (L.calls x s ws[s])).isSome ∨ SetupColl H D (xs.getD s []) (L.calls x s ws[s])) :
    CallsColl H D x xs ws := by
  rcases h with h | h
  · left
    rw [hR.2 x ws] at htrue ⊢
    obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1
      (findColl_isSome_of_sub H htrue (fun _ he => mem_calls_of_part hs he) h)
    exact ⟨u, v, huv, findColl_spec H _ htrue u v huv⟩
  · exact Or.inr (regAt_of_part hR hs hxs h)

def PartChecks (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → (∀ e ∈ L.calls x s w, H e.1 = e.2) →
    (∀ (c : ℕ → Coin), shows s x xO c → ∀ ms : List Msg,
      (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) → chk s x c ms) ∨
      (findColl (L.calls x s w)).isSome ∨ SetupColl H D xO (L.calls x s w)

def PartOpens (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (opening : Opening) (cmt : Msg → ℕ → Msg → Msg) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → (∀ e ∈ L.calls x s w, H e.1 = e.2) →
      (∀ (c : ℕ → Coin), shows s x xO c → ∀ i < L.rounds x, ∀ m, L.held x s w i = some m →
        ∃ cs, cs ⊆ L.calls x s w ∧ opening.opens (cmt x i xO) m cs) ∨
        (findColl (L.calls x s w)).isSome ∨ SetupColl H D xO (L.calls x s w)

def Staged (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form →
    ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin), (run H P D.V x ω r).out = .acc →
      ∀ s < D.nS x, shows s x ((runVstar H D I P x ω r).getD s []) (coinSeq r) ∧
        ∀ i < L.rounds x,
          cmt x i ((runVstar H D I P x ω r).getD s []) = ((run H P D.V x ω r).msgs.map Prod.fst).getD i []

def Decomp (H : Msg → Msg) (D : Deploy Coin) (IV : Verifier Coin) (shape : Form → Prop)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (c : ℕ → Coin) (ms : List Msg),
    (∀ s < D.nS x, chk s x c ms) → (run H (replay ms) IV (xI D x) () fun i => c i.val).out = .acc

def InnerConj (H : Msg → Msg) (D : Deploy Coin) (IV : Verifier Coin) (shape : Form → Prop) (nC : Msg → ℕ)
    (check : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (c : ℕ → Coin) (ms : List Msg),
    (∀ j < nC x, check j x c ms) → (run H (replay ms) IV (xI D x) () fun i => c i.val).out = .acc

def Covers (D : Deploy Coin) (shape : Form → Prop) (nC : Msg → ℕ) (check : ℕ → Msg → (ℕ → Coin) → List Msg → Prop)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ j < nC x, ∃ s < D.nS x,
    ∀ (c : ℕ → Coin) (ms : List Msg), chk s x c ms → check j x c ms

omit [Fintype Coin] [Nonempty Coin] in
theorem decomp_of_covers {H : Msg → Msg} {D : Deploy Coin} {IV : Verifier Coin} {shape : Form → Prop} {nC : Msg → ℕ}
    {check chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} (hI : InnerConj H D IV shape nC check)
    (hC : Covers D shape nC check chk) : Decomp H D IV shape chk := fun x st hst hf c ms h =>
  hI x st hst hf c ms fun j hj => by
    obtain ⟨s, hs, hs'⟩ := hC x st hst hf j hj
    exact hs' c ms (h s hs)

def PartCallsTrue (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : Layout) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → ∀ (xO : Msg) (w : Flock.Wit), Flock.check DO.parse xO w = true →
    ∀ e ∈ L.calls x s w, H e.1 = e.2

def CallsTrue (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
    ∀ e ∈ D.calls x ws, H e.1 = e.2

theorem CallsRecord.toTrue {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (h : CallsRecord H D I DO) : CallsTrue H D I DO := fun decO x Ω P ω r c ws hg => (h decO x Ω P ω r c ws hg).2

theorem callsTrue_of_parts {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {IV : Verifier Coin} {L : Layout} (hRef : Refuses H D) (hR : ReadsThrough H D IV L.msgs L.calls)
    (hT : ∀ s, PartCallsTrue H D DO L s) : CallsTrue H D I DO := by
  intro decO x Ω P ω r c ws hg e he
  obtain ⟨st, hst, -⟩ := hRef x Ω P ω r hg.1
  obtain ⟨hlen, hchk⟩ := eoGives_outerHolds hg
  rw [hR.2 x ws] at he
  obtain ⟨s, hs, he⟩ := part_of_mem_calls he
  exact hT s x st hst _ ws[s] (hchk s hs) e he

theorem callsRecord_of_parts {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {IV : Verifier Coin} {L : Layout} (hRef : Refuses H D) (hR : ReadsThrough H D IV L.msgs L.calls)
    (hT : ∀ s, PartCallsTrue H D DO L s)
    (hlog : ∀ (decO : List (ℕ × Col) → List T) (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω)
      (r : Fin (D.V.k x) → Coin) (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
      ∀ e ∈ (innerRun H D I x r ws).vLog, e ∈ D.calls x ws) :
    CallsRecord H D I DO := fun decO x Ω P ω r c ws hg =>
  ⟨hlog decO x Ω P ω r c ws hg, callsTrue_of_parts (I := I) hRef hR hT decO x Ω P ω r c ws hg⟩

theorem coll_of_sub {H : Msg → Msg} {l l' : List (Msg × Msg)} (hl' : ∀ p ∈ l', H p.1 = p.2)
    (hsub : ∀ p ∈ l, p ∈ l') (h : (findColl l).isSome) : ∃ u v, findColl l' = some (u, v) ∧ Collision H u v := by
  obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 (findColl_isSome_of_sub H hl' hsub h)
  exact ⟨u, v, huv, findColl_spec H l' hl' u v huv⟩

theorem coll_of_part {H : Msg → Msg} {calls : Msg → ℕ → Flock.Wit → List (Msg × Msg)} {x : Msg}
    {ws : List Flock.Wit} {s : ℕ} (hs : s < ws.length)
    (htrue : ∀ e ∈ (ws.mapIdx fun s w => calls x s w).flatten, H e.1 = e.2) (h : (findColl (calls x s ws[s])).isSome) :
    ∃ u v, findColl (ws.mapIdx fun s w => calls x s w).flatten = some (u, v) ∧ Collision H u v :=
  coll_of_sub htrue (fun _ he => mem_calls_of_part hs he) h

omit [Fintype Coin] [Nonempty Coin] in
theorem agree_or_coll {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {IV : Verifier Coin} {L : Layout} {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {opening : Opening} {cmt : Msg → ℕ → Msg → Msg} (hR : ReadsThrough H D IV L.msgs L.calls) (hVL : VstarLen D)
    (hO : ∀ s, PartOpens H D DO L shape shows opening cmt s) (hS : Staged H D I L shape shows cmt) {x : Msg} {st : CompiledStatement}
    (hst : D.parse x = some st) (hf : shape st.form) {Ω : Type} {P : Prover Ω} {ω : Ω} {r : Fin (D.V.k x) → Coin}
    (hacc : (run H P D.V x ω r).out = .acc) {ws : List Flock.Wit}
    (ho : OuterHolds DO (runVstar H D I P x ω r) (D.nS x) ws) (htrue : ∀ e ∈ D.calls x ws, H e.1 = e.2) :
    (∀ s (hs : s < ws.length), ∀ i < L.rounds x, ∀ m, L.held x s ws[s] i = some m → (L.msgs x ws).getD i [] = m) ∨
      CallsColl H D x (runVstar H D I P x ω r) ws := by
  obtain ⟨hlen, hchk⟩ := ho
  have hxs : ∀ s < ws.length, s < (runVstar H D I P x ω r).length := fun s hs =>
    lt_of_lt_of_le (hlen ▸ hs) (hVL x _)
  have hpt : ∀ s (hs : s < ws.length), ∀ e ∈ L.calls x s ws[s], H e.1 = e.2 := fun s hs e he =>
    htrue e (by rw [hR.2 x ws]; exact mem_calls_of_part hs he)
  by_cases hdiff : ∃ i < L.rounds x, ∃ s, ∃ hs : s < ws.length, ∃ s', ∃ hs' : s' < ws.length, ∃ m m',
      L.held x s ws[s] i = some m ∧ L.held x s' ws[s'] i = some m' ∧ m ≠ m'
  · obtain ⟨i, hi, s, hs, s', hs', m, m', hm, hm', hne⟩ := hdiff
    rcases hO s x st hst hf _ ws[s] (hchk s hs) (hpt s hs) with hop | hc
    · obtain ⟨cs, hcs, hop⟩ := hop _ (hS x st hst hf Ω P ω r hacc s (hlen ▸ hs)).1 i hi m hm
      rcases hO s' x st hst hf _ ws[s'] (hchk s' hs') (hpt s' hs') with hop' | hc'
      · obtain ⟨cs', hcs', hop'⟩ := hop' _ (hS x st hst hf Ω P ω r hacc s' (hlen ▸ hs')).1 i hi m' hm'
        rw [(hS x st hst hf Ω P ω r hacc s (hlen ▸ hs)).2 i hi] at hop
        rw [(hS x st hst hf Ω P ω r hacc s' (hlen ▸ hs')).2 i hi] at hop'
        refine Or.inr (Or.inl ?_)
        rw [hR.2 x ws] at htrue ⊢
        have hsub : cs ++ cs' ⊆ (ws.mapIdx fun s w => L.calls x s w).flatten := fun p hp =>
          (List.mem_append.1 hp).elim (fun h => mem_calls_of_part hs (hcs h)) fun h => mem_calls_of_part hs' (hcs' h)
        obtain ⟨q, hq, q', hq', hc⟩ := opening.bind H _ m m' cs cs' hop hop' hne fun e he => htrue e (hsub he)
        exact findColl_of_coll H htrue (List.map_subset _ hsub hq) (List.map_subset _ hsub hq') hc
      · exact Or.inr (callsColl_of_part hR hs' (hxs s' hs') htrue hc')
    · exact Or.inr (callsColl_of_part hR hs (hxs s hs) htrue hc)
  · push Not at hdiff
    exact Or.inl fun s hs i hi m hm =>
      Layout.msgs_getD hi hs hm fun s'' hs'' m'' hm'' => hdiff i hi s'' hs'' s hs m'' m hm'' hm

def VStarAcceptsR (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (shape : Form → Prop) : Prop :=
  ∀ (x : Msg) (s : CompiledStatement), D.parse x = some s → shape s.form →
    ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin), (run H P D.V x ω r).out = .acc →
      ∀ ws, OuterHolds DO (runVstar H D I P x ω r) (D.nS x) ws → (∀ e ∈ D.calls x ws, H e.1 = e.2) →
        (D.innerView H x r ws).out = .acc ∨ CallsColl H D x (runVstar H D I P x ω r) ws

omit [Fintype Coin] [Nonempty Coin] in
theorem vstarAccepts_of_parts {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {IV : Verifier Coin} {L : Layout} {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {opening : Opening} {cmt : Msg → ℕ → Msg → Msg}
    (hR : ReadsThrough H D IV L.msgs L.calls) (hVL : VstarLen D) (hC : ∀ s, PartChecks H D DO L shape shows chk s)
    (hO : ∀ s, PartOpens H D DO L shape shows opening cmt s) (hS : Staged H D I L shape shows cmt)
    (hD : Decomp H D IV shape chk) : VStarAcceptsR H D I DO shape := by
  intro x st hst hf Ω P ω r hacc ws ho htrue
  rcases agree_or_coll hR hVL hO hS hst hf hacc ho htrue with hag | hcoll
  · obtain ⟨hlen, hchk⟩ := ho
    have hxs : ∀ s < ws.length, s < (runVstar H D I P x ω r).length := fun s hs =>
      lt_of_lt_of_le (hlen ▸ hs) (hVL x _)
    by_cases hcol : ∃ s, ∃ hs : s < ws.length, (findColl (L.calls x s ws[s])).isSome ∨
        SetupColl H D ((runVstar H D I P x ω r).getD s []) (L.calls x s ws[s])
    · obtain ⟨s, hs, h⟩ := hcol
      exact Or.inr (callsColl_of_part hR hs (hxs s hs) htrue h)
    left
    rw [hR.1 x r ws]
    refine hD x st hst hf (coinSeq r) (L.msgs x ws) fun s hs => ?_
    have hs' : s < ws.length := hlen ▸ hs
    rcases hC s x st hst hf _ ws[s] (hchk s hs')
        (fun e he => htrue e (by rw [hR.2 x ws]; exact mem_calls_of_part hs' he)) with h | h
    · exact h (coinSeq r) (hS x st hst hf Ω P ω r hacc s hs).1 (L.msgs x ws) fun i hi m hm => hag s hs' i hi m hm
    · exact absurd ⟨s, hs', h⟩ hcol
  · exact Or.inr hcoll

omit [Fintype Coin] [Nonempty Coin] in
theorem vstarAccepts_dense {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {IV : Verifier Coin} {L : Layout} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {opening : Opening} {cmt : Msg → ℕ → Msg → Msg}
    (hR : ReadsThrough H D IV L.msgs L.calls) (hVL : VstarLen D)
    (hC : ∀ s, PartChecks H D DO L Form.Dense shows chk s) (hO : ∀ s, PartOpens H D DO L Form.Dense shows opening cmt s)
    (hS : Staged H D I L Form.Dense shows cmt) (hD : Decomp H D IV Form.Dense chk) :
    VStarAcceptsR H D I DO Form.Dense :=
  vstarAccepts_of_parts hR hVL hC hO hS hD

omit [Fintype Coin] [Nonempty Coin] in
theorem vstarAccepts_holo {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {IV : Verifier Coin} {L : Layout} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {opening : Opening} {cmt : Msg → ℕ → Msg → Msg}
    (hR : ReadsThrough H D IV L.msgs L.calls) (hVL : VstarLen D)
    (hC : ∀ s, PartChecks H D DO L Form.Holo shows chk s) (hO : ∀ s, PartOpens H D DO L Form.Holo shows opening cmt s)
    (hS : Staged H D I L Form.Holo shows cmt) (hD : Decomp H D IV Form.Holo chk) :
    VStarAcceptsR H D I DO Form.Holo :=
  vstarAccepts_of_parts hR hVL hC hO hS hD

omit [Fintype Coin] [Nonempty Coin] in
theorem vstarAccepts_sparse {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {IV : Verifier Coin} {L : Layout} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {opening : Opening} {cmt : Msg → ℕ → Msg → Msg}
    (hR : ReadsThrough H D IV L.msgs L.calls) (hVL : VstarLen D)
    (hC : ∀ s, PartChecks H D DO L Form.Sparse shows chk s)
    (hO : ∀ s, PartOpens H D DO L Form.Sparse shows opening cmt s)
    (hS : Staged H D I L Form.Sparse shows cmt) (hD : Decomp H D IV Form.Sparse chk) :
    VStarAcceptsR H D I DO Form.Sparse :=
  vstarAccepts_of_parts hR hVL hC hO hS hD

def InnerAcceptsR (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
    (D.innerView H x r ws).out = .acc ∨ CallsColl H D x (runVstar H D I P x ω r) ws

theorem innerAcceptsR_of_vstar {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} (hRef : Refuses H D) (hT : CallsTrue H D I DO)
    (hD : VStarAcceptsR H D I DO Form.Dense) (hH : VStarAcceptsR H D I DO Form.Holo)
    (hS : VStarAcceptsR H D I DO Form.Sparse) : InnerAcceptsR H D I DO := by
  intro decO x Ω P ω r c ws hg
  obtain ⟨s, hs, -⟩ := hRef x Ω P ω r hg.1
  have ho := eoGives_outerHolds hg
  have ht := hT decO x Ω P ω r c ws hg
  rcases Form.shape_cases s.form with hf | hf | hf
  · exact hD x s hs hf Ω P ω r hg.1 ws ho ht
  · exact hH x s hs hf Ω P ω r hg.1 ws ho ht
  · exact hS x s hs hf Ω P ω r hg.1 ws ho ht

theorem innerAcceptsR_of_parts {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {IV : Verifier Coin} {L : Layout} {opening : Opening}
    {showsD showsH showsS : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {chkD chkH chkS : ℕ → Msg → (ℕ → Coin) → List Msg → Prop}
    {cmtD cmtH cmtS : Msg → ℕ → Msg → Msg} (hRef : Refuses H D) (hT : CallsTrue H D I DO)
    (hR : ReadsThrough H D IV L.msgs L.calls) (hVL : VstarLen D)
    (hCD : ∀ s, PartChecks H D DO L Form.Dense showsD chkD s)
    (hOD : ∀ s, PartOpens H D DO L Form.Dense showsD opening cmtD s)
    (hSD : Staged H D I L Form.Dense showsD cmtD) (hDD : Decomp H D IV Form.Dense chkD)
    (hCH : ∀ s, PartChecks H D DO L Form.Holo showsH chkH s)
    (hOH : ∀ s, PartOpens H D DO L Form.Holo showsH opening cmtH s)
    (hSH : Staged H D I L Form.Holo showsH cmtH) (hDH : Decomp H D IV Form.Holo chkH)
    (hCS : ∀ s, PartChecks H D DO L Form.Sparse showsS chkS s)
    (hOS : ∀ s, PartOpens H D DO L Form.Sparse showsS opening cmtS s) (hSS : Staged H D I L Form.Sparse showsS cmtS)
    (hDS : Decomp H D IV Form.Sparse chkS) : InnerAcceptsR H D I DO :=
  innerAcceptsR_of_vstar hRef hT (vstarAccepts_dense hR hVL hCD hOD hSD hDD)
    (vstarAccepts_holo hR hVL hCH hOH hSH hDH) (vstarAccepts_sparse hR hVL hCS hOS hSS hDS)

/-- Fixed and public rounds need no opening to bind: two runs that share their coins before a round agree on the statement's message and on the run's own there. -/
def CallsOpenPlayed (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (opening : Opening) (fixed : Msg → ℕ → Option Msg) (pub : Msg → ℕ → Bool) : Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
    ∀ j < (D.innerView H x r ws).msgs.length,
      (∀ m, fixed x j = some m → ((D.innerView H x r ws).msgs.map Prod.fst).getD j [] = m) ∧
      (pub x j = true →
        ((D.innerView H x r ws).msgs.map Prod.fst).getD j [] = ((run H P D.V x ω r).msgs.map Prod.fst).getD j []) ∧
      (fixed x j = none → pub x j = false → (∃ cs, cs ⊆ D.calls x ws ∧
        opening.opens (((run H P D.V x ω r).msgs.map Prod.fst).getD j [])
          (((D.innerView H x r ws).msgs.map Prod.fst).getD j []) cs) ∨ CallsColl H D x (runVstar H D I P x ω r) ws)

/-- No part holds the link message, a public round, so readers of the witnesses alone don't read it. -/
def ReadsPub (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) (L : Layout) :
    Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
    ∀ j < (D.innerView H x r ws).msgs.length, L.pubAt x j = true →
      ((D.innerView H x r ws).msgs.map Prod.fst).getD j [] = ((run H P D.V x ω r).msgs.map Prod.fst).getD j []

/-- No part gives it: Flock's last message holds the inner session's record, which V\*'s statements stage in the clear, and its proof files, which V\*'s parts hold across them. -/
def FinalOpens (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (opening : Opening) : Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
    ∀ j < (D.innerView H x r ws).msgs.length, I.V.k (xI D x) ≤ j + 1 → (∃ cs, cs ⊆ D.calls x ws ∧
      opening.opens (((run H P D.V x ω r).msgs.map Prod.fst).getD j [])
        (((D.innerView H x r ws).msgs.map Prod.fst).getD j []) cs) ∨ CallsColl H D x (runVstar H D I P x ω r) ws

def HeldPub (D : Deploy Coin) (I : Inner Coin W) (L : Layout) (shape : Form → Prop) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form →
    I.V.k (xI D x) ≤ L.rounds x ∧ ∀ i, i + 1 < I.V.k (xI D x) →
      L.pubAt x i = true ∨ (L.fixedAt x i).isSome ∨ ∃ s < D.nS x, (L.msgAt x s i).isSome

theorem opensPlayed_of_parts {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {L : Layout} {shape : Form → Prop} {opening : Opening} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {cmt : Msg → ℕ → Msg → Msg} (hR : ReadsThrough H D I.V L.msgs L.calls) (hVL : VstarLen D)
    (hH : HeldPub D I L shape) (hO : ∀ s, PartOpens H D DO L shape shows opening cmt s)
    (hS : Staged H D I L shape shows cmt) (hPub : ReadsPub H D I DO L) (hFin : FinalOpens H D I DO opening)
    {decO : List (ℕ × Col) → List T} {x : Msg} {Ω : Type} {P : Prover Ω} {ω : Ω} {r : Fin (D.V.k x) → Coin}
    {c : (EO D I DO decO).C x} {ws : List Flock.Wit} (hg : EOGives H D I DO decO P x ω r c ws)
    (htrue : ∀ e ∈ D.calls x ws, H e.1 = e.2) {st : CompiledStatement} (hst : D.parse x = some st) (hf : shape st.form) :
    ∀ j < (D.innerView H x r ws).msgs.length,
      (∀ m, L.fixedAt x j = some m → ((D.innerView H x r ws).msgs.map Prod.fst).getD j [] = m) ∧
      (L.pubAt x j = true →
        ((D.innerView H x r ws).msgs.map Prod.fst).getD j [] = ((run H P D.V x ω r).msgs.map Prod.fst).getD j []) ∧
      (L.fixedAt x j = none → L.pubAt x j = false → (∃ cs, cs ⊆ D.calls x ws ∧
        opening.opens (((run H P D.V x ω r).msgs.map Prod.fst).getD j [])
          (((D.innerView H x r ws).msgs.map Prod.fst).getD j []) cs) ∨ CallsColl H D x (runVstar H D I P x ω r) ws) := by
  intro j hj
  have hplay : j < I.V.k (xI D x) := by
    have h := hj
    rw [hR.1 x r ws] at h
    exact lt_of_lt_of_le h (run_length_le H _ I.V _ () _)
  have hview : ((D.innerView H x r ws).msgs.map Prod.fst).getD j [] = (L.msgs x ws).getD j [] := by
    have h := hj
    rw [hR.1 x r ws] at h ⊢
    simp only [Transcript.seen] at h ⊢
    exact run_replay_fst H I.V (xI D x) (L.msgs x ws) _ h
  have hjr : j < L.rounds x := lt_of_lt_of_le hplay (hH x st hst hf).1
  refine ⟨fun m hm => hview.trans (Layout.msgs_getD_fixed hjr hm), hPub decO x Ω P ω r c ws hg j hj,
    fun hfx hpb => ?_⟩
  by_cases hlast : I.V.k (xI D x) ≤ j + 1
  · exact hFin decO x Ω P ω r c ws hg j hj hlast
  obtain ⟨hlen, hchk⟩ := eoGives_outerHolds hg
  have hxs : ∀ s < ws.length, s < (runVstar H D I P x ω r).length := fun s hs =>
    lt_of_lt_of_le (hlen ▸ hs) (hVL x _)
  rw [hview]
  rcases (hH x st hst hf).2 j (by omega) with hp | hp | ⟨s, hs, hh⟩
  · rw [hpb] at hp
    exact absurd hp Bool.false_ne_true
  · rw [hfx] at hp
    exact absurd hp (by simp)
  obtain ⟨s₀, hs₀, hm⟩ := Layout.msgs_getD_held hjr (hlen ▸ hs) hfx hh
  rcases hO s₀ x st hst hf _ ws[s₀] (hchk s₀ hs₀)
      (fun e he => htrue e (by rw [hR.2 x ws]; exact mem_calls_of_part hs₀ he)) with ho | hc
  · obtain ⟨cs, hcs, ho⟩ := ho _ (hS x st hst hf Ω P ω r hg.1 s₀ (hlen ▸ hs₀)).1 j hjr _ hm
    rw [(hS x st hst hf Ω P ω r hg.1 s₀ (hlen ▸ hs₀)).2 j hjr] at ho
    refine Or.inl ⟨cs, fun e he => ?_, ho⟩
    rw [hR.2 x ws]
    exact mem_calls_of_part hs₀ (hcs he)
  · exact Or.inr (callsColl_of_part hR hs₀ (hxs s₀ hs₀) htrue hc)

theorem callsOpenPlayed_of_parts {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {L : Layout} {opening : Opening} {showsD showsH showsS : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {cmtD cmtH cmtS : Msg → ℕ → Msg → Msg} (hRef : Refuses H D) (hT : CallsTrue H D I DO)
    (hR : ReadsThrough H D I.V L.msgs L.calls) (hVL : VstarLen D) (hPub : ReadsPub H D I DO L)
    (hFin : FinalOpens H D I DO opening)
    (hHD : HeldPub D I L Form.Dense) (hOD : ∀ s, PartOpens H D DO L Form.Dense showsD opening cmtD s)
    (hSD : Staged H D I L Form.Dense showsD cmtD)
    (hHH : HeldPub D I L Form.Holo) (hOH : ∀ s, PartOpens H D DO L Form.Holo showsH opening cmtH s)
    (hSH : Staged H D I L Form.Holo showsH cmtH)
    (hHS : HeldPub D I L Form.Sparse) (hOS : ∀ s, PartOpens H D DO L Form.Sparse showsS opening cmtS s)
    (hSS : Staged H D I L Form.Sparse showsS cmtS) : CallsOpenPlayed H D I DO opening L.fixedAt L.pubAt := by
  intro decO x Ω P ω r c ws hg
  obtain ⟨st, hst, -⟩ := hRef x Ω P ω r hg.1
  have ht := hT decO x Ω P ω r c ws hg
  rcases Form.shape_cases st.form with hf | hf | hf
  · exact opensPlayed_of_parts hR hVL hHD hOD hSD hPub hFin hg ht hst hf
  · exact opensPlayed_of_parts hR hVL hHH hOH hSH hPub hFin hg ht hst hf
  · exact opensPlayed_of_parts hR hVL hHS hOS hSS hPub hFin hg ht hst hf

def regColl (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (j : ℕ)
    (t : Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x × (EO D I DO decO).C x) :
    Prop :=
  ∃ u v, findCollIn (D.calls x (((EO D I DO decO).eval H P x t.1 t.2.1 t.2.2.2.1).1.getD []) ++
      D.calls x (((EO D I DO decO).eval H P x t.1 (fork j t.2.1 t.2.2.1) t.2.2.2.2).1.getD []))
    (setupCalls H D (runVstar H D I P x t.1 t.2.1) ++
      setupCalls H D (runVstar H D I P x t.1 (fork j t.2.1 t.2.2.1))) = some (u, v) ∧ Collision H u v

def forkReg (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (j : ℕ)
    (t : Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x × (EO D I DO decO).C x) :
    Prop :=
  forkColl H D I DO decO P x j t ∨ regColl H D I DO decO P x j t

theorem PrBr_nonneg (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) :
    0 ≤ PrBr H D I DO decO P x := by
  unfold PrBr; split
  · exact Pr_nonneg _
  · exact le_rfl

theorem PrBr_eq_avg (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg)
    (hk : 0 < D.V.k x) :
    PrBr H D I DO decO P x = (∑ j : Fin (D.V.k x), Pr (regColl H D I DO decO P x j)) / D.V.k x := by
  simp only [PrBr, hk, ↓reduceDIte, Finder.Pr]
  let e : Fin (D.V.k x) × Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x ×
      (EO D I DO decO).C x ≃ (Br H D I DO decO P x hk).S :=
    { toFun := fun t => ((t.1, t.2.1, t.2.2.1, t.2.2.2.1), t.2.2.2.2.1, t.2.2.2.2.2)
      invFun := fun s => (s.1.1, s.1.2.1, s.1.2.2.1, s.1.2.2.2, s.2.1, s.2.2)
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  rw [← Pr_equiv e, Pr_guess]
  rfl

theorem sum_forkReg_le (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) :
    ∑ j : Fin (D.V.k x), Pr (forkReg H D I DO decO P x j) ≤
      D.V.k x * (PrBe H D I DO decO P x + PrBr H D I DO decO P x) := by
  by_cases hk : 0 < D.V.k x
  · have hk' : (D.V.k x : ℝ) ≠ 0 := by exact_mod_cast hk.ne'
    rw [mul_add, PrBe_eq_avg H D I DO decO P x hk, PrBr_eq_avg H D I DO decO P x hk, mul_div_cancel₀ _ hk',
      mul_div_cancel₀ _ hk', ← Finset.sum_add_distrib]
    exact Finset.sum_le_sum fun j _ => Pr_or_le _ _
  · have h0 : D.V.k x = 0 := Nat.eq_zero_of_not_pos hk
    have : IsEmpty (Fin (D.V.k x)) := by rw [h0]; infer_instance
    have h0' : (D.V.k x : ℝ) = 0 := by exact_mod_cast h0
    rw [Finset.univ_eq_empty, Finset.sum_empty, h0', zero_mul]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem coll_or_reg {H : Msg → Msg} {D : Deploy Coin} {c₁ c₂ : List (Msg × Msg)} {xs₁ xs₂ : List Msg}
    (htrue : ∀ e ∈ c₁ ++ c₂, H e.1 = e.2)
    (h : (findColl (c₁ ++ c₂)).isSome ∨ (findCollIn c₁ (setupCalls H D xs₁)).isSome = true ∨
      (findCollIn c₂ (setupCalls H D xs₂)).isSome = true) :
    (∃ u v, findColl (c₁ ++ c₂) = some (u, v) ∧ Collision H u v) ∨
      ∃ u v, findCollIn (c₁ ++ c₂) (setupCalls H D xs₁ ++ setupCalls H D xs₂) = some (u, v) ∧ Collision H u v := by
  have hall : ∀ p ∈ (setupCalls H D xs₁ ++ setupCalls H D xs₂) ++ (c₁ ++ c₂), H p.1 = p.2 := fun p hp =>
    (List.mem_append.1 hp).elim
      (fun hp => (List.mem_append.1 hp).elim (setupCalls_true H D xs₁ p) (setupCalls_true H D xs₂ p)) (htrue p)
  have hreg : (findCollIn (c₁ ++ c₂) (setupCalls H D xs₁ ++ setupCalls H D xs₂)).isSome = true →
      ∃ u v, findCollIn (c₁ ++ c₂) (setupCalls H D xs₁ ++ setupCalls H D xs₂) = some (u, v) ∧ Collision H u v :=
    fun h => by
      obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 h
      exact ⟨u, v, huv, findCollIn_spec H _ _ hall u v huv⟩
  rcases h with h | h | h
  · obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 h
    exact Or.inl ⟨u, v, huv, findColl_spec H _ htrue u v huv⟩
  · exact Or.inr (hreg (findCollIn_isSome_of_sub (fun e he => List.mem_append_left _ he)
      (fun e he => List.mem_append_right _ (List.mem_append_left _ he)) h))
  · exact Or.inr (hreg (findCollIn_isSome_of_sub (fun e he => List.mem_append_right _ he)
      (fun e he => List.mem_append_right _ (List.mem_append_right _ he)) h))

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem collOrReg_of_callsColl {H : Msg → Msg} {D : Deploy Coin} {x : Msg} {ws₁ ws₂ : List Flock.Wit}
    {xs₁ xs₂ : List Msg} (htrue : ∀ e ∈ D.calls x ws₁ ++ D.calls x ws₂, H e.1 = e.2)
    (h : CallsColl H D x xs₁ ws₁ ∨ CallsColl H D x xs₂ ws₂) :
    (findColl (D.calls x ws₁ ++ D.calls x ws₂)).isSome ∨
      (findCollIn (D.calls x ws₁) (setupCalls H D xs₁)).isSome = true ∨
      (findCollIn (D.calls x ws₂) (setupCalls H D xs₂)).isSome = true := by
  rcases h with (⟨u, v, huv, -⟩ | h) | (⟨u, v, huv, -⟩ | h)
  · exact Or.inl (findColl_isSome_of_sub H htrue (fun p hp => List.mem_append_left _ hp)
      (Option.isSome_iff_exists.2 ⟨(u, v), huv⟩))
  · exact Or.inr (Or.inl h)
  · exact Or.inl (findColl_isSome_of_sub H htrue (fun p hp => List.mem_append_right _ hp)
      (Option.isSome_iff_exists.2 ⟨(u, v), huv⟩))
  · exact Or.inr (Or.inr h)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem coll_or_reg_split {H : Msg → Msg} {D : Deploy Coin} {c₁ c₂ : List (Msg × Msg)} {xs₁ xs₂ : List Msg}
    {l : List (Msg × Msg)} (htrue : ∀ e ∈ c₁ ++ c₂, H e.1 = e.2)
    (hsub : ∀ e ∈ l, e ∈ c₁ ++ c₂ ∨ e ∈ setupCalls H D xs₁ ++ setupCalls H D xs₂) (h : (findColl l).isSome) :
    (∃ u v, findColl (c₁ ++ c₂) = some (u, v) ∧ Collision H u v) ∨
      ∃ u v, findCollIn (c₁ ++ c₂) (setupCalls H D xs₁ ++ setupCalls H D xs₂) = some (u, v) ∧ Collision H u v := by
  have hS : ∀ p ∈ setupCalls H D xs₁ ++ setupCalls H D xs₂, H p.1 = p.2 := fun p hp =>
    (List.mem_append.1 hp).elim (setupCalls_true H D xs₁ p) (setupCalls_true H D xs₂ p)
  have hl : ∀ p ∈ l, H p.1 = p.2 := fun p hp => (hsub p hp).elim (htrue p) (hS p)
  have hall : ∀ q ∈ (setupCalls H D xs₁ ++ setupCalls H D xs₂) ++ (c₁ ++ c₂), H q.1 = q.2 := fun q hq =>
    (List.mem_append.1 hq).elim (hS q) (htrue q)
  have hreg : (findCollIn (c₁ ++ c₂) (setupCalls H D xs₁ ++ setupCalls H D xs₂)).isSome = true →
      ∃ u v, findCollIn (c₁ ++ c₂) (setupCalls H D xs₁ ++ setupCalls H D xs₂) = some (u, v) ∧ Collision H u v :=
    fun h => by
      obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 h
      exact ⟨u, v, huv, findCollIn_spec H _ _ hall u v huv⟩
  obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 h
  have hc := findColl_spec H l hl u v huv
  obtain ⟨hu, hv⟩ := findColl_mem l u v huv
  obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hu
  obtain ⟨p', hp', rfl⟩ := List.mem_map.1 hv
  have h2 : p'.2 = p.2 := by rw [← hl p' hp', ← hl p hp]; exact hc.2.symm
  rcases hsub p hp with hpc | hps
  · rcases hsub p' hp' with hpc' | hps'
    · exact Or.inl (findColl_of_coll H htrue (List.mem_map_of_mem hpc) (List.mem_map_of_mem hpc') hc)
    · exact Or.inr (hreg (findCollIn_isSome hps' (List.mem_append_right _ hpc) hc.1 h2.symm))
  · refine Or.inr (hreg (findCollIn_isSome hps ?_ (Ne.symm hc.1) h2))
    rcases hsub p' hp' with hpc' | hps'
    · exact List.mem_append_right _ hpc'
    · exact List.mem_append_left _ hps'

theorem forkColl_of_openingPlayed {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {opening : Opening} {fixed : Msg → ℕ → Option Msg} {pub : Msg → ℕ → Bool}
    (hV : InnerViewRun H D I DO)
    (hO : CallsOpenPlayed H D I DO opening fixed pub) (hT : CallsTrue H D I DO) (decO : List (ℕ × Col) → List T)
    (x : Msg)
    {Ω : Type} (P : Prover Ω) {j : ℕ}
    (t : Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x × (EO D I DO decO).C x)
    (h : OpensDiffer H D I DO decO P x j t) : forkReg H D I DO decO P x j t := by
  classical
  obtain ⟨ws, ws', hg, hg', hd⟩ := h
  have hex : ∃ i, ((D.innerView H x t.2.1 ws).msgs.map Prod.fst).getD i [] ≠
      ((D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.map Prod.fst).getD i [] := ⟨j, hd⟩
  have hi := Nat.find_spec hex
  have hmin : ∀ i < Nat.find hex, ((D.innerView H x t.2.1 ws).msgs.map Prod.fst).getD i [] =
      ((D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.map Prod.fst).getD i [] :=
    fun i h => not_not.1 (Nat.find_min hex h)
  have hij : Nat.find hex ≤ j := Nat.find_min' hex hd
  generalize Nat.find hex = i at hi hmin hij
  have hc : ∀ i' < i, coinSeq (innerCoins D I x t.2.1) i' =
      coinSeq (innerCoins D I x (fork j t.2.1 t.2.2.1)) i' := fun i' hi' => by
    unfold coinSeq innerCoins coinSeq
    split_ifs with h1 h2
    · exact fork_agree j t.2.1 t.2.2.1 ⟨i', h2⟩ (by simp only; omega)
    · rfl
    · rfl
  have hv := congrArg Seen.msgs (hV decO x Ω P t.1 t.2.1 _ ws hg)
  have hv' := congrArg Seen.msgs (hV decO x Ω P t.1 (fork j t.2.1 t.2.2.1) _ ws' hg')
  simp only [innerRun, Transcript.seen] at hv hv'
  have hboth : i < (D.innerView H x t.2.1 ws).msgs.length ∧
      i < (D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.length := by
    by_cases h1 : i < (D.innerView H x t.2.1 ws).msgs.length
    · refine ⟨h1, ?_⟩
      rw [hv']
      exact run_replay_plays H I.V (xI D x) _ _ _ _ _ hmin hc (hv ▸ h1)
    · by_cases h2 : i < (D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.length
      · refine absurd ?_ h1
        rw [hv]
        exact run_replay_plays H I.V (xI D x) _ _ _ _ _ (fun i h => (hmin i h).symm) (fun i h => (hc i h).symm)
          (hv' ▸ h2)
      · refine absurd ?_ hi
        push Not at h1 h2
        rw [List.getD_eq_default _ _ (by simpa using h1), List.getD_eq_default _ _ (by simpa using h2)]
  have hall : ∀ e ∈ D.calls x ws ++ D.calls x ws', H e.1 = e.2 := fun e he =>
    (List.mem_append.1 he).elim (hT decO x Ω P t.1 t.2.1 _ ws hg e) (hT decO x Ω P t.1 _ _ ws' hg' e)
  unfold forkReg forkColl regColl
  rw [hg.2, hg'.2]
  have hrun := run_fst_getD H P D.V x t.1 i fun i' hi' => fork_agree j t.2.1 t.2.2.1 i' (by omega)
  obtain ⟨hfx₁, hpb₁, hop₁⟩ := hO decO x Ω P t.1 t.2.1 _ ws hg _ hboth.1
  obtain ⟨hfx₂, hpb₂, hop₂⟩ := hO decO x Ω P t.1 (fork j t.2.1 t.2.2.1) _ ws' hg' _ hboth.2
  cases hfi : fixed x i with
  | some m => exact absurd ((hfx₁ m hfi).trans (hfx₂ m hfi).symm) hi
  | none =>
  cases hpi : pub x i with
  | true => exact absurd ((hpb₁ hpi).trans (hrun.trans (hpb₂ hpi).symm)) hi
  | false =>
  rcases hop₁ hfi hpi with ⟨cs, hcs, ho⟩ | hcc
  · rcases hop₂ hfi hpi with ⟨cs', hcs', ho'⟩ | hcc'
    · rw [hrun] at ho
      have hsub : cs ++ cs' ⊆ D.calls x ws ++ D.calls x ws' := fun e he =>
        (List.mem_append.1 he).elim (fun he => List.mem_append_left _ (hcs he))
          fun he => List.mem_append_right _ (hcs' he)
      obtain ⟨q, hq, q', hq', hcol⟩ := opening.bind H _ _ _ cs cs' ho ho' hi fun e he => hall e (hsub he)
      exact Or.inl (findColl_of_coll H hall (List.map_subset _ hsub hq) (List.map_subset _ hsub hq') hcol)
    · exact coll_or_reg hall (collOrReg_of_callsColl hall (Or.inr hcc'))
  · exact coll_or_reg hall (collOrReg_of_callsColl hall (Or.inl hcc))

theorem sum_opensDiffer_le_played {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {opening : Opening} {fixed : Msg → ℕ → Option Msg} {pub : Msg → ℕ → Bool}
    (hV : InnerViewRun H D I DO)
    (hO : CallsOpenPlayed H D I DO opening fixed pub) (hT : CallsTrue H D I DO) (hR : InnerRounds D I)
    (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    ∑ j : Fin (I.V.k (xI D x)), Pr (OpensDiffer H D I DO decO P x j) ≤
      D.V.k x * (PrBe H D I DO decO P x + PrBr H D I DO decO P x) := by
  have hq : ∀ j, 0 ≤ Pr (forkReg H D I DO decO P x j) := fun j => Pr_nonneg _
  calc ∑ j : Fin (I.V.k (xI D x)), Pr (OpensDiffer H D I DO decO P x j)
      ≤ ∑ j : Fin (I.V.k (xI D x)), Pr (forkReg H D I DO decO P x j) :=
        Finset.sum_le_sum fun j _ => Pr_mono fun t => forkColl_of_openingPlayed hV hO hT decO x P t
    _ ≤ ∑ j : Fin (D.V.k x), Pr (forkReg H D I DO decO P x j) := by
        rw [Fin.sum_univ_eq_sum_range (fun j => Pr (forkReg H D I DO decO P x j)),
          Fin.sum_univ_eq_sum_range (fun j => Pr (forkReg H D I DO decO P x j))]
        exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono (hR x)) fun j _ _ => hq j
    _ ≤ _ := sum_forkReg_le H D I DO decO P x

def CallsLogColl (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
    (∀ e ∈ (innerRun H D I x r ws).vLog, e ∈ D.calls x ws ∨ e ∈ setupCalls H D (runVstar H D I P x ω r)) ∨
      CallsColl H D x (runVstar H D I P x ω r) ws

def LogDecomp (H : Msg → Msg) (D : Deploy Coin) (IV : Verifier Coin) (shape : Form → Prop)
    (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (c : ℕ → Coin) (ms : List Msg),
    ∀ e ∈ (run H (replay ms) IV (xI D x) () fun i => c i.val).vLog, ∃ s < D.nS x, e ∈ logs s x c ms

def LogDecompPub (H : Msg → Msg) (D : Deploy Coin) (IV : Verifier Coin) (shape : Form → Prop)
    (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)) (pub : Msg → List (Msg × Msg)) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (c : ℕ → Coin) (ms : List Msg),
    ∀ e ∈ (run H (replay ms) IV (xI D x) () fun i => c i.val).vLog, (∃ s < D.nS x, e ∈ logs s x c ms) ∨ e ∈ pub x

omit [Fintype Coin] [Nonempty Coin] in
theorem logDecompPub_of_logDecomp {H : Msg → Msg} {D : Deploy Coin} {IV : Verifier Coin} {shape : Form → Prop}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} (h : LogDecomp H D IV shape logs) :
    LogDecompPub H D IV shape logs fun _ => [] :=
  fun x st hst hf c ms e he => Or.inl (h x st hst hf c ms e he)

def PubReg (H : Msg → Msg) (D : Deploy Coin) (shape : Form → Prop) (pub : Msg → List (Msg × Msg)) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ ps : List (Msg × Out), ∀ e ∈ pub x,
    e ∈ setupCalls H D (D.vstar x ps)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem pubReg_nil (H : Msg → Msg) (D : Deploy Coin) (shape : Form → Prop) : PubReg H D shape fun _ => [] :=
  fun _ _ _ _ _ _ he => by simp at he

def PartLogs (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg))
    (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → (∀ e ∈ L.calls x s w, H e.1 = e.2) →
    (∀ (c : ℕ → Coin), shows s x xO c → ∀ ms : List Msg,
      (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) → ∀ e ∈ logs s x c ms, e ∈ L.calls x s w) ∨
      (findColl (L.calls x s w)).isSome ∨ SetupColl H D xO (L.calls x s w)

theorem logColl_of_parts {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {L : Layout} {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {pub : Msg → List (Msg × Msg)} {opening : Opening}
    {cmt : Msg → ℕ → Msg → Msg} (hR : ReadsThrough H D I.V L.msgs L.calls) (hVL : VstarLen D)
    (hL : ∀ s, PartLogs H D DO L shape shows logs s) (hO : ∀ s, PartOpens H D DO L shape shows opening cmt s)
    (hS : Staged H D I L shape shows cmt) (hLD : LogDecompPub H D I.V shape logs pub) (hPR : PubReg H D shape pub)
    {decO : List (ℕ × Col) → List T} {x : Msg} {Ω : Type} {P : Prover Ω}
    {ω : Ω} {r : Fin (D.V.k x) → Coin} {c : (EO D I DO decO).C x} {ws : List Flock.Wit}
    (hg : EOGives H D I DO decO P x ω r c ws) (htrue : ∀ e ∈ D.calls x ws, H e.1 = e.2) {st : CompiledStatement}
    (hst : D.parse x = some st) (hf : shape st.form) :
    (∀ e ∈ (innerRun H D I x r ws).vLog, e ∈ D.calls x ws ∨ e ∈ setupCalls H D (runVstar H D I P x ω r)) ∨
      CallsColl H D x (runVstar H D I P x ω r) ws := by
  have ho := eoGives_outerHolds hg
  rcases agree_or_coll hR hVL hO hS hst hf hg.1 ho htrue with hag | hcoll
  · obtain ⟨hlen, hchk⟩ := ho
    have hxs : ∀ s < ws.length, s < (runVstar H D I P x ω r).length := fun s hs =>
      lt_of_lt_of_le (hlen ▸ hs) (hVL x _)
    have hrun : innerRun H D I x r ws =
        run H (replay (L.msgs x ws)) I.V (xI D x) () fun i => coinSeq r i.val := by
      unfold innerRun
      rw [hR.1 x r ws]
      exact run_replay_self H I.V (xI D x) (L.msgs x ws) _
    by_cases hcol : ∃ s, ∃ hs : s < ws.length, (findColl (L.calls x s ws[s])).isSome ∨
        SetupColl H D ((runVstar H D I P x ω r).getD s []) (L.calls x s ws[s])
    · obtain ⟨s, hs, h⟩ := hcol
      exact Or.inr (callsColl_of_part hR hs (hxs s hs) htrue h)
    refine Or.inl fun e he => ?_
    rw [hrun] at he
    rcases hLD x st hst hf (coinSeq r) (L.msgs x ws) e he with ⟨s, hs, hes⟩ | hpub
    swap
    · exact Or.inr (hPR x st hst hf _ e hpub)
    left
    have hs' : s < ws.length := hlen ▸ hs
    rw [hR.2 x ws]
    rcases hL s x st hst hf _ ws[s] (hchk s hs')
        (fun e he => htrue e (by rw [hR.2 x ws]; exact mem_calls_of_part hs' he)) with h | h
    · exact mem_calls_of_part hs' (h (coinSeq r) (hS x st hst hf Ω P ω r hg.1 s hs).1 (L.msgs x ws)
        (fun i hi m hm => hag s hs' i hi m hm) e hes)
    · exact absurd ⟨s, hs', h⟩ hcol
  · exact Or.inr hcoll

theorem callsLogColl_of_parts {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {L : Layout} {opening : Opening} {showsD showsH showsS : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {logsD logsH logsS : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    {pubD pubH pubS : Msg → List (Msg × Msg)}
    {cmtD cmtH cmtS : Msg → ℕ → Msg → Msg} (hRef : Refuses H D) (hT : CallsTrue H D I DO)
    (hR : ReadsThrough H D I.V L.msgs L.calls) (hVL : VstarLen D)
    (hLD : ∀ s, PartLogs H D DO L Form.Dense showsD logsD s)
    (hOD : ∀ s, PartOpens H D DO L Form.Dense showsD opening cmtD s)
    (hSD : Staged H D I L Form.Dense showsD cmtD) (hDD : LogDecompPub H D I.V Form.Dense logsD pubD)
    (hPD : PubReg H D Form.Dense pubD)
    (hLH : ∀ s, PartLogs H D DO L Form.Holo showsH logsH s)
    (hOH : ∀ s, PartOpens H D DO L Form.Holo showsH opening cmtH s)
    (hSH : Staged H D I L Form.Holo showsH cmtH) (hDH : LogDecompPub H D I.V Form.Holo logsH pubH)
    (hPH : PubReg H D Form.Holo pubH)
    (hLS : ∀ s, PartLogs H D DO L Form.Sparse showsS logsS s)
    (hOS : ∀ s, PartOpens H D DO L Form.Sparse showsS opening cmtS s) (hSS : Staged H D I L Form.Sparse showsS cmtS)
    (hDS : LogDecompPub H D I.V Form.Sparse logsS pubS) (hPS : PubReg H D Form.Sparse pubS) :
    CallsLogColl H D I DO := by
  intro decO x Ω P ω r c ws hg
  obtain ⟨st, hst, -⟩ := hRef x Ω P ω r hg.1
  have ht := hT decO x Ω P ω r c ws hg
  rcases Form.shape_cases st.form with hf | hf | hf
  · exact logColl_of_parts hR hVL hLD hOD hSD hDD hPD hg ht hst hf
  · exact logColl_of_parts hR hVL hLH hOH hSH hDH hPH hg ht hst hf
  · exact logColl_of_parts hR hVL hLS hOS hSS hDS hPS hg ht hst hf

theorem calls_true_of {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {decO : List (ℕ × Col) → List T} {Ω : Type} {P : Prover Ω} {x : Msg} (hT : CallsTrue H D I DO)
    (hN0 : CallsNil H D) (ω : Ω) (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) :
    ∀ e ∈ D.calls x ((eoOut H D I DO decO P x ω d).getD []), H e.1 = e.2 := by
  rcases hws : eoOut H D I DO decO P x ω d with _ | ws
  · exact hN0 x
  · exact hT decO x Ω P ω d.1 d.2 ws (eoGives_of_some hws)

theorem innerF_vLog_sub_coll {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {decO : List (ℕ × Col) → List T} {Ω : Type} {P : Prover Ω} {x : Msg} {decC : Msg → Msg → Coin}
    (hL : CallsLogColl H D I DO) (hpc : PublicCoin H I decC) (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) :
    (∀ e ∈ (innerF H D I DO decO P x decC ω d).vLog, e ∈ D.calls x ((eoOut H D I DO decO P x ω d).getD []) ∨
      e ∈ setupCalls H D (runVstar H D I P x ω d.1)) ∨
      CallsColl H D x (runVstar H D I P x ω d.1) ((eoOut H D I DO decO P x ω d).getD []) := by
  by_cases hk : keepD H D I DO decO P x ω d = true
  · have hk' : (eoOut H D I DO decO P x ω d).isSome = true ∧ Agrees H D I DO decO P x ω d := by
      simpa only [keepD, decide_eq_true_eq] using hk
    obtain ⟨ws, hws⟩ := Option.isSome_iff_exists.1 hk'.1
    rw [innerF_eq_innerRun hpc hws hk'.2, hws]
    exact hL decO x Ω P ω d.1 d.2 ws (eoGives_of_some hws)
  · exact Or.inl fun e he => by simp [innerF, hk, abortTranscript] at he

theorem PrBAbort_le_reg {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {decC : Msg → Msg → Coin} (hL : CallsLogColl H D I DO) (hT : CallsTrue H D I DO) (hN0 : CallsNil H D)
    (hpc : PublicCoin H I decC) (hR : InnerRounds D I) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    PrBAbort H I.V (innerP H D I DO decO P x decC) (xI D x) (keepA H D I DO decO P x) ≤
      D.V.k x * (PrBe H D I DO decO P x + PrBr H D I DO decO P x) := by
  have hcoll : ∀ (j : Fin (I.V.k (xI D x))) (ω : Ω) (ρ ρ' : Fin (I.V.k (xI D x)) → Coin)
      (a a' : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x),
      (findColl ((abortRun H (innerP H D I DO decO P x decC) I.V (xI D x) (keepA H D I DO decO P x) ω (ρ, a)).vLog ++
        (abortRun H (innerP H D I DO decO P x decC) I.V (xI D x) (keepA H D I DO decO P x) ω
          (forkCoins j ρ ρ', a')).vLog)).isSome →
      forkReg H D I DO decO P x j.val (ω, extend ρ a.1, extend ρ' a'.1, a.2, a'.2) := by
    intro j ω ρ ρ' a a' h
    rw [abortRun_eq (hR x), abortRun_eq (hR x)] at h
    unfold forkReg forkColl regColl
    simp only
    rw [extend_forkCoins]
    have htrue : ∀ p ∈ D.calls x ((eoOut H D I DO decO P x ω (extend ρ a.1, a.2)).getD []) ++
        D.calls x ((eoOut H D I DO decO P x ω (extend (forkCoins j ρ ρ') a'.1, a'.2)).getD []), H p.1 = p.2 :=
      fun p hp => (List.mem_append.1 hp).elim (calls_true_of hT hN0 ω _ p) (calls_true_of hT hN0 ω _ p)
    rcases innerF_vLog_sub_coll (decC := decC) hL hpc ω (extendSample (ρ, a)) with h1 | hc1
    · rcases innerF_vLog_sub_coll (decC := decC) hL hpc ω (extendSample (forkCoins j ρ ρ', a')) with h2 | hc2
      · refine coll_or_reg_split htrue (fun p hp => ?_) h
        rcases List.mem_append.1 hp with hp | hp
        · rcases h1 p hp with h | h
          · exact Or.inl (List.mem_append_left _ h)
          · exact Or.inr (List.mem_append_left _ h)
        · rcases h2 p hp with h | h
          · exact Or.inl (List.mem_append_right _ h)
          · exact Or.inr (List.mem_append_right _ h)
      · exact coll_or_reg htrue (collOrReg_of_callsColl htrue (Or.inr hc2))
    · exact coll_or_reg htrue (collOrReg_of_callsColl htrue (Or.inl hc1))
  have hle : PrBAbort H I.V (innerP H D I DO decO P x decC) (xI D x) (keepA H D I DO decO P x) ≤
      ∑ j : Fin (I.V.k (xI D x)), Pr (forkReg H D I DO decO P x j) := by
    unfold PrBAbort
    by_cases hk : 0 < I.V.k (xI D x)
    · simp only [hk, ↓reduceDIte]
      unfold Finder.Pr
      have hS : 0 ≤ ∑ j : Fin (I.V.k (xI D x)), Pr (forkReg H D I DO decO P x j) :=
        Finset.sum_nonneg fun j _ => Pr_nonneg _
      refine ((Pr_mono fun s hs => ?_).trans_eq
        (((unifMap_bAbort (Ω := Ω) (C := (EO D I DO decO).C x) (hR x)).Pr
          (fun t => forkReg H D I DO decO P x t.1.val t.2)).trans (Pr_guess _))).trans
        (div_le_self hS (by exact_mod_cast hk))
      obtain ⟨u, v, huv, -⟩ := hs
      exact hcoll s.1.1 s.1.2.1 s.1.2.2.1 s.1.2.2.2 s.2.1 s.2.2 (Option.isSome_iff_exists.2 ⟨(u, v), huv⟩)
    · simp only [hk, ↓reduceDIte]
      exact Finset.sum_nonneg fun j _ => Pr_nonneg _
  refine hle.trans (le_trans ?_ (sum_forkReg_le H D I DO decO P x))
  rw [Fin.sum_univ_eq_sum_range (fun j => Pr (forkReg H D I DO decO P x j)),
    Fin.sum_univ_eq_sum_range (fun j => Pr (forkReg H D I DO decO P x j))]
  exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono (hR x)) fun j _ _ => Pr_nonneg _

def RunPairs (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (decC : Msg → Msg → Coin) : Prop :=
  ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin), (run H P D.V x ω r).out = .acc →
    I.V.k (xI D x) + D.kS x ≤ (run H P D.V x ω r).msgs.length ∧ ∀ i < I.V.k (xI D x) + D.kS x,
      ∃ m, ((run H P D.V x ω r).msgs.getD i ([], .rej)).2 = .msg m ∧ decC x m = coinSeq r i

def VstarShows (D : Deploy Coin) (I : Inner Coin W) (L : Layout) (shape : Form → Prop) (decC : Msg → Msg → Coin)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form → ∀ ps : List (Msg × Out),
    ps.length = I.V.k (xI D x) + D.kS x → ∀ c : ℕ → Coin,
      (∀ i < ps.length, ∃ m, (ps.getD i ([], .rej)).2 = .msg m ∧ decC x m = c i) →
        ∀ s < D.nS x, shows s x ((D.vstar x ps).getD s []) c ∧
          ∀ i < L.rounds x, cmt x i ((D.vstar x ps).getD s []) = (ps.map Prod.fst).getD i []

omit [Fintype Coin] [Nonempty Coin] in
theorem staged_of_vstarShows {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {L : Layout}
    {shape : Form → Prop} {decC : Msg → Msg → Coin} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {cmt : Msg → ℕ → Msg → Msg} (hP : RunPairs H D I decC) (hV : VstarShows D I L shape decC shows cmt)
    (hL : ∀ x, L.rounds x ≤ I.V.k (xI D x) + D.kS x) : Staged H D I L shape shows cmt := by
  intro x st hst hf Ω P ω r hacc s hs
  obtain ⟨hlen, hpr⟩ := hP x Ω P ω r hacc
  have hpl : ((run H P D.V x ω r).msgs.take (I.V.k (xI D x) + D.kS x)).length = I.V.k (xI D x) + D.kS x := by
    simp only [List.length_take]
    omega
  have hc : ∀ i < ((run H P D.V x ω r).msgs.take (I.V.k (xI D x) + D.kS x)).length, ∃ m,
      (((run H P D.V x ω r).msgs.take (I.V.k (xI D x) + D.kS x)).getD i ([], .rej)).2 = .msg m ∧
        decC x m = coinSeq r i := by
    intro i hi
    rw [hpl] at hi
    obtain ⟨m, hm, hd⟩ := hpr i hi
    refine ⟨m, ?_, hd⟩
    rwa [List.getD_eq_getElem?_getD, List.getElem?_take_of_lt hi, ← List.getD_eq_getElem?_getD]
  obtain ⟨h1, h2⟩ := hV x st hst hf _ hpl (coinSeq r) hc s hs
  refine ⟨h1, fun i hi => ?_⟩
  unfold runVstar
  rw [h2 i hi]
  have hi' : i < I.V.k (xI D x) + D.kS x := Nat.lt_of_lt_of_le hi (hL x)
  simp only [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_take_of_lt hi']

def VStarParts (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) (L : Layout)
    (opening : Opening) (shape : Form → Prop) : Prop :=
  ∃ (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop)
    (cmt : Msg → ℕ → Msg → Msg) (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg))
    (pub : Msg → List (Msg × Msg)),
    (∀ s, PartChecks H D DO L shape shows chk s) ∧ (∀ s, PartOpens H D DO L shape shows opening cmt s) ∧
      (∀ s, PartLogs H D DO L shape shows logs s) ∧ Staged H D I L shape shows cmt ∧ Decomp H D I.V shape chk ∧
      LogDecompPub H D I.V shape logs pub ∧ PubReg H D shape pub ∧ HeldPub D I L shape

theorem codeFacts_of_parts {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {L : Layout} {opening : Opening} (hRef : Refuses H D) (hR : ReadsThrough H D I.V L.msgs L.calls)
    (hVL : VstarLen D) (hT : ∀ s, PartCallsTrue H D DO L s) (hPub : ReadsPub H D I DO L)
    (hFin : FinalOpens H D I DO opening) (hD : VStarParts H D I DO L opening Form.Dense)
    (hH : VStarParts H D I DO L opening Form.Holo) (hS : VStarParts H D I DO L opening Form.Sparse) :
    InnerViewRun H D I DO ∧ CallsNil H D ∧ CallsTrue H D I DO ∧ CallsOpenPlayed H D I DO opening L.fixedAt L.pubAt ∧
      CallsLogColl H D I DO ∧ InnerAcceptsR H D I DO := by
  obtain ⟨showsD, chkD, cmtD, logsD, pubD, hCD, hOD, hLD, hSD, hDD, hLDD, hPD, hHD⟩ := hD
  obtain ⟨showsH, chkH, cmtH, logsH, pubH, hCH, hOH, hLH, hSH, hDH, hLDH, hPH, hHH⟩ := hH
  obtain ⟨showsS, chkS, cmtS, logsS, pubS, hCS, hOS, hLS, hSS, hDS, hLDS, hPS, hHS⟩ := hS
  have hTr : CallsTrue H D I DO := callsTrue_of_parts hRef hR hT
  exact ⟨innerViewRun_of_readsThrough hR, callsNil_of_readsThrough hR, hTr,
    callsOpenPlayed_of_parts hRef hTr hR hVL hPub hFin hHD hOD hSD hHH hOH hSH hHS hOS hSS,
    callsLogColl_of_parts hRef hTr hR hVL hLD hOD hSD hLDD hPD hLH hOH hSH hLDH hPH hLS hOS hSS hLDS hPS,
    innerAcceptsR_of_parts hRef hTr hR hVL hCD hOD hSD hDD hCH hOH hSH hDH hCS hOS hSS hDS⟩

def VStarPartsStaging (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) (H : Msg → Msg) (L : Layout)
    (opening : Opening) (decC : Msg → Msg → Coin) (shape : Form → Prop) : Prop :=
  ∃ (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop)
    (cmt : Msg → ℕ → Msg → Msg) (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg))
    (pub : Msg → List (Msg × Msg)),
    (∀ s, PartChecks H D DO L shape shows chk s) ∧ (∀ s, PartOpens H D DO L shape shows opening cmt s) ∧
      (∀ s, PartLogs H D DO L shape shows logs s) ∧ VstarShows D I L shape decC shows cmt ∧
        Decomp H D I.V shape chk ∧ LogDecompPub H D I.V shape logs pub ∧ PubReg H D shape pub ∧ HeldPub D I L shape

omit [Fintype Coin] [Nonempty Coin] in
theorem vstarParts_of_staging {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {L : Layout} {opening : Opening} {decC : Msg → Msg → Coin} {shape : Form → Prop} (hP : RunPairs H D I decC)
    (hL : ∀ x, L.rounds x ≤ I.V.k (xI D x) + D.kS x) (h : VStarPartsStaging D I DO H L opening decC shape) :
    VStarParts H D I DO L opening shape := by
  obtain ⟨shows, chk, cmt, logs, pub, hC, hO, hLg, hV, hD, hLD, hPR, hH⟩ := h
  exact ⟨shows, chk, cmt, logs, pub, hC, hO, hLg, staged_of_vstarShows hP hV hL, hD, hLD, hPR, hH⟩

theorem codeFacts_of_staging {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {L : Layout} {opening : Opening} {decC : Msg → Msg → Coin} (hRef : Refuses H D)
    (hR : ReadsThrough H D I.V L.msgs L.calls) (hVL : VstarLen D) (hT : ∀ s, PartCallsTrue H D DO L s)
    (hP : RunPairs H D I decC) (hPub : ReadsPub H D I DO L) (hFin : FinalOpens H D I DO opening)
    (hL : ∀ x, L.rounds x ≤ I.V.k (xI D x) + D.kS x) (hD : VStarPartsStaging D I DO H L opening decC Form.Dense)
    (hH : VStarPartsStaging D I DO H L opening decC Form.Holo)
    (hS : VStarPartsStaging D I DO H L opening decC Form.Sparse) :
    InnerViewRun H D I DO ∧ CallsNil H D ∧ CallsTrue H D I DO ∧ CallsOpenPlayed H D I DO opening L.fixedAt L.pubAt ∧
      CallsLogColl H D I DO ∧ InnerAcceptsR H D I DO :=
  codeFacts_of_parts hRef hR hVL hT hPub hFin (vstarParts_of_staging hP hL hD) (vstarParts_of_staging hP hL hH)
    (vstarParts_of_staging hP hL hS)

end Facts

/-- Binds only where `salt` and `enc` are injective: two openings of one commit string to different messages collide in their salts' hashes or in their messages' digests. -/
def Opening.twoCalls (salt enc : Msg → Msg) (hsalt : Function.Injective salt) (henc : Function.Injective enc)
    (split : Msg → Msg × Msg) (dig : Msg → Msg → Msg) : Opening where
  opens a m cs := ∃ y, (salt y, (split a).2) ∈ cs ∧ (enc m, dig (split a).1 y) ∈ cs
  bind H a m m' cs cs' := by
    rintro ⟨y, hy, hm⟩ ⟨y', hy', hm'⟩ hne hall
    have hin : ∀ {p}, p ∈ cs → p.1 ∈ (cs ++ cs').map Prod.fst := fun hp =>
      List.mem_map.2 ⟨_, List.mem_append_left _ hp, rfl⟩
    have hin' : ∀ {p}, p ∈ cs' → p.1 ∈ (cs ++ cs').map Prod.fst := fun hp =>
      List.mem_map.2 ⟨_, List.mem_append_right _ hp, rfl⟩
    by_cases h : y = y'
    · subst h
      exact ⟨enc m, hin hm, enc m', hin' hm', fun e => hne (henc e),
        (hall _ (List.mem_append_left _ hm)).trans (hall _ (List.mem_append_right _ hm')).symm⟩
    · exact ⟨salt y, hin hy, salt y', hin' hy', fun e => h (hsalt e),
        (hall _ (List.mem_append_left _ hy)).trans (hall _ (List.mem_append_right _ hy')).symm⟩

end Claims.Recursion.VStar
