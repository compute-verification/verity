module

public import Proofs.Flock.VBridge.Algebra.Zerocheck
public import Proofs.Flock.Soundness.Refine.Frames
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.Combinatorics.Matroid.Init
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import VCVio.Prelude.Core

/-! The verifier's statement binding and zerocheck return `.ok` when the proof's messages pass the zerocheck's checks
(`bindAndZerocheck_of_checks`). -/

@[expose] public section

namespace Flock.RecAlgebra

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine Transcript

def AnswersL (r : Round) (L : List UInt8) : Prop := ∀ c : ByteArray, c.data.toList = L → r.answers c

theorem AnswersL.of_eq {r : Round} {L L' : List UInt8} (h : AnswersL r L) (e : L' = L) : AnswersL r L' := e ▸ h

theorem answersL_of_retains {r : Round} {L : List UInt8} (h : Retains r L) : AnswersL r L := by
  intro c hc
  unfold Retains at h
  unfold Round.answers
  cases hb : r.bytes with
  | none => rw [hb] at h; cases h
  | some B =>
      rw [hb] at h
      have h' : B.data.toList = L := Option.some.inj (h : some B.data.toList = some L)
      exact ByteArray.ext (Array.ext' (hc.trans h'.symm))

section Steps

variable {α β : Type}

theorem bind_of_ok {x : TM α} {f : α → TM β} {t t₁ : Transcript} {a : α} (h : x t = .ok (a, t₁)) :
    (x >>= f) t = f a t₁ := by
  change (StateT.bind x f) t = _
  unfold StateT.bind
  change Except.bind (x t) _ = _
  rw [h]
  rfl

theorem label_eq (l : String) (t : Transcript) :
    label l t = .ok ((), { t with pending := t.pending ++ header 2 0 l.toUTF8.size ++ padded l.toUTF8 }) := rfl

theorem absorbBytes_eq (b : ByteArray) (t : Transcript) :
    absorbBytes b t = .ok ((), { t with pending := t.pending ++ header 5 0 b.size ++ padded b }) := rfl

theorem absorbSlice_eq (vs : Array F128) (t : Transcript) :
    absorbSlice vs t = .ok ((), { t with pending := vs.foldl F128.pushBytes (t.pending ++ header 3 2 vs.size) }) := rfl

theorem absorbF_eq (v : F128) (t : Transcript) :
    absorbF v t = .ok ((), { t with pending := F128.pushBytes (t.pending ++ header 3 1 1) v }) := rfl

theorem check_eq {c : Bool} (h : c = true) (why : String) (t : Transcript) : Flock.check c why t = .ok ((), t) := by
  subst h; rfl

theorem squeeze_of_answers {kind : UInt8} {n : ℕ} (hn : n ≠ 0) {t : Transcript} (hN : (t.rounds[t.k]!).n = n)
    (ha : AnswersL (t.rounds[t.k]!) (t.pending.data.toList ++ (Frame.sq kind n).enc)) :
    squeeze kind n t = .ok ((t.rounds[t.k]!).coins, ⟨t.rounds, t.k + 1, ByteArray.empty⟩) := by
  have hlt : t.k < t.rounds.size := by
    by_contra hge
    rw [getElem!_neg t.rounds t.k hge] at hN
    exact hn hN.symm
  rw [getElem!_pos t.rounds t.k hlt] at hN ha ⊢
  unfold squeeze
  dsimp only
  rw [ite_eq_right hn, Array.getElem?_eq_getElem hlt]
  dsimp only
  rw [ite_eq_right (show ¬ (t.rounds[t.k].n ≠ n) from fun h => h hN),
    ite_eq_right (show ¬ ¬ t.rounds[t.k].answers (framed t kind n) from fun h => h (ha _ (framed_enc t kind n)))]

theorem sampleVec_of_answers {n : ℕ} (hn : n ≠ 0) {t : Transcript} (hN : (t.rounds[t.k]!).n = n)
    (ha : AnswersL (t.rounds[t.k]!) (t.pending.data.toList ++ (Frame.sq 2 n).enc)) :
    sampleVec n t = .ok ((t.rounds[t.k]!).coins, ⟨t.rounds, t.k + 1, ByteArray.empty⟩) :=
  squeeze_of_answers hn hN ha

theorem sample_of_answers {t : Transcript} (hN : (t.rounds[t.k]!).n = 1)
    (ha : AnswersL (t.rounds[t.k]!) (t.pending.data.toList ++ (Frame.sq 1 1).enc)) :
    sample t = .ok ((t.rounds[t.k]!).coins.getD 0 0, ⟨t.rounds, t.k + 1, ByteArray.empty⟩) := by
  unfold sample
  rw [bind_of_ok (squeeze_of_answers Nat.one_ne_zero hN ha)]
  rfl

theorem forIn_rounds_of {σ : Type} (body : ℕ → σ → TM (ForInStep σ)) (step : ℕ → σ → Array F128 → σ)
    (Q : ℕ → Round → Prop)
    (hbody : ∀ i s (t : Transcript), t.pending = ByteArray.empty → Q i (t.rounds[t.k]!) →
      body i s t = .ok (.yield (step i s (t.rounds[t.k]!).coins), ⟨t.rounds, t.k + 1, ByteArray.empty⟩)) :
    ∀ (n a base : ℕ) (s : σ) (t : Transcript), t.k = base + a → t.pending = ByteArray.empty →
      (∀ i ∈ List.range' a n, Q i (t.rounds[base + i]!)) →
      (forIn (List.range' a n) s body : TM σ) t =
        .ok ((List.range' a n).foldl (fun s i => step i s (t.rounds[base + i]!).coins) s,
          ⟨t.rounds, t.k + n, ByteArray.empty⟩) := by
  intro n
  induction n with
  | zero =>
      intro a base s t _ hp _
      obtain ⟨R, k, P⟩ := t
      dsimp only at hp
      subst hp
      simp only [List.range'_zero, List.forIn_nil, List.foldl_nil, Nat.add_zero]
      rfl
  | succ n ih =>
      intro a base s t hk hp hQ
      rw [forIn_range'_cons, bind_of_ok (hbody a s t hp (by rw [hk]; exact hQ a (by simp)))]
      dsimp only
      rw [ih (a + 1) base _ ⟨t.rounds, t.k + 1, ByteArray.empty⟩ (by dsimp only; omega) rfl
        (fun i hi => hQ i (by simp only [List.mem_range'_1] at hi ⊢; omega))]
      dsimp only
      rw [List.range'_succ, List.foldl_cons, ← hk, show t.k + 1 + n = t.k + (n + 1) by omega]

theorem forIn_range_of {σ : Type} (body : ℕ → σ → TM (ForInStep σ)) (step : ℕ → σ → Array F128 → σ)
    (Q : ℕ → Round → Prop)
    (hbody : ∀ i s (t : Transcript), t.pending = ByteArray.empty → Q i (t.rounds[t.k]!) →
      body i s t = .ok (.yield (step i s (t.rounds[t.k]!).coins), ⟨t.rounds, t.k + 1, ByteArray.empty⟩))
    (n base : ℕ) (s : σ) (t : Transcript) (hk : t.k = base + 0) (hp : t.pending = ByteArray.empty)
    (hQ : ∀ i ∈ List.range' 0 n, Q i (t.rounds[base + i]!)) :
    (forIn [:n] s body : TM σ) t =
      .ok ((List.range' 0 n).foldl (fun s i => step i s (t.rounds[base + i]!).coins) s,
        ⟨t.rounds, t.k + n, ByteArray.empty⟩) := by
  rw [forIn_range_eq]
  exact forIn_rounds_of body step Q hbody n 0 base s t hk hp hQ

end Steps

theorem zcFold_congr (r : Array F128) (p : ZerocheckProof) {C C' : ℕ → Array F128} (n : ℕ) (claim : F128)
    (h : ∀ i < n, (C i).getD 0 0 = (C' i).getD 0 0) : zcFold r p C n claim = zcFold r p C' n claim := by
  unfold zcFold
  suffices e : ∀ (l : List ℕ) (s : F128 × Array F128), (∀ i ∈ l, i < n) →
      l.foldl (fun s i => zStep r p.rounds i s (C i)) s = l.foldl (fun s i => zStep r p.rounds i s (C' i)) s from
    e _ _ fun i hi => by simp only [List.mem_range'_1] at hi; omega
  intro l
  induction l with
  | nil => intro s _; rfl
  | cons i l ih =>
      intro s hl
      rw [List.foldl_cons, List.foldl_cons, ih _ fun j hj => hl j (List.mem_cons_of_mem _ hj)]
      congr 1
      simp only [zStep, h i (hl i List.mem_cons_self)]

theorem bindAndZerocheck_of_checks (m : ℕ) (hm : 14 ≤ m) (digest cap : ByteArray) (p : ZerocheckProof)
    (t : Transcript)
    (hn₀ : (t.rounds[t.k]!).n = 6) (hn₁ : (t.rounds[t.k + 1]!).n = m - 13) (hn₂ : (t.rounds[t.k + 2]!).n = 1)
    (hnr : ∀ i < m - 6, (t.rounds[t.k + 3 + i]!).n = 1)
    (ha₀ : AnswersL (t.rounds[t.k]!) (t.pending.data.toList ++ encs [.label "flock-r1cs-v0".toUTF8, .bytes digest,
      .bytes cap, .label "flock-zerocheck-v0".toUTF8, .sq 2 6]))
    (ha₁ : AnswersL (t.rounds[t.k + 1]!) (encs [.sq 2 (m - 13)]))
    (ha₂ : AnswersL (t.rounds[t.k + 2]!) (encs [.vec p.round1ab.toList, .vec p.round1c.toList, .sq 1 1]))
    (har : ∀ i < m - 6,
      AnswersL (t.rounds[t.k + 3 + i]!) (encs [.one (p.rounds[i]!).1, .one (p.rounds[i]!).2, .sq 1 1]))
    {r : Array F128} {z : F128} {rho : Array F128} {C : ℕ → Array F128}
    (hr : r = pinned ++ (t.rounds[t.k + 1]!).coins) (hz : z = (t.rounds[t.k + 2]!).coins.getD 0 0)
    (hC : ∀ i < m - 6, (C i).getD 0 0 = (t.rounds[t.k + 3 + i]!).coins.getD 0 0)
    (hpc : dot (lagrange nodesL den6Inv z) p.round1c = p.finalC)
    (hfold : zcFold r p C (m - 6) (zcStart p z) = (p.finalA * p.finalB, rho)) :
    ∃ t', bindAndZerocheck m digest cap p t =
        .ok ({ z := z, rho := rho, rRest := r, vA := p.finalA, vB := p.finalB, vC := p.finalC }, t') ∧
      t'.rounds = t.rounds ∧ t'.k = t.k + 3 + (m - 6) ∧
      t'.pending.data.toList = encs [.one p.finalA, .one p.finalB] := by
  subst hr hz
  rw [zcFold_congr _ _ _ _ (C' := fun i => (t.rounds[t.k + 3 + i]!).coins) hC] at hfold
  have h1 : (zcFold (pinned ++ (t.rounds[t.k + 1]!).coins) p (fun i => (t.rounds[t.k + 3 + i]!).coins) (m - 6)
      (zcStart p ((t.rounds[t.k + 2]!).coins.getD 0 0))).1 = p.finalA * p.finalB := by rw [hfold]
  have h2 : rho = (zcFold (pinned ++ (t.rounds[t.k + 1]!).coins) p (fun i => (t.rounds[t.k + 3 + i]!).coins) (m - 6)
      (zcStart p ((t.rounds[t.k + 2]!).coins.getD 0 0))).2 := by rw [hfold]
  subst h2
  refine ⟨⟨t.rounds, t.k + 3 + (m - 6),
    F128.pushBytes (F128.pushBytes (ByteArray.empty ++ header 3 1 1) p.finalA ++ header 3 1 1) p.finalB⟩,
    ?_, rfl, rfl, ?_⟩
  · unfold bindAndZerocheck
    refine Eq.trans (bind_of_ok (label_eq _ _)) ?_
    refine Eq.trans (bind_of_ok (absorbBytes_eq _ _)) ?_
    refine Eq.trans (bind_of_ok (absorbBytes_eq _ _)) ?_
    refine Eq.trans (bind_of_ok (label_eq _ _)) ?_
    refine Eq.trans (bind_of_ok (sampleVec_of_answers (by decide) hn₀ (ha₀.of_eq (by
      dsimp only
      simp only [label_enc, absorbBytes_enc]
      simp [encs])))) ?_
    refine Eq.trans (bind_of_ok (sampleVec_of_answers (by omega) hn₁ (ha₁.of_eq (by simp [encs])))) ?_
    refine Eq.trans (bind_of_ok (absorbSlice_eq _ _)) ?_
    refine Eq.trans (bind_of_ok (absorbSlice_eq _ _)) ?_
    refine Eq.trans (bind_of_ok (sample_of_answers hn₂ (ha₂.of_eq (by
      dsimp only
      simp only [absorbSlice_enc]
      simp [encs])))) ?_
    refine Eq.trans (bind_of_ok (check_eq (beq_iff_eq.mpr hpc) _ _)) ?_
    refine Eq.trans (bind_of_ok (forIn_range_of _ (zStep (pinned ++ (t.rounds[t.k + 1]!).coins) p.rounds)
      (fun i R => R.n = 1 ∧ AnswersL R (encs [.one (p.rounds[i]!).1, .one (p.rounds[i]!).2, .sq 1 1])) ?_ (m - 6)
      (t.k + 3) _ _ rfl rfl
      (fun i hi => by
        have hi' : i < m - 6 := by simp only [List.mem_range'_1] at hi; omega
        exact ⟨hnr i hi', har i hi'⟩))) ?_
    · intro i s tt hp hQ
      rcases hg : p.rounds[i]! with ⟨g1, ginf⟩
      simp only [hg] at hQ
      refine Eq.trans (bind_of_ok (absorbF_eq _ _)) ?_
      refine Eq.trans (bind_of_ok (absorbF_eq _ _)) ?_
      refine Eq.trans (bind_of_ok (sample_of_answers hQ.1 (hQ.2.of_eq (by
        dsimp only
        simp only [absorbF_enc, hp]
        simp [encs])))) ?_
      simp only [zStep, hg]
      rfl
    refine Eq.trans (bind_of_ok (check_eq (beq_iff_eq.mpr h1) _ _)) ?_
    refine Eq.trans (bind_of_ok (absorbF_eq _ _)) ?_
    refine Eq.trans (bind_of_ok (absorbF_eq _ _)) ?_
    rfl
  · simp only [absorbF_enc]
    simp [encs]

theorem zerocheck_plays_of_checks (m : ℕ) (hm : 15 ≤ m) (digest cap : ByteArray) (p : ZerocheckProof)
    (t : Transcript) (hwf : RoundsWF t.rounds)
    (hn₀ : (t.rounds[t.k]!).n = 6) (hn₁ : (t.rounds[t.k + 1]!).n = m - 13) (hn₂ : (t.rounds[t.k + 2]!).n = 1)
    (hnr : ∀ i < m - 6, (t.rounds[t.k + 3 + i]!).n = 1)
    (ha₀ : AnswersL (t.rounds[t.k]!) (t.pending.data.toList ++ encs [.label "flock-r1cs-v0".toUTF8, .bytes digest,
      .bytes cap, .label "flock-zerocheck-v0".toUTF8, .sq 2 6]))
    (ha₁ : AnswersL (t.rounds[t.k + 1]!) (encs [.sq 2 (m - 13)]))
    (ha₂ : AnswersL (t.rounds[t.k + 2]!) (encs [.vec p.round1ab.toList, .vec p.round1c.toList, .sq 1 1]))
    (har : ∀ i < m - 6,
      AnswersL (t.rounds[t.k + 3 + i]!) (encs [.one (p.rounds[i]!).1, .one (p.rounds[i]!).2, .sq 1 1]))
    {r : Array F128} {z : F128} {rho : Array F128} {C : ℕ → Array F128}
    (hr : r = pinned ++ (t.rounds[t.k + 1]!).coins) (hz : z = (t.rounds[t.k + 2]!).coins.getD 0 0)
    (hC : ∀ i < m - 6, (C i).getD 0 0 = (t.rounds[t.k + 3 + i]!).coins.getD 0 0)
    (hpc : dot (lagrange nodesL den6Inv z) p.round1c = p.finalC)
    (hfold : zcFold r p C (m - 6) (zcStart p z) = (p.finalA * p.finalB, rho)) :
    ∃ t', bindAndZerocheck m digest cap p t =
        .ok ({ z := z, rho := rho, rRest := r, vA := p.finalA, vB := p.finalB, vC := p.finalC }, t') ∧
      t'.rounds = t.rounds ∧ t'.k = t.k + 3 + (m - 6) ∧
      t'.pending.data.toList = encs [.one p.finalA, .one p.finalB] ∧
      ∃ zc', Plays fOf (Model.zerocheck execArith m) (zcMsgs m p) (consumed t.rounds t.k t'.k) (some zc') ∧
        ZcRel m { z := z, rho := rho, rRest := r, vA := p.finalA, vB := p.finalB, vC := p.finalC } zc' := by
  obtain ⟨t', h, hrt, hkt, hpt⟩ := bindAndZerocheck_of_checks m (by omega) digest cap p t hn₀ hn₁ hn₂ hnr ha₀ ha₁
    ha₂ har hr hz hC hpc hfold
  exact ⟨t', h, hrt, hkt, hpt, (zerocheck_refines m hm digest cap p t t' _ hwf h).2.2⟩

end Flock.RecAlgebra
