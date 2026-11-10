module

public import Proofs.Flock.SessionLoop.SkelPiop
public import Proofs.Flock.Claims.ZK.FrameProg
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
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.ZK.FrameInner
import Proofs.Flock.Claims.ZK.FrameOuter

/-!
# The frame's state through the PIOP's scheduled rounds

A rep's frame checks each scheduled round against itself (`Loop.schedReqs`). Where the schedule's round 0 is the
zerocheck's binding with the stream's domain and its later rounds are the PIOP's (`piopFr`, `piopK`, `piopN`), the frame
is in the algebraic phase from the binding on, outside the inner proof and level 0, at the absorbed position of the
values the rounds before carry, owing the round's coins (`Loop.run_piop`). The algebraic phase's schedule is such a
schedule (`Sched.rounds_piop`), so the loop's state at a rep's PIOP round is that (`Loop.repState_piop`).
-/

@[expose] public section

namespace Claims.ZKServer.Loop

open _root_.Flock (F128)
open _root_.Flock.Firewall.Frame (Cfg Masker roundFree LABEL_ZC)
open FlockSoundness.Refine (Frame encs)

/-- **The frame through the PIOP's round `a`**, from a rep's start past an earlier rep's coins: where the schedule's
round 0 is the zerocheck's binding with the stream's domain, its rounds `1` to `a` are the PIOP's, and the rounds and
coins up to round `a` neither end the rep nor leave it owing none, the frame is in the algebraic phase, outside the inner
proof and level 0, at the absorbed position of the values of rounds `0` to `a`, owing round `a`'s coins. -/
theorem run_piop {c : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (P : Array F128)
    (sched : Array (ByteArray × ℕ)) (cr : Array (Array F128)) {m kLog : ℕ} {digest cap dom root : ByteArray}
    {lay : _root_.Flock.Zk.PadLayout} {mk : ℕ → F128} (hm : 14 ≤ m) (hm' : m < 2 ^ 64)
    (hdg : digest.size < 2 ^ 64) (hcap : cap.size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (h0 : sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest,
      .bytes cap, .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6)) (a : ℕ) :
    (∀ k, 1 ≤ k → k ≤ a → sched.getD k (.empty, 0) =
        (ByteArray.mk (encs (piopFr m kLog digest cap root lay mk k) ++ (Frame.sq (piopK k) (piopN m k)).enc).toArray,
          piopN m k)) →
      (piopV m kLog digest cap root lay mk (a + 1)).length ≤ c.order.size →
      (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr a)).2 = false →
      (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr a)).1.m.owed ≠ none →
      (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr a)).1.m.owed = some (piopN m a) ∧
      (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr a)).1.m.on = true ∧
      (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr a)).1.m.inner = none ∧
      (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr a)).1.m.l0 = false ∧
      (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr a)).1.m.pos =
        (piopV m kLog digest cap root lay mk (a + 1)).length := by
  induction a with
  | zero =>
    intro _ _ h2 ho
    have e0 : schedReqs P sched cr 0 =
        (if P.isEmpty then [] else [.prior P.size (_root_.Flock.Firewall.Frame.wordBytes P)]) ++
          [.round (sched.getD 0 (.empty, 0)).2 (sched.getD 0 (.empty, 0)).1
            (.sched (sched.getD 0 (.empty, 0)).1)] := by
      unfold schedReqs
      rw [List.range_zero, List.flatMap_nil, List.append_nil]
    have hv : (piopV m kLog digest cap root lay mk (0 + 1)).length = 0 := by
      rw [piopV_succ]
      simp [piopV, piopFr, fvals]
    rw [e0, h0] at h2 ho ⊢
    dsimp only at h2 ho ⊢
    obtain ⟨p, hp⟩ := run_prior c P _ h2
    rw [hp] at h2 ho ⊢
    rcases run_sched_cons (st := { m := { prior := p } }) rfl h2 with ⟨out, sx, why, hck, he⟩ | he
    · rw [he]
      have hsx : sx = { ({ prior := p } : Masker) with on := true, owed := some 6 } ∨
          sx = { ({ prior := p } : Masker) with stopped := true, on := true, owed := some 6 } := by
        rcases check_self hck with ⟨fr, hr⟩ | ⟨fr, hr⟩
        · obtain ⟨o, fr', hr'⟩ := roundFree_zcHead c { prior := p } digest cap dom hc rfl rfl rfl hdg hcap hdom
            (Nat.zero_le _)
          rw [hr'] at hr
          simp only [Except.ok.injEq, Prod.mk.injEq] at hr
          exact Or.inl hr.2.1.symm
        · obtain ⟨o, fr', hr'⟩ := roundFree_zcHead c { ({ prior := p } : Masker) with stopped := true } digest cap
            dom hc rfl rfl rfl hdg hcap hdom (Nat.zero_le _)
          rw [hr'] at hr
          simp only [Except.ok.injEq, Prod.mk.injEq] at hr
          exact Or.inr hr.2.1.symm
      rcases hsx with rfl | rfl <;> exact ⟨rfl, rfl, rfl, rfl, hv.symm⟩
    · rw [he] at ho
      exact (ho rfl).elim
  | succ a ih =>
    intro hs hb h2 ho
    have hb' : (piopV m kLog digest cap root lay mk (a + 1)).length ≤ c.order.size :=
      (piopV_mono m kLog digest cap root lay mk (Nat.le_succ _)).trans hb
    have hkn : (piopK (a + 1) = 2 ∧ piopN m (a + 1) = m - 13 ∧ a = 0) ∨
        (piopK (a + 1) = 1 ∧ piopN m (a + 1) = 1 ∧ 1 ≤ a) := by
      rcases Nat.eq_zero_or_pos a with rfl | ha
      · exact .inl ⟨rfl, rfl, rfl⟩
      · refine .inr ⟨?_, ?_, ha⟩
        · unfold piopK; rw [if_neg (by omega)]
        · unfold piopN; rw [if_neg (by omega), if_neg (by omega)]
    have hk : piopK (a + 1) = 2 ∨ (piopK (a + 1) = 1 ∧ piopN m (a + 1) = 1) := by
      rcases hkn with ⟨h1, -, -⟩ | ⟨h1, h2, -⟩
      · exact .inl h1
      · exact .inr ⟨h1, h2⟩
    have h0' : ¬ (piopN m (a + 1) = 0 ∧ piopK (a + 1) = 2) := by
      rcases hkn with ⟨-, h2, -⟩ | ⟨h1, -, -⟩
      · rintro ⟨h, -⟩
        omega
      · rintro ⟨-, h⟩
        rw [h1] at h
        exact absurd h (by decide)
    have hn : piopN m (a + 1) < 2 ^ 64 := by
      rcases hkn with ⟨-, h2, -⟩ | ⟨-, h2, -⟩ <;> omega
    have hmp : (∀ f ∈ piopFr m kLog digest cap root lay mk (a + 1), midFrame f) ∧
        (∀ f ∈ piopFr m kLog digest cap root lay mk (a + 1), plainFrame f) := by
      rcases Nat.eq_zero_or_pos a with rfl | ha
      · have e : piopFr m kLog digest cap root lay mk (0 + 1) = [] := rfl
        rw [e]
        exact ⟨fun f hf => by simp at hf, fun f hf => by simp at hf⟩
      · exact piopFr_frames m kLog digest cap root lay mk (by omega)
    rw [schedReqs_succ, run_append] at h2 ho ⊢
    cases hr : (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr a)).2 with
    | true =>
      rw [if_pos hr, hr] at h2
      cases h2
    | false =>
      simp only [hr, Bool.false_eq_true, ↓reduceIte] at h2 ho ⊢
      obtain ⟨s1, ha, he1⟩ := run_coins_cons c _ _ _ h2
      obtain ⟨-, hon, hi, hl, hpos⟩ := ih (fun k h1 h2 => hs k h1 (by omega)) hb' hr
        (fun h => answer_owed_none _ h ha)
      have hs1 := answer_alg c _ _ hi hl ha
      have hs1o : s1.owed = none := by rw [hs1]
      have hs1on : s1.on = true := by rw [hs1]; exact hon
      have hs1i : s1.inner = none := by rw [hs1]; exact hi
      have hs1l : s1.l0 = false := by rw [hs1]; exact hl
      have hs1p : s1.pos = (piopV m kLog digest cap root lay mk (a + 1)).length := by rw [hs1]; exact hpos
      rw [he1] at h2 ho ⊢
      rw [hs (a + 1) (by omega) le_rfl] at h2 ho ⊢
      dsimp only at h2 ho ⊢
      rcases run_sched_cons (by exact hs1o) h2 with ⟨out, sx, why, hck, he2⟩ | he2
      · rw [he2]
        have key : ∀ s' : Masker, s'.on = true → s'.inner = none → s'.l0 = false →
            s'.pos = (piopV m kLog digest cap root lay mk (a + 1)).length → ∀ fr,
            roundFree c s' (ByteArray.mk (encs (piopFr m kLog digest cap root lay mk (a + 1)) ++
              (Frame.sq (piopK (a + 1)) (piopN m (a + 1))).enc).toArray) (piopN m (a + 1)) = .ok (out, sx, fr) →
            sx.owed = some (piopN m (a + 1)) ∧ sx.on = true ∧ sx.inner = none ∧ sx.l0 = false ∧
              sx.pos = (piopV m kLog digest cap root lay mk (a + 1 + 1)).length := by
          intro s' h1 h2' h3 h4 fr hr'
          obtain ⟨o, fr', hr'', -⟩ := roundFree_on c s' _ _ _ hc h1 h2' hmp.1 hmp.2 hk h0' hn
            (by rw [h4, nAbs_fvals, ← List.length_append, ← piopV_succ]; exact hb)
          have e := Except.ok.inj (hr'.symm.trans hr'')
          simp only [Prod.mk.injEq] at e
          have hp : s'.pos + nAbs (piopFr m kLog digest cap root lay mk (a + 1)) =
              (piopV m kLog digest cap root lay mk (a + 1 + 1)).length := by
            rw [h4, piopV_succ m kLog digest cap root lay mk (a + 1), List.length_append, nAbs_fvals]
          rw [e.2.1]
          exact ⟨rfl, h1, h2', h3, hp⟩
        rcases check_self hck with ⟨fr, hr'⟩ | ⟨fr, hr'⟩
        · exact key s1 hs1on hs1i hs1l hs1p fr hr'
        · exact key { s1 with stopped := true } hs1on hs1i hs1l hs1p fr hr'
      · rw [he2] at ho
        exact (ho hs1o).elim

/-- **The loop's state at a rep's PIOP round `a + 1`**: where the frame takes round `a`'s coins at the rep's state
after it, with the values of rounds `0` to `a` inside the configuration's order, it is in the algebraic phase, outside
the inner proof and level 0, at their absorbed position (`run_piop` on the skeleton's rounds, `Sched.rounds_piop`). -/
theorem repState_piop {c c'' : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr0 cr : Array (Array F128)} {ns : ℕ → ℕ} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched) (hm : 14 ≤ t.st.m) (hm' : t.st.m < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hdom : dom.size < 2 ^ 64) {a : ℕ} (ha : a < piopNR t.st.m t.st.fold.kLog)
    (hb : (piopV t.st.m t.st.fold.kLog t.st.digest (_root_.Flock.Firewall.Finish.flat t.cm0.cap) Sched.root0
      (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0) (fun _ => 0) (a + 1)).length ≤ c.order.size)
    {cs : Array F128} {m : Masker} (hA : (repState c t rep dom xs hi cr0 cr ns (a + 1)).answer c'' cs = .ok m) :
    m.on = true ∧ m.inner = none ∧ m.l0 = false ∧
      m.pos = (piopV t.st.m t.st.fold.kLog t.st.digest (_root_.Flock.Firewall.Finish.flat t.cm0.cap) Sched.root0
        (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0) (fun _ => 0) (a + 1)).length := by
  obtain ⟨hsz, hpi⟩ := Sched.rounds_piop hm hm' hal
  obtain ⟨ha0, -⟩ := Sched.rounds_two hm hm' hal
  obtain ⟨rest, hrest⟩ := schedSoFar_pre hal hs
  have hNR : 6 ≤ piopNR t.st.m t.st.fold.kLog := by unfold piopNR; omega
  have hg : ∀ k, k < alg.size → sched.getD k (.empty, 0) = (alg[k]?).getD (.empty, 0) := fun k hk => by
    rw [hrest, Array.getD_eq_getD_getElem?, Array.getElem?_append_left hk]
  have h0 : sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8,
      .bytes t.st.digest, .bytes (_root_.Flock.Firewall.Finish.flat t.cm0.cap),
      .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6) := by
    rw [hg 0 (by omega), ha0, Option.getD_some, head_encs]
  have hsk : ∀ k, 1 ≤ k → k ≤ a → sched.getD k (.empty, 0) = (ByteArray.mk (encs (piopFr t.st.m t.st.fold.kLog
      t.st.digest (_root_.Flock.Firewall.Finish.flat t.cm0.cap) Sched.root0
      (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0) (fun _ => 0) k) ++
      (Frame.sq (piopK k) (piopN t.st.m k)).enc).toArray, piopN t.st.m k) := fun k h1 h2 => by
    rw [hg k (by omega), hpi k (by omega) (by omega), Option.getD_some]
  rw [repState_at c t rep dom xs hi cr0 cr ns a hal hs (by rw [hrest, Array.size_append]; omega)] at hA
  dsimp only at hA
  cases h2 : (_root_.Flock.Firewall.Frame.run c {} (schedReqs
      (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr a)).2
  · rw [h2] at hA
    simp only [Bool.false_eq_true, ↓reduceIte] at hA
    obtain ⟨-, hon, hin, hl, hpos⟩ := run_piop hc _ sched cr hm hm' hdg hcap hdom h0 a hsk hb h2
      (fun e => answer_owed_none cs e hA)
    rw [answer_alg c'' _ cs hin hl hA]
    exact ⟨hon, hin, hl, hpos⟩
  · rw [h2] at hA
    simp only [↓reduceIte] at hA
    exact absurd hA (answer_owed_none cs rfl)

end Claims.ZKServer.Loop
