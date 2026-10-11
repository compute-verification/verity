import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysReplay
import Proofs.Flock.Soundness.Discharge.Composed.Inner.Lincheck

/-!
# `Zk.replay`'s lincheck terms are `Lincheck`'s at the transcript's coins

`replay_lincheck`: on a run of `Zk.replay` from `t`, the "lincheck final" constraint (`cons[2]`), the `ab` claim's value
`abValue` and the `ab` claim `claims[0]` are `lcFinal`, `lcAbValue` and `lcAbClaim` at the coins of the rounds the run
consumed, and the circuit fold the replay called returned the entries `lcFinal` reads. Round `t.k + i`'s first coin is
`coinAt t i` (`(t.rounds[t.k + i]!).coins.getD 0 0`, what `sample` reads; `zk_replay_plays`'s numbering): the
zerocheck's `rSkip`, `rOuter`, `z` at `0, 1, 2`, its residual coins `ρ_j` at `3 + j` (`rhoOf`), then the lincheck's
`α`, `β` at
`3 + (m − 6)`, `4 + (m − 6)`, its round coins `t_j` at `5 + (m − 6) + j` (`tsOf`) and `r′` after them.

`foldAll_lcV`: `lincheck_complete`'s honest `z_partial` (`hzp`), the witness's block-folded columns folded at the round
coins, is entry `i` of the witness evaluated at the `ab` claim's point beyond the skip bits, `Σ_u eq(x, u)·w[64u + i]`
(the ring switch's `zPartialOf (claimOf m (lcAbClaim …)) w i`, written out).
-/

namespace FlockSoundness.Discharge.Composed.Lincheck

open Flock FlockLevel3 Model Refine

local notation "𝔽" => FlockLevel3.GF128

/-- Round `t.k + i`'s first coin, as `sample` reads it. -/
abbrev coinAt (t : Flock.Transcript) (i : ℕ) : F128 := (t.rounds[t.k + i]!).coins.getD 0 0

/-- The zerocheck's `m − 6` residual coins `ρ_j` (rounds `t.k + 3 + j`). -/
def rhoOf (m : ℕ) (t : Flock.Transcript) : Array F128 := Array.ofFn fun j : Fin (m - 6) => coinAt t (3 + j.val)

/-- The lincheck's `k_log − 6` round coins `t_j` (rounds `t.k + 5 + (m − 6) + j`). -/
def tsOf (m kLog : ℕ) (t : Flock.Transcript) : Array F128 :=
  Array.ofFn fun j : Fin (kLog - 6) => coinAt t (5 + (m - 6) + j.val)

/-- One round of `Zk.replay`'s lincheck loop on its state `(running, ts)`, at the round's coins `c`. -/
def lcLoopStep (lay : Zk.PadLayout) (rounds : Array (F128 × F128)) (j : ℕ) (s : Zk.Lin × Array F128)
    (c : Array F128) : Zk.Lin × Array F128 :=
  let (e1v, einfv) := rounds[j]!
  let rr := c.getD 0 0
  let e1 := Zk.Lin.masked e1v (lay.lc j 0) lay.len
  let einf := Zk.Lin.masked einfv (lay.lc j 1) lay.len
  let e0 := s.1.add e1
  let c1 := (e0.add e1).add einf
  (((einf.scale (rr * rr)).add (c1.scale rr)).add e0, s.2.push rr)

/-- The loop's running claim is `lcStep`'s fold, on any coin array that holds the rounds' coins. -/
theorem lcLoopStep_fold (lay : Zk.PadLayout) (rounds : Array (F128 × F128)) (C : ℕ → Array F128) (ts : Array F128)
    (x : Zk.Lin) (acc : Array F128) : ∀ n, (∀ j < n, ts[j]! = (C j).getD 0 0) →
      ((List.range n).foldl (fun s i => lcLoopStep lay rounds i s (C i)) (x, acc)).1 =
        (List.range n).foldl (lcStep lay rounds ts) x
  | 0, _ => rfl
  | n + 1, hC => by
      rw [List.range_succ, List.foldl_append, List.foldl_append, List.foldl_cons, List.foldl_cons, List.foldl_nil,
        List.foldl_nil, ← lcLoopStep_fold lay rounds C ts x acc n (fun j hj => hC j (by omega))]
      simp only [lcLoopStep, lcStep, hC n (by omega)]

/-- **`Zk.replay`'s lincheck terms at the transcript's coins.** On a run of `Zk.replay` from `t` (at `m ≥ 14`, where the
outer coins are one round), the circuit fold at the lincheck's `α`, `β`, the skip weights at `z`, `ρ[.. k_log − 6]` and
the round coins returned some `combv`, and `cons[2]`, `abValue` and `claims[0]` are `lcFinal`, `lcAbValue` and
`lcAbClaim` on the proof's masked messages at those coins. -/
theorem replay_lincheck (st : Setup) (lay : Zk.PadLayout) (p : Proof) (root : ByteArray) (hm : 14 ≤ st.m)
    {t t' : Flock.Transcript} {rp : Zk.Replayed} (h : Zk.replay st lay p root t = .ok (rp, t')) :
    ∃ combv : Array F128,
      st.fold.folded (coinAt t (3 + (st.m - 6))) (coinAt t (4 + (st.m - 6))) (skipWeights (coinAt t 2))
          ((rhoOf st.m t).extract 0 (st.kLog - 6)) (tsOf st.m st.kLog t) = .ok combv ∧
        rp.cons[2]! = ("lincheck final", lcFinal lay p.zerocheck.finalA p.zerocheck.finalB
          (coinAt t (3 + (st.m - 6))) (coinAt t (4 + (st.m - 6))) p.lincheck.rounds p.lincheck.zPartial
          (tsOf st.m st.kLog t) (st.kLog - 6) combv) ∧
        rp.abValue = lcAbValue lay p.lincheck.zPartial (coinAt t (5 + (st.m - 6) + (st.kLog - 6))) ∧
        rp.claims[0]! = lcAbClaim st.m st.kLog (rhoOf st.m t) (tsOf st.m st.kLog t)
          (coinAt t (5 + (st.m - 6) + (st.kLog - 6))) := by
  unfold Zk.replay at h
  dsimp only at h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbBytes_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨t₄, k₄, h⟩ := label_bind h
  obtain ⟨t₅, k₅, h⟩ := label_bind h
  obtain ⟨t₆, k₆, h⟩ := absorbBytes_bind h
  obtain ⟨-, -, t₇, r₇, kk₇, h⟩ := sampleVec_bind (by decide) h
  obtain ⟨-, -, t₈, r₈, kk₈, h⟩ := sampleVec_bind (by omega) h
  obtain ⟨t₉, k₉, h⟩ := absorbSlice_bind h
  obtain ⟨t₁₀, k₁₀, h⟩ := absorbSlice_bind h
  obtain ⟨-, -, t₁₁, r₁₁, kk₁₁, h⟩ := sample_bind h
  obtain ⟨s, t₁₂, hloop, h⟩ := bind_ok h
  obtain ⟨t₁₃, k₁₃, h⟩ := absorbF_bind h
  obtain ⟨t₁₄, k₁₄, h⟩ := absorbF_bind h
  obtain ⟨t₁₅, k₁₅, h⟩ := label_bind h
  obtain ⟨-, -, t₁₆, r₁₆, kk₁₆, h⟩ := sample_bind h
  obtain ⟨-, -, t₁₇, r₁₇, kk₁₇, h⟩ := sample_bind h
  obtain ⟨s', t₁₈, hloop', h⟩ := bind_ok h
  rcases hf : st.fold.folded ((t₁₅.rounds[t₁₅.k]!).coins.getD 0 0) ((t₁₆.rounds[t₁₆.k]!).coins.getD 0 0)
      (skipWeights ((t₁₀.rounds[t₁₀.k]!).coins.getD 0 0)) (s.2.extract 0 (st.kLog - 6)) s'.2 with err | v
  · rw [hf] at h
    obtain ⟨_, _, h₀, _⟩ := bind_ok h
    cases h₀
  rw [hf] at h
  simp only [pure_bind] at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₁₉, k₁₉, h⟩ := absorbSlice_bind h
  obtain ⟨-, -, t₂₀, r₂₀, kk₂₀, h⟩ := sample_bind h
  obtain ⟨hrp, -⟩ := pure_ok h
  subst hrp
  rw [forIn_range_eq] at hloop hloop'
  obtain ⟨-, -, -, hs⟩ := ZkExec.forIn_push _ (fun i st' tt b tt' hb => by
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, nn, tt₃, rr₃, kk₃, hb⟩ := sample_bind hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      have e1 : tt₂.rounds = tt.rounds := by rw [kk₂.2, kk₁.2]
      have e2 : tt₂.k = tt.k := by rw [kk₂.1, kk₁.1]
      refine ⟨by rw [← e1, ← e2]; exact nn, by rw [rr₃, e1], by rw [kk₃, e2], ?_⟩
      rw [← e1, ← e2]
      exact ⟨_, rfl⟩) (st.m - 6) 0 _ t₁₁ t₁₂ s hloop
  obtain ⟨-, -, -, hs'⟩ := ZkExec.forIn_push _ (fun i st' tt b tt' hb => by
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, nn, tt₃, rr₃, kk₃, hb⟩ := sample_bind hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      have e1 : tt₂.rounds = tt.rounds := by rw [kk₂.2, kk₁.2]
      have e2 : tt₂.k = tt.k := by rw [kk₂.1, kk₁.1]
      refine ⟨by rw [← e1, ← e2]; exact nn, by rw [rr₃, e1], by rw [kk₃, e2], ?_⟩
      rw [← e1, ← e2]
      exact ⟨_, rfl⟩) (st.kLog - 6) 0 _ t₁₇ t₁₈ s' hloop'
  obtain ⟨-, -, -, hrun⟩ := forIn_rounds _ (lcLoopStep lay p.lincheck.rounds) 1 (fun j st' tt b tt' hb => by
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, nn, tt₃, rr₃, kk₃, hb⟩ := sample_bind hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      have e1 : tt₂.rounds = tt.rounds := by rw [kk₂.2, kk₁.2]
      have e2 : tt₂.k = tt.k := by rw [kk₂.1, kk₁.1]
      refine ⟨by rw [← e1, ← e2]; exact nn, by rw [rr₃, e1], by rw [kk₃, e2], ?_⟩
      rw [← e1, ← e2]
      rfl) (st.kLog - 6) 0 t₁₇.k _ t₁₇ t₁₈ s' (by omega) hloop'
  -- the rounds and the counter, in terms of `t`
  have hr₆ : t₆.rounds = t.rounds := by rw [k₆.2, k₅.2, k₄.2, k₃.2, k₂.2, k₁.2]
  have hk₆ : t₆.k = t.k := by rw [k₆.1, k₅.1, k₄.1, k₃.1, k₂.1, k₁.1]
  have hr₁₀ : t₁₀.rounds = t.rounds := by rw [k₁₀.2, k₉.2, r₈, r₇, hr₆]
  have hk₁₀ : t₁₀.k = t.k + 2 := by rw [k₁₀.1, k₉.1, kk₈, kk₇, hk₆]
  have hr₁₁ : t₁₁.rounds = t.rounds := by rw [r₁₁, hr₁₀]
  have hk₁₁ : t₁₁.k = t.k + 3 := by rw [kk₁₁, hk₁₀]
  have hr₁₂ : t₁₂.rounds = t.rounds ∧ t₁₂.k = t.k + 3 + (st.m - 6) := by
    obtain ⟨a, b, -, -⟩ := ZkExec.forIn_push _ (fun i st' tt b tt' hb => by
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, nn, tt₃, rr₃, kk₃, hb⟩ := sample_bind hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      have e1 : tt₂.rounds = tt.rounds := by rw [kk₂.2, kk₁.2]
      have e2 : tt₂.k = tt.k := by rw [kk₂.1, kk₁.1]
      refine ⟨by rw [← e1, ← e2]; exact nn, by rw [rr₃, e1], by rw [kk₃, e2], ?_⟩
      rw [← e1, ← e2]
      exact ⟨_, rfl⟩) (st.m - 6) 0 _ t₁₁ t₁₂ s hloop
    exact ⟨a.trans hr₁₁, by rw [b, hk₁₁]⟩
  have hr₁₅ : t₁₅.rounds = t.rounds := by rw [k₁₅.2, k₁₄.2, k₁₃.2, hr₁₂.1]
  have hk₁₅ : t₁₅.k = t.k + 3 + (st.m - 6) := by rw [k₁₅.1, k₁₄.1, k₁₃.1, hr₁₂.2]
  have hr₁₆ : t₁₆.rounds = t.rounds := by rw [r₁₆, hr₁₅]
  have hk₁₆ : t₁₆.k = t.k + 3 + (st.m - 6) + 1 := by rw [kk₁₆, hk₁₅]
  have hr₁₇ : t₁₇.rounds = t.rounds := by rw [r₁₇, hr₁₆]
  have hk₁₇ : t₁₇.k = t.k + 3 + (st.m - 6) + 2 := by rw [kk₁₇, hk₁₆]
  have hr₁₈ : t₁₈.rounds = t.rounds ∧ t₁₈.k = t.k + 3 + (st.m - 6) + 2 + (st.kLog - 6) := by
    obtain ⟨a, b, -, -⟩ := ZkExec.forIn_push _ (fun i st' tt b tt' hb => by
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, nn, tt₃, rr₃, kk₃, hb⟩ := sample_bind hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      have e1 : tt₂.rounds = tt.rounds := by rw [kk₂.2, kk₁.2]
      have e2 : tt₂.k = tt.k := by rw [kk₂.1, kk₁.1]
      refine ⟨by rw [← e1, ← e2]; exact nn, by rw [rr₃, e1], by rw [kk₃, e2], ?_⟩
      rw [← e1, ← e2]
      exact ⟨_, rfl⟩) (st.kLog - 6) 0 _ t₁₇ t₁₈ s' hloop'
    exact ⟨a.trans hr₁₇, by rw [b, hk₁₇]⟩
  have hr₁₉ : t₁₉.rounds = t.rounds := by rw [k₁₉.2, hr₁₈.1]
  have hk₁₉ : t₁₉.k = t.k + 3 + (st.m - 6) + 2 + (st.kLog - 6) := by rw [k₁₉.1, hr₁₈.2]
  -- the coins
  have hz : (t₁₀.rounds[t₁₀.k]!).coins.getD 0 0 = coinAt t 2 := by rw [hr₁₀, hk₁₀]
  have hα : (t₁₅.rounds[t₁₅.k]!).coins.getD 0 0 = coinAt t (3 + (st.m - 6)) := by
    rw [hr₁₅, hk₁₅, Nat.add_assoc]
  have hβ : (t₁₆.rounds[t₁₆.k]!).coins.getD 0 0 = coinAt t (4 + (st.m - 6)) := by
    rw [hr₁₆, hk₁₆, show t.k + 3 + (st.m - 6) + 1 = t.k + (4 + (st.m - 6)) by omega]
  have hrP : (t₁₉.rounds[t₁₉.k]!).coins.getD 0 0 = coinAt t (5 + (st.m - 6) + (st.kLog - 6)) := by
    rw [hr₁₉, hk₁₉, show t.k + 3 + (st.m - 6) + 2 + (st.kLog - 6) = t.k + (5 + (st.m - 6) + (st.kLog - 6)) by omega]
  obtain ⟨hssz, hsget⟩ := ZkExec.push_get s.2 (st.m - 6) _ hs
  obtain ⟨hs'sz, hs'get⟩ := ZkExec.push_get s'.2 (st.kLog - 6) _ hs'
  have hrho : s.2 = rhoOf st.m t := by
    refine Array.ext (by simp [rhoOf, hssz]) fun i h₁ h₂ => ?_
    rw [← getElem!_pos s.2 i h₁, hsget i (by omega), hr₁₁, hk₁₁]
    simp [rhoOf, Nat.add_assoc]
  have htsg : ∀ j < st.kLog - 6, s'.2[j]! = coinAt t (5 + (st.m - 6) + j) := fun j hj => by
    rw [hs'get j hj, hr₁₇, hk₁₇, show t.k + 3 + (st.m - 6) + 2 + j = t.k + (5 + (st.m - 6) + j) by omega]
  have hts : s'.2 = tsOf st.m st.kLog t := by
    refine Array.ext (by simp [tsOf, hs'sz]) fun i h₁ h₂ => ?_
    rw [← getElem!_pos s'.2 i h₁, htsg i (by omega)]
    simp [tsOf]
  have hrun' : s'.1 = lcRunning lay p.zerocheck.finalA p.zerocheck.finalB (coinAt t (3 + (st.m - 6)))
      (coinAt t (4 + (st.m - 6))) p.lincheck.rounds (tsOf st.m st.kLog t) (st.kLog - 6) := by
    rw [hrun, ← List.range_eq_range', hα, hβ]
    refine lcLoopStep_fold _ _ _ _ _ _ _ fun j hj => ?_
    rw [← hts, htsg j hj, hr₁₇, hk₁₇, show t.k + 3 + (st.m - 6) + 2 + j = t.k + (5 + (st.m - 6) + j) by omega]
  rw [hz, hα, hβ, hrho, hts] at hf
  refine ⟨v, hf, ?_, ?_, ?_⟩
  · show (((#[_] : Array (String × Zk.Lin)).push _).push ("lincheck final", s'.1.add (Zk.Lin.combo v
      ((Array.range 64).map fun i => Zk.Lin.masked p.lincheck.zPartial[i]! (lay.zp i) lay.len) lay.len)))[2]! = _
    rw [hrun']
    rfl
  · show Zk.Lin.combo (skipWeights ((t₁₉.rounds[t₁₉.k]!).coins.getD 0 0)) _ lay.len = _
    rw [hrP]
    rfl
  · refine (abcClaims_ab _ _ _ _ _).trans ?_
    show lcAbClaim _ _ s.2 s'.2 _ = _
    rw [hrho, hts, hrP]

/-! ## The honest `z_partial` -/

section Generic

variable {F : Type} [Field F] [Fintype F] [DecidableEq F]

/-- `abEval_eq` at any skip weights: a claim at the `ab` point evaluates through the folded columns. -/
theorem eval_abPoint (S : Statement) (zc : ZcOut F S.m) (ts : Fin (S.kLog - 6) → F) (v : F) (sk : Fin 64 → F)
    (z : Witness S.m) :
    (⟨v, sk, abPoint S zc ts⟩ : Model.Claim F S.m).eval z =
      ∑ s : Fin 64, sk s * foldAll (List.ofFn ts) (2 ^ S.kLog) (hcol S (zc.ρOut S) z) s.val := by
  have hk6 := S.hk6
  rw [claim_eval_split S _ (fun j => ts ⟨S.kLog - 6 - 1 - j.val, by omega⟩) (zc.ρOut S)
    (fun j => by simp only [abPoint, dite_eq_left j.isLt]) (fun j => ?_) z]
  · have hlen : foldAll (List.ofFn ts) (2 ^ S.kLog) = foldAll (List.ofFn ts) (2 ^ (S.kLog - 6) * 64) := by
      rw [two_pow_kLog S]
    rw [hlen, sum_block S.kLog hk6, Finset.sum_comm]
    refine Finset.sum_congr rfl fun sidx _ => ?_
    rw [foldAll_eq_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl fun widx _ => ?_
    have hmod : (sidx.val + 64 * widx.val) % 64 = sidx.val := by omega
    have hdiv : (sidx.val + 64 * widx.val) / 64 = widx.val := by omega
    have hskip : sk ⟨(sidx.val + 64 * widx.val) % 64, Nat.mod_lt _ (by norm_num)⟩ = sk sidx :=
      congrArg sk (Fin.ext hmod)
    simp only [hskip, hdiv]
    ring
  · simp only [abPoint, ZcOut.ρOut, show ¬ (S.kLog - 6 + j.val < S.kLog - 6) by omega, dite_eq_right,
      not_false_eq_true]
    rw [dite_eq_left]

/-- Address `64u + i`: skip position `i` of high address `u`. -/
def addrW {m : ℕ} (hm : 6 ≤ m) (u : Fin (2 ^ (m - 6))) (i : Fin 64) : Fin (2 ^ m) :=
  ⟨64 * u.val + i.val, by
    rw [show 2 ^ m = 2 ^ (m - 6) * 64 by rw [show (64 : ℕ) = 2 ^ 6 by norm_num, ← pow_add, Nat.sub_add_cancel hm]]
    have := u.isLt; have := i.isLt; omega⟩

omit [Fintype F] [DecidableEq F] in
/-- A claim's value is its skip weights' combination of the witness's skip positions at its point. -/
theorem eval_skip {m : ℕ} (hm : 6 ≤ m) (c : Model.Claim F m) (z : Witness m) :
    c.eval z =
      ∑ i : Fin 64, c.skip i * ∑ u : Fin (2 ^ (m - 6)), Model.eqAt c.x u.val * ((z (addrW hm u i)).val : F) := by
  classical
  have h2 : 2 ^ m = 2 ^ (m - 6) * 64 := by
    rw [show (64 : ℕ) = 2 ^ 6 by norm_num, ← pow_add, Nat.sub_add_cancel hm]
  let e : Fin (2 ^ (m - 6)) × Fin 64 ≃ Fin (2 ^ m) := finProdFinEquiv.trans (finCongr h2.symm)
  have he : ∀ q, e q = addrW hm q.1 q.2 := fun q => by
    ext; simp [e, addrW, finProdFinEquiv]; ring
  unfold Claim.eval
  rw [← e.sum_comp, Fintype.sum_prod_type, Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ => ?_
  refine Eq.trans ?_ (Finset.mul_sum _ _ _).symm
  refine Finset.sum_congr rfl fun u _ => ?_
  simp only [he]
  have hmod : (addrW hm u i).val % 64 = i.val := by simp only [addrW]; omega
  have hdiv : (addrW hm u i).val / 64 = u.val := by simp only [addrW]; omega
  have hskip : c.skip ⟨(addrW hm u i).val % 64, Nat.mod_lt _ (by norm_num)⟩ = c.skip i :=
    congrArg c.skip (Fin.ext hmod)
  rw [hskip, hdiv]
  ring

end Generic

/-- **The honest `z_partial` is the witness at the `ab` claim's point** (§10): the block-folded columns `lcV` folded at
the round coins are, at skip position `i`, `Σ_u eq(x, u)·w[64u + i]` for the `ab` claim's point `x` (the ring switch's
`zPartialOf`). -/
theorem foldAll_lcV (S : Statement) (w : Witness S.m) (rho ts : Array F128) (hrho : rho.size = S.m - 6)
    (hts : ts.size = S.kLog - 6) (rPrime : F128) (i : Fin 64) :
    foldAll (ts.toList.map fOf) (2 ^ S.kLog) (lcV S rho w) i.val =
      ∑ u : Fin (2 ^ (S.m - 6)), Model.eqAt (vecOf (S.m - 6) (lcAbClaim S.m S.kLog rho ts rPrime).x) u.val *
        ((w ⟨64 * u.val + i.val, by
          have hu := u.isLt; have hi := i.isLt; have h6 := le_trans S.hk6 S.hk
          rw [show 2 ^ S.m = 2 ^ (S.m - 6) * 64 by
            rw [show (64 : ℕ) = 2 ^ 6 by norm_num, ← pow_add, Nat.sub_add_cancel h6]]
          omega⟩).val : 𝔽) := by
  classical
  have hk6 := S.hk6
  have hkm := S.hk
  let zc : Model.ZcOut 𝔽 S.m := ⟨0, vecOf (S.m - 6) rho, vecOf (S.m - 6) rho, 0, 0, 0⟩
  let tsFin : Fin (S.kLog - 6) → 𝔽 := fun j => fOf ts[j.val]!
  have hV : lcV S rho w = hcol S (zc.ρOut S) w := by
    unfold lcV
    congr 1
    funext j
    simp only [Model.ZcOut.ρOut, zc]
    have hj : j.val < S.m - S.kLog := j.isLt
    rw [dite_eq_left (by omega)]
    rfl
  have hof : List.ofFn tsFin = ts.toList.map fOf := by
    refine List.ext_getElem (by simp [hts]) fun j h₁ h₂ => ?_
    rw [List.getElem_ofFn, List.getElem_map]
    simp only [tsFin]
    rw [getElem!_pos _ _ (by simp at h₁; omega)]
    simp
  have hxs : (ts.reverse ++ rho.extract (S.kLog - 6) (S.m - 6)).size = S.m - 6 := by
    simp [hts, hrho]; omega
  have hx : abPoint S zc tsFin = vecOf (S.m - 6) (lcAbClaim S.m S.kLog rho ts rPrime).x := by
    funext j
    simp only [abPoint, lcAbClaim, vecOf]
    rw [getElem!_pos _ j.val (by rw [hxs]; exact j.isLt), Array.getElem_append]
    by_cases hj : j.val < S.kLog - 6
    · have hj' : j.val < ts.reverse.size := by simp [hts, hj]
      rw [dite_eq_left hj, dite_eq_left hj', Array.getElem_reverse]
      simp only [tsFin]
      rw [getElem!_pos _ _ (by omega)]
      congr 2
      omega
    · have hj' : ¬ j.val < ts.reverse.size := by simp [hts, hj]
      rw [dite_eq_right hj, dite_eq_right hj', Array.getElem_extract]
      show fOf rho[j.val]! = _
      rw [getElem!_pos _ _ (by omega)]
      congr 2
      simp [hts]
      omega
  let sk : Fin 64 → 𝔽 := fun s => if s = i then 1 else 0
  have h2 := eval_skip (by omega) (⟨0, sk, abPoint S zc tsFin⟩ : Model.Claim 𝔽 S.m) w
  rw [eval_abPoint] at h2
  dsimp only at h2
  simp only [sk, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_univ, ite_true] at h2
  rw [hof, ← hV, hx] at h2
  exact h2

end FlockSoundness.Discharge.Composed.Lincheck
