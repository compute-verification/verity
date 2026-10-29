import Proofs.Flock.Soundness.Discharge.ZkLink.Analysis
import Proofs.Flock.Soundness.Discharge.ZkLink.Link
import Proofs.Flock.Soundness.LinkCarry

/-!
# `δ_link` for the compiled `--zk` session, under expected-time collision resistance

`Discharge/Exec/FlockLinkL.lean`'s link theorem for the compiled `--zk` session (`batchedSessionZC`), whose extractor
fills each drawn unit's level-0 oracle (68 lanes: the witness lanes and both reps' extra lanes) from the verifying
openings of `k` accepted reruns (`extractTableZ`). It holds at the compiled analysis `analysisZC` at any knowledge bound,
since the link term does not read it, under expected-time collision resistance of SHA-512 for an explicit finder
(`Assumptions.SHA512CRExpected`) and the binding of wires to commit strings (`ValueBindingZ`, named):

* **Readings** (`readsAtZC`): after an extraction, the values drawn units open a commit string to, from the padded
  witness each one's table settles on (`candZC`).
* **The trial** (`trialGZC`) for a commit string `c`: a draw that reads `c`, uniform among such draws; a fresh compiled
  session; if it accepts, `k` waits of at most `Rw` reruns each for a rerun in which it accepts. It reads what the
  extraction from the runs found reads at `c` (`rdTZC`), and costs `t'` per run it plays (`cTZC`), `t'(1 + k)` in
  expectation.
* **The value layer** (`XplurZC`) is the plurality of the trial's single readings at each commit string.
* **The finder** for `c` is `LinkFinder.finderG` on the trial. Its successes output collisions (`outCZC_spec`), so the
  trial's binding error at the plurality of `c` is at most the finder's collision probability plus `1/(eM)`
  (`errZC_le_prZC`), and under A2 at most `2t'(1 + k)/2^256 + 1/(eM)`.

Against the value layer of any values `Xv` (`XlayerZC`), the link term is at most
`Σ_c π_c (ε_c + k/(eR_w))/(1 − ρ)`, with `ε_c` the trial's binding error at `c` against `Xv c` (`errZC`, `LinkFinder.errAt`;
`zk_linkSoundC_at`). So a value layer other than the plurality, such as one value of a carried leaf shared by several
sampling blocks, has its link term from the trials' binding errors there.

At the plurality, `δ_link` is at most `Σ_c π_c·(Pr[c's finder outputs a collision] + 1/(eM) + k/(eR_w))/(1 − ρ)` at
every prover, with no hash premise (`zk_linkSoundPr`, `linkPrZC`). A2 is asked of one prover at a time (`LinkCRZC`), and
the closed form is `⊤` at any other (`linkBoundZC`). With `π c` the probability that a draw reads `c`, at the plurality
(`zk_linkSoundC`):
$$\delta_{\rm link}\le\frac{Q_s}{1-\rho}\Big(\frac{2(1+k)\,t'}{2^{256}}+\frac{1}{eM}+
\frac{k}{eR_w}\Big),\qquad Q_s=\sum_c\pi_c,$$
with `ρ ≥ N₀/(ek)` for every table, `N₀` its level-0 positions.
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Extract LinkFinder Audit Audit.Partition ZkSession
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]
variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

variable {L : Law n} {decode : DecoderZK (C := C) planZ tab} {Dh : Type}
  {Hc : List UInt8 → Dh} (vb : ValueBindingZ A P planZ tab decode Hc)
  (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R)

/-- The compiled session strategy on draw `ω`. -/
noncomputable abbrev stratωZC (ω : L.Ω) : Strategy (sessionZC A Hs (planZ (L.draw ω) R)) :=
  zcStrat A Hs planZ (τ ω)

/-- Everything after `Commit` on draw `ω`. -/
noncomputable abbrev gameωZC (ω : L.Ω) : Game (List (OutZC F D Sl)) :=
  sessAfterZC (gamesZC A Hs (planZ (L.draw ω) R) (stratωZC A Hs planZ R τ ω).1)

/-- The compiled session accepts. -/
abbrev accωZC (ω : L.Ω) : List (OutZC F D Sl) → Prop := acceptedZC (planZ (L.draw ω) R)

/-- The level-0 oracle extracted for drawn unit `u`'s table from the reruns `bs`. -/
noncomputable abbrev tabExtZC (ω : L.Ω) (u : Fin n) (bs : List (List (OutZC F D Sl))) : Oracle F :=
  extractTableZ Hs (tabOfZ planZ tab (L.draw ω) R u) (tab (L.draw ω) R u)
    ((stratωZC A Hs planZ R τ ω).1 (tab (L.draw ω) R u)) bs

open Classical in
/-- The padded witness the extraction settles on for unit `u`'s table: one near the extracted oracle that satisfies the
statement, when there is one. -/
noncomputable def candZC (ω : L.Ω) (u : Fin n) (bs : List (List (OutZC F D Sl))) :
    Option (Message F (tabOfZ planZ tab (L.draw ω) R u).sch.k0 (tabOfZ planZ tab (L.draw ω) R u).sch.level₀.logCols) :=
  if h : (tabOfZ planZ tab (L.draw ω) R u).Committed A (tabExtZC A Hs planZ tab R τ ω u bs) then
    some (Classical.choose h) else none

/-- An extraction opens `c` to value `v`, with the opening `o`: a drawn unit that reads `c` opens it
so from the message its table settles on. -/
def OpensAtZC (ω : L.Ω) (bs : List (List (OutZC F D Sl))) (c : vb.Pos) (v : vb.Val) (o : vb.Opening) :
    Prop :=
  ∃ u ∈ L.draw ω, ReadsZ A P planZ tab vb u c ∧ ∃ M, candZC A Hs planZ tab R τ ω u bs = some M ∧
    o = vb.opening (L.draw ω) R u M c ∧ vb.val o = v

open Classical in
/-- **The readings of commit string `c`** after an extraction from the reruns `bs`. -/
noncomputable def readsAtZC (ω : L.Ω) (bs : List (List (OutZC F D Sl))) (c : vb.Pos) : Finset vb.Val :=
  Finset.univ.filter fun v => ∃ o, OpensAtZC A Hs P planZ tab vb R τ ω bs c v o

/-- A trial's outcome: the draw, the fresh run, and the waits' runs. -/
abbrev TrialOutZC (F : Type) (L : Law n) (D Sl : Type) := L.Ω × List (OutZC F D Sl) × List (List (List (OutZC F D Sl)))

/-- **The trial for commit string `c`**: a draw that reads `c`, a fresh session, and `k` waits of at
most `Rw` reruns each. -/
noncomputable def trialGZC (c : vb.Pos) [Nonempty (ΩcZ A P planZ tab vb L c)] (k Rw : ℕ) : Game (TrialOutZC F L D Sl) :=
  (Game.draw (ΩcZ A P planZ tab vb L c)).bind fun w => (gameωZC A Hs planZ R τ w.1).bind fun fresh =>
    (runsG (runsG (gameωZC A Hs planZ R τ w.1) Rw) k).map fun bl => (w.1, fresh, bl)

/-- The prover's session strategy in every run. -/
noncomputable def trialStratZC (c : vb.Pos) [Nonempty (ΩcZ A P planZ tab vb L c)] (k Rw : ℕ) :
    Strategy (trialGZC A Hs P planZ tab vb R τ c k Rw) :=
  Strategy.bind _ _ (drawStrat _) fun w => Strategy.bind _ _ (stratωZC A Hs planZ R τ w.1).2 fun _ =>
    Strategy.map _ _ (runsStrat _ (runsStrat _ (stratωZC A Hs planZ R τ w.1).2 Rw) k)

open Classical in
/-- **What a trial reads at `c`**: nothing unless the fresh session accepts and every wait finds an
accepted rerun; then the readings of the extraction from those reruns. -/
noncomputable def rdTZC (c : vb.Pos) (e : TrialOutZC F L D Sl) : Finset vb.Val :=
  if accωZC planZ R e.1 e.2.1 then
    (firsts (accωZC planZ R e.1) e.2.2).elim ∅ fun bs => readsAtZC A Hs P planZ tab vb R τ e.1 bs c
  else ∅

open Classical in
/-- **A trial's cost**: `t'` per run it plays, the fresh one and, if it accepts, each wait's. -/
noncomputable def cTZC (t' : ℝ) (e : TrialOutZC F L D Sl) : ℝ :=
  t' * (1 + if accωZC planZ R e.1 e.2.1 then (e.2.2.map (waitCost (accωZC planZ R e.1))).sum else 0)

open Classical in
/-- An opening a trial found for value `v` at `c`, when it found one. -/
noncomputable def opOfVZC (c : vb.Pos) (e : TrialOutZC F L D Sl) (v : vb.Val) : vb.Opening :=
  Classical.epsilon fun o => accωZC planZ R e.1 e.2.1 ∧
    ∃ bs, firsts (accωZC planZ R e.1) e.2.2 = some bs ∧ OpensAtZC A Hs P planZ tab vb R τ e.1 bs c v o

open Classical in
/-- **The finder's output**: a collision from two openings of `c` with different values, the first
from its first trial, the second from that trial or from the first further trial that reads `c`. -/
noncomputable def outCZC (c : vb.Pos) (x : TrialOutZC F L D Sl × List (TrialOutZC F L D Sl)) :
    Option (List UInt8 × List UInt8) :=
  if h : ∃ p : TrialOutZC F L D Sl × vb.Val × vb.Val, p.2.1 ≠ p.2.2 ∧
      p.2.1 ∈ rdTZC A Hs P planZ tab vb R τ c x.1 ∧ p.2.2 ∈ rdTZC A Hs P planZ tab vb R τ c p.1 ∧
      (p.1 = x.1 ∨ nextRead (rdTZC A Hs P planZ tab vb R τ c) x.2 = some p.1) then
    some (vb.collide (opOfVZC A Hs P planZ tab vb R τ c x.1 (Classical.choose h).2.1)
      (opOfVZC A Hs P planZ tab vb R τ c (Classical.choose h).1 (Classical.choose h).2.2))
  else none

open Classical in
/-- **The plurality value at `c`**: the value the trial reads alone most often. -/
noncomputable def XvalZC (k Rw : ℕ) (c : vb.Pos) : vb.Val :=
  if h : Nonempty (ΩcZ A P planZ tab vb L c) then
    haveI := h
    Classical.choose (Finset.exists_max_image Finset.univ
      (pRd (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw)
        (rdTZC A Hs P planZ tab vb R τ c)) Finset.univ_nonempty)
  else default

/-- **The value layer**: each wire's bit in the plurality value of its commit string. -/
noncomputable def XplurZC (k Rw : ℕ) : Fin C.N → Bool :=
  fun g => vb.bit (XvalZC A Hs P planZ tab vb R τ k Rw (vb.pos g)) g

/-! ## Openings, and the finder's output -/

theorem cand_satZC {ω : L.Ω} {u : Fin n} {bs : List (List (OutZC F D Sl))}
    {M : Message F (tabOfZ planZ tab (L.draw ω) R u).sch.k0 (tabOfZ planZ tab (L.draw ω) R u).sch.level₀.logCols}
    (h : candZC A Hs planZ tab R τ ω u bs = some M) :
    (tabOfZ planZ tab (L.draw ω) R u).S.Satisfies
      (wit A (tabOfZ planZ tab (L.draw ω) R u).S (tabOfZ planZ tab (L.draw ω) R u).sch M) := by
  unfold candZC at h
  split_ifs at h with hc
  obtain rfl := Option.some.inj h
  exact (Classical.choose_spec hc).2

theorem opensAt_specZC {ω : L.Ω} {bs : List (List (OutZC F D Sl))} {c : vb.Pos} {v : vb.Val}
    {o : vb.Opening} (h : OpensAtZC A Hs P planZ tab vb R τ ω bs c v o) :
    vb.com o = vb.cs R c ∧ vb.val o = v := by
  obtain ⟨u, hu, ⟨g, hg, hgc⟩, M, hM, rfl, hv⟩ := h
  refine ⟨?_, hv⟩
  have h1 := (vb.opening_spec (L.draw ω) R u hu M (cand_satZC A Hs planZ tab R τ hM) g hg).1
  rw [hgc] at h1
  exact h1

theorem opOfVZC_spec {c : vb.Pos} {e : TrialOutZC F L D Sl} {v : vb.Val}
    (hv : v ∈ rdTZC A Hs P planZ tab vb R τ c e) :
    vb.com (opOfVZC A Hs P planZ tab vb R τ c e v) = vb.cs R c ∧
      vb.val (opOfVZC A Hs P planZ tab vb R τ c e v) = v := by
  classical
  unfold rdTZC at hv
  split_ifs at hv with hacc
  · cases hf : firsts (accωZC planZ R e.1) e.2.2 with
    | none => simp [hf] at hv
    | some bs =>
        simp only [hf, Option.elim_some] at hv
        unfold readsAtZC at hv
        obtain ⟨o, ho⟩ := (Finset.mem_filter.1 hv).2
        have hex : ∃ o, accωZC planZ R e.1 e.2.1 ∧ ∃ bs, firsts (accωZC planZ R e.1) e.2.2 = some bs ∧
            OpensAtZC A Hs P planZ tab vb R τ e.1 bs c v o := ⟨o, hacc, bs, hf, ho⟩
        obtain ⟨-, bs', -, hop⟩ := Classical.epsilon_spec hex
        exact opensAt_specZC A Hs P planZ tab vb R τ hop
  · simp at hv

/-- **Every success of the finder outputs a collision.** -/
theorem outCZC_spec (c : vb.Pos) (x : TrialOutZC F L D Sl × List (TrialOutZC F L D Sl))
    (hS : Succ (rdTZC A Hs P planZ tab vb R τ c) x) :
    ∃ a b, outCZC A Hs P planZ tab vb R τ c x = some (a, b) ∧ a ≠ b ∧ Hc a = Hc b :=
  outOf_spec (rdTZC A Hs P planZ tab vb R τ c) (opOfVZC A Hs P planZ tab vb R τ c) vb.collide
    (fun o => vb.com o = vb.cs R c) vb.val (fun _ _ hv => opOfVZC_spec A Hs P planZ tab vb R τ hv)
    (fun o o' h h' hv => vb.collide_spec o o' (h.trans h'.symm) hv) x hS

/-! ## The trial's expectations -/

theorem expect_trialZC (c : vb.Pos) [Nonempty (ΩcZ A P planZ tab vb L c)] (k Rw : ℕ)
    (F : TrialOutZC F L D Sl → ℝ) :
    expect F (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw) =
      (∑ w : ΩcZ A P planZ tab vb L c, expect (fun fresh =>
          expectN (runsG (gameωZC A Hs planZ R τ w.1) Rw)
            (runsStrat _ (stratωZC A Hs planZ R τ w.1).2 Rw) k fun bl => F (w.1, fresh, bl))
          (gameωZC A Hs planZ R τ w.1) (stratωZC A Hs planZ R τ w.1).2) /
        Fintype.card (ΩcZ A P planZ tab vb L c) := by
  unfold trialGZC trialStratZC
  rw [expect_bind, expect_draw]
  congr 1
  refine Finset.sum_congr rfl fun w _ => ?_
  rw [expect_bind]
  refine congrArg (fun f => expect f _ _) (funext fun fresh => ?_)
  rw [expect_map, expect_runsG]

theorem cT_nonnegZC {t' : ℝ} (ht : 0 ≤ t') (e : TrialOutZC F L D Sl) : 0 ≤ cTZC planZ R t' e := by
  classical
  unfold cTZC
  have h : 0 ≤ if accωZC planZ R e.1 e.2.1 then (e.2.2.map (waitCost (accωZC planZ R e.1))).sum
      else 0 := by
    split_ifs
    · exact List.sum_nonneg (by
        simp only [List.mem_map]
        rintro _ ⟨b, -, rfl⟩
        exact waitCost_nonneg _ b)
    · exact le_rfl
  exact mul_nonneg ht (by linarith)

/-- **A trial costs at most `t'(1 + k)` in expectation**, whatever the acceptance probability. -/
theorem expect_cT_leZC (c : vb.Pos) [Nonempty (ΩcZ A P planZ tab vb L c)] (k Rw : ℕ) {t' : ℝ}
    (ht : 0 ≤ t') :
    expect (cTZC planZ R t') (trialGZC A Hs P planZ tab vb R τ c k Rw)
      (trialStratZC A Hs P planZ tab vb R τ c k Rw) ≤ t' * (1 + k) := by
  classical
  rw [expect_trialZC, div_le_iff₀ (by exact_mod_cast Fintype.card_pos)]
  have hw : ∀ w : ΩcZ A P planZ tab vb L c, expect (fun fresh =>
      expectN (runsG (gameωZC A Hs planZ R τ w.1) Rw)
        (runsStrat _ (stratωZC A Hs planZ R τ w.1).2 Rw) k fun bl => cTZC planZ R t' (w.1, fresh, bl))
      (gameωZC A Hs planZ R τ w.1) (stratωZC A Hs planZ R τ w.1).2 ≤ t' * (1 + k) := by
    intro w
    set g := gameωZC A Hs planZ R τ w.1
    set sg := (stratωZC A Hs planZ R τ w.1).2
    set acc : List (OutZC F D Sl) → Prop := accωZC planZ R w.1
    have hblk : ∀ fresh, expectN (runsG g Rw) (runsStrat g sg Rw) k
        (fun bl => cTZC planZ R t' (w.1, fresh, bl)) =
          t' * (1 + (k * geom (expect (ind acc) g sg) Rw) * ind acc fresh) := by
      intro fresh
      by_cases ha : acc fresh
      · have e1 : (fun bl : List (List (List (OutZC F D Sl))) => cTZC planZ R t' (w.1, fresh, bl)) =
            fun bl => t' * (1 + (bl.map (waitCost acc)).sum) := by
          funext bl; simp only [cTZC, acc, ha, ↓reduceIte]
        rw [e1, expectN_const_mul, expectN_add, expectN_const, expectN_sum_map, expect_runsG,
          expectN_waitCost]
        simp only [ind, ha, ↓reduceIte, mul_one]
      · have e1 : (fun bl : List (List (List (OutZC F D Sl))) => cTZC planZ R t' (w.1, fresh, bl)) =
            fun _ => t' * 1 := by
          funext bl; simp only [cTZC, acc, ha, ↓reduceIte, add_zero]
        rw [e1, expectN_const]
        simp [ind, ha]
    rw [show (fun fresh => expectN (runsG g Rw) (runsStrat g sg Rw) k
        fun bl => cTZC planZ R t' (w.1, fresh, bl)) =
      fun fresh => t' * (1 + (k * geom (expect (ind acc) g sg) Rw) * ind acc fresh) from
        funext hblk, expect_const_mul, expect_add, expect_const, expect_const_mul]
    have hε0 : 0 ≤ expect (ind acc) g sg := expect_nonneg (ind_nonneg acc) g sg
    have hε1 : expect (ind acc) g sg ≤ 1 := expect_ind_le_one _ _ _
    have hg : expect (ind acc) g sg * geom (expect (ind acc) g sg) Rw ≤ 1 := by
      rw [mul_geom]; have := pow_nonneg (by linarith : (0 : ℝ) ≤ 1 - expect (ind acc) g sg) Rw
      linarith
    have : (k : ℝ) * geom (expect (ind acc) g sg) Rw * expect (ind acc) g sg ≤ k := by
      have hk : (0 : ℝ) ≤ k := Nat.cast_nonneg k
      nlinarith
    exact mul_le_mul_of_nonneg_left (by linarith) ht
  calc (∑ w : ΩcZ A P planZ tab vb L c, _) ≤ ∑ _w : ΩcZ A P planZ tab vb L c, t' * (1 + k) :=
        Finset.sum_le_sum fun w _ => hw w
    _ = t' * (1 + k) * Fintype.card (ΩcZ A P planZ tab vb L c) := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]; ring

open Classical in
/-- A trial's value for a function `Ψ` of the reruns: `Ψ` of the reruns its waits found, when the
fresh session accepted and every wait found one. -/
noncomputable def trialValZC (Ψ : L.Ω → List (List (OutZC F D Sl)) → ℝ) (e : TrialOutZC F L D Sl) : ℝ :=
  if accωZC planZ R e.1 e.2.1 then (firsts (accωZC planZ R e.1) e.2.2).elim 0 (Ψ e.1) else 0

theorem expect_trialValZC (c : vb.Pos) [Nonempty (ΩcZ A P planZ tab vb L c)] (k Rw : ℕ)
    (Ψ : L.Ω → List (List (OutZC F D Sl)) → ℝ) :
    expect (trialValZC planZ R Ψ) (trialGZC A Hs P planZ tab vb R τ c k Rw)
        (trialStratZC A Hs P planZ tab vb R τ c k Rw) =
      (∑ w : ΩcZ A P planZ tab vb L c,
          expect (ind (accωZC planZ R w.1)) (gameωZC A Hs planZ R τ w.1) (stratωZC A Hs planZ R τ w.1).2 *
            geom (expect (ind (accωZC planZ R w.1)) (gameωZC A Hs planZ R τ w.1)
              (stratωZC A Hs planZ R τ w.1).2) Rw ^ k *
            expectN (gameωZC A Hs planZ R τ w.1) (stratωZC A Hs planZ R τ w.1).2 k
              (fun bs => (bs.map (ind (accωZC planZ R w.1))).prod * Ψ w.1 bs)) /
        Fintype.card (ΩcZ A P planZ tab vb L c) := by
  classical
  rw [expect_trialZC]
  congr 1
  refine Finset.sum_congr rfl fun w _ => ?_
  set g := gameωZC A Hs planZ R τ w.1
  set sg := (stratωZC A Hs planZ R τ w.1).2
  set acc : List (OutZC F D Sl) → Prop := accωZC planZ R w.1
  set Xs := geom (expect (ind acc) g sg) Rw ^ k *
    expectN g sg k (fun bs => (bs.map (ind acc)).prod * Ψ w.1 bs)
  have hblk : ∀ fresh, expectN (runsG g Rw) (runsStrat g sg Rw) k
      (fun bl => trialValZC planZ R Ψ (w.1, fresh, bl)) = Xs * ind acc fresh := by
    intro fresh
    by_cases ha : acc fresh
    · have e1 : (fun bl : List (List (List (OutZC F D Sl))) => trialValZC planZ R Ψ (w.1, fresh, bl)) =
          fun bl => (firsts acc bl).elim 0 (Ψ w.1) := by
        funext bl; simp only [trialValZC, acc, ha, ↓reduceIte]
      rw [e1, expectN_firsts]
      simp [ind, ha, Xs]
    · have e1 : (fun bl : List (List (List (OutZC F D Sl))) => trialValZC planZ R Ψ (w.1, fresh, bl)) =
          fun _ => 0 := by
        funext bl; simp only [trialValZC, acc, ha, ↓reduceIte]
      rw [e1, expectN_const]
      simp [ind, ha]
  rw [show (fun fresh => expectN (runsG g Rw) (runsStrat g sg Rw) k
      fun bl => trialValZC planZ R Ψ (w.1, fresh, bl)) = fun fresh => Xs * ind acc fresh from
    funext hblk, expect_const_mul]
  simp only [Xs]
  ring

/-! ## The link mass at one commit string -/

open Classical in
/-- **The link event's trace at commit string `c`**: the extraction from the reruns `bs` reads a
value other than the plurality there. -/
noncomputable def psiZC (k Rw : ℕ) (c : vb.Pos) (ω : L.Ω) (bs : List (List (OutZC F D Sl))) : ℝ :=
  if ∃ v ∈ readsAtZC A Hs P planZ tab vb R τ ω bs c, v ≠ XvalZC A Hs P planZ tab vb R τ k Rw c then 1 else 0

theorem psi_nonnegZC (k Rw : ℕ) (c : vb.Pos) (ω : L.Ω) (bs : List (List (OutZC F D Sl))) :
    0 ≤ psiZC A Hs P planZ tab vb R τ k Rw c ω bs := by
  unfold psiZC; split_ifs <;> norm_num

theorem psi_le_oneZC (k Rw : ℕ) (c : vb.Pos) (ω : L.Ω) (bs : List (List (OutZC F D Sl))) :
    psiZC A Hs P planZ tab vb R τ k Rw c ω bs ≤ 1 := by
  unfold psiZC; split_ifs <;> norm_num

theorem Xval_maxZC (k Rw : ℕ) (c : vb.Pos) [hne : Nonempty (ΩcZ A P planZ tab vb L c)] (v : vb.Val) :
    pRd (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw)
        (rdTZC A Hs P planZ tab vb R τ c) v ≤
      pRd (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw)
        (rdTZC A Hs P planZ tab vb R τ c) (XvalZC A Hs P planZ tab vb R τ k Rw c) := by
  unfold XvalZC
  rw [dite_eq_left_of_eq_true (eq_true hne)]
  exact (Classical.choose_spec (Finset.exists_max_image Finset.univ _ Finset.univ_nonempty)).2 v
    (Finset.mem_univ v)

theorem readsAt_eq_emptyZC {c : vb.Pos} {ω : L.Ω} (h : ¬ ReadsDrawZ A P planZ tab vb c ω)
    (bs : List (List (OutZC F D Sl))) : readsAtZC A Hs P planZ tab vb R τ ω bs c = ∅ := by
  classical
  unfold readsAtZC
  refine Finset.filter_false_of_mem fun v _ => ?_
  rintro ⟨o, u, hu, hrd, -⟩
  exact h ⟨u, hu, hrd⟩

/-! ## The binding error against any value -/

/-- **The value layer of a value at each commit string**: each wire's bit in the value `Xv` gives its commit
string. The plurality value layer is the one at `XvalZC` (`XplurZC`). -/
noncomputable def XlayerZC (Xv : vb.Pos → vb.Val) : Fin C.N → Bool := fun g => vb.bit (Xv (vb.pos g)) g

open Classical in
/-- **The binding error at `c` against the value `x`**: the trial's (`LinkFinder.errAt`), the probability that it
reads two values or one other than `x`, and `0` when no draw reads `c`. -/
noncomputable def errZC (c : vb.Pos) (k Rw : ℕ) (x : vb.Val) : ℝ :=
  if h : Nonempty (ΩcZ A P planZ tab vb L c) then
    haveI := h
    errAt (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw)
      (rdTZC A Hs P planZ tab vb R τ c) x
  else 0

omit [Fintype K] in
theorem errZC_nonneg (c : vb.Pos) (k Rw : ℕ) (x : vb.Val) : 0 ≤ errZC A Hs P planZ tab vb R τ c k Rw x := by
  unfold errZC
  split_ifs
  · exact errAt_nonneg _ _ _ x
  · exact le_rfl

omit [Fintype K] in
theorem errZC_of (c : vb.Pos) [hne : Nonempty (ΩcZ A P planZ tab vb L c)] (k Rw : ℕ) (x : vb.Val) :
    errZC A Hs P planZ tab vb R τ c k Rw x =
      errAt (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw)
        (rdTZC A Hs P planZ tab vb R τ c) x := by
  unfold errZC
  rw [dite_eq_left_of_eq_true (eq_true hne)]

open Classical in
/-- **The link event's trace at commit string `c` against the value `x`**: the extraction from the reruns `bs` reads a
value other than `x` there. -/
noncomputable def psiAtZC (x : vb.Val) (c : vb.Pos) (ω : L.Ω) (bs : List (List (OutZC F D Sl))) : ℝ :=
  if ∃ v ∈ readsAtZC A Hs P planZ tab vb R τ ω bs c, v ≠ x then 1 else 0

omit [Fintype K] in
theorem psiAt_nonnegZC (x : vb.Val) (c : vb.Pos) (ω : L.Ω) (bs : List (List (OutZC F D Sl))) :
    0 ≤ psiAtZC A Hs P planZ tab vb R τ x c ω bs := by
  unfold psiAtZC; split_ifs <;> norm_num

omit [Fintype K] in
theorem psiAt_le_oneZC (x : vb.Val) (c : vb.Pos) (ω : L.Ω) (bs : List (List (OutZC F D Sl))) :
    psiAtZC A Hs P planZ tab vb R τ x c ω bs ≤ 1 := by
  unfold psiAtZC; split_ifs <;> norm_num

omit [Fintype K] in
theorem trialVal_psiAtZC (x : vb.Val) (c : vb.Pos) (e : TrialOutZC F L D Sl) :
    trialValZC planZ R (psiAtZC A Hs P planZ tab vb R τ x c) e =
      ind (fun e => ∃ v ∈ rdTZC A Hs P planZ tab vb R τ c e, v ≠ x) e := by
  classical
  unfold trialValZC rdTZC
  by_cases ha : accωZC planZ R e.1 e.2.1
  · cases hf : firsts (accωZC planZ R e.1) e.2.2 with
    | none => simp [ind, ha, hf]
    | some bs => simp [ind, ha, hf, psiAtZC]
  · simp [ind, ha]

/-- **The trace's mass in the trial is the binding error**: the trial's reruns read, at `c`, two values or one other
than `x` with probability `errAt` there. -/
theorem expect_psiAtZC (c : vb.Pos) [Nonempty (ΩcZ A P planZ tab vb L c)] (k Rw : ℕ) (x : vb.Val) :
    expect (trialValZC planZ R (psiAtZC A Hs P planZ tab vb R τ x c)) (trialGZC A Hs P planZ tab vb R τ c k Rw)
        (trialStratZC A Hs P planZ tab vb R τ c k Rw) =
      errAt (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw)
        (rdTZC A Hs P planZ tab vb R τ c) x := by
  set T := trialGZC A Hs P planZ tab vb R τ c k Rw
  set sT := trialStratZC A Hs P planZ tab vb R τ c k Rw
  set rd := rdTZC A Hs P planZ tab vb R τ c
  rw [show trialValZC planZ R (psiAtZC A Hs P planZ tab vb R τ x c) =
      fun e => ind (fun e => (rd e).Nonempty) e - ind (fun e => rd e = {x}) e from
    funext fun e => (trialVal_psiAtZC A Hs P planZ tab vb R τ x c e).trans (ind_exists_ne rd x e),
    expect_sub]
  have := rRd_eq T sT rd
  unfold rRd at this
  rw [this]
  unfold errAt pRd
  ring

/-- **The trial's binding error at the plurality**, under A2 for the finder: at most `2t'(1 + k)/2^256 + 1/(eM)`. -/
theorem trial_err_leZC (k Rw M : ℕ) (hM : 1 ≤ M) {t' : ℝ} (ht : 0 ≤ t') (c : vb.Pos)
    [Nonempty (ΩcZ A P planZ tab vb L c)]
    (hCR : Assumptions.SHA512CRExpected Hc (finderG (trialGZC A Hs P planZ tab vb R τ c k Rw) M)
      (finderStrat _ (trialStratZC A Hs P planZ tab vb R τ c k Rw) M) (outCZC A Hs P planZ tab vb R τ c)
      (finderCost (rdTZC A Hs P planZ tab vb R τ c) (cTZC planZ R t'))) :
    errAt (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw)
        (rdTZC A Hs P planZ tab vb R τ c) (XvalZC A Hs P planZ tab vb R τ k Rw c) ≤
      2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) := by
  have hmass := link_mass_le (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw)
    (rdTZC A Hs P planZ tab vb R τ c) Hc (outCZC A Hs P planZ tab vb R τ c) (cTZC planZ R t')
    (cT_nonnegZC planZ R ht) M hM (outCZC_spec A Hs P planZ tab vb R τ c) hCR
    (Xval_maxZC A Hs P planZ tab vb R τ k Rw c)
  have hcost := expect_cT_leZC A Hs P planZ tab vb R τ c k Rw ht
  have h2 : 2 * expect (cTZC planZ R t') (trialGZC A Hs P planZ tab vb R τ c k Rw)
      (trialStratZC A Hs P planZ tab vb R τ c k Rw) / 2 ^ (256 : ℝ) ≤ 2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) :=
    div_le_div_of_nonneg_right (by linarith) (by positivity)
  unfold errAt
  linarith

/-- **The binding error at the plurality**, under A2 for the finder whenever a draw reads `c`. -/
theorem errZC_le (k Rw M : ℕ) (hM : 1 ≤ M) {t' : ℝ} (ht : 0 ≤ t') (c : vb.Pos)
    (hCR : ∀ _ : Nonempty (ΩcZ A P planZ tab vb L c), Assumptions.SHA512CRExpected Hc
      (finderG (trialGZC A Hs P planZ tab vb R τ c k Rw) M)
      (finderStrat _ (trialStratZC A Hs P planZ tab vb R τ c k Rw) M) (outCZC A Hs P planZ tab vb R τ c)
      (finderCost (rdTZC A Hs P planZ tab vb R τ c) (cTZC planZ R t'))) :
    errZC A Hs P planZ tab vb R τ c k Rw (XvalZC A Hs P planZ tab vb R τ k Rw c) ≤
      2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) := by
  by_cases hne : Nonempty (ΩcZ A P planZ tab vb L c)
  · rw [errZC_of]
    exact trial_err_leZC A Hs P planZ tab vb R τ k Rw M hM ht c (hCR hne)
  · unfold errZC
    rw [dite_eq_right_of_eq_false (eq_false hne)]
    positivity

open Classical in
/-- **The link finder's collision probability at `c`**: the probability that the finder built on the trial for `c`
outputs a collision of `Hc` (`0` at a commit string no draw reads, which has no trial). -/
noncomputable def linkFinderPrZC (c : vb.Pos) (k Rw M : ℕ) : ℝ :=
  if _ : Nonempty (ΩcZ A P planZ tab vb L c) then
    (prob (fun x => ∃ a b, outCZC A Hs P planZ tab vb R τ c x = some (a, b) ∧ a ≠ b ∧ Hc a = Hc b)
      (finderG (trialGZC A Hs P planZ tab vb R τ c k Rw) M)
      (finderStrat _ (trialStratZC A Hs P planZ tab vb R τ c k Rw) M)).toReal
  else 0

theorem linkFinderPrZC_nonneg (c : vb.Pos) (k Rw M : ℕ) : 0 ≤ linkFinderPrZC A Hs P planZ tab vb R τ c k Rw M := by
  unfold linkFinderPrZC
  split_ifs
  · exact ENNReal.toReal_nonneg
  · exact le_rfl

/-- **The binding error at the plurality of `c`** at its finder's collision probability: at most
`Pr[the finder outputs a collision of Hc] + 1/(eM)`. -/
theorem errZC_le_prZC (k Rw M : ℕ) (hM : 1 ≤ M) (c : vb.Pos) :
    errZC A Hs P planZ tab vb R τ c k Rw (XvalZC A Hs P planZ tab vb R τ k Rw c) ≤
      linkFinderPrZC A Hs P planZ tab vb R τ c k Rw M + 1 / (Real.exp 1 * M) := by
  by_cases hne : Nonempty (ΩcZ A P planZ tab vb L c)
  · rw [errZC_of]
    unfold linkFinderPrZC
    rw [dif_pos hne]
    exact link_mass_le_pr (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw)
      (rdTZC A Hs P planZ tab vb R τ c) Hc (outCZC A Hs P planZ tab vb R τ c) M hM
      (outCZC_spec A Hs P planZ tab vb R τ c) (Xval_maxZC A Hs P planZ tab vb R τ k Rw c)
  · unfold errZC linkFinderPrZC
    rw [dite_eq_right_of_eq_false (eq_false hne), dite_eq_right_of_eq_false (eq_false hne)]
    positivity

/-- **The link finder's collision probability under A2**: at a commit string whose finder satisfies
`Assumptions.SHA512CRExpected`, at most `2t'(1 + k)/2^256`. -/
theorem linkFinderPrZC_le (k Rw M : ℕ) {t' : ℝ} (ht : 0 ≤ t') (c : vb.Pos)
    (hCR : ∀ _ : Nonempty (ΩcZ A P planZ tab vb L c), Assumptions.SHA512CRExpected Hc
      (finderG (trialGZC A Hs P planZ tab vb R τ c k Rw) M)
      (finderStrat _ (trialStratZC A Hs P planZ tab vb R τ c k Rw) M) (outCZC A Hs P planZ tab vb R τ c)
      (finderCost (rdTZC A Hs P planZ tab vb R τ c) (cTZC planZ R t'))) :
    linkFinderPrZC A Hs P planZ tab vb R τ c k Rw M ≤ 2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) := by
  unfold linkFinderPrZC
  split_ifs with hne
  · have h3 := hCR hne
    unfold Assumptions.SHA512CRExpected at h3
    have h4 := expect_finderCost_le (trialGZC A Hs P planZ tab vb R τ c k Rw)
      (trialStratZC A Hs P planZ tab vb R τ c k Rw) (rdTZC A Hs P planZ tab vb R τ c) (cTZC planZ R t')
      (cT_nonnegZC planZ R ht) M
    have hcost := expect_cT_leZC A Hs P planZ tab vb R τ c k Rw ht
    have h5 : expect (finderCost (rdTZC A Hs P planZ tab vb R τ c) (cTZC planZ R t'))
        (finderG (trialGZC A Hs P planZ tab vb R τ c k Rw) M)
        (finderStrat _ (trialStratZC A Hs P planZ tab vb R τ c k Rw) M) / 2 ^ (256 : ℝ) ≤
        2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) :=
      div_le_div_of_nonneg_right (by linarith) (by positivity)
    linarith
  · positivity

/-- **The ideal link mass at `c` against the value `x`**: the audit extractor's reading of `c` other than `x`,
jointly with acceptance and averaged over the draw, is at most `π c` times the binding error there plus `k/(eRw)` for
the waits that give up. -/
theorem ideal_mass_le_atZC (x : vb.Val) (k Rw : ℕ) (hRw : 1 ≤ Rw) (c : vb.Pos) :
    (∑ ω : L.Ω, expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 *
        expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
          (psiAtZC A Hs P planZ tab vb R τ x c ω)) / Fintype.card L.Ω ≤
      (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω *
        (errZC A Hs P planZ tab vb R τ c k Rw x + k / (Real.exp 1 * Rw)) := by
  classical
  set E := errZC A Hs P planZ tab vb R τ c k Rw x with hE
  set f : L.Ω → ℝ := fun ω => expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω)
      (stratωZC A Hs planZ R τ ω).2 *
    expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
      (psiAtZC A Hs P planZ tab vb R τ x c ω) with hf
  have hΩ : (0 : ℝ) < Fintype.card L.Ω := by exact_mod_cast Fintype.card_pos
  have hB : 0 ≤ E + k / (Real.exp 1 * Rw) := add_nonneg (errZC_nonneg A Hs P planZ tab vb R τ c k Rw x) (by positivity)
  -- off the draws that read `c`, the trace vanishes
  have hoff : ∀ ω, ¬ ReadsDrawZ A P planZ tab vb c ω → f ω = 0 := by
    intro ω hω
    have hpsi : psiAtZC A Hs P planZ tab vb R τ x c ω = fun _ => 0 := funext fun bs => by
      simp [psiAtZC, readsAt_eq_emptyZC A Hs P planZ tab vb R τ hω bs]
    simp only [hf, hpsi, expectAcc_zero, mul_zero]
  have hsum : ∑ ω : L.Ω, f ω = ∑ w : ΩcZ A P planZ tab vb L c, f w.1 := by
    rw [← Finset.sum_filter_of_ne (p := fun ω => ReadsDrawZ A P planZ tab vb c ω)
      (fun ω _ hne => by by_contra h; exact hne (hoff ω h))]
    exact Finset.sum_subtype _ (fun ω => by simp) f
  rw [hsum, div_le_iff₀ hΩ]
  by_cases hne : Nonempty (ΩcZ A P planZ tab vb L c)
  · have hw : ∀ w : ΩcZ A P planZ tab vb L c, f w.1 ≤
        expect (ind (accωZC planZ R w.1)) (gameωZC A Hs planZ R τ w.1) (stratωZC A Hs planZ R τ w.1).2 *
            geom (expect (ind (accωZC planZ R w.1)) (gameωZC A Hs planZ R τ w.1)
              (stratωZC A Hs planZ R τ w.1).2) Rw ^ k *
            expectN (gameωZC A Hs planZ R τ w.1) (stratωZC A Hs planZ R τ w.1).2 k
              (fun bs => (bs.map (ind (accωZC planZ R w.1))).prod *
                psiAtZC A Hs P planZ tab vb R τ x c w.1 bs) + k / (Real.exp 1 * Rw) := fun w =>
      expectAcc_le_waits _ _ _ k Rw hRw _ (psiAt_nonnegZC A Hs P planZ tab vb R τ x c w.1)
        (psiAt_le_oneZC A Hs P planZ tab vb R τ x c w.1)
    have htr := expect_trialValZC A Hs P planZ tab vb R τ c k Rw (psiAtZC A Hs P planZ tab vb R τ x c)
    have hmass : expect (trialValZC planZ R (psiAtZC A Hs P planZ tab vb R τ x c))
        (trialGZC A Hs P planZ tab vb R τ c k Rw) (trialStratZC A Hs P planZ tab vb R τ c k Rw) ≤ E := by
      rw [expect_psiAtZC, hE, errZC_of]
    have hcard : (0 : ℝ) < Fintype.card (ΩcZ A P planZ tab vb L c) := by exact_mod_cast Fintype.card_pos
    rw [htr, div_le_iff₀ hcard] at hmass
    calc ∑ w : ΩcZ A P planZ tab vb L c, f w.1
        ≤ ∑ w : ΩcZ A P planZ tab vb L c, (expect (ind (accωZC planZ R w.1)) (gameωZC A Hs planZ R τ w.1)
              (stratωZC A Hs planZ R τ w.1).2 *
            geom (expect (ind (accωZC planZ R w.1)) (gameωZC A Hs planZ R τ w.1)
              (stratωZC A Hs planZ R τ w.1).2) Rw ^ k *
            expectN (gameωZC A Hs planZ R τ w.1) (stratωZC A Hs planZ R τ w.1).2 k
              (fun bs => (bs.map (ind (accωZC planZ R w.1))).prod *
                psiAtZC A Hs P planZ tab vb R τ x c w.1 bs) + k / (Real.exp 1 * Rw)) :=
          Finset.sum_le_sum fun w _ => hw w
      _ = (∑ w : ΩcZ A P planZ tab vb L c, expect (ind (accωZC planZ R w.1)) (gameωZC A Hs planZ R τ w.1)
              (stratωZC A Hs planZ R τ w.1).2 *
            geom (expect (ind (accωZC planZ R w.1)) (gameωZC A Hs planZ R τ w.1)
              (stratωZC A Hs planZ R τ w.1).2) Rw ^ k *
            expectN (gameωZC A Hs planZ R τ w.1) (stratωZC A Hs planZ R τ w.1).2 k
              (fun bs => (bs.map (ind (accωZC planZ R w.1))).prod *
                psiAtZC A Hs P planZ tab vb R τ x c w.1 bs)) +
          Fintype.card (ΩcZ A P planZ tab vb L c) * (k / (Real.exp 1 * Rw)) := by
        rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
      _ ≤ E * Fintype.card (ΩcZ A P planZ tab vb L c) +
          Fintype.card (ΩcZ A P planZ tab vb L c) * (k / (Real.exp 1 * Rw)) := by linarith
      _ = _ := by field_simp
  · have hempty : ∀ w : ΩcZ A P planZ tab vb L c, False := fun w => hne ⟨w⟩
    have h0 : (∑ w : ΩcZ A P planZ tab vb L c, f w.1) = 0 :=
      Finset.sum_eq_zero fun w _ => (hempty w).elim
    rw [h0]
    have : 0 ≤ (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω * (E + k / (Real.exp 1 * Rw)) :=
      mul_nonneg (div_nonneg (Nat.cast_nonneg _) hΩ.le) hB
    nlinarith

/-- **The ideal link mass at `c`**: the audit extractor's reading of `c` other than the
plurality, jointly with acceptance and averaged over the draw, is at most `π c` times the trial's
bound plus `k/(eRw)` for the waits that give up. -/
theorem ideal_mass_leZC (k Rw M : ℕ) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {t' : ℝ} (ht : 0 ≤ t')
    (c : vb.Pos)
    (hCR : ∀ _ : Nonempty (ΩcZ A P planZ tab vb L c), Assumptions.SHA512CRExpected Hc
      (finderG (trialGZC A Hs P planZ tab vb R τ c k Rw) M)
      (finderStrat _ (trialStratZC A Hs P planZ tab vb R τ c k Rw) M) (outCZC A Hs P planZ tab vb R τ c)
      (finderCost (rdTZC A Hs P planZ tab vb R τ c) (cTZC planZ R t'))) :
    (∑ ω : L.Ω, expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 *
        expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
          (psiZC A Hs P planZ tab vb R τ k Rw c ω)) / Fintype.card L.Ω ≤
      (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω *
        (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) := by
  refine (ideal_mass_le_atZC A Hs P planZ tab vb R τ (XvalZC A Hs P planZ tab vb R τ k Rw c) k Rw hRw c).trans ?_
  refine mul_le_mul_of_nonneg_left ?_ (div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _))
  linarith [errZC_le A Hs P planZ tab vb R τ k Rw M hM ht c hCR]

/-- **The ideal link mass at `c`** at the finder's collision probability (`ideal_mass_le_atZC` at the plurality, `errZC_le_prZC`). -/
theorem ideal_mass_le_prZC (k Rw M : ℕ) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) (c : vb.Pos) :
    (∑ ω : L.Ω, expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 *
        expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
          (psiZC A Hs P planZ tab vb R τ k Rw c ω)) / Fintype.card L.Ω ≤
      (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω *
        (linkFinderPrZC A Hs P planZ tab vb R τ c k Rw M + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) := by
  refine (ideal_mass_le_atZC A Hs P planZ tab vb R τ (XvalZC A Hs P planZ tab vb R τ k Rw c) k Rw hRw c).trans ?_
  refine mul_le_mul_of_nonneg_left ?_ (div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _))
  linarith [errZC_le_prZC A Hs P planZ tab vb R τ k Rw M hM c]

/-! ## The link theorem -/

/-- **The link event leaves a trace**: if drawn unit `u`'s extracted table is committed but every
close satisfying message decodes, on `u`'s wires, differently from the value layer of `Xv`, then some commit
string `c` reads a value other than `Xv c`. -/
theorem linkEv_le_psiAtZC (Xv : vb.Pos → vb.Val) {ω : L.Ω} {u : Fin n} (hu : u ∈ L.draw ω)
    (bs : List (List (OutZC F D Sl))) :
    ind (fun bs => LinkEvZK A P planZ tab decode (L.draw ω) R u (XlayerZC A P planZ tab vb Xv)
      (tabExtZC A Hs planZ tab R τ ω u bs)) bs ≤ ∑ c, psiAtZC A Hs P planZ tab vb R τ (Xv c) c ω bs := by
  classical
  have hsum0 : 0 ≤ ∑ c, psiAtZC A Hs P planZ tab vb R τ (Xv c) c ω bs :=
    Finset.sum_nonneg fun c _ => psiAt_nonnegZC A Hs P planZ tab vb R τ (Xv c) c ω bs
  by_cases hL : LinkEvZK A P planZ tab decode (L.draw ω) R u (XlayerZC A P planZ tab vb Xv)
      (tabExtZC A Hs planZ tab R τ ω u bs)
  · obtain ⟨hc, hall⟩ := hL
    have hspec := Classical.choose_spec hc
    obtain ⟨g, hg, hdec⟩ := hall (Classical.choose hc) hspec.1 hspec.2
    have hcand : candZC A Hs planZ tab R τ ω u bs = some (Classical.choose hc) := by
      unfold candZC; rw [dite_eq_left_of_eq_true (eq_true hc)]
    have hop : OpensAtZC A Hs P planZ tab vb R τ ω bs (vb.pos g)
        (vb.val (vb.opening (L.draw ω) R u (Classical.choose hc) (vb.pos g)))
        (vb.opening (L.draw ω) R u (Classical.choose hc) (vb.pos g)) :=
      ⟨u, hu, ⟨g, hg, rfl⟩, _, hcand, rfl, rfl⟩
    have hmem : vb.val (vb.opening (L.draw ω) R u (Classical.choose hc) (vb.pos g)) ∈
        readsAtZC A Hs P planZ tab vb R τ ω bs (vb.pos g) := by
      unfold readsAtZC; exact Finset.mem_filter.2 ⟨Finset.mem_univ _, _, hop⟩
    have hbit := (vb.opening_spec (L.draw ω) R u hu (Classical.choose hc) hspec.2 g hg).2
    have hne : vb.val (vb.opening (L.draw ω) R u (Classical.choose hc) (vb.pos g)) ≠ Xv (vb.pos g) := by
      intro heq
      apply hdec
      rw [← hbit, heq]
      rfl
    have hpsi : psiAtZC A Hs P planZ tab vb R τ (Xv (vb.pos g)) (vb.pos g) ω bs = 1 := by
      unfold psiAtZC; rw [ite_eq_left ⟨_, hmem, hne⟩]
    have h1 : ind (fun bs => LinkEvZK A P planZ tab decode (L.draw ω) R u
        (XlayerZC A P planZ tab vb Xv) (tabExtZC A Hs planZ tab R τ ω u bs)) bs = 1 := by
      unfold ind; rw [ite_eq_left ⟨hc, hall⟩]
    rw [h1, ← hpsi]
    exact Finset.single_le_sum (f := fun c => psiAtZC A Hs P planZ tab vb R τ (Xv c) c ω bs)
      (fun c _ => psiAt_nonnegZC A Hs P planZ tab vb R τ (Xv c) c ω bs) (Finset.mem_univ (vb.pos g))
  · have h0 : ind (fun bs => LinkEvZK A P planZ tab decode (L.draw ω) R u
        (XlayerZC A P planZ tab vb Xv) (tabExtZC A Hs planZ tab R τ ω u bs)) bs = 0 := by
      unfold ind; rw [ite_eq_right hL]
    rw [h0]
    exact hsum0

/-- **A2 for one prover's link finders** (`Assumptions.SHA512CRExpected`): at the registration `R` and the
prover's continuation `τ`, for the finder built for each commit string a draw can read. A2 is a hypothesis about one
finder, since some finder outputs a collision after no evaluations: a prover that hard-codes two openings of one commit
string has link finders that find one. So the link theorem asks A2 only of the finders of the prover it bounds.

Assumption record. Claim: `ecr/sha-512` (`Claims.ExpectedCR`). It assumes that no link finder built from this prover
finds a SHA-512 collision with probability above its expected SHA-512 evaluations over `2^256`. That is the generic
bound for a hash with 512-bit outputs, and no SHA-512 collision is known. A hash premise: it is
`Assumptions.SHA512CRExpected` at each commit string's finder, so it reduces to `ExpectedCR` and asks nothing of the
verifier. -/
def LinkCRZC (k Rw M : ℕ) (t' : ℝ) : Prop :=
  ∀ (c : vb.Pos) (_ : Nonempty (ΩcZ A P planZ tab vb L c)), Assumptions.SHA512CRExpected Hc
    (finderG (trialGZC A Hs P planZ tab vb R τ c k Rw) M)
    (finderStrat _ (trialStratZC A Hs P planZ tab vb R τ c k Rw) M) (outCZC A Hs P planZ tab vb R τ c)
    (finderCost (rdTZC A Hs P planZ tab vb R τ c) (cTZC planZ R t'))

open Classical in
/-- **The link bound, prover by prover**: `Q_s/(1 − ρ)·(2t'(1 + k)/2^256 + 1/(eM) + k/(eR_w))` at a prover whose link
finders satisfy A2 (`LinkCRZC`), and `⊤` at any other. -/
noncomputable def linkBoundZC (k Rw M : ℕ) (ρ t' : ℝ) (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R) :
    ℝ≥0∞ :=
  if LinkCRZC A Hs P planZ tab vb R τ k Rw M t' then
    ENNReal.ofReal ((∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω) *
      (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))
  else ⊤

/-- **The link bound against the value layer of `Xv`, prover by prover**: `Σ_c π_c (ε_c + k/(eR_w))/(1 − ρ)`, with
`ε_c` the binding error at `c` against `Xv R τ c` (`errZC`). -/
noncomputable def errBoundZC (Xv : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → vb.Pos → vb.Val)
    (k Rw : ℕ) (ρ : ℝ) (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R) : ℝ≥0∞ :=
  ENNReal.ofReal ((∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω *
    (errZC A Hs P planZ tab vb R τ c k Rw (Xv R τ c) + k / (Real.exp 1 * Rw))) / (1 - ρ))

/-- **The link theorem against any value layer.** For the batched audit with the expected-time extractor
(`analysisZC`, at any knowledge bound) and the value layer of any values `Xv` (`XlayerZC`), given the binding of wires
to commit strings (`ValueBindingZ`), at every prover
$$\delta_{\rm link}\le\frac{1}{1-\rho}\sum_c\pi_c\Big(\varepsilon_c+\frac{k}{eR_w}\Big),$$
with `ε_c` the trial's binding error at `c` against `Xv c` (`errZC`), `π_c` the probability that a draw reads `c`,
`ρ` a bound on every table's `N₀/(ek)` and `Rw` the waits' length. -/
theorem zk_linkSoundC_at
    (hlow : LoweringSoundZK A P planZ tab decode) {k : ℕ} {Rw : ℕ} (hRw : 1 ≤ Rw)
    {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a)
    (Xv : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → vb.Pos → vb.Val) :
    (analysisZC (L := L) A Hs P planZ tab decode (fun R τ => XlayerZC A P planZ tab vb (Xv R τ)) k
        (fun S R j => (hr S R j).trans_lt hρ) hlow a hj).LinkSound
      (errBoundZC A Hs P planZ tab vb Xv k Rw ρ) := by
  intro R τ tgt
  unfold errBoundZC
  have hρ1 : 0 < 1 - ρ := by linarith
  set f : vb.Pos → L.Ω → ℝ := fun c ω => expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω)
      (stratωZC A Hs planZ R τ ω).2 *
    expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
      (psiAtZC A Hs P planZ tab vb R τ (Xv R τ c) c ω) with hf
  set Λ : L.Ω → ℝ := fun ω => (∑ c, f c ω) / (1 - ρ) with hΛ
  have hf0 : ∀ c ω, 0 ≤ f c ω := fun c ω => mul_nonneg (expect_nonneg (ind_nonneg _) _ _)
    (expectAcc_nonneg _ _ _ _ (psiAt_nonnegZC A Hs P planZ tab vb R τ (Xv R τ c) c ω))
  have hΛ0 : ∀ ω, 0 ≤ Λ ω := fun ω =>
    div_nonneg (Finset.sum_nonneg fun c _ => hf0 c ω) hρ1.le
  have step1 : ∀ ω, atTgt (L.draw ω) (tgt ω) (fun u =>
      (analysisZC (L := L) A Hs P planZ tab decode (fun R τ => XlayerZC A P planZ tab vb (Xv R τ)) k
        (fun S R j => (hr S R j).trans_lt hρ) hlow a hj).link (L.draw ω) R (τ ω) u
        ((analysisZC (L := L) A Hs P planZ tab decode (fun R τ => XlayerZC A P planZ tab vb (Xv R τ)) k
          (fun S R j => (hr S R j).trans_lt hρ) hlow a hj).committed R τ)) ≤ ENNReal.ofReal (Λ ω) := by
    intro ω
    refine atTgt_le (fun u hu => ?_) (tgt ω)
    show ENNReal.ofReal (accProbZC A Hs (planZ (L.draw ω) R) (zcStrat A Hs planZ (τ ω)) *
        extractProbAccZ A Hs (planZ (L.draw ω) R) (tab (L.draw ω) R u) k
          (zcStrat A Hs planZ (τ ω))
          (LinkEvZK A P planZ tab decode (L.draw ω) R u (XlayerZC A P planZ tab vb (Xv R τ))) /
        (1 - rateZ (planZ (L.draw ω) R) (tab (L.draw ω) R u) k)) ≤ ENNReal.ofReal (Λ ω)
    apply ENNReal.ofReal_le_ofReal
    have hε0 : 0 ≤ accProbZC A Hs (planZ (L.draw ω) R) (zcStrat A Hs planZ (τ ω)) := expect_nonneg (ind_nonneg _) _ _
    have hLp0 := extractProbAccZ_nonneg A Hs (planZ (L.draw ω) R) (tab (L.draw ω) R u) k
      (zcStrat A Hs planZ (τ ω))
      (LinkEvZK A P planZ tab decode (L.draw ω) R u (XlayerZC A P planZ tab vb (Xv R τ)))
    have hLp : extractProbAccZ A Hs (planZ (L.draw ω) R) (tab (L.draw ω) R u) k
        (zcStrat A Hs planZ (τ ω))
        (LinkEvZK A P planZ tab decode (L.draw ω) R u (XlayerZC A P planZ tab vb (Xv R τ))) ≤
          expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
            (fun bs => ∑ c, psiAtZC A Hs P planZ tab vb R τ (Xv R τ c) c ω bs) :=
      expectAcc_mono _ _ _ _ fun bs => linkEv_le_psiAtZC A Hs P planZ tab vb R τ (Xv R τ) hu bs
    have hrr : 1 - ρ ≤ 1 - rateZ (planZ (L.draw ω) R) (tab (L.draw ω) R u) k := by
      linarith [hr (L.draw ω) R (tab (L.draw ω) R u)]
    have hsum : expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 *
        expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
          (fun bs => ∑ c, psiAtZC A Hs P planZ tab vb R τ (Xv R τ c) c ω bs) = ∑ c, f c ω := by
      rw [expectAcc_sum, Finset.mul_sum]
    calc accProbZC A Hs (planZ (L.draw ω) R) (zcStrat A Hs planZ (τ ω)) *
          extractProbAccZ A Hs (planZ (L.draw ω) R) (tab (L.draw ω) R u) k
            (zcStrat A Hs planZ (τ ω))
            (LinkEvZK A P planZ tab decode (L.draw ω) R u (XlayerZC A P planZ tab vb (Xv R τ))) /
          (1 - rateZ (planZ (L.draw ω) R) (tab (L.draw ω) R u) k)
        ≤ accProbZC A Hs (planZ (L.draw ω) R) (zcStrat A Hs planZ (τ ω)) *
          extractProbAccZ A Hs (planZ (L.draw ω) R) (tab (L.draw ω) R u) k
            (zcStrat A Hs planZ (τ ω))
            (LinkEvZK A P planZ tab decode (L.draw ω) R u (XlayerZC A P planZ tab vb (Xv R τ))) /
          (1 - ρ) := div_le_div_of_nonneg_left (mul_nonneg hε0 hLp0) hρ1 hrr
      _ ≤ expect (ind (accωZC planZ R ω)) (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 *
          expectAcc (gameωZC A Hs planZ R τ ω) (stratωZC A Hs planZ R τ ω).2 (accωZC planZ R ω) k
            (fun bs => ∑ c, psiAtZC A Hs P planZ tab vb R τ (Xv R τ c) c ω bs) / (1 - ρ) :=
          div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hLp hε0) hρ1.le
      _ = Λ ω := by rw [hsum]
  refine (avg_mono step1).trans ?_
  rw [avg_ofReal Λ hΛ0]
  apply ENNReal.ofReal_le_ofReal
  have hc : ∀ c, (∑ ω, f c ω) / Fintype.card L.Ω ≤
      (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω *
        (errZC A Hs P planZ tab vb R τ c k Rw (Xv R τ c) + k / (Real.exp 1 * Rw)) := fun c =>
    ideal_mass_le_atZC A Hs P planZ tab vb R τ (Xv R τ c) k Rw hRw c
  calc (∑ ω, Λ ω) / Fintype.card L.Ω
      = (∑ c, (∑ ω, f c ω) / Fintype.card L.Ω) / (1 - ρ) := by
        simp only [hΛ, div_eq_mul_inv, ← Finset.sum_mul]
        rw [Finset.sum_comm]
        ring
    _ ≤ _ := div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ => hc c) hρ1.le

/-- **The link bound at the link finders' collision probabilities**, prover by prover:
`Σ_c π_c·(Pr[c's finder outputs a collision] + 1/(eM) + k/(eR_w))/(1 − ρ)`, with `π_c` the probability that a draw
reads commit string `c` and `Pr[…]` its finder's (`linkFinderPrZC`). -/
noncomputable def linkPrZC (k Rw M : ℕ) (ρ : ℝ) (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R) : ℝ≥0∞ :=
  ENNReal.ofReal ((∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω *
    (linkFinderPrZC A Hs P planZ tab vb R τ c k Rw M + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw))) / (1 - ρ))

/-- **`linkPrZC` under A2**: at most `linkBoundZC`, which is its closed form at a prover whose link finders satisfy A2
(`LinkCRZC`, `linkFinderPrZC_le`) and `⊤` at any other. -/
theorem linkPrZC_le_linkBoundZC {k Rw M : ℕ} {ρ : ℝ} (hρ : ρ < 1) {t' : ℝ} (ht : 0 ≤ t') (R : Reg)
    (τ : Cont L Reg (batchedSessionZC A Hs planZ) R) :
    linkPrZC A Hs P planZ tab vb k Rw M ρ R τ ≤ linkBoundZC A Hs P planZ tab vb k Rw M ρ t' R τ := by
  unfold linkBoundZC
  split_ifs with hCR
  · refine ENNReal.ofReal_le_ofReal (div_le_div_of_nonneg_right ?_ (by linarith))
    rw [Finset.sum_mul]
    exact Finset.sum_le_sum fun c _ => mul_le_mul_of_nonneg_left
      (by linarith [linkFinderPrZC_le A Hs P planZ tab vb R τ k Rw M ht c (hCR c)]) (by positivity)
  · exact le_top

/-- **`linkPrZC` in closed form under A2**: at a prover whose link finders satisfy A2 (`LinkCRZC`),
`linkPrZC ≤ Q_s/(1 − ρ)·(2t'(1 + k)/2^256 + 1/(eM) + k/(eR_w))`. -/
theorem linkPrZC_le_closed {k Rw M : ℕ} {ρ : ℝ} (hρ : ρ < 1) {t' : ℝ} (ht : 0 ≤ t') (R : Reg)
    (τ : Cont L Reg (batchedSessionZC A Hs planZ) R) (hA2 : LinkCRZC A Hs P planZ tab vb R τ k Rw M t') :
    linkPrZC A Hs P planZ tab vb k Rw M ρ R τ ≤
      ENNReal.ofReal ((∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω) *
        (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) := by
  have h := linkPrZC_le_linkBoundZC A Hs P planZ tab vb (k := k) (Rw := Rw) (M := M) hρ ht R τ
  rwa [linkBoundZC, if_pos hA2] at h

/-- **The link theorem: `δ_link` at the link finders' collision probabilities, prover by prover.** For the batched
audit with the expected-time extractor (`analysisZC`, at any knowledge bound) and the plurality value layer (`XplurZC`),
given the binding of wires to commit strings (`ValueBindingZ`), at every prover,
$$\delta_{\rm link}\le\frac{1}{1-\rho}\sum_c\pi_c\Big(\Pr[B_c\text{ outputs a collision}]+\frac{1}{eM}+
\frac{k}{eR_w}\Big),$$
with `B_c` the explicit finder built from the prover for commit string `c` (`linkFinderPrZC`), `π_c` the probability
that a draw reads `c`, `ρ` a bound on every table's `N₀/(ek)`, `M` the finder's further trials and `Rw` the waits'
length (`linkPrZC`). It is `zk_linkSoundC_at` at the plurality, each binding error bounded by its finder's collision
probability (`errZC_le_prZC`). No hash premise: A2 enters only in `linkPrZC_le_linkBoundZC`. -/
theorem zk_linkSoundPr
    (hlow : LoweringSoundZK A P planZ tab decode) {k : ℕ} {Rw M : ℕ} (hRw : 1 ≤ Rw)
    (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) :
    (analysisZC (L := L) A Hs P planZ tab decode (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
        (fun S R j => (hr S R j).trans_lt hρ) hlow a hj).LinkSound
      (linkPrZC A Hs P planZ tab vb k Rw M ρ) := by
  intro R τ tgt
  refine (zk_linkSoundC_at A Hs P planZ tab vb hlow hRw hρ hr hj
    (fun R τ => XvalZC A Hs P planZ tab vb R τ k Rw) R τ tgt).trans ?_
  unfold errBoundZC linkPrZC
  apply ENNReal.ofReal_le_ofReal
  have hρ1 : 0 < 1 - ρ := by linarith
  refine div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ => ?_) hρ1.le
  refine mul_le_mul_of_nonneg_left ?_ (div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _))
  linarith [errZC_le_prZC A Hs P planZ tab vb R τ k Rw M hM c]

/-- **The link theorem: `δ_link` under expected-time collision resistance, prover by prover.** For the batched audit
with the expected-time extractor (`analysisZC`, at any knowledge bound) and the plurality value layer (`XplurZC`), given
the binding of wires to commit strings (`ValueBindingZ`): at every prover whose explicit finder of each commit string satisfies A2 (`LinkCRZC`),
$$\delta_{\rm link}\le\frac{Q_s}{1-\rho}\Big(\frac{2t'(1+k)}{2^{256}}+\frac{1}{eM}+\frac{k}{eR_w}\Big),$$
with `Q_s = Σ_c π_c` the expected number of commit strings the drawn units read, `ρ` a bound on every table's `N₀/(ek)`,
`t'` the SHA-512 evaluations of one session run, `M` the finder's further trials and `Rw` the waits' length. At any other
prover the bound is `⊤` (`linkBoundZC`). It is `zk_linkSoundC_at` at the plurality, each binding error bounded under
A2 (`errZC_le`). -/
theorem zk_linkSoundC
    (hlow : LoweringSoundZK A P planZ tab decode) {k : ℕ} {Rw M : ℕ} (hRw : 1 ≤ Rw)
    (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateZ (planZ S R) j k ≤ ρ)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) {t' : ℝ} (ht : 0 ≤ t') :
    (analysisZC (L := L) A Hs P planZ tab decode (fun R τ => XplurZC A Hs P planZ tab vb R τ k Rw) k
        (fun S R j => (hr S R j).trans_lt hρ) hlow a hj).LinkSound
      (linkBoundZC A Hs P planZ tab vb k Rw M ρ t') := by
  intro R τ tgt
  unfold linkBoundZC
  split_ifs with hCR
  swap
  · exact le_top
  refine (zk_linkSoundC_at A Hs P planZ tab vb hlow hRw hρ hr hj
    (fun R τ => XvalZC A Hs P planZ tab vb R τ k Rw) R τ tgt).trans ?_
  unfold errBoundZC
  apply ENNReal.ofReal_le_ofReal
  have hρ1 : 0 < 1 - ρ := by linarith
  rw [Finset.sum_mul]
  refine div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ => ?_) hρ1.le
  refine mul_le_mul_of_nonneg_left ?_ (div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _))
  linarith [errZC_le A Hs P planZ tab vb R τ k Rw M hM ht c (hCR c)]

end FlockSoundness.Discharge.ZkLink
