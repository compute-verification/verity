import Proofs.Flock.Soundness.Discharge.ZkSession.Numbers
import Proofs.Flock.Soundness.Audit.FlockLink

/-!
# `δ_link` for `--zk` sessions under expected-time collision resistance (interactive-oracle layer)

`zk_flock_count_execOS_m1` leaves the link term as the named hypothesis `LinkSound δ_link`. It is discharged here in the
shape of `Audit.Partition.flock_batched_linkSoundE`, for the protocol as Lean defines it at the interactive-oracle layer
(`sessionZK`), under expected-time collision resistance of SHA-512 for an explicit finder
(`Assumptions.SHA512CRExpected`):

* **The binding** (`ValueBindingZ`): `Audit.Partition.ValueBinding` for `--zk` tables. A message that packs a witness
  satisfying a drawn unit's statement opens the commit string of each of the unit's wires to a value whose commitment is
  the registered one and whose bit on the wire is the decoded one; two openings of one commit string with different values
  give a collision of `Hc`.
* **Readings** (`readsZ`): on a draw, the values that drawn units reading a commit string open it to, from every padded
  witness near their table's level-0 oracle that satisfies the statement. At this layer the oracle is the prover's
  `Commit` message, so a draw fixes it: the prover's strategy on the draw is deterministic.
* **The trial** for a commit string `c` is a draw that reads `c`, uniform among such draws, at cost `t'` (one run of the
  prover's `Commit`, and the decoding of its oracles). **The value layer** (`XplurZ`) is the plurality of the trial's
  single readings at each commit string. **The finder** is `LinkFinder.finderG` on the trial; its successes output
  collisions (`outZ_spec`), so under A2 the trial's link mass at `c` is at most `2t'/2^256 + 1/(eM)` (`trialZ_mass_le`).

The link event at a drawn unit gives a reading other than the plurality at some commit string, whatever the session's
coins (`linkEvZK_le_psi`, with `prob_out_le`: a table's output oracle is the one `Commit` sent). So
(`zk_linkSoundE`)
$$\delta_{\rm link}\le Q_s\Big(\frac{2t'}{2^{256}}+\frac{1}{eM}\Big),\qquad Q_s=\sum_c\pi_c,$$
with `π_c` the probability that a draw reads `c`. At M0 (`flock_batched_linkSoundE`) the extraction from accepted reruns
adds `k/(eR_w)` and the factor `1/(1 − ρ)`; here extraction is the oracle itself. A2 is asked of one prover at a time
(`LinkCRZ`), and the bound is `⊤` at any other (`linkBoundZ`).

`zk_flock_count_execOS_m1_cr`: `zk_flock_count_execOS_m1` with the committed transcript the plurality value layer and
`δ_link` that bound.

**What this does not cover.** The finder reads the oracle the prover sends at `Commit`; the protocol as run sends a Merkle
cap, and the oracle is what extraction from the cap's openings recovers. The compiled counterpart, whose finder runs the
compiled prover and extracts, is `Discharge/ZkLink/Compiled.lean`'s open guarantee.
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Audit LinkFinder ZkSession
open scoped ENNReal NNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

/-! ## A table's output oracle is the one `Commit` sent -/

omit [Fintype K] in
theorem value_after_out (A : Arith F K) (T : TabZK F mPts) (C : Oracle F) (pts : Fin 2 → Fin mPts → F)
    (Q : Oracle F → Prop) (hQ : ¬ Q C) : value (fun out => Q out.2) (T.after A C pts) = 0 := by
  simp only [TabZK.after, tableAfterZK, bind_eq, pure_eq]
  rw [value_bind_ret, value_congr (q := fun _ => False) (fun acc => by simp [hQ]), value_false]

omit [Fintype K] in
open Classical in
/-- **An event of table `j`'s output oracle** has probability at most its indicator at the oracle the strategy's
`Commit` sent: every table outputs its own level-0 oracle. -/
theorem prob_out_le (A : Arith F K) (ts : List (TabZK F mPts)) (j : Fin ts.length) (Q : Oracle F → Prop)
    (s : Strategy (sessionZK A ts)) :
    prob (fun outs => Q (outs.getD j dfltZK).2) (sessionZK A ts) s ≤ if Q (s.1 j) then 1 else 0 := by
  classical
  split_ifs with hQ
  · exact prob_le_one _ _ _
  · have h0 : value (fun outs => Q (outs.getD j dfltZK).2)
        (Game.coin (Fin 2 → Fin mPts → F) fun pts =>
          batch (List.ofFn fun i => (ts.get i).after A (s.1 i) pts)) = 0 := by
      rw [value_coin]
      refine Finset.sum_eq_zero fun pts _ => ?_
      have hs := ofFn_split (fun i => (ts.get i).after A (s.1 i) pts) j
      have hl : ((List.ofFn fun i => (ts.get i).after A (s.1 i) pts).take j).length = j := by
        simp [j.isLt.le]
      have h := value_batch_getD_le (fun out : Bool × Oracle F => Q out.2) dfltZK
        ((ts.get j).after A (s.1 j) pts) ((List.ofFn fun i => (ts.get i).after A (s.1 i) pts).drop (j + 1))
        ((List.ofFn fun i => (ts.get i).after A (s.1 i) pts).take j)
      rw [hl, ← hs, value_after_out A (ts.get j) (s.1 j) pts Q hQ] at h
      rw [le_zero_iff.1 h, mul_zero]
    exact (prob_le_value _ (Game.coin (Fin 2 → Fin mPts → F) fun pts =>
      batch (List.ofFn fun i => (ts.get i).after A (s.1 i) pts)) s.2).trans h0.le

/-! ## The binding, readings and the trial -/

section Link

variable (A : Arith F K) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

/-- **The binding of wires to registered commit strings for `--zk` tables** (`Audit.Partition.ValueBinding`'s form,
over `TabZK` plans). Each wire is held by a commit string (`pos`). A message that packs a witness satisfying drawn unit
`u`'s table's statement opens the commit string of each of `u`'s wires: the opening's commitment is the string registered
at `R` (`cs`), and its value's bit on the wire is the decoded one. Two openings of one string with different values give
a collision of `Hc` (`collide`). -/
structure ValueBindingZ (decode : DecoderZK (C := C) planZ tab) {Dh : Type} (Hc : List UInt8 → Dh) where
  Pos : Type
  [finPos : Fintype Pos]
  [decPos : DecidableEq Pos]
  Val : Type
  [finVal : Fintype Val]
  [decVal : DecidableEq Val]
  [inhVal : Inhabited Val]
  pos : Fin C.N → Pos
  bit : Val → Fin C.N → Bool
  Opening : Type
  [inhOp : Inhabited Opening]
  Com : Type
  val : Opening → Val
  com : Opening → Com
  cs : Reg → Pos → Com
  opening : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) →
    Message F (tabOfZ planZ tab S R u).sch.k0 (tabOfZ planZ tab S R u).sch.level₀.logCols → Pos → Opening
  opening_spec : ∀ S R u, u ∈ S → ∀ M, (tabOfZ planZ tab S R u).S.Satisfies
      (wit A (tabOfZ planZ tab S R u).S (tabOfZ planZ tab S R u).sch M) →
    ∀ g ∈ P.io u, com (opening S R u M (pos g)) = cs R (pos g) ∧
      bit (val (opening S R u M (pos g))) g = decode S R u M g
  collide : Opening → Opening → List UInt8 × List UInt8
  collide_spec : ∀ o o', com o = com o' → val o ≠ val o' →
    (collide o o').1 ≠ (collide o o').2 ∧ Hc (collide o o').1 = Hc (collide o o').2

attribute [instance] ValueBindingZ.finPos ValueBindingZ.decPos ValueBindingZ.finVal ValueBindingZ.decVal
  ValueBindingZ.inhVal ValueBindingZ.inhOp

variable {L : Law n} {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : ValueBindingZ A P planZ tab decode Hc) (R : Reg) (τ : Cont L Reg (batchedSessionZK A planZ) R)

/-- The level-0 oracles the prover's `Commit` sends on draw `ω`. -/
noncomputable abbrev oraclesω (ω : L.Ω) : Fin (planZ (L.draw ω) R).length → Oracle F :=
  (zkStrat A planZ (τ ω)).1

/-- Drawn unit `u` reads commit string `c`. -/
def ReadsZ (u : Fin n) (c : vb.Pos) : Prop := ∃ g ∈ P.io u, vb.pos g = c

/-- A draw reads commit string `c`. -/
def ReadsDrawZ (c : vb.Pos) (ω : L.Ω) : Prop := ∃ u ∈ L.draw ω, ReadsZ A P planZ tab vb u c

/-- **On draw `ω`, commit string `c` opens to `v` by the opening `o`**: a drawn unit that reads `c` opens it so from a
padded witness near its table's oracle that satisfies the statement. -/
def OpensAtZ (ω : L.Ω) (c : vb.Pos) (v : vb.Val) (o : vb.Opening) : Prop :=
  ∃ u ∈ L.draw ω, ReadsZ A P planZ tab vb u c ∧
    ∃ M ∈ padWits A (tabOfZ planZ tab (L.draw ω) R u).sch (tabOfZ planZ tab (L.draw ω) R u).t
      (tabOfZ planZ tab (L.draw ω) R u).c₀ (oraclesω A planZ R τ ω (tab (L.draw ω) R u)),
      (tabOfZ planZ tab (L.draw ω) R u).S.Satisfies
        (wit A (tabOfZ planZ tab (L.draw ω) R u).S (tabOfZ planZ tab (L.draw ω) R u).sch M) ∧
      o = vb.opening (L.draw ω) R u M c ∧ vb.val o = v

open Classical in
/-- **The readings of commit string `c`** on draw `ω`. -/
noncomputable def readsZ (ω : L.Ω) (c : vb.Pos) : Finset vb.Val :=
  Finset.univ.filter fun v => ∃ o, OpensAtZ A P planZ tab vb R τ ω c v o

/-- The draws that read `c`. -/
abbrev ΩcZ (L : Law n) (c : vb.Pos) : Type := {ω : L.Ω // ReadsDrawZ A P planZ tab vb c ω}

open Classical in
noncomputable instance instFintypeΩcZ (L : Law n) (c : vb.Pos) : Fintype (ΩcZ A P planZ tab vb L c) := Subtype.fintype _

/-- **The trial for commit string `c`**: a draw that reads `c`, uniform among such draws. -/
noncomputable abbrev trialZ (c : vb.Pos) [Nonempty (ΩcZ A P planZ tab vb L c)] : Game (ΩcZ A P planZ tab vb L c) :=
  Game.draw (ΩcZ A P planZ tab vb L c)

/-- What a trial reads at `c`: the readings of its draw. -/
noncomputable def rdZ (c : vb.Pos) (w : ΩcZ A P planZ tab vb L c) : Finset vb.Val :=
  readsZ A P planZ tab vb R τ w.1 c

open Classical in
/-- An opening a trial found for value `v` at `c`, when it found one. -/
noncomputable def opOfZ (c : vb.Pos) (w : ΩcZ A P planZ tab vb L c) (v : vb.Val) : vb.Opening :=
  Classical.epsilon fun o => OpensAtZ A P planZ tab vb R τ w.1 c v o

open Classical in
/-- **The finder's output**: a collision from two openings of `c` with different values, the first from its first
trial, the second from that trial or from the first further trial that reads `c`. -/
noncomputable def outZ (c : vb.Pos) (x : ΩcZ A P planZ tab vb L c × List (ΩcZ A P planZ tab vb L c)) :
    Option (List UInt8 × List UInt8) :=
  if h : ∃ p : ΩcZ A P planZ tab vb L c × vb.Val × vb.Val, p.2.1 ≠ p.2.2 ∧
      p.2.1 ∈ rdZ A P planZ tab vb R τ c x.1 ∧ p.2.2 ∈ rdZ A P planZ tab vb R τ c p.1 ∧
      (p.1 = x.1 ∨ nextRead (rdZ A P planZ tab vb R τ c) x.2 = some p.1) then
    some (vb.collide (opOfZ A P planZ tab vb R τ c x.1 (Classical.choose h).2.1)
      (opOfZ A P planZ tab vb R τ c (Classical.choose h).1 (Classical.choose h).2.2))
  else none

open Classical in
/-- **The plurality value at `c`**: the value the trial reads alone most often. -/
noncomputable def XvalZ (c : vb.Pos) : vb.Val :=
  if h : Nonempty (ΩcZ A P planZ tab vb L c) then
    haveI := h
    Classical.choose (Finset.exists_max_image Finset.univ
      (pRd (trialZ A P planZ tab vb c) (drawStrat _) (rdZ A P planZ tab vb R τ c)) Finset.univ_nonempty)
  else default

/-- **The value layer**: each wire's bit in the plurality value of its commit string. -/
noncomputable def XplurZ : Fin C.N → Bool :=
  fun g => vb.bit (XvalZ A P planZ tab vb R τ (vb.pos g)) g

/-! ## Openings, and the finder's output -/

omit [Fintype K] in
theorem opensAtZ_spec {ω : L.Ω} {c : vb.Pos} {v : vb.Val} {o : vb.Opening}
    (h : OpensAtZ A P planZ tab vb R τ ω c v o) : vb.com o = vb.cs R c ∧ vb.val o = v := by
  obtain ⟨u, hu, ⟨g, hg, hgc⟩, M, -, hsat, rfl, hv⟩ := h
  refine ⟨?_, hv⟩
  have h1 := (vb.opening_spec (L.draw ω) R u hu M hsat g hg).1
  rw [hgc] at h1
  exact h1

omit [Fintype K] in
theorem opOfZ_spec {c : vb.Pos} {w : ΩcZ A P planZ tab vb L c} {v : vb.Val}
    (hv : v ∈ rdZ A P planZ tab vb R τ c w) :
    vb.com (opOfZ A P planZ tab vb R τ c w v) = vb.cs R c ∧ vb.val (opOfZ A P planZ tab vb R τ c w v) = v := by
  classical
  unfold rdZ readsZ at hv
  obtain ⟨o, ho⟩ := (Finset.mem_filter.1 hv).2
  exact opensAtZ_spec A P planZ tab vb R τ (Classical.epsilon_spec (p := fun o =>
    OpensAtZ A P planZ tab vb R τ w.1 c v o) ⟨o, ho⟩)

omit [Fintype K] in
/-- **Every success of the finder outputs a collision.** -/
theorem outZ_spec (c : vb.Pos) (x : ΩcZ A P planZ tab vb L c × List (ΩcZ A P planZ tab vb L c))
    (hS : Succ (rdZ A P planZ tab vb R τ c) x) :
    ∃ a b, outZ A P planZ tab vb R τ c x = some (a, b) ∧ a ≠ b ∧ Hc a = Hc b := by
  classical
  have hex : ∃ p : ΩcZ A P planZ tab vb L c × vb.Val × vb.Val, p.2.1 ≠ p.2.2 ∧
      p.2.1 ∈ rdZ A P planZ tab vb R τ c x.1 ∧ p.2.2 ∈ rdZ A P planZ tab vb R τ c p.1 ∧
      (p.1 = x.1 ∨ nextRead (rdZ A P planZ tab vb R τ c) x.2 = some p.1) := by
    rcases hS with h2 | ⟨v, hv, e', hn, hne⟩
    · obtain ⟨a, ha, b, hb, hab⟩ := Finset.one_lt_card.1 (by omega :
        1 < (rdZ A P planZ tab vb R τ c x.1).card)
      exact ⟨(x.1, a, b), hab, ha, hb, Or.inl rfl⟩
    · have hnon : (rdZ A P planZ tab vb R τ c e').Nonempty := by
        have := List.find?_some hn
        simpa using this
      have hw : ∃ w ∈ rdZ A P planZ tab vb R τ c e', w ≠ v := by
        by_contra hall
        push Not at hall
        apply hne
        obtain ⟨w₀, hw₀⟩ := hnon
        refine Finset.eq_singleton_iff_unique_mem.2 ⟨?_, hall⟩
        rw [← hall w₀ hw₀]
        exact hw₀
      obtain ⟨w, hw, hwv⟩ := hw
      exact ⟨(e', v, w), Ne.symm hwv, by rw [hv]; exact Finset.mem_singleton_self v, hw, Or.inr hn⟩
  unfold outZ
  rw [dite_eq_left_of_eq_true (eq_true hex)]
  obtain ⟨hne, h1, h2, -⟩ := Classical.choose_spec hex
  have s1 := opOfZ_spec A P planZ tab vb R τ h1
  have s2 := opOfZ_spec A P planZ tab vb R τ h2
  exact ⟨_, _, rfl, vb.collide_spec _ _ (s1.1.trans s2.1.symm) (by rw [s1.2, s2.2]; exact hne)⟩

/-! ## The link mass at one commit string -/

open Classical in
/-- **The link event's trace at commit string `c`**: draw `ω` reads a value other than the plurality there. -/
noncomputable def psiZ (c : vb.Pos) (ω : L.Ω) : ℝ :=
  if ∃ v ∈ readsZ A P planZ tab vb R τ ω c, v ≠ XvalZ A P planZ tab vb R τ c then 1 else 0

omit [Fintype K] in
theorem psiZ_nonneg (c : vb.Pos) (ω : L.Ω) : 0 ≤ psiZ A P planZ tab vb R τ c ω := by
  unfold psiZ; split_ifs <;> norm_num

omit [Fintype K] in
theorem XvalZ_max (c : vb.Pos) [hne : Nonempty (ΩcZ A P planZ tab vb L c)] (v : vb.Val) :
    pRd (trialZ A P planZ tab vb c) (drawStrat _) (rdZ A P planZ tab vb R τ c) v ≤
      pRd (trialZ A P planZ tab vb c) (drawStrat _) (rdZ A P planZ tab vb R τ c)
        (XvalZ A P planZ tab vb R τ c) := by
  unfold XvalZ
  rw [dite_eq_left_of_eq_true (eq_true hne)]
  exact (Classical.choose_spec (Finset.exists_max_image Finset.univ _ Finset.univ_nonempty)).2 v
    (Finset.mem_univ v)

omit [Fintype K] in
theorem readsZ_eq_empty {c : vb.Pos} {ω : L.Ω} (h : ¬ ReadsDrawZ A P planZ tab vb c ω) :
    readsZ A P planZ tab vb R τ ω c = ∅ := by
  classical
  unfold readsZ
  refine Finset.filter_false_of_mem fun v _ => ?_
  rintro ⟨o, u, hu, hrd, -⟩
  exact h ⟨u, hu, hrd⟩

/-- **A2 for one prover's link finders** (`Assumptions.SHA512CRExpected`): at the registration `R` and the prover's
continuation `τ`, for the finder built for each commit string a draw can read, at cost `t'` per trial. As at M0
(`Audit.Partition.LinkCR`), A2 is asked only of the finders of the prover it bounds. -/
def LinkCRZ (M : ℕ) (t' : ℝ) : Prop :=
  ∀ (c : vb.Pos) (_ : Nonempty (ΩcZ A P planZ tab vb L c)), Assumptions.SHA512CRExpected Hc
    (finderG (trialZ A P planZ tab vb c) M) (finderStrat _ (drawStrat _) M) (outZ A P planZ tab vb R τ c)
    (finderCost (rdZ A P planZ tab vb R τ c) fun _ => t')

omit [Fintype K] in
/-- **The trial's link mass at `c`**, under A2 for the finder: at most `2t'/2^256 + 1/(eM)`. -/
theorem trialZ_mass_le (M : ℕ) (hM : 1 ≤ M) {t' : ℝ} (ht : 0 ≤ t') (c : vb.Pos)
    [Nonempty (ΩcZ A P planZ tab vb L c)]
    (hCR : Assumptions.SHA512CRExpected Hc (finderG (trialZ A P planZ tab vb c) M) (finderStrat _ (drawStrat _) M)
      (outZ A P planZ tab vb R τ c) (finderCost (rdZ A P planZ tab vb R τ c) fun _ => t')) :
    (∑ w : ΩcZ A P planZ tab vb L c, psiZ A P planZ tab vb R τ c w.1) / Fintype.card (ΩcZ A P planZ tab vb L c) ≤
      2 * t' / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) := by
  set T := trialZ (L := L) A P planZ tab vb c
  set sT : Strategy T := drawStrat _
  set rd := rdZ A P planZ tab vb R τ c
  set Xv := XvalZ A P planZ tab vb R τ c
  have hmass := link_mass_le T sT rd Hc (outZ A P planZ tab vb R τ c) (fun _ => t') (fun _ => ht) M hM
    (outZ_spec A P planZ tab vb R τ c) hCR (XvalZ_max A P planZ tab vb R τ c)
  have hpsi : (fun w : ΩcZ A P planZ tab vb L c => psiZ A P planZ tab vb R τ c w.1) =
      fun w => ind (fun e => (rd e).Nonempty) w - ind (fun e => rd e = {Xv}) w := by
    funext w
    rw [← Partition.ind_exists_ne rd Xv w]
    unfold psiZ ind
    split_ifs with h1 h2 h2 <;> first | rfl | exact absurd h1 h2 | exact absurd h2 h1
  have heq : (∑ w : ΩcZ A P planZ tab vb L c, psiZ A P planZ tab vb R τ c w.1) /
      Fintype.card (ΩcZ A P planZ tab vb L c) = sRd T sT rd + ((∑ v, pRd T sT rd v) - pRd T sT rd Xv) := by
    rw [← expect_draw (ΩcZ A P planZ tab vb L c) (fun w => psiZ A P planZ tab vb R τ c w.1), hpsi, expect_sub]
    have := rRd_eq T sT rd
    unfold rRd at this
    rw [this]
    unfold pRd
    ring
  rw [heq]
  rw [expect_const] at hmass
  exact hmass

/-- **The link mass at `c`, over the draw**: the probability that a draw reads a value other than the plurality at `c`
is at most `π c` times the trial's bound. -/
theorem drawZ_mass_le (M : ℕ) (hM : 1 ≤ M) {t' : ℝ} (ht : 0 ≤ t') (c : vb.Pos)
    (hCR : ∀ _ : Nonempty (ΩcZ A P planZ tab vb L c), Assumptions.SHA512CRExpected Hc
      (finderG (trialZ A P planZ tab vb c) M) (finderStrat _ (drawStrat _) M) (outZ A P planZ tab vb R τ c)
      (finderCost (rdZ A P planZ tab vb R τ c) fun _ => t')) :
    (∑ ω : L.Ω, psiZ A P planZ tab vb R τ c ω) / Fintype.card L.Ω ≤
      (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω *
        (2 * t' / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M)) := by
  classical
  have hΩ : (0 : ℝ) < Fintype.card L.Ω := by exact_mod_cast Fintype.card_pos
  have hB : 0 ≤ 2 * t' / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) := by positivity
  have hoff : ∀ ω, ¬ ReadsDrawZ A P planZ tab vb c ω → psiZ A P planZ tab vb R τ c ω = 0 := by
    intro ω hω
    simp [psiZ, readsZ_eq_empty A P planZ tab vb R τ hω]
  have hsum : ∑ ω : L.Ω, psiZ A P planZ tab vb R τ c ω =
      ∑ w : ΩcZ A P planZ tab vb L c, psiZ A P planZ tab vb R τ c w.1 := by
    rw [← Finset.sum_filter_of_ne (p := fun ω => ReadsDrawZ A P planZ tab vb c ω)
      (fun ω _ hne => by by_contra h; exact hne (hoff ω h))]
    exact Finset.sum_subtype _ (fun ω => by simp) _
  rw [hsum, div_le_iff₀ hΩ]
  by_cases hne : Nonempty (ΩcZ A P planZ tab vb L c)
  · have hmass := trialZ_mass_le A P planZ tab vb R τ M hM ht c (hCR hne)
    have hcard : (0 : ℝ) < Fintype.card (ΩcZ A P planZ tab vb L c) := by exact_mod_cast Fintype.card_pos
    rw [div_le_iff₀ hcard] at hmass
    calc ∑ w : ΩcZ A P planZ tab vb L c, psiZ A P planZ tab vb R τ c w.1
        ≤ (2 * t' / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M)) * Fintype.card (ΩcZ A P planZ tab vb L c) := hmass
      _ = _ := by field_simp
  · have h0 : (∑ w : ΩcZ A P planZ tab vb L c, psiZ A P planZ tab vb R τ c w.1) = 0 :=
      Finset.sum_eq_zero fun w _ => (hne ⟨w⟩).elim
    rw [h0]
    have : 0 ≤ (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω *
        (2 * t' / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M)) :=
      mul_nonneg (div_nonneg (Nat.cast_nonneg _) hΩ.le) hB
    nlinarith

/-! ## The link theorem -/

omit [Fintype K] in
/-- **The link event leaves a trace**: if drawn unit `u`'s table's oracle is `CommittedPad` but every satisfying padded
witness near it decodes, on `u`'s wires, differently from the value layer, then some commit string reads a value other
than its plurality. -/
theorem linkEvZK_le_psi {ω : L.Ω} {u : Fin n} (hu : u ∈ L.draw ω)
    (hL : LinkEvZK A P planZ tab decode (L.draw ω) R u (XplurZ A P planZ tab vb R τ)
      (oraclesω A planZ R τ ω (tab (L.draw ω) R u))) :
    1 ≤ ∑ c, psiZ A P planZ tab vb R τ c ω := by
  classical
  obtain ⟨⟨M, hM, hsat⟩, hall⟩ := hL
  obtain ⟨g, hg, hdec⟩ := hall M hM hsat
  have hop : OpensAtZ A P planZ tab vb R τ ω (vb.pos g)
      (vb.val (vb.opening (L.draw ω) R u M (vb.pos g))) (vb.opening (L.draw ω) R u M (vb.pos g)) :=
    ⟨u, hu, ⟨g, hg, rfl⟩, M, hM, hsat, rfl, rfl⟩
  have hmem : vb.val (vb.opening (L.draw ω) R u M (vb.pos g)) ∈ readsZ A P planZ tab vb R τ ω (vb.pos g) := by
    unfold readsZ; exact Finset.mem_filter.2 ⟨Finset.mem_univ _, _, hop⟩
  have hbit := (vb.opening_spec (L.draw ω) R u hu M hsat g hg).2
  have hne : vb.val (vb.opening (L.draw ω) R u M (vb.pos g)) ≠ XvalZ A P planZ tab vb R τ (vb.pos g) := by
    intro heq
    apply hdec
    rw [← hbit, heq]
    rfl
  have hpsi : psiZ A P planZ tab vb R τ (vb.pos g) ω = 1 := by
    unfold psiZ; rw [ite_eq_left_iff.2 fun h => absurd ⟨_, hmem, hne⟩ h]
  rw [← hpsi]
  exact Finset.single_le_sum (fun c _ => psiZ_nonneg A P planZ tab vb R τ c ω) (Finset.mem_univ (vb.pos g))

variable (L) in
open Classical in
/-- **The link bound, prover by prover**: `Q_s·(2t'/2^256 + 1/(eM))` at a prover whose link finders satisfy A2
(`LinkCRZ`), and `⊤` at any other. -/
noncomputable def linkBoundZ (M : ℕ) (t' : ℝ) (R : Reg) (τ : Cont L Reg (batchedSessionZK A planZ) R) : ℝ≥0∞ :=
  if LinkCRZ A P planZ tab vb R τ M t' then
    ENNReal.ofReal ((∑ c : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω) *
      (2 * t' / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M)))
  else ⊤

/-- **The `--zk` link theorem: `δ_link` under expected-time collision resistance, prover by prover**
(interactive-oracle layer). For the batched `--zk` audit (`analysisZK`) with the plurality value layer (`XplurZ`), given
the binding of wires to commit strings (`ValueBindingZ`): at every prover whose explicit finder of each commit string
satisfies A2 at cost `t'` per trial (`LinkCRZ`),
$$\delta_{\rm link}\le Q_s\Big(\frac{2t'}{2^{256}}+\frac{1}{eM}\Big),$$
with `Q_s = Σ_c π_c` the expected number of commit strings the drawn units read and `M` the finder's further trials. At
any other prover the bound is `⊤` (`linkBoundZ`). -/
theorem zk_linkSoundE (hlow : LoweringSoundZK A P planZ tab decode) {M : ℕ} (hM : 1 ≤ M) {t' : ℝ} (ht : 0 ≤ t') :
    (analysisZK (L := L) A P planZ tab decode (fun R τ => XplurZ A P planZ tab vb R τ) hlow).LinkSound
      (linkBoundZ A P planZ tab L vb M t') := by
  intro R τ tgt
  unfold linkBoundZ
  split_ifs with hCR
  swap
  · exact le_top
  set B := 2 * t' / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) with hBdef
  set Λ : L.Ω → ℝ := fun ω => ∑ c, psiZ A P planZ tab vb R τ c ω with hΛ
  have hΛ0 : ∀ ω, 0 ≤ Λ ω := fun ω => Finset.sum_nonneg fun c _ => psiZ_nonneg A P planZ tab vb R τ c ω
  have step1 : ∀ ω, atTgt (L.draw ω) (tgt ω) (fun u =>
      (analysisZK (L := L) A P planZ tab decode (fun R τ => XplurZ A P planZ tab vb R τ) hlow).link (L.draw ω) R
        (τ ω) u ((analysisZK (L := L) A P planZ tab decode (fun R τ => XplurZ A P planZ tab vb R τ) hlow).committed
          R τ)) ≤ ENNReal.ofReal (Λ ω) := by
    intro ω
    refine atTgt_le (fun u hu => ?_) (tgt ω)
    show prob (fun outs => acceptedZK (planZ (L.draw ω) R) outs ∧
      LinkEvZK A P planZ tab decode (L.draw ω) R u (XplurZ A P planZ tab vb R τ)
        (outs.getD (tab (L.draw ω) R u) dfltZK).2) (sessionZK A (planZ (L.draw ω) R)) (zkStrat A planZ (τ ω)) ≤
      ENNReal.ofReal (Λ ω)
    refine (prob_mono (q := fun outs => LinkEvZK A P planZ tab decode (L.draw ω) R u (XplurZ A P planZ tab vb R τ)
      (outs.getD (tab (L.draw ω) R u) dfltZK).2) (fun _ h => h.2) _ _).trans
      ((prob_out_le A (planZ (L.draw ω) R) (tab (L.draw ω) R u) _ (zkStrat A planZ (τ ω))).trans ?_)
    split_ifs with hL
    · rw [← ENNReal.ofReal_one]
      exact ENNReal.ofReal_le_ofReal (linkEvZK_le_psi A P planZ tab vb R τ hu hL)
    · exact zero_le
  refine (avg_mono step1).trans ?_
  rw [Partition.avg_ofReal Λ hΛ0]
  apply ENNReal.ofReal_le_ofReal
  have hc : ∀ c, (∑ ω, psiZ A P planZ tab vb R τ c ω) / Fintype.card L.Ω ≤
      (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω * B := fun c =>
    drawZ_mass_le A P planZ tab vb R τ M hM ht c (hCR c)
  calc (∑ ω, Λ ω) / Fintype.card L.Ω
      = ∑ c, (∑ ω, psiZ A P planZ tab vb R τ c ω) / Fintype.card L.Ω := by
        simp only [hΛ, div_eq_mul_inv, ← Finset.sum_mul]
        rw [Finset.sum_comm]
    _ ≤ ∑ c, (Fintype.card (ΩcZ A P planZ tab vb L c) : ℝ) / Fintype.card L.Ω * B :=
        Finset.sum_le_sum fun c _ => hc c
    _ = _ := by rw [← Finset.sum_mul]

end Link

/-! ## The count curve at M1's parameters -/

section Count

variable (A : Arith F K) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

/-- **The `--zk` count curve at the live draw, every table at M1's parameters, `δ_link` under A2**
(interactive-oracle layer): with the committed transcript the plurality value layer (`XplurZ`) of a binding of wires to
registered commit strings (`ValueBindingZ`),
`Pr[accept ∧ at least K wrong units] ≤ miss(K) + 2^-203 + Q_s·(2t'/2^256 + 1/(eM))` at every prover whose link finders
satisfy A2 (`LinkCRZ`), and `⊤` on the right at any other. -/
theorem zk_flock_count_execOS_m1_cr (hA : A.Correct)
    (hM1 : ∀ S R (j : Fin (planZ S R).length), ((planZ S R).get j).M1)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ planZ tab S R u).S u)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : ValueBindingZ A P planZ tab (decodeOfPlace A P planZ tab place) Hc)
    {M : ℕ} (hM : 1 ≤ M) {t' : ℝ} (ht : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZK A planZ))) :
    prob (fun o => o.1 = true ∧ K ≤ (P.wrong (XplurZ A P planZ tab vb (reg σ) (cont σ))).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (batchedSessionZK A planZ)) σ ≤
      (Law.stratified σs ks hks).miss K + ENNReal.ofReal ((2 : ℝ) ^ (-(203 : ℝ))) +
        linkBoundZ A P planZ tab (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) vb M t'
          (reg σ) (cont σ) :=
  zk_flock_count_execOS_m1 A P planZ tab hA hM1 σs ks hks St hS Kd pop Lb src hA3 place
    (fun R τ => XplurZ A P planZ tab vb R τ)
    (zk_linkSoundE A P planZ tab vb (loweringSoundZK_of_place A P planZ tab place) hM ht) K σ

end Count

end FlockSoundness.Discharge.ZkLink
