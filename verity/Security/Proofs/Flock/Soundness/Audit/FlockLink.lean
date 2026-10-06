import Proofs.Flock.Soundness.Audit.FlockBatchedAcc
import Proofs.Flock.Soundness.LinkFinder

/-!
# The link theorem: `δ_link` under expected-time collision resistance (DESIGN.md §3)

The batched audit's link term (`analysisBE`) is discharged here for the plurality value layer,
under expected-time collision resistance of SHA-512 for an explicit finder
(`Assumptions.SHA512CRExpected`) and the binding of wires to commit strings (`ValueBinding`,
named).

* **The binding** (`ValueBinding`): each wire is held by a commit string (a leaf under the serving
  roots, `pos`). A satisfying witness of a drawn unit's table opens the commit string of each of the
  unit's wires, registered at `R`, to a value whose bit on the wire is the decoded one. Two openings
  of one string with different values give a SHA-512 collision. M0's `hm96-sha512` leaf layout
  discharges it.
* **Readings** (`readsAt`): after an extraction, the values drawn units open a commit string to.
* **The trial** (`trialG`) for a commit string `c`: a draw that reads `c`, uniform among such draws;
  a fresh session; if the session accepts, `k` waits of at most `Rw` reruns each for a rerun in which
  the session accepts. It reads what the extraction from the runs found reads at `c` (`rdT`), and
  costs `t'` per run it plays (`cT`), `t'·(1 + k)` in expectation.
* **The value layer** (`Xplur`) is the plurality of the trial's single readings at each commit
  string. It is a function of the registration state.
* **The finder** for `c` is `LinkFinder.finderG` on the trial. Its successes output collisions
  (`outC`), so under A2 the trial's link mass at `c` is at most `2t'(1 + k)/2^256 + 1/(eM)`.

A2 is asked of one prover at a time (`LinkCR`): some prover hard-codes two openings of a commit string, and its
finders find a collision, so no bound holds for every prover under A2. The theorem bounds the link term at each prover
whose finders satisfy A2, and gives `⊤` at any other (`linkBoundCR`).

The link event at a drawn unit gives a reading other than the plurality at some commit string, so
the audit's link term is at most the ideal link mass, summed over commit strings. The ideal extractor
and the trial differ only by the waits that give up (`k/(eRw)`). Each commit string is weighted by
the probability `π c` that a draw reads it (`flock_batched_linkSoundE`):
$$\delta_{\rm link}\le\frac{Q_s}{1-\rho}\Big(\frac{2(1+k)\,t'}{2^{256}}+\frac{1}{eM}+
\frac{k}{eR_w}\Big),\qquad Q_s=\sum_c\pi_c,$$
with `ρ ≥ N₀/(ek)` for every table. `Q_s` is the expected number of commit strings the drawn units
read, at most the `N_s` leaves under the serving roots. At `k = ⌈2N₀/e⌉`, `ρ = 1/2`, this is the
record's `t·N_s(8N₀/e)/2^256` with `Q_s` in place of `N_s`, up to lower-order terms.
-/

namespace FlockSoundness.Audit

open Game Model Accounting Extract LinkFinder
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

/-- **The binding of wires to registered commit strings** (named hypothesis): the statement's
Halevi–Micali checks, for each drawn unit's table. Each wire is held by a commit string (`pos`). A
message that satisfies a drawn unit's table opens the commit string of each of the unit's wires:
the opening's commitment is the string registered at `R` (`cs`), and its value's bit on the wire is
the decoded one. Two openings of one string with different values give a collision of `Hc`
(`collide`), as Halevi–Micali binding reduces to SHA-512 collision resistance. -/
structure ValueBinding (decode : DecoderB (C := C) (Reg := Reg) plan tab) {Dh : Type}
    (Hc : List UInt8 → Dh) where
  /-- The commit strings: the leaves under the serving roots. -/
  Pos : Type
  [finPos : Fintype Pos]
  [decPos : DecidableEq Pos]
  /-- The value behind a commit string. -/
  Val : Type
  [finVal : Fintype Val]
  [decVal : DecidableEq Val]
  [inhVal : Inhabited Val]
  /-- The commit string that holds each wire. -/
  pos : Fin C.N → Pos
  /-- A value's bit on each wire its string holds. -/
  bit : Val → Fin C.N → Bool
  /-- An opening of a commit string, and its commitment. -/
  Opening : Type
  [inhOp : Inhabited Opening]
  Com : Type
  val : Opening → Val
  com : Opening → Com
  /-- The commit string registered at each position. -/
  cs : Reg → Pos → Com
  /-- The opening a drawn unit's table's message gives at a commit string. -/
  opening : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) →
    Message 𝔽 (tabOf plan tab S R u).sch.k0 (tabOf plan tab S R u).sch.level₀.logCols → Pos → Opening
  opening_spec : ∀ S R u, u ∈ S → ∀ M, (tabOf plan tab S R u).S.Satisfies
      (witnessOf execArith (tabOf plan tab S R u).S.m (tabOf plan tab S R u).sch.k0
        (tabOf plan tab S R u).sch.level₀.logCols M) →
      ∀ g ∈ P.io u, com (opening S R u M (pos g)) = cs R (pos g) ∧
        bit (val (opening S R u M (pos g))) g = decode S R u M g
  /-- Two openings of one string with different values give a collision. -/
  collide : Opening → Opening → List UInt8 × List UInt8
  collide_spec : ∀ o o', com o = com o' → val o ≠ val o' →
    (collide o o').1 ≠ (collide o o').2 ∧ Hc (collide o o').1 = Hc (collide o o').2

attribute [instance] ValueBinding.finPos ValueBinding.decPos ValueBinding.finVal
  ValueBinding.decVal ValueBinding.inhVal ValueBinding.inhOp

variable {L : Law n} {decode : DecoderB (C := C) (Reg := Reg) plan tab} {Dh : Type}
  {Hc : List UInt8 → Dh} (vb : P.ValueBinding plan tab decode Hc)
  (R : Reg) (τ : Cont L Reg (batchedSession H E plan) R)

/-- The session strategy on draw `ω`. -/
noncomputable abbrev stratω (ω : L.Ω) : Strategy (sessionB execArith H E (plan (L.draw ω) R)) :=
  batchStrat H E plan (τ ω)

/-- Everything after `Commit` on draw `ω`. -/
noncomputable abbrev gameω (ω : L.Ω) : Game (List (OutC 𝔽 D)) :=
  sessAfterB (gamesB execArith H E (plan (L.draw ω) R) (stratω H E plan R τ ω).1)

/-- The session accepts. -/
abbrev accω (ω : L.Ω) : List (OutC 𝔽 D) → Prop := acceptedB (plan (L.draw ω) R)

/-- The table extracted for drawn unit `u` from the reruns `bs`. -/
noncomputable abbrev tabExt (ω : L.Ω) (u : Fin n) (bs : List (List (OutC 𝔽 D))) : Oracle 𝔽 :=
  extractTableS H E (tabOf plan tab (L.draw ω) R u).sch (tab (L.draw ω) R u)
    ((stratω H E plan R τ ω).1 (tab (L.draw ω) R u)) bs

open Classical in
/-- The message the extraction settles on for unit `u`'s table: a close message packing a
satisfying witness, when there is one. -/
noncomputable def cand (ω : L.Ω) (u : Fin n) (bs : List (List (OutC 𝔽 D))) :
    Option (Message 𝔽 (tabOf plan tab (L.draw ω) R u).sch.k0
      (tabOf plan tab (L.draw ω) R u).sch.level₀.logCols) :=
  if h : Committed execArith (tabOf plan tab (L.draw ω) R u).S (tabOf plan tab (L.draw ω) R u).sch
      (tabExt H E plan tab R τ ω u bs) then some (Classical.choose h) else none

/-- Drawn unit `u` reads commit string `c`. -/
def Reads (u : Fin n) (c : vb.Pos) : Prop := ∃ g ∈ P.io u, vb.pos g = c

/-- A draw reads commit string `c`. -/
def ReadsDraw (c : vb.Pos) (ω : L.Ω) : Prop := ∃ u ∈ L.draw ω, P.Reads plan tab vb u c

/-- An extraction opens `c` to value `v`, with the opening `o`: a drawn unit that reads `c` opens it
so from the message its table settles on. -/
def OpensAt (ω : L.Ω) (bs : List (List (OutC 𝔽 D))) (c : vb.Pos) (v : vb.Val) (o : vb.Opening) :
    Prop :=
  ∃ u ∈ L.draw ω, P.Reads plan tab vb u c ∧ ∃ M, cand H E plan tab R τ ω u bs = some M ∧
    o = vb.opening (L.draw ω) R u M c ∧ vb.val o = v

open Classical in
/-- **The readings of commit string `c`** after an extraction from the reruns `bs`. -/
noncomputable def readsAt (ω : L.Ω) (bs : List (List (OutC 𝔽 D))) (c : vb.Pos) : Finset vb.Val :=
  Finset.univ.filter fun v => ∃ o, P.OpensAt H E plan tab vb R τ ω bs c v o

/-- The draws that read `c`. -/
abbrev Ωc (L : Law n) (c : vb.Pos) : Type := {ω : L.Ω // P.ReadsDraw plan tab vb c ω}

open Classical in
noncomputable instance instFintypeΩc (L : Law n) (c : vb.Pos) : Fintype (P.Ωc plan tab vb L c) :=
  Subtype.fintype _

/-- A trial's outcome: the draw, the fresh run, and the waits' runs. -/
abbrev TrialOut (L : Law n) (D : Type) := L.Ω × List (OutC 𝔽 D) × List (List (List (OutC 𝔽 D)))

/-- **The trial for commit string `c`**: a draw that reads `c`, a fresh session, and `k` waits of at
most `Rw` reruns each. -/
noncomputable def trialG (c : vb.Pos) [Nonempty (P.Ωc plan tab vb L c)] (k Rw : ℕ) : Game (TrialOut L D) :=
  (Game.draw (P.Ωc plan tab vb L c)).bind fun w => (gameω H E plan R τ w.1).bind fun fresh =>
    (runsG (runsG (gameω H E plan R τ w.1) Rw) k).map fun bl => (w.1, fresh, bl)

/-- The prover's session strategy in every run. -/
noncomputable def trialStrat (c : vb.Pos) [Nonempty (P.Ωc plan tab vb L c)] (k Rw : ℕ) :
    Strategy (P.trialG H E plan tab vb R τ c k Rw) :=
  Strategy.bind _ _ (drawStrat _) fun w => Strategy.bind _ _ (stratω H E plan R τ w.1).2 fun _ =>
    Strategy.map _ _ (runsStrat _ (runsStrat _ (stratω H E plan R τ w.1).2 Rw) k)

open Classical in
/-- **What a trial reads at `c`**: nothing unless the fresh session accepts and every wait finds an
accepted rerun; then the readings of the extraction from those reruns. -/
noncomputable def rdT (c : vb.Pos) (e : TrialOut L D) : Finset vb.Val :=
  if accω plan R e.1 e.2.1 then
    (firsts (accω plan R e.1) e.2.2).elim ∅ fun bs => P.readsAt H E plan tab vb R τ e.1 bs c
  else ∅

open Classical in
/-- **A trial's cost**: `t'` per run it plays, the fresh one and, if it accepts, each wait's. -/
noncomputable def cT (t' : ℝ) (e : TrialOut L D) : ℝ :=
  t' * (1 + if accω plan R e.1 e.2.1 then (e.2.2.map (waitCost (accω plan R e.1))).sum else 0)

open Classical in
/-- An opening a trial found for value `v` at `c`, when it found one. -/
noncomputable def opOf (c : vb.Pos) (e : TrialOut L D) (v : vb.Val) : vb.Opening :=
  Classical.epsilon fun o => accω plan R e.1 e.2.1 ∧
    ∃ bs, firsts (accω plan R e.1) e.2.2 = some bs ∧ P.OpensAt H E plan tab vb R τ e.1 bs c v o

open Classical in
/-- **The finder's output**: a collision from two openings of `c` with different values, the first
from its first trial, the second from that trial or from the first further trial that reads `c`. -/
noncomputable def outC (c : vb.Pos) (x : TrialOut L D × List (TrialOut L D)) :
    Option (List UInt8 × List UInt8) :=
  if h : ∃ p : TrialOut L D × vb.Val × vb.Val, p.2.1 ≠ p.2.2 ∧
      p.2.1 ∈ P.rdT H E plan tab vb R τ c x.1 ∧ p.2.2 ∈ P.rdT H E plan tab vb R τ c p.1 ∧
      (p.1 = x.1 ∨ nextRead (P.rdT H E plan tab vb R τ c) x.2 = some p.1) then
    some (vb.collide (P.opOf H E plan tab vb R τ c x.1 (Classical.choose h).2.1)
      (P.opOf H E plan tab vb R τ c (Classical.choose h).1 (Classical.choose h).2.2))
  else none

open Classical in
/-- **The plurality value at `c`**: the value the trial reads alone most often. -/
noncomputable def Xval (k Rw : ℕ) (c : vb.Pos) : vb.Val :=
  if h : Nonempty (P.Ωc plan tab vb L c) then
    haveI := h
    Classical.choose (Finset.exists_max_image Finset.univ
      (pRd (P.trialG H E plan tab vb R τ c k Rw) (P.trialStrat H E plan tab vb R τ c k Rw)
        (P.rdT H E plan tab vb R τ c)) Finset.univ_nonempty)
  else default

/-- **The value layer**: each wire's bit in the plurality value of its commit string. -/
noncomputable def Xplur (k Rw : ℕ) : Fin C.N → Bool :=
  fun g => vb.bit (P.Xval H E plan tab vb R τ k Rw (vb.pos g)) g

/-! ## Openings, and the finder's output -/

theorem cand_sat {ω : L.Ω} {u : Fin n} {bs : List (List (OutC 𝔽 D))}
    {M : Message 𝔽 (tabOf plan tab (L.draw ω) R u).sch.k0
      (tabOf plan tab (L.draw ω) R u).sch.level₀.logCols}
    (h : cand H E plan tab R τ ω u bs = some M) :
    (tabOf plan tab (L.draw ω) R u).S.Satisfies (witnessOf execArith
      (tabOf plan tab (L.draw ω) R u).S.m (tabOf plan tab (L.draw ω) R u).sch.k0
      (tabOf plan tab (L.draw ω) R u).sch.level₀.logCols M) := by
  unfold cand at h
  split_ifs at h with hc
  obtain rfl := Option.some.inj h
  exact (Classical.choose_spec hc).2

theorem opensAt_spec {ω : L.Ω} {bs : List (List (OutC 𝔽 D))} {c : vb.Pos} {v : vb.Val}
    {o : vb.Opening} (h : P.OpensAt H E plan tab vb R τ ω bs c v o) :
    vb.com o = vb.cs R c ∧ vb.val o = v := by
  obtain ⟨u, hu, ⟨g, hg, hgc⟩, M, hM, rfl, hv⟩ := h
  refine ⟨?_, hv⟩
  have h1 := (vb.opening_spec (L.draw ω) R u hu M (cand_sat H E plan tab R τ hM) g hg).1
  rw [hgc] at h1
  exact h1

theorem opOf_spec {c : vb.Pos} {e : TrialOut L D} {v : vb.Val}
    (hv : v ∈ P.rdT H E plan tab vb R τ c e) :
    vb.com (P.opOf H E plan tab vb R τ c e v) = vb.cs R c ∧
      vb.val (P.opOf H E plan tab vb R τ c e v) = v := by
  classical
  unfold rdT at hv
  split_ifs at hv with hacc
  · cases hf : firsts (accω plan R e.1) e.2.2 with
    | none => simp [hf] at hv
    | some bs =>
        simp only [hf, Option.elim_some] at hv
        unfold readsAt at hv
        obtain ⟨o, ho⟩ := (Finset.mem_filter.1 hv).2
        have hex : ∃ o, accω plan R e.1 e.2.1 ∧ ∃ bs, firsts (accω plan R e.1) e.2.2 = some bs ∧
            P.OpensAt H E plan tab vb R τ e.1 bs c v o := ⟨o, hacc, bs, hf, ho⟩
        obtain ⟨-, bs', -, hop⟩ := Classical.epsilon_spec hex
        exact P.opensAt_spec H E plan tab vb R τ hop
  · simp at hv

/-- **Every success of the finder outputs a collision.** -/
theorem outC_spec (c : vb.Pos) (x : TrialOut L D × List (TrialOut L D))
    (hS : Succ (P.rdT H E plan tab vb R τ c) x) :
    ∃ a b, P.outC H E plan tab vb R τ c x = some (a, b) ∧ a ≠ b ∧ Hc a = Hc b := by
  classical
  have hex : ∃ p : TrialOut L D × vb.Val × vb.Val, p.2.1 ≠ p.2.2 ∧
      p.2.1 ∈ P.rdT H E plan tab vb R τ c x.1 ∧ p.2.2 ∈ P.rdT H E plan tab vb R τ c p.1 ∧
      (p.1 = x.1 ∨ nextRead (P.rdT H E plan tab vb R τ c) x.2 = some p.1) := by
    rcases hS with h2 | ⟨v, hv, e', hn, hne⟩
    · obtain ⟨a, ha, b, hb, hab⟩ := Finset.one_lt_card.1 (by omega :
        1 < (P.rdT H E plan tab vb R τ c x.1).card)
      exact ⟨(x.1, a, b), hab, ha, hb, Or.inl rfl⟩
    · have hnon : (P.rdT H E plan tab vb R τ c e').Nonempty := by
        have := List.find?_some hn
        simpa using this
      have hw : ∃ w ∈ P.rdT H E plan tab vb R τ c e', w ≠ v := by
        by_contra hall
        push Not at hall
        apply hne
        obtain ⟨w₀, hw₀⟩ := hnon
        refine Finset.eq_singleton_iff_unique_mem.2 ⟨?_, hall⟩
        rw [← hall w₀ hw₀]
        exact hw₀
      obtain ⟨w, hw, hwv⟩ := hw
      exact ⟨(e', v, w), Ne.symm hwv, by rw [hv]; exact Finset.mem_singleton_self v, hw, Or.inr hn⟩
  unfold outC
  rw [dite_eq_left_of_eq_true (eq_true hex)]
  obtain ⟨hne, h1, h2, -⟩ := Classical.choose_spec hex
  have s1 := P.opOf_spec H E plan tab vb R τ h1
  have s2 := P.opOf_spec H E plan tab vb R τ h2
  exact ⟨_, _, rfl, vb.collide_spec _ _ (s1.1.trans s2.1.symm) (by rw [s1.2, s2.2]; exact hne)⟩

/-! ## The trial's expectations -/

theorem expect_trial (c : vb.Pos) [Nonempty (P.Ωc plan tab vb L c)] (k Rw : ℕ)
    (F : TrialOut L D → ℝ) :
    expect F (P.trialG H E plan tab vb R τ c k Rw) (P.trialStrat H E plan tab vb R τ c k Rw) =
      (∑ w : P.Ωc plan tab vb L c, expect (fun fresh =>
          expectN (runsG (gameω H E plan R τ w.1) Rw)
            (runsStrat _ (stratω H E plan R τ w.1).2 Rw) k fun bl => F (w.1, fresh, bl))
          (gameω H E plan R τ w.1) (stratω H E plan R τ w.1).2) /
        Fintype.card (P.Ωc plan tab vb L c) := by
  unfold trialG trialStrat
  rw [expect_bind, expect_draw]
  congr 1
  refine Finset.sum_congr rfl fun w _ => ?_
  rw [expect_bind]
  refine congrArg (fun f => expect f _ _) (funext fun fresh => ?_)
  rw [expect_map, expect_runsG]

theorem cT_nonneg {t' : ℝ} (ht : 0 ≤ t') (e : TrialOut L D) : 0 ≤ cT plan R t' e := by
  classical
  unfold cT
  have h : 0 ≤ if accω plan R e.1 e.2.1 then (e.2.2.map (waitCost (accω plan R e.1))).sum
      else 0 := by
    split_ifs
    · exact List.sum_nonneg (by
        simp only [List.mem_map]
        rintro _ ⟨b, -, rfl⟩
        exact waitCost_nonneg _ b)
    · exact le_rfl
  exact mul_nonneg ht (by linarith)

/-- **A trial costs at most `t'(1 + k)` in expectation**, whatever the acceptance probability. -/
theorem expect_cT_le (c : vb.Pos) [Nonempty (P.Ωc plan tab vb L c)] (k Rw : ℕ) {t' : ℝ}
    (ht : 0 ≤ t') :
    expect (cT plan R t') (P.trialG H E plan tab vb R τ c k Rw)
      (P.trialStrat H E plan tab vb R τ c k Rw) ≤ t' * (1 + k) := by
  classical
  rw [expect_trial, div_le_iff₀ (by exact_mod_cast Fintype.card_pos)]
  have hw : ∀ w : P.Ωc plan tab vb L c, expect (fun fresh =>
      expectN (runsG (gameω H E plan R τ w.1) Rw)
        (runsStrat _ (stratω H E plan R τ w.1).2 Rw) k fun bl => cT plan R t' (w.1, fresh, bl))
      (gameω H E plan R τ w.1) (stratω H E plan R τ w.1).2 ≤ t' * (1 + k) := by
    intro w
    set g := gameω H E plan R τ w.1
    set sg := (stratω H E plan R τ w.1).2
    set acc : List (OutC 𝔽 D) → Prop := accω plan R w.1
    have hblk : ∀ fresh, expectN (runsG g Rw) (runsStrat g sg Rw) k
        (fun bl => cT plan R t' (w.1, fresh, bl)) =
          t' * (1 + (k * geom (expect (ind acc) g sg) Rw) * ind acc fresh) := by
      intro fresh
      by_cases ha : acc fresh
      · have e1 : (fun bl : List (List (List (OutC 𝔽 D))) => cT plan R t' (w.1, fresh, bl)) =
            fun bl => t' * (1 + (bl.map (waitCost acc)).sum) := by
          funext bl; simp only [cT, acc, ha, ↓reduceIte]
        rw [e1, expectN_const_mul, expectN_add, expectN_const, expectN_sum_map, expect_runsG,
          expectN_waitCost]
        simp only [ind, ha, ↓reduceIte, mul_one]
      · have e1 : (fun bl : List (List (List (OutC 𝔽 D))) => cT plan R t' (w.1, fresh, bl)) =
            fun _ => t' * 1 := by
          funext bl; simp only [cT, acc, ha, ↓reduceIte, add_zero]
        rw [e1, expectN_const]
        simp [ind, ha]
    rw [show (fun fresh => expectN (runsG g Rw) (runsStrat g sg Rw) k
        fun bl => cT plan R t' (w.1, fresh, bl)) =
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
  calc (∑ w : P.Ωc plan tab vb L c, _) ≤ ∑ _w : P.Ωc plan tab vb L c, t' * (1 + k) :=
        Finset.sum_le_sum fun w _ => hw w
    _ = t' * (1 + k) * Fintype.card (P.Ωc plan tab vb L c) := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]; ring

open Classical in
/-- A trial's value for a function `Ψ` of the reruns: `Ψ` of the reruns its waits found, when the
fresh session accepted and every wait found one. -/
noncomputable def trialVal (Ψ : L.Ω → List (List (OutC 𝔽 D)) → ℝ) (e : TrialOut L D) : ℝ :=
  if accω plan R e.1 e.2.1 then (firsts (accω plan R e.1) e.2.2).elim 0 (Ψ e.1) else 0

theorem expect_trialVal (c : vb.Pos) [Nonempty (P.Ωc plan tab vb L c)] (k Rw : ℕ)
    (Ψ : L.Ω → List (List (OutC 𝔽 D)) → ℝ) :
    expect (trialVal plan R Ψ) (P.trialG H E plan tab vb R τ c k Rw)
        (P.trialStrat H E plan tab vb R τ c k Rw) =
      (∑ w : P.Ωc plan tab vb L c,
          expect (ind (accω plan R w.1)) (gameω H E plan R τ w.1) (stratω H E plan R τ w.1).2 *
            geom (expect (ind (accω plan R w.1)) (gameω H E plan R τ w.1)
              (stratω H E plan R τ w.1).2) Rw ^ k *
            expectN (gameω H E plan R τ w.1) (stratω H E plan R τ w.1).2 k
              (fun bs => (bs.map (ind (accω plan R w.1))).prod * Ψ w.1 bs)) /
        Fintype.card (P.Ωc plan tab vb L c) := by
  classical
  rw [expect_trial]
  congr 1
  refine Finset.sum_congr rfl fun w _ => ?_
  set g := gameω H E plan R τ w.1
  set sg := (stratω H E plan R τ w.1).2
  set acc : List (OutC 𝔽 D) → Prop := accω plan R w.1
  set Xs := geom (expect (ind acc) g sg) Rw ^ k *
    expectN g sg k (fun bs => (bs.map (ind acc)).prod * Ψ w.1 bs)
  have hblk : ∀ fresh, expectN (runsG g Rw) (runsStrat g sg Rw) k
      (fun bl => trialVal plan R Ψ (w.1, fresh, bl)) = Xs * ind acc fresh := by
    intro fresh
    by_cases ha : acc fresh
    · have e1 : (fun bl : List (List (List (OutC 𝔽 D))) => trialVal plan R Ψ (w.1, fresh, bl)) =
          fun bl => (firsts acc bl).elim 0 (Ψ w.1) := by
        funext bl; simp only [trialVal, acc, ha, ↓reduceIte]
      rw [e1, expectN_firsts]
      simp [ind, ha, Xs]
    · have e1 : (fun bl : List (List (List (OutC 𝔽 D))) => trialVal plan R Ψ (w.1, fresh, bl)) =
          fun _ => 0 := by
        funext bl; simp only [trialVal, acc, ha, ↓reduceIte]
      rw [e1, expectN_const]
      simp [ind, ha]
  rw [show (fun fresh => expectN (runsG g Rw) (runsStrat g sg Rw) k
      fun bl => trialVal plan R Ψ (w.1, fresh, bl)) = fun fresh => Xs * ind acc fresh from
    funext hblk, expect_const_mul]
  simp only [Xs]
  ring

/-- `1[∃ v ∈ S, v ≠ X] = 1[S nonempty] − 1[S = {X}]`. -/
theorem ind_exists_ne {τ' Val : Type} [DecidableEq Val] (rd : τ' → Finset Val) (Xv : Val)
    (e : τ') :
    ind (fun e => ∃ v ∈ rd e, v ≠ Xv) e =
      ind (fun e => (rd e).Nonempty) e - ind (fun e => rd e = {Xv}) e := by
  by_cases h1 : (rd e).Nonempty
  · by_cases h2 : rd e = {Xv}
    · simp [ind, h2]
    · have : ∃ v ∈ rd e, v ≠ Xv := by
        by_contra hall
        push Not at hall
        apply h2
        obtain ⟨w₀, hw₀⟩ := h1
        refine Finset.eq_singleton_iff_unique_mem.2 ⟨?_, hall⟩
        rw [← hall w₀ hw₀]; exact hw₀
      simp [ind, h1, h2, this]
  · have h2 : rd e ≠ {Xv} := fun h => h1 (h ▸ Finset.singleton_nonempty Xv)
    have : ¬ ∃ v ∈ rd e, v ≠ Xv := by
      rintro ⟨v, hv, -⟩; exact h1 ⟨v, hv⟩
    simp [ind, h1, h2, this]

/-! ## The link mass at one commit string -/

open Classical in
/-- **The link event's trace at commit string `c`**: the extraction from the reruns `bs` reads a
value other than the plurality there. -/
noncomputable def psi (k Rw : ℕ) (c : vb.Pos) (ω : L.Ω) (bs : List (List (OutC 𝔽 D))) : ℝ :=
  if ∃ v ∈ P.readsAt H E plan tab vb R τ ω bs c, v ≠ P.Xval H E plan tab vb R τ k Rw c then 1 else 0

theorem psi_nonneg (k Rw : ℕ) (c : vb.Pos) (ω : L.Ω) (bs : List (List (OutC 𝔽 D))) :
    0 ≤ P.psi H E plan tab vb R τ k Rw c ω bs := by
  unfold psi; split_ifs <;> norm_num

theorem psi_le_one (k Rw : ℕ) (c : vb.Pos) (ω : L.Ω) (bs : List (List (OutC 𝔽 D))) :
    P.psi H E plan tab vb R τ k Rw c ω bs ≤ 1 := by
  unfold psi; split_ifs <;> norm_num

theorem trialVal_psi (k Rw : ℕ) (c : vb.Pos) (e : TrialOut L D) :
    trialVal plan R (P.psi H E plan tab vb R τ k Rw c) e =
      ind (fun e => ∃ v ∈ P.rdT H E plan tab vb R τ c e,
        v ≠ P.Xval H E plan tab vb R τ k Rw c) e := by
  classical
  unfold trialVal rdT
  by_cases ha : accω plan R e.1 e.2.1
  · cases hf : firsts (accω plan R e.1) e.2.2 with
    | none => simp [ind, ha, hf]
    | some bs => simp [ind, ha, hf, psi]
  · simp [ind, ha]

theorem Xval_max (k Rw : ℕ) (c : vb.Pos) [hne : Nonempty (P.Ωc plan tab vb L c)] (v : vb.Val) :
    pRd (P.trialG H E plan tab vb R τ c k Rw) (P.trialStrat H E plan tab vb R τ c k Rw)
        (P.rdT H E plan tab vb R τ c) v ≤
      pRd (P.trialG H E plan tab vb R τ c k Rw) (P.trialStrat H E plan tab vb R τ c k Rw)
        (P.rdT H E plan tab vb R τ c) (P.Xval H E plan tab vb R τ k Rw c) := by
  unfold Xval
  rw [dite_eq_left_of_eq_true (eq_true hne)]
  exact (Classical.choose_spec (Finset.exists_max_image Finset.univ _ Finset.univ_nonempty)).2 v
    (Finset.mem_univ v)

theorem readsAt_eq_empty {c : vb.Pos} {ω : L.Ω} (h : ¬ P.ReadsDraw plan tab vb c ω)
    (bs : List (List (OutC 𝔽 D))) : P.readsAt H E plan tab vb R τ ω bs c = ∅ := by
  classical
  unfold readsAt
  refine Finset.filter_false_of_mem fun v _ => ?_
  rintro ⟨o, u, hu, hrd, -⟩
  exact h ⟨u, hu, hrd⟩

/-- **The trial's link mass at `c`**, under A2 for the finder: at most
`2t'(1 + k)/2^256 + 1/(eM)`. -/
theorem trial_mass_le (k Rw M : ℕ) (hM : 1 ≤ M) {t' : ℝ} (ht : 0 ≤ t') (c : vb.Pos)
    [Nonempty (P.Ωc plan tab vb L c)]
    (hCR : Assumptions.SHA512CRExpected Hc (finderG (P.trialG H E plan tab vb R τ c k Rw) M)
      (finderStrat _ (P.trialStrat H E plan tab vb R τ c k Rw) M) (P.outC H E plan tab vb R τ c)
      (finderCost (P.rdT H E plan tab vb R τ c) (cT plan R t'))) :
    expect (trialVal plan R (P.psi H E plan tab vb R τ k Rw c))
        (P.trialG H E plan tab vb R τ c k Rw) (P.trialStrat H E plan tab vb R τ c k Rw) ≤
      2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) := by
  set T := P.trialG H E plan tab vb R τ c k Rw
  set sT := P.trialStrat H E plan tab vb R τ c k Rw
  set rd := P.rdT H E plan tab vb R τ c
  set Xv := P.Xval H E plan tab vb R τ k Rw c
  have hmass := link_mass_le T sT rd Hc (P.outC H E plan tab vb R τ c) (cT plan R t')
    (cT_nonneg plan R ht) M hM (P.outC_spec H E plan tab vb R τ c) hCR
    (P.Xval_max H E plan tab vb R τ k Rw c)
  have hcost := P.expect_cT_le H E plan tab vb R τ c k Rw ht
  have heq : expect (trialVal plan R (P.psi H E plan tab vb R τ k Rw c)) T sT =
      sRd T sT rd + ((∑ v, pRd T sT rd v) - pRd T sT rd Xv) := by
    rw [show trialVal plan R (P.psi H E plan tab vb R τ k Rw c) =
        fun e => ind (fun e => (rd e).Nonempty) e - ind (fun e => rd e = {Xv}) e from
      funext fun e => (P.trialVal_psi H E plan tab vb R τ k Rw c e).trans (ind_exists_ne rd Xv e),
      expect_sub]
    have := rRd_eq T sT rd
    unfold rRd at this
    rw [this]
    unfold pRd
    ring
  have h2 : 2 * expect (cT plan R t') T sT / 2 ^ (256 : ℝ) ≤
      2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) :=
    div_le_div_of_nonneg_right (by linarith) (by positivity)
  linarith

/-- **The ideal link mass at `c`**: the audit extractor's reading of `c` other than the
plurality, jointly with acceptance and averaged over the draw, is at most `π c` times the trial's
bound plus `k/(eRw)` for the waits that give up. -/
theorem ideal_mass_le (k Rw M : ℕ) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {t' : ℝ} (ht : 0 ≤ t')
    (c : vb.Pos)
    (hCR : ∀ _ : Nonempty (P.Ωc plan tab vb L c), Assumptions.SHA512CRExpected Hc
      (finderG (P.trialG H E plan tab vb R τ c k Rw) M)
      (finderStrat _ (P.trialStrat H E plan tab vb R τ c k Rw) M) (P.outC H E plan tab vb R τ c)
      (finderCost (P.rdT H E plan tab vb R τ c) (cT plan R t'))) :
    (∑ ω : L.Ω, expect (ind (accω plan R ω)) (gameω H E plan R τ ω) (stratω H E plan R τ ω).2 *
        expectAcc (gameω H E plan R τ ω) (stratω H E plan R τ ω).2 (accω plan R ω) k
          (P.psi H E plan tab vb R τ k Rw c ω)) / Fintype.card L.Ω ≤
      (Fintype.card (P.Ωc plan tab vb L c) : ℝ) / Fintype.card L.Ω *
        (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) := by
  classical
  set f : L.Ω → ℝ := fun ω => expect (ind (accω plan R ω)) (gameω H E plan R τ ω)
      (stratω H E plan R τ ω).2 *
    expectAcc (gameω H E plan R τ ω) (stratω H E plan R τ ω).2 (accω plan R ω) k
      (P.psi H E plan tab vb R τ k Rw c ω) with hf
  have hΩ : (0 : ℝ) < Fintype.card L.Ω := by exact_mod_cast Fintype.card_pos
  have hB : 0 ≤ 2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) +
      k / (Real.exp 1 * Rw) := by positivity
  -- off the draws that read `c`, the trace vanishes
  have hoff : ∀ ω, ¬ P.ReadsDraw plan tab vb c ω → f ω = 0 := by
    intro ω hω
    have hpsi : P.psi H E plan tab vb R τ k Rw c ω = fun _ => 0 := funext fun bs => by
      simp [psi, P.readsAt_eq_empty H E plan tab vb R τ hω bs]
    simp only [hf, hpsi, expectAcc_zero, mul_zero]
  have hsum : ∑ ω : L.Ω, f ω = ∑ w : P.Ωc plan tab vb L c, f w.1 := by
    rw [← Finset.sum_filter_of_ne (p := fun ω => P.ReadsDraw plan tab vb c ω)
      (fun ω _ hne => by by_contra h; exact hne (hoff ω h))]
    exact Finset.sum_subtype _ (fun ω => by simp) f
  rw [hsum, div_le_iff₀ hΩ]
  by_cases hne : Nonempty (P.Ωc plan tab vb L c)
  · have hw : ∀ w : P.Ωc plan tab vb L c, f w.1 ≤
        expect (ind (accω plan R w.1)) (gameω H E plan R τ w.1) (stratω H E plan R τ w.1).2 *
            geom (expect (ind (accω plan R w.1)) (gameω H E plan R τ w.1)
              (stratω H E plan R τ w.1).2) Rw ^ k *
            expectN (gameω H E plan R τ w.1) (stratω H E plan R τ w.1).2 k
              (fun bs => (bs.map (ind (accω plan R w.1))).prod *
                P.psi H E plan tab vb R τ k Rw c w.1 bs) + k / (Real.exp 1 * Rw) := fun w =>
      expectAcc_le_waits _ _ _ k Rw hRw _ (P.psi_nonneg H E plan tab vb R τ k Rw c w.1)
        (P.psi_le_one H E plan tab vb R τ k Rw c w.1)
    have htr := P.expect_trialVal H E plan tab vb R τ c k Rw (P.psi H E plan tab vb R τ k Rw c)
    have hmass := P.trial_mass_le H E plan tab vb R τ k Rw M hM ht c (hCR hne)
    have hcard : (0 : ℝ) < Fintype.card (P.Ωc plan tab vb L c) := by exact_mod_cast Fintype.card_pos
    rw [htr, div_le_iff₀ hcard] at hmass
    calc ∑ w : P.Ωc plan tab vb L c, f w.1
        ≤ ∑ w : P.Ωc plan tab vb L c, (expect (ind (accω plan R w.1)) (gameω H E plan R τ w.1)
              (stratω H E plan R τ w.1).2 *
            geom (expect (ind (accω plan R w.1)) (gameω H E plan R τ w.1)
              (stratω H E plan R τ w.1).2) Rw ^ k *
            expectN (gameω H E plan R τ w.1) (stratω H E plan R τ w.1).2 k
              (fun bs => (bs.map (ind (accω plan R w.1))).prod *
                P.psi H E plan tab vb R τ k Rw c w.1 bs) + k / (Real.exp 1 * Rw)) :=
          Finset.sum_le_sum fun w _ => hw w
      _ = (∑ w : P.Ωc plan tab vb L c, expect (ind (accω plan R w.1)) (gameω H E plan R τ w.1)
              (stratω H E plan R τ w.1).2 *
            geom (expect (ind (accω plan R w.1)) (gameω H E plan R τ w.1)
              (stratω H E plan R τ w.1).2) Rw ^ k *
            expectN (gameω H E plan R τ w.1) (stratω H E plan R τ w.1).2 k
              (fun bs => (bs.map (ind (accω plan R w.1))).prod *
                P.psi H E plan tab vb R τ k Rw c w.1 bs)) +
          Fintype.card (P.Ωc plan tab vb L c) * (k / (Real.exp 1 * Rw)) := by
        rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
      _ ≤ (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M)) *
            Fintype.card (P.Ωc plan tab vb L c) +
          Fintype.card (P.Ωc plan tab vb L c) * (k / (Real.exp 1 * Rw)) := by linarith
      _ = _ := by field_simp
  · have hempty : ∀ w : P.Ωc plan tab vb L c, False := fun w => hne ⟨w⟩
    have h0 : (∑ w : P.Ωc plan tab vb L c, f w.1) = 0 :=
      Finset.sum_eq_zero fun w _ => (hempty w).elim
    rw [h0]
    have : 0 ≤ (Fintype.card (P.Ωc plan tab vb L c) : ℝ) / Fintype.card L.Ω *
        (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) :=
      mul_nonneg (div_nonneg (Nat.cast_nonneg _) hΩ.le) hB
    nlinarith

/-! ## The link theorem -/

/-- **The link event leaves a trace**: if drawn unit `u`'s extracted table is committed but every
close satisfying message decodes, on `u`'s wires, differently from the value layer, then some commit
string reads a value other than its plurality. -/
theorem linkEv_le_psi (k Rw : ℕ) {ω : L.Ω} {u : Fin n} (hu : u ∈ L.draw ω)
    (bs : List (List (OutC 𝔽 D))) :
    ind (fun bs => P.LinkEvB plan tab decode (L.draw ω) R u (P.Xplur H E plan tab vb R τ k Rw)
      (tabExt H E plan tab R τ ω u bs)) bs ≤ ∑ c, P.psi H E plan tab vb R τ k Rw c ω bs := by
  classical
  have hsum0 : 0 ≤ ∑ c, P.psi H E plan tab vb R τ k Rw c ω bs :=
    Finset.sum_nonneg fun c _ => P.psi_nonneg H E plan tab vb R τ k Rw c ω bs
  by_cases hL : P.LinkEvB plan tab decode (L.draw ω) R u (P.Xplur H E plan tab vb R τ k Rw)
      (tabExt H E plan tab R τ ω u bs)
  · obtain ⟨hc, hall⟩ := hL
    have hspec := Classical.choose_spec hc
    obtain ⟨g, hg, hdec⟩ := hall (Classical.choose hc) hspec.1 hspec.2
    have hcand : cand H E plan tab R τ ω u bs = some (Classical.choose hc) := by
      unfold cand; rw [dite_eq_left_of_eq_true (eq_true hc)]
    have hop : P.OpensAt H E plan tab vb R τ ω bs (vb.pos g)
        (vb.val (vb.opening (L.draw ω) R u (Classical.choose hc) (vb.pos g)))
        (vb.opening (L.draw ω) R u (Classical.choose hc) (vb.pos g)) :=
      ⟨u, hu, ⟨g, hg, rfl⟩, _, hcand, rfl, rfl⟩
    have hmem : vb.val (vb.opening (L.draw ω) R u (Classical.choose hc) (vb.pos g)) ∈
        P.readsAt H E plan tab vb R τ ω bs (vb.pos g) := by
      unfold readsAt; exact Finset.mem_filter.2 ⟨Finset.mem_univ _, _, hop⟩
    have hbit := (vb.opening_spec (L.draw ω) R u hu (Classical.choose hc) hspec.2 g hg).2
    have hne : vb.val (vb.opening (L.draw ω) R u (Classical.choose hc) (vb.pos g)) ≠
        P.Xval H E plan tab vb R τ k Rw (vb.pos g) := by
      intro heq
      apply hdec
      rw [← hbit, heq]
      rfl
    have hpsi : P.psi H E plan tab vb R τ k Rw (vb.pos g) ω bs = 1 := by
      unfold psi; rw [ite_eq_left ⟨_, hmem, hne⟩]
    have h1 : ind (fun bs => P.LinkEvB plan tab decode (L.draw ω) R u
        (P.Xplur H E plan tab vb R τ k Rw) (tabExt H E plan tab R τ ω u bs)) bs = 1 := by
      unfold ind; rw [ite_eq_left ⟨hc, hall⟩]
    rw [h1, ← hpsi]
    exact Finset.single_le_sum (fun c _ => P.psi_nonneg H E plan tab vb R τ k Rw c ω bs)
      (Finset.mem_univ (vb.pos g))
  · have h0 : ind (fun bs => P.LinkEvB plan tab decode (L.draw ω) R u
        (P.Xplur H E plan tab vb R τ k Rw) (tabExt H E plan tab R τ ω u bs)) bs = 0 := by
      unfold ind; rw [ite_eq_right hL]
    rw [h0]
    exact hsum0

theorem avg_ofReal {Cc : Type} [Fintype Cc] [Nonempty Cc] (f : Cc → ℝ) (hf : ∀ c, 0 ≤ f c) :
    avg (fun c => ENNReal.ofReal (f c)) = ENNReal.ofReal ((∑ c, f c) / Fintype.card Cc) := by
  have hc : (0 : ℝ) < Fintype.card Cc := by exact_mod_cast Fintype.card_pos
  unfold avg
  rw [div_eq_mul_inv, Finset.sum_mul, ENNReal.ofReal_sum_of_nonneg fun c _ =>
    mul_nonneg (hf c) (inv_nonneg.2 hc.le)]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [ENNReal.ofReal_mul (hf c), ENNReal.ofReal_inv_of_pos hc, ENNReal.ofReal_natCast, mul_comm]

/-- **A2 for one prover's link finders** (`Assumptions.SHA512CRExpected`): at the registration `R` and the
prover's continuation `τ`, for the finder built for each commit string a draw can read. A2 is a hypothesis about one
finder, since some finder outputs a collision after no evaluations: a prover that hard-codes two openings of one commit
string has link finders that find one. So the link theorem asks A2 only of the finders of the prover it bounds. -/
def LinkCR (k Rw M : ℕ) (t' : ℝ) : Prop :=
  ∀ (c : vb.Pos) (_ : Nonempty (P.Ωc plan tab vb L c)), Assumptions.SHA512CRExpected Hc
    (finderG (P.trialG H E plan tab vb R τ c k Rw) M)
    (finderStrat _ (P.trialStrat H E plan tab vb R τ c k Rw) M) (P.outC H E plan tab vb R τ c)
    (finderCost (P.rdT H E plan tab vb R τ c) (cT plan R t'))

open Classical in
/-- **The link bound, prover by prover**: `Q_s/(1 − ρ)·(2t'(1 + k)/2^256 + 1/(eM) + k/(eR_w))` at a prover whose link
finders satisfy A2 (`LinkCR`), and `⊤` at any other. -/
noncomputable def linkBoundCR (k Rw M : ℕ) (ρ t' : ℝ) (R : Reg) (τ : Cont L Reg (batchedSession H E plan) R) :
    ℝ≥0∞ :=
  if P.LinkCR H E plan tab vb R τ k Rw M t' then
    ENNReal.ofReal ((∑ c : vb.Pos, (Fintype.card (P.Ωc plan tab vb L c) : ℝ) / Fintype.card L.Ω) *
      (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))
  else ⊤

/-- **The link theorem: `δ_link` under expected-time collision resistance, prover by prover.** For the batched audit
with the expected-time extractor (`analysisBE`) and the plurality value layer (`Xplur`), given the binding of wires to
commit strings (`ValueBinding`): at every prover whose explicit finder of each commit string satisfies A2 (`LinkCR`),
$$\delta_{\rm link}\le\frac{Q_s}{1-\rho}\Big(\frac{2t'(1+k)}{2^{256}}+\frac{1}{eM}+\frac{k}{eR_w}\Big),$$
with `Q_s = Σ_c π_c` the expected number of commit strings the drawn units read, `ρ` a bound on every table's `N₀/(ek)`,
`t'` the SHA-512 evaluations of one session run, `M` the finder's further trials and `Rw` the waits' length. At any other
prover the bound is `⊤` (`linkBoundCR`). -/
theorem flock_batched_linkSoundE
    (hlow : P.LoweringSoundB plan tab decode) {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw)
    (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1) (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ}
    (ht : 0 ≤ t') :
    (P.analysisBE (L := L) H E plan tab decode (fun R τ => P.Xplur H E plan tab vb R τ k Rw) k hk
        (fun S R j => (hr S R j).trans_lt hρ) hlow).LinkSound
      (P.linkBoundCR H E plan tab vb k Rw M ρ t') := by
  intro R τ tgt
  unfold linkBoundCR
  split_ifs with hCR
  swap
  · exact le_top
  set B := 2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)
    with hBdef
  have hρ1 : 0 < 1 - ρ := by linarith
  set f : vb.Pos → L.Ω → ℝ := fun c ω => expect (ind (accω plan R ω)) (gameω H E plan R τ ω)
      (stratω H E plan R τ ω).2 *
    expectAcc (gameω H E plan R τ ω) (stratω H E plan R τ ω).2 (accω plan R ω) k
      (P.psi H E plan tab vb R τ k Rw c ω) with hf
  set Λ : L.Ω → ℝ := fun ω => (∑ c, f c ω) / (1 - ρ) with hΛ
  have hf0 : ∀ c ω, 0 ≤ f c ω := fun c ω => mul_nonneg (expect_nonneg (ind_nonneg _) _ _)
    (expectAcc_nonneg _ _ _ _ (P.psi_nonneg H E plan tab vb R τ k Rw c ω))
  have hΛ0 : ∀ ω, 0 ≤ Λ ω := fun ω =>
    div_nonneg (Finset.sum_nonneg fun c _ => hf0 c ω) hρ1.le
  have step1 : ∀ ω, atTgt (L.draw ω) (tgt ω) (fun u =>
      (P.analysisBE (L := L) H E plan tab decode (fun R τ => P.Xplur H E plan tab vb R τ k Rw) k hk
        (fun S R j => (hr S R j).trans_lt hρ) hlow).link (L.draw ω) R (τ ω) u
        ((P.analysisBE (L := L) H E plan tab decode (fun R τ => P.Xplur H E plan tab vb R τ k Rw) k hk
          (fun S R j => (hr S R j).trans_lt hρ) hlow).committed R τ)) ≤ ENNReal.ofReal (Λ ω) := by
    intro ω
    refine atTgt_le (fun u hu => ?_) (tgt ω)
    show ENNReal.ofReal (accProb H E plan (τ ω) *
        extractProbAccB execArith H E (plan (L.draw ω) R) (tab (L.draw ω) R u) k
          (batchStrat H E plan (τ ω))
          (P.LinkEvB plan tab decode (L.draw ω) R u (P.Xplur H E plan tab vb R τ k Rw)) /
        (1 - rateB (plan (L.draw ω) R) (tab (L.draw ω) R u) k)) ≤ ENNReal.ofReal (Λ ω)
    apply ENNReal.ofReal_le_ofReal
    have hε0 : 0 ≤ accProb H E plan (τ ω) := expect_nonneg (ind_nonneg _) _ _
    have hLp0 := extractProbAccB_nonneg execArith H E (plan (L.draw ω) R) (tab (L.draw ω) R u) k
      (batchStrat H E plan (τ ω))
      (P.LinkEvB plan tab decode (L.draw ω) R u (P.Xplur H E plan tab vb R τ k Rw))
    have hLp : extractProbAccB execArith H E (plan (L.draw ω) R) (tab (L.draw ω) R u) k
        (batchStrat H E plan (τ ω))
        (P.LinkEvB plan tab decode (L.draw ω) R u (P.Xplur H E plan tab vb R τ k Rw)) ≤
          expectAcc (gameω H E plan R τ ω) (stratω H E plan R τ ω).2 (accω plan R ω) k
            (fun bs => ∑ c, P.psi H E plan tab vb R τ k Rw c ω bs) :=
      expectAcc_mono _ _ _ _ fun bs => P.linkEv_le_psi H E plan tab vb R τ k Rw hu bs
    have hrr : 1 - ρ ≤ 1 - rateB (plan (L.draw ω) R) (tab (L.draw ω) R u) k := by
      linarith [hr (L.draw ω) R (tab (L.draw ω) R u)]
    have hsum : expect (ind (accω plan R ω)) (gameω H E plan R τ ω) (stratω H E plan R τ ω).2 *
        expectAcc (gameω H E plan R τ ω) (stratω H E plan R τ ω).2 (accω plan R ω) k
          (fun bs => ∑ c, P.psi H E plan tab vb R τ k Rw c ω bs) = ∑ c, f c ω := by
      rw [expectAcc_sum, Finset.mul_sum]
    calc accProb H E plan (τ ω) *
          extractProbAccB execArith H E (plan (L.draw ω) R) (tab (L.draw ω) R u) k
            (batchStrat H E plan (τ ω))
            (P.LinkEvB plan tab decode (L.draw ω) R u (P.Xplur H E plan tab vb R τ k Rw)) /
          (1 - rateB (plan (L.draw ω) R) (tab (L.draw ω) R u) k)
        ≤ accProb H E plan (τ ω) *
          extractProbAccB execArith H E (plan (L.draw ω) R) (tab (L.draw ω) R u) k
            (batchStrat H E plan (τ ω))
            (P.LinkEvB plan tab decode (L.draw ω) R u (P.Xplur H E plan tab vb R τ k Rw)) /
          (1 - ρ) := div_le_div_of_nonneg_left (mul_nonneg hε0 hLp0) hρ1 hrr
      _ ≤ expect (ind (accω plan R ω)) (gameω H E plan R τ ω) (stratω H E plan R τ ω).2 *
          expectAcc (gameω H E plan R τ ω) (stratω H E plan R τ ω).2 (accω plan R ω) k
            (fun bs => ∑ c, P.psi H E plan tab vb R τ k Rw c ω bs) / (1 - ρ) :=
          div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hLp hε0) hρ1.le
      _ = Λ ω := by rw [hsum]
  refine (avg_mono step1).trans ?_
  rw [avg_ofReal Λ hΛ0]
  apply ENNReal.ofReal_le_ofReal
  have hΩ : (0 : ℝ) < Fintype.card L.Ω := by exact_mod_cast Fintype.card_pos
  have hc : ∀ c, (∑ ω, f c ω) / Fintype.card L.Ω ≤
      (Fintype.card (P.Ωc plan tab vb L c) : ℝ) / Fintype.card L.Ω * B := fun c =>
    P.ideal_mass_le H E plan tab vb R τ k Rw M hRw hM ht c (hCR c)
  calc (∑ ω, Λ ω) / Fintype.card L.Ω
      = (∑ c, (∑ ω, f c ω) / Fintype.card L.Ω) / (1 - ρ) := by
        simp only [hΛ, div_eq_mul_inv, ← Finset.sum_mul]
        rw [Finset.sum_comm]
        ring
    _ ≤ (∑ c, (Fintype.card (P.Ωc plan tab vb L c) : ℝ) / Fintype.card L.Ω * B) / (1 - ρ) :=
        div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ => hc c) hρ1.le
    _ = _ := by rw [← Finset.sum_mul]

end Partition

end FlockSoundness.Audit
