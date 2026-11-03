module

public import Proofs.Flock.SampledGeneric.JointTransfers
public import Proofs.Flock.Recursive.Multi

/-!
# The sampled audit for a proof system with one session per sampled proof block

Flock proves a sample in one batched session. A proof system may instead prove each sampled proof block in a session
of its own (`BlockSystem`). The compiler runs the sampled proof blocks' sessions in a row, in increasing order, each
with its own live coins, and accepts when every one accepts (`BlockSystem.seq`, a `ProofSystem`). The prover names its
strategy for each session at that session's start, after every session before it (`Recursion.seqC`), so the row is
played by any prover of the sessions, adaptive to the earlier sessions' coins.

The composition's knowledge extractor and trials are built from each proof block's (`BlockExtractor`, `BlockTrials`):

* `BlockExtractor.seq`: at a sampled proof block, its own session's knowledge term and binding error at the strategy
  the prover names there, in expectation over the row. They cover acceptance (`BlockExtractor.seq_covers`): the row
  accepts only when that session does.
* `BlockTrials.seq`: the trial at a position `c` draws a sample and a sampled proof block that reads `c`, uniform among
  such pairs (`BlockTrials.PairsAt`), plays the sample's sessions, then the proof block's own trial at `c` from the
  strategy named for its session. Its weight `wt c` is the expected number of sampled proof blocks that read `c`. Its
  cost, recorded in its output, is the proof block's trial's plus the sessions' (`sessCost`).
* `BlockTrials.seq_transfers`: the binding error at the picked proof block is at most the sum over the sample's proof
  blocks (`BlockTrials.link_seq_le`), and averaged over the sample that sum is `Σ_c wt_c (errOf_c + slack)/(1 − ρ)`
  (`BlockTrials.sum_wt_errOf_seq`, an exchange of sums over pairs of a sampled proof block and a position it reads).

Each proof block's extractor reruns its own session at its place in the row, so two proof blocks' readings of one
position come from different trials; the weight counts each sampled proof block that reads a position once.

`Flock.SecurityProofs.GenericSampledKSBlocks` is `GenericSampledKSPa` at the composition: a proof system with one
session per sampled proof block instantiates the compiler with no change to its objects. `Toy` is an instance with
perfect per-block soundness, where every hypothesis holds, ecr/sha-512 included, and the theorem gives
`Pr[accept] ≤ P_a({u : R u = false}) + ε_bind` (`Toy.bound`).
-/

@[expose] public section

namespace FlockSoundness.SampledGeneric

open Game Audit LinkFinder Discharge.Recursion
open scoped ENNReal

variable {n : ℕ} {Reg : Type} {Dh : Type} {Hc : List UInt8 → Dh}

/-! ## Expectations -/

theorem exG_ofReal {α : Type} (F : α → ℝ) (hF : ∀ a, 0 ≤ F a) :
    ∀ (g : Game α) (s : Strategy g), exG (fun a => ENNReal.ofReal (F a)) g s = ENNReal.ofReal (expect F g s)
  | .ret _, _ => rfl
  | .send _ k, s => exG_ofReal F hF (k s.1) s.2
  | @Game.coin _ _ _ _ k, s =>
      (congrArg avg (funext fun c => exG_ofReal F hF (k c) (s c))).trans
        (avg_ofReal_eq (fun c => expect F (k c) (s c)) fun _ => expect_nonneg hF _ _)

theorem expect_sum_add_div {α ι : Type} (t : Finset ι) (f : ι → α → ℝ) (k d : ℝ) (g : Game α) (s : Strategy g) :
    expect (fun a => (∑ i ∈ t, (f i a + k)) / d) g s = (∑ i ∈ t, (expect (f i) g s + k)) / d := by
  simp only [div_eq_inv_mul]
  rw [expect_const_mul, expect_sum]
  congr 1
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [expect_add, expect_const]

/-! ## The sample in increasing order -/

/-- **The `j`-th proof block of the sample `S`**, in increasing order. -/
def blockAt (S : Finset (Fin n)) (j : Fin S.card) : Fin n := (S.orderIsoOfFin rfl j).1

/-- **The place of the sampled proof block `u` in the sample `S`**, in increasing order. -/
def placeOf (S : Finset (Fin n)) (u : Fin n) (h : u ∈ S) : Fin S.card := (S.orderIsoOfFin rfl).symm ⟨u, h⟩

theorem blockAt_placeOf (S : Finset (Fin n)) (u : Fin n) (h : u ∈ S) : blockAt S (placeOf S u h) = u :=
  congrArg Subtype.val ((S.orderIsoOfFin rfl).apply_symm_apply ⟨u, h⟩)

theorem blockAt_mem (S : Finset (Fin n)) (j : Fin S.card) : blockAt S j ∈ S := (S.orderIsoOfFin rfl j).2

/-- Every session of a row accepted. -/
def allAccept {k : ℕ} {T : Fin k → Type} (rs : (j : Fin k) → T j × Bool) : Bool := decide (∀ j, (rs j).2 = true)

theorem allAccept_eq_true {k : ℕ} {T : Fin k → Type} {rs : (j : Fin k) → T j × Bool} :
    allAccept rs = true ↔ ∀ j, (rs j).2 = true := by
  unfold allAccept
  exact decide_eq_true_iff

/-! ## A proof system with one session per sampled proof block -/

/-- **A proof system that proves each sampled proof block in its own session**: proof block `u`'s session on the
sample `S` against the registration `R`, which accepts or rejects. -/
structure BlockSystem (n : ℕ) (Reg : Type) where
  session : Finset (Fin n) → Reg → Fin n → Game Bool

namespace BlockSystem

variable (BS : BlockSystem n Reg)

/-- What the row of sessions on a sample records: for each sampled proof block, in increasing order, the strategy the
prover named for its session and the session's verdict. -/
abbrev Rec (S : Finset (Fin n)) (R : Reg) : Type :=
  (j : Fin S.card) → Strategy (BS.session S R (blockAt S j)) × Bool

/-- **The sampled proof blocks' sessions in a row**, in increasing order, each with its own live coins and played by a
strategy the prover names at its start, after every session before it (`Recursion.seqC`). -/
def sessions (S : Finset (Fin n)) (R : Reg) : Game (BS.Rec S R) :=
  seqC S.card (T := fun j => Strategy (BS.session S R (blockAt S j))) (α := fun _ => Bool)
    (fun j => BS.session S R (blockAt S j)) fun _ t => t

/-- **The compiled proof system**: the sampled proof blocks' sessions in a row, accepting when every one accepts. -/
def seq : ProofSystem n Reg where
  session S R := (BS.sessions S R).map allAccept

open Classical in
/-- **A bound on one session, at each of its strategies, bounds it in the row** at the strategy the prover names
there (`Recursion.exG_seqC_le`). -/
theorem prob_sessions_le (S : Finset (Fin n)) (R : Reg) (j : Fin S.card) (E : Bool → Prop)
    (F : Strategy (BS.session S R (blockAt S j)) → ℝ≥0∞)
    (hF : ∀ t, prob E (BS.session S R (blockAt S j)) t ≤ F t) (s : Strategy (BS.sessions S R)) :
    prob (fun rs => E (rs j).2) (BS.sessions S R) s ≤ exG (fun rs => F (rs j).1) (BS.sessions S R) s := by
  rw [prob_eq_exG]
  exact exG_seqC_le S.card (T := fun j => Strategy (BS.session S R (blockAt S j))) (α := fun _ => Bool)
    (fun j => BS.session S R (blockAt S j)) (fun _ t => t) j (fun r => if E r.2 then 1 else 0) F
    (fun t => (prob_eq_exG E _ t).symm.trans_le (hF t)) s

end BlockSystem

/-! ## The per-block extractor and its composition -/

/-- **The knowledge extractor's guarantee at a proof block's own session**: from a prover's strategy `t` for proof
block `u`'s session on the sample `S` against `R`, `ks S R u t` bounds `Pr[the session accepts ∧ extraction of u's
witness fails]`, and `link S R u t X` bounds `Pr[it accepts ∧ the extracted witness opens a position u reads to a value
other than X's]`. -/
structure BlockExtractor (BS : BlockSystem n Reg) (B : Binding Reg Hc) where
  ks : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → Strategy (BS.session S R u) → ℝ≥0∞
  link : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → Strategy (BS.session S R u) → B.Trace → ℝ≥0∞

namespace BlockExtractor

variable {BS : BlockSystem n Reg} {B : Binding Reg Hc} (BE : BlockExtractor BS B)

/-- **The two terms cover acceptance of each session**: at every sampled proof block incorrect in a trace, its session
accepts with probability at most the knowledge term plus the binding error, from every strategy. -/
def Covers (Correct : Reg → B.Trace → Fin n → Prop) : Prop :=
  ∀ S R u (t : Strategy (BS.session S R u)) (Xv : B.Trace), u ∈ S → ¬ Correct R Xv u →
    prob (· = true) (BS.session S R u) t ≤ BE.ks S R u t + BE.link S R u t Xv

/-- **The composition's extractor**: at a sampled proof block, its own session's knowledge term and binding error at the
strategy the prover names there, in expectation over the row of sessions; `0` off the sample. -/
noncomputable def seq : Extractor BS.seq B where
  ks S R τ u := if h : u ∈ S then
      exG (fun rs => BE.ks S R (blockAt S (placeOf S u h)) (rs (placeOf S u h)).1) (BS.sessions S R)
        (Strategy.ofMap allAccept (BS.sessions S R) τ)
    else 0
  link S R τ u Xv := if h : u ∈ S then
      exG (fun rs => BE.link S R (blockAt S (placeOf S u h)) (rs (placeOf S u h)).1 Xv) (BS.sessions S R)
        (Strategy.ofMap allAccept (BS.sessions S R) τ)
    else 0

theorem seq_ks {S : Finset (Fin n)} {R : Reg} (τ : Strategy (BS.seq.session S R)) {u : Fin n} (h : u ∈ S) :
    BE.seq.ks S R τ u = exG (fun rs => BE.ks S R (blockAt S (placeOf S u h)) (rs (placeOf S u h)).1)
      (BS.sessions S R) (Strategy.ofMap allAccept (BS.sessions S R) τ) := dite_eq_left h

theorem seq_link {S : Finset (Fin n)} {R : Reg} (τ : Strategy (BS.seq.session S R)) {u : Fin n} (h : u ∈ S)
    (Xv : B.Trace) :
    BE.seq.link S R τ u Xv = exG (fun rs => BE.link S R (blockAt S (placeOf S u h)) (rs (placeOf S u h)).1 Xv)
      (BS.sessions S R) (Strategy.ofMap allAccept (BS.sessions S R) τ) := dite_eq_left h

/-- **The composition's terms cover acceptance**: the row accepts only when the sampled proof block's own session
does, and that session's terms cover it at the strategy the prover names there. -/
theorem seq_covers {Correct : Reg → B.Trace → Fin n → Prop} (h : BE.Covers Correct) : BE.seq.Covers Correct := by
  intro S R τ u Xv hu hX
  have hX' : ¬ Correct R Xv (blockAt S (placeOf S u hu)) := by rw [blockAt_placeOf]; exact hX
  calc prob (· = true) (BS.seq.session S R) τ
      = prob (fun rs => allAccept rs = true) (BS.sessions S R) (Strategy.ofMap allAccept (BS.sessions S R) τ) :=
        prob_map _ _ _ _
    _ ≤ prob (fun rs => (rs (placeOf S u hu)).2 = true) (BS.sessions S R)
          (Strategy.ofMap allAccept (BS.sessions S R) τ) :=
        prob_mono (fun rs (hrs : allAccept rs = true) => allAccept_eq_true.1 hrs (placeOf S u hu)) _ _
    _ ≤ exG (fun rs => BE.ks S R (blockAt S (placeOf S u hu)) (rs (placeOf S u hu)).1 +
          BE.link S R (blockAt S (placeOf S u hu)) (rs (placeOf S u hu)).1 Xv) (BS.sessions S R)
          (Strategy.ofMap allAccept (BS.sessions S R) τ) :=
        BS.prob_sessions_le S R (placeOf S u hu) (· = true) _
          (fun t => h S R _ t Xv (blockAt_mem S _) hX') _
    _ = BE.seq.ks S R τ u + BE.seq.link S R τ u Xv := by
        rw [exG_add, BE.seq_ks τ hu, BE.seq_link τ hu]

/-- **The knowledge term the audit pays**, from a bound `κ` on each session's: at most the largest `κ` over the
sample, averaged over the sample. -/
theorem ksAvg_seq_le (κ : Finset (Fin n) → Reg → Fin n → ℝ≥0∞) (hks : ∀ S R u t, u ∈ S → BE.ks S R u t ≤ κ S R u)
    (L : Law n) (R : Reg) (τ : Cont L Reg BS.seq.session R) :
    BE.seq.ksAvg L R τ ≤ avg fun ω => ⨆ u ∈ L.draw ω, κ (L.draw ω) R u :=
  avg_mono fun ω => iSup₂_mono fun u hu => by
    rw [BE.seq_ks (τ ω) hu]
    refine (exG_mono (fun rs : BS.Rec (L.draw ω) R => hks _ R _ (rs (placeOf _ u hu)).1 (blockAt_mem _ _)) _ _).trans
      (le_of_eq ?_)
    rw [exG_const, blockAt_placeOf]

end BlockExtractor

/-! ## The per-block trials and their composition -/

/-- **A proof block's trials**: the positions proof block `u` reads (`reads u`), and at a prover's strategy `t` for
its session on the sample `S` against `R` and a position `c`, a game the strategy plays (`T`, by `sT`): the extractor
rerun from `t`. It reads values at `c` (`rd`), each with an opening (`op`), and costs `cost` SHA-512 calls; one session
of `u` played by `t` costs `sessCost` (the prover's calls included). `ρ` and `slack` are what the extractor loses to its
waits. -/
structure BlockTrials (BS : BlockSystem n Reg) (B : Binding Reg Hc) where
  Out : Type
  reads : Fin n → Finset B.Pos
  T : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → Strategy (BS.session S R u) → B.Pos → Game Out
  sT : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → (t : Strategy (BS.session S R u)) → (c : B.Pos) →
    Strategy (T S R u t c)
  rd : Finset (Fin n) → Reg → Fin n → B.Pos → Out → Finset B.Val
  op : Finset (Fin n) → Reg → Fin n → B.Pos → Out → B.Val → B.Opening
  cost : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → Strategy (BS.session S R u) → Out → ℝ
  sessCost : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → Strategy (BS.session S R u) → ℝ
  ρ : ℝ
  slack : ℝ

namespace BlockTrials

variable {BS : BlockSystem n Reg} {B : Binding Reg Hc} (BT : BlockTrials BS B)

/-- **Each reading comes with an opening of the registered commitment** to the value read. -/
def Opens : Prop :=
  ∀ S R u c o v, v ∈ BT.rd S R u c o → B.com (BT.op S R u c o v) = B.cs R c ∧ B.val (BT.op S R u c o v) = v

/-- **The trials' numbers are in range**: costs nonnegative, `ρ < 1`, and the slack nonnegative. -/
def WellFormed : Prop :=
  (∀ S R u t o, 0 ≤ BT.cost S R u t o) ∧ (∀ S R u t, 0 ≤ BT.sessCost S R u t) ∧ BT.ρ < 1 ∧ 0 ≤ BT.slack

/-- **The transfer from each session's binding error to its trials**: at every sampled proof block and every strategy
for its session, the binding error against a trace `X` is at most `Σ_{c ∈ reads u} (errAt_c(X c) + slack)/(1 − ρ)`. -/
def Transfers (BE : BlockExtractor BS B) : Prop :=
  ∀ S R u (t : Strategy (BS.session S R u)) (Xv : B.Trace), u ∈ S →
    BE.link S R u t Xv ≤ ENNReal.ofReal ((∑ c ∈ BT.reads u,
      (errAt (BT.T S R u t c) (BT.sT S R u t c) (BT.rd S R u c) (Xv c) + BT.slack)) / (1 - BT.ρ))

/-- The pairs of a sample and a sampled proof block that reads `c`. -/
abbrev PairsAt (L : Law n) (c : B.Pos) : Type := {p : L.Ω × Fin n // p.2 ∈ L.draw p.1 ∧ c ∈ BT.reads p.2}

open Classical in
noncomputable instance instFintypePairsAt (L : Law n) (c : B.Pos) : Fintype (BT.PairsAt L c) := Subtype.fintype _

/-- **The trial at `c` from the pair `(ω, u)`**: the sample's sessions in a row, then `u`'s own trial at `c` from the
strategy the prover named for `u`'s session. It outputs the pair, the trial's output and its cost: the trial's plus the
sessions'. -/
noncomputable def pairG (L : Law n) (R : Reg) (c : B.Pos) (ω : L.Ω) (u : Fin n) (h : u ∈ L.draw ω) :
    Game ((L.Ω × Fin n) × BT.Out × ℝ) :=
  (BS.sessions (L.draw ω) R).bind fun rs =>
    (BT.T (L.draw ω) R (blockAt _ (placeOf _ u h)) (rs (placeOf _ u h)).1 c).map fun o =>
      ((ω, blockAt _ (placeOf _ u h)), o, BT.cost (L.draw ω) R _ (rs (placeOf _ u h)).1 o +
        ∑ j, BT.sessCost (L.draw ω) R (blockAt _ j) (rs j).1)

/-- The prover's state at registration plays the sessions; the trial is played by `sT`. -/
noncomputable def pairS (L : Law n) (R : Reg) (τ : Cont L Reg BS.seq.session R) (c : B.Pos) (ω : L.Ω) (u : Fin n)
    (h : u ∈ L.draw ω) : Strategy (BT.pairG L R c ω u h) :=
  Strategy.bind _ _ (Strategy.ofMap allAccept (BS.sessions (L.draw ω) R) (τ ω)) fun rs =>
    Strategy.map _ _ (BT.sT (L.draw ω) R (blockAt _ (placeOf _ u h)) (rs (placeOf _ u h)).1 c)

/-- **The composition's trials** at the sampling design `L`: at `c`, a pair of a sample and a sampled proof block that
reads `c`, uniform among those, then that pair's trial (`pairG`); weight the expected number of sampled proof blocks
that read `c`, and the per-block `ρ` and slack. -/
noncomputable def seq (L : Law n) : Trials L BS.seq B where
  Out := (L.Ω × Fin n) × BT.Out × ℝ
  Read c := Nonempty (BT.PairsAt L c)
  T R _ c h := haveI : Nonempty (BT.PairsAt L c) := h
    (Game.draw (BT.PairsAt L c)).bind fun p => BT.pairG L R c p.1.1 p.1.2 p.2.1
  sT R τ c h := haveI : Nonempty (BT.PairsAt L c) := h
    Strategy.bind _ _ (drawStrat _) fun p => BT.pairS L R τ c p.1.1 p.1.2 p.2.1
  rd R _ c e := BT.rd (L.draw e.1.1) R e.1.2 c e.2.1
  op R _ c e v := BT.op (L.draw e.1.1) R e.1.2 c e.2.1 v
  cost _ _ e := max e.2.2 0
  wt c := (Fintype.card (BT.PairsAt L c) : ℝ) / Fintype.card L.Ω
  ρ := BT.ρ
  slack := BT.slack

theorem seq_opens (h : BT.Opens) (L : Law n) : (BT.seq L).Opens :=
  fun R _ c _ v hv => h _ R _ c _ v hv

theorem seq_wellFormed (h : BT.WellFormed) (L : Law n) : (BT.seq L).WellFormed :=
  ⟨fun _ _ _ => le_max_right _ _, fun _ => div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _), h.2.2.1, h.2.2.2⟩

/-- **The binding error of the pair `(ω, u)` at `c`**: `u`'s trial's, in expectation over the row of sessions on the
sample `ω`; `0` when `u` is not sampled. -/
noncomputable def pairErr (L : Law n) (R : Reg) (τ : Cont L Reg BS.seq.session R) (Xv : B.Trace) (ω : L.Ω)
    (u : Fin n) (c : B.Pos) : ℝ :=
  if h : u ∈ L.draw ω then
    expect (fun rs : BS.Rec (L.draw ω) R =>
        errAt (BT.T (L.draw ω) R (blockAt _ (placeOf _ u h)) (rs (placeOf _ u h)).1 c)
          (BT.sT (L.draw ω) R (blockAt _ (placeOf _ u h)) (rs (placeOf _ u h)).1 c)
          (BT.rd (L.draw ω) R (blockAt _ (placeOf _ u h)) c) (Xv c))
      (BS.sessions (L.draw ω) R) (Strategy.ofMap allAccept (BS.sessions (L.draw ω) R) (τ ω))
  else 0

theorem pairErr_of_mem (L : Law n) (R : Reg) (τ : Cont L Reg BS.seq.session R) (Xv : B.Trace) {ω : L.Ω}
    {u : Fin n} (h : u ∈ L.draw ω) (c : B.Pos) :
    BT.pairErr L R τ Xv ω u c =
      expect (fun rs : BS.Rec (L.draw ω) R =>
          errAt (BT.T (L.draw ω) R (blockAt _ (placeOf _ u h)) (rs (placeOf _ u h)).1 c)
            (BT.sT (L.draw ω) R (blockAt _ (placeOf _ u h)) (rs (placeOf _ u h)).1 c)
            (BT.rd (L.draw ω) R (blockAt _ (placeOf _ u h)) c) (Xv c))
        (BS.sessions (L.draw ω) R) (Strategy.ofMap allAccept (BS.sessions (L.draw ω) R) (τ ω)) :=
  dite_eq_left h

theorem pairErr_nonneg (L : Law n) (R : Reg) (τ : Cont L Reg BS.seq.session R) (Xv : B.Trace) (ω : L.Ω)
    (u : Fin n) (c : B.Pos) : 0 ≤ BT.pairErr L R τ Xv ω u c := by
  unfold pairErr
  split_ifs
  · exact expect_nonneg (fun _ => errAt_nonneg _ _ _ _) _ _
  · exact le_rfl

/-- **The binding error at a sampled proof block** is at most its pairs' binding errors at the positions it reads, by
its own session's transfer at the strategy the prover names there. -/
theorem link_seq_le {BE : BlockExtractor BS B} (hwf : BT.WellFormed) (hT : BT.Transfers BE) (L : Law n) (R : Reg)
    (τ : Cont L Reg BS.seq.session R) (Xv : B.Trace) (ω : L.Ω) {u : Fin n} (h : u ∈ L.draw ω) :
    BE.seq.link (L.draw ω) R (τ ω) u Xv ≤
      ENNReal.ofReal ((∑ c ∈ BT.reads u, (BT.pairErr L R τ Xv ω u c + BT.slack)) / (1 - BT.ρ)) := by
  rw [BE.seq_link (τ ω) h]
  refine (exG_mono (fun rs : BS.Rec (L.draw ω) R =>
    hT _ R _ (rs (placeOf _ u h)).1 Xv (blockAt_mem _ _)) _ _).trans (le_of_eq ?_)
  rw [exG_ofReal, expect_sum_add_div, congrArg BT.reads (blockAt_placeOf _ u h)]
  · refine congrArg (fun x => ENNReal.ofReal (x / (1 - BT.ρ))) (Finset.sum_congr rfl fun c _ => ?_)
    rw [BT.pairErr_of_mem L R τ Xv h c]
  · exact fun rs => div_nonneg (Finset.sum_nonneg fun c _ => add_nonneg (errAt_nonneg _ _ _ _) hwf.2.2.2)
      (sub_nonneg.2 hwf.2.2.1.le)

/-- **The composed trials' binding error at a position some sample reads**: its pairs' binding errors, averaged over
the pairs (`LinkFinder.errAt_eq_expect`). -/
theorem errOf_seq (L : Law n) (R : Reg) (τ : Cont L Reg BS.seq.session R) (Xv : B.Trace) (c : B.Pos)
    (h : Nonempty (BT.PairsAt L c)) :
    (BT.seq L).errOf R τ c (Xv c) =
      (∑ p : BT.PairsAt L c, BT.pairErr L R τ Xv p.1.1 p.1.2 c) / Fintype.card (BT.PairsAt L c) := by
  unfold Trials.errOf
  split_ifs with h'
  · rw [errAt_eq_expect]
    refine (expect_bind _ (Game.draw (BT.PairsAt L c)) (fun p => BT.pairG L R c p.1.1 p.1.2 p.2.1)
      (drawStrat _) (fun p => BT.pairS L R τ c p.1.1 p.1.2 p.2.1)).trans
      ((expect_draw _ _).trans (congrArg (· / (Fintype.card (BT.PairsAt L c) : ℝ))
        (Finset.sum_congr rfl fun p _ => ?_)))
    rw [BT.pairErr_of_mem L R τ Xv p.2.1 c]
    unfold pairG pairS
    refine (expect_bind _ _ _ _ _).trans ?_
    refine congrArg (fun f => expect f _ _) (funext fun rs => ?_)
    exact (expect_map _ _ _ _).trans (errAt_eq_expect _ _ _ _).symm
  · exact absurd h h'

/-- **The composed trials' weighted binding error** is the pairs' binding errors at the positions each sampled proof
block reads, summed over the sample and averaged over the sample. -/
theorem sum_wt_errOf_seq (L : Law n) (R : Reg) (τ : Cont L Reg BS.seq.session R) (Xv : B.Trace) :
    ∑ c, (BT.seq L).wt c * ((BT.seq L).errOf R τ c (Xv c) + BT.slack) =
      (∑ ω, ∑ u ∈ L.draw ω, ∑ c ∈ BT.reads u, (BT.pairErr L R τ Xv ω u c + BT.slack)) / Fintype.card L.Ω := by
  have hc : ∀ c, (BT.seq L).wt c * ((BT.seq L).errOf R τ c (Xv c) + BT.slack) =
      (∑ q ∈ Finset.univ.filter (fun q : L.Ω × Fin n => q.2 ∈ L.draw q.1 ∧ c ∈ BT.reads q.2),
        (BT.pairErr L R τ Xv q.1 q.2 c + BT.slack)) / Fintype.card L.Ω := by
    intro c
    rw [Finset.sum_subtype (F := BT.instFintypePairsAt L c)
      (Finset.univ.filter (fun q : L.Ω × Fin n => q.2 ∈ L.draw q.1 ∧ c ∈ BT.reads q.2)) (fun q => by simp)]
    show (Fintype.card (BT.PairsAt L c) : ℝ) / Fintype.card L.Ω * ((BT.seq L).errOf R τ c (Xv c) + BT.slack) = _
    by_cases h : Nonempty (BT.PairsAt L c)
    · have hP : (Fintype.card (BT.PairsAt L c) : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
      rw [BT.errOf_seq L R τ Xv c h, Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
      field_simp
    · have : IsEmpty (BT.PairsAt L c) := not_nonempty_iff.1 h
      simp [Fintype.card_eq_zero]
  rw [Finset.sum_congr rfl fun c _ => hc c, ← Finset.sum_div]
  congr 1
  refine (Finset.sum_comm' (t' := Finset.univ.filter fun q : L.Ω × Fin n => q.2 ∈ L.draw q.1)
    (s' := fun q => BT.reads q.2) fun c q => ?_).trans ?_
  · simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact and_comm
  · rw [Finset.sum_filter, Fintype.sum_prod_type]
    refine Finset.sum_congr rfl fun ω _ => ?_
    rw [← Finset.sum_filter]
    exact Finset.sum_congr (by ext u; simp) fun _ _ => rfl

/-- **The composition transfers its binding error to its trials**: the binding error at the picked proof block is at
most the sum over the sample's proof blocks, which averaged over the sample is `Σ_c wt_c (errOf_c + slack)/(1 − ρ)`. -/
theorem seq_transfers {BE : BlockExtractor BS B} (hwf : BT.WellFormed) (hT : BT.Transfers BE) (L : Law n) :
    (BT.seq L).Transfers BE.seq := by
  intro R τ tgt Xv
  have hQ : ∀ ω u, 0 ≤ (∑ c ∈ BT.reads u, (BT.pairErr L R τ Xv ω u c + BT.slack)) / (1 - BT.ρ) := fun ω u =>
    div_nonneg (Finset.sum_nonneg fun c _ => add_nonneg (BT.pairErr_nonneg L R τ Xv ω u c) hwf.2.2.2)
      (sub_nonneg.2 hwf.2.2.1.le)
  refine (avg_mono fun ω => atTgt_le (fun u hu => (BT.link_seq_le hwf hT L R τ Xv ω hu).trans
    (ENNReal.ofReal_le_ofReal (Finset.single_le_sum (f := fun u => (∑ c ∈ BT.reads u,
      (BT.pairErr L R τ Xv ω u c + BT.slack)) / (1 - BT.ρ)) (fun u _ => hQ ω u) hu))) (tgt ω)).trans
    (le_of_eq ?_)
  rw [avg_ofReal_eq _ fun ω => Finset.sum_nonneg fun u _ => hQ ω u]
  show _ = ENNReal.ofReal ((∑ c, (BT.seq L).wt c * ((BT.seq L).errOf R τ c (Xv c) + BT.slack)) / (1 - BT.ρ))
  rw [BT.sum_wt_errOf_seq L R τ Xv]
  congr 1
  simp only [← Finset.sum_div]
  exact div_right_comm _ _ _

end BlockTrials

end FlockSoundness.SampledGeneric

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.SampledGeneric
open scoped ENNReal

/-- **The sampled audit's knowledge soundness for a proof system with one session per sampled proof block.** For a
proof system that proves each sampled proof block in its own session (`BlockSystem`), the sessions run in a row
(`BlockSystem.seq`), whose per-block extractor's terms cover each session's acceptance (`BlockExtractor.Covers`), with
per-block trials that open the registered commitment (`BlockTrials.Opens`) and carry each session's binding error
(`BlockTrials.Transfers`): at every prover `σ` whose link finders satisfy ecr/sha-512 (`Trials.CR` of the composed
trials `BlockTrials.seq`),
$$\Pr[\text{accept}]\le P_a(F)+\varepsilon_{\rm ks}(\sigma)+\varepsilon_{\rm bind}(\sigma),$$
with `F` the incorrect set of the plurality of the composed trials (`Trials.plur`), `ε_ks` the composed extractor's
knowledge term (`BlockExtractor.seq`, `Extractor.ksAvg`) and `ε_bind` the composed trials' binding term
(`Trials.bindErr`). -/
def GenericSampledKSBlocks : Prop :=
  ∀ (n : ℕ) (Reg Dh : Type) (Hc : List UInt8 → Dh) (L : Law n) (BS : BlockSystem n Reg) (B : Binding Reg Hc)
    (Correct : Reg → B.Trace → Fin n → Prop) (BE : BlockExtractor BS B) (BT : BlockTrials BS B) (M : ℕ),
    BE.Covers Correct → BT.Opens → BT.WellFormed → BT.Transfers BE → 1 ≤ M →
    ∀ σ : Strategy (audit L Reg BS.seq.session), (BT.seq L).CR (reg σ) (cont σ) M →
      prob (fun o => o.1 = true) (audit L Reg BS.seq.session) σ ≤
        L.escape (incorrect B Correct (reg σ) ((BT.seq L).plur (reg σ) (cont σ))) +
          BE.seq.ksAvg L (reg σ) (cont σ) + (BT.seq L).bindErr (reg σ) (cont σ) M

end Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.SampledGeneric in
/-- **Per-block sessions** (`Flock.Guarantees.GenericSampledKSBlocks`): `GenericSampledKSPa` at the composition's
extractor and trials, with the committed trace the plurality of the composed trials. -/
theorem Flock.SecurityProofs.GenericSampledKSBlocks : Flock.Guarantees.GenericSampledKSBlocks := by
  intro n Reg Dh Hc L BS B Correct BE BT M hcov hop hwf hT hM σ hCR
  exact Flock.SecurityProofs.GenericSampledKSPa n Reg Dh Hc L BS.seq B Correct BE.seq (BT.seq L) (BT.seq L).plur M
    (BE.seq_covers hcov) (BT.seq_opens hop L) (BT.seq_wellFormed hwf L) (BT.seq_transfers hwf hT L)
    (BT.seq L).plurality_plur hM σ hCR

/-! ## A toy instance: perfect per-block soundness -/

namespace FlockSoundness.SampledGeneric.Toy

open Game Audit LinkFinder Discharge.Recursion
open scoped ENNReal

variable {n : ℕ} {Dh : Type} {Hc : List UInt8 → Dh}

/-- **The registration in the clear**: one bit per position, which opens to itself; perfectly binding. -/
abbrev binding (n : ℕ) (Hc : List UInt8 → Dh) : Binding (Fin n → Bool) Hc where
  Pos := Fin n
  Val := Bool
  Opening := Bool
  Com := Bool
  val o := o
  com o := o
  cs R c := R c
  collide _ _ := ([], [])
  collide_spec _ _ h h' := absurd h h'

/-- **Proof block `u`'s session**: the prover sends a bit, and the verifier accepts when it is set and so is `u`'s
registered bit. -/
def system (n : ℕ) : BlockSystem n (Fin n → Bool) where
  session _ R u := .send Bool fun b => .ret (b && R u)

/-- Proof block `u` is correct in a trace when the trace's bit at `u` is set. -/
def bitSet (n : ℕ) (Hc : List UInt8 → Dh) : (Fin n → Bool) → (binding n Hc).Trace → Fin n → Prop :=
  fun _ Xv u => Xv u = true

/-- **Perfect per-block soundness**: no knowledge error, and a binding error only where the registration's bit differs
from the trace's. -/
def extractor (n : ℕ) (Hc : List UInt8 → Dh) : BlockExtractor (system n) (binding n Hc) where
  ks _ _ _ _ := 0
  link _ R u _ Xv := if R u = Xv u then 0 else 1

/-- **The trials read the registered bit** at no cost. -/
def trials (n : ℕ) (Hc : List UInt8 → Dh) : BlockTrials (system n) (binding n Hc) where
  Out := Unit
  reads u := {u}
  T _ _ _ _ _ := .ret ()
  sT _ _ _ _ _ := PUnit.unit
  rd _ R _ c _ := {R c}
  op _ _ _ _ _ v := v
  cost _ _ _ _ _ := 0
  sessCost _ _ _ _ := 0
  ρ := 0
  slack := 0

theorem errAt_ret (a x : Bool) :
    errAt (Game.ret ()) PUnit.unit (fun _ : Unit => ({a} : Finset Bool)) x = if a = x then 0 else 1 := by
  refine (errAt_eq_expect _ _ _ _).trans ?_
  show ind (fun _ : Unit => ({a} : Finset Bool).Nonempty) () - ind (fun _ : Unit => ({a} : Finset Bool) = {x}) () = _
  by_cases h : a = x
  · subst h
    simp [ind]
  · simp [ind, h]

theorem covers : (extractor n Hc).Covers (bitSet n Hc) := by
  intro S R u t Xv _ hX
  show prob (· = true) (Game.ret (t.1 && R u)) t.2 ≤ 0 + if R u = Xv u then 0 else 1
  by_cases h : R u = Xv u
  · have hR : R u = false := h.trans (Bool.eq_false_iff.2 hX)
    rw [ite_eq_left h, add_zero]
    simp [prob, hR]
  · rw [ite_eq_right h, zero_add]
    exact prob_le_one _ _ _

theorem opens : (trials n Hc).Opens :=
  fun _ _ _ _ _ _ hv => ⟨Finset.mem_singleton.1 hv, rfl⟩

theorem wellFormed : (trials n Hc).WellFormed :=
  ⟨fun _ _ _ _ _ => le_rfl, fun _ _ _ _ => le_rfl, zero_lt_one, le_rfl⟩

theorem transfers : (trials n Hc).Transfers (extractor n Hc) := by
  intro S R u t Xv _
  show (if R u = Xv u then (0 : ℝ≥0∞) else 1) ≤ ENNReal.ofReal ((∑ c ∈ ({u} : Finset (Fin n)),
    (errAt (Game.ret ()) PUnit.unit (fun _ : Unit => ({R c} : Finset Bool)) (Xv c) + 0)) / (1 - 0))
  rw [Finset.sum_singleton, errAt_ret]
  split_ifs with h
  · exact zero_le
  · norm_num

private theorem loopCost_nonneg' {τ Val : Type} [Fintype Val] [DecidableEq Val] (rd : τ → Finset Val) {c : τ → ℝ}
    (hc : ∀ e, 0 ≤ c e) : ∀ l : List τ, 0 ≤ loopCost rd c l
  | [] => by simp [loopCost]
  | e :: es => by
      simp only [loopCost]
      have := loopCost_nonneg' rd hc es
      split_ifs <;> linarith [hc e]

private theorem finderCost_nonneg' {τ Val : Type} [Fintype Val] [DecidableEq Val] (rd : τ → Finset Val)
    {c : τ → ℝ} (hc : ∀ e, 0 ≤ c e) (x : τ × List τ) : 0 ≤ finderCost rd c x := by
  unfold finderCost
  split_ifs
  · exact add_nonneg (hc _) (loopCost_nonneg' rd hc _)
  · simpa using hc x.1

private theorem prob_eq_zero {α : Type} {p : α → Prop} (h : ∀ a, ¬ p a) (g : Game α) (s : Strategy g) :
    prob p g s = 0 :=
  le_antisymm ((prob_mono (q := fun _ => False) (fun a ha => (h a ha).elim) g s).trans (prob_false g s).le) zero_le

/-- **ecr/sha-512 holds at every prover's link finders**: every trial reads the registered bit, so no finder outputs a
pair. -/
theorem cr (L : Law n) (R : Fin n → Bool) (τ : Cont L (Fin n → Bool) (system n).seq.session R) (M : ℕ) :
    ((trials n Hc).seq L).CR R τ M := by
  intro c h
  have hnone : ∀ x, ((trials n Hc).seq L).finderOut R τ c x = none := fun x => by
    unfold Trials.finderOut outOf
    refine dite_eq_right ?_
    rintro ⟨p, hne, h1, h2, -⟩
    exact hne ((Finset.mem_singleton.1 h1).trans (Finset.mem_singleton.1 h2).symm)
  have h0 : ∀ b, ¬ ∃ x y, ((trials n Hc).seq L).finderOut R τ c b = some (x, y) ∧ x ≠ y ∧ Hc x = Hc y :=
    fun b ⟨x, y, hxy, _⟩ => by rw [hnone] at hxy; exact Option.some_ne_none _ hxy.symm
  show (prob _ _ _).toReal ≤ expect _ _ _ / 2 ^ (256 : ℝ)
  rw [prob_eq_zero h0, ENNReal.toReal_zero]
  exact div_nonneg (expect_nonneg (fun x => finderCost_nonneg' _ (fun e => le_max_right _ _) x) _ _) (by positivity)

theorem pRd_seq (L : Law n) (R : Fin n → Bool) (τ : Cont L (Fin n → Bool) (system n).seq.session R) (c : Fin n)
    (h : ((trials n Hc).seq L).Read c) (w : Bool) :
    pRd (((trials n Hc).seq L).T R τ c h) (((trials n Hc).seq L).sT R τ c h) (((trials n Hc).seq L).rd R τ c) w =
      if R c = w then 1 else 0 := by
  unfold pRd
  by_cases hw : R c = w
  · rw [ite_eq_left hw]
    have : (ind fun e => ((trials n Hc).seq L).rd R τ c e = {w}) = fun _ => 1 :=
      funext fun _ => ite_eq_left (congrArg (fun v => ({v} : Finset Bool)) hw)
    exact (congrArg (fun f => expect f _ _) this).trans (expect_const _ _ _)
  · rw [ite_eq_right hw]
    have : (ind fun e => ((trials n Hc).seq L).rd R τ c e = {w}) = fun _ => 0 :=
      funext fun _ => ite_eq_right fun he => hw (Finset.singleton_inj.1 he)
    exact (congrArg (fun f => expect f _ _) this).trans (expect_const _ _ _)

/-- **The committed trace is the registration** at every position some sample reads. -/
theorem plur_seq (L : Law n) (R : Fin n → Bool) (τ : Cont L (Fin n → Bool) (system n).seq.session R) (c : Fin n)
    (h : ((trials n Hc).seq L).Read c) : ((trials n Hc).seq L).plur R τ c = R c := by
  have hmax := ((trials n Hc).seq L).plurality_plur R τ c h (R c)
  have h3 := ((pRd_seq L R τ c h (R c)).symm.trans_le hmax).trans_eq
    (pRd_seq L R τ c h (((trials n Hc).seq L).plur R τ c))
  by_contra hne
  have h4 : ¬ R c = ((trials n Hc).seq L).plur R τ c := fun e => hne e.symm
  norm_num [h4] at h3

/-- **Every proof block whose registered bit is clear is incorrect in the committed trace.** -/
theorem sub_incorrect (L : Law n) (R : Fin n → Bool) (τ : Cont L (Fin n → Bool) (system n).seq.session R) :
    Finset.univ.filter (fun u => R u = false) ⊆
      incorrect (binding n Hc) (bitSet n Hc) R (((trials n Hc).seq L).plur R τ) := by
  intro u hu
  rw [mem_incorrect]
  have hR : R u = false := (Finset.mem_filter.1 hu).2
  show ¬ ((trials n Hc).seq L).plur R τ u = true
  by_cases h : ((trials n Hc).seq L).Read u
  · rw [plur_seq L R τ u h, hR]
    exact Bool.false_ne_true
  · simp only [Trials.plur, dite_eq_right h]
    exact Bool.false_ne_true

theorem ksAvg_seq (L : Law n) (R : Fin n → Bool) (τ : Cont L (Fin n → Bool) (system n).seq.session R) :
    (extractor n Hc).seq.ksAvg L R τ = 0 := by
  refine le_antisymm ?_ zero_le
  refine ((extractor n Hc).ksAvg_seq_le (fun _ _ _ => 0) (fun _ _ _ _ _ => le_rfl) L R τ).trans ?_
  exact (avg_mono (g := fun _ => 0) fun ω => iSup₂_le fun u hu => le_rfl).trans (avg_const (C := L.Ω) 0).le

/-- **The theorem at the toy instance**: at every sampling design and every prover, the audit accepts with probability
at most the sampling design's acceptance probability at the proof blocks whose registered bit is clear, plus
`ε_bind`. -/
theorem bound (L : Law n) (M : ℕ) (hM : 1 ≤ M) (σ : Strategy (audit L (Fin n → Bool) (system n).seq.session)) :
    prob (fun o => o.1 = true) (audit L (Fin n → Bool) (system n).seq.session) σ ≤
      L.escape (Finset.univ.filter fun u => reg σ u = false) +
        ((trials n Hc).seq L).bindErr (reg σ) (cont σ) M := by
  refine (_root_.Flock.SecurityProofs.GenericSampledKSBlocks n (Fin n → Bool) Dh Hc L (system n) (binding n Hc)
    (bitSet n Hc) (extractor n Hc) (trials n Hc) M covers opens wellFormed transfers hM σ (cr L _ _ M)).trans ?_
  rw [ksAvg_seq, add_zero]
  exact add_le_add (Audit.Law.escape_anti L (sub_incorrect L _ _)) le_rfl

end FlockSoundness.SampledGeneric.Toy
