import Proofs.Flock.SamplingBoundary
import Proofs.Flock.Soundness.Discharge.ZkLink.Public

/-!
# The sampling claims' faulty units are wrong on the registration

The sampling claims (`Claims.Sampling`) count a slot faulty (`Claims.Sampling.Faulty`) when no interior makes its proof
units correct on the witness's openings; the game form (`Flock.Guarantees.EndToEnd`) counts a unit wrong on the
registration (`wrongRegZ`) at a transcript, `XplurZC` there. This file relates the two, in the direction a bound on the
game's event gives a bound on the claims': at a one-stage statement whose program, partition and openings are the
game's circuit, partition and transcript, every faulty slot is a unit wrong on every registration.

* `ofOp`: a game gate as a gate of the Boolean profile; a row (the AND of two XORs) has none, so `GatesAgree` holds only
  of a circuit without rows;
* `GatesAgree C p`: the program `p` is the circuit `C` gate by gate, and returns only `C`'s outputs;
* `UnitsAgree P Q`: the partition `Q` has one slot per unit of `P`, slot `u` one proof unit, the gates of unit `u`;
* `OpensOn P val Xt`: the openings `val` carry the transcript `Xt` on every committed wire of `P`;
* `runIn_localEval`: a unit's gates computed by the claims' `runIn` are the game's `localEval`;
* `not_faulty_of_correct`: a unit correct in `Xt` is not faulty;
* `wrongRegZ_of_faulty`, `card_faulty_le_card_wrongRegZ`: a faulty slot is a unit wrong on the registration, so there
  are at most as many faulty slots as units wrong on it.
-/

namespace FlockSoundness.Discharge.ZkLink.SamplingFaulty

open Game Model Accounting Audit ZkSession

/-- **A game gate as a gate of the Boolean profile**: an input, `And_v1`, `Xor_v1` and `Not_v1` of the same wires; a row
has none. -/
def ofOp {N : ℕ} : Op N → Option _root_.Claims.Sampling.BGate
  | .input => some .input
  | .and a b => some (.and [a.val, b.val])
  | .xor a b => some (.xor [a.val, b.val])
  | .not a => some (.not a.val)
  | .row _ _ => none

/-- **The program `p` is the circuit `C`, gate by gate**: its gates are `C`'s (`ofOp`), and every gate it returns is one
of `C`'s outputs. -/
def GatesAgree (C : Circuit) (p : _root_.Claims.Sampling.Prog) : Prop :=
  p.gates.length = C.N ∧ (∀ g : Fin C.N, p.gates[g.val]? = ofOp (C.op g)) ∧
    ∀ o ∈ p.outs, ∃ h : o < C.N, (⟨o, h⟩ : Fin C.N) ∈ C.outputs

/-- **The one-stage partition `Q` is `P`**: one slot per unit, slot `u` the proof block of the gates of unit `u`. -/
def UnitsAgree {C : Circuit} {n : ℕ} (P : Partition C n) (Q : _root_.Claims.Sampling.Partition) : Prop :=
  Q.blocks.length = n ∧ ∀ u : Fin n, ∃ l : List ℕ, Q.blocks[u.val]? = some l ∧
    ∀ g, g ∈ l ↔ ∃ h : g < C.N, P.unit ⟨g, h⟩ = some u

/-- **The openings `val` carry the transcript `Xt`** on every committed wire of `P`. -/
def OpensOn {C : Circuit} {n : ℕ} (P : Partition C n) (val : ℕ → Option Bool) (Xt : Fin C.N → Bool) : Prop :=
  ∀ g ∈ P.committed, val g.val = some (Xt g)

theorem ofOp_ne_const {N : ℕ} (o : Op N) (b : Bool) : ofOp o ≠ some (.const b) := by
  cases o <;> simp [ofOp]

theorem ofOp_reads {N : ℕ} {o : Op N} {x : _root_.Claims.Sampling.BGate} (h : ofOp o = some x) :
    x.reads = o.args.map Fin.val := by
  cases o <;> simp only [ofOp, Option.some.injEq, reduceCtorEq] at h <;> subst h <;>
    simp [_root_.Claims.Sampling.BGate.reads, Op.args]

theorem isInput_of_ofOp {N : ℕ} {o : Op N} (h : ofOp o = some .input) : o.isInput = true := by
  cases o <;> simp_all [ofOp, Op.isInput]

theorem ofOp_val {N : ℕ} {o : Op N} {x : _root_.Claims.Sampling.BGate} (h : ofOp o = some x)
    (hi : o.isInput = false) {rd : ℕ → Option Bool} {v : Fin N → Bool} (own : Option Bool)
    (hrd : ∀ a ∈ o.args, rd a.val = some (v a)) : x.val rd own = some (o.apply v) := by
  cases o with
  | input => simp [Op.isInput] at hi
  | and a b =>
    simp only [ofOp, Option.some.injEq] at h
    subst h
    simp [_root_.Claims.Sampling.BGate.val, hrd a (by simp [Op.args]), hrd b (by simp [Op.args]), Op.apply]
  | xor a b =>
    simp only [ofOp, Option.some.injEq] at h
    subst h
    simp [_root_.Claims.Sampling.BGate.val, hrd a (by simp [Op.args]), hrd b (by simp [Op.args]), Op.apply]
  | not a =>
    simp only [ofOp, Option.some.injEq] at h
    subst h
    simp [_root_.Claims.Sampling.BGate.val, hrd a (by simp [Op.args]), Op.apply]
  | row a b => simp [ofOp] at h

section Units

variable {C : Circuit} {n : ℕ} {P : Partition C n} {p : _root_.Claims.Sampling.Prog}

theorem read_eq_of_agree (hG : GatesAgree C p) (ref : ℕ → Option Bool) (a : Fin C.N) : p.read ref a.val = ref a.val :=
  _root_.Claims.Sampling.Boundary.read_eq p fun b h => ofOp_ne_const _ b ((hG.2.1 a).symm.trans h)

/-- A gate another unit's gate reads is committed. -/
theorem mem_committed_of_reads {u : Fin n} {g a : Fin C.N} (hu : P.unit g = some u) (ha : a ∈ (C.op g).args)
    (hau : P.unit a ≠ some u) : a ∈ P.committed := by
  simp only [Partition.committed, Finset.mem_filter, Finset.mem_univ, true_and]
  exact Or.inr (Or.inr ⟨g, ha, by rw [hu]; exact fun h => hau h.symm⟩)

/-- **The claims' `runIn` is the game's `localEval`** on a unit's gates, when every value read from outside the unit is
the opening of a transcript. -/
theorem runIn_localEval (hG : GatesAgree C p) {val : ℕ → Option Bool} {Xt : Fin C.N → Bool} (hO : OpensOn P val Xt)
    {u : Fin n} {l : List ℕ} (hl : ∀ g, g ∈ l ↔ ∃ h : g < C.N, P.unit ⟨g, h⟩ = some u) {ref : ℕ → Option Bool}
    (href : ∀ i, i ∉ l → ref i = val i) :
    ∀ m (g : Fin C.N), g.val = m → P.unit g = some u → p.runIn ref l g.val = some (P.localEval Xt u g) := by
  intro m
  induction m using Nat.strong_induction_on with
  | _ m ih =>
  intro g hgm hu
  have hni : (C.op g).isInput = false := by
    cases h : (C.op g).isInput
    · rfl
    · exact absurd ((P.input_iff g).2 h) (by simp [hu])
  have hlt : g.val < p.gates.length := by rw [hG.1]; exact g.isLt
  have hx : p.gates[g.val]? = some p.gates[g.val] := List.getElem?_eq_getElem hlt
  have hx' : ofOp (C.op g) = some p.gates[g.val] := (hG.2.1 g).symm.trans hx
  rw [_root_.Claims.Sampling.Boundary.runIn_eq, hx, Option.bind_some, P.localEval_of_mem Xt hu]
  refine ofOp_val hx' hni _ fun a ha => ?_
  have hag : a.val < g.val := C.topo g a ha
  by_cases hal : a.val ∈ l
  · have hau : P.unit a = some u := by
      obtain ⟨_, h⟩ := (hl a.val).1 hal
      exact h
    simp only [hal, ↓reduceIte, hag]
    exact ih a.val (hgm ▸ hag) a rfl hau
  · have hau : P.unit a ≠ some u := fun h => hal ((hl a.val).2 ⟨a.isLt, h⟩)
    simp only [hal, ↓reduceIte]
    rw [read_eq_of_agree hG, href _ hal, hO a (mem_committed_of_reads hu ha hau), P.localEval_of_not_mem Xt hau]

/-- **A unit correct in the transcript is not faulty**, at a one-stage statement whose program, partition and openings
are the game's circuit, partition and transcript. -/
theorem not_faulty_of_correct {H : _root_.Claims.Msg → _root_.Claims.Msg} {Cm : _root_.Claims.Sampling.Commitment Bool}
    {s : _root_.Claims.Sampling.AuditStatement} {Q : _root_.Claims.Sampling.Partition} {w : _root_.Claims.Sampling.Wit}
    {Xt : Fin C.N → Bool} (hG : GatesAgree C s.prog) (hU : UnitsAgree P Q) (hO : OpensOn P (w.val H Cm s) Xt)
    {u : Fin n} (hc : P.Correct Xt u) : ¬ _root_.Claims.Sampling.Faulty H Cm s Q w u.val := by
  obtain ⟨l, hQ, hl⟩ := hU.2 u
  intro hF
  apply hF
  refine ⟨l, hQ, fun _ => false, ?_⟩
  have href : ∀ i, i ∉ l → s.prog.refIn l (w.val H Cm s) (fun _ => false) i = w.val H Cm s i := by
    intro i hi
    simp [_root_.Claims.Sampling.Prog.refIn, hi]
  intro g hg
  obtain ⟨hgN, hgu⟩ := (hl g).1 hg
  refine ⟨fun x hx i hi hil => ?_, fun hb => ?_⟩
  · rw [ofOp_reads ((hG.2.1 ⟨g, hgN⟩).symm.trans hx)] at hi
    obtain ⟨a, ha, rfl⟩ := List.mem_map.1 hi
    have hau : P.unit a ≠ some u := fun h => hil ((hl a.val).2 ⟨a.isLt, h⟩)
    rw [read_eq_of_agree hG, href _ hil, hO a (mem_committed_of_reads hgu ha hau)]
    simp
  · have hgc : (⟨g, hgN⟩ : Fin C.N) ∈ P.committed := by
      simp only [Partition.committed, Finset.mem_filter, Finset.mem_univ, true_and]
      simp only [_root_.Claims.Sampling.Prog.Boundary, Bool.or_eq_true, List.contains_iff_mem,
        decide_eq_true_eq, List.any_eq_true, List.mem_range, Bool.and_eq_true, decide_not,
        Bool.not_eq_true', decide_eq_false_iff_not] at hb
      rcases hb with (hout | hin) | ⟨g', hg', hg'l, hrd⟩
      · obtain ⟨h, hmem⟩ := hG.2.2 g hout
        exact Or.inr (Or.inl hmem)
      · exact Or.inl (isInput_of_ofOp ((hG.2.1 ⟨g, hgN⟩).symm.trans hin))
      · have hg'N : g' < C.N := hG.1 ▸ hg'
        have hx : s.prog.gates[g']? = some s.prog.gates[g'] := List.getElem?_eq_getElem hg'
        rw [hx] at hrd
        simp only [Option.map_some, Option.getD_some] at hrd
        rw [ofOp_reads ((hG.2.1 ⟨g', hg'N⟩).symm.trans hx)] at hrd
        obtain ⟨a, ha, hag⟩ := List.mem_map.1 hrd
        have hae : a = ⟨g, hgN⟩ := Fin.ext hag
        subst hae
        refine Or.inr (Or.inr ⟨⟨g', hg'N⟩, ha, ?_⟩)
        rw [hgu]
        exact fun h => hg'l ((hl g').2 ⟨hg'N, h⟩)
    have hrefg : s.prog.refIn l (w.val H Cm s) (fun _ => false) g = w.val H Cm s g := by
      simp [_root_.Claims.Sampling.Prog.refIn, hb]
    refine ⟨Xt ⟨g, hgN⟩, by rw [hrefg]; exact hO _ hgc, ?_⟩
    rw [runIn_localEval hG hO hl href g ⟨g, hgN⟩ rfl hgu, hc _ hgc hgu]

/-- **A faulty slot is a wrong unit** of the transcript. -/
theorem wrong_of_faulty {H : _root_.Claims.Msg → _root_.Claims.Msg} {Cm : _root_.Claims.Sampling.Commitment Bool}
    {s : _root_.Claims.Sampling.AuditStatement} {Q : _root_.Claims.Sampling.Partition} {w : _root_.Claims.Sampling.Wit}
    {Xt : Fin C.N → Bool} (hG : GatesAgree C s.prog) (hU : UnitsAgree P Q) (hO : OpensOn P (w.val H Cm s) Xt)
    {u : Fin n} (hF : _root_.Claims.Sampling.Faulty H Cm s Q w u.val) : u ∈ P.wrong Xt :=
  P.mem_wrong.2 fun hc => not_faulty_of_correct hG hU hO hc hF

end Units

section Reg

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable (A : Arith F K) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)
  {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : ValueBindingZ A P planZ tab decode Hc)

omit [Fintype K] [DecidableEq K] in
/-- **A faulty slot is a unit wrong on the registration** `R` at the transcript `Xt`, at a one-stage statement whose
program, partition and openings are the game's circuit, partition and `Xt`. -/
theorem wrongRegZ_of_faulty {H : _root_.Claims.Msg → _root_.Claims.Msg} {Cm : _root_.Claims.Sampling.Commitment Bool}
    {s : _root_.Claims.Sampling.AuditStatement} {Q : _root_.Claims.Sampling.Partition} {w : _root_.Claims.Sampling.Wit}
    {Xt : Fin C.N → Bool} (hG : GatesAgree C s.prog) (hU : UnitsAgree P Q) (hO : OpensOn P (w.val H Cm s) Xt)
    (R : Reg) {u : Fin n} (hF : _root_.Claims.Sampling.Faulty H Cm s Q w u.val) :
    u ∈ wrongRegZ A P planZ tab vb R Xt :=
  wrong_subset_wrongRegZ A P planZ tab vb R Xt (wrong_of_faulty hG hU hO hF)

omit [Fintype K] [DecidableEq K] in
/-- **At most as many faulty slots as units wrong on the registration** `R` at the transcript `Xt`: so the game's bound
on `K₀ ≤ #wrongRegZ` bounds the claims' `K₀ ≤ #faulty`. -/
theorem card_faulty_le_card_wrongRegZ {H : _root_.Claims.Msg → _root_.Claims.Msg} {Cm : _root_.Claims.Sampling.Commitment Bool}
    {s : _root_.Claims.Sampling.AuditStatement} {Q : _root_.Claims.Sampling.Partition} {w : _root_.Claims.Sampling.Wit}
    {Xt : Fin C.N → Bool} (hG : GatesAgree C s.prog) (hU : UnitsAgree P Q) (hO : OpensOn P (w.val H Cm s) Xt)
    (R : Reg) : (_root_.Claims.Sampling.faulty H Cm s Q w).card ≤ (wrongRegZ A P planZ tab vb R Xt).card := by
  classical
  calc (_root_.Claims.Sampling.faulty H Cm s Q w).card
      ≤ ((wrongRegZ A P planZ tab vb R Xt).map Fin.valEmbedding).card := by
        refine Finset.card_le_card fun v hv => ?_
        simp only [_root_.Claims.Sampling.faulty, Finset.mem_filter, Finset.mem_range] at hv
        rw [hU.1] at hv
        exact Finset.mem_map.2 ⟨⟨v, hv.1⟩, wrongRegZ_of_faulty A P planZ tab vb hG hU hO R hv.2, rfl⟩
    _ = (wrongRegZ A P planZ tab vb R Xt).card := Finset.card_map _

end Reg

end FlockSoundness.Discharge.ZkLink.SamplingFaulty
