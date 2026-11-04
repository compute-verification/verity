import Definitions.NetworkCertifier.SampledCharge
import Proofs.NetworkCertifier.KSWired
import Proofs.NetworkCertifier.Wires
import Proofs.Flock.Claims.Witnesses.Sampling

/-!
# Three toys for the wired sampled audit's premises and count

The sampled audit counts a wire-wrong proof block as incorrect (`incorrect`, `RWired`, `KSBy` at the wired check), a layout's wiring
fits its program only when its wires are acyclic (`WiresAcyclic`, inside `BoundaryGates.Fits`), and a free input that
writes a wire stays a free input (`freeInputs`). Each toy is a program at which dropping one of the three lets accepted
traces carry more observation sequences than `SampledLeakAccepted`'s bound
`2^#(freeInputs \ reg) · #faultySets(δ) · 2^β · Dj^k`, here at `δ = 0`, no registered input, `β = 0`, one step and
`Dj = 1`.

The program `flipProg` is two inputs, each negated, the negations returned and at the two slots of one egress link
(`flipEgress`); its partition `flipPart` is the proof blocks `{0, 1}` and `{2, 3}` of one
class drawn `all`, so at `δ = 0` the one possible faulty set is empty (`flip_faultySets`). Every opening passes
(`accCm`). `carried` is the set of sequences that the transfer gates of some trace carry, with every status complete,
whose proof blocks counted bad are in `M_δ`.

* **The wire flip** (`flipLay`): the second input is wired to the first negation, so one input is free and the bound
  is `2`. A trace whose two proof blocks are each correct, its wired input the opposite of its writer, has no faulty
  proof block: the sampling relation at the faulty set (`R`, what `KS` gives) holds there and `RWired` doesn't
  (`flip_excludes`). Counted at the faulty set, the traces carry all four rows, past the bound; counted at the
  incorrect set, two (`flip_meets`).
* **The two-cycle** (`cycleLay`): each input wired to the other's negation. The layout fits but for its wires' cycle
  (`cycle_fits_iff`), no input is free and the bound is `1`, and two traces with no incorrect proof block carry two
  rows (`cycle_excludes`).
* **The pass-through** (`passLay`): the second input is wired to the first, so the wire's chain ends at once at an input
  no wire or ingress holds. It is a later copy of an input bit read through its first copy, which `flipProg` doesn't
  return: the layout fits because `BoundaryGates.WiresFit` admits an input writer. Only the reader's copy leaves the
  free inputs: the first input stays free and the bound is `2`, which the traces with no incorrect proof block meet
  (`pass_meets`). They carry two rows, so a count that left the chain's end out too, `1`, is exceeded (`pass_excludes`).
-/

namespace NetTiming.WiredToys

open Claims Claims.Sampling Claims.Witness.Sampling

/-! ## The program, its partition and layouts -/

/-- Two inputs, each negated, the negations returned. -/
def flipProg : Prog := ⟨[.input, .not 0, .input, .not 2], [1, 3]⟩

/-- One class drawn `all`, its one stratum the proof blocks `{0, 1}` and `{2, 3}`. -/
def flipPart : Partition := ⟨[[[[0, 1], [2, 3]]]]⟩

/-- `flipProg`'s statement, its one class drawn `all`. -/
def flipStmt : AuditStatement := ⟨[], flipProg, ⟨"", 0, []⟩, ⟨[.all]⟩, []⟩

/-- One egress link of two slots at every step: slot `0` at the first negation, slot `1` at the second, each an empty
frame. -/
def flipEgress : ℕ → Fin 1 → Fin 2 → ℕ × List ℕ := fun _ _ σ => (if σ = 0 then 1 else 3, [])

/-- No ingress, and the second input wired to the first negation. -/
def flipLay : BoundaryGates 1 2 where
  egress := flipEgress
  ingress _ := []
  wires g := if g = 2 then some 1 else none

/-- No ingress, and each input wired to the other's negation: a cycle. -/
def cycleLay : BoundaryGates 1 2 where
  egress := flipEgress
  ingress _ := []
  wires g := if g = 0 then some 3 else if g = 2 then some 1 else none

/-- No ingress, and the second input wired to the first: a pass-through, its writer an input no wire holds. -/
def passLay : BoundaryGates 1 2 where
  egress := flipEgress
  ingress _ := []
  wires g := if g = 2 then some 0 else none

/-- The operator, which sends nothing. -/
def flipOp : ℕ → List (Obs 1 1 2) → List Bool := fun _ _ => []

/-- One step whose slot `0` holds an empty frame when `p` and slot `1` when `q`, every status complete. -/
def flipObs (p q : Bool) : Obs 1 1 2 :=
  ⟨fun _ σ => if (if σ = 0 then p else q) then some [] else none, 0, fun _ => .complete⟩

/-- Openings of the four gates at the values `a`, `b`, `c` and `d`. -/
def wit4 (a b c d : Bool) : Wit := ⟨[⟨0, a, []⟩, ⟨1, b, []⟩, ⟨2, c, []⟩, ⟨3, d, []⟩]⟩

/-- **The sequences the traces carry whose bad proof blocks are in `M_δ`**: the observation sequences of `k` steps with
jitter below `Dj` and every status complete that the transfer gates of some trace `w` carry (`Carries`), `bad w` in
`M_δ`. At `bad = faulty` the traces are the ones `R δ` admits, and at `bad = incorrect` the ones `RWired δ` admits. -/
def carried {L K S : ℕ} (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (P : Partition)
    (lay : BoundaryGates L S) (k : ℕ) (δ : ℝ) (bad : Wit → Finset ℕ) (op : ℕ → List (Obs L K S) → List Bool)
    (Dj : ℕ) : Set (List (Obs L K S)) :=
  {o | o.length = k ∧ (∀ x ∈ o, x.jitter < Dj) ∧ Covered k o ∧
    ∃ w, bad w ∈ s.design.M P δ ∧ Carries lay s.prog (w.val H Cm s) op k o}

/-! ## The design: at `δ = 0` no proof block may be bad -/

theorem window_two {F : Finset ℕ} (hF : F ⊆ Finset.range 2) : window 0 2 F = F := by
  ext v
  simp only [window, Finset.mem_image, Finset.mem_filter]
  constructor
  · rintro ⟨v', ⟨hv', -⟩, rfl⟩
    simpa using hv'
  · intro hv
    exact ⟨v, ⟨hv, Nat.zero_le v, by simpa using hF hv⟩, Nat.sub_zero v⟩

theorem flip_Pa (F : Finset ℕ) : flipStmt.design.Pa flipPart F = if window 0 2 F = ∅ then 1 else 0 := by
  simp [Design.Pa, flipStmt, flipPart, classesPa, OneStage.Pa]

theorem flip_M {F : Finset ℕ} (hF : F ⊆ Finset.range 2) : F ∈ flipStmt.design.M flipPart 0 ↔ F = ∅ := by
  show (0 : ℝ) < flipStmt.design.Pa flipPart F ↔ F = ∅
  rw [flip_Pa, window_two hF]
  split_ifs with h <;> simp [h]

theorem flip_faultySets : faultySets flipStmt.design flipPart 0 = {∅} := by
  ext F
  simp only [faultySets, Finset.mem_filter, Finset.mem_powerset, Finset.mem_singleton]
  constructor
  · rintro ⟨hsub, hM, -⟩
    exact (flip_M hsub).1 hM
  · rintro rfl
    exact ⟨Finset.empty_subset _, (flip_M (Finset.empty_subset _)).2 rfl, Finset.disjoint_empty_left _⟩

theorem flip_free : (freeInputs flipProg flipLay 1 \ ∅).card = 1 := by
  decide

theorem cycle_free : (freeInputs flipProg cycleLay 1 \ ∅).card = 0 := by
  decide

theorem flip_bound :
    ((2 ^ (freeInputs flipProg flipLay 1 \ ∅).card * (faultySets flipStmt.design flipPart 0).card * 2 ^ 0 * 1 ^ 1 :
      ℕ) : ℕ∞) = ((2 : ℕ) : ℕ∞) := by
  rw [flip_free, flip_faultySets, Finset.card_singleton]
  rfl

theorem cycle_bound :
    ((2 ^ (freeInputs flipProg cycleLay 1 \ ∅).card * (faultySets flipStmt.design flipPart 0).card * 2 ^ 0 * 1 ^ 1 :
      ℕ) : ℕ∞) = ((1 : ℕ) : ℕ∞) := by
  rw [cycle_free, flip_faultySets, Finset.card_singleton]
  rfl

/-- Under `passLay` only the reader's copy leaves the free inputs: the first input, the wire's writer, stays. -/
theorem pass_end : 0 ∈ freeInputs flipProg passLay 1 := by
  decide

theorem pass_free : (freeInputs flipProg passLay 1 \ ∅).card = 1 := by
  decide

theorem pass_erase : ((freeInputs flipProg passLay 1).erase 0).card = 0 := by
  decide

theorem pass_bound :
    ((2 ^ (freeInputs flipProg passLay 1 \ ∅).card * (faultySets flipStmt.design flipPart 0).card * 2 ^ 0 * 1 ^ 1 :
      ℕ) : ℕ∞) = ((2 : ℕ) : ℕ∞) := by
  rw [pass_free, flip_faultySets, Finset.card_singleton]
  rfl

/-- The bound with the wire's chain's end left out of the free inputs too. -/
theorem pass_bound_erase :
    ((2 ^ ((freeInputs flipProg passLay 1).erase 0).card * (faultySets flipStmt.design flipPart 0).card * 2 ^ 0 *
      1 ^ 1 : ℕ) : ℕ∞) = ((1 : ℕ) : ℕ∞) := by
  rw [pass_erase, flip_faultySets, Finset.card_singleton]
  rfl

/-! ## The layouts fit but for the wires' cycle -/

theorem reads_lt {b c : ℕ} (h : b ∈ (flipProg.gates[c]?.map BGate.reads).getD []) : b < c := by
  rcases Nat.lt_or_ge c 4 with hc | hc
  · interval_cases c <;> simp [flipProg, BGate.reads] at h <;> omega
  · have hn : flipProg.gates[c]? = none := by simp [flipProg]; omega
    simp [hn] at h

theorem depends_le {i g : ℕ} (h : Depends flipProg i g) : i ≤ g := by
  unfold Depends at h
  induction h with
  | refl => exact le_rfl
  | tail _ hbc ih => exact ih.trans (reads_lt hbc).le

theorem flip_wired {g w : ℕ} (hw : flipLay.wires g = some w) : g = 2 ∧ w = 1 := by
  have h' : (if g = 2 then some 1 else none) = some w := hw
  split at h'
  · exact ⟨‹_›, (Option.some.inj h').symm⟩
  · cases h'

theorem cycle_wired {g w : ℕ} (hw : cycleLay.wires g = some w) : (g = 0 ∧ w = 3) ∨ (g = 2 ∧ w = 1) := by
  have h' : (if g = 0 then some 3 else if g = 2 then some 1 else none) = some w := hw
  split at h'
  · exact Or.inl ⟨‹_›, (Option.some.inj h').symm⟩
  · split at h'
    · exact Or.inr ⟨‹_›, (Option.some.inj h').symm⟩
    · cases h'

theorem pass_wired {g w : ℕ} (hw : passLay.wires g = some w) : g = 2 ∧ w = 0 := by
  have h' : (if g = 2 then some 0 else none) = some w := hw
  split at h'
  · exact ⟨‹_›, (Option.some.inj h').symm⟩
  · cases h'

/-- A layout with `flipLay`'s egress and ingress whose wired gates are inputs and whose writers are returned or inputs
fits `flipProg` exactly when its wires are acyclic. -/
theorem fits_iff (lay : BoundaryGates 1 2) (he : lay.egress = flipEgress) (hi : lay.ingress = fun _ => [])
    (hw : ∀ g w, lay.wires g = some w → (g = 0 ∨ g = 2) ∧ (w ∈ flipProg.outs ∨ flipProg.gates[w]? = some .input)) :
    lay.Fits flipProg 1 ↔ WiresAcyclic flipProg lay.wires := by
  refine ⟨fun h => by obtain ⟨-, -, -, -, h⟩ := h; exact h, fun h => ⟨fun t _ ℓ σ g hg => ?_, fun t _ g hg => ?_,
    fun t ht t' ht' htt' => absurd htt' (by omega), fun g w hgw => ?_, h⟩⟩
  · rw [he] at hg
    by_cases hσ : σ = 0 <;> simp [slotGates, flipEgress, hσ] at hg <;> subst hg <;> decide
  · rw [hi] at hg
    simp at hg
  · obtain ⟨hg, hw'⟩ := hw g w hgw
    exact ⟨by rcases hg with rfl | rfl <;> rfl, fun t _ => by rw [hi]; simp, hw'⟩

theorem flip_acyclic : WiresAcyclic flipProg flipLay.wires := by
  refine ⟨fun _ => 0, fun g w g' hg hg' hd => ?_⟩
  obtain ⟨-, rfl⟩ := flip_wired hg
  rcases hc : flipLay.wires g' with _ | w'
  · exact absurd hc hg'
  · obtain ⟨rfl, -⟩ := flip_wired hc
    exact absurd (depends_le hd) (by omega)

theorem flip_fits : flipLay.Fits flipProg 1 :=
  (fits_iff flipLay rfl rfl fun _ _ hw => by
    obtain ⟨rfl, rfl⟩ := flip_wired hw
    exact ⟨Or.inr rfl, Or.inl (by decide)⟩).2 flip_acyclic

/-- **`cycleLay` fits `flipProg` exactly when its wires are acyclic**: every other condition of `BoundaryGates.Fits`
holds. -/
theorem cycle_fits_iff : cycleLay.Fits flipProg 1 ↔ WiresAcyclic flipProg cycleLay.wires :=
  fits_iff cycleLay rfl rfl fun _ _ hw => by
    rcases cycle_wired hw with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
    · exact ⟨Or.inl rfl, Or.inl (by decide)⟩
    · exact ⟨Or.inr rfl, Or.inl (by decide)⟩

theorem pass_acyclic : WiresAcyclic flipProg passLay.wires := by
  refine ⟨fun _ => 0, fun g w g' hg hg' hd => ?_⟩
  obtain ⟨-, rfl⟩ := pass_wired hg
  rcases hc : passLay.wires g' with _ | w'
  · exact absurd hc hg'
  · obtain ⟨rfl, -⟩ := pass_wired hc
    exact absurd (depends_le hd) (by omega)

/-- `passLay` fits `flipProg`: its writer is an input, which `flipProg` doesn't return. -/
theorem pass_fits : passLay.Fits flipProg 1 :=
  (fits_iff passLay rfl rfl fun _ _ hw => by
    obtain ⟨rfl, rfl⟩ := pass_wired hw
    exact ⟨Or.inr rfl, Or.inr rfl⟩).2 pass_acyclic

/-- `passLay`'s writer, the first input, is no gate `flipProg` returns. -/
theorem pass_writer_unreturned : 0 ∉ flipProg.outs := by
  decide

/-- **`cycleLay`'s wires are a cycle**: the first input's writer depends on the second input, and the second's on the
first, so no rank falls along both. -/
theorem cycle_not_acyclic : ¬ WiresAcyclic flipProg cycleLay.wires := by
  rintro ⟨r, hr⟩
  have h1 := hr 0 3 2 rfl (by simp [cycleLay]) (depends_of_read (x := .not 2) rfl (by simp [BGate.reads]))
  have h2 := hr 2 1 0 rfl (by simp [cycleLay]) (depends_of_read (x := .not 0) rfl (by simp [BGate.reads]))
  omega

/-! ## The traces -/

theorem wit4_val (H : Msg → Msg) (a b c d : Bool) :
    (wit4 a b c d).val H accCm flipStmt 0 = some a ∧ (wit4 a b c d).val H accCm flipStmt 1 = some b ∧
      (wit4 a b c d).val H accCm flipStmt 2 = some c ∧ (wit4 a b c d).val H accCm flipStmt 3 = some d := by
  simp [Wit.val, wit4, CommitmentScheme.Opens, accCm, OC.eval]

theorem refIn_of_boundary {C : Prog} {u : List ℕ} {g : ℕ} (hg : C.Boundary u g = true) (val : ℕ → Option Bool)
    (ι : ℕ → Bool) : C.refIn u val ι g = val g := by
  simp [Prog.refIn, hg]

theorem runIn_input {C : Prog} {i : ℕ} (hi : C.gates[i]? = some .input) (ref : ℕ → Option Bool) (u : List ℕ) :
    C.runIn ref u i = ref i := by
  rw [Prog.runIn, hi]
  rfl

theorem runIn_not {C : Prog} {i j : ℕ} (hi : C.gates[i]? = some .input) (hj : C.gates[j]? = some (.not i))
    (hij : i < j) (ref : ℕ → Option Bool) : C.runIn ref [i, j] j = (ref i).map (!·) := by
  rw [Prog.runIn, hj]
  simp [BGate.val, hij, runIn_input hi]

/-- An input and its negation are correct on values that negate. -/
theorem pair_correct {C : Prog} {i j : ℕ} (hi : C.gates[i]? = some .input) (hj : C.gates[j]? = some (.not i))
    (hij : i < j) {ref : ℕ → Option Bool} {b : Bool} (h0 : ref i = some b) (h1 : ref j = some (!b)) :
    C.Correct ref [i, j] := by
  intro g hg
  rcases List.mem_cons.1 hg with rfl | hg
  · refine ⟨fun x hx k hk _ => ?_, fun _ => ⟨b, h0, by rw [runIn_input hi, h0]⟩⟩
    rw [hi] at hx
    obtain rfl : BGate.input = x := Option.some.inj hx
    simp [BGate.reads] at hk
  · obtain rfl : g = j := List.mem_singleton.1 hg
    refine ⟨fun x hx k hk hku => ?_, fun _ => ⟨!b, h1, by simp [runIn_not hi hj hij, h0]⟩⟩
    rw [hj] at hx
    obtain rfl : BGate.not i = x := Option.some.inj hx
    simp only [BGate.reads, List.mem_singleton] at hk
    subst hk
    exact absurd List.mem_cons_self hku

/-- On a correct input and its negation, both committed, the negation's value negates the input's. -/
theorem pair_val {C : Prog} {i j : ℕ} (hi : C.gates[i]? = some .input) (hj : C.gates[j]? = some (.not i))
    (hij : i < j) (hbi : C.Boundary [i, j] i = true) (hbj : C.Boundary [i, j] j = true) {val : ℕ → Option Bool}
    {ι : ℕ → Bool} (h : C.Correct (C.refIn [i, j] val ι) [i, j]) : val j = (val i).map (!·) := by
  obtain ⟨b, hb, hrun⟩ := (h j (by simp)).2 hbj
  rw [runIn_not hi hj hij, refIn_of_boundary hbi] at hrun
  rw [refIn_of_boundary hbj] at hb
  rw [hb, hrun]

/-- A trace whose negations negate its inputs has no faulty proof block. -/
theorem flip_faulty (H : Msg → Msg) {a b c d : Bool} (hb : b = !a) (hd : d = !c) :
    faulty H accCm flipStmt flipPart (wit4 a b c d) = ∅ := by
  classical
  subst hb hd
  obtain ⟨h0, h1, h2, h3⟩ := wit4_val H a (!a) c (!c)
  refine Finset.eq_empty_of_forall_notMem fun v hv => ?_
  obtain ⟨hv, hf⟩ := Finset.mem_filter.1 hv
  have hv2 : v < 2 := Finset.mem_range.1 hv
  apply hf
  interval_cases v
  · exact ⟨[0, 1], rfl, fun _ => false, pair_correct (C := flipProg) (i := 0) (j := 1) rfl rfl Nat.one_pos
      ((refIn_of_boundary (by decide) _ _).trans h0) ((refIn_of_boundary (by decide) _ _).trans h1)⟩
  · exact ⟨[2, 3], rfl, fun _ => false, pair_correct (C := flipProg) (i := 2) (j := 3) rfl rfl (by omega)
      ((refIn_of_boundary (by decide) _ _).trans h2) ((refIn_of_boundary (by decide) _ _).trans h3)⟩

theorem wireWrong_eq_empty {H : Msg → Msg} {Cm : CommitmentScheme Bool} {s : AuditStatement} {P : Partition}
    {wires : ℕ → Option ℕ} {w : Wit} (h : ∀ g, WireRight s.prog wires (w.val H Cm s) g) :
    wireWrong H Cm s P wires w = ∅ := by
  classical
  refine Finset.eq_empty_of_forall_notMem fun _ hv => ?_
  obtain ⟨-, _, -, g, -, hg⟩ := Finset.mem_filter.1 hv
  exact hg (h g)

theorem cycle_wireRight (H : Msg → Msg) (a : Bool) (g : ℕ) :
    WireRight flipStmt.prog cycleLay.wires ((wit4 a (!a) (!a) a).val H accCm flipStmt) g := by
  obtain ⟨h0, h1, h2, h3⟩ := wit4_val H a (!a) (!a) a
  intro w hw
  rcases cycle_wired hw with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
  · exact ⟨fun c => by simp [flipStmt, flipProg], a, h0, h3⟩
  · exact ⟨fun c => by simp [flipStmt, flipProg], !a, h2, h1⟩

/-- Under `cycleLay`'s wiring, a trace whose inputs are its writers' values has no incorrect proof block. -/
theorem cycle_incorrect (H : Msg → Msg) (a : Bool) :
    incorrect H accCm flipStmt flipPart cycleLay.wires (wit4 a (!a) (!a) a) = ∅ := by
  rw [incorrect, flip_faulty H (a := a) (b := !a) (c := !a) (d := a) rfl (by simp),
    wireWrong_eq_empty (cycle_wireRight H a), Finset.empty_union]

theorem pass_wireRight (H : Msg → Msg) (a : Bool) (g : ℕ) :
    WireRight flipStmt.prog passLay.wires ((wit4 a (!a) a (!a)).val H accCm flipStmt) g := by
  obtain ⟨h0, -, h2, -⟩ := wit4_val H a (!a) a (!a)
  intro w hw
  obtain ⟨rfl, rfl⟩ := pass_wired hw
  exact ⟨fun c => by simp [flipStmt, flipProg], a, h2, h0⟩

/-- Under `passLay`'s wiring, a trace whose second input copies the first has no incorrect proof block. -/
theorem pass_incorrect (H : Msg → Msg) (a : Bool) :
    incorrect H accCm flipStmt flipPart passLay.wires (wit4 a (!a) a (!a)) = ∅ := by
  rw [incorrect, flip_faulty H (a := a) (b := !a) (c := a) (d := !a) rfl rfl,
    wireWrong_eq_empty (pass_wireRight H a), Finset.empty_union]

/-- Under `flipLay`'s wiring, a trace with no incorrect proof block has negations of opposite values. -/
theorem flip_vals (H : Msg → Msg) {w : Wit} (h : incorrect H accCm flipStmt flipPart flipLay.wires w = ∅) :
    ∃ b, w.val H accCm flipStmt 1 = some b ∧ w.val H accCm flipStmt 3 = some (!b) := by
  classical
  obtain ⟨hf, hw⟩ := Finset.union_eq_empty.1 h
  have hnf : ¬ Faulty H accCm flipStmt flipPart w 1 := fun hf1 => by
    have h1 : (1 : ℕ) ∈ faulty H accCm flipStmt flipPart w :=
      Finset.mem_filter.2 ⟨Finset.mem_range.2 (by decide), hf1⟩
    rw [hf] at h1
    exact Finset.notMem_empty 1 h1
  have hnw : ¬ WireWrong H accCm flipStmt flipPart flipLay.wires w 1 := fun hw1 => by
    have h1 : (1 : ℕ) ∈ wireWrong H accCm flipStmt flipPart flipLay.wires w :=
      Finset.mem_filter.2 ⟨Finset.mem_range.2 (by decide), hw1⟩
    rw [hw] at h1
    exact Finset.notMem_empty 1 h1
  obtain ⟨u, hu, ι, hc⟩ := Classical.not_not.1 hnf
  obtain rfl : [2, 3] = u := Option.some.inj hu
  have h32 := pair_val (C := flipProg) (i := 2) (j := 3) rfl rfl (by omega) (by decide) (by decide) hc
  have hr : WireRight flipStmt.prog flipLay.wires (w.val H accCm flipStmt) 2 := by
    by_contra hn
    exact hnw ⟨[2, 3], rfl, 2, Or.inl (by simp), hn⟩
  obtain ⟨-, b, h2, h1⟩ := hr 1 rfl
  exact ⟨b, h1, by simp [h32, h2]⟩

/-- Under `passLay`'s wiring, a trace with no incorrect proof block has negations of one value. -/
theorem pass_vals (H : Msg → Msg) {w : Wit} (h : incorrect H accCm flipStmt flipPart passLay.wires w = ∅) :
    ∃ b, w.val H accCm flipStmt 1 = some b ∧ w.val H accCm flipStmt 3 = some b := by
  classical
  obtain ⟨hf, hw⟩ := Finset.union_eq_empty.1 h
  have hnf : ∀ v, v < flipPart.blocks.length → ¬ Faulty H accCm flipStmt flipPart w v := fun v hv hfv => by
    have h1 : v ∈ faulty H accCm flipStmt flipPart w := Finset.mem_filter.2 ⟨Finset.mem_range.2 hv, hfv⟩
    rw [hf] at h1
    exact Finset.notMem_empty v h1
  have hnw : ¬ WireWrong H accCm flipStmt flipPart passLay.wires w 1 := fun hw1 => by
    have h1 : (1 : ℕ) ∈ wireWrong H accCm flipStmt flipPart passLay.wires w :=
      Finset.mem_filter.2 ⟨Finset.mem_range.2 (by decide), hw1⟩
    rw [hw] at h1
    exact Finset.notMem_empty 1 h1
  obtain ⟨u0, hu0, ι0, hc0⟩ := Classical.not_not.1 (hnf 0 (by decide))
  obtain rfl : [0, 1] = u0 := Option.some.inj hu0
  obtain ⟨u1, hu1, ι1, hc1⟩ := Classical.not_not.1 (hnf 1 (by decide))
  obtain rfl : [2, 3] = u1 := Option.some.inj hu1
  have h10 := pair_val (C := flipProg) (i := 0) (j := 1) rfl rfl Nat.one_pos (by decide) (by decide) hc0
  have h32 := pair_val (C := flipProg) (i := 2) (j := 3) rfl rfl (by omega) (by decide) (by decide) hc1
  have hr : WireRight flipStmt.prog passLay.wires (w.val H accCm flipStmt) 2 := by
    by_contra hn
    exact hnw ⟨[2, 3], rfl, 2, Or.inl (by simp), hn⟩
  obtain ⟨-, b, h2, h0⟩ := hr 0 rfl
  exact ⟨!b, by simp [h10, h0], by simp [h32, h2]⟩

/-! ## The rows the traces carry -/

theorem flipRows {ref : ℕ → Option Bool} {p q : Bool} (h1 : ref 1 = some p) (h3 : ref 3 = some q) (t : ℕ)
    (ℓ : Fin 1) : rowOf (traceVal flipProg ref) (flipEgress t ℓ) = (flipObs p q).rows ℓ := by
  have t1 : traceVal flipProg ref 1 = p := by simp [traceVal, Prog.read, flipProg, h1]
  have t3 : traceVal flipProg ref 3 = q := by simp [traceVal, Prog.read, flipProg, h3]
  funext σ
  by_cases hσ : σ = 0 <;> simp [rowOf, flipEgress, flipObs, hσ, t1, t3]

theorem flipObs_inj {p q p' q' : Bool} (h : [flipObs p q] = [flipObs p' q']) : p = p' ∧ q = q' := by
  have h' : flipObs p q = flipObs p' q' := List.head_eq_of_cons_eq h
  have h0 := congrArg (fun x : Obs 1 1 2 => (x.rows 0 0).isSome) h'
  have h1 := congrArg (fun x : Obs 1 1 2 => (x.rows 0 1).isSome) h'
  revert h0 h1
  cases p <;> cases q <;> cases p' <;> cases q' <;> decide

theorem obs_eq {x : Obs 1 1 2} {p q : Bool} (hr : ∀ ℓ, x.rows ℓ = (flipObs p q).rows ℓ) (hj : x.jitter < 1)
    (hs : ∀ κ, x.status κ = .complete) : x = flipObs p q := by
  obtain ⟨rows, jitter, status⟩ := x
  obtain rfl : jitter = 0 := Nat.lt_one_iff.1 hj
  obtain rfl : rows = (flipObs p q).rows := funext hr
  obtain rfl : status = fun _ => .complete := funext hs
  rfl

theorem carries_flipObs {lay : BoundaryGates 1 2} (he : lay.egress = flipEgress) (hi : lay.ingress = fun _ => [])
    {ref : ℕ → Option Bool} {p q : Bool} (h1 : ref 1 = some p) (h3 : ref 3 = some q) :
    Carries lay flipProg ref flipOp 1 [flipObs p q] := by
  intro t ht
  obtain rfl : t = 0 := by omega
  refine ⟨Nat.one_pos, fun ℓ => ?_, ?_⟩
  · rw [he]
    exact (flipRows h1 h3 0 ℓ).symm
  · simp [hi, flipOp]

theorem covered_flipObs (p q : Bool) : Covered 1 [flipObs p q] := by
  intro t ht _ κ
  obtain rfl : t = 0 := by omega
  rfl

theorem mem_carried {lay : BoundaryGates 1 2} (he : lay.egress = flipEgress) (hi : lay.ingress = fun _ => [])
    (H : Msg → Msg) {bad : Wit → Finset ℕ} {w : Wit} (hw : bad w = ∅) {p q : Bool}
    (h1 : w.val H accCm flipStmt 1 = some p) (h3 : w.val H accCm flipStmt 3 = some q) :
    [flipObs p q] ∈ carried H accCm flipStmt flipPart lay 1 0 bad flipOp 1 :=
  ⟨rfl, by simp [flipObs], covered_flipObs p q, w, by rw [hw]; exact (flip_M (Finset.empty_subset _)).2 rfl,
    carries_flipObs he hi h1 h3⟩

/-- Counted at the faulty set, the traces under `flipLay` carry every pair of rows. -/
theorem flip_four (H : Msg → Msg) :
    (Set.univ : Set (Bool × Bool)).encard ≤
      (carried H accCm flipStmt flipPart flipLay 1 0 (faulty H accCm flipStmt flipPart) flipOp 1).encard := by
  refine Set.encard_le_encard_of_injOn (f := fun pq => [flipObs pq.1 pq.2]) (fun ⦃pq⦄ _ => ?_)
    (fun ⦃pq⦄ _ ⦃pq'⦄ _ h => ?_)
  · obtain ⟨-, h1, -, h3⟩ := wit4_val H (!pq.1) pq.1 (!pq.2) pq.2
    exact mem_carried rfl rfl H (flip_faulty H (by simp) (by simp)) h1 h3
  · obtain ⟨hp, hq⟩ := flipObs_inj h
    exact Prod.ext hp hq

/-- Counted at the incorrect set, the traces under `flipLay` carry rows of opposite slots only. -/
theorem flip_two (H : Msg → Msg) :
    carried H accCm flipStmt flipPart flipLay 1 0 (incorrect H accCm flipStmt flipPart flipLay.wires) flipOp 1 ⊆
      (fun b => [flipObs b (!b)]) '' Set.univ := by
  classical
  rintro o ⟨hlen, hjit, hcov, w, hM, hcar⟩
  have hsub : incorrect H accCm flipStmt flipPart flipLay.wires w ⊆ Finset.range 2 :=
    Finset.union_subset (Finset.filter_subset _ _) (Finset.filter_subset _ _)
  obtain ⟨b, h1, h3⟩ := flip_vals H ((flip_M hsub).1 hM)
  obtain ⟨x, rfl⟩ := List.length_eq_one_iff.1 hlen
  obtain ⟨_, hrows, _⟩ := hcar 0 Nat.one_pos
  exact ⟨b, Set.mem_univ _, congrArg (fun y => [y])
    (obs_eq (fun ℓ => (hrows ℓ).trans (flipRows h1 h3 0 ℓ)) (hjit x (by simp))
      (fun κ => hcov 0 Nat.one_pos Nat.one_pos κ)).symm⟩

/-- Counted at the incorrect set, the traces under `cycleLay` carry two rows. -/
theorem cycle_two (H : Msg → Msg) :
    (Set.univ : Set Bool).encard ≤
      (carried H accCm flipStmt flipPart cycleLay 1 0 (incorrect H accCm flipStmt flipPart cycleLay.wires) flipOp
        1).encard := by
  refine Set.encard_le_encard_of_injOn (f := fun a => [flipObs (!a) a]) (fun ⦃a⦄ _ => ?_)
    (fun ⦃a⦄ _ ⦃a'⦄ _ h => ?_)
  · obtain ⟨-, h1, -, h3⟩ := wit4_val H a (!a) (!a) a
    exact mem_carried rfl rfl H (cycle_incorrect H a) h1 h3
  · exact (flipObs_inj h).2

/-- Counted at the incorrect set, the traces under `passLay` carry rows of one value at both slots only. -/
theorem pass_rows (H : Msg → Msg) :
    carried H accCm flipStmt flipPart passLay 1 0 (incorrect H accCm flipStmt flipPart passLay.wires) flipOp 1 ⊆
      (fun b => [flipObs b b]) '' Set.univ := by
  classical
  rintro o ⟨hlen, hjit, hcov, w, hM, hcar⟩
  have hsub : incorrect H accCm flipStmt flipPart passLay.wires w ⊆ Finset.range 2 :=
    Finset.union_subset (Finset.filter_subset _ _) (Finset.filter_subset _ _)
  obtain ⟨b, h1, h3⟩ := pass_vals H ((flip_M hsub).1 hM)
  obtain ⟨x, rfl⟩ := List.length_eq_one_iff.1 hlen
  obtain ⟨_, hrows, _⟩ := hcar 0 Nat.one_pos
  exact ⟨b, Set.mem_univ _, congrArg (fun y => [y])
    (obs_eq (fun ℓ => (hrows ℓ).trans (flipRows h1 h3 0 ℓ)) (hjit x (by simp))
      (fun κ => hcov 0 Nat.one_pos Nat.one_pos κ)).symm⟩

/-- Counted at the incorrect set, the traces under `passLay` carry two rows, one for each value of the free input. -/
theorem pass_two (H : Msg → Msg) :
    (Set.univ : Set Bool).encard ≤
      (carried H accCm flipStmt flipPart passLay 1 0 (incorrect H accCm flipStmt flipPart passLay.wires) flipOp
        1).encard := by
  refine Set.encard_le_encard_of_injOn (f := fun a => [flipObs (!a) (!a)]) (fun ⦃a⦄ _ => ?_)
    (fun ⦃a⦄ _ ⦃a'⦄ _ h => ?_)
  · obtain ⟨-, h1, -, h3⟩ := wit4_val H a (!a) a (!a)
    exact mem_carried rfl rfl H (pass_incorrect H a) h1 h3
  · have h' := (flipObs_inj h).1
    revert h'
    cases a <;> cases a' <;> decide

/-! ## The toys -/

/-- **The wire flip.** `flipLay` fits `flipProg`, and the trace whose inputs are both `false` and negations both
`true` has each proof block correct and its wired input the opposite of its writer: the sampling relation at the
faulty set holds there (`R`, what `KS` gives at `δ = 0`), and the one at the incorrect set doesn't (`RWired`). Counted
at the faulty set, the traces carry more observation sequences than `SampledLeakAccepted`'s bound. -/
theorem flip_excludes (H : Msg → Msg) :
    flipLay.Fits flipProg 1 ∧
      R H accCm (fun _ => some flipStmt) (fun _ => some flipPart) 0 [] (wit4 false true false true) ∧
      ¬ RWired H accCm (fun _ => some flipStmt) (fun _ => some flipPart) (fun _ => flipLay.wires) 0 []
        (wit4 false true false true) ∧
      ¬ (carried H accCm flipStmt flipPart flipLay 1 0 (faulty H accCm flipStmt flipPart) flipOp 1).encard ≤
        ((2 ^ (freeInputs flipProg flipLay 1 \ ∅).card * (faultySets flipStmt.design flipPart 0).card * 2 ^ 0 *
          1 ^ 1 : ℕ) : ℕ∞) := by
  classical
  obtain ⟨-, h1, h2, -⟩ := wit4_val H false true false true
  refine ⟨flip_fits, ⟨flipStmt, flipPart, rfl, rfl, ?_⟩, ?_, ?_⟩
  · show faulty H accCm flipStmt flipPart (wit4 false true false true) ∈ flipStmt.design.M flipPart 0
    rw [flip_faulty H (a := false) (b := true) (c := false) (d := true) rfl rfl]
    exact (flip_M (Finset.empty_subset _)).2 rfl
  · rintro ⟨s, P, hs, hP, hM⟩
    obtain rfl : flipStmt = s := Option.some.inj hs
    obtain rfl : flipPart = P := Option.some.inj hP
    have hw : (1 : ℕ) ∈ incorrect H accCm flipStmt flipPart flipLay.wires (wit4 false true false true) :=
      Finset.mem_union_right _ (Finset.mem_filter.2 ⟨Finset.mem_range.2 (by decide),
        ⟨[2, 3], rfl, 2, Or.inl (by simp), fun hr => by
          obtain ⟨-, b, hb2, hb1⟩ := hr 1 rfl
          rw [h2] at hb2
          rw [h1] at hb1
          exact absurd ((Option.some.inj hb2).trans (Option.some.inj hb1).symm) (by decide)⟩⟩)
    have hsub : incorrect H accCm flipStmt flipPart flipLay.wires (wit4 false true false true) ⊆ Finset.range 2 :=
      Finset.union_subset (Finset.filter_subset _ _) (Finset.filter_subset _ _)
    rw [(flip_M hsub).1 hM] at hw
    exact Finset.notMem_empty 1 hw
  · rw [flip_bound]
    intro h
    have h4 := (flip_four H).trans h
    rw [Set.encard_univ, ENat.card_eq_coe_fintype_card] at h4
    exact absurd (ENat.natCast_le_natCast.1 h4) (by decide)

/-- **The wire flip, at the incorrect set**: counted there, the traces under `flipLay` carry at most
`SampledLeakAccepted`'s bound of observation sequences. -/
theorem flip_meets (H : Msg → Msg) :
    (carried H accCm flipStmt flipPart flipLay 1 0 (incorrect H accCm flipStmt flipPart flipLay.wires) flipOp
        1).encard ≤
      ((2 ^ (freeInputs flipProg flipLay 1 \ ∅).card * (faultySets flipStmt.design flipPart 0).card * 2 ^ 0 *
        1 ^ 1 : ℕ) : ℕ∞) := by
  rw [flip_bound]
  calc _ ≤ ((fun b => [flipObs b (!b)]) '' Set.univ).encard := Set.encard_le_encard (flip_two H)
    _ ≤ (Set.univ : Set Bool).encard := Set.encard_image_le _ _
    _ = ((2 : ℕ) : ℕ∞) := by rw [Set.encard_univ, ENat.card_eq_coe_fintype_card, Fintype.card_bool]

/-- **The two-cycle.** `cycleLay` fits `flipProg` but for its wires' cycle, and the traces with no incorrect proof
block under it carry more observation sequences than `SampledLeakAccepted`'s bound. -/
theorem cycle_excludes (H : Msg → Msg) :
    (cycleLay.Fits flipProg 1 ↔ WiresAcyclic flipProg cycleLay.wires) ∧ ¬ WiresAcyclic flipProg cycleLay.wires ∧
      ¬ (carried H accCm flipStmt flipPart cycleLay 1 0 (incorrect H accCm flipStmt flipPart cycleLay.wires) flipOp
          1).encard ≤
        ((2 ^ (freeInputs flipProg cycleLay 1 \ ∅).card * (faultySets flipStmt.design flipPart 0).card * 2 ^ 0 *
          1 ^ 1 : ℕ) : ℕ∞) := by
  refine ⟨cycle_fits_iff, cycle_not_acyclic, ?_⟩
  rw [cycle_bound]
  intro h
  have h2 := (cycle_two H).trans h
  rw [Set.encard_univ, ENat.card_eq_coe_fintype_card] at h2
  exact absurd (ENat.natCast_le_natCast.1 h2) (by decide)

/-- **The pass-through, counted with its chain's end**: the first input, which writes `passLay`'s one wire, stays a
free input, and counted at the incorrect set the traces under `passLay` carry at most `SampledLeakAccepted`'s bound of
observation sequences. -/
theorem pass_meets (H : Msg → Msg) :
    (carried H accCm flipStmt flipPart passLay 1 0 (incorrect H accCm flipStmt flipPart passLay.wires) flipOp
        1).encard ≤
      ((2 ^ (freeInputs flipProg passLay 1 \ ∅).card * (faultySets flipStmt.design flipPart 0).card * 2 ^ 0 *
        1 ^ 1 : ℕ) : ℕ∞) := by
  rw [pass_bound]
  calc _ ≤ ((fun b => [flipObs b b]) '' Set.univ).encard := Set.encard_le_encard (pass_rows H)
    _ ≤ (Set.univ : Set Bool).encard := Set.encard_image_le _ _
    _ = ((2 : ℕ) : ℕ∞) := by rw [Set.encard_univ, ENat.card_eq_coe_fintype_card, Fintype.card_bool]

/-- **The pass-through.** `passLay` fits `flipProg`, its one wire's chain ending at the first input, which `flipProg`
doesn't return and which stays a free input. The traces with no incorrect proof block under it carry more observation
sequences than the bound would be with the chain's end left out of the free inputs too. -/
theorem pass_excludes (H : Msg → Msg) :
    passLay.Fits flipProg 1 ∧ 0 ∉ flipProg.outs ∧ 0 ∈ freeInputs flipProg passLay 1 ∧
      ¬ (carried H accCm flipStmt flipPart passLay 1 0 (incorrect H accCm flipStmt flipPart passLay.wires) flipOp
          1).encard ≤
        ((2 ^ ((freeInputs flipProg passLay 1).erase 0).card * (faultySets flipStmt.design flipPart 0).card * 2 ^ 0 *
          1 ^ 1 : ℕ) : ℕ∞) := by
  refine ⟨pass_fits, pass_writer_unreturned, pass_end, ?_⟩
  rw [pass_bound_erase]
  intro h
  have h2 := (pass_two H).trans h
  rw [Set.encard_univ, ENat.card_eq_coe_fintype_card] at h2
  exact absurd (ENat.natCast_le_natCast.1 h2) (by decide)

end NetTiming.WiredToys
