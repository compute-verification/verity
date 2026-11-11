module

public import Definitions.Claims.SamplingTwoReg
public import Proofs.Flock.SamplingGeneric.Reg
public import Proofs.Flock.SamplingGeneric.TwoCap
public import Proofs.Flock.SamplingGeneric.TwoCount
import Proofs.Flock.Claims.Aggregate
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Finder
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Binding
import Proofs.Flock.Claims.Sampling.Extract
import Proofs.Flock.SamplingGeneric.Assemble
import Proofs.Flock.Soundness.Rewinding

/-!
# The two-stage committed form's event

The primary conjunct of `CompiledTwo.KSRegTwo`, from the sessions' bound at `CompilerKSTwo`'s premises: an accepted
run whose sample holds a proof block of the committed incorrect set (`CompiledTwo.FregTwo`) has that block's session
unfound at its first slot, or its extractions disagree with themselves or open a gate off the committed boundary trace
(the one-stage form's events at `L.toCompiled`: `Compiled.InConf`, `Compiled.OffReg`), or open a gate off its replay
block's committed interior (`OffIreg`).

* `correct_refIn_congr`: a proof block's correctness on its replay block's values reads the boundary values only below
  the program's gates and the interior values only at the replay block's gates.
* `iopensAt_some`, `iopensAt_of_mem`: what a run's extractions open under a replay block's interior roots is a kept
  opening that passes, and, at a run that doesn't disagree with itself, every such opening.
* `iop`, `iplur`, `OffIreg`, `IConflict`: at a tape and first-stage coins, the second stages' openings of the program's
  gates, each under the interior roots of the replay block that holds it (`gateBlock`); the committed interiors as their
  plurality; a run off them; and two second stages that open a gate two ways.
* `pr_offIreg_le`: **the rewinding lemma at the committed interior**: at a tape and first-stage coins, a second stage's
  extractions open a gate off the committed interior with probability at most `N·√Pr[two second stages conflict]`.
* `pr_iconflict_le`, `ex_offIreg_le`: two second stages at shared first-stage coins see one interior root per replay
  block (`Sessions.interior`), so their conflict is a collision `B_conf,2` finds (`CompiledTwo.BconfTwo`); averaged,
  the run is off the committed interior with probability at most `N·√Pr[B_conf,2]`.
* `regTwo_split`: **the deterministic half**, at any boundary trace: at an accepted run that samples a proof block
  incorrect at the boundary trace and its replay block's committed interior, the first such block's session is unfound
  at its first slot, or the run disagrees with itself, or is off the boundary trace or the committed interior;
  otherwise `Sessions.opens` at the block's replay block makes it correct at those traces.
* `Compiled.pr_unfound_le_at`, `Compiled.ex_offReg_le`: the sessions' bound at a selection that reads the run's coins
  off the second stage only, and the rewinding lemma at the committed boundary trace averaged over the tape.
* `eventTwo_split`, `eventTwo_le`: the event at a tape and a boundary trace as the union of those four, and, at the
  committed boundary trace, **the primary conjunct of `CompiledTwo.KSRegTwo`**, at most `ε_reg,2`.

The deterministic half and the event's union read the boundary trace as a parameter, so the form at the boundary trace
read at the statement (`SamplingGeneric/TwoRegX.lean`) takes them as they are.

The committed interior and the committed incorrect set read no second-stage coin (`CompiledTwo.iregVal_forkOn`,
`CompiledTwo.FregTwo_forkOn`), and the incorrect set is the grid's incorrect proof blocks at the committed traces
(`CompiledTwo.FregTwo_eq`), all in `SamplingGeneric/TwoCap.lean`.
-/

@[expose] public section

namespace Claims.Sampling

/-- **What a proof block's correctness reads of its replay block's values**: on a topological program that `Pt`
partitions, a proof block of `Pt` is correct at `refIn rb val ι` exactly when it is at `refIn rb val' ι'`, for boundary
values that agree below the program's gates and interior values that agree at `rb`'s gates below them. -/
theorem correct_refIn_congr {C : Prog} {Pt : Partition} (htop : C.Topological) (hP : Pt.Partitions C) {v : ℕ}
    {u : List ℕ} (hu : Pt.blocks[v]? = some u) (rb : List ℕ) {val val' : ℕ → Option Bool} {ι ι' : ℕ → Bool}
    (hval : ∀ g < C.gates.length, val g = val' g) (hι : ∀ g < C.gates.length, g ∈ rb → ι g = ι' g) :
    C.Correct (C.refIn rb val ι) u ↔ C.Correct (C.refIn rb val' ι') u := by
  have href : ∀ g < C.gates.length, C.refIn rb val ι g = C.refIn rb val' ι' g := by
    intro g hg
    unfold Prog.refIn
    split_ifs with h
    · rw [hι g hg h.1]
    · exact hval g hg
  have hlt : ∀ g ∈ u, g < C.gates.length := fun g hg => hP.2.1 g (mem_gates hu hg)
  exact C.correct_congr (fun g hg => href g (hlt g hg)) fun g hg x hx i hi _ =>
    href i ((htop.1 g x hx i hi).trans (hlt g hg))

end Claims.Sampling

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Opening Partition InScopeTwo cell replayOf forkOn wrongInAt
  gateBlock gateBlock_of_mem correct_refIn_congr val_assemble ιOf ιOf_eq wrongAllAt wrongAllAt_subset)

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The sessions' bound at a selection that reads the first stage's coins**, averaged over the tape: the run accepts
and the proof block at the picked slot is not found at the slot's coins with probability at most
`κ_s + √(k(x)·γ_s·Pr[B])`, for a selection that reads the run's coins off the second stage only. -/
theorem pr_unfound_le_at (H : Msg → Msg) (L : Compiled Coin A) {x : Msg} {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (β : Ω → (Fin (L.V.k x) → Coin) → ℝ) (hβ0 : ∀ ω r, 0 ≤ β ω r)
    (hβ : Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => β p.1 p.2) ≤ PrB H L.V P x)
    (sel : Ω → (Fin (L.V.k x) → Coin) → List ℕ → Option (Fin (L.nPU x)))
    (hsel : ∀ ω r e, sel ω (forkOn (L.stage2 x) r e) = sel ω r)
    (hκ : ∀ ω r (sl : List ℕ → Option (Fin (L.nPU x))), Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      (run H P L.V x ω (forkOn (L.stage2 x) r p.1)).out = .acc ∧
        ∃ i, sl (L.checked x (forkOn (L.stage2 x) r p.1)) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.stage2 x) r p.1))[i.val]? = some u.val ∧
            ¬ L.Found H x (run H P L.V x ω) u (forkOn (L.stage2 x) r p.1) (p.2 i u)) ≤
        L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r)) :
    Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      (run H P L.V x ω rc.1).out = .acc ∧ ∃ i, sel ω rc.1 (L.checked x rc.1) = some i ∧
        ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
          ¬ L.Found H x (run H P L.V x ω) u rc.1 (rc.2 i u)) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) := by
  have hγ0 : 0 ≤ (L.V.k x : ℝ) * L.γs x := mul_nonneg (Nat.cast_nonneg _) (L.γs_nonneg x)
  calc Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        (run H P L.V x ω rc.1).out = .acc ∧ ∃ i, sel ω rc.1 (L.checked x rc.1) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
            ¬ L.Found H x (run H P L.V x ω) u rc.1 (rc.2 i u))
      ≤ Ex (fun ω : Ω => L.κs x + Ex fun r => Real.sqrt (L.V.k x * L.γs x * β ω r)) :=
        Ex_mono fun ω => (Pr_forkOn_prod (L.stage2 x) (fun (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) =>
            (run H P L.V x ω r).out = .acc ∧ ∃ i, sel ω r (L.checked x r) = some i ∧
              ∃ u : Fin (L.nPU x), (L.checked x r)[i.val]? = some u.val ∧
                ¬ L.Found H x (run H P L.V x ω) u r (c i u))).trans_le
          ((Ex_mono fun r => (Pr_mono fun p hp => by rwa [hsel] at hp).trans (hκ ω r (sel ω r))).trans_eq
            (by rw [Ex_add, Ex_const]))
    _ = L.κs x + Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => Real.sqrt (L.V.k x * L.γs x * β p.1 p.2)) := by
        rw [Ex_add, Ex_const,
          Ex_prod (fun p : Ω × (Fin (L.V.k x) → Coin) => Real.sqrt (L.V.k x * L.γs x * β p.1 p.2))]
    _ ≤ L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) := by
        refine add_le_add le_rfl ((Ex_sqrt_le _ fun p => mul_nonneg hγ0 (hβ0 _ _)).trans (Real.sqrt_le_sqrt ?_))
        rw [Ex_mul_left]
        exact mul_le_mul_of_nonneg_left hβ hγ0

/-- **The rewinding lemma at the committed trace, averaged over the tape**: at most `N·√Pr[B_conf]` (Jensen's
inequality). -/
theorem ex_offReg_le {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2) ≤
      L.nG x * Real.sqrt ((Bconf H L Ω P x).Pr H) := by
  refine (Ex_mono fun ω => L.pr_offReg_le H P x ω).trans ?_
  rw [Ex_mul_left]
  exact mul_le_mul_of_nonneg_left ((Ex_sqrt_le _ fun _ => Pr_nonneg _).trans
    (Real.sqrt_le_sqrt (pr_conflict_le hbind hs P))) (Nat.cast_nonneg _)

end Compiled

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## A run's interior openings -/

/-- What a run's extractions open under replay block `ρ`'s interior roots at gate `g` is a kept opening there that
passes. -/
theorem iopensAt_some {H : Msg → Msg} {L : CompiledTwo Coin A} {Ω : Type} {P : Prover Ω} {x : Msg} {ω : Ω}
    {s : AuditStatement} (hs : L.parse x = some s) {r : Fin (L.V.k x) → Coin} {c : L.SlotEC x} {ρ g : ℕ} {b : Bool}
    (h : L.iopensAt H P x ω r c ρ g = some b) :
    ∃ q ∈ L.kept H P x s ω r c, q.1 = L.iroots x (run H P L.V x ω r).seen ρ ∧ q.2.pos = g ∧ q.2.val = b ∧
      L.commit.Opens H q.1 q.2 = true := by
  simp only [iopensAt, hs] at h
  obtain ⟨q, hq, rfl⟩ := Option.map_eq_some_iff.1 h
  have hp := List.find?_some hq
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hp
  exact ⟨q, List.mem_of_find?_eq_some hq, hp.1.1, hp.1.2, rfl, hp.2⟩

/-- At a run whose kept openings agree with themselves, each kept opening under `ρ`'s interior roots that passes is
what the run's extractions open there. -/
theorem iopensAt_of_mem {H : Msg → Msg} {L : CompiledTwo Coin A} {Ω : Type} {P : Prover Ω} {x : Msg} {ω : Ω}
    {s : AuditStatement} (hs : L.parse x = some s) {r : Fin (L.V.k x) → Coin} {c : L.SlotEC x}
    (hagree : ∀ q ∈ L.kept H P x s ω r c, ∀ q' ∈ L.kept H P x s ω r c, q.1 = q'.1 → q.2.pos = q'.2.pos →
      q.2.val = q'.2.val)
    {ρ : ℕ} {q : List Msg × Opening Bool} (hq : q ∈ L.kept H P x s ω r c)
    (h1 : q.1 = L.iroots x (run H P L.V x ω r).seen ρ) (hok : L.commit.Opens H q.1 q.2 = true) :
    L.iopensAt H P x ω r c ρ q.2.pos = some q.2.val := by
  have hsome : (L.iopensAt H P x ω r c ρ q.2.pos).isSome := by
    simp only [iopensAt, hs, Option.isSome_map]
    exact List.find?_isSome.2 ⟨q, hq, by rw [← h1]; simp [hok]⟩
  obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1 hsome
  obtain ⟨q', hq', h1', h2', h3', -⟩ := iopensAt_some hs hb
  rw [hb, ← h3']
  exact congrArg some (hagree q' hq' q hq (h1'.trans h1.symm) h2')

/-! ## The rewinding lemma at the committed interior -/

/-- At a tape and first-stage coins `r`, a second stage's and the extractor's coins `ec` open gate `g` under the
interior roots of the replay block of `Pρ` that holds it; `none` at a gate no replay block holds. -/
noncomputable def iop (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (Pρ : Partition) (r : Fin (L.V.k x) → Coin) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : ℕ) : Option Bool :=
  (gateBlock Pρ g).bind fun ρ => L.iopensAt H P x ω (forkOn (L.stage2 x) r ec.1) ec.2 ρ g

/-- `iop` on the program's gates, as the rewinding lemma's table. -/
noncomputable def iopTable (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (Pρ : Partition) (r : Fin (L.V.k x) → Coin) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) :
    Option Bool :=
  L.iop H P x ω Pρ r ec g

/-- The committed interior of the replay block of `Pρ` that holds gate `g`; `true` at a gate no replay block holds. -/
noncomputable def iplur (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (Pρ : Partition) (r : Fin (L.V.k x) → Coin) (g : ℕ) : Bool :=
  ((gateBlock Pρ g).map fun ρ => L.iregVal H P x ω r ρ g).getD true

/-- A run's extractions open a gate of the program, under the interior roots of the replay block of `Pρ` that holds it,
off that block's committed interior. -/
def OffIreg (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (Pρ : Partition)
    (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) : Prop :=
  ∃ g < L.nG x, ∃ ρ, gateBlock Pρ g = some ρ ∧ ∃ b, L.iopensAt H P x ω r c ρ g = some b ∧ L.iregVal H P x ω r ρ g ≠ b

/-- Two second stages on one tape and one first stage open a gate of the program two ways, each under the interior
roots of the replay block of `Pρ` that holds it. -/
def IConflict (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (Pρ : Partition)
    (r : Fin (L.V.k x) → Coin)
    (ee : ((Fin (L.V.k x) → Coin) × L.SlotEC x) × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) : Prop :=
  ∃ g : Fin (L.nG x), (L.iopTable H P x ω Pρ r ee.1 g).isSome ∧ (L.iopTable H P x ω Pρ r ee.2 g).isSome ∧
    L.iopTable H P x ω Pρ r ee.1 g ≠ L.iopTable H P x ω Pρ r ee.2 g

/-- **The rewinding lemma at the committed interior**: at a tape and first-stage coins, a second stage's extractions
open a gate of the program off the committed interior with probability at most `N·√Pr[conflict]`, `N` the program's
gates (`FlockSoundness.Rewinding.prBad_le`, the committed interior being the second stages' openings' plurality). -/
theorem pr_offIreg_le (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (Pρ : Partition) (r : Fin (L.V.k x) → Coin) :
    Pr (fun ec : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffIreg H P x ω Pρ (forkOn (L.stage2 x) r ec.1) ec.2) ≤
      L.nG x * Real.sqrt (Pr fun ee : ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
        ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.IConflict H P x ω Pρ r ee) := by
  classical
  have hcard : (0 : ℝ) < Fintype.card ((Fin (L.V.k x) → Coin) × L.SlotEC x) := by
    exact_mod_cast Fintype.card_pos
  have hcnt : ∀ (g : Fin (L.nG x)) (b : Bool),
      (#(FlockSoundness.Rewinding.openedTo (L.iopTable H P x ω Pρ r) g b) : ℝ) =
        Pr (fun ec : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.iop H P x ω Pρ r ec g = some b) *
          Fintype.card ((Fin (L.V.k x) → Coin) × L.SlotEC x) := by
    intro g b
    rw [Pr_eq_card_of_mem (fun ec : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.iop H P x ω Pρ r ec g = some b)
      (FlockSoundness.Rewinding.openedTo (L.iopTable H P x ω Pρ r) g b)
      fun ec => mem_filter.trans (and_iff_right (mem_univ _)), div_mul_cancel₀ _ hcard.ne']
  have hpl : FlockSoundness.Rewinding.IsPlurality (L.iopTable H P x ω Pρ r) fun g => L.iplur H P x ω Pρ r g := by
    intro g v
    have h : Pr (fun ec : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.iop H P x ω Pρ r ec g = some v) ≤
        Pr (fun ec : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
          L.iop H P x ω Pρ r ec g = some (L.iplur H P x ω Pρ r g)) := by
      cases hgb : gateBlock Pρ g with
      | none => exact Pr_mono fun ec hec => by simp [iop, hgb] at hec
      | some ρ =>
        simp only [iop, iplur, hgb, Option.bind_some, Option.map_some, Option.getD_some]
        exact le_plur (fun b => Pr fun ec : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
          L.iopensAt H P x ω (forkOn (L.stage2 x) r ec.1) ec.2 ρ g = some b) v
    have h' : (#(FlockSoundness.Rewinding.openedTo (L.iopTable H P x ω Pρ r) g v) : ℝ) ≤
        #(FlockSoundness.Rewinding.openedTo (L.iopTable H P x ω Pρ r) g (L.iplur H P x ω Pρ r g)) := by
      rw [hcnt, hcnt]
      exact mul_le_mul_of_nonneg_right h hcard.le
    exact_mod_cast h'
  have hbad := FlockSoundness.Rewinding.prBad_le (op := L.iopTable H P x ω Pρ r)
    (fun g => L.iplur H P x ω Pρ r g) hpl (L.nG x)
    fun ec => (card_filter_le _ _).trans (card_univ.trans (Fintype.card_fin _)).le
  have hoff : Pr (fun ec : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      L.OffIreg H P x ω Pρ (forkOn (L.stage2 x) r ec.1) ec.2) ≤
      (#(FlockSoundness.Rewinding.bad (L.iopTable H P x ω Pρ r) fun g => L.iplur H P x ω Pρ r g) : ℝ) /
        Fintype.card ((Fin (L.V.k x) → Coin) × L.SlotEC x) := by
    calc Pr (fun ec : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
          L.OffIreg H P x ω Pρ (forkOn (L.stage2 x) r ec.1) ec.2)
        ≤ Pr (fun ec : (Fin (L.V.k x) → Coin) × L.SlotEC x => ∃ g,
            FlockSoundness.Rewinding.Off (L.iopTable H P x ω Pρ r) (fun g => L.iplur H P x ω Pρ r g) ec g) :=
          Pr_mono fun ec hec => by
            obtain ⟨g, hg, ρ, hgb, b, hb, hne⟩ := hec
            have hiop : L.iop H P x ω Pρ r ec g = some b := by
              simp only [iop, hgb, Option.bind_some]
              exact hb
            have hpl' : L.iplur H P x ω Pρ r g = L.iregVal H P x ω r ρ g := by
              simp only [iplur, hgb, Option.map_some, Option.getD_some]
            refine ⟨⟨g, hg⟩, Option.isSome_iff_exists.2 ⟨b, hiop⟩, fun h => hne ?_⟩
            have h' : some b = some (L.iplur H P x ω Pρ r g) := hiop.symm.trans h
            rw [iregVal_forkOn]
            exact hpl'.symm.trans (Option.some.inj h').symm
      _ = _ := Pr_eq_card_of_mem _
          (FlockSoundness.Rewinding.bad (L.iopTable H P x ω Pρ r) fun g => L.iplur H P x ω Pρ r g)
          fun ec => by simp only [FlockSoundness.Rewinding.bad, mem_filter, mem_univ, true_and]
  have hconf : (#(FlockSoundness.Rewinding.conflict (L.iopTable H P x ω Pρ r)) : ℝ) /
      (Fintype.card ((Fin (L.V.k x) → Coin) × L.SlotEC x) : ℝ) ^ 2 =
      Pr fun ee : ((Fin (L.V.k x) → Coin) × L.SlotEC x) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        L.IConflict H P x ω Pρ r ee := by
    rw [Pr_eq_card_of_mem (fun ee : ((Fin (L.V.k x) → Coin) × L.SlotEC x) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        L.IConflict H P x ω Pρ r ee) (FlockSoundness.Rewinding.conflict (L.iopTable H P x ω Pρ r))
        fun ee => by simp only [FlockSoundness.Rewinding.conflict, mem_filter, mem_univ, true_and]; exact Iff.rfl,
      Fintype.card_prod ((Fin (L.V.k x) → Coin) × L.SlotEC x) ((Fin (L.V.k x) → Coin) × L.SlotEC x),
      Nat.cast_mul, sq]
  calc Pr (fun ec : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffIreg H P x ω Pρ (forkOn (L.stage2 x) r ec.1) ec.2)
      ≤ _ := hoff
    _ ≤ _ := hbad
    _ = L.nG x * Real.sqrt (Pr fun ee : ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
          ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.IConflict H P x ω Pρ r ee) := by
      rw [Fintype.card_fin, hconf, Real.sqrt_mul (mul_self_nonneg (L.nG x : ℝ)),
        Real.sqrt_mul_self (Nat.cast_nonneg _)]

/-! ## Binding: two second stages' conflicts are `B_conf,2`'s collisions -/

/-- An event of a tape, first-stage coins and two second stages on them that gives a collision `B_conf,2`'s output is
at most `Pr[B_conf,2]`. -/
theorem le_bconfTwo (H : Msg → Msg) (L : CompiledTwo Coin A) {x : Msg} {s : AuditStatement}
    (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    {Q : Ω × (Fin (L.V.k x) → Coin) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
      ((Fin (L.V.k x) → Coin) × L.SlotEC x) → Prop}
    (hQ : ∀ q, Q q → ∃ u v,
      findColl (Sampling.ownLog H (run H P L.V x q.1) (L.runExt x s (forkOn (L.stage2 x) q.2.1 q.2.2.1.1)
          (run H P L.V x q.1 (forkOn (L.stage2 x) q.2.1 q.2.2.1.1)).seen q.2.2.1.2) ++
        Sampling.ownLog H (run H P L.V x q.1) (L.runExt x s (forkOn (L.stage2 x) q.2.1 q.2.2.2.1)
          (run H P L.V x q.1 (forkOn (L.stage2 x) q.2.1 q.2.2.2.1)).seen q.2.2.2.2)) = some (u, v) ∧
        Collision H u v) :
    Pr Q ≤ (BconfTwo H L Ω P x).Pr H := by
  simp only [BconfTwo, hs]
  exact Pr_mono hQ

/-- **Two second stages' conflict at the committed interior**: at most `Pr[B_conf,2]`. Both read gate `g` under the
interior roots of one replay block, which the two second stages share (`Sessions.interior`). -/
theorem pr_iconflict_le {H : Msg → Msg} {L : CompiledTwo Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    {x : Msg} {s : AuditStatement} (hs : L.parse x = some s) (Pρ : Partition) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) :
    Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => Pr fun ee : ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
      ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.IConflict H P x p.1 Pρ p.2 ee) ≤ (BconfTwo H L Ω P x).Pr H := by
  have hEx : Pr (fun q : Ω × (Fin (L.V.k x) → Coin) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
        ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.IConflict H P x q.1 Pρ q.2.1 q.2.2) =
      Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => Pr fun ee : ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
        ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.IConflict H P x p.1 Pρ p.2 ee) := by
    rw [Sampling.Pr_prod_eq_Ex, Ex_prod]
    exact congrArg Ex (funext fun ω => Sampling.Pr_prod_eq_Ex _)
  rw [← hEx]
  refine le_bconfTwo H L hs P fun q hq => ?_
  obtain ⟨g, h1, h2, hne⟩ := hq
  obtain ⟨b1, hb1⟩ := Option.isSome_iff_exists.1 h1
  obtain ⟨b2, hb2⟩ := Option.isSome_iff_exists.1 h2
  rw [hb1, hb2] at hne
  simp only [iopTable, iop] at hb1 hb2
  obtain ⟨ρ, hgb, hb1'⟩ := Option.bind_eq_some_iff.1 hb1
  obtain ⟨ρ', hgb', hb2'⟩ := Option.bind_eq_some_iff.1 hb2
  rw [hgb, Option.some.injEq] at hgb'
  rw [← hgb'] at hb2'
  obtain ⟨a, ha, ha1, ha2, ha3, -⟩ := iopensAt_some hs hb1'
  obtain ⟨a', ha', ha1', ha2', ha3', -⟩ := iopensAt_some hs hb2'
  have hroots : a.1 = a'.1 := by
    rw [ha1, ha1', hS.interior x Ω P q.1 q.2.1 q.2.2.1.1 ρ, hS.interior x Ω P q.1 q.2.1 q.2.2.2.1 ρ]
  exact Compiled.bconf_of_disagree hbind s P q.1 (forkOn (L.stage2 x) q.2.1 q.2.2.1.1, q.2.2.1.2)
    (forkOn (L.stage2 x) q.2.1 q.2.2.2.1, q.2.2.2.2) (Or.inl ha) (Or.inr ha') hroots (ha2.trans ha2'.symm)
    (by rw [ha3, ha3']; exact fun h => hne (congrArg some h))

/-- **The rewinding lemma at the committed interior, averaged over the tape and the first stage**: at most
`N·√Pr[B_conf,2]` (Jensen's inequality). -/
theorem ex_offIreg_le {H : Msg → Msg} {L : CompiledTwo Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    {x : Msg} {s : AuditStatement} (hs : L.parse x = some s) (Pρ : Partition) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) :
    Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffIreg H P x ω Pρ rc.1 rc.2) ≤
      L.nG x * Real.sqrt ((BconfTwo H L Ω P x).Pr H) :=
  calc Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffIreg H P x ω Pρ rc.1 rc.2)
      ≤ Ex (fun ω : Ω => Ex fun r : Fin (L.V.k x) → Coin => L.nG x * Real.sqrt (Pr fun ee :
          ((Fin (L.V.k x) → Coin) × L.SlotEC x) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
            L.IConflict H P x ω Pρ r ee)) :=
        Ex_mono fun ω => (Pr_forkOn_prod (L.stage2 x) (fun (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) =>
          L.OffIreg H P x ω Pρ r c)).trans_le (Ex_mono fun r => L.pr_offIreg_le H P x ω Pρ r)
    _ = Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => L.nG x * Real.sqrt (Pr fun ee :
          ((Fin (L.V.k x) → Coin) × L.SlotEC x) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
            L.IConflict H P x p.1 Pρ p.2 ee)) :=
        (Ex_prod (fun p : Ω × (Fin (L.V.k x) → Coin) => L.nG x * Real.sqrt (Pr fun ee :
          ((Fin (L.V.k x) → Coin) × L.SlotEC x) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
            L.IConflict H P x p.1 Pρ p.2 ee))).symm
    _ = L.nG x * Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => Real.sqrt (Pr fun ee :
          ((Fin (L.V.k x) → Coin) × L.SlotEC x) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
            L.IConflict H P x p.1 Pρ p.2 ee)) := Ex_mul_left _ _
    _ ≤ L.nG x * Real.sqrt ((BconfTwo H L Ω P x).Pr H) :=
        mul_le_mul_of_nonneg_left ((Ex_sqrt_le _ fun _ => Pr_nonneg _).trans
          (Real.sqrt_le_sqrt (pr_iconflict_le hbind hS hs Pρ P))) (Nat.cast_nonneg _)

/-! ## The deterministic half -/

set_option synthInstance.maxSize 1024 in
/-- **The deterministic half, at a boundary trace**: at an accepted run whose sample holds, at slot `i`, a proof block
`u` incorrect at the boundary trace `tr` and its replay block's committed interior (`wrongAllAt`), the session of `i`
is not extracted at its first slot `i₀`'s coins, or the run's extractions disagree with themselves, or open a gate off
`tr` or off a replay block's committed interior. Otherwise the boundary values that are `tr` on the program's gates and
the kept openings elsewhere, and the interior values that are `u`'s replay block's committed interior on its gates and
the kept openings elsewhere, agree with every opening the session's witness reveals at `u` that passes, so they make
`u` correct (`Sessions.opens`); and `tr` and the committed interior agree with them where `u`'s correctness reads
(`correct_refIn_congr`), so `u` is correct at them, which `u ∈ wrongAllAt` denies. The boundary trace is the reader's:
the committed trace at the run's tape (`FregTwo_eq`), or one read at the statement. -/
theorem regTwo_split {H : Msg → Msg} {L : CompiledTwo Coin A} (hS : L.Sessions H) {x : Msg} {s : AuditStatement}
    {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s) (hPt : L.partition s = some Pt)
    (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) (hacc : (run H P L.V x ω r).out = .acc) (i u : Fin (L.nPU x))
    (hiu : (L.checked x r)[i.val]? = some u.val) (tr : ℕ → Option Bool)
    (hF : u.val ∈ wrongAllAt s Pt d.part tr (L.iregVal H P x ω r)) (i₀ u₀ : Fin (L.nPU x))
    (hlead : L.leadOf x (L.checked x r) i = some i₀) (hu₀ : (L.checked x r)[i₀.val]? = some u₀.val) :
    ¬ L.Found H x (run H P L.V x ω) u₀ r (c i₀ u₀) ∨ L.InConf H P x s ω r c ∨
      (∃ g < L.nG x, ∃ b, L.opensAt H P x ω r c g = some b ∧ tr g ≠ some b) ∨
      L.OffIreg H P x ω d.part r c := by
  have hu := List.mem_of_getElem? hiu
  by_cases hfd : L.Found H x (run H P L.V x ω) u₀ r (c i₀ u₀)
  swap
  · exact Or.inl hfd
  by_cases hconf : L.InConf H P x s ω r c
  · exact Or.inr (Or.inl hconf)
  by_cases hoff : ∃ g < L.nG x, ∃ b, L.opensAt H P x ω r c g = some b ∧ tr g ≠ some b
  · exact Or.inr (Or.inr (Or.inl hoff))
  by_cases hioff : L.OffIreg H P x ω d.part r c
  · exact Or.inr (Or.inr (Or.inr hioff))
  exfalso
  have hagree : ∀ q ∈ L.kept H P x s ω r c, ∀ q' ∈ L.kept H P x s ω r c, q.1 = q'.1 → q.2.pos = q'.2.pos →
      q.2.val = q'.2.val :=
    fun q hq q' hq' h1 h2 => Classical.byContradiction fun hne => hconf ⟨q, hq, q', hq', h1, h2, hne⟩
  have hreg : ∀ g < L.nG x, ∀ b, L.opensAt H P x ω r c g = some b → tr g = some b :=
    fun g hg b hb => Classical.byContradiction fun hne => hoff ⟨g, hg, b, hb, hne⟩
  have hireg : ∀ g < L.nG x, ∀ ρ, gateBlock d.part g = some ρ → ∀ b, L.iopensAt H P x ω r c ρ g = some b →
      L.iregVal H P x ω r ρ g = b :=
    fun g hg ρ hρ b hb => Classical.byContradiction fun hne => hioff ⟨g, hg, ρ, hρ, b, hb, hne⟩
  have hnG : L.nG x = s.prog.gates.length := by simp only [Compiled.nG, hs]
  obtain ⟨xS, vS, hsess, hsh, w, hw, hR⟩ := hfd
  obtain ⟨hk, hc, hfirst⟩ := Compiled.of_leadOf hlead
  have hsu : L.session x u (run H P L.V x ω r).seen = some (xS, vS) := by
    rw [← hS.shared x Ω P ω r _ _ _ _ hacc hu₀ hiu hk]
    exact hsess
  have hub : u.val ∈ L.sessBlocks x (L.checked x r) i₀ := by
    rw [Compiled.sessBlocks, hc]
    exact List.mem_filterMap.2 ⟨i.val, Compiled.self_mem_sessSlots hiu u.isLt, hiu⟩
  obtain ⟨b, ρ, rb, hb, hρ, hrb, hcor⟩ := hS.opens x s Pt d Ω P ω r u xS vS w hs hPt hd hin hacc hu hsu hR
  simp only [wrongAllAt, mem_biUnion, mem_product, mem_range] at hF
  obtain ⟨⟨ρ', t⟩, -, hwr⟩ := hF
  simp only [wrongInAt, mem_filter] at hwr
  obtain ⟨hcell, hnot⟩ := hwr
  have hρ' : ρ' = ρ := by
    simp only [cell, mem_filter] at hcell
    exact Option.some.inj (hcell.2.1.symm.trans hρ)
  rw [hρ'] at hnot
  obtain ⟨V, hV⟩ : ∃ V : ℕ → Option Bool, ∀ g, V g =
      if g < s.prog.gates.length then tr g else L.opensAt H P x ω r c g := ⟨_, fun _ => rfl⟩
  obtain ⟨I, hI⟩ : ∃ I : ℕ → Bool, ∀ g, I g =
      if g < s.prog.gates.length ∧ g ∈ rb then L.iregVal H P x ω r ρ g
      else ιOf (L.kept H P x s ω r c) (L.iroots x (run H P L.V x ω r).seen ρ) g := ⟨_, fun _ => rfl⟩
  have hc' : s.prog.Correct (s.prog.refIn rb V I) b := hcor V I fun q hq hok => by
    have hqk : q ∈ L.kept H P x s ω r c :=
      L.mem_runExt H x (run H P L.V x ω) s r _ c hu₀ u₀.isLt hfirst
        (L.sessExt_mem H x (run H P L.V x ω) s _ u₀ r (c i₀ u₀) hsess hsh hw hub hq hok)
    refine ⟨fun hr => ?_, fun hi => ?_⟩
    · have hv : L.opensAt H P x ω r c q.2.pos = some q.2.val := by
        rw [L.opensAt_eq H P x ω hs]
        rw [hr] at hok
        exact val_assemble hagree hqk hr hok
      rw [hV]
      by_cases hg : q.2.pos < s.prog.gates.length
      · rw [ite_eq_left hg]
        exact hreg _ (by rw [hnG]; exact hg) _ hv
      · rw [ite_eq_right hg]
        exact hv
    · rw [hI]
      by_cases hg : q.2.pos < s.prog.gates.length ∧ q.2.pos ∈ rb
      · rw [ite_eq_left hg]
        exact hireg _ (by rw [hnG]; exact hg.1) ρ (gateBlock_of_mem hin.2.2.1 hrb hg.2) _
          (iopensAt_of_mem hs hagree hqk hi hok)
      · rw [ite_eq_right hg, ← hi]
        exact ιOf_eq hagree hqk
  exact hnot ⟨b, rb, hb, hrb, (correct_refIn_congr hin.1 hin.2.1 hb rb (fun g hg => by rw [hV, ite_eq_left hg])
    fun g hg hgr => by rw [hI, ite_eq_left ⟨hg, hgr⟩]).1 hc'⟩

/-! ## The event -/

set_option synthInstance.maxSize 1024 in
/-- **The event at a tape, at a boundary trace**: an accepted run that samples a proof block incorrect at the boundary
trace `tr` and its replay block's committed interior has the first such block's session unfound at its first slot (at
`leadSel (selOf …)`), or disagrees with itself, or is off `tr` or off the committed interior (`regTwo_split`). -/
theorem eventTwo_split {H : Msg → Msg} {L : CompiledTwo Coin A} (hS : L.Sessions H) {x : Msg} {s : AuditStatement}
    {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s) (hPt : L.partition s = some Pt)
    (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} (P : Prover Ω) (ω : Ω) (tr : ℕ → Option Bool) :
    Pr (fun r : Fin (L.V.k x) → Coin => (run H P L.V x ω r).out = .acc ∧
      ∃ u ∈ L.checked x r, u ∈ wrongAllAt s Pt d.part tr (L.iregVal H P x ω r)) ≤
      Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => (run H P L.V x ω rc.1).out = .acc ∧
        ∃ i, L.leadSel x (L.selOf x (wrongAllAt s Pt d.part tr (L.iregVal H P x ω rc.1)))
            (L.checked x rc.1) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
            ¬ L.Found H x (run H P L.V x ω) u rc.1 (rc.2 i u)) +
      (Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2) +
        (Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
            ∃ g < L.nG x, ∃ b, L.opensAt H P x ω rc.1 rc.2 g = some b ∧ tr g ≠ some b) +
          Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffIreg H P x ω d.part rc.1 rc.2))) := by
  have hIO := (Sampling.Pr_union_le
    (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2)
    (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      (∃ g < L.nG x, ∃ b, L.opensAt H P x ω rc.1 rc.2 g = some b ∧ tr g ≠ some b) ∨
        L.OffIreg H P x ω d.part rc.1 rc.2)).trans
      (add_le_add le_rfl (Sampling.Pr_union_le
        (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
          ∃ g < L.nG x, ∃ b, L.opensAt H P x ω rc.1 rc.2 g = some b ∧ tr g ≠ some b)
        (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffIreg H P x ω d.part rc.1 rc.2)))
  refine (Pr_fst (β := L.SlotEC x) (fun r : Fin (L.V.k x) → Coin => (run H P L.V x ω r).out = .acc ∧
    ∃ u ∈ L.checked x r, u ∈ wrongAllAt s Pt d.part tr (L.iregVal H P x ω r))).symm.trans_le
      (Sampling.Pr_or_le (fun rc hrc => ?_) le_rfl hIO)
  obtain ⟨hacc, v, hvl, hvF⟩ := hrc
  obtain ⟨i, hi, u, hiu, huF⟩ := L.selOf_some x (hS.slots x rc.1)
    (fun w hw => by rw [Compiled.nPU_eq hs hPt]; exact mem_range.1 (wrongAllAt_subset s Pt d.part _ _ hw)) hvl hvF
  obtain ⟨i₀, u₀, hlead, hu₀⟩ := Compiled.leadOf_some hiu u.isLt
  rcases regTwo_split hS hs hPt hd hin P ω rc.1 rc.2 hacc i u hiu tr huF i₀ u₀ hlead hu₀ with h | h | h | h
  · exact Or.inl ⟨hacc, i₀, by rw [Compiled.leadSel, hi, Option.bind_some, hlead], u₀, hu₀, h⟩
  · exact Or.inr (Or.inl h)
  · exact Or.inr (Or.inr (Or.inl h))
  · exact Or.inr (Or.inr (Or.inr h))

/-- **The primary conjunct of `KSRegTwo`, from the sessions' bound**: at each tape, an accepted run that samples a
proof block of `FregTwo` has the first such block's session unfound at its first slot (the sessions' bound at
`leadSel (selOf …)`, which reads the run's coins off the second stage only), or its extractions disagree with
themselves (`B_conf`), open a gate off the committed boundary trace (the rewinding lemma at the tape), or open a gate
off a replay block's committed interior (the rewinding lemma at the tape and the first stage, and `B_conf,2`):
`eventTwo_split` at the committed boundary trace. -/
theorem eventTwo_le {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H) (hbind : L.commit.Binds H)
    (hS : L.Sessions H) :
    ∀ x, L.Scope x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwo H P x p.1 p.2) ≤ L.εregTwo H x P := by
  intro x hsc Ω _ _ P
  obtain ⟨s, Pt, d, hs, hPt, hd, hin⟩ := hsc
  have hSK : L.toCompiled.SessionKS H := Compiled.sessionKS_of hA ⟨hS.fixed, hS.runs⟩
  obtain ⟨β, hβ0, hβ, hκ⟩ := hSK x Ω P
  have hT : Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      (run H P L.V x ω rc.1).out = .acc ∧
        ∃ i, L.leadSel x (L.selOf x (wrongAllAt s Pt d.part (L.regVal H P x ω) (L.iregVal H P x ω rc.1)))
            (L.checked x rc.1) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
            ¬ L.Found H x (run H P L.V x ω) u rc.1 (rc.2 i u)) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) :=
    Compiled.pr_unfound_le_at H L.toCompiled P β hβ0 hβ
      (fun ω r => L.leadSel x (L.selOf x (wrongAllAt s Pt d.part (L.regVal H P x ω) (L.iregVal H P x ω r))))
      (fun ω r e => by simp only [iregVal_forkOn]) fun ω r sl => hκ ω r sl
  have hI := Compiled.pr_inConf_le hbind hs P
  have hO : Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      ∃ g < L.nG x, ∃ b, L.opensAt H P x ω rc.1 rc.2 g = some b ∧ L.regVal H P x ω g ≠ some b) ≤
      L.nG x * Real.sqrt ((Compiled.Bconf H L.toCompiled Ω P x).Pr H) := Compiled.ex_offReg_le hbind hs P
  have hIC := ex_offIreg_le hbind hS hs d.part P
  simp only [FregTwo_eq H L P hs hPt hd]
  refine (Sampling.Pr_prod_eq_Ex _).trans_le
    ((Ex_mono fun ω => eventTwo_split hS hs hPt hd hin P ω (L.regVal H P x ω)).trans ?_)
  rw [Ex_add, Ex_add, Ex_add]
  unfold εregTwo Compiled.εreg
  linarith

end CompiledTwo

end Claims.SamplingGeneric
