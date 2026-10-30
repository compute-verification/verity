import Proofs.Flock.Claims.Sampling.Select.Design
import Proofs.Flock.Claims.Sampling.Select.DrawLaw

/-!
# The sampling scope check from the deployed design (`Claims.Sampling.ScopeMatches`, `Refuses`)

The readable scope (`InScope`) is a program part (a topologically ordered program, and a partition of it into proof
units) and a design part (`Design.WellFormed`). The design part holds of every statement the sampled verifier lays out
and draws on (`designWellFormed`), so the scope check need check only the program part:

* `ProgramScope`: the scope check takes a statement exactly when it parses, its query evaluates, and the program part
  holds.
* `ChecksScope`: a run the verifier accepts is of a statement the scope check takes.
* `scopeMatches_of`: `ScopeMatches` where the statement is the verifier's own (`DrawSlots`) and the scope check is the
  program part.
* `refuses_of`: `Refuses` from `ScopeMatches` where the verifier runs its scope check.
-/

open SampledProofs

namespace Claims.Sampling

open Properties.SampledProofs Properties.SampledProofs.SecurityProofs

variable {Coin Col T : Type}

/-- **The scope check is the readable scope's program part**: it takes a statement exactly when it parses, its query
evaluates to a partition, the program is topologically ordered and the partition partitions it into proof units whose
committed values fit their public size classes. The design part it need not check (`scopeMatches_of`). The program is the statement's bit
gates, its word gates' Boolean lowering, and a `Q_word` unit is the set of its word gates' bit gates. -/
def ProgramScope (D : Deploy Coin Col T) : Prop :=
  ∀ x, D.scopeCheck x = true ↔ ∃ s Pt, D.parse x = some s ∧ D.partition s = some Pt ∧
    s.prog.Topological ∧ Pt.Partitions s.prog

/-- **The verifier runs its scope check**: a run it accepts is of a statement its scope check takes. -/
def ChecksScope [Inhabited Coin] (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ x (Ω : Type) (P : Prover Ω) ω r, (run H P D.V x ω r).out = .acc → D.scopeCheck x = true

/-- **`ChecksScope` from the verifier's decision**: where the verifier answers `acc` only at statements its scope check
takes, every run it accepts is of one (`run_acc_next`). -/
theorem checksScope_of_next [Inhabited Coin] (H : Msg → Msg) (D : Deploy Coin Col T)
    (h : ∀ x as cs, ((D.V.next x as cs).eval H).1 = .acc → D.scopeCheck x = true) : ChecksScope H D := by
  intro x Ω P ω r hacc
  obtain ⟨as, cs, has⟩ := run_acc_next H P D.V x ω r hacc
  exact h x as cs has

/-- **The readable scope is the verifier's**: where its statements are its own layout and design (`DrawSlots`) and its
scope check is the program part (`ProgramScope`), the design is well formed on the partition (`designWellFormed`), so
the scope check is the whole readable scope. -/
theorem scopeMatches_of [Fintype Coin] (D : Deploy Coin Col T) (hslots : DrawSlots D) (hprog : ProgramScope D) :
    ScopeMatches D := by
  intro x
  rw [hprog x]
  constructor
  · rintro ⟨s, Pt, hx, hP, htop, hpart, -⟩
    exact ⟨s, Pt, hx, hP, htop, hpart⟩
  · rintro ⟨s, Pt, hx, hP, htop, hpart⟩
    obtain ⟨rule, d, query, table, -, l, -, -, -, hl, hshape, hds, -, -⟩ := hslots x s Pt hx hP
    exact ⟨s, _, hx, hP, htop, hpart,
      (Design.wellFormed_of_shape _ hshape).2 (designWellFormed rule d query table l s.design hl hds)⟩

/-- **The verifier refuses what is out of scope**: an accepted run's statement passed the scope check (`ChecksScope`),
which is the readable scope (`ScopeMatches`). -/
theorem refuses_of [Inhabited Coin] (H : Msg → Msg) (D : Deploy Coin Col T) (hscope : ScopeMatches D)
    (hchecks : ChecksScope H D) :
    Refuses H D :=
  fun x Ω P ω r hacc => (hscope x).2 (hchecks x Ω P ω r hacc)

end Claims.Sampling
