module

public import Proofs.Flock.Claims.Sampling.Select.Design
public import Proofs.Flock.Claims.Sampling.Select.SampleDesign

/-! The sampling scope check from the deployed design, which need check only the readable scope's program part. -/

@[expose] public section

open Spot

namespace Claims.Sampling

open Properties.Spot Properties.Spot.SecurityProofs

variable {Coin Col T : Type}

def ProgramScope (D : Deploy Coin Col T) : Prop :=
  ∀ x, D.scopeCheck x = true ↔ ∃ s Pt, D.parse x = some s ∧ D.partition s = some Pt ∧
    s.prog.Topological ∧ Pt.Partitions s.prog

def ChecksScope [Inhabited Coin] (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ x (Ω : Type) (P : Prover Ω) ω r, (run H P D.V x ω r).out = .acc → D.scopeCheck x = true

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

theorem refuses_of [Inhabited Coin] (H : Msg → Msg) (D : Deploy Coin Col T) (hscope : ScopeMatches D)
    (hchecks : ChecksScope H D) :
    Refuses H D :=
  fun x Ω P ω r hacc => (hscope x).2 (hchecks x Ω P ω r hacc)

end Claims.Sampling
