import Proofs.Flock.Soundness.Discharge.ZkLink.Public
import Definitions.Claims.Sampling

/-!
# The claims' faulty set inside the game's wrong set

`Claims.Sampling.faulty` counts the slots of the claims' partition whose proof block no interior makes correct on the
witness's openings; `FlockSoundness.Discharge.ZkLink.wrongRegZ` counts the game's units not right on the registration at
a transcript (`EndToEnd` reads it at `XplurZC`). `faulty_card_le_wrongRegZ`: when every slot is some game unit's
(`slot`), at most as many slots are faulty as units are wrong, given two facts per game unit `u` at the slot's proof
block `b u`, each a hypothesis here:

* `Lowers`: the claims program's gates `need` of `b u` (its gates and the gates they read) map to the game circuit's
  wires (`φ`), and on values that are the circuit's evaluation at those wires `b u` is correct (`Prog.Correct`). It reads no prover: it is the claims program and the game circuit agreeing on the unit.
* `Binds`: every opening family that opens `u`'s registered commit strings on its committed wires (`ValueBindingZ.cs`)
  agrees with the witness's opened value (`Wit.val`) at each committed, non-constant gate of `need` that `b u` reads or
  commits, at a wire of `P.io u`. It is the event that the witness's openings and the registration's agree, which a
  collision breaks.

So a unit right on the registration has its slot not faulty (`not_faulty_of_rightRegZ`), at the interior that is the
circuit's evaluation. -/

namespace FlockSoundness.Discharge.ZkLink.WrongSet

open Game Model Accounting Audit ZkSession

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] {mPts : ℕ}
variable (A : Arith F K) {C : Circuit} {n : ℕ} (P : Audit.Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)
  {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : ValueBindingZ A P planZ tab decode Hc)

/-- The claims program's gates a proof block reads or computes: its gates, and the gates they read. -/
def need (Cp : Claims.Sampling.Prog) (b : List ℕ) : List ℕ :=
  b ++ b.flatMap fun g => (Cp.gates[g]?.map Claims.Sampling.BGate.reads).getD []

/-- **The claims unit lowers to the game unit**: on values that are the circuit's evaluation on `Xv`'s inputs at the
wires `φ` maps `need` to (constants aside), the proof block `b` is correct. -/
def Lowers (Cp : Claims.Sampling.Prog) (b : List ℕ) (φ : ℕ → Fin C.N) (Xv : Fin C.N → Bool) : Prop :=
  ∀ ref : ℕ → Option Bool, (∀ g ∈ need Cp b, (∀ c, Cp.gates[g]? ≠ some (.const c)) → ref g = some (C.eval Xv (φ g))) →
    Cp.Correct ref b

/-- **The witness's openings agree with the registration's** at game unit `u`: every opening family that opens `u`'s
registered commit strings on its committed wires has, at each non-constant gate of `need` that `b` reads from outside or
commits, a wire of `P.io u` whose opened bit is the witness's opened value. -/
def Binds (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.Commitment Bool) (s : Claims.Sampling.AuditStatement)
    (w : Claims.Sampling.Wit) (b : List ℕ)
    (φ : ℕ → Fin C.N) (R : Reg) (u : Fin n) : Prop :=
  ∀ op : vb.Pos → vb.Opening, (∀ g ∈ P.io u, vb.com (op (vb.pos g)) = vb.cs R (vb.pos g)) →
    ∀ g ∈ need s.prog b, (∀ c, s.prog.gates[g]? ≠ some (.const c)) →
      (g ∉ b ∨ s.prog.Boundary b g = true) →
      φ g ∈ P.io u ∧ w.val H Cm s g = some (vb.bit (vb.val (op (vb.pos (φ g)))) (φ g))

variable {A P planZ tab vb}

/-- **A unit right on the registration has its slot not faulty**, at the interior that is the circuit's evaluation. -/
theorem not_faulty_of_rightRegZ {H : Claims.Msg → Claims.Msg} {Cm : Claims.Sampling.Commitment Bool}
    {s : Claims.Sampling.AuditStatement}
    {Pc : Claims.Sampling.Partition} {w : Claims.Sampling.Wit} {R : Reg} {Xv : Fin C.N → Bool} {u : Fin n} {v : ℕ}
    {b : List ℕ} {φ : ℕ → Fin C.N}
    (hr : Pc.blocks[v]? = some b) (hlow : Lowers s.prog b φ Xv) (hbind : Binds A P planZ tab vb H Cm s w b φ R u)
    (hu : RightRegZ A P planZ tab vb R Xv u) : ¬ Claims.Sampling.Faulty H Cm s Pc w v := by
  obtain ⟨-, op, hop⟩ := hu
  intro hf
  refine hf ⟨b, hr, fun g => C.eval Xv (φ g), hlow _ fun g hg hc => ?_⟩
  unfold Claims.Sampling.Prog.refIn
  split_ifs with hin
  · rfl
  · have hB : g ∉ b ∨ s.prog.Boundary b g = true := by
      by_cases h1 : g ∈ b
      · exact Or.inr (by simpa [h1] using hin)
      · exact Or.inl h1
    obtain ⟨hio, hw⟩ := hbind op (fun g hg => (hop g hg).1) g hg hc hB
    rw [hw, (hop (φ g) hio).2]

/-- **At most as many slots are faulty as game units are wrong on the registration**: every slot is some unit's
(`slot`), and at each unit its slot's proof block lowers to it (`Lowers`) and the witness's openings agree with the
registration's (`Binds`). -/
theorem faulty_card_le_wrongRegZ (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.Commitment Bool)
    (s : Claims.Sampling.AuditStatement)
    (Pc : Claims.Sampling.Partition) (w : Claims.Sampling.Wit) (R : Reg) (Xv : Fin C.N → Bool) (slot : Fin n → ℕ)
    (b : Fin n → List ℕ) (φ : ℕ → Fin C.N)
    (hcov : ∀ v < Pc.blocks.length, ∃ u, slot u = v) (hr : ∀ u, Pc.blocks[slot u]? = some (b u))
    (hlow : ∀ u, Lowers s.prog (b u) φ Xv) (hbind : ∀ u, Binds A P planZ tab vb H Cm s w (b u) φ R u) :
    (Claims.Sampling.faulty H Cm s Pc w).card ≤ (wrongRegZ A P planZ tab vb R Xv).card := by
  classical
  calc (Claims.Sampling.faulty H Cm s Pc w).card ≤ ((wrongRegZ A P planZ tab vb R Xv).image slot).card := by
        apply Finset.card_le_card
        intro v hv
        simp only [Claims.Sampling.faulty, Finset.mem_filter, Finset.mem_range] at hv
        obtain ⟨u, rfl⟩ := hcov v hv.1
        refine Finset.mem_image.2 ⟨u, (mem_wrongRegZ A P planZ tab vb).2 fun h => ?_, rfl⟩
        exact not_faulty_of_rightRegZ (hr u) (hlow u) (hbind u) h hv.2
    _ ≤ _ := Finset.card_image_le

end FlockSoundness.Discharge.ZkLink.WrongSet
