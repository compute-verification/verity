import Proofs.Flock.Claims.Bridge.WrongSet

/-!
# `Binds` or a collision

`WrongSet.Binds` couples two extractors' openings at a game unit `u`: the claims witness's (`Wit.val`, under the
statement's roots) and any opening family of `u`'s registered commit strings (`ValueBindingZ.cs`, the game's). Given that
every registered opening reads as a claims opening at the gate's position that passes the commitment (`Transports`), and
that the witness opens every committed, non-constant gate the unit needs (`Covers`), `Binds` holds unless two openings
that pass at one position under the statement's roots carry different values (`binds_or_disagree`). Under the
commitment's binding (`CommitmentScheme.Binds`, proved at the deployed record), the `H` calls of those two openings' checks
collide (`binds_or_logColl`): a collision among the inputs the two openings hashed, which cr/sha-512 charges per prover. -/

namespace FlockSoundness.Discharge.ZkLink.WrongSet

open Game Model Accounting Audit ZkSession

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] {mPts : ℕ}
variable (A : Arith F K) {C : Circuit} {n : ℕ} (P : Audit.Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)
  {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : ValueBindingZ A P planZ tab decode Hc)

/-- **The registered openings read as claims openings**: an opening `q` of `u`'s registered commit string at the wire
`φ g` is, as `tr g q`, an opening at gate `g`'s position, of the bit `q` carries on that wire, that passes the
commitment under the statement's roots. -/
def Transports (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool) (s : Claims.Sampling.AuditStatement)
    (R : Reg) (u : Fin n) (φ : ℕ → Fin C.N) (tr : ℕ → vb.Opening → Claims.Sampling.Opening Bool) : Prop :=
  ∀ g q, φ g ∈ P.io u → vb.com q = vb.cs R (vb.pos (φ g)) →
    (tr g q).pos = g ∧ (tr g q).val = vb.bit (vb.val q) (φ g) ∧ Cm.Opens H s.roots (tr g q) = true

/-- **The witness covers the unit**: at each committed, non-constant gate of `need` that `r` reads from outside or
commits, `φ` lands in `P.io u` and the witness opens the gate. -/
def Covers (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool) (s : Claims.Sampling.AuditStatement)
    (w : Claims.Sampling.Wit) (r : Claims.Sampling.RU) (φ : ℕ → Fin C.N) (u : Fin n) : Prop :=
  ∀ g ∈ need s.prog r, (∀ b, s.prog.gates[g]? ≠ some (.const b)) →
    (g ∉ Claims.Sampling.RU.gates r ∨ s.prog.Boundary (Claims.Sampling.RU.gates r) g = true) →
      φ g ∈ P.io u ∧ (w.val H Cm s g).isSome

variable {A P planZ tab vb}

/-- **`Binds`, or two passing openings at one position that disagree.** -/
theorem binds_or_disagree {H : Claims.Msg → Claims.Msg} {Cm : Claims.Sampling.CommitmentScheme Bool}
    {s : Claims.Sampling.AuditStatement} {w : Claims.Sampling.Wit} {r : Claims.Sampling.RU} {φ : ℕ → Fin C.N} {R : Reg}
    {u : Fin n} {tr : ℕ → vb.Opening → Claims.Sampling.Opening Bool}
    (hcov : Covers P H Cm s w r φ u) (htr : Transports A P planZ tab vb H Cm s R u φ tr) :
    Binds A P planZ tab vb H Cm s w r φ R u ∨
      ∃ o o' : Claims.Sampling.Opening Bool, o ∈ w.vals ∧ o.pos = o'.pos ∧ o.val ≠ o'.val ∧
        Cm.Opens H s.roots o = true ∧ Cm.Opens H s.roots o' = true := by
  by_cases hc : ∃ o o' : Claims.Sampling.Opening Bool, o ∈ w.vals ∧ o.pos = o'.pos ∧ o.val ≠ o'.val ∧
      Cm.Opens H s.roots o = true ∧ Cm.Opens H s.roots o' = true
  · exact Or.inr hc
  · refine Or.inl fun ops hops g hg hconst hB => ?_
    obtain ⟨hio, hsome⟩ := hcov g hg hconst hB
    refine ⟨hio, ?_⟩
    obtain ⟨htp, htv, htO⟩ := htr g (ops (vb.pos (φ g))) hio (hops (φ g) hio)
    rcases hfo : w.vals.find? (fun o => o.pos == g && Cm.Opens H s.roots o) with _ | o
    · rw [Claims.Sampling.Wit.val, hfo] at hsome
      simp at hsome
    · have hp := List.find?_some hfo
      have hm := List.mem_of_find?_eq_some hfo
      simp only [Bool.and_eq_true, beq_iff_eq] at hp
      have hval : o.val = (tr g (ops (vb.pos (φ g)))).val := by
        by_contra hne
        exact hc ⟨o, tr g (ops (vb.pos (φ g))), hm, hp.1.trans htp.symm, hne, hp.2, htO⟩
      rw [Claims.Sampling.Wit.val, hfo]
      exact congrArg some (hval.trans htv)

/-- **`Binds`, or a collision among the two openings' checks**, under the commitment's binding. -/
theorem binds_or_logColl {H : Claims.Msg → Claims.Msg} {Cm : Claims.Sampling.CommitmentScheme Bool}
    {s : Claims.Sampling.AuditStatement} {w : Claims.Sampling.Wit} {r : Claims.Sampling.RU} {φ : ℕ → Fin C.N} {R : Reg}
    {u : Fin n} {tr : ℕ → vb.Opening → Claims.Sampling.Opening Bool} (hb : Cm.Binds H)
    (hcov : Covers P H Cm s w r φ u) (htr : Transports A P planZ tab vb H Cm s R u φ tr) :
    Binds A P planZ tab vb H Cm s w r φ R u ∨
      ∃ o o' : Claims.Sampling.Opening Bool, o ∈ w.vals ∧
        Claims.Sampling.LogColl (((Cm.opens s.roots o).eval H).2 ++ ((Cm.opens s.roots o').eval H).2) := by
  rcases binds_or_disagree hcov htr with h | ⟨o, o', hm, hp, hv, h1, h2⟩
  · exact Or.inl h
  · exact Or.inr ⟨o, o', hm, hb s.roots o o' hp hv h1 h2⟩

/-- **`Binds` at every unit, or a collision**: `faulty_card_le_wrongRegZ`'s `hbind`, unless the checks of two openings
the witness and the registration pass at one position collide. -/
theorem binds_all_or_logColl {H : Claims.Msg → Claims.Msg} {Cm : Claims.Sampling.CommitmentScheme Bool}
    {s : Claims.Sampling.AuditStatement} {w : Claims.Sampling.Wit} {r : Fin n → Claims.Sampling.RU} {φ : ℕ → Fin C.N}
    {R : Reg} {tr : ℕ → vb.Opening → Claims.Sampling.Opening Bool} (hb : Cm.Binds H)
    (hcov : ∀ u, Covers P H Cm s w (r u) φ u) (htr : ∀ u, Transports A P planZ tab vb H Cm s R u φ tr) :
    (∀ u, Binds A P planZ tab vb H Cm s w (r u) φ R u) ∨
      ∃ o o' : Claims.Sampling.Opening Bool, o ∈ w.vals ∧
        Claims.Sampling.LogColl (((Cm.opens s.roots o).eval H).2 ++ ((Cm.opens s.roots o').eval H).2) := by
  by_cases hc : ∃ o o' : Claims.Sampling.Opening Bool, o ∈ w.vals ∧
      Claims.Sampling.LogColl (((Cm.opens s.roots o).eval H).2 ++ ((Cm.opens s.roots o').eval H).2)
  · exact Or.inr hc
  · exact Or.inl fun u => (binds_or_logColl hb (hcov u) (htr u)).resolve_right hc

end FlockSoundness.Discharge.ZkLink.WrongSet
