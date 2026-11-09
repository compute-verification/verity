module

public import Proofs.Flock.ZkAny.SampledAuditTwo
public import Proofs.Flock.ZkAny.SampledSessions

/-!
# A two-stage sampled audit's view at any values, its sampled sessions one protocol's, in sequence

`Flock.Guarantees.SampledAuditTwoAny` with its sampled sessions' release `ℓ` sessions of one protocol that meets
`Claims.ZKOut`, one after another (`sampledAuditTwo_zkOut`), as `sampledAudit_zkOut` has it for one stage. The sessions
are `Sessions` whose record is what the auditor holds before the second stage (`recTwoOf`: the admitted records and the
first-stage sample) and whose sample is the second-stage sample. So session `j`'s statement, its verifier `V*` and
`V*`'s tape are functions of the admitted records, both samples and the earlier sessions' messages; its prover's
parameters and its witness are functions of anything, the first-stage sample included.

* `recTwoOf`: the records a stream stack hands the loop at a first-stage sample, with that sample.
* `sampledAuditTwo_zkOut`: the two-stage view at `ℓ` sessions of a protocol that meets `ZKOut`.
-/

@[expose] public section

namespace FlockSoundness.ZK

open Claims Verity Verity.Game Verity.ValueSalts Verity.Assumptions
open scoped ENNReal

/-- **What the auditor holds before the second stage**: the records a stream stack hands the loop at the first-stage
sample `p.2`, its values `v p.2`, with that sample. -/
def recTwoOf {K S Ys V O T R X₁ : Type} (sv : Service K S Ys V O T R) {r : ℕ} {m : Fin r → ℕ} (v : X₁ → Fin r → V)
    (st : X₁ → StreamStack K V S O T R r m) (p : (K × ((i : Fin r) → Fin (m i) → S)) × X₁) :
    StreamRecord r S O T R × X₁ :=
  (streamRecOf sv (v p.2) (st p.2) p.1, p.2)

/-- **A two-stage sampled audit's view at any values, its sampled sessions `ℓ` sessions of one protocol that meets
`ZKOut`, in sequence.** At every value service, stream, first-stage values and stack, and key `k₀` of
`SampledAuditTwoAny`, and every `ℓ` sessions `Q` of a protocol that meets `ZKOut` at the prover `Pv i` within `ε i`
wherever its parameters `i` are `ok`: where the stack reads no key, every session's parameters are `ok`, its statement
and witness are in `Rl`, its verifier runs the protocol's `k` rounds, its error is at most `εS j` at each outcome, its
simulator's coins are those of `CB j` through `eC`, and at every first-stage sample an ideal stream of the values is
within `δ` of one of the dummy values `v₀` there, the audit's view at the sessions' outcomes `s` is
`SampledAuditTwoAny`'s bound, with the sessions' simulators run in turn (`Tup.sims`) and their errors added
(`Tup.errs εS ℓ s`). -/
theorem sampledAuditTwo_zkOut {K S Ys V O T R X₁ X₂ ι W St N₀ Ω₀ Θ : Type} [Fintype K] [Fintype S] [Fintype Ys]
    [Fintype T] [Fintype R] [Fintype X₁] [Fintype X₂] [Fintype N₀] [Nonempty N₀] [Fintype Ω₀] [Nonempty Ω₀]
    {CB : ℕ → Type} [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)] {H : Msg → Msg} {k : Msg → ℕ}
    {Pv : ι → W → Msg → Prover (N₀ × Ω₀)} {Rl : Rel W} {out : Msg → List (Msg × Out) → St} {Sm : SimOut St}
    {ε : ι → (x : Msg) → W → (Θ' : Type) → VStar Θ' → Θ' → St → ℝ} {ok : ι → Prop}
    (hZ : ∀ i, ok i → ZKOut (fun _ => N₀) (fun _ => Ω₀) H k (Pv i) Rl out Sm (ε i))
    (sv : Service K S Ys V O T R) (r : ℕ) (m : Fin r → ℕ) (b : ℕ) (v v₀ : X₁ → Fin r → V)
    (st : X₁ → StreamStack K V S O T R r m) (k₀ : K) (ℓ : ℕ)
    (Q : Sessions (StreamRecord r S O T R × X₁) X₂ ((K × ((i : Fin r) → Fin (m i) → S)) × X₁) ι W (N₀ × Ω₀) Θ
      (List (Msg × Out)))
    (eC : ∀ j ρ χ b, CB j ≃ Sm.C (Q.x j ρ χ b)) (εS : ℕ → St → ℝ≥0∞) (δ : ℝ≥0∞)
    (hown : ∀ χ₁, ZkServerOwnUser (st χ₁)) (hok : ∀ j p χ π b, ok (Q.i j p χ π b))
    (hR : ∀ j p χ π b, Rl (Q.x j (recTwoOf sv v st p) χ b) (Q.w j p χ π b))
    (hk : ∀ j ρ χ b, (Q.Vs j ρ χ b).k (Q.x j ρ χ b) = k (Q.x j ρ χ b))
    (hε : ∀ j p χ π b o, ENNReal.ofReal (ε (Q.i j p χ π b) (Q.x j (recTwoOf sv v st p) χ b) (Q.w j p χ π b) Θ
      (Q.Vs j (recTwoOf sv v st p) χ b) (Q.θ j (recTwoOf sv v st p) χ b) o) ≤ εS j o)
    (hδ : ∀ χ₁ P A, sv.streamIdeal m (v χ₁) P A ≤ sv.streamIdeal m (v₀ χ₁) P A + δ) (s : Tup (fun _ => St) ℓ)
    (E : StreamRecord r S O T R → X₁ → X₂ → Tup (fun _ => List (Msg × Out)) ℓ → Prop) :
    prCoin (fun q : (K × ((i : Fin r) → Fin (m i) → S)) × X₁ × X₂ × Tup (fun _ => N₀ × Ω₀) ℓ =>
        sv.streamAdmits q.1.1 b (streamRecOf sv (v q.2.1) (st q.2.1) q.1) ∧
          Tup.outs (Q.outc out H Pv (recTwoOf sv v st) (q.1, q.2.1) q.2.2.1)
              (Q.rel H Pv (recTwoOf sv v st) (q.1, q.2.1) q.2.2.1) ℓ q.2.2.2 = s ∧
            E (streamRecOf sv (v q.2.1) (st q.2.1) q.1) q.2.1 q.2.2.1
              (Tup.rels (Q.rel H Pv (recTwoOf sv v st) (q.1, q.2.1) q.2.2.1) ℓ q.2.2.2)) ≤
      (∑ d ∈ streamChoices r m b,
        (avg (fun q : X₁ × X₂ × Tup CB ℓ =>
            sv.streamIdeal m (v₀ q.1) (passed d)
                (fun ρs => E ρs q.1 q.2.1 (Tup.sims (Q.sim Sm H eC (ρs, q.1) q.2.1) ℓ s q.2.2)) +
              sv.streamPrfAdvantage (v q.1) d
                (fun ρs => E ρs q.1 q.2.1 (Tup.sims (Q.sim Sm H eC (ρs, q.1) q.2.1) ℓ s q.2.2))) + δ)) +
        avg (fun χ₁ => sv.streamForgeAdvantage (v χ₁) (st χ₁ k₀ (v χ₁))) + Tup.errs εS ℓ s := by
  have hS : ∀ p χ₁ χ₂, PerOutcome
      (Tup.outs (Q.outc out H Pv (recTwoOf sv v st) (p, χ₁) χ₂) (Q.rel H Pv (recTwoOf sv v st) (p, χ₁) χ₂) ℓ)
      (Tup.rels (Q.rel H Pv (recTwoOf sv v st) (p, χ₁) χ₂) ℓ)
      (Tup.sims (Q.sim Sm H eC (recTwoOf sv v st (p, χ₁)) χ₂) ℓ) (Tup.errs εS ℓ) := by
    intro p χ₁ χ₂
    refine PerOutcome.stream (A := fun _ => N₀ × Ω₀) (C := CB) (Q.outc out H Pv (recTwoOf sv v st) (p, χ₁) χ₂)
      (Q.rel H Pv (recTwoOf sv v st) (p, χ₁) χ₂) (Q.sim Sm H eC (recTwoOf sv v st (p, χ₁)) χ₂) εS ℓ
      fun j _ π => ?_
    have hP := perOutcome_of_zkOut
      (hZ _ (hok j (p, χ₁) χ₂ π (Tup.rels (Q.rel H Pv (recTwoOf sv v st) (p, χ₁) χ₂) j π)))
      (hR j (p, χ₁) χ₂ π (Tup.rels (Q.rel H Pv (recTwoOf sv v st) (p, χ₁) χ₂) j π))
      (Q.Vs j (recTwoOf sv v st (p, χ₁)) χ₂ (Tup.rels (Q.rel H Pv (recTwoOf sv v st) (p, χ₁) χ₂) j π))
      (Q.θ j (recTwoOf sv v st (p, χ₁)) χ₂ (Tup.rels (Q.rel H Pv (recTwoOf sv v st) (p, χ₁) χ₂) j π))
      (hk j (recTwoOf sv v st (p, χ₁)) χ₂ (Tup.rels (Q.rel H Pv (recTwoOf sv v st) (p, χ₁) χ₂) j π))
    exact (hP.comap_sim (eC j (recTwoOf sv v st (p, χ₁)) χ₂
      (Tup.rels (Q.rel H Pv (recTwoOf sv v st) (p, χ₁) χ₂) j π))).mono
      (hε j (p, χ₁) χ₂ π (Tup.rels (Q.rel H Pv (recTwoOf sv v st) (p, χ₁) χ₂) j π))
  exact Flock.SecurityProofs.SampledAuditTwoAny sv r m b v v₀ st k₀
    (fun p χ₁ χ₂ => Tup.outs (Q.outc out H Pv (recTwoOf sv v st) (p, χ₁) χ₂)
      (Q.rel H Pv (recTwoOf sv v st) (p, χ₁) χ₂) ℓ)
    (fun p χ₁ χ₂ => Tup.rels (Q.rel H Pv (recTwoOf sv v st) (p, χ₁) χ₂) ℓ)
    (fun ρs χ₁ χ₂ => Tup.sims (Q.sim Sm H eC (ρs, χ₁) χ₂) ℓ) (Tup.errs εS ℓ) δ hown hS hδ s E

end FlockSoundness.ZK
