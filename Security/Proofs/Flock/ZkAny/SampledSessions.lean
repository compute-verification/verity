module

public import Proofs.Flock.ZkAny.SampledAudit
public import Proofs.Flock.ZkAny.Stream
public import Proofs.Flock.ZkAny.RelayOut

/-!
# A sampled audit's view at any values, its sampled sessions one protocol's, in sequence

`Flock.Guarantees.SampledAuditAny` takes the sampled sessions' release simulated per outcome. Here that release is `ℓ`
sessions of one protocol that meets `Claims.ZKOut`, one after another (`sampledAudit_zkOut`). Session `j`'s statement,
its verifier `V*` and `V*`'s tape are functions of what the auditor has seen: the admitted record, the sample and the
earlier sessions' messages. Its prover's parameters (a worker and its values, say) and its witness are functions of
anything: the registration's randomness, the sample, the earlier sessions' randomness and their messages (`Sessions`).
Each session is `ZKOut` read per outcome (`perOutcome_of_zkOut`) at its own randomness, its simulator's coins read
through a bijection `eC` with a type of the session's index alone, its error at most `εS j`; the `ℓ` sessions are
simulated in turn (`PerOutcome.stream`) with the errors added, and `SampledAuditAny` gives the view.

* `recOf`: the record a stack hands the loop.
* `Sessions`: the sampled sessions' statements, verifiers, prover parameters and witnesses, at what each releases.
* `Sessions.rel`, `Sessions.outc`, `Sessions.sim`: session `j`'s released messages, its outcome and its simulator.
* `sampledAudit_zkOut`: the view at `ℓ` sessions of a protocol that meets `ZKOut`.
-/

@[expose] public section

namespace FlockSoundness.ZK

open Claims Verity Verity.Game Verity.ValueSalts Verity.Assumptions
open scoped ENNReal

/-- **The record a stack hands the loop** as registration `n`, at the key and sessions `p`. -/
def recOf {K S Ys V O T R : Type} (sv : Service K S Ys V O T R) (n : ℕ) {m : ℕ} (v : V) (st : Stack K V S O T R m)
    (p : K × (Fin m → S)) : RegistrationRecord S O T R :=
  sv.handed p.1 n v p.2 (st p.1 v (sv.views p.1 v p.2))

/-- **An audit's sampled sessions of one protocol**, oldest first, each releasing a `B`. Session `j` at the record `ρ`
the loop admitted, the sample `χ` and the earlier sessions' releases `b`: its statement `x`, its verifier `Vs` and the
verifier's tape `θ`. At the registration's randomness `p`, the sample, the earlier sessions' randomness `π` and their
releases: its prover's parameters `i` and its witness `w`. -/
structure Sessions (Rec Xs P ι W A Θ B : Type) where
  x : (j : ℕ) → Rec → Xs → Tup (fun _ => B) j → Msg
  Vs : (j : ℕ) → Rec → Xs → Tup (fun _ => B) j → VStar Θ
  θ : (j : ℕ) → Rec → Xs → Tup (fun _ => B) j → Θ
  i : (j : ℕ) → P → Xs → Tup (fun _ => A) j → Tup (fun _ => B) j → ι
  w : (j : ℕ) → P → Xs → Tup (fun _ => A) j → Tup (fun _ => B) j → W

namespace Sessions

variable {Rec Xs P ι W A Θ St : Type} (Q : Sessions Rec Xs P ι W A Θ (List (Msg × Out)))

/-- **Session `j`'s messages** at its own randomness `ω`: the prover `Pv` at the session's parameters and witness,
against its verifier, at its statement, the record being `ρ p`. -/
def rel (H : Msg → Msg) (Pv : ι → W → Msg → Prover A) (ρ : P → Rec) (p : P) (χ : Xs) (j : ℕ)
    (π : Tup (fun _ => A) j) (b : Tup (fun _ => List (Msg × Out)) j) (ω : A) : List (Msg × Out) :=
  (runStar H (Pv (Q.i j p χ π b) (Q.w j p χ π b) (Q.x j (ρ p) χ b)) (Q.Vs j (ρ p) χ b) (Q.x j (ρ p) χ b) ω
    (Q.θ j (ρ p) χ b)).msgs

/-- **Session `j`'s outcome**, read off its messages by `out` at its statement. -/
def outc (out : Msg → List (Msg × Out) → St) (H : Msg → Msg) (Pv : ι → W → Msg → Prover A) (ρ : P → Rec) (p : P)
    (χ : Xs) (j : ℕ) (π : Tup (fun _ => A) j) (b : Tup (fun _ => List (Msg × Out)) j) (ω : A) : St :=
  out (Q.x j (ρ p) χ b) (Q.rel H Pv ρ p χ j π b ω)

/-- **Session `j`'s simulator** at the record `ρ`, the sample and the earlier simulated messages `b`, given its outcome
`o`: `Sm` given `o`, against the session's verifier, at its statement, its coins read through `eC`. -/
def sim (Sm : SimOut St) (H : Msg → Msg) {CB : ℕ → Type} (eC : ∀ j ρ χ b, CB j ≃ Sm.C (Q.x j ρ χ b)) (ρ : Rec)
    (χ : Xs) (j : ℕ) (_s : Tup (fun _ => St) j) (b : Tup (fun _ => List (Msg × Out)) j) (o : St) (c : CB j) :
    List (Msg × Out) :=
  (Sm.given o).out H (Q.Vs j ρ χ b) (Q.θ j ρ χ b) (Q.x j ρ χ b) (eC j ρ χ b c)

end Sessions

/-- **A sampled audit's view at any values, its sampled sessions `ℓ` sessions of one protocol that meets `ZKOut`, in
sequence.** At every value service, registration, stack and key `k₀` of `SampledAuditAny`, and every `ℓ` sessions `Q`
of a protocol that meets `ZKOut` at the prover `Pv i` within `ε i` wherever its parameters `i` are `ok`: where the stack
reads no key, every session's parameters are `ok`, its statement and witness are in `Rl`, its verifier runs the
protocol's `k` rounds, its error is at most `εS j` at each outcome, its simulator's coins are those of `CB j` through
`eC`, and an ideal record of `v` is within `δ` of one of `v₀`, the audit's view at the sessions' outcomes `s` is
`SampledAuditAny`'s bound, with the sessions' simulators run in turn (`Tup.sims`) and their errors added
(`Tup.errs εS ℓ s`). -/
theorem sampledAudit_zkOut {K S Ys V O T R Xs ι W St N₀ Ω₀ Θ : Type} [Fintype K] [Fintype S] [Fintype Ys] [Fintype T]
    [Fintype R] [Fintype Xs] [Fintype N₀] [Nonempty N₀] [Fintype Ω₀] [Nonempty Ω₀] {CB : ℕ → Type}
    [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)] {H : Msg → Msg} {k : Msg → ℕ}
    {Pv : ι → W → Msg → Prover (N₀ × Ω₀)} {Rl : Rel W} {out : Msg → List (Msg × Out) → St} {Sm : SimOut St}
    {ε : ι → (x : Msg) → W → (Θ' : Type) → VStar Θ' → Θ' → St → ℝ} {ok : ι → Prop}
    (hZ : ∀ i, ok i → ZKOut (fun _ => N₀) (fun _ => Ω₀) H k (Pv i) Rl out Sm (ε i))
    (sv : Service K S Ys V O T R) (n m : ℕ) (v v₀ : V) (st : Stack K V S O T R m) (k₀ : K) (ℓ : ℕ)
    (Q : Sessions (RegistrationRecord S O T R) Xs (K × (Fin m → S)) ι W (N₀ × Ω₀) Θ (List (Msg × Out)))
    (eC : ∀ j ρ χ b, CB j ≃ Sm.C (Q.x j ρ χ b)) (εS : ℕ → St → ℝ≥0∞) (δ : ℝ≥0∞) (hown : ZkServerOwnUser st)
    (hok : ∀ j p χ π b, ok (Q.i j p χ π b))
    (hR : ∀ j p χ π b, Rl (Q.x j (recOf sv n v st p) χ b) (Q.w j p χ π b))
    (hk : ∀ j ρ χ b, (Q.Vs j ρ χ b).k (Q.x j ρ χ b) = k (Q.x j ρ χ b))
    (hε : ∀ j p χ π b o, ENNReal.ofReal (ε (Q.i j p χ π b) (Q.x j (recOf sv n v st p) χ b) (Q.w j p χ π b) Θ
      (Q.Vs j (recOf sv n v st p) χ b) (Q.θ j (recOf sv n v st p) χ b) o) ≤ εS j o)
    (hδ : ∀ A, sv.ideal n m v A ≤ sv.ideal n m v₀ A + δ) (s : Tup (fun _ => St) ℓ)
    (E : RegistrationRecord S O T R → Xs → Tup (fun _ => List (Msg × Out)) ℓ → Prop) :
    prCoin (fun q : (K × (Fin m → S)) × Xs × Tup (fun _ => N₀ × Ω₀) ℓ =>
        sv.admits q.1.1 (recOf sv n v st q.1) ∧
          Tup.outs (Q.outc out H Pv (recOf sv n v st) q.1 q.2.1) (Q.rel H Pv (recOf sv n v st) q.1 q.2.1) ℓ q.2.2 = s ∧
            E (recOf sv n v st q.1) q.2.1 (Tup.rels (Q.rel H Pv (recOf sv n v st) q.1 q.2.1) ℓ q.2.2)) ≤
      m * (avg (fun q : Xs × Tup CB ℓ =>
          sv.ideal n m v₀ (fun ρ => E ρ q.1 (Tup.sims (Q.sim Sm H eC ρ q.1) ℓ s q.2)) +
            sv.prfAdvantage n m v (fun ρ => E ρ q.1 (Tup.sims (Q.sim Sm H eC ρ q.1) ℓ s q.2))) + δ) +
        sv.forgeAdvantage n m v (st k₀ v) + Tup.errs εS ℓ s := by
  have hS : ∀ p χ, PerOutcome (Tup.outs (Q.outc out H Pv (recOf sv n v st) p χ) (Q.rel H Pv (recOf sv n v st) p χ) ℓ)
      (Tup.rels (Q.rel H Pv (recOf sv n v st) p χ) ℓ) (Tup.sims (Q.sim Sm H eC (recOf sv n v st p) χ) ℓ)
      (Tup.errs εS ℓ) := by
    intro p χ
    refine PerOutcome.stream (A := fun _ => N₀ × Ω₀) (C := CB) (Q.outc out H Pv (recOf sv n v st) p χ)
      (Q.rel H Pv (recOf sv n v st) p χ) (Q.sim Sm H eC (recOf sv n v st p) χ) εS ℓ fun j _ π => ?_
    have hP := perOutcome_of_zkOut (hZ _ (hok j p χ π (Tup.rels (Q.rel H Pv (recOf sv n v st) p χ) j π)))
      (hR j p χ π (Tup.rels (Q.rel H Pv (recOf sv n v st) p χ) j π))
      (Q.Vs j (recOf sv n v st p) χ (Tup.rels (Q.rel H Pv (recOf sv n v st) p χ) j π))
      (Q.θ j (recOf sv n v st p) χ (Tup.rels (Q.rel H Pv (recOf sv n v st) p χ) j π))
      (hk j (recOf sv n v st p) χ (Tup.rels (Q.rel H Pv (recOf sv n v st) p χ) j π))
    exact (hP.comap_sim (eC j (recOf sv n v st p) χ (Tup.rels (Q.rel H Pv (recOf sv n v st) p χ) j π))).mono
      (hε j p χ π (Tup.rels (Q.rel H Pv (recOf sv n v st) p χ) j π))
  exact Flock.SecurityProofs.SampledAuditAny sv n m v v₀ st k₀
    (fun p χ => Tup.outs (Q.outc out H Pv (recOf sv n v st) p χ) (Q.rel H Pv (recOf sv n v st) p χ) ℓ)
    (fun p χ => Tup.rels (Q.rel H Pv (recOf sv n v st) p χ) ℓ) (fun ρ χ => Tup.sims (Q.sim Sm H eC ρ χ) ℓ)
    (Tup.errs εS ℓ) δ hown hS hδ s E

end FlockSoundness.ZK
