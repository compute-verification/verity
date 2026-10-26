import Definitions.Claims
import Definitions.Pouw.Dither

/-!
# Dither's work-soundness: its named premises (trusted)

The two premises of `Models.Pouw.Guarantees.Dither.DitherWorkSound` besides the device model (G1, G2, P1, P2) and
the slots of Theorem O and ABSORB, which are the model's (`Definitions.Pouw.Dither`):

* `ServiceSoundDither`: the proof service is sound up to its proof layer, adversary by adversary, in the form of
  `ServiceSoundStagedPerProver` (C-Flock's binding and zero knowledge, SHA-512 collision resistance only through each
  adversary's own finders, live coins from the auditor's challenger), on a period cut into rounds;
* `ImportWithinCertifier`: an accepted period imported at most `K` bits, the network certifier's count.

Neither says anything about which algorithms a prover runs, nor what an accepted run's rounds cost: that is
ABSORB's slot (`Pouw.Dither.Absorb`), whose standing is Daniel's open decision.
-/

namespace Pouw.Assumptions

open Pouw.RelativeSecurity Pouw.Dither

/-- `pouw-dither-service` (proposed; `pouw-rs-staged-service`'s form per prover, on a period of rounds). **The proof
service is sound up to its proof layer, each break bounded through the adversary's own collision finders.** The
service's seed is the period's live coins (`e`): `R` rounds,
each a first coin (Dither's dither `(R, U)`) and a check (a cell and block), drawn by the auditor's challenger after
the round's commitments (S5, live coins on `trusted-nebius-host`). For every adversary in `G` that commits first and
every value `r` of its own coins:
- round `w`'s binding break has probability at most `δb + βb · Pr[fb finds a collision of sha]`, its zero-knowledge
  break `δz + βz · Pr[fz …]` (`ServiceSoundStagedPerProver`'s terms and finders; F1 and F2, C-Flock's binding and zero
  knowledge; N1, SHA-512 commitments);
- every run the verifier accepts is accepted by the play's game, every round's check passing on the transcript its
  round program committed, or breaks binding or zero knowledge in some round.

It assumes nothing of SHA-512 (`ecr/sha-512` enters only the floor's bound, `breakBound`) and nothing of the
adversary's algorithms: a round program is any program of `G`, fixed before its round's coins. What the rounds cost
beside the run is not here: ABSORB's slot (`Pouw.Dither.Absorb`) states it.
A service whose fold is sound only under an empirical premise (KLI₂₄ for the li fold) does not meet this form. -/
def ServiceSoundDither (G : Accel) (S : Service) (P : RoundProto) {R : ℕ} (run : PeriodMap G S P R)
    (e : S.Seed ≃ Paths (P.Draw × P.Chk) R) (sha : Claims.Msg → Claims.Msg) (fb fz : Finders G.Mc S R)
    (δb βb δz βz : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : G.Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (G.Mc.eval M (σ, r)).1 = (G.Mc.eval M (σ', r)).1) → ∀ r : Ω',
      (∀ w, prob ((run Ω' M r).Bind w) ≤ δb + βb * (fb Ω' M r w).Pr sha) ∧
      (∀ w, prob ((run Ω' M r).ZK w) ≤ δz + βz * (fz Ω' M r w).Pr sha) ∧
      ∀ x, S.verify (G.Mc.eval M (e.symm x, r)).1 (e.symm x) (G.Mc.eval M (e.symm x, r)).2 = true →
        (run Ω' M r).play.accepts x ∨ ∃ w, (run Ω' M r).Bind w x ∨ (run Ω' M r).ZK w x

/-- `pouw-dither-import` (proposed; the network certifier's count, `trusted-nebius-host`). **An accepted period
imported at most `K` bits.** The network certifier sits on the period's devices' links, on a machine the prover
doesn't control, and signs the bits that enter; the period's verdict reads its signed count and refuses a period past
`K`. So every run the verifier accepts read at most `K` bits from outside the period's devices. -/
def ImportWithinCertifier (G : Accel) (S : Service) (K : ℝ) : Prop :=
  ∀ (Ω' : Type) [Fintype Ω'] (M : G.Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)) (ω : S.Seed × Ω'),
    S.verify (G.Mc.eval M ω).1 ω.1 (G.Mc.eval M ω).2 = true → G.ext M ω ≤ K

end Pouw.Assumptions
