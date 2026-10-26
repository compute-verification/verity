import Properties.Verity.Game
import Proofs.Core.Game.Prob
import Proofs.Flock.Bridge.Opened
import Proofs.Flock.Bridge.Project
import Proofs.Flock.Soundness.Game.Prob

/-!
# Sampled proofs' target from C-Flock's session

`Verity.Vocab.SampledProofsSound` (`Properties.Verity.Game`) bounds the probability that sampled proofs' verdict
accepts while some transcript the commitment opens to has more than an `ε` share of the circuit file's units unbound,
with no break. Here it follows from C-Flock's guarantee on its live session (`SampledKSLivePub`'s form,
`SampledSession.KS`), at a vocabulary whose sampled fields are the session's (`SampledSession.vocab`):

* the verdict reads only a registered file, and a file other than the registered one with its digest is a break, so
  the event is empty unless the file is the registered one (`SampledSession.file`);
* the claim and the collision are functions of `θ`, the file and `c` alone. Outside both, every witness fails the
  session's sampling relation (`card_badIn_lt_of_relS`, through `Opens` read as the session's openings), so the event
  lies inside "the sampled component accepts with no forged release";
* that acceptance is no likelier than the session's at some strategy of it (`SampledSession.Couples`: the device's
  releases are the session's coins, ONE-4, and v2's acceptance is the session's, ONE-3p), where it is at most the
  session's error.

`sampledProofsSound` takes `Couples`, `KS` and the reading of `Opens` (ONE-7's `place`) as hypotheses.
-/

namespace Verity

open Accounting Flock.Bridge Claims.Sampling
open scoped ENNReal

/-- **Sampled proofs' session**: the registered circuit file at each parameters `θ`; each file's game circuit, its
partition into units and its constants; at `θ`, the sampling statement's descriptor, query, draw size `kd`, and `K₀`;
at `θ` and the commitment `c`, the roots, the commitment the openings pass under, and C-Flock's live session (`G`, with
its outcomes `Out`, its acceptance `Acc` and the witness `wit` its extractor reads at each strategy and outcome), whose
error is `δ θ c`. -/
structure SampledSession where
  file : ByteArray → ByteArray
  C : ByteArray → FlockSoundness.Audit.Circuit
  n : ByteArray → ℕ
  P : (F : ByteArray) → FlockSoundness.Audit.Partition (C F) (n F)
  κ : (F : ByteArray) → Fin (C F).N → Option Bool
  kd : ByteArray → ℕ
  K₀ : ByteArray → ℕ
  desc : ByteArray → Claims.Msg
  q : ByteArray → Query
  roots : ByteArray → ByteArray → List Claims.Msg
  H : Claims.Msg → Claims.Msg
  Cm : ByteArray → ByteArray → Commitment Bool
  Out : ByteArray → ByteArray → Type
  G : (θ c : ByteArray) → FlockSoundness.Game (Out θ c)
  Acc : (θ c : ByteArray) → Out θ c → Prop
  wit : (θ c : ByteArray) → FlockSoundness.Game.Strategy (G θ c) → Out θ c → Wit
  δ : ByteArray → ByteArray → ℝ

namespace SampledSession

variable (S : SampledSession)

/-- The session's sampling statement at `θ` and `c`, over the registered file's circuit. -/
abbrev stmt (θ c : ByteArray) : Claims.Sampling.Stmt :=
  stmtOf (S.P (S.file θ)) (S.κ (S.file θ)) (S.kd θ) (S.desc θ) (S.q θ) (S.roots θ c)

/-- The sampling relation's `δ` at `θ`: `C(n − K₀, kd)/C(n, kd)`, `n` the registered file's units. -/
noncomputable abbrev δS (θ : ByteArray) : ℝ :=
  ((S.n (S.file θ) - S.K₀ θ).choose (S.kd θ) : ℝ) / ((S.n (S.file θ)).choose (S.kd θ) : ℝ)

/-- **C-Flock's guarantee on the session** (`SampledKSLivePub`'s conclusion): at every strategy, the session accepts
with its witness outside the sampling relation with probability at most `δ θ c`. -/
def KS : Prop :=
  ∀ (θ c : ByteArray) (τ : FlockSoundness.Game.Strategy (S.G θ c)),
    FlockSoundness.Game.prob
      (fun x => S.Acc θ c x ∧ ¬ RelS S.H (S.Cm θ c) (S.δS θ) (S.stmt θ c) (partOf (S.P (S.file θ))) (S.wit θ c τ x))
      (S.G θ c) τ ≤ ENNReal.ofReal (S.δ θ c)

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α]

/-- **The vocabulary whose sampled fields are the session's**: each file's circuit, units and partition, and the
registered file (ONE-7), with the other subprotocols' names and the published values as given. -/
noncomputable def vocab (place : (θ F c : ByteArray) → Fin (S.C F).N → Option Place)
    (programDigest : ByteArray → Option Hash.Digest) (rounds : ByteArray → List ℕ)
    (coinsRead : Audit → Sent → List Release)
    (world : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c) → Outcome → World d α)
    (certOf : ByteArray → Certificates) (priceOf : ByteArray → Price) (delta : ByteArray → Audit → ℝ≥0∞) :
    Vocab d α where
  circuitOf F := Verity.Circuit.ofAudit (S.C F)
  units := S.n
  partsOf F := Verity.Partition.ofAudit (S.P F)
  place := place
  programDigest := programDigest
  registered := S.file
  rounds := rounds
  coinsRead := coinsRead
  world := world
  certOf := certOf
  priceOf := priceOf
  delta := delta

/-- **The sampled component's acceptance couples to the session's**: at every `θ` and `c`, at the registered file, and
every strategy of the period's game, some strategy of the session accepts at least as often as sampled proofs'
component accepts on the device's releases with no forged release (ONE-4: the releases are the session's coins; ONE-3p:
v2's acceptance is the session's). -/
def Couples (V : Vocab d α) : Prop :=
  ∀ (θ c : ByteArray) (σ : Game.Strategy (V.game θ (S.file θ) c)), ∃ τ : FlockSoundness.Game.Strategy (S.G θ c),
    Game.prob (fun o => V.Verdict .sampled o ∧ ¬ V.Forged o) (V.game θ (S.file θ) c) σ ≤
      FlockSoundness.Game.prob (S.Acc θ c) (S.G θ c) τ

end SampledSession

namespace Vocab

open SampledSession

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α]

/-- An event no outcome after `θ`, `F` and `c` meets has probability `0`. -/
theorem prob_game_le_zero (V : Vocab d α) (θ F c : ByteArray) (E : Outcome → Prop)
    (hE : ∀ log e, ¬ E ⟨θ, F, c, log, e⟩) (σ : Game.Strategy (V.game θ F c)) : Game.prob E (V.game θ F c) σ ≤ 0 :=
  prob_bind_le_of_le (releases (V.rounds θ)) (fun log => Game.send ByteArray fun e => Game.ret ⟨θ, F, c, log, e⟩)
    (fun log t => by rw [Game.prob_send]; simp [Game.prob, hE]) σ

/-- **Sampled proofs' target from C-Flock's session**: at the session's vocabulary, where the parameters' digest is the
registered file's, `Opens` reads the session's openings (a registered constant as its value), the session couples to
the sampled component's acceptance, C-Flock's guarantee holds on the session, fewer than `K₀` units is at most an `ε`
share, and the session's error is at most the published one, `SampledProofsSound` holds. -/
theorem sampledProofsSound (S : SampledSession) (place : (θ F c : ByteArray) → Fin (S.C F).N → Option Place)
    (programDigest : ByteArray → Option Hash.Digest) (rounds : ByteArray → List ℕ)
    (coinsRead : Audit → Sent → List Release)
    (world : (θ F c : ByteArray) → Game.Strategy (play (rounds θ) θ F c) → Outcome → World d α)
    (certOf : ByteArray → Certificates) (priceOf : ByteArray → Price) (delta : ByteArray → Audit → ℝ≥0∞)
    (hreg : ∀ θ h, programDigest θ = some h → sha512 (S.file θ).data.toList = h)
    (hOpens : ∀ (θ c : ByteArray) (g : Fin (S.C (S.file θ)).N) (v : BitVec 1),
      (S.vocab place programDigest rounds coinsRead world certOf priceOf delta).Opens θ (S.file θ) c g v ↔
        OpensAt (S.C (S.file θ)) (S.κ (S.file θ)) S.H (S.Cm θ c) (S.roots θ c) g (v.getLsbD 0))
    (hCouple : S.Couples (S.vocab place programDigest rounds coinsRead world certOf priceOf delta)) (hKS : S.KS)
    (hε : ∀ θ, (S.K₀ θ : ℝ) - 1 ≤ (certOf θ).ε * S.n (S.file θ))
    (hδ : ∀ θ c, ENNReal.ofReal (S.δ θ c) ≤ delta θ .sampled) :
    (S.vocab place programDigest rounds coinsRead world certOf priceOf delta).SampledProofsSound := by
  classical
  intro θ F c σ
  by_cases hReg : (S.vocab place programDigest rounds coinsRead world certOf priceOf delta).Registered θ F
  swap
  · exact (prob_game_le_zero _ θ F c _ (fun _ _ h => hReg h.1.1) σ).trans (zero_le _)
  by_cases hFC : (S.vocab place programDigest rounds coinsRead world certOf priceOf delta).FileCollides θ F
  · exact (prob_game_le_zero _ θ F c _ (fun _ _ h => h.2.2 (Or.inr (Or.inr hFC))) σ).trans (zero_le _)
  have hF : F = S.file θ := by
    by_contra hne
    exact hFC ⟨hne, (hreg θ _ hReg).symm⟩
  subst hF
  by_cases hH : (S.vocab place programDigest rounds coinsRead world certOf priceOf delta).CommitHolds θ (S.file θ) c ∨
      (S.vocab place programDigest rounds coinsRead world certOf priceOf delta).Collides θ (S.file θ) c
  · refine (Game.prob_mono (q := fun _ => False) ?_ _ _).trans (by rw [Game.prob_false]; exact zero_le _)
    rintro o ⟨-, hn, hb⟩
    rcases hH with h | h
    · exact hn h
    · exact hb (Or.inr (Or.inl h))
  obtain ⟨hnH, hnC⟩ := not_or.1 hH
  have hU : OpensUnique (S.C (S.file θ)) (S.κ (S.file θ)) S.H (S.Cm θ c) (S.roots θ c) := by
    intro g v v' h h'
    by_contra hne
    refine hnC ⟨g, BitVec.ofBool v, BitVec.ofBool v', fun he => hne ?_, (hOpens θ c g _).2 (by simpa using h),
      (hOpens θ c g _).2 (by simpa using h')⟩
    simpa using congrArg (fun x : BitVec 1 => x.getLsbD 0) he
  unfold Vocab.CommitHolds at hnH
  obtain ⟨X, hX⟩ := not_forall.1 hnH
  obtain ⟨hXo, hgt⟩ := not_imp.1 hX
  have hT : OpensTo (S.P (S.file θ)) (S.κ (S.file θ)) S.H (S.Cm θ c) (S.roots θ c) X := by
    intro g _ v hv
    rw [hXo g (BitVec.ofBool v) ((hOpens θ c g _).2 (by simpa using hv))]
    simpa using hv
  have hcard :
      ((S.vocab place programDigest rounds coinsRead world certOf priceOf delta).unbound θ (S.file θ) c X).card =
        (badIn (S.P (S.file θ)) (S.κ (S.file θ)) S.H (S.Cm θ c) (S.roots θ c) X).card := by
    congr 1
    ext u
    simp only [Vocab.unbound, badIn, Finset.mem_filter, Finset.mem_univ, true_and]
    refine or_congr Iff.rfl ⟨fun ⟨g, hg, hn⟩ => ⟨g, hg, fun h => hn ((hOpens θ c g _).2 h)⟩,
      fun ⟨g, hg, hn⟩ => ⟨g, hg, fun h => hn ((hOpens θ c g _).1 h)⟩⟩
  have hnR : ∀ w, ¬ RelS S.H (S.Cm θ c) (S.δS θ) (S.stmt θ c) (partOf (S.P (S.file θ))) w := by
    intro w hR
    have hlt := card_badIn_lt_of_relS (S.P (S.file θ)) (S.κ (S.file θ)) (S.kd θ) (S.K₀ θ) (S.desc θ) (S.q θ)
      (S.roots θ c) S.H (S.Cm θ c) w hU hR hT
    apply hgt
    rw [hcard]
    have h1 : ((badIn (S.P (S.file θ)) (S.κ (S.file θ)) S.H (S.Cm θ c) (S.roots θ c) X).card : ℝ) + 1 ≤
        S.K₀ θ := by
      exact_mod_cast Nat.succ_le_of_lt hlt
    have h2 : ((badIn (S.P (S.file θ)) (S.κ (S.file θ)) S.H (S.Cm θ c) (S.roots θ c) X).card : ℝ) ≤
        S.K₀ θ - 1 := by
      linarith
    exact h2.trans (hε θ)
  obtain ⟨τ, hτ⟩ := hCouple θ c σ
  calc _ ≤ Game.prob (fun o => (S.vocab place programDigest rounds coinsRead world certOf priceOf delta).Verdict
          .sampled o ∧ ¬ (S.vocab place programDigest rounds coinsRead world certOf priceOf delta).Forged o)
          ((S.vocab place programDigest rounds coinsRead world certOf priceOf delta).game θ (S.file θ) c) σ :=
        Game.prob_mono (fun o ⟨hv, -, hb⟩ => ⟨hv, fun hf => hb (Or.inl hf)⟩) _ _
    _ ≤ FlockSoundness.Game.prob (S.Acc θ c) (S.G θ c) τ := hτ
    _ ≤ FlockSoundness.Game.prob
          (fun x => S.Acc θ c x ∧
            ¬ RelS S.H (S.Cm θ c) (S.δS θ) (S.stmt θ c) (partOf (S.P (S.file θ))) (S.wit θ c τ x))
          (S.G θ c) τ :=
        FlockSoundness.Game.prob_mono (fun x hx => ⟨hx, hnR _⟩) _ _
    _ ≤ ENNReal.ofReal (S.δ θ c) := hKS θ c τ
    _ ≤ delta θ .sampled := hδ θ c

end Vocab

end Verity
