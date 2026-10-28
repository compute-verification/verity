import Definitions.Pouw.RelativeSecurity

/-!
# Relative security's collision finder, built from the adversary (trusted)

One adversary class (the lean-proofs skill): a guarantee holds for every prover strategy, and its break term
is the probability that one explicit finder, built from that strategy and run in about its time, exhibits a SHA-512
collision. Collision resistance is then applied to that finder alone, outside the theorem. The floors per prover of
`Definitions.Pouw.Guarantees.RelativeSecurity` read their service premises at this finder, and conclude at its
collision probability:

* the verifier's SHA-512 inputs (`HashedService`), what a finder can read off a run;
* round `w`'s fork of two coin paths (`Paths.splice`);
* **the fork finder** (`forkFinder`, one per adversary, value of its coins and round; `forkFinders`): run the adversary
  from its commitment on a seed and on `k` continuations forked from it at round `w`, and output the first collision
  among the verifier's SHA-512 inputs in those runs. At `k = 1` it is the binding finder; at C-Flock's `k` reruns
  it is the link finder's form (two openings of one committed leaf, both checked by the verifier);
* an adversary's error at its finders' collision probabilities (`breakPr`);
* the machine of functions at no cost (`funMachine`), where the nontrivial lemma's adversary lives.

The finder's cost is the verifier's SHA-512 calls in its runs plus each run of the adversary at its worst-case time
(`Machine.time`), so `ecr/sha-512` (`Claims.ExpectedCR`), applied to it outside the floors, reads a cost no choice of
the analysis sets. A finder the analysis picks freely (`Finders`) can declare any cost and output a fixed collision,
which is why the floors no longer take one.
-/

namespace Pouw.RelativeSecurity

open Finset

/-- **A service with its verifier's SHA-512 inputs**: `hashes c σ a` lists the strings the verifier hashes when it
checks commitment `c`, seed `σ` and answer `a`, in order: the leaves and nodes of every opening it checks against a
committed root, and the hashes inside every proof it verifies. A collision finder reads a run's collisions off them. -/
structure HashedService extends Service where
  hashes : Com → Seed → Ans → List Claims.Msg

instance HashedService.instNonemptySeed (S : HashedService) : Nonempty S.Seed :=
  Fintype.card_pos_iff.mp (by rw [S.card_seed]; positivity)

/-- **Round `i`'s fork of two coin paths**: `x`'s first `i` rounds, then `y`'s. -/
def Paths.splice {Ω : Type} : {n : ℕ} → ℕ → Paths Ω n → Paths Ω n → Paths Ω n
  | 0, _, x, _ => x
  | _ + 1, 0, _, y => y
  | _ + 1, i + 1, x, y => (x.1, Paths.splice i x.2 y.2)

/-- The seeds of a fork at round `w`: `σ`, then each continuation `σ' i`, which keeps `σ`'s first `w` rounds and
draws the rest afresh. -/
def forkSeeds (S : HashedService) {Ω : Type} {R : ℕ} (e : S.Seed ≃ Paths Ω R) (w : Fin R) {k : ℕ} (σ : S.Seed)
    (σ' : Fin k → S.Seed) : List S.Seed :=
  σ :: List.ofFn fun i => e.symm (Paths.splice w.val (e σ) (e (σ' i)))

/-- The verifier's SHA-512 inputs along the fork's runs of `M` at its coins `r`, run after run. -/
def forkLog {Mc : Machine} (S : HashedService) {Ω : Type} {R : ℕ} (e : S.Seed ≃ Paths Ω R) {Ω' : Type}
    (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)) (r : Ω') (w : Fin R) {k : ℕ} (σ : S.Seed)
    (σ' : Fin k → S.Seed) : List Claims.Msg :=
  (forkSeeds S e w σ σ').flatMap fun s => S.hashes (Mc.eval M (s, r)).1 s (Mc.eval M (s, r)).2

/-- **The fork finder `B^M`** at the adversary's coins `r` and round `w`, with `k` continuations. Its coins are a
seed and `k` more; it runs `M` at `r` on the seed and on each continuation forked from it at round `w`
(`forkSeeds`), and outputs the first collision of `sha` among the verifier's SHA-512 inputs in those `k + 1` runs
(`Claims.findColl`). Its cost is those inputs' count plus `k + 1` runs of `M` at `M`'s worst-case time, which counts
`M`'s own SHA-512 calls wherever a call costs at least a unit of the machine's time. -/
noncomputable def forkFinder {Mc : Machine} (S : HashedService) {Ω : Type} {R : ℕ} (e : S.Seed ≃ Paths Ω R)
    (sha : Claims.Msg → Claims.Msg) (k : ℕ) {Ω' : Type} (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)) (r : Ω')
    (w : Fin R) : Claims.Finder where
  S := S.Seed × (Fin k → S.Seed)
  out s := Claims.findColl ((forkLog S e M r w s.1 s.2).map fun u => (u, sha u))
  cost s := (forkLog S e M r w s.1 s.2).length + (k + 1) * ⌈Mc.time M⌉₊

/-- **The fork finders of every adversary**, as `Finders`: `forkFinder` at each adversary, value of its coins and
round. -/
noncomputable def forkFinders {Mc : Machine} (S : HashedService) {Ω : Type} {R : ℕ} (e : S.Seed ≃ Paths Ω R)
    (sha : Claims.Msg → Claims.Msg) (k : ℕ) : Finders Mc S.toService R :=
  fun _ _ M r w => forkFinder S e sha k M r w

/-- **An adversary's error at its finders' collision probabilities**: at each value `r` of its coins, the luck term
`ε`, the constants `δb` and `δz` of each of the `R` rounds' binding and zero-knowledge breaks, and its binding and
zero-knowledge finders' collision probabilities summed over the rounds at the weights `βb` and `βz`, capped at 1;
then averaged over `r`. It assumes nothing of SHA-512: `ecr/sha-512` at the finders turns each probability into a
number outside the floors, and `breakBound` is that reading (`breakPr ≤ breakBound` at every budget). -/
noncomputable def breakPr {Mc : Machine} {S : Service} {R : ℕ} (sha : Claims.Msg → Claims.Msg)
    (fb fz : Finders Mc S R) (ε δb βb δz βz : ℝ) (Ω : Type) [Fintype Ω]
    (M : Mc.Alg (S.Seed × Ω) (S.Com × S.Ans)) : ENNReal :=
  ∑ r : Ω, (Fintype.card Ω : ENNReal)⁻¹ *
    min 1 (ENNReal.ofReal
      (ε + R * (δb + δz) + βb * (∑ w, (fb Ω M r w).Pr sha) + βz * ∑ w, (fz Ω M r w).Pr sha))

/-- **The machine of functions at no cost**: an algorithm is a function, and every run costs nothing. An adversary
there is charged nothing, so a floor at a positive charge is a bound on its acceptance alone. -/
def funMachine : Machine where
  Alg I O := I → O
  eval f := f
  time _ := 0
  cost _ _ := 0
  size _ := 0
  seq f g := g ∘ f
  pair f g x := (f x, g x)
  fst := Prod.fst
  snd := Prod.snd
  eval_seq _ _ _ := rfl
  eval_pair _ _ _ := rfl
  eval_fst _ := rfl
  eval_snd _ := rfl
  time_nonneg _ := le_rfl
  cost_le_time _ _ := le_rfl
  size_le_time _ := le_rfl
  size_nonneg _ := le_rfl
  time_seq_le _ _ := by norm_num
  le_time_seq _ _ := le_rfl
  time_pair_le _ _ := by norm_num
  le_time_pair_left _ _ := le_rfl
  le_time_pair_right _ _ := le_rfl
  time_fst := rfl
  time_snd := rfl

end Pouw.RelativeSecurity
