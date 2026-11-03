import Proofs.Properties.Verity.Recursive
import Proofs.Flock.Claims.Recursion.Wald

/-!
# The recursive configuration's binding finders

The binding term of the recursive audit's error reads `ecr/sha-512` (`Claims.ExpectedCR`) at two finders built from
the strategy's session, each a `Claims.Finder` whose cost counts every SHA-512 call it makes:

* `Dep.bindAt`, two independent plays and `collOf` on their witnesses, whose clause gives the term
  `√(min(W, 2n) · E[bindBudget] / 2^256)` (`Dep.delta`);
* `Dep.chaseAt`, one play and, where its witness is off the plurality at a bit, `M` more plays at that bit, `M` read
  off the strategy's weights there, and `collOf` on the first pair that disagrees, whose clause gives the term
  `chaseBudget / ((1 − e^{−β/2}) · 2^256)` with `chaseBudget = (2 + β n) · E[playBudget] + dColl` (`Dep.deltaChase`).

A play's cost (`Dep.playCost`) is its run's calls, its extraction's (`Extractor.eval`'s, every rerun's run included)
and the extraction's inner-view calls (`Dep.ivCalls`), which `E*` makes outside its program when it reads each inner
rerun's view off the outer witnesses. Testing a play's witness (the recursive relation, `readX`) makes no call: the
relation reads no oracle. `collOf` is an oracle program, so its calls are its log's. The error (`Dep.errAt`) is the
smaller of the two conditional terms.
-/

namespace Verity.Recursive

open scoped ENNReal
open Claims (Msg)
open FlockDeploy (Coin)
open RecDeploy
open Accounting (Device)

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α]

/-- **The bridge from two readings to a collision**: at two statements and witnesses, an oracle program at SHA-512 that
outputs a pair of messages, its calls the log `Claims.OC.eval` keeps. -/
abbrev CollOf : Type := Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Claims.OC (Option (Msg × Msg))

/-- **`collOf` at a pair of statements and witnesses, or at none**: its output and its SHA-512 calls, and nothing
at no pair. -/
def collRun (collOf : CollOf) : Option (Msg × Claims.Flock.Wit × Msg × Claims.Flock.Wit) → Option (Msg × Msg) × ℕ
  | some (x₁, w₁, x₂, w₂) =>
    (((collOf x₁ w₁ x₂ w₂).eval FlockDeploy.sha512).1, ((collOf x₁ w₁ x₂ w₂).eval FlockDeploy.sha512).2.length)
  | none => (none, 0)

/-- **The file's wire bits by unit**, in the order `Dep.chaseAt` reads them: by unit, then wire, then bit. -/
noncomputable def unitBits (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) :
    List (Fin (V.units θ F) × Σ g : Fin (V.circuitOf θ F).N, Fin ((V.circuitOf θ F).w g)) :=
  (List.finRange (V.units θ F)).flatMap fun u => (List.finRange (V.circuitOf θ F).N).flatMap fun g =>
    (List.finRange ((V.circuitOf θ F).w g)).map fun i => (u, ⟨g, i⟩)

namespace Dep

variable (q : Dep) (V : Vocab d α) {k : Verity.Registration.Composite} {θ c : ByteArray} {F : Array ByteArray}
  {σ : Game.Strategy (V.game θ F c)}

/-- **Some run of the deployed recursive verifier accepts at `x`**, at some prover, tape and coins: the statements the
call bounds and the level are read at. -/
def RunAccepts (x : Msg) : Prop :=
  ∃ (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Fin (q.D.V.k x) → Coin),
    (Claims.run FlockDeploy.sha512 P q.D.V x ω r).out = .acc

/-- The deployment's outer extractor `E_O`, which serves `E*`'s inner reruns. -/
noncomputable abbrev EO : Claims.Extractor Coin (List Claims.Flock.Wit) q.D.V :=
  Claims.Recursion.EO q.D q.I (deployZkRec _root_.Flock.Readable.relation) q.decO

/-- **The extraction's inner-view calls** at the prover `P`, the statement `x`, the tape `ω`, the judged run's coins
`r₀` and `E*`'s seed `c` (`Extractor.eval`'s arguments): `E*` reads each inner rerun's view off `E_O`'s witnesses
(`Claims.Recursion.dec`) by running the inner verifier at SHA-512, outside its program, so `Extractor.eval` doesn't
count those calls. This is their count, `Deploy.innerCalls` at each inner rerun (at its coins, `RP.rerunsAt` of the
inner `E^rep`'s program served by `E_O`) where `E_O` returns witnesses. -/
noncomputable def ivCallsAt {Ω : Type} (P : Claims.Prover Ω) (x : Msg) (ω : Ω) (r₀ : Fin (q.D.V.k x) → Coin)
    (c : q.EO.C x × (Fin (q.I.waits (Claims.Recursion.xI q.D x)) → Fin (q.I.cap (Claims.Recursion.xI q.D x)) →
      (Fin (q.D.V.k x) → Coin) × q.EO.C x)) : ℕ :=
  (((Claims.erepProg (q.I.waits (Claims.Recursion.xI q.D x)) (q.I.k (Claims.Recursion.xI q.D x))
      (q.I.cap (Claims.Recursion.xI q.D x)) q.I.opens (q.I.assemble (Claims.Recursion.xI q.D x))
      (q.I.check (Claims.Recursion.xI q.D x)) (r₀, c.1) c.2).rerunsAt FlockDeploy.sha512
    (Claims.servedRun FlockDeploy.sha512
      (Claims.Recursion.outerStage q.EO (Claims.Recursion.dec FlockDeploy.sha512 q.D) x)
      (Claims.run FlockDeploy.sha512 P q.D.V x ω))).map fun dr =>
    ((q.EO.eval FlockDeploy.sha512 P x ω dr.1 dr.2).1.elim 0 fun ws =>
      (q.D.innerCalls FlockDeploy.sha512 x (Claims.coinSeq dr.1) ws).length)).sum

/-- **The extraction's inner-view calls at a play** at the draw `ξ` and the sample `s` (`ivCallsAt`). -/
noncomputable def ivCalls (S : SessionIn V q k θ F c σ) (ξ : S.Ξ) (s : q.Sample V S ξ) : ℕ :=
  q.ivCallsAt (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2

/-- **A play's SHA-512 calls** at the draw `ξ` and the sample `s`: the run's, both parties', the extraction's
(`Extractor.eval`'s, every rerun's run included) with its rerun count, and the extraction's inner-view calls. -/
noncomputable def playCost (S : SessionIn V q k θ F c σ) (ξ : S.Ξ) (s : q.Sample V S ξ) : ℕ :=
  runCalls (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1) +
    (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).2.1 +
    (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).2.2 + q.ivCalls V S ξ s

/-- **A play's budget at the draw `ξ`**, a defined function of the strategy's cost `t'` and public counts: a run, and
an extraction of `runs` reruns in expectation (`Claims.Recursion.efficient`), each a run of the session and so at most
`t'` calls, with its rerun count, and `N_I` inner reruns in expectation, each reading an inner view of at most `nI`
calls: `(1 + r) · t' + r + N_I · nI`. -/
noncomputable def playBudget (S : SessionIn V q k θ F c σ) (nI : Msg → ℕ) (ξ : S.Ξ) : ℕ :=
  (1 + q.runs (S.x ξ)) * q.tCost V S + q.runs (S.x ξ) + Claims.Recursion.NI q.D q.I (S.x ξ) * nI (S.x ξ)

/-- **The binding finder at the draws `ξ₁` and `ξ₂`**: it plays the session twice, at `ξ₁` and at `ξ₂`, with
independent tapes, coins and seeds, runs the extractor on both, and runs `collOf` on the two witnesses. Its cost is the
two plays' calls (`playCost`) and `collOf`'s. -/
noncomputable def bindAt (Se : SessionIn V q k θ F c σ) (collOf : CollOf) (ξ₁ ξ₂ : Se.Ξ) : Claims.Finder where
  S := q.Sample V Se ξ₁ × q.Sample V Se ξ₂
  out s := (collRun collOf ((q.E.eval FlockDeploy.sha512 (Se.P ξ₁) (Se.x ξ₁) s.1.1 s.1.2.1 s.1.2.2).1.bind fun w₁ =>
    (q.E.eval FlockDeploy.sha512 (Se.P ξ₂) (Se.x ξ₂) s.2.1 s.2.2.1 s.2.2.2).1.map fun w₂ =>
      (Se.x ξ₁, w₁, Se.x ξ₂, w₂))).1
  cost s := q.playCost V Se ξ₁ s.1 + q.playCost V Se ξ₂ s.2 +
    (collRun collOf ((q.E.eval FlockDeploy.sha512 (Se.P ξ₁) (Se.x ξ₁) s.1.1 s.1.2.1 s.1.2.2).1.bind fun w₁ =>
      (q.E.eval FlockDeploy.sha512 (Se.P ξ₂) (Se.x ξ₂) s.2.1 s.2.2.1 s.2.2.2).1.map fun w₂ =>
        (Se.x ξ₁, w₁, Se.x ξ₂, w₂))).2

/-- **The binding finder's budget at the draws `ξ₁`, `ξ₂`**: the two plays' budgets and `collOf`'s calls `dColl`. -/
noncomputable def bindBudget (S : SessionIn V q k θ F c σ) (nI : Msg → ℕ) (dColl : ℕ) (ξ₁ ξ₂ : S.Ξ) : ℕ :=
  q.playBudget V S nI ξ₁ + q.playBudget V S nI ξ₂ + dColl

/-- **A play** of the session: a draw, and a sample at every draw, read at its own. Uniform plays are a uniform draw
and then a uniform sample at it, though the samples' sizes vary with the draw. -/
abbrev Play (S : SessionIn V q k θ F c σ) : Type := S.Ξ × ((ξ : S.Ξ) → q.Sample V S ξ)

open Classical in
/-- **A play's witness** at the draw `ξ` and the sample `s`: the extractor's, where the run accepts and the witness is
in the recursive relation, and none elsewhere. -/
noncomputable def playWit (S : SessionIn V q k θ F c σ) (ξ : S.Ξ) (s : q.Sample V S ξ) :
    Option Claims.Flock.Wit :=
  if (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc then
    (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1.filter fun wt => decide (q.R (S.x ξ) wt)
  else none

open Classical in
/-- **The bit a play is off at**: the least unit `u`, wire `g` of `u` and bit `i` of `g` (in `unitBits`' order) where
the play's witness is read off the plurality `Dep.extract`, at a unit the draw holds, and none where the play has no
witness or none is off. A fixed rule of the play and the plurality. -/
noncomputable def chaseSel (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript) (ξ : S.Ξ) (s : q.Sample V S ξ) :
    Option (Fin (V.units θ F) × Σ g : Fin (V.circuitOf θ F).N, Fin ((V.circuitOf θ F).w g)) :=
  (q.playWit V S ξ s).bind fun wt => (unitBits V θ F).find? fun b =>
    decide (b.1 ∈ S.drawOf ξ ∧ b.2.1 ∈ (V.partsOf θ F).io b.1 ∧
      (readX (S.x ξ) wt b.2.1).getLsbD b.2.2 = !(q.extract V S readX b.2.1).getLsbD b.2.2)

/-- **`chaseAt`'s rate**: `β = √(8/n)` at `n` units. -/
noncomputable def chaseBeta (n : ℕ) : ℝ := Real.sqrt (8 / n)

/-- **`chaseAt`'s plays at a bit** `b`: `⌈β / q_b⌉`, `q_b` the probability that a play reads bit `b.2` of the wire
`b.1` (either value; `Dep.weight` at both). A function of the strategy's weights, as the plurality is. -/
noncomputable def chaseM (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript)
    (b : Σ g : Fin (V.circuitOf θ F).N, Fin ((V.circuitOf θ F).w g)) : ℕ :=
  ⌈chaseBeta (V.units θ F) / (q.weight V S readX b.1 b.2 true + q.weight V S readX b.1 b.2 false)⌉₊

/-- **`chaseAt`'s further plays**: at each wire bit `b`, `chaseM b` independent plays, of which `chaseAt` reads those
at the bit its first play is off at. -/
abbrev ChasePlays (S : SessionIn V q k θ F c σ) (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript) :
    Type :=
  (b : Σ g : Fin (V.circuitOf θ F).N, Fin ((V.circuitOf θ F).w g)) → Fin (q.chaseM V S readX b) → q.Play V S

open Classical in
/-- **A play's witness reading the plurality at the bit `b`**: the play's witness, where a unit the draw holds has the
wire `b.1` and the witness reads `Dep.extract`'s bit there, and none elsewhere (`wEvent` at the plurality's bit). -/
noncomputable def chaseHit (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript)
    (b : Σ g : Fin (V.circuitOf θ F).N, Fin ((V.circuitOf θ F).w g)) (t : q.Play V S) : Option Claims.Flock.Wit :=
  (q.playWit V S t.1 (t.2 t.1)).filter fun wt =>
    decide ((∃ u ∈ S.drawOf t.1, b.1 ∈ (V.partsOf θ F).io u) ∧
      (readX (S.x t.1) wt b.1).getLsbD b.2 = (q.extract V S readX b.1).getLsbD b.2)

/-- **The pair `chaseAt` hands `collOf`** at its first play `p.1` and further plays `p.2`: where the first play is off
at the unit and bit `b` (`chaseSel`), the first play's statement and witness with the first of `b`'s `chaseM b`
further plays that reads the plurality at `b`; none elsewhere. -/
noncomputable def chasePair (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript) (p : q.Play V S × q.ChasePlays V S readX) :
    Option (Msg × Claims.Flock.Wit × Msg × Claims.Flock.Wit) :=
  (q.chaseSel V S readX p.1.1 (p.1.2 p.1.1)).bind fun b => (q.playWit V S p.1.1 (p.1.2 p.1.1)).bind fun w₀ =>
    ((List.finRange (q.chaseM V S readX b.2)).filterMap fun j =>
      (q.chaseHit V S readX b.2 (p.2 b.2 j)).map fun w => (S.x (p.2 b.2 j).1, w)).head?.map
        fun xw => (S.x p.1.1, w₀, xw.1, xw.2)

open Classical in
/-- **The chasing finder**: it plays the session once; where that play is off the plurality at the unit and bit `b`
(`chaseSel`), it plays `chaseM b` more times, independently, and runs `collOf` on the first play's witness and the
first further one that reads the plurality at `b` (`chasePair`). Its cost is every play's calls (`playCost`) and
`collOf`'s. Its rule reads the strategy's weights (`chaseM`, and the plurality through `chaseSel`), as fixed data. -/
noncomputable def chaseAt (Se : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript) (collOf : CollOf) : Claims.Finder where
  S := q.Play V Se × q.ChasePlays V Se readX
  out p := (collRun collOf (q.chasePair V Se readX p)).1
  cost p := q.playCost V Se p.1.1 (p.1.2 p.1.1) +
    ((q.chaseSel V Se readX p.1.1 (p.1.2 p.1.1)).elim 0 fun b =>
      ∑ j : Fin (q.chaseM V Se readX b.2), q.playCost V Se (p.2 b.2 j).1 ((p.2 b.2 j).2 (p.2 b.2 j).1)) +
    (collRun collOf (q.chasePair V Se readX p)).2

/-- **The chasing finder's budget**: `(2 + β n) · E[playBudget] + dColl`, a defined function of the strategy's cost `t'`
and public counts, `n` the file's units and the budget averaged over the draws. -/
noncomputable def chaseBudget (S : SessionIn V q k θ F c σ) (nI : Msg → ℕ) (dColl : ℕ) : ℝ :=
  (2 + chaseBeta (V.units θ F) * V.units θ F) * Claims.Ex (fun ξ : S.Ξ => (q.playBudget V S nI ξ : ℝ)) + dColl

/-- **`FinderCR`: the named SHA-512 premises at the strategy, asked of the finders built from it, at the binding
finder.** Each session prover's finders meet C-Flock's premises (`SessionCR`, at the inner and setup call bounds `nI`
and `vS`; `B`'s runs are the session's, at most `t'` calls each), and each binding finder (`bindAt`) meets
`ecr/sha-512` (`Claims.ExpectedCR`) at its own cost. -/
def FinderCR (S : SessionIn V q k θ F c σ) (collOf : CollOf) (nI vS : Msg → ℕ) : Prop :=
  (∀ ξ, q.SessionCR (nI (S.x ξ)) (vS (S.x ξ)) (S.x ξ) S.Ω (S.P ξ)) ∧
    ∀ ξ₁ ξ₂, Claims.ExpectedCR FlockDeploy.sha512 (q.bindAt V S collOf ξ₁ ξ₂)

/-- **`ChaseCR`: `FinderCR` at the chasing finder.** The session provers' premises (`SessionCR`), and the chasing
finder (`chaseAt`) meets `ecr/sha-512` (`Claims.ExpectedCR`) at its own cost. -/
def ChaseCR (S : SessionIn V q k θ F c σ) (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript)
    (collOf : CollOf) (nI vS : Msg → ℕ) : Prop :=
  (∀ ξ, q.SessionCR (nI (S.x ξ)) (vS (S.x ξ)) (S.x ξ) S.Ω (S.P ξ)) ∧
    Claims.ExpectedCR FlockDeploy.sha512 (q.chaseAt V S readX collOf)

/-- **The error's number at the binding finder**, from `t'` and public parameters: the draw's miss `δs`, plus
`BitsLin2On`'s `t' · 2^{-λ}`, plus the binding term `√(min(W, 2n) · E[bindBudget] / 2^256)`, with `W` the file's wire
bits, `n` its units and the budget averaged over pairs of draws. -/
noncomputable def delta (S : SessionIn V q k θ F c σ) (nI : Msg → ℕ) (dColl : ℕ) (δs lam : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal (δs + q.tCost V S * (2 : ℝ) ^ (-lam) +
    Real.sqrt (min ((∑ g, (V.circuitOf θ F).w g : ℕ) : ℝ) (2 * V.units θ F) *
      (Claims.Ex (fun p : S.Ξ × S.Ξ => (q.bindBudget V S nI dColl p.1 p.2 : ℝ)) / 2 ^ (256 : ℝ))))

/-- **The error's number at the chasing finder**: `δs + t' · 2^{-λ}`, plus the binding term
`chaseBudget / ((1 − e^{−β/2}) · 2^256)`: where a play is off the plurality, the chasing finder's further plays read
the plurality at its bit with probability at least `1 − e^{−β/2}`. -/
noncomputable def deltaChase (S : SessionIn V q k θ F c σ) (nI : Msg → ℕ) (dColl : ℕ) (δs lam : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal (δs + q.tCost V S * (2 : ℝ) ^ (-lam) +
    q.chaseBudget V S nI dColl / ((1 - Real.exp (-(chaseBeta (V.units θ F) / 2))) * 2 ^ (256 : ℝ)))

open Classical in
/-- **The error at the strategy's session `S`**: the smaller of the two conditional terms, each the shared
`condError P δ := if P then δ else ⊤` (the shape of `Vocab.error`): `delta` where the finders built from the strategy
meet `FinderCR`, `deltaChase` where they meet `ChaseCR`, and `⊤` where they meet neither. -/
noncomputable def errAt (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript) (collOf : CollOf) (dColl : ℕ)
    (nI vS : Msg → ℕ) (δs lam : ℝ) : ℝ≥0∞ :=
  min (if q.FinderCR V S collOf nI vS then q.delta V S nI dColl δs lam else ⊤)
    (if q.ChaseCR V S readX collOf nI vS then q.deltaChase V S nI dColl δs lam else ⊤)

end Dep

/-- **Its conclusion fails for a verifier that accepts everything**: at a strategy whose extracted transcript fails
the claim, where the finders built from it meet `FinderCR` and the error's number (`Dep.delta`) is below one. -/
theorem sampledProofsRecursive_nontrivial (q : Dep) (V : Vocab d α) (k : Verity.Registration.Composite)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript) (collOf : CollOf) (dColl : ℕ)
    (nI vS : Msg → ℕ) (δs lam : ℝ)
    (hbad : ¬ V.Holds k θ F c σ .sampled ⟨θ, F, c, [], ByteArray.empty, #[]⟩)
    (hcr : q.FinderCR V S collOf nI vS) (hlt : q.delta V S nI dColl δs lam < 1) :
    ¬ Game.prob (fun o => True ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ ≤
      q.errAt V S readX collOf dColl nI vS δs lam := by
  intro h
  have h1 : Game.prob (fun o => True ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ = 1 :=
    prob_play_eq_one (V.rounds θ) θ F c (fun o => True ∧ ¬ V.Holds k θ F c σ .sampled o)
      (fun _ _ => ⟨trivial, hbad⟩) σ
  have h2 : q.errAt V S readX collOf dColl nI vS δs lam ≤ q.delta V S nI dColl δs lam := by
    unfold Dep.errAt
    rw [if_pos hcr]
    exact min_le_left _ _
  exact lt_irrefl (1 : ℝ≥0∞) ((h1.symm.trans_le (h.trans h2)).trans_lt hlt)

end Verity.Recursive
