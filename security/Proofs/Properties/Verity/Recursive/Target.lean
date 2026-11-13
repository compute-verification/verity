module

public import Proofs.Properties.Verity.Recursive
public import Proofs.Flock.Claims.Prob
public import Proofs.Flock.Claims.Recursion.BitsLin2
public import Proofs.Flock.Claims.Recursion.DeployFacts
public import Proofs.Core.Circuit
public import Proofs.Properties.Verity.Majority

/-!
# The target, proved: sampled proofs at the recursive configuration

`spotRecursive` proves the target's bound at each strategy from the property `acc_of_accepts` states
(`AccOfAccepts`, the statement of `acc_of_accepts` at the deployment), at `acc_of_accepts`.

The proof, at a strategy `σ` whose finders meet `FinderCR` (elsewhere the error is `⊤`), where the files pass the
commit phase's check (elsewhere no play is accepted) and sampled proofs' claim fails on the extracted transcript `X`
(elsewhere the event is empty), so that the unbound units `U` number more than `ε n`:
* each accepted play shows, through the coupling (`SessionIn.names_shows`) and G, an accepted run of the recursive
  verifier at a dense statement (`verdict_run`);
* at each sample `ξ` such a play has, the sample misses `U`, or the run accepts while the extractor returns no witness in
  the recursive relation (`PrBad`, at most `Dep.ksAt`: `t' · 2^{-λ}` by `RecursiveKSAt` through
  `bitsLin2On_of_ksLin2On`, and at a statement at the deployment's tables also `2^{-167.7} + t' · 2^{-174.95}` through
  `bitsLin2KOn_of_ksLin2On` and `Dep.kappa_eps_oneTable`), or the
  extractor's witness, correct and opened on every sampled proof block (`hread`), differs from `X` at some bit of a wire of a sampled
  proof block of `U` (`unbound_drawn`), the bit `X`'s plurality didn't pick (`offEvent`, `pr_verdict_le`);
* averaged over the samples, the miss is at most `δs` (`hdraw`), and the last event is at most `√(min(W, 2n) · K)`,
  `K = E[bindBudget] / 2^256`, with `W` the file's wire bits and `n` its units. Both bounds pick one piece per play
  (`units_bound`), giving each piece a share `a` at most `p`, and rest on `hcoll`, which turns two independent plays
  that read a sampled wire differently into the binding finder's collision, so that a play off `X` at a bit collides with
  every play that reads `X`'s bit there (`majority_le_coll`):
  - at the bits (`ex_off_le_bits`), `p` is the bit's minority weight, at most its majority's
    (`minority_le_majority`), so `∑ a p ≤ K` and `(∑ a)² ≤ W ∑ a² ≤ WK`;
  - at the units (`ex_off_le_units`), `p_u` is the probability of sampling `u`, and the majority at any bit of `u`'s
    wires weighs at least `p_u / 2` (`drawn_le_weight`), so `∑ a p ≤ 2K` and `(∑ a)² ≤ n ∑ a² ≤ 2nK`.

The averaging is `Verity.Majority`'s, which the two-stage leg reads too: the session's plays are a `Majority.Plays`
(`Dep.plays`), `Dep.weight` and `Dep.extract` are its weight and plurality, and each lemma above is the shared one at
them.
-/

@[expose] public section

namespace Verity.Recursive

open scoped ENNReal
open Claims (Msg Out)
open FlockDeploy (Coin)
open RecDeploy

open Verity.Majority (ex_ind pr_or_le ex_add3)
open Claims (Ex Pr Ex_nonneg Ex_mono Ex_div Pr_nonneg Pr_mono Pr_le_one Pr_false)

/-- **`acc_of_accepts`' property at the deployment `q`** (`acc_of_accepts`' statement): where the compiled recursive component accepts
blobs that a statement names, that show a run of the deployed recursive verifier with the component's staged rows, and
whose records hold the run's live coins, the statement is dense and the run accepts. -/
def AccOfAccepts (q : Dep) : Prop :=
  ∀ (p : VStar.Recursive.Params) (c : Spot.OneStage.Commitment) (e : VStar.Recursive.Evidence)
    (bs : Core.Blobs) (pub : Spot.OneStage.Public) (prof : Core.Integrity),
    VStar.Recursive.checkBlobs p c e bs = .accept pub prof → ∀ x, Names p e bs x →
    ∀ (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Fin (q.D.V.k x) → Coin),
      Shows q.Bd.IVStar q.st x e.proofs bs (Claims.run FlockDeploy.sha512 P q.D.V x ω r).seen →
      RowsShow q.Bd.IVStar q.st x e.proofs bs (Claims.run FlockDeploy.sha512 P q.D.V x ω r).seen →
      LiveCoins q.Bd.IVStar q.st x (Claims.run FlockDeploy.sha512 P q.D.V x ω r).seen →
      RecDeploy.DenseAt x ∧ (Claims.run FlockDeploy.sha512 P q.D.V x ω r).out = .acc

/-- `AccOfAccepts` is `acc_of_accepts`. -/
theorem accOfAccepts (q : Dep) : AccOfAccepts q :=
  fun p c e bs _ _ hacc x hnames _ P ω r hshow hrows hlive =>
    acc_of_accepts q.wE q.w q.b q.aux q.nOf q.hd q.recRest p c e bs hacc x hnames P ω r hshow hrows hlive

/-- **Some run of the deployed recursive verifier accepts at `x`**, at some prover, tape and coins: the statements the
call bounds and the level are read at. -/
def Dep.RunAccepts (q : Dep) (x : Msg) : Prop :=
  ∃ (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Fin (q.D.V.k x) → Coin),
    (Claims.run FlockDeploy.sha512 P q.D.V x ω r).out = .acc

/-- **`κ ≤ 2^{-167.7}` and `ε ≤ 2^{-174.95}` at a statement at the deployment's tables**, at an inner setup of at most
`2^32` calls and call bounds `n ≤ 2^32` and `v ≤ nS x · 138543104` (`setupCalls_deployedB_le_cap`'s, at most
`16 · 138543104 ≤ 2^32` at the verifier's 16-part cap): `RecDeploy.kappa_eps_oneTable_cap` at the deployed recursive
record. -/
theorem Dep.kappa_eps_oneTable (q : Dep) {x : Msg} (h : q.OneTable x) (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 32)
    {n v : ℕ} (hn : n ≤ 2 ^ 32) (hv : v ≤ nS x * 138543104) :
    Claims.Recursion.κ q.D q.I x ≤ 2 ^ (-(167.7 : ℝ)) ∧
      Claims.Recursion.epsLin2 q.D q.I FlockDeploy.KS.CpDep x n v ≤ 2 ^ (-(174.95 : ℝ)) :=
  kappa_eps_oneTable_cap _root_.Flock.Readable.relation { deployedB q.b with tr := recTr q.aux q.nOf q.hd q.recRest }
    (DeployedR.layR OpenRows.elemR) (FlockDeploy.nextR DeployedR.rdR)
    (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols q.wE FlockDeploy.joinR
    rowInDeployed DeployedR.finalR q.w q.decI h hS hn hv

open Accounting (Device)

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α]

/-- **An accepted play shows an accepted run at a dense statement**: where the recursive configuration's verdict holds of
the play's outcome at the sample `ξ` and the session's tape and coins `s`, the coupling's statement is dense and the
deployed recursive verifier accepts the session's run at `s`. -/
theorem verdict_run (q : Dep) (hG : AccOfAccepts q) (V : Vocab d α) (k : Verity.Registration.Composite)
    {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} (S : SessionIn V q k θ F c σ)
    (hlive : ∀ ξ ω r, LiveCoins q.Bd.IVStar q.st (S.x ξ)
      (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) ω r).seen)
    (ξ : S.Ξ) (s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin)) (h : RecVerdict V k (S.out ξ s)) :
    RecDeploy.DenseAt (S.x ξ) ∧ (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2).out = .acc := by
  obtain ⟨-, -, a, cb, eb, hcase, hda, j, p, hj, hp⟩ := h
  obtain ⟨p', c', ev, hp', -, he, pub, prof, hch⟩ := id hda
  rw [hj] at hp'
  change VStar.Recursive.Sampled.paramsOfJson j = .ok p' at hp'
  rw [hp] at hp'
  cases hp'
  cases ev with
  | inl e₀ =>
      change VStar.Recursive.Sampled.checkBlobs (.inr p) c' (.inl e₀)
        (Core.Blobs.ofArray (S.out ξ s).sent.blobs) = .accept pub prof at hch
      simp [VStar.Recursive.Sampled.checkBlobs] at hch
  | inr e =>
      have hch' : VStar.Recursive.checkBlobs p c' e (Core.Blobs.ofArray (S.out ξ s).sent.blobs) = .accept pub prof :=
        hch
      obtain ⟨hN, hS, hR⟩ := S.names_shows ξ s a cb eb j p e hcase hda hj hp he
      exact hG p c' e _ pub prof hch' (S.x ξ) hN S.Ω (S.P ξ) s.1 s.2 hS hR (hlive ξ s.1 s.2)

/-- **The recursive session's plays** (`Majority.Plays` at the session `S` and the reading `readX`): the samples `ξ`,
a run with the extractor's seed at each (`Dep.Sample`), the run's accept, the units `ξ` samples, the extractor's
witness, the recursive relation at `ξ`'s statement and `readX` there. `Dep.weight` and `Dep.extract` are its weight and
plurality. -/
noncomputable def Dep.plays (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace) :
    Majority.Plays (V.circuitOf θ F) (V.units θ F) (V.partsOf θ F) where
  Ξ := S.Ξ
  Smp := q.Sample V S
  Wit := Claims.Flock.Wit
  acc ξ s := (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc
  drawn ξ _ u := u ∈ S.drawOf ξ
  wit ξ s := (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1
  R ξ := q.R (S.x ξ)
  read ξ := readX (S.x ξ)

theorem Dep.weight_plays (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace) :
    q.weight V S readX = (q.plays V S readX).weight := rfl

theorem Dep.extract_plays (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace) :
    q.extract V S readX = (q.plays V S readX).extract := by
  funext g
  unfold Dep.extract Majority.Plays.extract
  rw [Dep.weight_plays]

/-- **The event a bit's weight counts** (`Dep.weight`'s): at the sample `ξ` and a run with the extractor's seed `s`, the
run accepts, `g` is a wire of a unit `ξ` samples, and the extractor returns a witness in the recursive relation whose
reading has `bit` at bit `i` of `g`. -/
def wEvent (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray} {σ : Game.Strategy (V.game θ F c)}
    {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (g : Fin (V.circuitOf θ F).N) (i : ℕ) (bit : Bool) (ξ : S.Ξ) (s : q.Sample V S ξ) : Prop :=
  (q.plays V S readX).wEvent g i bit ξ s

theorem weight_eq (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray} {σ : Game.Strategy (V.game θ F c)}
    {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (g : Fin (V.circuitOf θ F).N) (i : ℕ) (bit : Bool) :
    q.weight V S readX g i bit = Ex fun ξ => Pr (wEvent q V S readX g i bit ξ) := rfl

/-- **The extracted transcript's bit is the plurality's**: bit `i` of gate `g` of `Dep.extract` is set exactly where its
weight at `true` exceeds its weight at `false`. -/
theorem extract_bit (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray} {σ : Game.Strategy (V.game θ F c)}
    {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (g : Fin (V.circuitOf θ F).N) {i : ℕ} (hi : i < (V.circuitOf θ F).w g) :
    (q.extract V S readX g).getLsbD i = true ↔
      q.weight V S readX g i false < q.weight V S readX g i true := by
  rw [Dep.extract_plays, Dep.weight_plays]
  exact (q.plays V S readX).extract_bit g hi

/-- **The plurality's minority weighs at most its majority**: at a bit `i` of a wire `g`, the weight of the bit
`Dep.extract` didn't pick is at most that of the bit it picked. -/
theorem minority_le_majority (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (g : Fin (V.circuitOf θ F).N) {i : ℕ} (hi : i < (V.circuitOf θ F).w g) :
    q.weight V S readX g i (!(q.extract V S readX g).getLsbD i) ≤
      q.weight V S readX g i ((q.extract V S readX g).getLsbD i) := by
  rw [Dep.extract_plays, Dep.weight_plays]
  exact (q.plays V S readX).minority_le_majority g hi

/-- **The event the binding term counts at the unit `u`**: at the sample `ξ` and a run with the extractor's seed `s`, the
run accepts, `ξ` samples `u`, and the extractor returns a witness in the recursive relation whose reading differs from
`Xt` at a bit of a wire of `u`. -/
def offEvent (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray} {σ : Game.Strategy (V.game θ F c)}
    {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace) (Xt : (V.circuitOf θ F).Trace)
    (u : Fin (V.units θ F)) (ξ : S.Ξ) (s : q.Sample V S ξ) : Prop :=
  (q.plays V S readX).offEvent Xt u ξ s

/-- **A run's sample of the unit `u`**: at the sample `ξ` and a run with the extractor's seed `s`, the run accepts, `ξ`
samples `u`, and the extractor returns a witness in the recursive relation. -/
def drawEvent (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray} {σ : Game.Strategy (V.game θ F c)}
    {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ) (u : Fin (V.units θ F)) (ξ : S.Ξ)
    (s : q.Sample V S ξ) : Prop :=
  (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc ∧ u ∈ S.drawOf ξ ∧
    ∃ wt, (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1 = some wt ∧ q.R (S.x ξ) wt

/-- **The binding finder's collision** (`bindAt`'s event) at the samples `ξ₁`, `ξ₂` and the runs with seeds `s₁`, `s₂`. -/
def collEvent (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray} {σ : Game.Strategy (V.game θ F c)}
    {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (ξ₁ : S.Ξ) (s₁ : q.Sample V S ξ₁) (ξ₂ : S.Ξ) (s₂ : q.Sample V S ξ₂) : Prop :=
  ∃ m m', (q.bindAt V S collOf dColl ξ₁ ξ₂).out (s₁, s₂) = some (m, m') ∧ Claims.Collision FlockDeploy.sha512 m m'

/-- **The event the binding term counts at the wire bit `b`**: `wEvent` at bit `b.2` of the wire `b.1`, reading the
bit `Dep.extract` didn't pick there. -/
def bitEvent (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray} {σ : Game.Strategy (V.game θ F c)}
    {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (b : Σ g : Fin (V.circuitOf θ F).N, Fin ((V.circuitOf θ F).w g)) (ξ : S.Ξ) (s : q.Sample V S ξ) : Prop :=
  (q.plays V S readX).bitEvent b ξ s

/-- **`hcoll` is the plays' binding at the binding finder's collision**: where `collOf`, from two witnesses in the
recursive relation that read a wire of units sampled in both differently, outputs a SHA-512 collision, two runs whose
extracted witnesses do so are a collision of `bindAt`. -/
theorem plays_binds (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (hcoll : ∀ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf θ F).N), q.R (S.x ξ₁) w₁ → q.R (S.x ξ₂) w₂ →
      (∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf θ F).io u) → (∃ u ∈ S.drawOf ξ₂, g ∈ (V.partsOf θ F).io u) →
      readX (S.x ξ₁) w₁ g ≠ readX (S.x ξ₂) w₂ g →
      ∃ m m', collOf (S.x ξ₁) w₁ (S.x ξ₂) w₂ = some (m, m') ∧ Claims.Collision FlockDeploy.sha512 m m') :
    (q.plays V S readX).Binds (collEvent q V S collOf dColl) := by
  intro ξ₁ s₁ ξ₂ s₂ w₁ w₂ g he₁ he₂ hR₁ hR₂ hd₁ hd₂ hne
  have he₁' : (q.E.eval FlockDeploy.sha512 (S.P ξ₁) (S.x ξ₁) s₁.1 s₁.2.1 s₁.2.2).1 = some w₁ := he₁
  have he₂' : (q.E.eval FlockDeploy.sha512 (S.P ξ₂) (S.x ξ₂) s₂.1 s₂.2.1 s₂.2.2).1 = some w₂ := he₂
  obtain ⟨m, m', hm, hcol⟩ := hcoll ξ₁ ξ₂ w₁ w₂ g hR₁ hR₂ hd₁ hd₂ hne
  refine ⟨m, m', ?_, hcol⟩
  simp only [Dep.bindAt, he₁', he₂']
  exact hm

/-- **A run off the plurality collides with every run on it**: where a run at the sample `ξ₁` with seed `s₁` returns a
witness in the recursive relation whose reading, at a bit `i` of a wire `g` that `ξ₁` samples, is not the bit
`Dep.extract` picked, the weight of the picked bit is at most the probability, over a second, independent sample and
run, that the binding finder at the two collides (`hcoll`). -/
theorem majority_le_coll (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (hcoll : ∀ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf θ F).N), q.R (S.x ξ₁) w₁ → q.R (S.x ξ₂) w₂ →
      (∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf θ F).io u) → (∃ u ∈ S.drawOf ξ₂, g ∈ (V.partsOf θ F).io u) →
      readX (S.x ξ₁) w₁ g ≠ readX (S.x ξ₂) w₂ g →
      ∃ m m', collOf (S.x ξ₁) w₁ (S.x ξ₂) w₂ = some (m, m') ∧ Claims.Collision FlockDeploy.sha512 m m')
    {ξ₁ : S.Ξ} {s₁ : q.Sample V S ξ₁} {w₁ : Claims.Flock.Wit} {g : Fin (V.circuitOf θ F).N} {i : ℕ}
    (hd₁ : ∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf θ F).io u)
    (he₁ : (q.E.eval FlockDeploy.sha512 (S.P ξ₁) (S.x ξ₁) s₁.1 s₁.2.1 s₁.2.2).1 = some w₁) (hR₁ : q.R (S.x ξ₁) w₁)
    (hb₁ : (readX (S.x ξ₁) w₁ g).getLsbD i = !(q.extract V S readX g).getLsbD i) :
    q.weight V S readX g i ((q.extract V S readX g).getLsbD i) ≤
      Ex (fun ξ₂ => Pr (collEvent q V S collOf dColl ξ₁ s₁ ξ₂)) := by
  rw [Dep.extract_plays] at hb₁
  rw [Dep.extract_plays, Dep.weight_plays]
  exact (q.plays V S readX).majority_le_coll (collEvent q V S collOf dColl)
    (plays_binds q V S readX collOf dColl hcoll) (s₁ := s₁) hd₁ he₁ hR₁ hb₁

/-- **Two independent plays collide at most `E[bindBudget] / 2^256`** under `FinderCR`'s binding premise (`hbind`):
the binding finder at each pair of samples is the pair of plays. -/
theorem ex_coll_le (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (hbind : ∀ ξ₁ ξ₂, (q.bindAt V S collOf dColl ξ₁ ξ₂).Pr FlockDeploy.sha512 ≤
      (q.bindBudget V S dColl ξ₁ ξ₂ : ℝ) / 2 ^ (256 : ℝ)) :
    Ex (fun p : S.Ξ × S.Ξ => Pr (fun s : q.Sample V S p.1 × q.Sample V S p.2 =>
      collEvent q V S collOf dColl p.1 s.1 p.2 s.2)) ≤
      Ex (fun p : S.Ξ × S.Ξ => (q.bindBudget V S dColl p.1 p.2 : ℝ)) / 2 ^ (256 : ℝ) :=
  Majority.ex_pairs_le (Smp := q.Sample V S) (collEvent q V S collOf dColl)
    (fun ξ₁ ξ₂ => (q.bindBudget V S dColl ξ₁ ξ₂ : ℝ)) _ hbind

/-- **The plurality's bit weighs at least half a sampled proof block**: at a bit `i` of a wire `g` of the unit `u`, the weight of
the bit `Dep.extract` picked there is at least half the probability that the run accepts, samples `u` and the extractor
returns a witness in the recursive relation. -/
theorem drawn_le_weight (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace) {u : Fin (V.units θ F)}
    {g : Fin (V.circuitOf θ F).N} (hg : g ∈ (V.partsOf θ F).io u) {i : ℕ} (hi : i < (V.circuitOf θ F).w g) :
    Ex (fun ξ => Pr (drawEvent q V S u ξ)) ≤ 2 * q.weight V S readX g i ((q.extract V S readX g).getLsbD i) := by
  rw [Dep.extract_plays, Dep.weight_plays]
  exact (q.plays V S readX).drawn_le_weight hg hi

/-- **The binding term at the bits**: averaged over the samples, the probability that the run accepts while the
extractor's witness differs from the plurality `Dep.extract` at a bit of a wire of a sampled proof block is at most
`√(W · E[bindBudget] / 2^256)`, `W` the file's wire bits (`Majority.Plays.ex_off_le_bits` at the session's plays): a
run off the plurality at a bit collides, through `hcoll`, with every run that reads the plurality's bit there, which
weighs at least the bit's minority (`minority_le_majority`). -/
theorem ex_off_le_bits (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (hcoll : ∀ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf θ F).N), q.R (S.x ξ₁) w₁ → q.R (S.x ξ₂) w₂ →
      (∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf θ F).io u) → (∃ u ∈ S.drawOf ξ₂, g ∈ (V.partsOf θ F).io u) →
      readX (S.x ξ₁) w₁ g ≠ readX (S.x ξ₂) w₂ g →
      ∃ m m', collOf (S.x ξ₁) w₁ (S.x ξ₂) w₂ = some (m, m') ∧ Claims.Collision FlockDeploy.sha512 m m')
    (hbind : ∀ ξ₁ ξ₂, (q.bindAt V S collOf dColl ξ₁ ξ₂).Pr FlockDeploy.sha512 ≤
      (q.bindBudget V S dColl ξ₁ ξ₂ : ℝ) / 2 ^ (256 : ℝ)) :
    Ex (fun ξ => Pr (fun s : q.Sample V S ξ => ∃ u, offEvent q V S readX (q.extract V S readX) u ξ s)) ≤
      Real.sqrt ((∑ g, (V.circuitOf θ F).w g : ℕ) *
        (Ex (fun p : S.Ξ × S.Ξ => (q.bindBudget V S dColl p.1 p.2 : ℝ)) / 2 ^ (256 : ℝ))) := by
  rw [Dep.extract_plays]
  exact (q.plays V S readX).ex_off_le_bits (collEvent q V S collOf dColl) (plays_binds q V S readX collOf dColl hcoll)
    (ex_coll_le q V S collOf dColl hbind)

/-- **The binding term at the units**: averaged over the samples, the probability that the run accepts while the
extractor's witness differs from the plurality `Dep.extract` at a bit of a wire of a sampled proof block is at most
`√(2n · E[bindBudget] / 2^256)`, `n` the file's units (`Majority.Plays.ex_off_le_units` at the session's plays): a run
off the plurality at a bit of a wire of `u` collides, through `hcoll`, with every run that reads the plurality's bit
there, which weighs at least half `u`'s sample (`drawn_le_weight`). -/
theorem ex_off_le_units (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (hcoll : ∀ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf θ F).N), q.R (S.x ξ₁) w₁ → q.R (S.x ξ₂) w₂ →
      (∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf θ F).io u) → (∃ u ∈ S.drawOf ξ₂, g ∈ (V.partsOf θ F).io u) →
      readX (S.x ξ₁) w₁ g ≠ readX (S.x ξ₂) w₂ g →
      ∃ m m', collOf (S.x ξ₁) w₁ (S.x ξ₂) w₂ = some (m, m') ∧ Claims.Collision FlockDeploy.sha512 m m')
    (hbind : ∀ ξ₁ ξ₂, (q.bindAt V S collOf dColl ξ₁ ξ₂).Pr FlockDeploy.sha512 ≤
      (q.bindBudget V S dColl ξ₁ ξ₂ : ℝ) / 2 ^ (256 : ℝ)) :
    Ex (fun ξ => Pr (fun s : q.Sample V S ξ => ∃ u, offEvent q V S readX (q.extract V S readX) u ξ s)) ≤
      Real.sqrt (2 * V.units θ F *
        (Ex (fun p : S.Ξ × S.Ξ => (q.bindBudget V S dColl p.1 p.2 : ℝ)) / 2 ^ (256 : ℝ))) := by
  rw [Dep.extract_plays]
  exact (q.plays V S readX).ex_off_le_units (collEvent q V S collOf dColl) (plays_binds q V S readX collOf dColl hcoll)
    (ex_coll_le q V S collOf dColl hbind)

/-- **A sampled unbound unit gives a wire bit off the plurality**: where the witness's reading is correct on, and opened
at, every unit the sample holds (`hread`), a sampled proof block unbound in `X` has a wire where `X` differs from the reading, at
some bit (`Majority.unbound_drawn` at the file's partition). -/
theorem unbound_drawn (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray} {Xw Xe : (V.circuitOf θ F).Trace}
    {u : Fin (V.units θ F)}
    (hc : (V.partsOf θ F).Correct Xw u) (ho : ∀ g ∈ (V.partsOf θ F).io u, V.Opens θ F c g (Xw g))
    (hu : u ∈ Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) Xe) :
    ∃ g ∈ (V.partsOf θ F).io u, ∃ i < (V.circuitOf θ F).w g, (Xw g).getLsbD i = !(Xe g).getLsbD i :=
  Majority.unbound_drawn (V.partsOf θ F) (V.Opens θ F c) hc ho hu

/-- **One sample's share**: at the sample `ξ`, the probability over the session's tape and coins that the recursive
configuration's verdict holds is at most the sample's miss of the unbound units (one where `ξ` misses them), plus
the knowledge-soundness term at its statement (`Dep.ksAt`: `RecursiveKSAt`'s `BitsLin2On`, and at the deployment's
tables the least of it and its `BitsLin2KOn` with `Dep.kappa_eps_oneTable`), plus the probability that the run accepts
while the extractor's witness differs from the plurality at a bit of a wire of a sampled proof block (`offEvent`). -/
theorem pr_verdict_le (q : Dep) (hG : AccOfAccepts q)
    (hKS : Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR q.wE q.w (deployedB q.b)
      q.aux q.nOf q.hd q.recRest)
    (hdecI : Claims.Recursion.decOK q.Bd.deploy q.decI)
    (hdecO : Claims.Recursion.decOK (deployZkRec _root_.Flock.Readable.relation) q.decO)
    (nI vS : Msg → ℕ) (lam : ℝ)
    (hn : ∀ x, q.RunAccepts x → ∀ cs ws, (q.D.innerCalls FlockDeploy.sha512 x cs ws).length ≤ nI x)
    (hv : ∀ x, q.RunAccepts x → ∀ ps,
      (Claims.Recursion.setupCalls FlockDeploy.sha512 q.D (q.D.vstar x ps)).length ≤ vS x)
    (hlam : ∀ x, q.RunAccepts x → lam ≤ Claims.Recursion.lamLin2 q.D q.I FlockDeploy.KS.CpDep x (nI x) (vS x))
    (hcap : ∀ x, q.RunAccepts x → q.OneTable x →
      (innerSetupIns (xIOf x)).length ≤ 2 ^ 32 ∧ nI x ≤ 2 ^ 32 ∧ vS x ≤ nS x * 138543104)
    (V : Vocab d α) (k : Verity.Registration.Composite) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (hcr : ∀ ξ, q.SessionCR (nI (S.x ξ)) (vS (S.x ξ)) (S.x ξ) S.Ω (S.P ξ))
    (hread : ∀ ξ wt, q.R (S.x ξ) wt → ∀ u ∈ S.drawOf ξ,
      (V.partsOf θ F).Correct (readX (S.x ξ) wt) u ∧
        ∀ g ∈ (V.partsOf θ F).io u, V.Opens θ F c g (readX (S.x ξ) wt g))
    (hlive : ∀ ξ ω r, LiveCoins q.Bd.IVStar q.st (S.x ξ)
      (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) ω r).seen)
    (ξ : S.Ξ) :
    Pr (fun s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) => RecVerdict V k (S.out ξ s)) ≤
      (if Disjoint (S.drawOf ξ) (Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (q.extract V S readX)) then 1
        else 0) + q.ksAt V S lam (S.x ξ) +
      Pr (fun s : q.Sample V S ξ => ∃ u, offEvent q V S readX (q.extract V S readX) u ξ s) := by
  classical
  set Xe := q.extract V S readX with hXdef
  set U := Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) Xe with hUdef
  have hsum0 : 0 ≤ Pr (fun s : q.Sample V S ξ => ∃ u, offEvent q V S readX Xe u ξ s) := Pr_nonneg _
  have hc0 : 0 ≤ q.ksAt V S lam (S.x ξ) := by
    unfold Dep.ksAt
    split_ifs
    · exact le_min (by positivity) (by positivity)
    · positivity
  by_cases hex : ∃ s, RecVerdict V k (S.out ξ s)
  swap
  · push_neg at hex
    have h0 : Pr (fun s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) => RecVerdict V k (S.out ξ s)) = 0 :=
      (le_antisymm (le_trans (Pr_mono fun s hs => hex s hs) (le_of_eq Pr_false)) (Pr_nonneg _))
    rw [h0]
    have : (0 : ℝ) ≤ if Disjoint (S.drawOf ξ) U then 1 else 0 := by split_ifs <;> norm_num
    linarith
  obtain ⟨s₀, hs₀⟩ := hex
  have hdense := (verdict_run q hG V k S hlive ξ s₀ hs₀).1
  have hrun : q.RunAccepts (S.x ξ) := ⟨S.Ω, S.P ξ, s₀.1, s₀.2, (verdict_run q hG V k S hlive ξ s₀ hs₀).2⟩
  by_cases hD : Disjoint (S.drawOf ξ) U
  · rw [if_pos hD]
    have := Pr_le_one (fun s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) => RecVerdict V k (S.out ξ s))
    linarith
  rw [if_neg hD, zero_add]
  -- `RecursiveKSAt` at the session's prover: `PrBad ≤ t' · 2^{-λ}`, and `≤ κ + t' · ε` at the deployment's tables
  have hcrξ := hcr ξ
  have hlamB : Pr (fun s : q.Sample V S ξ =>
      (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc ∧
        ¬ q.R.ext (S.x ξ) (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1) ≤
      q.tCost V S * (2 : ℝ) ^ (-lam) := by
    have hB := Claims.Recursion.bitsLin2On_of_ksLin2On ?hS hKS
    case hS => exact RecDeploy.sessionAccepts _ _ _
    have h1 := hB q.decI hdecI q.decO hdecO (S.x ξ) hdense S.Ω (S.P ξ) (q.tCost V S) (le_max_left _ _)
      (fun ω r => le_trans (S.calls_le ξ ω r) (le_max_right _ _)) (nI (S.x ξ)) (hn _ hrun) (vS (S.x ξ))
      (hv _ hrun) hcrξ.1 hcrξ.2.1 hcrξ.2.2.1 hcrξ.2.2.2.1 hcrξ.2.2.2.2
    refine le_trans h1 (mul_le_mul_of_nonneg_left ?_ (Nat.cast_nonneg _))
    exact Real.rpow_le_rpow_of_exponent_le (by norm_num) (neg_le_neg (hlam _ hrun))
  have hbad : Pr (fun s : q.Sample V S ξ =>
      (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc ∧
        ¬ q.R.ext (S.x ξ) (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1) ≤
      q.ksAt V S lam (S.x ξ) := by
    unfold Dep.ksAt
    split_ifs with h1t
    · refine le_min ?_ hlamB
      have hB := Claims.Recursion.bitsLin2KOn_of_ksLin2On ?hS hKS
      case hS => exact RecDeploy.sessionAccepts _ _ _
      have h1 := hB q.decI hdecI q.decO hdecO (S.x ξ) hdense S.Ω (S.P ξ) (q.tCost V S) (le_max_left _ _)
        (fun ω r => le_trans (S.calls_le ξ ω r) (le_max_right _ _)) (nI (S.x ξ)) (hn _ hrun) (vS (S.x ξ))
        (hv _ hrun) hcrξ.1 hcrξ.2.1 hcrξ.2.2.1 hcrξ.2.2.2.1 hcrξ.2.2.2.2
      obtain ⟨hSc, hnc, hvc⟩ := hcap _ hrun h1t
      obtain ⟨hκ, hε⟩ := q.kappa_eps_oneTable h1t hSc hnc hvc
      exact le_trans h1 (add_le_add hκ (mul_le_mul_of_nonneg_left hε (Nat.cast_nonneg _)))
    · exact hlamB
  -- the verdict's event, over the run and the extractor's seed
  have hlift : Pr (fun s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) => RecVerdict V k (S.out ξ s)) =
      Pr (fun s : q.Sample V S ξ => RecVerdict V k (S.out ξ (s.1, s.2.1))) :=
    (Claims.Pr_drop (γ := q.E.C (S.x ξ)) (fun ω r => RecVerdict V k (S.out ξ (ω, r)))).symm
  rw [hlift]
  obtain ⟨u₀, hu₀d, hu₀U⟩ := Finset.not_disjoint_iff.1 hD
  calc Pr (fun s : q.Sample V S ξ => RecVerdict V k (S.out ξ (s.1, s.2.1)))
      ≤ Pr (fun s : q.Sample V S ξ =>
          ((Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc ∧
            ¬ q.R.ext (S.x ξ) (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1) ∨
          ∃ u, offEvent q V S readX Xe u ξ s) := by
        refine Pr_mono fun s hs => ?_
        have hacc := (verdict_run q hG V k S hlive ξ (s.1, s.2.1) hs).2
        by_cases hR : q.R.ext (S.x ξ) (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1
        · right
          cases hw : (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1 with
          | none => rw [hw] at hR; exact (show False from hR).elim
          | some wt =>
              rw [hw] at hR
              obtain ⟨hcu, hou⟩ := hread ξ wt hR u₀ hu₀d
              obtain ⟨g, hg, i, hi, hbit⟩ := unbound_drawn V hcu hou hu₀U
              exact ⟨u₀, hacc, hu₀d, wt, hw, hR, g, hg, i, hi, hbit⟩
        · exact Or.inl ⟨hacc, hR⟩
    _ ≤ Pr (fun s : q.Sample V S ξ =>
          (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc ∧
            ¬ q.R.ext (S.x ξ) (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1) +
        Pr (fun s : q.Sample V S ξ => ∃ u, offEvent q V S readX Xe u ξ s) := pr_or_le _ _
    _ ≤ q.ksAt V S lam (S.x ξ) + Pr (fun s : q.Sample V S ξ => ∃ u, offEvent q V S readX Xe u ξ s) :=
        add_le_add hbad le_rfl

/-- **The bound at one coupling**: averaged over the samples, the probability that the recursive configuration's verdict
holds is at most the sample's miss `δs`, plus the knowledge-soundness term averaged over the samples (`Dep.ksAt`), plus
`√(min(W, 2n) · E[bindBudget] / 2^256)`, with `W` the file's wire bits and `n` its units (`ex_off_le_bits`,
`ex_off_le_units`). -/
theorem ex_verdict_le (q : Dep) (hG : AccOfAccepts q)
    (hKS : Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR q.wE q.w (deployedB q.b)
      q.aux q.nOf q.hd q.recRest)
    (hdecI : Claims.Recursion.decOK q.Bd.deploy q.decI)
    (hdecO : Claims.Recursion.decOK (deployZkRec _root_.Flock.Readable.relation) q.decO)
    (nI vS : Msg → ℕ) (lam : ℝ)
    (hn : ∀ x, q.RunAccepts x → ∀ cs ws, (q.D.innerCalls FlockDeploy.sha512 x cs ws).length ≤ nI x)
    (hv : ∀ x, q.RunAccepts x → ∀ ps,
      (Claims.Recursion.setupCalls FlockDeploy.sha512 q.D (q.D.vstar x ps)).length ≤ vS x)
    (hlam : ∀ x, q.RunAccepts x → lam ≤ Claims.Recursion.lamLin2 q.D q.I FlockDeploy.KS.CpDep x (nI x) (vS x))
    (hcap : ∀ x, q.RunAccepts x → q.OneTable x →
      (innerSetupIns (xIOf x)).length ≤ 2 ^ 32 ∧ nI x ≤ 2 ^ 32 ∧ vS x ≤ nS x * 138543104)
    (V : Vocab d α) (k : Verity.Registration.Composite) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (hcr : q.FinderCR V S collOf dColl nI vS) (δs : ℝ)
    (hdraw : Pr (fun ξ : S.Ξ =>
      Disjoint (S.drawOf ξ) (Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (q.extract V S readX))) ≤ δs)
    (hread : ∀ ξ wt, q.R (S.x ξ) wt → ∀ u ∈ S.drawOf ξ,
      (V.partsOf θ F).Correct (readX (S.x ξ) wt) u ∧
        ∀ g ∈ (V.partsOf θ F).io u, V.Opens θ F c g (readX (S.x ξ) wt g))
    (hcoll : ∀ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf θ F).N), q.R (S.x ξ₁) w₁ → q.R (S.x ξ₂) w₂ →
      (∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf θ F).io u) → (∃ u ∈ S.drawOf ξ₂, g ∈ (V.partsOf θ F).io u) →
      readX (S.x ξ₁) w₁ g ≠ readX (S.x ξ₂) w₂ g →
      ∃ m m', collOf (S.x ξ₁) w₁ (S.x ξ₂) w₂ = some (m, m') ∧ Claims.Collision FlockDeploy.sha512 m m')
    (hlive : ∀ ξ ω r, LiveCoins q.Bd.IVStar q.st (S.x ξ)
      (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) ω r).seen) :
    Ex (fun ξ => Pr fun s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) => RecVerdict V k (S.out ξ s)) ≤
      δs + Ex (fun ξ => q.ksAt V S lam (S.x ξ)) +
        Real.sqrt (min ((∑ g, (V.circuitOf θ F).w g : ℕ) : ℝ) (2 * V.units θ F) *
          (Ex (fun p : S.Ξ × S.Ξ => (q.bindBudget V S dColl p.1 p.2 : ℝ)) / 2 ^ (256 : ℝ))) := by
  classical
  refine le_trans (Ex_mono fun ξ => pr_verdict_le q hG hKS hdecI hdecO nI vS lam hn hv hlam hcap V k S readX hcr.1
    hread hlive ξ) ?_
  rw [ex_add3, ex_ind]
  refine add_le_add (add_le_add hdraw le_rfl) ?_
  rcases le_total ((∑ g, (V.circuitOf θ F).w g : ℕ) : ℝ) (2 * V.units θ F) with h | h
  · rw [min_eq_left h]
    exact ex_off_le_bits q V S readX collOf dColl hcoll hcr.2
  · rw [min_eq_right h]
    exact ex_off_le_units q V S readX collOf dColl hcoll hcr.2

/-- **The target: sampled proofs at the recursive configuration**, in the per-prover form over the causal game (R4′,
R5, R6), with the recursive configuration's verdict (`RecVerdict`). At composite `k`, the parameters `θ`, the
committed files `F`, the commitment `c` and a strategy `σ`, with `S` its recursive session (`SessionIn`), the
probability that the recursive component accepts while sampled proofs' claim fails on the extractor's transcript
(`¬ Holds k .sampled`) is at most `Dep.errAt` at `S`, the shared `condError FinderCR δ`: where `FinderCR` holds at
the finders built from `σ`, `δs + E_ξ[ksAt] + √(min(bits, 2n) · E[bindBudget(σ)] / 2^256)` with `n` the units,
and `⊤` elsewhere, where `ksAt` is `t'(σ) · 2^{-λ}`, and at a sample whose statement is at the deployment's tables the
least of it and `2^{-167.7} + t'(σ) · 2^{-174.95}` (`Dep.ksAt`), with `t'(σ)` the strategy's SHA-512 calls in one run
(`Vocab.cost`) and every finder's budget defined from it. There is no hypothesis on the strategy but `live-verifier`.

It takes these hypotheses:
* `RecursiveKSAt` at the deployed brings (`hKS`);
* the decoders meet the list-decoding spec (`hdecI`, `hdecO`);
* the inner and setup call bounds `nI`, `vS` and the level `λ` (`hn`, `hv`, `hlam`), and at the deployment's tables
  the inner setup's calls and `nI` at most `2^32` and `vS` at most `nS · 138543104`, `setupCalls_deployedB_le_cap`'s
  bound (`hcap`), at the statements some run of the deployed recursive verifier accepts (`Dep.RunAccepts`);
* `K₀` covers the claim's `ε` share (`hK₀`), and the session's sample misses any `K₀` units with probability at most `δs`
  (`hdraw`);
* the vocabulary's extractor is the recursive configuration's (`hX`, `Dep.extract` at `S`);
* the deployment's reading `readX` of a witness in the recursive relation is correct at, and opened by the commitment
  on, every unit the statement samples (`hread`); and `collOf`, from two such witnesses that read a wire of units sampled
  in both differently, outputs a SHA-512 collision (`hcoll`);
* the records hold the run's live coins (`LiveCoins`, `live-verifier`). `live-verifier` is admissible until the
  component reads the coins through the challenger's live-coin part's signed device log; then this hypothesis goes.

The proof, at a strategy whose finders meet `FinderCR` (elsewhere the error is `⊤`), where the files pass the commit
phase's check (elsewhere no play is accepted) and the claim fails (elsewhere the event is empty), is `ex_verdict_le`
through the coupling's `prob_le`, at `acc_of_accepts`' property. -/
theorem spotRecursive (q : Dep)
    (hKS : Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR q.wE q.w (deployedB q.b)
      q.aux q.nOf q.hd q.recRest)
    (hdecI : Claims.Recursion.decOK q.Bd.deploy q.decI)
    (hdecO : Claims.Recursion.decOK (deployZkRec _root_.Flock.Readable.relation) q.decO)
    (nI vS : Msg → ℕ) (lam : ℝ)
    (hn : ∀ x, q.RunAccepts x → ∀ cs ws, (q.D.innerCalls FlockDeploy.sha512 x cs ws).length ≤ nI x)
    (hv : ∀ x, q.RunAccepts x → ∀ ps,
      (Claims.Recursion.setupCalls FlockDeploy.sha512 q.D (q.D.vstar x ps)).length ≤ vS x)
    (hlam : ∀ x, q.RunAccepts x → lam ≤ Claims.Recursion.lamLin2 q.D q.I FlockDeploy.KS.CpDep x (nI x) (vS x))
    (hcap : ∀ x, q.RunAccepts x → q.OneTable x →
      (innerSetupIns (xIOf x)).length ≤ 2 ^ 32 ∧ nI x ≤ 2 ^ 32 ∧ vS x ≤ nS x * 138543104)
    (V : Vocab d α) (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl K₀ : ℕ) (δs : ℝ)
    (hK₀ : ∀ m : ℕ, (V.certOf θ).ε * V.units θ F < m → K₀ ≤ m)
    (hdraw : ∀ U : Finset (Fin (V.units θ F)), K₀ ≤ U.card → Claims.Pr (fun ξ : S.Ξ => Disjoint (S.drawOf ξ) U) ≤ δs)
    (hX : V.extract θ F c σ = q.extract V S readX)
    (hread : ∀ ξ wt, V.Registered θ F → q.R (S.x ξ) wt → ∀ u ∈ S.drawOf ξ,
      (V.partsOf θ F).Correct (readX (S.x ξ) wt) u ∧ ∀ g ∈ (V.partsOf θ F).io u, V.Opens θ F c g (readX (S.x ξ) wt g))
    (hcoll : ∀ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf θ F).N), q.R (S.x ξ₁) w₁ → q.R (S.x ξ₂) w₂ →
      (∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf θ F).io u) → (∃ u ∈ S.drawOf ξ₂, g ∈ (V.partsOf θ F).io u) →
      readX (S.x ξ₁) w₁ g ≠ readX (S.x ξ₂) w₂ g →
      ∃ m m', collOf (S.x ξ₁) w₁ (S.x ξ₂) w₂ = some (m, m') ∧ Claims.Collision FlockDeploy.sha512 m m')
    (hlive : ∀ ξ ω r, LiveCoins q.Bd.IVStar q.st (S.x ξ)
      (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) ω r).seen) :
    Game.prob (fun o => RecVerdict V k o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ ≤
      q.errAt V S collOf dColl nI vS δs lam := by
  classical
  unfold Dep.errAt
  split_ifs with hcr
  swap
  · exact le_top
  -- where the claim holds, the event is empty
  by_cases hH : V.Holds k θ F c σ .sampled ⟨θ, F, c, .empty, [], .empty, [], [], .empty, #[]⟩
  · refine le_trans (Game.prob_mono (q := fun _ => False) (fun o ho => ho.2 hH) _ σ) ?_
    rw [Game.prob_false]
    exact zero_le
  -- where the files fail the commit phase's check, no play is accepted
  by_cases hreg : V.Registered θ F
  swap
  · rw [show Game.prob (fun o => RecVerdict V k o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ = 0 from
      prob_play_eq_zero (V.families θ) (D := V.signer θ) θ F c _ (fun _ h3 h => hreg (h3.1 ▸ h3.2.1 ▸ h.1.1)) σ]
    exact zero_le
  -- more than an `ε` share of the units is unbound in the extracted transcript
  have hU : K₀ ≤ (Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (q.extract V S readX)).card := by
    refine hK₀ _ ?_
    have hH' : ¬ ((Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (V.extract θ F c σ)).card : ℝ) ≤
        (V.certOf θ).ε * V.units θ F := hH
    rw [hX] at hH'
    exact not_le.mp hH'
  calc Game.prob (fun o => RecVerdict V k o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ
      ≤ ENNReal.ofReal (Ex fun ξ => Pr fun s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) =>
          (fun o => RecVerdict V k o ∧ ¬ V.Holds k θ F c σ .sampled o) (S.out ξ s)) := S.prob_le _
    _ ≤ ENNReal.ofReal (Ex fun ξ => Pr fun s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) =>
          RecVerdict V k (S.out ξ s)) :=
        ENNReal.ofReal_le_ofReal (Ex_mono fun ξ => Pr_mono fun s hs => hs.1)
    _ ≤ q.delta V S dColl δs lam := by
        unfold Dep.delta
        refine ENNReal.ofReal_le_ofReal ?_
        exact ex_verdict_le q (accOfAccepts q) hKS hdecI hdecO nI vS lam hn hv hlam hcap V k S readX collOf dColl hcr
          δs (hdraw _ hU) (fun ξ wt hR u hu => hread ξ wt hreg hR u hu) hcoll hlive

end Verity.Recursive
