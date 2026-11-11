module

public import Proofs.Flock.Soundness.Audit.Extraction
public import Proofs.Flock.Soundness.LinkCarry

/-!
# The sampled audit as a compiler, generic in the proof system: its objects

The sampled audit compiles a proof system for proof blocks and a sampling design into an audit (`Audit.audit`): the
prover registers its commitment to the trace, the verifier samples proof blocks by the sampling design with live coins
after the registration, and the proof system's session proves the sampled proof blocks against the registration. This
file defines what the compiler takes and what its theorem reads; `Statements.lean` states the theorem.

* `Binding`: the registration's binding commitment to the trace: positions, the value behind each, openings, and
  binding as a reduction to the hash (two openings of one commitment with different values are a collision).
* `ProofSystem`: the proof system's session on each sample against each registration.
* `incorrect`: the incorrect set `F` of a trace, for the proof blocks' correctness the statement reads.
* `Extractor`: the knowledge extractor's guarantee at a sampled proof block, as two terms (`ks`: extraction fails;
  `link`: it succeeds, but opens a position the proof block reads to a value other than the trace's) that cover
  acceptance at every sampled proof block incorrect in the trace (`Extractor.Covers`). `Extractor.ksAvg` is the
  knowledge term the audit pays: at one sampled proof block, the worst, averaged over the sample.
* `Trials`: the extractor rerun for one position, which defines the committed trace (a plurality of what the trials read
  there, `Trials.Plurality`), the link finders (`Trials.CR`, ecr/sha-512 at them) and the binding term
  (`Trials.bindErr`, `ε_bind`), with the transfer from the binding error to the trials (`Trials.Transfers`).
* `JointExtractor`: the textbook form of knowledge soundness, an extractor that is a game played after the judged session,
  with a relation on witnesses (`JointExtractor.KnowledgeSound`). It gives an `Extractor` and `Trials` with no slack
  (`toExtractor`, `toTrials`; `Flock.Guarantees.GenericSampledKSJoint`).
* `Exp`, `Close`, `Zk`, `Hider`, `auditView`, `auditSim`: the audit's zero knowledge, from the proof system's and the
  commitment's hiding.
-/

@[expose] public section

namespace FlockSoundness.SampledGeneric

open Game Audit LinkFinder
open scoped ENNReal

/-! ## The commitment to the trace -/

/-- **The registration's binding commitment to the trace**: its positions (`Pos`), the value behind each (`Val`),
openings with their value and commitment, the commitment a registration holds at each position (`cs`), and binding as
a reduction to the hash `Hc`: two openings of one commitment with different values give a collision (`collide`). -/
structure Binding (Reg : Type) {Dh : Type} (Hc : List UInt8 → Dh) where
  Pos : Type
  [finPos : Fintype Pos]
  [decPos : DecidableEq Pos]
  Val : Type
  [finVal : Fintype Val]
  [decVal : DecidableEq Val]
  [inhVal : Inhabited Val]
  Opening : Type
  [inhOp : Inhabited Opening]
  Com : Type
  val : Opening → Val
  com : Opening → Com
  cs : Reg → Pos → Com
  collide : Opening → Opening → List UInt8 × List UInt8
  collide_spec : ∀ o o', com o = com o' → val o ≠ val o' →
    (collide o o').1 ≠ (collide o o').2 ∧ Hc (collide o o').1 = Hc (collide o o').2

attribute [instance] Binding.finPos Binding.decPos Binding.finVal Binding.decVal Binding.inhVal Binding.inhOp

variable {n : ℕ} {Reg : Type} {Dh : Type} {Hc : List UInt8 → Dh}

/-- **A trace**: a value at each position of the commitment. -/
abbrev Binding.Trace (B : Binding Reg Hc) : Type := B.Pos → B.Val

/-! ## The proof system and its extractor -/

/-- **A proof system for proof blocks**, as the compiler runs it: its session on each sample of the `n` proof blocks
against each registration, which accepts or rejects. The session may prove the sample at once (Flock's batched
session) or run one session per sampled proof block. -/
structure ProofSystem (n : ℕ) (Reg : Type) where
  session : Finset (Fin n) → Reg → Game Bool

variable {B : Binding Reg Hc}

open Classical in
/-- **The incorrect set `F`** of the trace `X` at the registration `R`: the proof blocks not correct in it, for the
proof blocks' correctness `Correct` the audit's statement reads. -/
noncomputable def incorrect (B : Binding Reg Hc) (Correct : Reg → B.Trace → Fin n → Prop) (R : Reg) (X : B.Trace) : Finset (Fin n) :=
  Finset.univ.filter fun u => ¬ Correct R X u

/-- **The knowledge extractor's guarantee at a sampled proof block**: from a prover's session strategy `τ` on the sample
`S` against the registration `R`, `ks S R τ u` bounds `Pr[the session accepts ∧ extraction of u's witness fails]`, and
`link S R τ u X` bounds `Pr[it accepts ∧ the extracted witness opens a position u reads to a value other than X's]`.
Each may depend on the prover (its finders' advantages). -/
structure Extractor (PS : ProofSystem n Reg) (B : Binding Reg Hc) where
  ks : (S : Finset (Fin n)) → (R : Reg) → Strategy (PS.session S R) → Fin n → ℝ≥0∞
  link : (S : Finset (Fin n)) → (R : Reg) → Strategy (PS.session S R) → Fin n → B.Trace → ℝ≥0∞

namespace Extractor

variable {PS : ProofSystem n Reg} (Ex : Extractor PS B)

/-- **The two terms cover acceptance**: at every sampled proof block incorrect in a trace, the session accepts with
probability at most the knowledge term plus the binding error against that trace, from every session strategy. -/
def Covers (Correct : Reg → B.Trace → Fin n → Prop) : Prop :=
  ∀ S R (τ : Strategy (PS.session S R)) u (X : B.Trace), u ∈ S → ¬ Correct R X u →
    prob (· = true) (PS.session S R) τ ≤ Ex.ks S R τ u + Ex.link S R τ u X

/-- **The knowledge term the audit pays**: on each sample, the largest knowledge term over its proof blocks, averaged
over the sample. The audit charges it at one sampled proof block, fixed before the session's first coin. -/
noncomputable def ksAvg (L : Law n) (R : Reg) (τ : Cont L Reg PS.session R) : ℝ≥0∞ :=
  avg fun ω => ⨆ u ∈ L.draw ω, Ex.ks (L.draw ω) R (τ ω) u

end Extractor

/-! ## The trials: the committed trace and the binding term -/

/-- **The extractor's trials** at the sampling design `L`. At a registration `R`, a prover state `τ` and a position `c`
that some sample reads (`Read c`), a game the prover's state plays (`T`, by `sT`): a sample that reads `c`, a fresh
session and the extractor's reruns. It reads values at `c` (`rd`), each with an opening (`op`), and costs `cost` SHA-512
calls. `wt c` weighs `c` (the probability that a sample reads it), and `ρ` and `slack` are what the extractor loses to its
waits (both `0` for an extractor that is a game, `JointExtractor.toTrials`). -/
structure Trials (L : Law n) (PS : ProofSystem n Reg) (B : Binding Reg Hc) where
  Out : Type
  Read : B.Pos → Prop
  T : (R : Reg) → Cont L Reg PS.session R → (c : B.Pos) → Read c → Game Out
  sT : (R : Reg) → (τ : Cont L Reg PS.session R) → (c : B.Pos) → (h : Read c) → Strategy (T R τ c h)
  rd : (R : Reg) → Cont L Reg PS.session R → B.Pos → Out → Finset B.Val
  op : (R : Reg) → Cont L Reg PS.session R → B.Pos → Out → B.Val → B.Opening
  cost : (R : Reg) → Cont L Reg PS.session R → Out → ℝ
  wt : B.Pos → ℝ
  ρ : ℝ
  slack : ℝ

namespace Trials

variable {L : Law n} {PS : ProofSystem n Reg} (Tr : Trials L PS B)

/-- **Each reading comes with an opening of the registered commitment** to the value read. -/
def Opens : Prop :=
  ∀ R τ c e v, v ∈ Tr.rd R τ c e → B.com (Tr.op R τ c e v) = B.cs R c ∧ B.val (Tr.op R τ c e v) = v

/-- **The trials' numbers are in range**: costs and weights nonnegative, `ρ < 1`, and the slack nonnegative. -/
def WellFormed : Prop :=
  (∀ R τ e, 0 ≤ Tr.cost R τ e) ∧ (∀ c, 0 ≤ Tr.wt c) ∧ Tr.ρ < 1 ∧ 0 ≤ Tr.slack

open Classical in
/-- **The binding error at `c` against the value `x`**: the probability that the trial reads two values or one other
than `x` there (`LinkFinder.errAt`), and `0` at a position no sample reads. -/
noncomputable def errOf (R : Reg) (τ : Cont L Reg PS.session R) (c : B.Pos) (x : B.Val) : ℝ :=
  if h : Tr.Read c then errAt (Tr.T R τ c h) (Tr.sT R τ c h) (Tr.rd R τ c) x else 0

/-- **The transfer from the binding error to the trials**: for every prover state, every rule that picks a sampled proof
block before the session's first coin and every trace `X`, the binding error at the picked proof block, averaged over the
sample, is at most `Σ_c wt_c (errOf_c(X c) + slack)/(1 − ρ)`. -/
def Transfers (Ex : Extractor PS B) : Prop :=
  ∀ (R : Reg) (τ : Cont L Reg PS.session R) (tgt : L.Ω → Option (Fin n)) (X : B.Trace),
    avg (fun ω => atTgt (L.draw ω) (tgt ω) fun u => Ex.link (L.draw ω) R (τ ω) u X) ≤
      ENNReal.ofReal ((∑ c, Tr.wt c * (Tr.errOf R τ c (X c) + Tr.slack)) / (1 - Tr.ρ))

/-- **The committed trace is a plurality of the trials**: at each position a sample reads, no value is read alone more
often than the trace's. The committed trace is a function of the prover's state at registration (`Xc R τ`), so it is
fixed before the sample; nobody computes it. -/
def Plurality (Xc : (R : Reg) → Cont L Reg PS.session R → B.Trace) : Prop :=
  ∀ R τ c (h : Tr.Read c) v,
    pRd (Tr.T R τ c h) (Tr.sT R τ c h) (Tr.rd R τ c) v ≤ pRd (Tr.T R τ c h) (Tr.sT R τ c h) (Tr.rd R τ c) (Xc R τ c)

open Classical in
/-- **The plurality**: at each position a sample reads, a value no other is read alone more often than; the
`Inhabited` value elsewhere. -/
noncomputable def plur (R : Reg) (τ : Cont L Reg PS.session R) : B.Trace := fun c =>
  if h : Tr.Read c then
    Classical.choose (Finset.exists_max_image Finset.univ
      (pRd (Tr.T R τ c h) (Tr.sT R τ c h) (Tr.rd R τ c)) Finset.univ_nonempty)
  else default

/-- **The link finder for `c`**: a trial, then up to `M` more until one reads anything (`LinkFinder.finderG`); its
output is a collision from two openings of `c` with different values (`LinkFinder.outOf`). -/
noncomputable def finderOut (R : Reg) (τ : Cont L Reg PS.session R) (c : B.Pos) :
    Tr.Out × List Tr.Out → Option (List UInt8 × List UInt8) :=
  outOf (Tr.rd R τ c) (Tr.op R τ c) B.collide

/-- **ecr/sha-512 at the prover's link finders** (`Assumptions.SHA512CRExpected`, named): at the registration `R` and the
prover's state `τ`, the finder for each position a sample reads finds a collision of `Hc` with probability at most its
expected SHA-512 calls over `2^256`. It is asked of one prover at a time: a prover that hard-codes two openings of one
commitment has finders that find one. -/
def CR (R : Reg) (τ : Cont L Reg PS.session R) (M : ℕ) : Prop :=
  ∀ c (h : Tr.Read c), Assumptions.SHA512CRExpected Hc (finderG (Tr.T R τ c h) M)
    (finderStrat _ (Tr.sT R τ c h) M) (Tr.finderOut R τ c) (finderCost (Tr.rd R τ c) (Tr.cost R τ))

open Classical in
/-- **A trial's expected cost at `c`** in SHA-512 calls, and `0` at a position no sample reads. -/
noncomputable def ecost (R : Reg) (τ : Cont L Reg PS.session R) (c : B.Pos) : ℝ :=
  if h : Tr.Read c then expect (Tr.cost R τ) (Tr.T R τ c h) (Tr.sT R τ c h) else 0

/-- **The binding term `ε_bind`**, prover by prover:
`Σ_c wt_c (2·E[cost_c]/2^256 + 1/(eM) + slack)/(1 − ρ)`, with `E[cost_c]` a trial's expected SHA-512 calls at `c`. -/
noncomputable def bindErr (R : Reg) (τ : Cont L Reg PS.session R) (M : ℕ) : ℝ≥0∞ :=
  ENNReal.ofReal ((∑ c, Tr.wt c * (2 * Tr.ecost R τ c / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + Tr.slack)) /
    (1 - Tr.ρ))

end Trials

/-! ## Knowledge soundness in the textbook form -/

/-- **A knowledge extractor as a game**: after the judged session on the sample `S` against `R`, the extractor's reruns,
played by a strategy built from the prover's (`strat`). The game outputs the judged session's verdict and a witness, or
none, for each sampled proof block; `cost` is its SHA-512 calls. `Rel S R u w`: `w` is a witness of `u`'s statement;
`reads u`: the positions `u` reads; `opens S R u w c`: the opening of `c` that `w` carries. -/
structure JointExtractor (PS : ProofSystem n Reg) (B : Binding Reg Hc) where
  Wit : Type
  Rel : Finset (Fin n) → Reg → Fin n → Wit → Prop
  reads : Fin n → Finset B.Pos
  opens : Finset (Fin n) → Reg → Fin n → Wit → B.Pos → B.Opening
  G : Finset (Fin n) → Reg → Game (Bool × (Fin n → Option Wit))
  strat : (S : Finset (Fin n)) → (R : Reg) → Strategy (PS.session S R) → Strategy (G S R)
  cost : Finset (Fin n) → Reg → Bool × (Fin n → Option Wit) → ℝ

namespace JointExtractor

variable {PS : ProofSystem n Reg} (J : JointExtractor PS B)

/-- **The game's first output is the judged session's verdict**: it accepts exactly as often as the session. -/
def Judged : Prop :=
  ∀ S R (τ : Strategy (PS.session S R)),
    prob (fun o => o.1 = true) (J.G S R) (J.strat S R τ) = prob (· = true) (PS.session S R) τ

/-- **A witness is sound for the statement**: at a sampled proof block, a witness of its statement opens the registered
commitment at every position the proof block reads, and the proof block is correct in every trace that agrees with
those openings. -/
def Sound (Correct : Reg → B.Trace → Fin n → Prop) : Prop :=
  ∀ S R u w, u ∈ S → J.Rel S R u w →
    (∀ c ∈ J.reads u, B.com (J.opens S R u w c) = B.cs R c) ∧
      ∀ X : B.Trace, (∀ c ∈ J.reads u, B.val (J.opens S R u w c) = X c) → Correct R X u

/-- **Extraction of `u` fails**: the game's output carries no witness of `u`'s statement. -/
def Fails (S : Finset (Fin n)) (R : Reg) (u : Fin n) (o : Bool × (Fin n → Option J.Wit)) : Prop :=
  ¬ ∃ w, o.2 u = some w ∧ J.Rel S R u w

/-- **Knowledge soundness at the knowledge error `κ`**, the textbook form: from every session strategy, at every
sampled proof block, `Pr[the judged session accepts ∧ extraction of u fails] ≤ κ`. `κ` may depend on the prover. -/
def KnowledgeSound (κ : (S : Finset (Fin n)) → (R : Reg) → Strategy (PS.session S R) → Fin n → ℝ≥0∞) : Prop :=
  ∀ S R (τ : Strategy (PS.session S R)) u, u ∈ S →
    prob (fun o => o.1 = true ∧ J.Fails S R u o) (J.G S R) (J.strat S R τ) ≤ κ S R τ u

/-- **The extractor's guarantee**: `ks` is `Pr[accept ∧ extraction of u fails]`, `link` is `Pr[accept ∧ the witness
extracted for u opens a position u reads to a value other than X's]`. -/
noncomputable def toExtractor : Extractor PS B where
  ks S R τ u := prob (fun o => o.1 = true ∧ J.Fails S R u o) (J.G S R) (J.strat S R τ)
  link S R τ u X := prob (fun o => o.1 = true ∧ ∃ w, o.2 u = some w ∧ J.Rel S R u w ∧
    ∃ c ∈ J.reads u, B.val (J.opens S R u w c) ≠ X c) (J.G S R) (J.strat S R τ)

/-- The samples that read `c`. -/
abbrev ReadsAt (L : Law n) (c : B.Pos) : Type := {ω : L.Ω // ∃ u ∈ L.draw ω, c ∈ J.reads u}

open Classical in
noncomputable instance instFintypeReadsAt (L : Law n) (c : B.Pos) : Fintype (J.ReadsAt L c) := Subtype.fintype _

open Classical in
/-- What the extractor's game, after sample `ω`, reads at `c`: the values the witnesses it extracted for the sampled
proof blocks that read `c` open `c` to, when the judged session accepted. -/
noncomputable def readsOut (L : Law n) (R : Reg) (c : B.Pos) (e : L.Ω × (Bool × (Fin n → Option J.Wit))) :
    Finset B.Val :=
  Finset.univ.filter fun v => e.2.1 = true ∧ ∃ u ∈ L.draw e.1, c ∈ J.reads u ∧
    ∃ w, e.2.2 u = some w ∧ J.Rel (L.draw e.1) R u w ∧ B.val (J.opens (L.draw e.1) R u w c) = v

/-- **The trials of an extractor that is a game**: a sample that reads `c`, uniform among those, then the extractor's
game against the prover's session strategy on it; it reads what `readsOut` reads, at the weight `wt c` the probability
that a sample reads `c`, with no slack (`ρ = 0`, `slack = 0`). -/
noncomputable def toTrials (L : Law n) : Trials L PS B where
  Out := L.Ω × (Bool × (Fin n → Option J.Wit))
  Read c := Nonempty (J.ReadsAt L c)
  T R _ c h := haveI : Nonempty (J.ReadsAt L c) := h
    (Game.draw (J.ReadsAt L c)).bind fun w => (J.G (L.draw w.1) R).map fun o => (w.1, o)
  sT R τ c h := haveI : Nonempty (J.ReadsAt L c) := h
    Strategy.bind _ _ (drawStrat _) fun w => Strategy.map _ _ (J.strat (L.draw w.1) R (τ w.1))
  rd R _ c e := J.readsOut L R c e
  op R _ c e v := Classical.epsilon fun o => ∃ u ∈ L.draw e.1, c ∈ J.reads u ∧
    ∃ w, e.2.2 u = some w ∧ J.Rel (L.draw e.1) R u w ∧ o = J.opens (L.draw e.1) R u w c ∧ B.val o = v
  cost R _ e := J.cost (L.draw e.1) R e.2
  wt c := (Fintype.card (J.ReadsAt L c) : ℝ) / Fintype.card L.Ω
  ρ := 0
  slack := 0

end JointExtractor

/-! ## Zero knowledge -/

/-- **A view experiment**: a game and the strategy that plays it (the honest prover, or a simulator's coins); its output
is the view. -/
structure Exp (V : Type) where
  g : Game V
  s : Strategy g

/-- `Pr[E]` in the experiment. -/
noncomputable def Exp.pr {V : Type} (e : Exp V) (E : V → Prop) : ℝ≥0∞ := prob E e.g e.s

/-- **Post-processing**: the experiment, then an experiment that reads its view. -/
def Exp.andThen {V W : Type} (e : Exp V) (f : V → Exp W) : Exp W :=
  ⟨e.g.bind fun v => (f v).g, Strategy.bind _ _ e.s fun v => (f v).s⟩

/-- **`ε`-close**: on every event the two experiments' probabilities differ by at most `ε`. -/
def Close {V : Type} (a b : Exp V) (ε : ℝ≥0∞) : Prop :=
  ∀ E : V → Prop, a.pr E ≤ b.pr E + ε ∧ b.pr E ≤ a.pr E + ε

/-- **A zero-knowledge proof system's views**, for statements `Stmt` with witnesses `Wit` against verifiers `VS`: the
honest prover's view from the witness (`real`), and the simulator's from the statement alone (`sim`). -/
structure Zk (Stmt Wit VS View : Type) where
  real : Stmt → Wit → VS → Exp View
  sim : Stmt → VS → Exp View

/-- **Zero knowledge at `ε_zk`**: for every statement and witness in the relation and every verifier, the honest
prover's view is `ε_zk`-close to the simulator's. -/
def Zk.ZK {Stmt Wit VS View : Type} (Z : Zk Stmt Wit VS View) (Rel : Stmt → Wit → Prop) (εzk : Stmt → ℝ≥0∞) : Prop :=
  ∀ x w, Rel x w → ∀ v, Close (Z.real x w v) (Z.sim x v) (εzk x)

/-- **A commitment to traces `Tr`** under the prover's uniform coins `Rnd`. -/
structure Hider (Tr Rnd Com : Type) where
  commit : Tr → Rnd → Com

namespace Hider

variable {Tr Rnd Com : Type} [Fintype Rnd] [Nonempty Rnd] (H : Hider Tr Rnd Com)

/-- The commitment to `X`, as an experiment over the prover's coins. -/
def exp (X : Tr) : Exp Com := ⟨.coin Rnd fun r => .ret (H.commit X r), fun _ => PUnit.unit⟩

/-- **Hiding at `ε_hide`** (named assumption): against every post-processing, the commitments to any two traces are
`ε_hide`-close. -/
def Hiding (εh : ℝ≥0∞) : Prop :=
  ∀ (W : Type) (X X' : Tr) (f : Com → Exp W), Close ((H.exp X).andThen f) ((H.exp X').andThen f) εh

end Hider

variable {Tr Rnd Com Wit VS View : Type} [Fintype Rnd] [Nonempty Rnd]

/-- **The audit's view against the verifier `v`**: the commitment to the trace `X` under the prover's coins, the sample
`v` takes after seeing it (`samp`), and the proof system's view of the session on that sample's statement, proved from
the trace's openings (`wit`). -/
def auditView (H : Hider Tr Rnd Com) (Z : Zk (Com × Finset (Fin n)) Wit VS View) (samp : VS → Com → Finset (Fin n))
    (wit : Tr → Rnd → Finset (Fin n) → Wit) (X : Tr) (v : VS) : Exp (Com × Finset (Fin n) × View) :=
  ⟨.coin Rnd fun r => (Z.real (H.commit X r, samp v (H.commit X r)) (wit X r (samp v (H.commit X r))) v).g.map
      fun w => (H.commit X r, samp v (H.commit X r), w),
    fun r => Strategy.map _ _ (Z.real (H.commit X r, samp v (H.commit X r)) (wit X r (samp v (H.commit X r))) v).s⟩

/-- **The audit's simulator against `v`**: a commitment to the fixed trace `X₀`, the sample `v` takes after it, and the
proof system's simulator on that sample's statement. -/
def auditSim (H : Hider Tr Rnd Com) (Z : Zk (Com × Finset (Fin n)) Wit VS View) (samp : VS → Com → Finset (Fin n))
    (X₀ : Tr) (v : VS) : Exp (Com × Finset (Fin n) × View) :=
  ⟨.coin Rnd fun r => (Z.sim (H.commit X₀ r, samp v (H.commit X₀ r)) v).g.map
      fun w => (H.commit X₀ r, samp v (H.commit X₀ r), w),
    fun r => Strategy.map _ _ (Z.sim (H.commit X₀ r, samp v (H.commit X₀ r)) v).s⟩

end FlockSoundness.SampledGeneric
