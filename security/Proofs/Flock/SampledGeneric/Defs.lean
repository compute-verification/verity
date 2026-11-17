module

public import Proofs.Flock.Soundness.Audit.Extraction
public import Proofs.Flock.Soundness.LinkCarry

/-! The objects the generic sampled audit compiles and its theorems read. -/

@[expose] public section

namespace FlockSoundness.SampledGeneric

open Game Audit LinkFinder
open scoped ENNReal

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

abbrev Binding.Trace (B : Binding Reg Hc) : Type := B.Pos → B.Val

structure ProofSystem (n : ℕ) (Reg : Type) where
  session : Finset (Fin n) → Reg → Game Bool

variable {B : Binding Reg Hc}

open Classical in
noncomputable def incorrect (B : Binding Reg Hc) (Correct : Reg → B.Trace → Fin n → Prop) (R : Reg) (X : B.Trace) : Finset (Fin n) :=
  Finset.univ.filter fun u => ¬ Correct R X u

/-- `ks` and `link` may depend on the prover (its finders' advantages). -/
structure Extractor (PS : ProofSystem n Reg) (B : Binding Reg Hc) where
  ks : (S : Finset (Fin n)) → (R : Reg) → Strategy (PS.session S R) → Fin n → ℝ≥0∞
  link : (S : Finset (Fin n)) → (R : Reg) → Strategy (PS.session S R) → Fin n → B.Trace → ℝ≥0∞

namespace Extractor

variable {PS : ProofSystem n Reg} (Ex : Extractor PS B)

def Covers (Correct : Reg → B.Trace → Fin n → Prop) : Prop :=
  ∀ S R (τ : Strategy (PS.session S R)) u (X : B.Trace), u ∈ S → ¬ Correct R X u →
    prob (· = true) (PS.session S R) τ ≤ Ex.ks S R τ u + Ex.link S R τ u X

/-- Charged at one sampled proof block, fixed before the session's first coin. -/
noncomputable def ksAvg (L : SamplingDesign n) (R : Reg) (τ : Cont L Reg PS.session R) : ℝ≥0∞ :=
  avg fun ω => ⨆ u ∈ L.draw ω, Ex.ks (L.draw ω) R (τ ω) u

end Extractor

/-- `ρ` and `slack` are what the extractor loses to its waits, both `0` for an extractor that is a game. -/
structure Trials (L : SamplingDesign n) (PS : ProofSystem n Reg) (B : Binding Reg Hc) where
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

variable {L : SamplingDesign n} {PS : ProofSystem n Reg} (Tr : Trials L PS B)

def Opens : Prop :=
  ∀ R τ c e v, v ∈ Tr.rd R τ c e → B.com (Tr.op R τ c e v) = B.cs R c ∧ B.val (Tr.op R τ c e v) = v

def WellFormed : Prop :=
  (∀ R τ e, 0 ≤ Tr.cost R τ e) ∧ (∀ c, 0 ≤ Tr.wt c) ∧ Tr.ρ < 1 ∧ 0 ≤ Tr.slack

open Classical in
noncomputable def errOf (R : Reg) (τ : Cont L Reg PS.session R) (c : B.Pos) (x : B.Val) : ℝ :=
  if h : Tr.Read c then errAt (Tr.T R τ c h) (Tr.sT R τ c h) (Tr.rd R τ c) x else 0

def Transfers (Ex : Extractor PS B) : Prop :=
  ∀ (R : Reg) (τ : Cont L Reg PS.session R) (tgt : L.Ω → Option (Fin n)) (X : B.Trace),
    avg (fun ω => atTgt (L.draw ω) (tgt ω) fun u => Ex.link (L.draw ω) R (τ ω) u X) ≤
      ENNReal.ofReal ((∑ c, Tr.wt c * (Tr.errOf R τ c (X c) + Tr.slack)) / (1 - Tr.ρ))

/-- The committed trace `Xc R τ` is fixed before the sample; nobody computes it. -/
def Plurality (Xc : (R : Reg) → Cont L Reg PS.session R → B.Trace) : Prop :=
  ∀ R τ c (h : Tr.Read c) v,
    pRd (Tr.T R τ c h) (Tr.sT R τ c h) (Tr.rd R τ c) v ≤ pRd (Tr.T R τ c h) (Tr.sT R τ c h) (Tr.rd R τ c) (Xc R τ c)

open Classical in
noncomputable def plur (R : Reg) (τ : Cont L Reg PS.session R) : B.Trace := fun c =>
  if h : Tr.Read c then
    Classical.choose (Finset.exists_max_image Finset.univ
      (pRd (Tr.T R τ c h) (Tr.sT R τ c h) (Tr.rd R τ c)) Finset.univ_nonempty)
  else default

noncomputable def finderOut (R : Reg) (τ : Cont L Reg PS.session R) (c : B.Pos) :
    Tr.Out × List Tr.Out → Option (List UInt8 × List UInt8) :=
  outOf (Tr.rd R τ c) (Tr.op R τ c) B.collide

/-- Asked of one prover at a time: a prover that hard-codes two openings of one commitment has finders that find one. -/
def CR (R : Reg) (τ : Cont L Reg PS.session R) (M : ℕ) : Prop :=
  ∀ c (h : Tr.Read c), Assumptions.SHA512CRExpected Hc (finderG (Tr.T R τ c h) M)
    (finderStrat _ (Tr.sT R τ c h) M) (Tr.finderOut R τ c) (finderCost (Tr.rd R τ c) (Tr.cost R τ))

open Classical in
noncomputable def ecost (R : Reg) (τ : Cont L Reg PS.session R) (c : B.Pos) : ℝ :=
  if h : Tr.Read c then expect (Tr.cost R τ) (Tr.T R τ c h) (Tr.sT R τ c h) else 0

noncomputable def bindErr (R : Reg) (τ : Cont L Reg PS.session R) (M : ℕ) : ℝ≥0∞ :=
  ENNReal.ofReal ((∑ c, Tr.wt c * (2 * Tr.ecost R τ c / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + Tr.slack)) /
    (1 - Tr.ρ))

end Trials

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

def Judged : Prop :=
  ∀ S R (τ : Strategy (PS.session S R)),
    prob (fun o => o.1 = true) (J.G S R) (J.strat S R τ) = prob (· = true) (PS.session S R) τ

def Sound (Correct : Reg → B.Trace → Fin n → Prop) : Prop :=
  ∀ S R u w, u ∈ S → J.Rel S R u w →
    (∀ c ∈ J.reads u, B.com (J.opens S R u w c) = B.cs R c) ∧
      ∀ X : B.Trace, (∀ c ∈ J.reads u, B.val (J.opens S R u w c) = X c) → Correct R X u

def Fails (S : Finset (Fin n)) (R : Reg) (u : Fin n) (o : Bool × (Fin n → Option J.Wit)) : Prop :=
  ¬ ∃ w, o.2 u = some w ∧ J.Rel S R u w

/-- `κ` may depend on the prover. -/
def KnowledgeSound (κ : (S : Finset (Fin n)) → (R : Reg) → Strategy (PS.session S R) → Fin n → ℝ≥0∞) : Prop :=
  ∀ S R (τ : Strategy (PS.session S R)) u, u ∈ S →
    prob (fun o => o.1 = true ∧ J.Fails S R u o) (J.G S R) (J.strat S R τ) ≤ κ S R τ u

noncomputable def toExtractor : Extractor PS B where
  ks S R τ u := prob (fun o => o.1 = true ∧ J.Fails S R u o) (J.G S R) (J.strat S R τ)
  link S R τ u X := prob (fun o => o.1 = true ∧ ∃ w, o.2 u = some w ∧ J.Rel S R u w ∧
    ∃ c ∈ J.reads u, B.val (J.opens S R u w c) ≠ X c) (J.G S R) (J.strat S R τ)

abbrev ReadsAt (L : SamplingDesign n) (c : B.Pos) : Type := {ω : L.Ω // ∃ u ∈ L.draw ω, c ∈ J.reads u}

open Classical in
noncomputable instance instFintypeReadsAt (L : SamplingDesign n) (c : B.Pos) : Fintype (J.ReadsAt L c) := Subtype.fintype _

open Classical in
noncomputable def readsOut (L : SamplingDesign n) (R : Reg) (c : B.Pos) (e : L.Ω × (Bool × (Fin n → Option J.Wit))) :
    Finset B.Val :=
  Finset.univ.filter fun v => e.2.1 = true ∧ ∃ u ∈ L.draw e.1, c ∈ J.reads u ∧
    ∃ w, e.2.2 u = some w ∧ J.Rel (L.draw e.1) R u w ∧ B.val (J.opens (L.draw e.1) R u w c) = v

noncomputable def toTrials (L : SamplingDesign n) : Trials L PS B where
  Out := L.Ω × (Bool × (Fin n → Option J.Wit))
  Read c := Nonempty (J.ReadsAt L c)
  T R _ c h := haveI : Nonempty (J.ReadsAt L c) := h
    (Game.draw (J.ReadsAt L c)).bind fun w => (J.G (L.draw w.1) R).map fun o => (w.1, o)
  sT R τ c h := haveI : Nonempty (J.ReadsAt L c) := h
    Strategy.bind _ _ (sampleStrat _) fun w => Strategy.map _ _ (J.strat (L.draw w.1) R (τ w.1))
  rd R _ c e := J.readsOut L R c e
  op R _ c e v := Classical.epsilon fun o => ∃ u ∈ L.draw e.1, c ∈ J.reads u ∧
    ∃ w, e.2.2 u = some w ∧ J.Rel (L.draw e.1) R u w ∧ o = J.opens (L.draw e.1) R u w c ∧ B.val o = v
  cost R _ e := J.cost (L.draw e.1) R e.2
  wt c := (Fintype.card (J.ReadsAt L c) : ℝ) / Fintype.card L.Ω
  ρ := 0
  slack := 0

end JointExtractor

structure Exp (V : Type) where
  g : Game V
  s : Strategy g

noncomputable def Exp.pr {V : Type} (e : Exp V) (E : V → Prop) : ℝ≥0∞ := prob E e.g e.s

def Exp.andThen {V W : Type} (e : Exp V) (f : V → Exp W) : Exp W :=
  ⟨e.g.bind fun v => (f v).g, Strategy.bind _ _ e.s fun v => (f v).s⟩

def Close {V : Type} (a b : Exp V) (ε : ℝ≥0∞) : Prop :=
  ∀ E : V → Prop, a.pr E ≤ b.pr E + ε ∧ b.pr E ≤ a.pr E + ε

structure Zk (Stmt Wit VS View : Type) where
  real : Stmt → Wit → VS → Exp View
  sim : Stmt → VS → Exp View

def Zk.ZK {Stmt Wit VS View : Type} (Z : Zk Stmt Wit VS View) (Rel : Stmt → Wit → Prop) (εzk : Stmt → ℝ≥0∞) : Prop :=
  ∀ x w, Rel x w → ∀ v, Close (Z.real x w v) (Z.sim x v) (εzk x)

structure Hider (Tr Rnd Com : Type) where
  commit : Tr → Rnd → Com

namespace Hider

variable {Tr Rnd Com : Type} [Fintype Rnd] [Nonempty Rnd] (H : Hider Tr Rnd Com)

def exp (X : Tr) : Exp Com := ⟨.coin Rnd fun r => .ret (H.commit X r), fun _ => PUnit.unit⟩

/-- A named assumption. -/
def Hiding (εh : ℝ≥0∞) : Prop :=
  ∀ (W : Type) (X X' : Tr) (f : Com → Exp W), Close ((H.exp X).andThen f) ((H.exp X').andThen f) εh

end Hider

variable {Tr Rnd Com Wit VS View : Type} [Fintype Rnd] [Nonempty Rnd]

def auditView (H : Hider Tr Rnd Com) (Z : Zk (Com × Finset (Fin n)) Wit VS View) (samp : VS → Com → Finset (Fin n))
    (wit : Tr → Rnd → Finset (Fin n) → Wit) (X : Tr) (v : VS) : Exp (Com × Finset (Fin n) × View) :=
  ⟨.coin Rnd fun r => (Z.real (H.commit X r, samp v (H.commit X r)) (wit X r (samp v (H.commit X r))) v).g.map
      fun w => (H.commit X r, samp v (H.commit X r), w),
    fun r => Strategy.map _ _ (Z.real (H.commit X r, samp v (H.commit X r)) (wit X r (samp v (H.commit X r))) v).s⟩

def auditSim (H : Hider Tr Rnd Com) (Z : Zk (Com × Finset (Fin n)) Wit VS View) (samp : VS → Com → Finset (Fin n))
    (X₀ : Tr) (v : VS) : Exp (Com × Finset (Fin n) × View) :=
  ⟨.coin Rnd fun r => (Z.sim (H.commit X₀ r, samp v (H.commit X₀ r)) v).g.map
      fun w => (H.commit X₀ r, samp v (H.commit X₀ r), w),
    fun r => Strategy.map _ _ (Z.sim (H.commit X₀ r, samp v (H.commit X₀ r)) v).s⟩

end FlockSoundness.SampledGeneric
