module

public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Basic.ENNReal.BigOperators
public import Mathlib.Basic.ENNReal.Inv
public import Mathlib.Order.CompleteLattice.Basic
public import Mathlib.Data.Fintype.Perm
public import Mathlib.Data.Fintype.Pi
public import Mathlib.Data.Fintype.Sigma
public import Definitions.Core.Draw
public import Definitions.Core.Game.Prob

/-! A window of calls, one sample that depends on the registration and then the calls' sessions in order, with `ServedTie` and `CollCounted` among its open premises. -/

@[expose] public section

namespace Pouw.Window

open scoped ENNReal
open Verity.Game

export Verity (Game)
export Verity.Game (Strategy prob avg prCoin)

variable {α β : Type}

noncomputable def thenExp {f : α → Game β} (F : (a : α) → Strategy (f a) → ℝ≥0∞) :
    (g : Game α) → Strategy (g.bind f) → ℝ≥0∞
  | .ret a, s => F a s
  | .send _ k, s => thenExp F (k s.1) s.2
  | @Verity.Game.coin _ C iF _ k, s => ∑ c : C, (@Fintype.card C iF : ℝ≥0∞)⁻¹ * thenExp F (k c) (s c)

export Verity (SamplingDesign)

def auditReg {Reg U : Type} (L : Reg → SamplingDesign U) (session : Finset U → Reg → Game Bool) : Game (Bool × Finset U) :=
  .send Reg fun R => .coin (L R).Ω fun ω => (session ((L R).draw ω) R).map fun ok => (ok, (L R).draw ω)

abbrev Cont {Reg U : Type} (L : Reg → SamplingDesign U) (session : Finset U → Reg → Game Bool) (R : Reg) :=
  ∀ ω : (L R).Ω, Strategy (session ((L R).draw ω) R)

section Audit

variable {Reg U : Type} {L : Reg → SamplingDesign U} {session : Finset U → Reg → Game Bool}

def reg (σ : Strategy (auditReg L session)) : Reg := σ.1

def cont (σ : Strategy (auditReg L session)) : Cont L session (reg σ) :=
  fun ω => Strategy.ofMap _ _ (σ.2 ω)

end Audit

abbrev WUnit {C : ℕ} (n : Fin C → ℕ) := (c : Fin C) × Fin (n c)

structure SeedIn (Tag Salt Pos Root : Type) where
  tag : Tag
  salt : Salt
  pos : Pos
  rootA : Root
  rootB : Root

/-- `pos X v` is a function of the unit's place `v` and `X` alone, so no message of a session sets it: a prover that could
pick a position inside a unit could pick its seed. -/
structure UnitStmts (Tr : Type) {C : ℕ} (n : Fin C → ℕ) where
  Tag : Type
  Salt : Type
  Pos : Type
  Root : Type
  tag : Tag
  salt : Salt
  pos : Tr → WUnit n → Pos
  rootA : Tr → WUnit n → Root
  rootB : Tr → WUnit n → Root
  computes : WUnit n → SeedIn Tag Salt Pos Root → Tr → Prop

namespace UnitStmts

variable {Tr : Type} {C : ℕ} {n : Fin C → ℕ}

def seedIn (U : UnitStmts Tr n) (X : Tr) (v : WUnit n) : SeedIn U.Tag U.Salt U.Pos U.Root :=
  ⟨U.tag, U.salt, U.pos X v, U.rootA X v, U.rootB X v⟩

open Classical in
/-- Wrong is about values, not work: a deficient unit whose committed outputs came out right by luck isn't incorrect. -/
noncomputable def incorrect (U : UnitStmts Tr n) (X : Tr) : Finset (WUnit n) :=
  Finset.univ.filter fun v => ¬ U.computes v (U.seedIn X v) X

end UnitStmts

/-- A unit is a self-contained tile, so its harm is its own `W_ref(tile)`: fan 1, with no steering term, as no unit sets
the noise another reads. -/
def harmSets {C : ℕ} {n : Fin C → ℕ} (Wref : WUnit n → ℕ) (ε : ℝ≥0∞) (B : Finset (WUnit n)) : Prop :=
  ε * ((∑ v, Wref v : ℕ) : ℝ≥0∞) ≤ ((∑ v ∈ B, Wref v : ℕ) : ℝ≥0∞)

section Window

variable {C : ℕ} {n : Fin C → ℕ}

def SamplingDesign.prod (L : (c : Fin C) → SamplingDesign (Fin (n c))) : SamplingDesign (WUnit n) where
  Ω := (c : Fin C) → (L c).Ω
  draw ω := Finset.univ.sigma fun c => (L c).draw (ω c)

def slice (S : Finset (WUnit n)) (c : Fin C) : Finset (Fin (n c)) :=
  Finset.univ.filter fun u => (⟨c, u⟩ : WUnit n) ∈ S

def pick (t : Option (WUnit n)) (c : Fin C) : Option (Fin (n c)) :=
  t.bind fun x => if h : x.1 = c then some (Fin.cast (congrArg n h) x.2) else none

open Classical in
noncomputable def atTgt {V : Type} (S : Finset V) (t : Option V) (f : V → ℝ≥0∞) : ℝ≥0∞ :=
  t.elim 0 fun u => if u ∈ S then f u else 0

def onAccept (g : Game Bool) : Bool → Game Bool
  | true => g
  | false => .ret false

noncomputable def onAcceptExp (g : Game Bool) (h : Strategy g → ℝ≥0∞) : (ok : Bool) → Strategy (onAccept g ok) → ℝ≥0∞
  | true, s => h s
  | false, _ => 0

def seqCalls (sess : Fin C → Game Bool) : List (Fin C) → Game Bool
  | [] => .ret true
  | c :: cs => (sess c).bind (onAccept (seqCalls sess cs))

def executionPhaseSession {Reg : Type} (sess : (c : Fin C) → Finset (Fin (n c)) → Reg → Game Bool) (S : Finset (WUnit n))
    (R : Reg) : Game Bool :=
  seqCalls (fun c => sess c (slice S c) R) (List.finRange C)

noncomputable def along (sess : Fin C → Game Bool) (F : (c : Fin C) → Strategy (sess c) → ℝ≥0∞) :
    (cs : List (Fin C)) → Strategy (seqCalls sess cs) → ℝ≥0∞
  | [], _ => 0
  | c :: cs, s => F c (Strategy.ofBind (onAccept (seqCalls sess cs)) (sess c) s) +
      thenExp (f := onAccept (seqCalls sess cs)) (onAcceptExp (seqCalls sess cs) fun s' => along sess F cs s') (sess c) s

/-- `cover` can hold only for a session that checks unit `u` at its place, never at one the prover picks. -/
structure CallAnalysis {Reg Tr : Type} (U : UnitStmts Tr n) (c : Fin C)
    (sess : Finset (Fin (n c)) → Reg → Game Bool) where
  ks : (S : Finset (Fin (n c))) → (R : Reg) → Strategy (sess S R) → Fin (n c) → ℝ≥0∞
  link : (S : Finset (Fin (n c))) → (R : Reg) → Strategy (sess S R) → Fin (n c) → Tr → ℝ≥0∞
  cover : ∀ S R (τ : Strategy (sess S R)) u X, u ∈ S → (⟨c, u⟩ : WUnit n) ∈ U.incorrect X →
    prob (· = true) (sess S R) τ ≤ ks S R τ u + link S R τ u X

/-- No constant bounds it for every state of a hashed session: a state that holds a hash collision passes with extraction
failing, so a constant holds only at the oracle layer. -/
def CallAnalysis.KnowledgeSound {Reg Tr : Type} {U : UnitStmts Tr n} {c : Fin C}
    {sess : Finset (Fin (n c)) → Reg → Game Bool} (A : CallAnalysis U c sess) (κ : ℝ≥0∞)
    (g : (S : Finset (Fin (n c))) → (R : Reg) → Strategy (sess S R) → ℝ≥0∞) : Prop :=
  ∀ S R (τ : Strategy (sess S R)) u, u ∈ S → A.ks S R τ u ≤ κ + g S R τ

/-- The committed transcript is a function of the prover's state at registration, so it is the same for every call and
every sample. -/
def BindSoundIn {Reg Tr : Type} {U : UnitStmts Tr n} (Lw : Reg → SamplingDesign (WUnit n))
    (sess : (c : Fin C) → Finset (Fin (n c)) → Reg → Game Bool)
    (committed : (R : Reg) → Cont Lw (executionPhaseSession sess) R → Tr) (A : (c : Fin C) → CallAnalysis U c (sess c))
    (c : Fin C) (εbind : (R : Reg) → Cont Lw (executionPhaseSession sess) R → ℝ≥0∞) : Prop :=
  ∀ (R : Reg) (τ : Cont Lw (executionPhaseSession sess) R) (tgt : (Lw R).Ω → Option (WUnit n)),
    avg (fun ω => along (fun c' => sess c' (slice ((Lw R).draw ω) c') R)
      (fun c' τc => if c' = c then
          atTgt (slice ((Lw R).draw ω) c') (pick (tgt ω) c')
            fun u => (A c').link (slice ((Lw R).draw ω) c') R τc u (committed R τ)
        else 0)
      (List.finRange C) (τ ω)) ≤ εbind R τ

/-- At one oracle no count bounds it: a state that holds a collision of the oracle makes it about 1. -/
noncomputable def coll {Reg : Type} (Lw : Reg → SamplingDesign (WUnit n))
    (sess : (c : Fin C) → Finset (Fin (n c)) → Reg → Game Bool)
    (g : (c : Fin C) → (S : Finset (Fin (n c))) → (R : Reg) → Strategy (sess c S R) → ℝ≥0∞) (R : Reg)
    (τ : Cont Lw (executionPhaseSession sess) R) : ℝ≥0∞ :=
  avg fun ω => along (fun c => sess c (slice ((Lw R).draw ω) c) R)
    (fun c τc => g c (slice ((Lw R).draw ω) c) R τc) (List.finRange C) (τ ω)

/-- It holds only where `σ H` is one oracle program's run at `H` and `qs` counts its calls; as stated, with `σ` a separate
play at each oracle and `qs` free, it is false at a true count and vacuous at a made-up one. No guarantee reads it. -/
def CollCounted {O Reg : Type} [Fintype O] (Lw : O → Reg → SamplingDesign (WUnit n))
    (sess : O → (c : Fin C) → Finset (Fin (n c)) → Reg → Game Bool)
    (g : (H : O) → (c : Fin C) → (S : Finset (Fin (n c))) → (R : Reg) → Strategy (sess H c S R) → ℝ≥0∞)
    (qs : (H : O) → Strategy (auditReg (Lw H) (executionPhaseSession (sess H))) → ℕ) (f : ℕ → ℝ≥0∞) : Prop :=
  ∀ (q : ℕ) (σ : (H : O) → Strategy (auditReg (Lw H) (executionPhaseSession (sess H)))), (∀ H, qs H (σ H) ≤ q) →
    avg (fun H => coll (Lw H) (sess H) (g H) (reg (σ H)) (cont (σ H))) ≤ f q

/-- For the keyed sample this is the keyed streams' pseudorandomness at the context `digest(R)` (`prf/sha-512`); for the
service's sample it holds at `η = 0` on the copy-closed sets (`CopyClosed`). -/
def MissProbWithin {Reg : Type} (Lw : Reg → SamplingDesign (WUnit n)) (L : Reg → (c : Fin C) → SamplingDesign (Fin (n c)))
    (𝓑 : Finset (WUnit n) → Prop) (η : ℝ≥0∞) : Prop :=
  ∀ R B, 𝓑 B → (Lw R).missProb B ≤ (SamplingDesign.prod (L R)).missProb B + η

/-- An instance whose readings forget the words (`In`, `Out` and `Root` with one element) makes every transcript anchored
and served. -/
structure ServedPass (Tr : Type) {C : ℕ} (n : Fin C → ℕ) (Pos : Type) where
  In : Type
  Out : Type
  Root : Type
  Leaf : Type
  anchor : Pos → In
  rootH3 : Root
  leaf : Pos → Leaf
  tileRead : Leaf → WUnit n → Out
  inputs : Tr → WUnit n → In
  outputs : Tr → WUnit n → Out
  rootOf : Tr → Root

def Anchored {Tr : Type} (U : UnitStmts Tr n) (P : ServedPass Tr n U.Pos) (X : Tr) : Prop :=
  ∀ v, P.inputs X v = P.anchor (U.pos X v)

def Served {Tr : Type} (U : UnitStmts Tr n) (P : ServedPass Tr n U.Pos) (X : Tr) : Prop :=
  P.rootOf X = P.rootH3 ∧ ∀ v, P.outputs X v = P.tileRead (P.leaf (U.pos X v)) v

/-- It makes the committed transcript the served computation's, not a self-consistent copy the prover chose. It is open:
`δin = 1` is the only bound, as no Program the verifier requires reads the `-h3` leaf. -/
def ServedTie {Reg Tr : Type} (Lw : Reg → SamplingDesign (WUnit n)) (sess : (c : Fin C) → Finset (Fin (n c)) → Reg → Game Bool)
    (committed : (R : Reg) → Cont Lw (executionPhaseSession sess) R → Tr) (U : UnitStmts Tr n) (P : ServedPass Tr n U.Pos)
    (δin : (R : Reg) → Cont Lw (executionPhaseSession sess) R → ℝ≥0∞) : Prop :=
  ∀ σ : Strategy (auditReg Lw (executionPhaseSession sess)),
    prob (fun o => o.1 = true ∧
        ¬ (Anchored U P (committed (reg σ) (cont σ)) ∧ Served U P (committed (reg σ) (cont σ))))
      (auditReg Lw (executionPhaseSession sess)) σ ≤ δin (reg σ) (cont σ)

end Window

def rank {N : ℕ} (D : Finset (Fin N)) (t : Fin N) : ℕ := (D.filter (· < t)).card

/-- The `K`-subset of the `N` tiles a permutation samples, the images of `0, …, K − 1`: under a uniform permutation, a
uniform `K`-subset. -/
def subsetDraw {N : ℕ} (K : ℕ) (π : Equiv.Perm (Fin N)) : Finset (Fin N) :=
  (Finset.univ.filter fun i : Fin N => i.val < K).image π

def sessions (K cap : ℕ) : ℕ := (K + cap - 1) / cap

/-- The session pads its instances with the filler tile, which is credited no work. -/
def chunk {N : ℕ} (cap : ℕ) (D : Finset (Fin N)) (σ : ℕ) : Finset (Fin N) :=
  D.filter fun t => σ * cap ≤ rank D t ∧ rank D t < (σ + 1) * cap

def serviceSamplingDesign (N K cap : ℕ) : SamplingDesign (WUnit fun _ : Fin (sessions K cap) => N) where
  Ω := Equiv.Perm (Fin N)
  draw π := Finset.univ.sigma fun σ => chunk cap (subsetDraw K π) σ.val

def share (K cap σ : ℕ) : ℕ := min cap (K - σ * cap)

def sessionSamplingDesign (N K cap σ : ℕ) : SamplingDesign (Fin N) where
  Ω := Equiv.Perm (Fin N)
  draw π := subsetDraw (share K cap σ) π

def CopyClosed {S N : ℕ} (B : Finset (WUnit fun _ : Fin S => N)) : Prop :=
  ∀ (σ σ' : Fin S) (t : Fin N), (⟨σ, t⟩ : WUnit fun _ : Fin S => N) ∈ B → ⟨σ', t⟩ ∈ B

/-- Its instance's public inputs are `t` and the window's common words, never `σ`, so its seed inputs and what it computes
are tile `t`'s. -/
def UnitStmts.copies {Tr : Type} {N : ℕ} (V : UnitStmts Tr fun _ : Fin 1 => N) (S : ℕ) :
    UnitStmts Tr fun _ : Fin S => N where
  Tag := V.Tag
  Salt := V.Salt
  Pos := V.Pos
  Root := V.Root
  tag := V.tag
  salt := V.salt
  pos X v := V.pos X ⟨0, v.2⟩
  rootA X v := V.rootA X ⟨0, v.2⟩
  rootB X v := V.rootB X ⟨0, v.2⟩
  computes v si X := V.computes ⟨0, v.2⟩ si X

def IncorrectCopyClosed {Tr : Type} {S N : ℕ} (U : UnitStmts Tr fun _ : Fin S => N) : Prop :=
  ∀ X, CopyClosed (U.incorrect X)

end Pouw.Window
