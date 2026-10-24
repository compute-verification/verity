import Definitions.Pouw.Window

/-! The proof-only part of `Definitions.Pouw.Window`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.Window

open scoped ENNReal

open Game

variable {α β : Type}

/-- **PoUW's harm family** (the `𝓑` the window audit reads for compute): the wrong-unit sets whose work is at least the
share `ε` of the window's, `∑_{v ∈ B} W_ref(v) ≥ ε · W` with `W = ∑_v W_ref(v)`, where `Wref` is the scheme's work
table read at each unit's template. Under decision 106 a unit is a self-contained `r × c` tile (`UnitStmts`), so its
harm is its own `W_ref(tile)`, `r·c` dots. That is fan 1, since a wrong tile is charged its own work and no other
unit's, and there is no steering term, since no unit sets the noise another reads. -/
def harmSets {C : ℕ} {n : Fin C → ℕ} (Wref : WUnit n → ℕ) (ε : ℝ≥0∞) (B : Finset (WUnit n)) : Prop :=
  ε * ((∑ v, Wref v : ℕ) : ℝ≥0∞) ≤ ((∑ v ∈ B, Wref v : ℕ) : ℝ≥0∞)

section Window

variable {C : ℕ} {n : Fin C → ℕ}

/-- **The window's collision term at the sessions' counted queries** (premise, open; the random-oracle card's, with gap
(4)'s `ε_id`): for windows indexed by the oracle `H`, uniform on a finite `O` (the window's game, its laws, sessions
and state terms at each oracle), every prover's plays `σ H` whose sessions make at most `q` oracle calls by the count
`qs` have collision term at most `f q` on average over the oracle.

It holds only where `σ H` is one oracle program's run at `H` and `qs H (σ H)` counts that program's calls. The random
oracle's collision bound is a statement about one program run against a uniform oracle, and its expected instance is
that bound, `f q ≈ q²/2^512` times the tables' factors. As stated, `σ` is a separate play at each oracle, which is the
non-uniform adversary that bound doesn't cover, and `qs` is free:
* at a true count it is false: the play that at each `H` holds a hardcoded collision of `H` makes 0 calls, and its
  `coll` is about 1 at every `H`, so the average is about 1, more than `f 0`;
* at a count that isn't a count of calls it is vacuous: `qs H σ := 0` where `coll` at `σ` is at most `f 0`, and
  otherwise some `Q` with `f Q ≥ 1`, meets it for every game.
`qs` is meant to be the prover model's count of its sessions' calls (gap (4)'s `ProverModel.sessCalls`, which its
`Prover.Bounded` counts), and that field isn't tied to an oracle program either. No guarantee reads `CollCounted`, and
before the random-oracle card discharges it, it is restated with gap (4)'s session-query tie, one definition for both:
over an oracle strategy of the window's game, one `QTree` per prover move, with `H` only in the verdict, so the plays
and their count can depend on `H` only through queries (note:compute-accounting/20261018T2336Z-finding-pw8-rered,
the recheck's (c)). -/
def CollCounted {O Reg : Type} [Fintype O] (Lw : O → Reg → Law (WUnit n))
    (sess : O → (c : Fin C) → Finset (Fin (n c)) → Reg → Game Bool)
    (g : (H : O) → (c : Fin C) → (S : Finset (Fin (n c))) → (R : Reg) → Strategy (sess H c S R) → ℝ≥0∞)
    (qs : (H : O) → Strategy (auditReg (Lw H) (windowSession (sess H))) → ℕ) (f : ℕ → ℝ≥0∞) : Prop :=
  ∀ (q : ℕ) (σ : (H : O) → Strategy (auditReg (Lw H) (windowSession (sess H)))), (∀ H, qs H (σ H) ≤ q) →
    avg (fun H => coll (Lw H) (sess H) (g H) (reg (σ H)) (cont (σ H))) ≤ f q

/-- **What the served pass committed**, as the window's units read it, fixed before the window's registration:
* `anchor p`: the words the served pass committed for the inputs (activations and weights) of the tile at position
  `p`, its anchors;
* `rootH3`: the pass's `-h3` root; `leaf p`, the `-h3` leaf at position `p` under it; `tileRead l v`, the words of leaf
  `l` that unit `v` reads (`tile_read` in `verity/core/protocols/pouw/window.py`);
* `inputs X v`, `outputs X v`: a transcript `X`'s values at unit `v`'s input and output wires; `rootOf X`, the `-h3`
  root the window's entries hold in `X`.
The positions are the units' statements' (`UnitStmts.Pos`), so a unit reads the anchors and the leaf at its own place in
the transcript. A reading of `ServedTie` names its instance: the window's wires and the pass's `-h3` tree. An instance
whose readings forget the words (`In`, `Out` and `Root` with one element) makes every transcript anchored and served. -/
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

/-- **An anchored transcript**: its values at every unit's input wires are the served pass's anchors at the unit's
position. -/
def Anchored {Tr : Type} (U : UnitStmts Tr n) (P : ServedPass Tr n U.Pos) (X : Tr) : Prop :=
  ∀ v, P.inputs X v = P.anchor (U.pos X v)

/-- **A served transcript**: its window entries hold the served pass's `-h3` root, and its outputs at every unit are
the words that unit reads from the `-h3` leaf at its position. -/
def Served {Tr : Type} (U : UnitStmts Tr n) (P : ServedPass Tr n U.Pos) (X : Tr) : Prop :=
  P.rootOf X = P.rootH3 ∧ ∀ v, P.outputs X v = P.tileRead (P.leaf (U.pos X v)) v

/-- **The tie to what's served** (premise `δ_in`, PoUW's reading of `WindowEps`; open on main): every prover's window
accepts while its committed transcript isn't anchored and served (`Anchored`, `Served`, against the served pass `P`)
with probability at most `δin(R, τ)`, a function of the window's state. `WindowEps` bounds the wrong units of whatever
transcript the analysis commits; this is what makes that transcript the served computation's, not a self-consistent copy
the prover chose. The units' statements `U` and the served pass `P` are fixed before the window, so the prover picks
neither.

It is open on main, where `δin = 1` is the only bound: the window's tile Program (`Pc8TileSide`) reads
`[ac, bc, pa, pb, ia, jb]` and passes neither `ia` nor `jb` on, its side ports' regions are empty, nothing outside
`window.py` calls `tile_read`, and `window_problem` is Python only, so no Program reads the `-h3` leaf. It is discharged
when the window's Program reads the `-h3` leaf in gates (`tile_read`) and `window_problem` is a Program: then the
outputs' half is that Program's opening of the `-h3` root, and the inputs' half is C-Flock's `AnchorsSound` (`δ_in`,
`Proofs/Flock/Soundness/Audit/OneStage.lean`), each a binding term at the window's state. -/
def ServedTie {Reg Tr : Type} (Lw : Reg → Law (WUnit n)) (sess : (c : Fin C) → Finset (Fin (n c)) → Reg → Game Bool)
    (committed : (R : Reg) → Cont Lw (windowSession sess) R → Tr) (U : UnitStmts Tr n) (P : ServedPass Tr n U.Pos)
    (δin : (R : Reg) → Cont Lw (windowSession sess) R → ℝ≥0∞) : Prop :=
  ∀ σ : Strategy (auditReg Lw (windowSession sess)),
    prob (fun o => o.1 = true ∧
        ¬ (Anchored U P (committed (reg σ) (cont σ)) ∧ Served U P (committed (reg σ) (cont σ))))
      (auditReg Lw (windowSession sess)) σ ≤ δin (reg σ) (cont σ)

end Window

/-- **The service's units' statements**: the tiles' statements `V` (a window of one call whose units are the `N`
tiles) read in every one of `S` sessions. Unit `⟨σ, t⟩` is tile `t` proved in session `σ`; its instance's public
inputs are `t` and the window's common words (`public_inputs` in `window.py`: the window root, its domain, the sides'
domains and `domain_h3`), never `σ`, so its seed inputs and what it computes are tile `t`'s. -/
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

/-- **The deployed wrong sets are copy-closed** (PoUW's reading of `WindowEps`, which it reads at
`𝓑 = harmSets W_ref ε ∧ CopyClosed`, and `ServiceEscapes`'s family): at every transcript, the wrong units hold a tile
in every session or in none. It holds for the service's statements (`Pouw.Window.wrongCopyClosed_copies`), and there
`𝓑`'s copy-closed conjunct is automatic (`Pouw.Window.harmSets_copyClosed_iff`). -/
def WrongCopyClosed {Tr : Type} {S N : ℕ} (U : UnitStmts Tr fun _ : Fin S => N) : Prop :=
  ∀ X, CopyClosed (U.wrong X)

end Pouw.Window
