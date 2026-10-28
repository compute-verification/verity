import Definitions.Accounting.TS.Run

/-!
# Heads, labelled programs and a cheater's labelled run (trusted)

EPS1-SPEC §0.4, §0.7a and §1.3 over a run (`Accounting.TS.Run`):

* `Head`: what heads an owned right fixed committed word: `T_κ` for a writer kind (an MMA kind, or a uniform class),
  `S_V` for a gather-free vector word, a table route per memory level, a side unit's route.
* `Labelled`: a program's labelled split into word production, fold, filler and skippable work (honest's, §0.4, and
  the floor program ♭'s, §0.7a). `Honest` adds honest's measured cycles and clock.
* `Anatomy`: a cheater's run labelled the same way, with its word production split by head (`nword h`), its owned
  right fixed words per head (`words h`), and `fixed`, the owned right committed words less the adaptive ones
  (`N_right − N_adapt`, §2.1).

Counts are per unit of honest work throughout; a window's totals divided by its units.
-/

namespace Accounting.TS

open Finset

/-- **What heads an owned right fixed committed word** (§1.3), and the word route that prices it (§0.5): a writer kind
`t` (an MMA kind, priced `T_κ`, or a uniform class), a gather-free vector instruction (`S_V`), a vector instruction
whose production reads a load at a data-dependent address served from memory level `l` (a table word), or side unit
`u`. -/
inductive Head (T L U : Type)
  | writer (t : T)
  | vector
  | table (l : L)
  | side (u : U)

/-- `Head T L U ≃ T ⊕ Unit ⊕ L ⊕ U`, for its `Fintype`. -/
def Head.equiv (T L U : Type) : Head T L U ≃ T ⊕ (Unit ⊕ (L ⊕ U)) where
  toFun
    | .writer t => .inl t
    | .vector => .inr (.inl ())
    | .table l => .inr (.inr (.inl l))
    | .side u => .inr (.inr (.inr u))
  invFun
    | .inl t => .writer t
    | .inr (.inl ()) => .vector
    | .inr (.inr (.inl l)) => .table l
    | .inr (.inr (.inr u)) => .side u
  left_inv h := by cases h <;> rfl
  right_inv s := by rcases s with t | ⟨⟨⟩⟩ | l | u <;> rfl

noncomputable instance instFintypeHead (T L U : Type) [Fintype T] [Fintype L] [Fintype U] : Fintype (Head T L U) :=
  Fintype.ofEquiv _ (Head.equiv T L U).symm

/-- **A labelled program** (§0.4), per unit of honest work: its word production `nw` (the MMAs and the operand loading
and forming the checked words need, and everything else it executes that isn't below), its fold `nf`, its filler
`nx` and its skippable work `ns` (work Verify neither checks nor needs). -/
structure Labelled (K : Type) where
  nw : K → ℝ
  nf : K → ℝ
  nx : K → ℝ
  ns : K → ℝ

/-- A labelled program's counts: `n = n^w + n^f + n^x + n^s`. -/
def Labelled.n {K : Type} (P : Labelled K) : K → ℝ := fun k => P.nw k + P.nf k + P.nx k + P.ns k

/-- **The honest kernel** `H`: its labelled split, its measured cycles per unit at saturation `C_H` and its measured
sustained SM clock `f_H`. -/
structure Honest (K : Type) where
  prog : Labelled K
  CH : ℝ
  fH : ℝ

/-- Honest's time per unit: `T(H) = C_H / f_H`. -/
noncomputable def Honest.time {K : Type} (H : Honest K) : ℝ := H.CH / H.fH

/-- **A cheater's labelled run** (§2.1): the run, its word production split by the head of the words it
produces (`nword h`), its fold `nfold` and filler `nfill`, its owned right fixed committed words per head
(`words h`), and `fixed`, its owned right committed words less the adaptive ones (`N_right − N_adapt`), all per unit
of honest work. The labelling is §0.4's and §1.3's (heads, ownership, copies); the named assumptions over an
`Anatomy` are claimed at that labelling. -/
structure Anatomy (K T L U : Type) where
  run : Run K
  nword : Head T L U → K → ℝ
  nfold : K → ℝ
  nfill : K → ℝ
  words : Head T L U → ℝ
  fixed : ℝ

variable {K T L U : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U]

/-- The labelling splits the run: every part is `≥ 0` and the parts sum to at most the run's counts, class by class. -/
def Anatomy.Split (A : Anatomy K T L U) : Prop :=
  (∀ h k, 0 ≤ A.nword h k) ∧ (∀ k, 0 ≤ A.nfold k) ∧ (∀ k, 0 ≤ A.nfill k) ∧ (∀ h, 0 ≤ A.words h) ∧
    ∀ k, ∑ h, A.nword h k + A.nfold k + A.nfill k ≤ A.run.n k

end Accounting.TS
