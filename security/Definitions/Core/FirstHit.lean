module

public import Definitions.Core.Interaction

/-!
# The first-hit collision finders

The finders a guarantee names when it takes `Claims.ExpectedCR` at a first-hit finder: what such a finder's seed is,
what it outputs and what it is charged. A prover's run with coins `ω` opens node `p` to a value, or doesn't open it
(`op ω p`). The finder makes a first run and `K` further runs, and at each node the first run opened it waits for
the first further run that opens the node again (`firstIdx`). Its bounds are proved in
`Proofs.Core.SamplingGeneric.FirstHit` and `Proofs.Core.SamplingGeneric.FirstHitClass`.

* **The seed** is a kept part `t`, the first run's coins and the `K` further runs' coins (`sampleFinderAt`'s `S`).
* **The output** (`outOfAt`) is `coll` at the first run, a further run that is the first to open some node the first
  run opened, and that node, where that is a pair.
* **The cost** is the first run's calls `c` and, at each further run, `c₂` of the nodes still waited for that the
  run's sample holds (`loopCostAt`). A node stops being waited for at the first further run that opens it.
* **By class** (`classFinderAt`): the nodes fall in classes (`cls`), and a further run is charged `cq` once for each
  class holding a node still waited for in its sample. The class of a point under a key is the least point with its
  key (`clsOf`).
-/

@[expose] public section

open Finset

namespace Verity.FirstHit

open Claims

variable {Ω P V : Type}

/-- **The first tape that opens `p`**, of the tapes `ρ 0, ρ 1, …`. -/
def firstIdx (op : Ω → P → Option V) (p : P) : (K : ℕ) → (Fin K → Ω) → Option (Fin K)
  | 0, _ => none
  | K + 1, ρ => if (op (ρ 0) p).isSome then some 0 else (firstIdx op p K fun i => ρ i.succ).map Fin.succ

/-- **The further runs' calls, by position**: a further tape's, at the nodes of `need` that its sample holds and that
have not been opened again. -/
def loopCostAt (op : Ω → P → Option V) (chk : Ω → P → Bool) (c₂ : Ω → Finset P → ℕ) :
    (K : ℕ) → Finset P → (Fin K → Ω) → ℕ
  | 0, _, _ => 0
  | K + 1, need, ρ =>
    c₂ (ρ 0) (need.filter fun p => chk (ρ 0) p) +
      loopCostAt op chk c₂ K (need.filter fun p => (op (ρ 0) p).isNone) fun i => ρ i.succ

/-- **The first-hit finder that pays by position**, with the kept part in its seed: its seed is the kept part, the
first run's coins and the further runs' coins; its cost is the first run's calls and, at each further run, what the
nodes still waited for that its sample holds cost there (`c₂`). -/
noncomputable def sampleFinderAt {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P]
    (op : T → Ω → P → Option V) (chk : T → Ω → P → Bool) (c : T → Ω → ℕ) (c₂ : T → Ω → Finset P → ℕ) (K : ℕ)
    (out : T × (Ω × (Fin K → Ω)) → Option (Msg × Msg)) : Finder where
  S := T × (Ω × (Fin K → Ω))
  out := out
  cost q :=
    c q.1 q.2.1 + loopCostAt (op q.1) (chk q.1) (c₂ q.1) K (univ.filter fun p => (op q.1 q.2.1 p).isSome) q.2.2

open Classical in
/-- **The pair the by-position finder outputs**, from `coll`, which reads two runs' tapes and a node: `coll` at the
first run, a further run that is the first to open some node the first run opened, and that node, where that is a
pair. What it reads of the further run is what that node costs there, which the loop charges. -/
noncomputable def outOfAt (op : Ω → P → Option V) (coll : Ω → Ω → P → Option (Msg × Msg)) (K : ℕ) (ω : Ω)
    (ρ : Fin K → Ω) : Option (Msg × Msg) :=
  if h : ∃ jp : Fin K × P, ((op ω jp.2).isSome ∧ firstIdx op jp.2 K ρ = some jp.1) ∧
      (coll ω (ρ jp.1) jp.2).isSome then
    coll ω (ρ h.choose.1) h.choose.2
  else none

variable {Q : Type}

/-- **The first-hit finder that pays by class**, with the kept part in its seed: the seed and the output are
`sampleFinderAt`'s; its cost is the first run's calls and, at each further run, `cq` of each class that holds a node
still waited for in the run's sample. -/
noncomputable def classFinderAt {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P]
    [DecidableEq Q] (cls : P → Q) (op : T → Ω → P → Option V) (chk : T → Ω → P → Bool) (c : T → Ω → ℕ)
    (cq : T → Ω → Q → ℕ) (K : ℕ) (out : T × (Ω × (Fin K → Ω)) → Option (Msg × Msg)) : Finder :=
  sampleFinderAt op chk c (fun t ω J => ∑ q ∈ J.image cls, cq t ω q) K out

/-- **The class of a point under a key**: the least point with its key. -/
def clsOf {n : ℕ} {Key : Type} [DecidableEq Key] (key : Fin n → Key) (g : Fin n) : Fin n :=
  (univ.filter fun g' => key g' = key g).min' ⟨g, mem_filter.2 ⟨mem_univ _, rfl⟩⟩

end Verity.FirstHit
