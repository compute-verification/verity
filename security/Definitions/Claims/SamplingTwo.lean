module

public import Definitions.Claims.SamplingGeneric

/-!
# Claims: two-stage sampling at the footprint

The sampled audit's two-stage design (registered `{"rule": "two-stage", "grain": Q, "replay": LAW, "k": K}`), on
`Claims.Sampling`'s statement, witness and commitment and `Claims.SamplingGeneric`'s compiled verifier, read as they
stand, so neither changes. Replay blocks are the blocks of a replay partition that the proof-block partition refines.
Serving commits every value read across a replay-block boundary (the registration, the statement's roots). The first
stage's coins sample replay blocks under a one-stage rule over them (the replay sample); the prover then commits each
sampled replay block's interior (the interior registration, `iroots`); the second stage's coins, fresh, sample `k_t`
proof blocks of each stratum `t` inside each sampled replay block. One-stage designs are the case whose replay blocks
are the proof blocks, with `k = 1` and no interior registration.

1. **The design** (`Design.Two`): the replay partition (`part`, the statement's replay query evaluated on its
   program), the first stage's rules, one per class of it (`replay`), and the second stage's `k` per stratum of the
   proof-block partition (`k`). A cell (`cell`) is a replay block's proof blocks in one stratum; `n_{ρ,t}` is its size.
   A replay sample `S`'s proof blocks are its replay blocks' cells (`sampleBlocks`), and the footprint bits
   `M = ⌈log₂ ∏_{ρ,t}(n_{ρ,t} + 1)⌉` (`footBits`) cover a set's counts per cell.
2. **The footprint** (`footprint`) at interior values `ι`, one interior per replay block: replay block `ρ`'s incorrect
   proof blocks in stratum `t` on its boundary values, which the statement's roots commit, and its interior `ι ρ`. A
   replay block's strata are read at one interior, so that its cells' counts are those of one assignment of its values.
   With one stratum, its least footprint is the fewest incorrect proof blocks of any interior consistent with its
   boundary, fixed before any coin; `ρ` is an incorrect replay block when every interior leaves one.
3. **The acceptance probability at the footprint** (`Design.Two.Pa`):
   `P̄_a(w) = E_S ∏_{ρ∈S} e_ρ(w)`, `e_ρ(w) = ∏_t C(n_{ρ,t} − w(ρ, t), k_t)/C(n_{ρ,t}, k_t)` (`Design.Two.esc`, the
   hypergeometric miss, `escCell`), written through the first stage's `P_a`; the confidence set
   `M̄_δ = {w : P̄_a(w) > δ}` (`Design.Two.M`); and the relation `RTwo`: at some interiors, the extracted boundary
   trace's footprint is in `M̄_δ`. The cap reads two rates, a first stage that catches each replay block at rate `r₁`
   (`Design.Two.FirstRate`) and a second stage that samples each cell at rate `r₂` (`Design.Two.SecondRate`), and the
   replay blocks a footprint marks incorrect (`Design.Two.wrongBlocks`).
4. **The compiled verifier** (`CompiledTwo`): a `Compiled` with the statement's two-stage design (`two`) and the
   replay sample (`replaySample`). Its second stage is `Compiled.stage2`, and `iroots x v ρ` is the roots a run shows for
   replay block `ρ`'s interior. Its extractor, errors and numbers are `Compiled`'s.
5. **The contract** (`CompiledTwo.Sessions`): `Compiled.Sessions` with its draw law split by stage and its openings
   read at a replay block. The first stage misses a set `T` of replay blocks with probability at most `P_a(T)`
   (`replayDraws`); at the first stage's coins, the second stage misses a set `X` of proof blocks with probability at
   most `∏_{ρ∈S} ∏_t C(n_{ρ,t} − |X ∩ cell|, k_t)/C(n_{ρ,t}, k_t)` (`secondDraws`); the waits' bits cover the
   footprint bits and the replay sample's proof blocks, `M ≤ n_out` and `|sampleBlocks S| ≤ n_in` (`unions`); the
   interior roots precede the second stage (`interior`); and a witness in `A`'s relation reveals openings under which
   its proof block is correct on its replay block's boundary values and interior values (`opens`). `slots`, `refuses`,
   `fixed` and `runs` are `Sessions`'.
6. **The claims**: `CompiledTwo.KS`, at `Compiled`'s `κ_δ`, `γ` and `ε_bind`, and the compiler's statement
   `CompilerKSTwo`; the extractor's cost, `Compiled.Efficient` at the two-stage verifier's own waits' counts `k_out`
   and `k_in` (`CompilerEfficientTwo`), and so `CompiledTwo.KSAt`, KS read at a time budget at `Compiled`'s `work`
   (`CompilerKSAtTwo`); `CompiledTwo.Exfil` and `CompilerExfilTwo`, the exfiltration bound (7).
7. **What the incorrect proof blocks can serve**: a set `T` of proof blocks is admissible at `δ` when its counts per cell
   (`counts`) are a footprint in `M̄_δ` (`admissibleTwo`); `b(T)` is the values its proof blocks commit (`bitsOf`). The
   influence set at input values `a` (`influenceTwo`) is the served outputs (`served`) of every boundary trace with
   inputs `a` (`inputsOf`) that `RTwo` takes, and `RInflTwo` holds when a trace's served outputs are in it at its own
   inputs. `CompiledTwo.Exfil` is KS at `RInflTwo` and `|influenceTwo| ≤ Σ_{T admissible} 2^{b(T)}`: on fixed inputs, a
   trace whose proof blocks outside `T` are correct serves outputs fixed by `T`'s committed bits, so exporting `β` bits
   needs `2^β ≤ Σ_{T admissible} 2^{b(T)} ≤ 2^H·Σ_{j≤K} C(n, j)` (every admissible set at most `K` proof blocks and `H`
   bits, `n` proof blocks).

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.Sampling

/-! ## 1. The design -/

/-- A slot's stratum among strata of sizes `ns`, in order; `ns.length` past them. -/
def stratumIdx : List ℕ → ℕ → ℕ
  | [], _ => 0
  | n :: ns, v => if v < n then 0 else stratumIdx ns (v - n) + 1

/-- **The strata** of `P`: its classes' strata in population order, each its proof blocks. -/
def Partition.strata (P : Partition) : List (List (List ℕ)) := P.classes.flatten

/-- **Slot `v`'s stratum** in `P`: its index in `P.strata`. -/
def Partition.stratumOf (P : Partition) (v : ℕ) : ℕ := stratumIdx (P.strata.map List.length) v

/-- **The replay block of proof block `v`** of `Pt`: the first block of the replay partition `Pρ` that holds each of its
gates; `none` when none does. -/
def replayOf (Pt Pρ : Partition) (v : ℕ) : Option ℕ :=
  (List.range Pρ.blocks.length).find? fun ρ =>
    match Pt.blocks[v]?, Pρ.blocks[ρ]? with
    | some u, some r => u.all r.contains
    | _, _ => false

/-- **A cell**: the proof blocks of `Pt` in replay block `ρ` of `Pρ` and in stratum `t` of `Pt`. Its size is
`n_{ρ,t}`. -/
def cell (Pt Pρ : Partition) (ρ t : ℕ) : Finset ℕ :=
  (Finset.range Pt.blocks.length).filter fun v => replayOf Pt Pρ v = some ρ ∧ Pt.stratumOf v = t

/-- **A replay sample's proof blocks**: the proof blocks of `Pt` whose replay block in `Pρ` is in `S`, every cell of
`S`'s replay blocks. -/
def sampleBlocks (Pt Pρ : Partition) (S : List ℕ) : Finset ℕ :=
  (Finset.range Pt.blocks.length).filter fun v => ∃ ρ ∈ S, replayOf Pt Pρ v = some ρ

/-- **A two-stage sampling design**, as the verifier evaluates it on a statement: the replay partition (`part`, the
statement's replay query on its program, which the proof-block partition refines), the first stage's rules, one per
class of the replay partition (`replay`, a one-stage design over replay blocks), and the second stage's `k` per stratum
of the proof-block partition (`k`), each stratum's past the list `0`. -/
structure Design.Two where
  part : Partition
  replay : Design
  k : List ℕ

/-- **The two-stage design is well formed** on the proof-block partition `Pt`: its first stage's rules are well formed
on the replay partition, and it has one `k ≥ 1` per stratum of `Pt`. -/
def Design.Two.WellFormed (d : Design.Two) (Pt : Partition) : Prop :=
  d.replay.WellFormed d.part ∧ d.k.length = Pt.strata.length ∧ ∀ k ∈ d.k, 1 ≤ k

/-- **The footprint bits** `M = ⌈log₂ ∏_{ρ,t} (n_{ρ,t} + 1)⌉`, over the grid of replay blocks and strata: a set of
proof blocks has `|Y ∩ cell(ρ, t)| ≤ n_{ρ,t}` in each cell, so its counts per cell take at most `2^M` values. -/
def footBits (Pt : Partition) (d : Design.Two) : ℕ :=
  Nat.clog 2 (∏ p ∈ Finset.range d.part.blocks.length ×ˢ Finset.range Pt.strata.length,
    ((cell Pt d.part p.1 p.2).card + 1))

/-! ## 2. The footprint -/

open Classical in
/-- **The incorrect proof blocks of cell `(ρ, t)` at the interior values `ι`**: each is incorrect on replay block `ρ`'s
values, its boundary's the committed ones `w` opens and its interior's `ι` (`Prog.refIn` at the replay block). -/
noncomputable def wrongIn (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (Pt Pρ : Partition) (w : Wit)
    (ρ t : ℕ) (ι : ℕ → Bool) : Finset ℕ :=
  (cell Pt Pρ ρ t).filter fun v => ¬ ∃ u r, Pt.blocks[v]? = some u ∧ Pρ.blocks[ρ]? = some r ∧
    s.prog.Correct (s.prog.refIn r (w.val H Cm s) ι) u

/-- **The footprint at interior values `ι`** `w(ρ, t)`, `ι ρ` replay block `ρ`'s interior: cell `(ρ, t)`'s incorrect
proof blocks on `ρ`'s boundary values, which the statement's roots commit at serve time, and its interior `ι ρ`. Every
stratum of `ρ` is read at the one interior `ι ρ`. -/
noncomputable def footprint (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (Pt Pρ : Partition) (w : Wit)
    (ι : ℕ → ℕ → Bool) (ρ t : ℕ) : ℕ :=
  (wrongIn H Cm s Pt Pρ w ρ t (ι ρ)).card

/-! ## 3. The acceptance probability at the footprint -/

/-- **A cell's escape** (the hypergeometric miss): a uniform `min(k, n)`-subset of `n` proof blocks misses `m` of them
with probability `C(n − m, k')/C(n, k')`, `k' = min(k, n)`. -/
noncomputable def escCell (n k m : ℕ) : ℝ := ((n - m).choose (min k n) : ℝ) / (n.choose (min k n) : ℝ)

/-- **Replay block `ρ`'s escape** at a footprint `fp`: `e_ρ = ∏_t C(n_{ρ,t} − fp(ρ, t), k_t)/C(n_{ρ,t}, k_t)`, the
probability that the second stage samples no incorrect proof block of a sampled `ρ`. -/
noncomputable def Design.Two.esc (d : Design.Two) (Pt : Partition) (fp : ℕ → ℕ → ℕ) (ρ : ℕ) : ℝ :=
  ∏ t ∈ Finset.range Pt.strata.length, escCell (cell Pt d.part ρ t).card (d.k.getD t 0) (fp ρ t)

/-- **The acceptance probability at a footprint**, `P̄_a(fp) = E_S ∏_{ρ∈S} e_ρ` over the first stage's sample `S`.
Catching each replay block independently with probability `1 − e_ρ`, it is the probability that `S` misses the caught
ones, `Σ_T P_a(T)·∏_{ρ∈T}(1 − e_ρ)·∏_{ρ∉T} e_ρ` at the first stage's `P_a`. One-stage's `P_a(B)` is its case at
`B`'s indicator, replay blocks the proof blocks and `k = 1`. -/
noncomputable def Design.Two.Pa (d : Design.Two) (Pt : Partition) (fp : ℕ → ℕ → ℕ) : ℝ :=
  ∑ T ∈ (Finset.range d.part.blocks.length).powerset,
    d.replay.Pa d.part T *
      ((∏ ρ ∈ T, (1 - d.esc Pt fp ρ)) * ∏ ρ ∈ Finset.range d.part.blocks.length \ T, d.esc Pt fp ρ)

/-- **The confidence set** `M̄_δ`: every footprint `fp` with `P̄_a(fp) > δ`. -/
def Design.Two.M (d : Design.Two) (Pt : Partition) (δ : ℝ) : Set (ℕ → ℕ → ℕ) := {fp | δ < d.Pa Pt fp}

/-- **The first stage catches each replay block at rate `r`**: it misses every set `T` of replay blocks with
probability at most `(1 − r)^{|T|}`, as independent catches at rate `r` would. `r` is a hypothesis: a first stage that
samples a uniform `Q`-subset of the population's `N` replay blocks and every input replay block has it at `r = Q/N`
(`Design.Two.firstRate_subsetInputs`), and a two-stage rule's replay design is a `rate:NUM/DEN` design, whose count is
at least `NUM/DEN` of whatever replay blocks it is taken over. -/
def Design.Two.FirstRate (d : Design.Two) (r : ℝ) : Prop :=
  ∀ T ⊆ Finset.range d.part.blocks.length, d.replay.Pa d.part T ≤ (1 - r) ^ T.card

/-- **The second stage samples each cell at rate `r`**: in every nonempty cell `(ρ, t)` it samples
`min(k_t, n_{ρ,t}) ≥ r·n_{ρ,t}` proof blocks. `r` is a hypothesis that reads no rule's rate: a two-stage rule names
the counts `k_t`, each fixed, so the largest `r` that holds is the least `min(k_t, n_{ρ,t}) / n_{ρ,t}` over the cells,
which the cells' sizes set. -/
def Design.Two.SecondRate (d : Design.Two) (Pt : Partition) (r : ℝ) : Prop :=
  ∀ ρ < d.part.blocks.length, ∀ t < Pt.strata.length, 0 < (cell Pt d.part ρ t).card →
    r * (cell Pt d.part ρ t).card ≤ ((min (d.k.getD t 0) (cell Pt d.part ρ t).card : ℕ) : ℝ)

/-- **The replay blocks a footprint marks incorrect**: those with an incorrect proof block in the cell of some
stratum. -/
def Design.Two.wrongBlocks (d : Design.Two) (Pt : Partition) (fp : ℕ → ℕ → ℕ) : Finset ℕ :=
  (Finset.range d.part.blocks.length).filter fun ρ => ∃ t < Pt.strata.length, 0 < fp ρ t

/-- **The two-stage relation** at `δ`, on the statement's bytes: it parses, its proof-block partition and two-stage
design evaluate (`partition`, `two`, the verifier's own), and at some interior values, one interior per replay block,
the footprint of the boundary trace `w` opens is in `M̄_δ`. The interiors are the relation's to pick: serving commits
only the boundary. -/
def RTwo (H : Msg → Msg) (Cm : CommitmentScheme Bool) (parse : Msg → Option AuditStatement)
    (partition : AuditStatement → Option Partition) (two : AuditStatement → Option Design.Two) (δ : ℝ) : Rel Wit :=
  fun x w => ∃ s Pt d, parse x = some s ∧ partition s = some Pt ∧ two s = some d ∧
    ∃ ι : ℕ → ℕ → Bool, footprint H Cm s Pt d.part w ι ∈ d.M Pt δ

/-- **The two-stage scope, readable**: a topologically ordered program, partitioned into proof blocks (`Partitions`),
each within a block of the replay partition, whose blocks are disjoint, and a well-formed two-stage design. -/
def InScopeTwo (Pt : Partition) (d : Design.Two) (s : AuditStatement) : Prop :=
  s.prog.Topological ∧ Pt.Partitions s.prog ∧ d.part.gates.Nodup ∧
    (∀ v < Pt.blocks.length, (replayOf Pt d.part v).isSome) ∧ d.WellFormed Pt

/-! ## 7. What the incorrect proof blocks can serve -/

/-- `Y`'s proof blocks in cell `(ρ, t)`: the footprint of a set of proof blocks. -/
def counts (Pt Pρ : Partition) (Y : Finset ℕ) (ρ t : ℕ) : ℕ := (Y ∩ cell Pt Pρ ρ t).card

open Classical in
/-- **The admissible sets** at `δ`: the sets `T` of proof blocks whose counts per cell are a footprint in `M̄_δ`, the
incorrect sets a boundary trace `RTwo` takes can have. -/
noncomputable def admissibleTwo (Pt : Partition) (d : Design.Two) (δ : ℝ) : Finset (Finset ℕ) :=
  (Finset.range Pt.blocks.length).powerset.filter fun T => counts Pt d.part T ∈ d.M Pt δ

/-- **The committed bits** `b(T) = Σ_{v∈T} |out(v)|` of a set `T` of proof blocks: the values each commits. -/
def bitsOf (s : AuditStatement) (Pt : Partition) (T : Finset ℕ) : ℕ :=
  ∑ v ∈ T, committed s.prog (Pt.blocks.getD v [])

/-- The values `w` opens at the program's input gates, `none` at every other gate. -/
def inputsOf (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (w : Wit) (g : ℕ) : Option Bool :=
  if s.prog.gates[g]? = some .input then w.val H Cm s g else none

/-- **The served outputs** of `w`: the values it opens at the program's returned gates, a constant's its own
(`Prog.read`), an unopened one `false`. -/
def served (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (w : Wit) :
    Fin s.prog.outs.length → Bool :=
  fun i => (s.prog.read (w.val H Cm s) s.prog.outs[i]).getD false

open Classical in
/-- **The influence set** at `δ` and input values `a`: the served outputs of every boundary trace with inputs `a` whose
footprint is in `M̄_δ` at some interiors, the outputs `RTwo` lets a prover serve on those inputs. -/
noncomputable def influenceTwo (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (Pt : Partition)
    (d : Design.Two) (δ : ℝ) (a : ℕ → Option Bool) : Finset (Fin s.prog.outs.length → Bool) :=
  Finset.univ.filter fun y => ∃ w : Wit, inputsOf H Cm s w = a ∧
    (∃ ι : ℕ → ℕ → Bool, footprint H Cm s Pt d.part w ι ∈ d.M Pt δ) ∧ served H Cm s w = y

/-- **The influence relation** at `δ`, on the statement's bytes: it parses, its partitions evaluate, and the served
outputs of `w` are in the influence set at `w`'s own inputs. Every boundary trace in `RTwo` is in it. -/
def RInflTwo (H : Msg → Msg) (Cm : CommitmentScheme Bool) (parse : Msg → Option AuditStatement)
    (partition : AuditStatement → Option Partition) (two : AuditStatement → Option Design.Two) (δ : ℝ) : Rel Wit :=
  fun x w => ∃ s Pt d, parse x = some s ∧ partition s = some Pt ∧ two s = some d ∧
    served H Cm s w ∈ influenceTwo H Cm s Pt d δ (inputsOf H Cm s w)

end Claims.Sampling

namespace Claims.SamplingGeneric

open Claims
open Claims.Sampling (AuditStatement Partition InScopeTwo RTwo RInflTwo influenceTwo admissibleTwo bitsOf cell escCell
  replayOf sampleBlocks footBits forkOn KSLinOn Bext)

/-! ## 4. The compiled verifier -/

/-- **A compiled two-stage sampled verifier over `A`**: a compiled sampled verifier (`Compiled`) with the two-stage
design it evaluates on a statement (`two`) and the replay blocks the first stage samples at a run's coins
(`replaySample`). Its second stage is `stage2`, the rounds after the interior registration, whose sample is the
proof blocks `checked`; `iroots x v ρ` is the roots the run `v` shows for replay block `ρ`'s interior values. -/
structure CompiledTwo (Coin : Type) (A : Arg Coin) extends Compiled Coin A where
  two : AuditStatement → Option Claims.Sampling.Design.Two
  replaySample : (x : Msg) → (Fin (V.k x) → Coin) → List ℕ

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The two-stage scope, on the statement's bytes**: it parses, its proof-block partition and two-stage design
evaluate, and they are in scope (`InScopeTwo`). -/
def Scope (L : CompiledTwo Coin A) (x : Msg) : Prop :=
  ∃ s Pt d, L.parse x = some s ∧ L.partition s = some Pt ∧ L.two s = some d ∧ InScopeTwo Pt d s

/-! ## 5. The contract -/

/-- **The two-stage compiler's contract** at `H`: `Compiled.Sessions` with its draw law split by stage and its openings
read at a replay block.

* `replayDraws`: on a statement in scope, the first stage's sample misses a set `T` of replay blocks with probability
  at most the first stage's `P_a(T)`.
* `secondDraws`: on a statement in scope, at a run's coins `r`, fresh second-stage coins sample no proof block of a set
  `X` with probability at most `∏_{ρ∈S} ∏_t C(n_{ρ,t} − |X ∩ cell(ρ, t)|, k_t)/C(n_{ρ,t}, k_t)`, `S` the replay
  blocks `r` samples: the second stage's sample is, in each sampled replay block, a uniform `k_t`-subset of each cell,
  independently.
* `slots`, `shared`, `refuses` (at `CompiledTwo.Scope`), `fixed` and `runs`: `Sessions`'.
* `unions`: on a statement in scope, both waits' unions are within their bits: the outer wait's, over the sets' counts
  per cell, `M ≤ n_out` (`footBits`), and the inner wait's, over sets of the proof blocks of the replay blocks `r`
  samples, `|sampleBlocks| ≤ n_in` at every `r` of which some second-stage fork is accepted.
* `interior`: the roots a run shows for a replay block's interior values are the same at fresh second-stage coins.
* `opens`: on a statement in scope, at an accepted run that samples `u` and shows its session, a witness in `A`'s
  relation at the session's statement reveals openings such that any values and interior values agreeing with those
  that pass (under the statement's roots, and under the interior roots the run showed for `u`'s replay block `ρ`)
  make proof block `u` correct on `ρ`'s values (`Prog.refIn` at `ρ`).
* `keys`: `Sessions`', the sample's slots hold at most `n_sess` keys, so the extractor runs `A`'s extractor at most
  `n_sess` times per run (`CompilerEfficientTwo` reads it). -/
structure Sessions (H : Msg → Msg) (L : CompiledTwo Coin A) : Prop where
  replayDraws : ∀ x s Pt d, L.parse x = some s → L.partition s = some Pt → L.two s = some d → InScopeTwo Pt d s →
    ∀ T ⊆ Finset.range d.part.blocks.length,
      Pr (fun r : Fin (L.V.k x) → Coin => ∀ ρ ∈ L.replaySample x r, ρ ∉ T) ≤ d.replay.Pa d.part T
  secondDraws : ∀ x s Pt d, L.parse x = some s → L.partition s = some Pt → L.two s = some d → InScopeTwo Pt d s →
    ∀ (r : Fin (L.V.k x) → Coin) (X : Finset ℕ),
      Pr (fun e : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x (forkOn (L.stage2 x) r e), u ∉ X) ≤
        ∏ ρ ∈ (Finset.range d.part.blocks.length).filter (· ∈ L.replaySample x r),
          ∏ t ∈ Finset.range Pt.strata.length,
            escCell (cell Pt d.part ρ t).card (d.k.getD t 0) (X ∩ cell Pt d.part ρ t).card
  slots : ∀ x r, (L.checked x r).length ≤ L.nPU x
  shared : ∀ x (Ω : Type) (P : Prover Ω) ω r i j u u', (run H P L.V x ω r).out = .acc →
    (L.checked x r)[i]? = some u → (L.checked x r)[j]? = some u' →
    L.skey x (L.checked x r) i = L.skey x (L.checked x r) j →
    L.session x u (run H P L.V x ω r).seen = L.session x u' (run H P L.V x ω r).seen
  unions : ∀ x s Pt d, L.parse x = some s → L.partition s = some Pt → L.two s = some d → InScopeTwo Pt d s →
    footBits Pt d ≤ L.nOut x ∧ ∀ (Ω : Type) (P : Prover Ω) ω (r e : Fin (L.V.k x) → Coin),
      (run H P L.V x ω (forkOn (L.stage2 x) r e)).out = .acc →
        (sampleBlocks Pt d.part (L.replaySample x r)).card ≤ L.nIn x
  refuses : ∀ x (Ω : Type) (P : Prover Ω) ω r, (run H P L.V x ω r).out = .acc → L.Scope x
  interior : ∀ x (Ω : Type) (P : Prover Ω) ω r e ρ,
    L.iroots x (run H P L.V x ω (forkOn (L.stage2 x) r e)).seen ρ = L.iroots x (run H P L.V x ω r).seen ρ
  fixed : ∀ x r u, u ∈ L.checked x r →
    L.rounds x u ⊆ L.stage2 x ∧ ∀ e, L.checked x (forkOn (L.rounds x u) r e) = L.checked x r
  runs : ∀ x (Ω : Type) (P : Prover Ω) (ω : Ω) r u, u ∈ L.checked x r →
    ∃ (xS : Msg) (P' : Prover Unit), A.shape xS = L.sh x u ∧ A.κ xS ≤ L.κb x u ∧ A.γ xS ≤ L.γb x u ∧
      A.V.k xS = (L.rounds x u).card ∧
      (∀ r' : Fin (L.V.k x) → Coin, (∀ t ∉ L.rounds x u, r' t = r t) →
        (run H P L.V x ω r').out = .acc → (run H P' A.V xS () (L.coinsOf x u xS r')).out = .acc) ∧
      ∀ r' : Fin (L.V.k x) → Coin, (∀ t ∉ L.rounds x u, (∀ t' ∈ L.rounds x u, t < t') → r' t = r t) →
        L.session x u (run H P L.V x ω r').seen = some (xS, (run H P' A.V xS () (L.coinsOf x u xS r')).seen) ∧
          ∀ q ∈ (run H P' A.V xS () (L.coinsOf x u xS r')).vLog, q ∈ (run H P L.V x ω r').vLog
  opens : ∀ x s Pt d (Ω : Type) (Pv : Prover Ω) ω r u xS vS w,
    L.parse x = some s → L.partition s = some Pt → L.two s = some d → InScopeTwo Pt d s →
    (run H Pv L.V x ω r).out = .acc → u ∈ L.checked x r → L.session x u (run H Pv L.V x ω r).seen = some (xS, vS) →
    A.R xS w →
    ∃ b ρ rb, Pt.blocks[u]? = some b ∧ replayOf Pt d.part u = some ρ ∧ d.part.blocks[ρ]? = some rb ∧
      ∀ (val : ℕ → Option Bool) (ι : ℕ → Bool),
        (∀ q ∈ L.reveal s u (run H Pv L.V x ω r).seen w, L.commit.Opens H q.1 q.2 = true →
          (q.1 = s.roots → val q.2.pos = some q.2.val) ∧
          (q.1 = L.iroots x (run H Pv L.V x ω r).seen ρ → ι q.2.pos = q.2.val)) →
        s.prog.Correct (s.prog.refIn rb val ι) b
  keys : ∀ x r, ((Finset.range (L.checked x r).length).image (L.skey x (L.checked x r))).card ≤ L.nSess x

/-! ## 6. The claims -/

/-- **KS at the footprint**, on the statements in scope, at every `δ ≥ 0`: `Pr[acc ∧ no interiors put the footprint in
M̄_δ] ≤ δ + ε_ks + ε_bind`, at `Compiled`'s extractor, `κ_δ`, `γ` and `ε_bind = Pr[B_ext]` (`Compiled.KS` at
`RTwo`). -/
def KS (H : Msg → Msg) (L : CompiledTwo Coin A) : Prop :=
  ∀ δ : ℝ, 0 ≤ δ →
    KSLinOn H L.V L.Scope (RTwo H L.commit L.parse L.partition L.two δ) L.E (L.κ δ) L.γ (fun _ => 1) (Bext H L.V L.E)

/-- **The exfiltration bound at the footprint**, on the statements in scope, at every `δ ≥ 0`:
`Pr[acc ∧ the served outputs are outside the influence set] ≤ δ + ε_ks + ε_bind` at `KS`'s extractor and errors
(`RInflTwo`), and the influence set at any input values has at most `Σ_{T ∈ admissible} 2^{b(T)}` members: a set `T` of
incorrect proof blocks can serve no more than its committed bits choose. -/
def Exfil (H : Msg → Msg) (L : CompiledTwo Coin A) : Prop :=
  ∀ δ : ℝ, 0 ≤ δ →
    KSLinOn H L.V L.Scope (RInflTwo H L.commit L.parse L.partition L.two δ) L.E (L.κ δ) L.γ (fun _ => 1)
        (Bext H L.V L.E) ∧
      ∀ x s Pt d (a : ℕ → Option Bool), L.parse x = some s → L.partition s = some Pt → L.two s = some d →
        InScopeTwo Pt d s →
        (influenceTwo H L.commit s Pt d δ a).card ≤ ∑ T ∈ admissibleTwo Pt d δ, 2 ^ bitsOf s Pt T

/-- **KS at the footprint read at a time budget** (`Compiled.KSAt` at `RTwo`), on the statements in scope, at every
`δ ≥ 0`: `Pr[acc ∧ no interiors put the footprint in M̄_δ]` is at most
`κ_δ + √γ·sbB(2T) + ecrBound(B_ext at T, 2·T·work)` for every prover and every budget `T` on one run's SHA-512 work
(`Sampling.KSAt`), and `B_ext` at `T` costs at most `2·T·work` in expectation (`Sampling.BextAtCostLe`), at
`Compiled`'s extractor, errors and `work`, whose waits' counts `k_out` and `k_in` are the two-stage verifier's own. -/
def KSAt (H : Msg → Msg) (L : CompiledTwo Coin A) : Prop :=
  (∀ δ : ℝ, 0 ≤ δ →
    Sampling.KSAt H L.V L.Scope (RTwo H L.commit L.parse L.partition L.two δ) L.E (L.κ δ) L.γ L.work) ∧
    Sampling.BextAtCostLe H L.V L.E L.work

end CompiledTwo

/-- **The two-stage compiler is knowledge sound at the footprint**: over every proof system for proof blocks that is
knowledge sound, every compiled two-stage verifier whose commitment binds and that meets the contract is, at the
one-stage compiler's errors. -/
def CompilerKSTwo : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : CompiledTwo Coin A)
    (H : Msg → Msg), A.KS H → L.commit.Binds H → L.Sessions H → L.KS H

/-- **The two-stage compiler bounds exfiltration at the footprint**: at `CompilerKSTwo`'s premises, every compiled
two-stage verifier meets `CompiledTwo.Exfil`. -/
def CompilerExfilTwo : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : CompiledTwo Coin A)
    (H : Msg → Msg), A.KS H → L.commit.Binds H → L.Sessions H → L.Exfil H

/-- **The two-stage compiler's KS read at a time budget**: at `CompilerKSTwo`'s premises and the opening checks' count,
every compiled two-stage verifier meets `CompiledTwo.KSAt` (a corollary of `CompilerKSTwo` and
`CompilerEfficientTwo`). -/
def CompilerKSAtTwo : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : CompiledTwo Coin A)
    (H : Msg → Msg), A.KS H → L.commit.Binds H → L.Sessions H → L.OpenCallsLe H → L.KSAt H

/-- **The two-stage extractor's cost**: the two-stage extractor is `Compiled.E`, so every compiled two-stage verifier
that meets the contract has `Compiled.Efficient`'s three bounds at its own waits' counts, from the opening checks'
count: reruns at most `1 + k_out + k_out·k_in·(1 + n_sess·e_s)` in expectation, and own SHA-512 calls at most
`k_out·k_in·n_PU·openCalls`. -/
def CompilerEfficientTwo : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : CompiledTwo Coin A)
    (H : Msg → Msg), L.Sessions H → L.OpenCallsLe H → L.Efficient H

end Claims.SamplingGeneric
