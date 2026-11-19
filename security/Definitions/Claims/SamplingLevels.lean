module

public import Definitions.Claims.SamplingTwoReg
public import Definitions.Claims.LevelsProofBlocks

/-!
# Claims: sampling at levels, one statement

SPOT's audit at any number of levels, on `Claims.Sampling`'s statement, witness and commitment and
`Claims.SamplingGeneric`'s compiled verifier, read as they stand, with core's levels (`Verity.Levels`, the page's
Definition 3.13) as its design. The one-level audit (`Claims.Sampling.R`, `Compiled.KS`, `Compiled.KSReg`) and the
two-level audit (`RTwo`, `CompiledTwo.KS`, `CompiledTwo.KSRegTwo`) are its cases at one level and at two, where core's
levels realize their claims designs (`Proofs/Flock/SamplingGeneric/Levels.lean`).

1. **The levels on a statement**: core's levels over the proof blocks' slots, `Fin |Pt.blocks|`, whose last level's
   blocks are the slots one by one (`Verity.Levels.SlotsAreBlocks` at `slot`), and the top partition (`top`), a claims
   partition whose blocks hold the committed interiors: none at one level, where a proof block is its own top block;
   at two levels or more, the partition whose blocks are the first level's (`TopIs`).
2. **The miss**: the level-by-level sample `Verity.Levels.design` misses a set `B` of proof blocks with probability
   `missProb (B.image slot)`, and `M_δ` (`levM`) is every set it misses with probability above `δ`.
3. **The relation** (`RLev`): at some interior values, one per top block, the proof blocks incorrect on their top
   block's values (`incorrectLev`) are in `M_δ`.
4. **The compiled verifier** (`CompiledLev`): a `Compiled` with the levels and the top partition it evaluates on a
   statement, the samples of the levels above the last at a run's coins (`upper`), and the rounds of each level's
   sample between the first and the last (`stages`; the last level's are `secondStage`). Its contract
   (`CompiledLev.Sessions`) is `Compiled.Sessions` with its sampling clause read level by level, each level's sample
   drawn on its own rounds from the core level's design at each block the level above sampled, and the interiors
   fixed before any level below the first; its claims are `CompiledLev.KS` and the committed form
   `CompiledLev.KSRegLev`.
5. **The compiler's statements**: `CompilerKSLev`, `CompilerKSRegLev`, and `CompilerLev`, both.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.Sampling

/-! ## 1. The levels on a statement -/

/-- **Proof block `v` as a block of the slots**: `{v}` among `Fin |Pt.blocks|`, empty past the last slot. -/
def slot (Pt : Partition) (v : ℕ) : Finset (Fin Pt.blocks.length) := Finset.univ.filter fun i => i.val = v

/-- **The top block of proof block `v`**: `v` itself with no top partition; otherwise the block of the top partition
that holds it (`replayOf`). -/
def topOf (Pt : Partition) : Option Partition → ℕ → Option ℕ
  | none, v => some v
  | some Q, v => replayOf Pt Q v

/-- **The top partition is the levels' first**: with none, one level; with `Q`, two levels or more, `Q`'s blocks
disjoint, each proof block in one of them, and two proof blocks in one first-level block exactly when they are in one
block of `Q`. -/
def TopIs (Pt : Partition) (L : Verity.Levels (Fin Pt.blocks.length)) : Option Partition → Prop
  | none => L.levels.length = 1
  | some Q => Q.gates.Nodup ∧ (∀ v < Pt.blocks.length, (replayOf Pt Q v).isSome) ∧
      ∃ l rest, L.levels = l :: rest ∧ rest ≠ [] ∧
        ∀ v w : Fin Pt.blocks.length, replayOf Pt Q v = replayOf Pt Q w ↔ ∃ X ∈ l.blocks.parts, v ∈ X ∧ w ∈ X

/-! ## 2. The miss -/

/-- **The confidence set at levels** `M_δ`: every set `B` of proof blocks that the level-by-level sample misses with
probability above `δ`. -/
def levM (Pt : Partition) (L : Verity.Levels (Fin Pt.blocks.length)) (δ : ℝ) : Set (Finset ℕ) :=
  {B | δ < (L.design.missProb (B.image (slot Pt))).toReal}

/-! ## 3. The relation -/

open Classical in
/-- **The incorrect proof blocks at boundary values `val` and interior values `ι`**, one interior per top block: each
proof block incorrect on its top block's values, the boundary's `val` and the interior's `ι ρ` (`Prog.refIn` at the
top block). -/
noncomputable def incorrectLev (s : AuditStatement) (Pt : Partition) (top : Option Partition) (val : ℕ → Option Bool)
    (ι : ℕ → ℕ → Bool) : Finset ℕ :=
  (Finset.range Pt.blocks.length).filter fun v => ¬ ∃ u ρ r, Pt.blocks[v]? = some u ∧ topOf Pt top v = some ρ ∧
    (top.getD Pt).blocks[ρ]? = some r ∧ s.prog.Correct (s.prog.refIn r val (ι ρ)) u

/-- **The relation at levels** at `δ`, on the statement's bytes: it parses, its proof-block partition, levels and top
partition evaluate (`partition`, `levels`, `top`, the verifier's own), and at some interior values, one per top block,
the proof blocks incorrect on the boundary trace `w` opens are in `M_δ`. The interiors are the relation's to pick:
serving commits only the boundary. -/
def RLev (H : Msg → Msg) (Cm : CommitmentScheme Bool) (parse : Msg → Option AuditStatement)
    (partition : AuditStatement → Option Partition)
    (levels : (s : AuditStatement) → (Pt : Partition) → Option (Verity.Levels (Fin Pt.blocks.length)))
    (top : AuditStatement → Option Partition) (δ : ℝ) : Rel Wit :=
  fun x w => ∃ s Pt L, parse x = some s ∧ partition s = some Pt ∧ levels s Pt = some L ∧
    ∃ ι : ℕ → ℕ → Bool, incorrectLev s Pt (top s) (w.val H Cm s) ι ∈ levM Pt L δ

/-! ## 4. Scope -/

/-- **The scope at levels, readable**: a topologically ordered program, partitioned into proof blocks (`Partitions`),
levels whose last level's blocks are the proof blocks one by one, and a top partition that is the levels' first. -/
def InScopeLev (Pt : Partition) (L : Verity.Levels (Fin Pt.blocks.length)) (top : Option Partition)
    (s : AuditStatement) : Prop :=
  s.prog.Topological ∧ Pt.Partitions s.prog ∧ L.SlotsAreBlocks Pt (slot Pt) ∧ TopIs Pt L top

/-! ## 5. Where core's levels realize the claims designs -/

/-- **Top block `ρ`'s slots**: the proof blocks of `Pt` whose block in the top partition `Q` is `ρ`. -/
def topBlock (Pt Q : Partition) (ρ : ℕ) : Finset (Fin Pt.blocks.length) :=
  Finset.univ.filter fun v => replayOf Pt Q v = some ρ

/-- **One level realizes the claims design**: at every statement in scope (`InScope`), some one level whose blocks are
the proof blocks misses each set of them with the statement's design's `P_a` (`Verity.Levels.MissesAsDesign`). -/
def LevelsRealizeOne : Prop :=
  ∀ (s : AuditStatement) (Pt : Partition), InScope Pt s → ∃ L : Verity.Levels (Fin Pt.blocks.length),
    TopIs Pt L none ∧ L.SlotsAreBlocks Pt (slot Pt) ∧ L.MissesAsDesign s.design Pt (slot Pt)

/-- **Two levels realize the two-stage design `d`** on `Pt`: their first level's blocks are `d`'s replay blocks
(`TopIs`), their last the proof blocks; their sample misses each set `B` of proof blocks with `Design.Two.Pa` at
`B`'s counts per cell; the first level's design misses a set `T` of its blocks with probability at least the replay
design's `P_a` at the nonempty replay blocks in `T`; and the second level's design at a replay block's proof blocks
misses `X`'s with probability at least `∏_t C(n_{ρ,t} − |X ∩ cell|, k_t)/C(n_{ρ,t}, k_t)`. -/
def Design.Two.RealizedBy (d : Design.Two) (Pt : Partition) (L : Verity.Levels (Fin Pt.blocks.length)) : Prop :=
  TopIs Pt L (some d.part) ∧ L.SlotsAreBlocks Pt (slot Pt) ∧
    (∀ B ⊆ Finset.range Pt.blocks.length,
      (L.design.missProb (B.image (slot Pt))).toReal = d.Pa Pt (counts Pt d.part B)) ∧
    ∃ l₁ l₂, L.levels = [l₁, l₂] ∧
      (∀ T, d.replay.Pa d.part ((Finset.range d.part.blocks.length).filter fun ρ =>
          (topBlock Pt d.part ρ).Nonempty ∧ topBlock Pt d.part ρ ∈ T) ≤
        ((l₁.design Finset.univ).missProb T).toReal) ∧
      ∀ ρ < d.part.blocks.length, (topBlock Pt d.part ρ).Nonempty → ∀ X : Finset ℕ,
        ∏ t ∈ Finset.range Pt.strata.length,
            escCell (cell Pt d.part ρ t).card (d.k.getD t 0) (X ∩ cell Pt d.part ρ t).card ≤
          ((l₂.design (topBlock Pt d.part ρ)).missProb (X.image (slot Pt))).toReal

/-- **Two levels realize the two-stage designs**: at every statement in scope at two stages (`InScopeTwo`), some
two levels realize its design (`Design.Two.RealizedBy`). -/
def LevelsRealizeTwo : Prop :=
  ∀ (s : AuditStatement) (Pt : Partition) (d : Design.Two), InScopeTwo Pt d s →
    ∃ L : Verity.Levels (Fin Pt.blocks.length), d.RealizedBy Pt L

end Claims.Sampling

namespace Claims.SamplingGeneric

open Claims
open Claims.Sampling (AuditStatement Partition InScopeLev RLev slot topOf incorrectLev levM forkOn KSLinOn
  Bext incorrectInAt waitAccC flatMapRP assemble slack Opening)

/-! ## 5. The compiled verifier -/

/-- **A compiled sampled verifier at levels over `A`**: a compiled sampled verifier (`Compiled`) with the levels it
evaluates on a statement and its partition (`levels`), the top partition (`top`), the samples of the levels above the
last at a run's coins (`upper`, the `j`-th the blocks level `j` samples; the last level's sample is `checked`), and the
rounds of each level's sample between the first and the last (`stages`, the `j`-th the rounds of level `j + 1`'s; the
first level's are the rounds off every other, the last level's `secondStage`). -/
structure CompiledLev (Coin : Type) (A : Arg Coin) extends Compiled Coin A where
  levels : (s : AuditStatement) → (Pt : Partition) → Option (Verity.Levels (Fin Pt.blocks.length))
  top : AuditStatement → Option Partition
  upper : (x : Msg) → (Pt : Partition) → (Fin (V.k x) → Coin) → List (Finset (Finset (Fin Pt.blocks.length)))
  stages : (x : Msg) → ℕ → Finset (Fin (V.k x))

namespace CompiledLev

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The scope at levels, on the statement's bytes**: it parses, its proof-block partition and levels evaluate, and
they are in scope with its top partition (`InScopeLev`). -/
def Scope (L : CompiledLev Coin A) (x : Msg) : Prop :=
  ∃ s Pt Lv, L.parse x = some s ∧ L.partition s = some Pt ∧ L.levels s Pt = some Lv ∧ InScopeLev Pt Lv (L.top s) s

/-! ## 6. The contract -/

/-- **The samples, level by level**, on a statement with partition `Pt` and levels `Lv`.

* One level: as `Compiled.Sessions.draws`, for each set `B` of proof blocks a weight of the coins bounds the
  probability over fresh second-stage coins that the sample misses `B`, and averages at most `Lv`'s miss of `B`.
* The first of two or more: the run's coins sample level `0`'s blocks (`upper`'s head) missing a set `T` of them with
  probability at most the first level's design's miss of `T`.
* Each middle level `j + 1`: at a run's coins `r`, fresh coins on its rounds (`stages j`) give the product of any
  escapes `e ≤ 1` of its sampled blocks at most, in expectation, the product over level `j`'s sampled blocks `X` of
  the expectation of the product over a draw of level `j + 1`'s design at `X`, and leave the samples above it as they
  are.
* The last of two or more: at a run's coins `r`, fresh second-stage coins sample no proof block of a set `X` with
  probability at most the product, over the blocks `Y` the level above sampled, of the last level's design's miss at
  `Y` of `X`'s slots.
* Every run's coins give one sample per level above the last: `upper` is empty at one level, and its last entry is the
  sample of the level above the last. -/
def Samples (L : CompiledLev Coin A) (x : Msg) (Pt : Partition) (Lv : Verity.Levels (Fin Pt.blocks.length)) : Prop :=
  (Lv.levels.length = 1 → ∀ B ⊆ Finset.range Pt.blocks.length, ∃ esc : (Fin (L.V.k x) → Coin) → ℝ,
      (∀ r, 0 ≤ esc r) ∧
      (∀ r, Pr (fun d : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x (forkOn (L.secondStage x) r d), u ∉ B) ≤
        esc r) ∧ Ex esc ≤ (Lv.design.missProb (B.image (slot Pt))).toReal) ∧
  (∀ l l' rest, Lv.levels = l :: l' :: rest → ∀ T : Finset (Finset (Fin Pt.blocks.length)),
      Pr (fun r : Fin (L.V.k x) → Coin => Disjoint ((L.upper x Pt r).headD ∅) T) ≤
        ((l.design Finset.univ).missProb T).toReal) ∧
  (∀ (j : ℕ) l', Lv.levels[j + 1]? = some l' → j + 2 < Lv.levels.length →
      ∀ (r : Fin (L.V.k x) → Coin) (e : Finset (Fin Pt.blocks.length) → ENNReal), (∀ Y, e Y ≤ 1) →
        Ex (fun d : Fin (L.V.k x) → Coin =>
            (∏ Y ∈ (L.upper x Pt (forkOn (L.stages x j) r d)).getD (j + 1) ∅, e Y).toReal) ≤
          (∏ X ∈ (L.upper x Pt r).getD j ∅,
            Verity.Game.avg fun ω => ∏ Y ∈ (l'.design X).draw ω, e Y).toReal ∧
        ∀ d i, i ≤ j → (L.upper x Pt (forkOn (L.stages x j) r d))[i]? = (L.upper x Pt r)[i]?) ∧
  (∀ lm, Lv.levels.getLast? = some lm → 2 ≤ Lv.levels.length →
      ∀ (r : Fin (L.V.k x) → Coin) (X : Finset ℕ),
        Pr (fun e : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x (forkOn (L.secondStage x) r e), u ∉ X) ≤
          ∏ Y ∈ (L.upper x Pt r).getLastD ∅, ((lm.design Y).missProb (X.image (slot Pt))).toReal) ∧
  ∀ r, (L.upper x Pt r).length + 1 = Lv.levels.length

/-- **The waits' unions**: the outer wait's, and each middle level's wait's, over sets of the `n_PU` proof blocks
within `n_out` bits; with the top partition, also the inner wait's over the proof blocks inside the blocks the level
above the last sampled, `≤ n_in` at every `r` of which some second-stage fork is accepted. -/
def Unions (H : Msg → Msg) (L : CompiledLev Coin A) (x : Msg) (Pt : Partition) : Option Partition → Prop
  | none => L.nPU x ≤ L.nOut x
  | some _ => L.nPU x ≤ L.nOut x ∧ ∀ (Ω : Type) (P : Prover Ω) ω (r e : Fin (L.V.k x) → Coin),
      (run H P L.V x ω (forkOn (L.secondStage x) r e)).out = .acc →
        (((L.upper x Pt r).getLastD ∅).biUnion id).card ≤ L.nIn x

/-- **The compiler's contract at levels** at `H`: `Compiled.Sessions` with its sampling clause read level by level
(`Samples`) and its openings read at a top block.

* `samples`: `Samples`, on a statement in scope.
* `unions`: `Unions`, on a statement in scope.
* `slots`, `shared`, `keys`, `refuses` (at `CompiledLev.Scope`), `fixed` and `runs`: `Compiled.Sessions`'.
* `interior`: the interior roots a run shows are the same under fresh coins on the rounds of every level below the
  first, the last's included: the prover fixes them after the first level's coins.
* `opens`: on a statement in scope, at an accepted run that samples `u` and shows its session, a witness in `A`'s
  relation at the session's statement reveals openings such that any values and interior values agreeing with those
  that pass (under the statement's roots, and under the interior roots the run showed for `u`'s top block `ρ`) make
  proof block `u` correct on `ρ`'s values (`Prog.refIn` at `ρ`). -/
structure Sessions (H : Msg → Msg) (L : CompiledLev Coin A) : Prop where
  samples : ∀ x s Pt Lv, L.parse x = some s → L.partition s = some Pt → L.levels s Pt = some Lv →
    InScopeLev Pt Lv (L.top s) s → L.Samples x Pt Lv
  unions : ∀ x s Pt Lv, L.parse x = some s → L.partition s = some Pt → L.levels s Pt = some Lv →
    InScopeLev Pt Lv (L.top s) s → L.Unions H x Pt (L.top s)
  slots : ∀ x r, (L.checked x r).length ≤ L.nPU x
  shared : ∀ x (Ω : Type) (P : Prover Ω) ω r i j u u', (run H P L.V x ω r).out = .acc →
    (L.checked x r)[i]? = some u → (L.checked x r)[j]? = some u' →
    L.skey x (L.checked x r) i = L.skey x (L.checked x r) j →
    L.session x u (run H P L.V x ω r).seen = L.session x u' (run H P L.V x ω r).seen
  keys : ∀ x r, ((Finset.range (L.checked x r).length).image (L.skey x (L.checked x r))).card ≤ L.nSess x
  refuses : ∀ x (Ω : Type) (P : Prover Ω) ω r, (run H P L.V x ω r).out = .acc → L.Scope x
  interior : ∀ x (Ω : Type) (P : Prover Ω) ω r e ρ,
    L.iroots x (run H P L.V x ω (forkOn (L.secondStage x) r e)).seen ρ = L.iroots x (run H P L.V x ω r).seen ρ ∧
      ∀ j, L.iroots x (run H P L.V x ω (forkOn (L.stages x j) r e)).seen ρ = L.iroots x (run H P L.V x ω r).seen ρ
  fixed : ∀ x r u, u ∈ L.checked x r →
    L.rounds x u ⊆ L.secondStage x ∧ ∀ e, L.checked x (forkOn (L.rounds x u) r e) = L.checked x r
  runs : ∀ x (Ω : Type) (P : Prover Ω) (ω : Ω) r u, u ∈ L.checked x r →
    ∃ (xS : Msg) (P' : Prover Unit), A.shape xS = L.sh x u ∧ A.κ xS ≤ L.κb x u ∧ A.γ xS ≤ L.γb x u ∧
      A.V.k xS = (L.rounds x u).card ∧
      (∀ r' : Fin (L.V.k x) → Coin, (∀ t ∉ L.rounds x u, r' t = r t) →
        (run H P L.V x ω r').out = .acc → (run H P' A.V xS () (L.coinsOf x u xS r')).out = .acc) ∧
      ∀ r' : Fin (L.V.k x) → Coin, (∀ t ∉ L.rounds x u, (∀ t' ∈ L.rounds x u, t < t') → r' t = r t) →
        L.session x u (run H P L.V x ω r').seen = some (xS, (run H P' A.V xS () (L.coinsOf x u xS r')).seen) ∧
          ∀ q ∈ (run H P' A.V xS () (L.coinsOf x u xS r')).vLog, q ∈ (run H P L.V x ω r').vLog
  opens : ∀ x s Pt Lv (Ω : Type) (Pv : Prover Ω) ω r u xS vS w,
    L.parse x = some s → L.partition s = some Pt → L.levels s Pt = some Lv → InScopeLev Pt Lv (L.top s) s →
    (run H Pv L.V x ω r).out = .acc → u ∈ L.checked x r → L.session x u (run H Pv L.V x ω r).seen = some (xS, vS) →
    A.R xS w →
    ∃ b ρ rb, Pt.blocks[u]? = some b ∧ topOf Pt (L.top s) u = some ρ ∧ ((L.top s).getD Pt).blocks[ρ]? = some rb ∧
      ∀ (val : ℕ → Option Bool) (ι : ℕ → Bool),
        (∀ q ∈ L.reveal s u (run H Pv L.V x ω r).seen w, L.commit.Opens H q.1 q.2 = true →
          (q.1 = s.roots → val q.2.pos = some q.2.val) ∧
          (q.1 = L.iroots x (run H Pv L.V x ω r).seen ρ → ι q.2.pos = q.2.val)) →
        s.prog.Correct (s.prog.refIn rb val ι) b

/-! ## 7. The extractor at levels -/

/-- The levels strictly between the first and the last at `x`; `0` off scope and at one or two levels. -/
def mid (L : CompiledLev Coin A) (x : Msg) : ℕ :=
  ((L.parse x).bind fun s => (L.partition s).bind fun Pt => (L.levels s Pt).map fun Lv => Lv.levels.length - 2).getD 0

/-- The rounds of the middle levels' samples at `x`, from the second level down. -/
def forks (L : CompiledLev Coin A) (x : Msg) : List (Finset (Fin (L.V.k x))) := (List.range (L.mid x)).map (L.stages x)

/-- The extractor's coins below one kept run with `m` middle waits left: each middle wait's `K_out` coin vectors and,
for each of its `k_out` kept runs, the coins below it; at the last, the inner wait's `K_in` and one `SlotEC` per kept
inner run. -/
def WaitC (L : CompiledLev Coin A) (x : Msg) : ℕ → Type
  | 0 => (Fin (L.capIn x) → Fin (L.V.k x) → Coin) × (Fin (L.kIn x) → L.SlotEC x)
  | m + 1 => (Fin (L.capOut x) → Fin (L.V.k x) → Coin) × (Fin (L.kOut x) → WaitC L x m)

noncomputable instance instFintypeWaitC (L : CompiledLev Coin A) (x : Msg) : (m : ℕ) → Fintype (L.WaitC x m)
  | 0 => inferInstanceAs (Fintype (_ × _))
  | m + 1 =>
    have := instFintypeWaitC L x m
    inferInstanceAs (Fintype ((Fin (L.capOut x) → Fin (L.V.k x) → Coin) × (Fin (L.kOut x) → L.WaitC x m)))

instance instNonemptyWaitC (L : CompiledLev Coin A) (x : Msg) : (m : ℕ) → Nonempty (L.WaitC x m)
  | 0 => inferInstanceAs (Nonempty (_ × _))
  | m + 1 =>
    have := instNonemptyWaitC L x m
    inferInstanceAs (Nonempty ((Fin (L.capOut x) → Fin (L.V.k x) → Coin) × (Fin (L.kOut x) → L.WaitC x m)))

/-- **Below a kept run** `r`: rerun fresh on the next middle level's rounds until `k_out` accepted runs are in hand
(giving up after `K_out`) and go below each; past the middle levels, `Compiled.E`'s inner wait on the second stage and
each kept inner run's extractions (`runExt`). -/
noncomputable def below (L : CompiledLev Coin A) (x : Msg) (s : AuditStatement) :
    (fs : List (Finset (Fin (L.V.k x)))) → (Fin (L.V.k x) → Coin) → L.WaitC x fs.length →
      RP (Fin (L.V.k x) → Coin) (List (List Msg × Opening Bool))
  | [], r, c =>
    (waitAccC (L.kIn x) ((List.finRange (L.capIn x)).map fun l => forkOn (L.secondStage x) r (c.1 l)) []).bind
      fun inner => flatMapRP (fun li : Fin (L.kIn x) × ((Fin (L.V.k x) → Coin) × Seen) =>
        L.runExt x s li.2.1 li.2.2 (c.2 li.1)) ((List.finRange (L.kIn x)).zip inner)
  | f :: fs, r, c =>
    (waitAccC (L.kOut x) ((List.finRange (L.capOut x)).map fun l => forkOn f r (c.1 l)) []).bind
      fun kept => flatMapRP (fun jo : Fin (L.kOut x) × ((Fin (L.V.k x) → Coin) × Seen) =>
        below L x s fs jo.2.1 (c.2 jo.1)) ((List.finRange (L.kOut x)).zip kept)

/-- **The sampling extractor at levels**, one wait per level: replay the judged run; if it accepted, rerun on fresh
coins until `k_out` accepted runs are in hand (giving up after `K_out`); below each (`below`), one wait per middle level
and the inner wait; and assemble. At one or two levels, `Compiled.E` on its coins rearranged. -/
noncomputable def Elev (L : CompiledLev Coin A) : Extractor Coin Sampling.Wit L.V where
  C x := (Fin (L.capOut x) → Fin (L.V.k x) → Coin) × (Fin (L.kOut x) → L.WaitC x (L.forks x).length)
  ext x r₀ c := .rerun r₀ fun v₀ =>
    match L.parse x with
    | none => .ret none
    | some s =>
      if v₀.out = .acc then
        (waitAccC (L.kOut x) ((List.finRange (L.capOut x)).map c.1) []).bind fun outer =>
          (flatMapRP (fun jo : Fin (L.kOut x) × ((Fin (L.V.k x) → Coin) × Seen) =>
              L.below x s (L.forks x) jo.2.1 (c.2 jo.1))
            ((List.finRange (L.kOut x)).zip outer)).bind fun found => .ret (some (assemble s found))
      else .ret none

/-- The error below a kept run with `m` middle waits left, `a` the error past the inner wait's union: each wait's
truncation plus its factor `c` times the error below it. -/
noncomputable def waitErr (L : CompiledLev Coin A) (x : Msg) : ℕ → ℝ → ℝ
  | 0, a => (2 * L.kIn x / L.capIn x + Real.exp (-(L.kIn x : ℝ) / 8)) + slack * a
  | m + 1, a => (2 * L.kOut x / L.capOut x + Real.exp (-(L.kOut x : ℝ) / 8)) + slack * waitErr L x m a

/-- **The knowledge error at levels at `δ`**: the outer wait's truncation plus its factor `c` times the error below a
kept run (`waitErr`) at `δ + κ_s`; `Compiled.κ` at one or two levels. -/
noncomputable def κLev (L : CompiledLev Coin A) (δ : ℝ) (x : Msg) : ℝ :=
  (2 * L.kOut x / L.capOut x + Real.exp (-(L.kOut x : ℝ) / 8)) + slack * L.waitErr x (L.mid x) (δ + L.κs x)

/-- **`γ` at levels**: `c^{2(m+2)}·k(x)·γ_s` at `m` middle levels; `Compiled.γ` at one or two levels. -/
noncomputable def γLev (L : CompiledLev Coin A) (x : Msg) : ℝ := slack ^ (2 * (L.mid x + 2)) * (L.V.k x * L.γs x)

/-! ## 8. The claims -/

/-- **KS at levels**, on the statements in scope, at every `δ ≥ 0`: `Pr[acc ∧ no interiors put the incorrect set in
M_δ] ≤ δ + ε_ks + ε_bind`, at the extractor with one wait per level (`Elev`), `κLev δ`, `γLev` and
`ε_bind = Pr[B_ext]` (`Compiled.KS` at `RLev`). -/
def KS (H : Msg → Msg) (L : CompiledLev Coin A) : Prop :=
  ∀ δ : ℝ, 0 ≤ δ →
    KSLinOn H L.V L.Scope (RLev H L.commit L.parse L.partition L.levels L.top δ) L.Elev (L.κLev δ) L.γLev
      (fun _ => 1) (Bext H L.V L.Elev)

/-- The two-stage verifier the committed interiors and their binding are read on: `L`'s compiled verifier. Neither
reads `two` or `replaySample`. -/
def twoView (L : CompiledLev Coin A) : CompiledTwo Coin A :=
  { toCompiled := L.toCompiled, two := fun _ => none, replaySample := fun _ _ => [] }

/-- **The committed incorrect set at levels** at the prover's tape `ω` and the run's coins `r`: with no top
partition, `Compiled.Freg`; with the top partition `Q`, over the grid of `Q`'s blocks and the strata, the proof
blocks incorrect at the committed boundary trace (`Compiled.regVal`) and their top block's committed interior
(`CompiledTwo.iregVal`), as `CompiledTwo.FregTwo` reads them. Empty on a statement that doesn't parse or whose
partition doesn't evaluate. -/
noncomputable def FregLev (H : Msg → Msg) (L : CompiledLev Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin (L.V.k x) → Coin) : Finset ℕ :=
  match L.parse x with
  | none => ∅
  | some s =>
    match L.partition s, L.top s with
    | some _, none => L.Freg H P x ω
    | some Pt, some Q =>
      (Finset.range Q.blocks.length ×ˢ Finset.range Pt.strata.length).biUnion fun p =>
        incorrectInAt s Pt Q (L.regVal H P x ω) p.1 p.2 (L.twoView.iregVal H P x ω r p.1)
    | none, _ => ∅

/-- **The committed form's error at levels**: `Compiled.εreg` with no top partition; otherwise
`CompiledTwo.εregTwo`'s, with the committed interiors' binding across runs. -/
noncomputable def εregLev (H : Msg → Msg) (L : CompiledLev Coin A) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) : ℝ :=
  match L.parse x >>= L.top with
  | none => L.εreg H x P
  | some _ => L.twoView.εregTwo H x P

/-- **KS, the committed form at levels**, on the statements in scope, at every prover: the run accepts with a sampled
proof block in the committed incorrect set (`FregLev`) with probability at most `ε_reg` (`εregLev`); so, by the levels'
samples, at every `δ ≥ 0` it accepts with no interiors putting the proof blocks incorrect on the committed boundary
trace in `M_δ` with probability at most `δ + ε_reg`. -/
def KSRegLev (H : Msg → Msg) (L : CompiledLev Coin A) : Prop :=
  ∀ x, L.Scope x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregLev H P x p.1 p.2) ≤ L.εregLev H x P ∧
      ∀ s Pt Lv, L.parse x = some s → L.partition s = some Pt → L.levels s Pt = some Lv → ∀ δ : ℝ, 0 ≤ δ →
        Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
          ∀ ι : ℕ → ℕ → Bool, incorrectLev s Pt (L.top s) (L.regVal H P x p.1) ι ∉ levM Pt Lv δ) ≤
          δ + L.εregLev H x P

end CompiledLev

/-! ## 9. The compiler's statements -/

/-- **The sampled compiler at levels is knowledge sound**: over every proof system for proof blocks that is knowledge
sound, every compiled verifier at levels whose commitment binds and that meets the contract meets `CompiledLev.KS`. -/
def CompilerKSLev : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : CompiledLev Coin A)
    (H : Msg → Msg), A.KS H → L.commit.Binds H → L.Sessions H → L.KS H

/-- **The committed form at levels**: at `CompilerKSLev`'s premises, every compiled verifier at levels meets
`CompiledLev.KSRegLev`. -/
def CompilerKSRegLev : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : CompiledLev Coin A)
    (H : Msg → Msg), A.KS H → L.commit.Binds H → L.Sessions H → L.KSRegLev H

/-- **The sampled compiler at levels**: `CompilerKSLev` and `CompilerKSRegLev`, the one statement over levels. -/
def CompilerLev : Prop := CompilerKSLev ∧ CompilerKSRegLev

end Claims.SamplingGeneric
