module

public import Definitions.Claims.SamplingTwoReg

/-!
# Claims: sampling at levels, one statement

SPOT's audit at any number of levels (the page's Definition 3.13), on `Claims.Sampling`'s statement, witness and
commitment and `Claims.SamplingGeneric`'s compiled verifier, read as they stand. The one-level audit
(`Claims.Sampling.R`, `Compiled.KS`, `Compiled.KSReg`) and the two-level audit (`RTwo`, `CompiledTwo.KS`,
`CompiledTwo.KSRegTwo`) are its cases at no inner level and at one.

1. **The design at levels** (`Design.Lev`), as the verifier evaluates it on a statement, as `Design.Two` is: the outer
   design, a one-stage design over the blocks of the coarsest partition, and the inner levels from coarse to fine
   (`Design.Inner`): each a partition whose blocks the level above samples, and the count `k_t` sampled in each
   stratum `t` of the next finer partition inside each sampled block. The finest partition is the statement's proof
   blocks. With no inner level the outer design samples the proof blocks themselves. The sample it describes is core's
   level-by-level sample (`Verity.Levels.design`).
2. **The nested miss** (`Design.Lev.Pa`), by recursion on the inner levels: a uniform `min(k, n)`-subset of a cell
   escapes with the expectation of its members' escapes (`subsetEsc`), a cell of proof blocks with the hypergeometric
   miss (`escCell`), and the outer design's `P_a` weighs the top blocks' escapes. With no inner level it is the
   outer design's `P_a`; with one it is `Design.Two.Pa`, which is `Verity.Levels.design`'s miss where two levels
   realize it (`Verity.Levels.missProb_two_toReal`).
3. **The relation** (`RLev`): at some interior values, one per top block, the proof blocks incorrect on their top
   block's values (`Design.Lev.incorrect`) are in `M_δ` (`Design.Lev.M`). With no inner level the top block of a proof
   block is itself, and the interiors are each proof block's own.
4. **The compiled verifier** (`CompiledLev`): a `Compiled` with the levels it evaluates on a statement (`levels`), the
   samples of the levels above the last at a run's coins (`upper`), and each inner level's rounds but the last's
   (`stages`; the last level's are `secondStage`). Its contract (`CompiledLev.Sessions`) is `Compiled.Sessions` with
   its sampling clause read level by level, and its claims are `CompiledLev.KS` and the committed form
   `CompiledLev.KSRegLev`.
5. **The compiler's statements**: `CompilerKSLev`, `CompilerKSRegLev`, and `CompilerLev`, both.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.Sampling

/-! ## 1. The levels -/

/-- **An inner level**: the partition whose blocks the level above samples (`part`), and the count `k_t` this level
samples in each stratum `t` of the next finer partition inside each sampled block, each stratum's past the list `0`. -/
structure Design.Inner where
  part : Partition
  k : List ℕ

/-- **A design at levels**, coarse to fine: the outer design over the blocks of the coarsest partition, and the inner levels. With
no inner level the outer design samples the proof blocks. -/
structure Design.Lev where
  outer : Design
  inner : List Design.Inner

/-- One-level sampling under the design `d`. -/
def Design.Lev.ofDesign (d : Design) : Design.Lev := ⟨d, []⟩

/-- Two-level sampling under the two-stage design `d`. -/
def Design.Lev.ofTwo (d : Design.Two) : Design.Lev := ⟨d.replay, [⟨d.part, d.k⟩]⟩

/-- **The top partition**, whose blocks the outer design samples: the first inner level's, or the proof blocks'. -/
def Design.Lev.top (L : Design.Lev) (Pt : Partition) : Partition :=
  match L.inner with
  | [] => Pt
  | l :: _ => l.part

/-- **The top block of proof block `v`**: the block of the top partition that holds it; `v` itself with no inner
level. -/
def Design.Lev.topOf (L : Design.Lev) (Pt : Partition) (v : ℕ) : Option ℕ :=
  match L.inner with
  | [] => some v
  | l :: _ => replayOf Pt l.part v

/-! ## 2. The nested miss -/

/-- **A uniform subset's escape**: the expectation, over a uniform `min(k, n)`-subset of the `n` members of `C`, of the
product of its members' escapes `e`. -/
noncomputable def subsetEsc (C : Finset ℕ) (k : ℕ) (e : ℕ → ℝ) : ℝ :=
  (∑ S ∈ C.powersetCard (min k C.card), ∏ X ∈ S, e X) / (C.card.choose (min k C.card) : ℝ)

/-- **Block `ρ`'s escape at level `l`**, the levels below it `rest`, at the set `B` of incorrect proof blocks of
`Pt`: the probability that the levels below sample no proof block of `B` inside `ρ`. Above the proof blocks each
stratum's cell escapes as a uniform subset of its blocks (`subsetEsc`); a cell of proof blocks with the hypergeometric
miss (`escCell`). -/
noncomputable def Design.Inner.esc : Design.Inner → List Design.Inner → Partition → Finset ℕ → ℕ → ℝ
  | l, [], Pt, B, ρ =>
    ∏ t ∈ Finset.range Pt.strata.length,
      escCell (cell Pt l.part ρ t).card (l.k.getD t 0) (B ∩ cell Pt l.part ρ t).card
  | l, l' :: rest, Pt, B, ρ =>
    ∏ t ∈ Finset.range l'.part.strata.length,
      subsetEsc (cell l'.part l.part ρ t) (l.k.getD t 0) (Design.Inner.esc l' rest Pt B)

/-- **The acceptance probability** `P_a(B)` of the levels at a set `B` of proof blocks of `Pt`, the nested miss: the
outer design's `P_a` with no inner level; otherwise, catching each top block `ρ` with probability `1 − e_ρ`
(`Design.Inner.esc`), the probability that the outer design's sample misses the caught ones,
`Σ_T P_a(T)·∏_{ρ∈T}(1 − e_ρ)·∏_{ρ∉T} e_ρ`. -/
noncomputable def Design.Lev.Pa (L : Design.Lev) (Pt : Partition) (B : Finset ℕ) : ℝ :=
  match L.inner with
  | [] => L.outer.Pa Pt B
  | l :: rest =>
    ∑ T ∈ (Finset.range l.part.blocks.length).powerset,
      L.outer.Pa l.part T *
        ((∏ ρ ∈ T, (1 - l.esc rest Pt B ρ)) * ∏ ρ ∈ Finset.range l.part.blocks.length \ T, l.esc rest Pt B ρ)

/-- **The confidence set** `M_δ`: every set `B` of proof blocks with `P_a(B) > δ`. -/
def Design.Lev.M (L : Design.Lev) (Pt : Partition) (δ : ℝ) : Set (Finset ℕ) := {B | δ < L.Pa Pt B}

/-! ## 3. The relation -/

open Classical in
/-- **The incorrect proof blocks at boundary values `val` and interior values `ι`**, one interior per top block: each
proof block incorrect on its top block's values, the boundary's `val` and the interior's `ι ρ` (`Prog.refIn` at the top
block). -/
noncomputable def Design.Lev.incorrect (L : Design.Lev) (s : AuditStatement) (Pt : Partition) (val : ℕ → Option Bool)
    (ι : ℕ → ℕ → Bool) : Finset ℕ :=
  (Finset.range Pt.blocks.length).filter fun v => ¬ ∃ u ρ r, Pt.blocks[v]? = some u ∧ L.topOf Pt v = some ρ ∧
    (L.top Pt).blocks[ρ]? = some r ∧ s.prog.Correct (s.prog.refIn r val (ι ρ)) u

/-- **The relation at levels** at `δ`, on the statement's bytes: it parses, its proof-block partition and levels
evaluate (`partition`, `levels`, the verifier's own), and at some interior values, one per top block, the proof blocks
incorrect on the boundary trace `w` opens are in `M_δ`. The interiors are the relation's to pick: serving commits only
the boundary. -/
def RLev (H : Msg → Msg) (Cm : CommitmentScheme Bool) (parse : Msg → Option AuditStatement)
    (partition : AuditStatement → Option Partition) (levels : AuditStatement → Option Design.Lev) (δ : ℝ) : Rel Wit :=
  fun x w => ∃ s Pt L, parse x = some s ∧ partition s = some Pt ∧ levels s = some L ∧
    ∃ ι : ℕ → ℕ → Bool, L.incorrect s Pt (w.val H Cm s) ι ∈ L.M Pt δ

/-! ## 4. Scope -/

/-- **The inner levels refine down to `Pt`**: each level's partition has disjoint blocks, each block of the next finer
partition is in one of its blocks, and it has one `k ≥ 1` per stratum of the next finer partition. -/
def Design.Inner.Chain : List Design.Inner → Partition → Prop
  | [], _ => True
  | [l], Pt => l.part.gates.Nodup ∧ (∀ v < Pt.blocks.length, (replayOf Pt l.part v).isSome) ∧
      l.k.length = Pt.strata.length ∧ ∀ k ∈ l.k, 1 ≤ k
  | l :: l' :: rest, Pt => l.part.gates.Nodup ∧
      (∀ v < l'.part.blocks.length, (replayOf l'.part l.part v).isSome) ∧
      l.k.length = l'.part.strata.length ∧ (∀ k ∈ l.k, 1 ≤ k) ∧ Design.Inner.Chain (l' :: rest) Pt

/-- **The scope at levels, readable**: a topologically ordered program, partitioned into proof blocks (`Partitions`),
an outer design well formed on the top partition, and inner levels that refine down to the proof blocks. -/
def InScopeLev (Pt : Partition) (L : Design.Lev) (s : AuditStatement) : Prop :=
  s.prog.Topological ∧ Pt.Partitions s.prog ∧ L.outer.WellFormed (L.top Pt) ∧ Design.Inner.Chain L.inner Pt

end Claims.Sampling

namespace Claims.SamplingGeneric

open Claims
open Claims.Sampling (AuditStatement Partition Design InScopeLev RLev cell escCell replayOf sampleBlocks
  footBits forkOn KSLinOn Bext subsetEsc incorrectInAt footprintAt)

/-! ## 5. The compiled verifier -/

/-- **A compiled sampled verifier at levels over `A`**: a compiled sampled verifier (`Compiled`) with the levels it
evaluates on a statement (`levels`), the samples of the levels above the last at a run's coins (`upper`, the `j`-th
the blocks of the `j`-th inner level's partition it samples; the last level's sample is `checked`), and the rounds of
each inner level's sample but the first's (`stages`, the `j`-th the rounds of the `(j+1)`-th's; the last level's are
`secondStage`). -/
structure CompiledLev (Coin : Type) (A : Arg Coin) extends Compiled Coin A where
  levels : AuditStatement → Option Design.Lev
  upper : (x : Msg) → (Fin (V.k x) → Coin) → List (List ℕ)
  stages : (x : Msg) → ℕ → Finset (Fin (V.k x))

namespace CompiledLev

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The scope at levels, on the statement's bytes**: it parses, its proof-block partition and levels evaluate, and
they are in scope (`InScopeLev`). -/
def Scope (L : CompiledLev Coin A) (x : Msg) : Prop :=
  ∃ s Pt Lv, L.parse x = some s ∧ L.partition s = some Pt ∧ L.levels s = some Lv ∧ InScopeLev Pt Lv s

/-! ## 6. The contract -/

/-- **The last level's sample**, at a run's coins `r`: with no inner level, as `Compiled.Sessions.draws`, for each set
`B` of proof blocks a weight of the coins bounds the probability over fresh second-stage coins that the sample misses
`B`, and averages at most the outer design's `P_a(B)`; otherwise fresh second-stage coins sample no proof block of a
set `X` with probability at most `∏_{ρ∈S} ∏_t C(n_{ρ,t} − |X ∩ cell|, k_t)/C(n_{ρ,t}, k_t)`, `S` the last inner
level's blocks `r` samples. -/
def LastSamples (L : CompiledLev Coin A) (x : Msg) (Pt : Partition) (Lv : Design.Lev) : Prop :=
  match Lv.inner.getLast? with
  | none => ∀ B ⊆ Finset.range Pt.blocks.length, ∃ esc : (Fin (L.V.k x) → Coin) → ℝ, (∀ r, 0 ≤ esc r) ∧
      (∀ r, Pr (fun d : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x (forkOn (L.secondStage x) r d), u ∉ B) ≤
        esc r) ∧ Ex esc ≤ Lv.outer.Pa Pt B
  | some l => ∀ (r : Fin (L.V.k x) → Coin) (X : Finset ℕ),
      Pr (fun e : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x (forkOn (L.secondStage x) r e), u ∉ X) ≤
        ∏ ρ ∈ (Finset.range l.part.blocks.length).filter (· ∈ (L.upper x r).getLastD []),
          ∏ t ∈ Finset.range Pt.strata.length,
            escCell (cell Pt l.part ρ t).card (l.k.getD t 0) (X ∩ cell Pt l.part ρ t).card

/-- **The waits' unions**: with no inner level, the outer wait's over sets of the `n_PU` proof blocks within its bits;
otherwise the outer wait's over the counts per cell of the last inner level, `M ≤ n_out` (`footBits`), and the inner
wait's over the proof blocks of the last inner level's sampled blocks, `≤ n_in` at every `r` of which some
second-stage fork is accepted. -/
def Unions (H : Msg → Msg) (L : CompiledLev Coin A) (x : Msg) (Pt : Partition) (Lv : Design.Lev) : Prop :=
  match Lv.inner.getLast? with
  | none => L.nPU x ≤ L.nOut x
  | some l => footBits Pt ⟨l.part, Lv.outer, l.k⟩ ≤ L.nOut x ∧ ∀ (Ω : Type) (P : Prover Ω) ω (r e : Fin (L.V.k x) → Coin),
      (run H P L.V x ω (forkOn (L.secondStage x) r e)).out = .acc →
        (sampleBlocks Pt l.part ((L.upper x r).getLastD [])).card ≤ L.nIn x

/-- **The compiler's contract at levels** at `H`: `Compiled.Sessions` with its sampling clause read level by level and
its openings read at a top block.

* `firstSamples`: on a statement in scope with an inner level, the outer design's sample misses a set `T` of top blocks
  with probability at most the outer design's `P_a(T)`.
* `midSamples`: on a statement in scope, at a run's coins `r`, fresh coins on the rounds of inner level `j + 1`'s
  sample give the product of any escapes `e ∈ [0, 1]` of its sampled blocks at most, in expectation, the product over
  level `j`'s sampled blocks and strata of a uniform subset's escape (`subsetEsc`), and leave the samples above it as
  they are. Only audits of three levels or more read it.
* `lastSamples`: `LastSamples`.
* `unions`: `Unions`.
* `slots`, `shared`, `keys`, `refuses` (at `CompiledLev.Scope`), `interior`, `fixed` and `runs`: `Compiled.Sessions`'.
* `opens`: on a statement in scope, at an accepted run that samples `u` and shows its session, a witness in `A`'s
  relation at the session's statement reveals openings such that any values and interior values agreeing with those
  that pass (under the statement's roots, and under the interior roots the run showed for `u`'s top block `ρ`) make
  proof block `u` correct on `ρ`'s values (`Prog.refIn` at `ρ`). -/
structure Sessions (H : Msg → Msg) (L : CompiledLev Coin A) : Prop where
  firstSamples : ∀ x s Pt Lv, L.parse x = some s → L.partition s = some Pt → L.levels s = some Lv →
    InScopeLev Pt Lv s → ∀ l rest, Lv.inner = l :: rest → ∀ T ⊆ Finset.range l.part.blocks.length,
      Pr (fun r : Fin (L.V.k x) → Coin => ∀ ρ ∈ (L.upper x r).headD [], ρ ∉ T) ≤ Lv.outer.Pa l.part T
  midSamples : ∀ x s Pt Lv, L.parse x = some s → L.partition s = some Pt → L.levels s = some Lv →
    InScopeLev Pt Lv s → ∀ (j : ℕ) (l l' : Design.Inner), Lv.inner[j]? = some l → Lv.inner[j + 1]? = some l' →
      ∀ (r : Fin (L.V.k x) → Coin) (e : ℕ → ℝ), (∀ Y, 0 ≤ e Y ∧ e Y ≤ 1) →
        Ex (fun d : Fin (L.V.k x) → Coin =>
            ∏ Y ∈ (Finset.range l'.part.blocks.length).filter
              (· ∈ (L.upper x (forkOn (L.stages x j) r d)).getD (j + 1) []), e Y) ≤
          ∏ ρ ∈ (Finset.range l.part.blocks.length).filter (· ∈ (L.upper x r).getD j []),
            ∏ t ∈ Finset.range l'.part.strata.length, subsetEsc (cell l'.part l.part ρ t) (l.k.getD t 0) e ∧
        ∀ d i, i ≤ j → (L.upper x (forkOn (L.stages x j) r d))[i]? = (L.upper x r)[i]?
  lastSamples : ∀ x s Pt Lv, L.parse x = some s → L.partition s = some Pt → L.levels s = some Lv →
    InScopeLev Pt Lv s → L.LastSamples x Pt Lv
  slots : ∀ x r, (L.checked x r).length ≤ L.nPU x
  shared : ∀ x (Ω : Type) (P : Prover Ω) ω r i j u u', (run H P L.V x ω r).out = .acc →
    (L.checked x r)[i]? = some u → (L.checked x r)[j]? = some u' →
    L.skey x (L.checked x r) i = L.skey x (L.checked x r) j →
    L.session x u (run H P L.V x ω r).seen = L.session x u' (run H P L.V x ω r).seen
  keys : ∀ x r, ((Finset.range (L.checked x r).length).image (L.skey x (L.checked x r))).card ≤ L.nSess x
  unions : ∀ x s Pt Lv, L.parse x = some s → L.partition s = some Pt → L.levels s = some Lv →
    InScopeLev Pt Lv s → L.Unions H x Pt Lv
  refuses : ∀ x (Ω : Type) (P : Prover Ω) ω r, (run H P L.V x ω r).out = .acc → L.Scope x
  interior : ∀ x (Ω : Type) (P : Prover Ω) ω r e ρ,
    L.iroots x (run H P L.V x ω (forkOn (L.secondStage x) r e)).seen ρ = L.iroots x (run H P L.V x ω r).seen ρ
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
    L.parse x = some s → L.partition s = some Pt → L.levels s = some Lv → InScopeLev Pt Lv s →
    (run H Pv L.V x ω r).out = .acc → u ∈ L.checked x r → L.session x u (run H Pv L.V x ω r).seen = some (xS, vS) →
    A.R xS w →
    ∃ b ρ rb, Pt.blocks[u]? = some b ∧ Lv.topOf Pt u = some ρ ∧ (Lv.top Pt).blocks[ρ]? = some rb ∧
      ∀ (val : ℕ → Option Bool) (ι : ℕ → Bool),
        (∀ q ∈ L.reveal s u (run H Pv L.V x ω r).seen w, L.commit.Opens H q.1 q.2 = true →
          (q.1 = s.roots → val q.2.pos = some q.2.val) ∧
          (q.1 = L.iroots x (run H Pv L.V x ω r).seen ρ → ι q.2.pos = q.2.val)) →
        s.prog.Correct (s.prog.refIn rb val ι) b

/-! ## 7. The claims -/

/-- **KS at levels**, on the statements in scope, at every `δ ≥ 0`: `Pr[acc ∧ no interiors put the incorrect set in
M_δ] ≤ δ + ε_ks + ε_bind`, at `Compiled`'s extractor, `κ_δ`, `γ` and `ε_bind = Pr[B_ext]` (`Compiled.KS` at
`RLev`). -/
def KS (H : Msg → Msg) (L : CompiledLev Coin A) : Prop :=
  ∀ δ : ℝ, 0 ≤ δ →
    KSLinOn H L.V L.Scope (RLev H L.commit L.parse L.partition L.levels δ) L.E (L.κ δ) L.γ (fun _ => 1)
      (Bext H L.V L.E)

/-- The two-stage verifier the committed interiors and their binding are read on: `L`'s compiled verifier, its last
inner level's sample as the replay sample. Neither reads its `two`. -/
def twoView (L : CompiledLev Coin A) : CompiledTwo Coin A :=
  { toCompiled := L.toCompiled, two := fun _ => none, replaySample := fun x r => (L.upper x r).getLastD [] }

/-- **The committed incorrect set at levels** at the prover's tape `ω` and the run's coins `r`: with no inner level,
`Compiled.Freg`; otherwise over the grid of top blocks and strata, the proof blocks incorrect at the committed boundary
trace (`Compiled.regVal`) and their top block's committed interior (`CompiledTwo.iregVal`), as `CompiledTwo.FregTwo`
reads them. Empty on a statement that doesn't parse or whose partition or levels don't evaluate. -/
noncomputable def FregLev (H : Msg → Msg) (L : CompiledLev Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin (L.V.k x) → Coin) : Finset ℕ :=
  match L.parse x with
  | none => ∅
  | some s =>
    match L.partition s, L.levels s with
    | some _, some ⟨_, []⟩ => L.Freg H P x ω
    | some Pt, some ⟨_, l :: _⟩ =>
      (Finset.range l.part.blocks.length ×ˢ Finset.range Pt.strata.length).biUnion fun p =>
        incorrectInAt s Pt l.part (L.regVal H P x ω) p.1 p.2 (L.twoView.iregVal H P x ω r p.1)
    | _, _ => ∅

/-- **The committed form's error at levels**: `Compiled.εreg` with no inner level; otherwise `CompiledTwo.εregTwo`'s,
with the committed interiors' binding across runs. -/
noncomputable def εregLev (H : Msg → Msg) (L : CompiledLev Coin A) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) : ℝ :=
  match L.parse x >>= L.levels with
  | some ⟨_, []⟩ => L.εreg H x P
  | _ => L.twoView.εregTwo H x P

/-- **KS, the committed form at levels**, on the statements in scope, at every prover: the run accepts with a sampled
proof block in the committed incorrect set (`FregLev`) with probability at most `ε_reg` (`εregLev`); so, by the levels'
samples, at every `δ ≥ 0` it accepts with no interiors putting the proof blocks incorrect on the committed boundary
trace in `M_δ` with probability at most `δ + ε_reg`. -/
def KSRegLev (H : Msg → Msg) (L : CompiledLev Coin A) : Prop :=
  ∀ x, L.Scope x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregLev H P x p.1 p.2) ≤ L.εregLev H x P ∧
      ∀ s Pt Lv, L.parse x = some s → L.partition s = some Pt → L.levels s = some Lv → ∀ δ : ℝ, 0 ≤ δ →
        Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
          ∀ ι : ℕ → ℕ → Bool, Lv.incorrect s Pt (L.regVal H P x p.1) ι ∉ Lv.M Pt δ) ≤
          δ + L.εregLev H x P

end CompiledLev

/-! ## 8. The compiler's statements -/

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

/-- **The sampled compiler at levels**: `CompilerKSLev` and `CompilerKSRegLev`. -/
def CompilerLev : Prop := CompilerKSLev ∧ CompilerKSRegLev

end Claims.SamplingGeneric
