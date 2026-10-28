import Definitions.Pouw.Assumptions.PearlC.TTOutRev1

/-!
# Per-row seeds (P2) as the per-unit game on the split layout (trusted definitions; nothing assumed)

P2 seeds A's noise per row: row `i` of unit `u` draws E_A from the oracle input the per-unit scheme uses for a one-row
unit ("Row additivity for P2: it reduces to the per-unit TT_OUT").
So the row-seeded game on a workload `U` is the per-unit game on its **split workload**, where every row is its own
unit, and rev1's TT_OUT on the split workload bounds it (`ttOutRowSeed_of_ttOut`, and per audit tile
`ttOutTileRowSeed_of_ttOutTile`, in `RowSeedGamma.lean`). Row additivity is then a theorem, and no new assumption is
added. The error runs over the `R = Σ_u m_u` declared rows: `(q + R)/2^128`.

The reduction reads four conditions, all definitions here, each for the statement reviewer:
- **(C0) `PearlCDevice.RowLocal d`:** the record's ticket, U and flags of word `(i, j)` read row `i` of A′ (and A's
  noise line `i`) only. It holds of every `devAt p G pr` by `rfl` (`devAt_rowLocal`); for an abstract record, credits
  aren't additive over rows without it.
- **(C1) `RowSeedNoise sem sem'`:** the per-row noise `sem'` is `sem` on the split workload, row by row.
- **(C2) the per-row cap:** `pearlCProtocolDevRev1RowCap` and `pearlCTilesDevRev1RowCap` are rev1's protocol and tiles
  with the cap checked for every row (every credited row, per tile). An aggregate cap doesn't transfer: a row over its
  own cap is excluded in the split game but would be credited in the row-seeded one. The per-row cap implies the
  aggregate one. At a census k the served check caps each row at `ρr = 4ρ` and the tile at `ρ`
  (`pearlCProtocolDevRev1RowCapAt`, `pearlCTilesDevRev1RowCapAt`), which (C2) at `ρr` bounds (`RowCapAtGamma.lean`).
- **(C3) `CostModel.SplitClosed CM`:** regrouping rows into units doesn't change what a program costs or queries.
  `CostModel` is abstract, so this is a hypothesis on `CM`, a property of W1 (which charges instructions, not which rows
  form a unit), and no cryptographic assumption.

The reduction also takes `ρ ≤ 1` and the record's add and BF16 prices `≥ 0`, so that a row correct in the split game
earns a credit `≥ 0`: arithmetic, discharged at the concrete records by `norm_num`.

`TTOutRowSeed` and `TTOutTileRowSeed` are the row-seeded statements, the target forms the reduction proves (not named
assumptions): `TTOut` and `TTOutTile` with the error read at the row count.

Three additions, also definitions only:
- **The code-form seed (`-h3`).** (C1-h3) `CodeSeedNoise sem' semC
  π`: the code's seed is C1's up to a fixed permutation of the oracle's inputs. (C5) `CostModel.RelabelClosed CM`:
  relabelling a program's oracle keeps its cost and calls. With them the row-seeded TT_OUT carries to the code's form
  at the same error (`ttOutRowSeed_code`, in `RowSeedCode.lean`).
- **The fragment argument's counting side** ("The fragment argument, made rigorous"):
  `SkipSetU`, `undebitedSkippable` and `fragCount`, which `fragSkip_count_le` (`RowSeedFragment.lean`) bounds a skip
  program's undebited slots by.
- **The fragment argument's skip class** (option (a) of the fragment-draw ruling):
  `SkipProgram` and its class, `SkipEmbed`, (C4) `CostModel.FragmentPriced`, (C6) `PearlCSem.RowDrawn`, `rowWords` and
  `QTree.hits`, which `qtree_union_bound` and `ttOutRowSeed_skipClass` (`RowSeedSkip.lean`) read. The corollary's one
  named assumption, `Assumptions.FragDraw`, is in `RowSeedAssumptions.lean`. P2 rests on `ttOutRowSeed_of_ttOut` (and
  its tile twin) alone, with no named assumption; the corollary is a supporting line, and it takes no TT_OUT.
-/

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

variable {Q R S : Type}

/-! ## The split layout -/

/-- The rows before unit `u`: `Σ_{v < u} m_v`. -/
def rowOffset (L : UnitLayout) (u : ℕ) : ℕ := ∑ v ∈ range u, (L.shape v).m

/-- The declared rows, `R = Σ_u m_u`. -/
def rowsOf (L : UnitLayout) : ℕ := rowOffset L L.N

/-- Row `i` of unit `u` as a unit of the split layout: `offset(u) + i`. -/
def rowIdx (L : UnitLayout) (u i : ℕ) : ℕ := rowOffset L u + i

/-- The unit that split unit `v` is a row of: the last `u < N` with `offset(u) ≤ v`. -/
def unitOfRow (L : UnitLayout) (v : ℕ) : ℕ := Nat.findGreatest (fun u => rowOffset L u ≤ v) (L.N - 1)

/-- The row of its unit that split unit `v` is. -/
def rowOfRow (L : UnitLayout) (v : ℕ) : ℕ := v - rowOffset L (unitOfRow L v)

/-- **The split layout:** unit `u` becomes `m_u` units of shape `(1, n_u, k_u)` with `u`'s weight id, its row `i` at
`rowIdx L u i`. -/
def splitLayout (L : UnitLayout) : UnitLayout where
  N := rowsOf L
  shape v := ⟨1, (L.shape (unitOfRow L v)).n, (L.shape (unitOfRow L v)).k⟩
  wid v := L.wid (unitOfRow L v)

/-- The split workload: the split layout, with the same weights. -/
def splitWorkload (U : Workload) : Workload := ⟨splitLayout U.layout, U.weight⟩

/-- Row `i` of a matrix of codes, as a one-row matrix. -/
def rowCodes (M : Codes) (i : ℕ) : Codes := fun _ l => M i l

/-- Row `i` of a unit's commitment: that row of its activations, ticket words and useful words. -/
def rowCommit (c : Commit) (i : ℕ) : Commit := ⟨rowCodes c.act i, rowCodes c.ticket i, rowCodes c.useful i⟩

/-- A transcript relabelled row by row: split unit `v` commits its row of its unit's commitment, and is blank where its
unit is. -/
def splitTranscript (L : UnitLayout) (τ : CommitLog) : CommitLog := fun v =>
  if v < rowsOf L then (τ (unitOfRow L v)).map fun c => rowCommit c (rowOfRow L v) else none

/-! ## The conditions -/

/-- **(C0) A row-local device record:** the ticket and flags of word `(i, j)`, and its U, are those of the one-row
matrix of A′'s row `i` (with A's noise line `i`), at every shape. -/
def PearlCDevice.RowLocal (d : PearlCDevice) : Prop :=
  (∀ {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code) (B : Matrix (Fin n) (Fin k) Code) (i : Fin m) (j : Fin n),
    d.ticket A B i j = d.ticket (fun _ : Fin 1 => A i) B 0 j ∧
      ∀ t, d.flags A B i j t = d.flags (fun _ : Fin 1 => A i) B 0 j t) ∧
  ∀ {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code) (B : Matrix (Fin n) (Fin k) Code)
    (FA FB : Matrix (Fin k) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) ℕ) (beB : Matrix (Fin n) (Fin 32) ℕ)
    (i : Fin m) (j : Fin n),
    d.useful A B FA FB beA beB i j = d.useful (fun _ : Fin 1 => A i) B FA FB (fun _ => beA i) beB 0 j

/-- The noise `sem` draws for row `i` of unit `u` as its own unit: `sem`'s noise at the one-row unit `rowIdx u i` of the
split workload, on that row's activations. -/
noncomputable def PearlCSem.splitNoise (sem : PearlCSem Q R S) (H : Q → R) (s : S) (U : Workload) (u i : ℕ)
    (act : Codes) :=
  sem.noise H s (splitWorkload U) (rowIdx U.layout u i) (rowCodes act i)

/-- **(C1) Per-row seeds are the split layout's unit seeds.** `sem'` has `sem`'s forming chain, scales and floor, and
its noise at unit `u` is, row by row, `sem`'s at the one-row units of the split workload: row `i`'s E_A is row 0 of
`splitNoise`'s, and the per-job F_A, F_B and E_B are `splitNoise`'s. So the row seed binds exactly what the one-row
unit's seed binds (the same oracle input, with the row's leaf as the unit's A root), and no oracle programming is
needed. -/
def RowSeedNoise (sem sem' : PearlCSem Q R S) : Prop :=
  sem'.fc = sem.fc ∧ sem'.alpha = sem.alpha ∧ sem'.beta = sem.beta ∧ sem'.floor = sem.floor ∧
    ∀ (H : Q → R) (s : S) (U : Workload) (u : ℕ) (act : Codes), u < U.layout.N →
      ∀ (i : ℕ) (hi : i < (U.layout.shape u).m),
        (∀ t, (sem'.noise H s U u act).EA ⟨i, hi⟩ t = (sem.splitNoise H s U u i act).EA ⟨0, Nat.one_pos⟩ t) ∧
        (∀ (l : ℕ) (hl : l < (U.layout.shape u).k)
            (hl' : l < ((splitWorkload U).layout.shape (rowIdx U.layout u i)).k) (t : Fin Params.pi.r),
          (sem'.noise H s U u act).FA ⟨l, hl⟩ t = (sem.splitNoise H s U u i act).FA ⟨l, hl'⟩ t ∧
            (sem'.noise H s U u act).FB ⟨l, hl⟩ t = (sem.splitNoise H s U u i act).FB ⟨l, hl'⟩ t) ∧
        ∀ (j : ℕ) (hj : j < (U.layout.shape u).n)
            (hj' : j < ((splitWorkload U).layout.shape (rowIdx U.layout u i)).n) (t : Fin Params.pi.r),
          (sem'.noise H s U u act).EB ⟨j, hj⟩ t = (sem.splitNoise H s U u i act).EB ⟨j, hj'⟩ t

/-- **(C2) rev1 with the cap checked per row:** `pearlCProtocolDevRev1` whose unit is within the cap when every row is,
row `i`'s debit (`tileDebitRev1` on the row's words) at most `ρ` times its share of the credit. The credit, `W_ref`,
words and rules are rev1's. -/
noncomputable def pearlCProtocolDevRev1RowCap (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Protocol Q R S :=
  { pearlCProtocolDevRev1 d sem ρ with
    capOK := fun H s U u act => ∀ i < (U.layout.shape u).m,
      sem.tileDebitRev1 d H s U u {i} (range (U.layout.shape u).n) act ≤
        ρ * creditDevRev1 d (U.layout.shape u) * tileShare (U.layout.shape u) {i} (range (U.layout.shape u).n) }

/-- **(C2) rev1's audit tiles with the cap checked per credited row:** `pearlCTilesDevRev1` whose tile is within the
cap when every credited row of it is, at most `ρ` times that row's share of the credit. -/
noncomputable def pearlCTilesDevRev1RowCap (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : TileRules Q R S :=
  { pearlCTilesDevRev1 d sem ρ with
    capOK := fun H s U g act =>
      let tl := auditTiling 64 64 U.layout
      let sh := U.layout.shape (tl.unit g)
      ∀ i ∈ sem.passRows sh act (tl.rows g),
        sem.tileDebitRev1 d H s U (tl.unit g) {i} (tl.cols g) act ≤
          ρ * creditDevRev1 d sh * tileShare sh {i} (tl.cols g) }

/-- **(C2) at a wider row cap:** `pearlCProtocolDevRev1` at the cap `ρ`, whose unit is within the cap when its debit is
at most `ρ` times its credit and every row's is at most `ρr` times its share (`pearlCProtocolDevRev1RowCap` at `ρr`).
The credit, `W_ref`, words and rules are rev1's at `ρ`. -/
noncomputable def pearlCProtocolDevRev1RowCapAt (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρr : ℚ) :
    Protocol Q R S :=
  { pearlCProtocolDevRev1 d sem ρ with
    capOK := fun H s U u act =>
      (pearlCProtocolDevRev1 d sem ρ).capOK H s U u act ∧ (pearlCProtocolDevRev1RowCap d sem ρr).capOK H s U u act }

/-- **(C2)'s tiles at a wider row cap**: `pearlCTilesDevRev1` at the cap `ρ`, whose tile is
within the cap when its debit is at most `ρ` times its credited rows' credit and every credited row's is at most `ρr`
times that row's share (`pearlCTilesDevRev1RowCap` at `ρr`). The credit is rev1's at `ρ`, `(1 − ρ)·C` times the
credited rows' share. -/
noncomputable def pearlCTilesDevRev1RowCapAt (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρr : ℚ) :
    TileRules Q R S :=
  { pearlCTilesDevRev1 d sem ρ with
    capOK := fun H s U g act =>
      (pearlCTilesDevRev1 d sem ρ).capOK H s U g act ∧ (pearlCTilesDevRev1RowCap d sem ρr).capOK H s U g act }

/-- **(C3) The cost model is closed under splitting rows into units:** every program on a workload has a program on its
split workload whose transcript is the original's relabelled row by row (`splitTranscript`), at no more cost and no
more calls, for every oracle, state and salt. -/
def CostModel.SplitClosed (CM : CostModel Q R S) : Prop :=
  ∀ (prog : CM.Prog) (U : Workload) (act : ℕ → Codes), ∃ (prog' : CM.Prog) (act' : ℕ → Codes),
    ∀ (H : Q → R) (σ : State) (s : S),
      CM.run prog' H (splitWorkload U) act' σ s = splitTranscript U.layout (CM.run prog H U act σ s) ∧
        CM.cost prog' H (splitWorkload U) act' σ s ≤ CM.cost prog H U act σ s ∧
        CM.calls prog' H (splitWorkload U) act' σ s ≤ CM.calls prog H U act σ s

/-! ## The row-seeded statements (target forms) -/

/-- **The row-seeded TT_OUT** (a target form, proved from rev1's TT_OUT, not assumed): `TTOut` with the error read at
the row count, `εPearlC q R = (q + R)/2^128` for the `R = rowsOf U.layout ≤ 2^64` declared rows. -/
def TTOutRowSeed [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : UnitLayout → Prop) (γ₀ : ℝ) : Prop :=
  CM.Queries ∧
  ∀ U : Workload, U.InDomain P D → rowsOf U.layout ≤ 2 ^ 64 → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 →
    ∀ A : Adversary CM, A.Bounded U T q →
      pr (fun x : (Q → R) × S =>
        T / (1 - γ₀) < ∑ u ∈ correctSet P U x.1 x.2 (A.transcript U x.1 x.2),
          P.credit x.1 x.2 U u (actOf (A.transcript U x.1 x.2) u))
        ≤ εPearlC q (rowsOf U.layout)

/-- **The row-seeded TT_OUT per audit tile** (a target form): `TTOutTile` with the error read at the row count. -/
def TTOutTileRowSeed [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : UnitLayout → Prop) (γ₀ : ℝ) : Prop :=
  CM.Queries ∧
  ∀ U : Workload, U.InDomain P D → rowsOf U.layout ≤ 2 ^ 64 → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 →
    ∀ A : Adversary CM, A.Bounded U T q →
      pr (fun x : (Q → R) × S =>
        T / (1 - γ₀) < ∑ g ∈ goodTiles P TR U x.1 x.2 (A.transcript U x.1 x.2),
          TR.credit x.1 x.2 U g (actOf (A.transcript U x.1 x.2) ((TR.tiling U.layout).unit g)))
        ≤ εPearlC q (rowsOf U.layout)

/-! ## The code-form seed (`-h3`) -/

/-- **(C1-h3) The code-form seed:** `semC` draws its noise from the oracle at the code's inputs, and a fixed permutation
`π L` of the oracle's inputs (one per layout) carries `sem'`'s inputs to them: `semC` at `H` is `sem'` at `H ∘ π L`.
For P2's rows, `π L` takes the one-row unit's input at `rowIdx L u i` to the code's, indexed `u·2³² + i` with `leaf_i`
framed inside unit `u`; the chain, scales and floor are `sem'`'s. -/
def CodeSeedNoise (sem' semC : PearlCSem Q R S) (π : UnitLayout → Q ≃ Q) : Prop :=
  semC.fc = sem'.fc ∧ semC.alpha = sem'.alpha ∧ semC.beta = sem'.beta ∧ semC.floor = sem'.floor ∧
    ∀ (H : Q → R) (s : S) (U : Workload) (u : ℕ) (act : Codes),
      semC.noise H s U u act = sem'.noise (H ∘ π U.layout) s U u act

/-- **(C5) The cost model is closed under relabelling the oracle:** for a fixed permutation `π` of the oracle's inputs,
every program has one that, against `H`, commits what the original commits against `H ∘ π`, at no more cost and no
more calls. -/
def CostModel.RelabelClosed (CM : CostModel Q R S) : Prop :=
  ∀ (π : Q ≃ Q) (prog : CM.Prog) (U : Workload) (act : ℕ → Codes), ∃ prog' : CM.Prog,
    ∀ (H : Q → R) (σ : State) (s : S),
      CM.run prog' H U act σ s = CM.run prog (H ∘ π) U act σ s ∧
        CM.cost prog' H U act σ s ≤ CM.cost prog (H ∘ π) U act σ s ∧
        CM.calls prog' H U act σ s ≤ CM.calls prog (H ∘ π) U act σ s

/-! ## The fragment argument: skip sets and the fragment count -/

/-- The mask of the atoms in `S`, over a word's `atoms`. -/
def omitMask (atoms : List AtomIn) (S : Finset ℕ) : List Bool :=
  List.ofFn fun t : Fin atoms.length => decide ((t : ℕ) ∈ S)

/-- **The word with the atoms in `S` omitted**, under the step `p` and promotion every `G` atoms: each group's atoms
folded by `p.step` from `+0` with the omitted ones passed through, and the groups' words promoted in order from `0`, the
groups those of `Accum.chainP p G`. -/
def chainOmitP (p : Pipeline) (G : ℕ) (atoms : List AtomIn) (S : Finset ℕ) : Option ℚ :=
  ((atoms.toChunks G).zip ((omitMask atoms S).toChunks G)).foldl
    (fun t q => promote t (Skip.foldMask (SkipP.stepP p) 0 q.1 q.2)) (some 0)

/-- **`S` is a skip set of the word:** omitting the atoms in `S` leaves its final value unchanged. -/
def SkipSetU (p : Pipeline) (G : ℕ) (atoms : List AtomIn) (S : Finset ℕ) : Prop :=
  chainOmitP p G atoms S = Accum.chainP p G atoms

/-- **Atom `t` of the word over rows `a` and `b` is undebited skippable:** it lies in some skip set of the word, and the
record's flags (`SkipP.flagsAt p G`, debit (A)) don't flag it. -/
def undebitedSkippable (p : Pipeline) (G : ℕ) {k : ℕ} (a b : Fin k → Code) (t : ℕ) : Prop :=
  (∃ S : Finset ℕ, t ∈ S ∧ SkipSetU p G (rowAtoms a b) S) ∧ SkipP.flagsAt p G k a b t = false

open Classical in
/-- **Row `i`'s fragment count** `Y_i`: the `(J, t)`, `J` one of the `n/8` aligned 8-column groups and `t` one of the
`k/32` atoms, with `t` undebited skippable in every word `(i, j)`, `j ∈ J`. -/
noncomputable def fragCount (p : Pipeline) (G : ℕ) {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code)
    (B : Matrix (Fin n) (Fin k) Code) (i : Fin m) : ℕ :=
  ((range (n / 8) ×ˢ range (k / 32)).filter fun Jt =>
    ∀ j : Fin n, (j : ℕ) / 8 = Jt.1 → undebitedSkippable p G (A i) (B j) Jt.2).card

/-- The atoms omitted from word `(i, j)`: those of the slots `(i / 16, j / 8, t)` whose patch spares the word. -/
def omittedAt (slots : Finset (ℕ × ℕ × ℕ)) (patch : ℕ × ℕ × ℕ → Finset (ℕ × ℕ)) (i j : ℕ) : Finset ℕ :=
  (slots.filter fun s => s.1 = i / 16 ∧ s.2.1 = j / 8 ∧ (i, j) ∉ patch s).image fun s => s.2.2

end Pouw.PearlC

/-! ## The skip class (option (a) of the fragment-draw ruling) -/

namespace Pouw.QTree

/-- **The program hits the bad sets** `Bad` against `H`: some query `x` on its path gets an answer in `Bad x`. -/
def hits {Q R α : Type} (Bad : Q → R → Prop) (H : Q → R) : QTree Q R α → Prop
  | done _ => False
  | query x k => Bad x (H x) ∨ hits Bad H (k (H x))

end Pouw.QTree

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

variable {Q R S : Type}

/-- **A skip program on a unit** (the ruling's item 2; `a-commit-latency.md`, "The fragment argument, made rigorous"):
the activations it commits, and the `mma.sync m16n8k32` instructions it omits, each a slot `(b, J, t)` (16-row block
`b`, 8-column group `J`, atom `t`) with the words `patch s` it routes to generic cores. A patched word keeps the slot's
atom (`SkipProgram.out`), so the patched values are the chain's, not free: free values on 5 words a slot would let 26
slots over one fragment leave all 128 of its words unconstrained. The unit's shape is the workload's. -/
structure SkipProgram where
  act : Codes
  slots : Finset (ℕ × ℕ × ℕ)
  patch : ℕ × ℕ × ℕ → Finset (ℕ × ℕ)

/-- The skip program's slots lie inside an `m × n × k` unit, and each patches at most 5 words. -/
def SkipProgram.WellFormed (m n k : ℕ) (sp : SkipProgram) : Prop :=
  ∀ s ∈ sp.slots, 16 * s.1 + 16 ≤ m ∧ s.2.1 < n / 8 ∧ s.2.2 < k / 32 ∧ (sp.patch s).card ≤ 5

/-- **The skip program's words** on the noisy operands `A`, `B`: word `(i, j)` is the chain over `A i` and `B j` with
the atoms of the slots over it that don't patch it (`omittedAt`) passed through (`chainOmitP`). -/
def SkipProgram.out (p : Pipeline) (G : ℕ) (sp : SkipProgram) {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code)
    (B : Matrix (Fin n) (Fin k) Code) (i : Fin m) (j : Fin n) : Option ℚ :=
  chainOmitP p G (rowAtoms (A i) (B j)) (omittedAt sp.slots sp.patch i j)

/-- Every word of the skip program is the honest chain's. -/
def SkipProgram.Correct (p : Pipeline) (G : ℕ) (sp : SkipProgram) {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code)
    (B : Matrix (Fin n) (Fin k) Code) : Prop :=
  ∀ (i : Fin m) (j : Fin n), sp.out p G A B i j = Accum.chainP p G (rowAtoms (A i) (B j))

/-- Each slot's atom is unflagged (`SkipP.flagsAt p G`, debit (A)) in all of the slot's unpatched words. -/
def SkipProgram.Undebited (p : Pipeline) (G : ℕ) (sp : SkipProgram) {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code)
    (B : Matrix (Fin n) (Fin k) Code) : Prop :=
  ∀ s ∈ sp.slots, ∀ (i : Fin m) (j : Fin n), (i : ℕ) / 16 = s.1 → (j : ℕ) / 8 = s.2.1 →
    ((i : ℕ), (j : ℕ)) ∉ sp.patch s → SkipP.flagsAt p G k (A i) (B j) s.2.2 = false

/-- **The skip class** on unit `u` against the oracle `H`: the skip program is well formed, its committed activations
pass the domain rules and are 32-bit words (an FP32 commitment holds nothing else), and on the unit's noisy operands
from them (`unitChain`) every word is correct and every slot is undebited. A slot whose atom is flagged in only some of
its unpatched words saves 4,096 units but is debited `32 + 32/G` a flagged word; the class leaves such slots out. -/
def SkipProgram.InClass (sem : PearlCSem Q R S) (p : Pipeline) (G : ℕ) (s : S) (U : Workload) (u : ℕ)
    (sp : SkipProgram) (H : Q → R) : Prop :=
  sp.WellFormed (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k ∧
    sem.OperandOK (U.layout.shape u).m (U.layout.shape u).k sp.act ∧
    (∀ i < (U.layout.shape u).m, ∀ l < (U.layout.shape u).k, sp.act i l < 2 ^ 32) ∧
    sp.Correct p G (sem.unitChain H s U u sp.act).1 (sem.unitChain H s U u sp.act).2 ∧
    sp.Undebited p G (sem.unitChain H s U u sp.act).1 (sem.unitChain H s U u sp.act).2

/-- **Skip programs in the cost model** on unit `u` of `U` (the ruling's item 2): a program `embed sp` for each skip
program, which commits its activations and its words at unit `u`, and the honest program `honest`, which commits the
unit's own activations and the honest chain's words. `CostModel` itself is unchanged. -/
structure SkipEmbed (CM : CostModel Q R S) (sem : PearlCSem Q R S) (p : Pipeline) (G : ℕ) (U : Workload) (u : ℕ) where
  embed : SkipProgram → CM.Prog
  honest : CM.Prog
  run_embed : ∀ (sp : SkipProgram) (H : Q → R) (act : ℕ → Codes) (σ : State) (s : S), ∃ c : Commit,
    CM.run (embed sp) H U act σ s u = some c ∧ c.act = sp.act ∧
      ∀ (i : Fin (U.layout.shape u).m) (j : Fin (U.layout.shape u).n),
        wordQ (c.ticket i j) = sp.out p G (sem.unitChain H s U u sp.act).1 (sem.unitChain H s U u sp.act).2 i j
  run_honest : ∀ (H : Q → R) (act : ℕ → Codes) (σ : State) (s : S), ∃ c : Commit,
    CM.run honest H U act σ s u = some c ∧ c.act = act u ∧
      ∀ (i : Fin (U.layout.shape u).m) (j : Fin (U.layout.shape u).n),
        wordQ (c.ticket i j) =
          Accum.chainP p G (rowAtoms ((sem.unitChain H s U u (act u)).1 i) ((sem.unitChain H s U u (act u)).2 j))

/-- **(C4) Whole-fragment pricing**, a cost-model condition like (C3), for the statement reviewer: an embedded skip
program costs at least the honest program's cost less 4,096 units per omitted slot, for every oracle, activation input,
state and salt. Each omitted `m16n8k32` saves at most its whole fragment, and patches cost at least 0. On a
per-instruction W1 model it is a lemma: the cost is a sum of instruction prices, and a skip program deletes 4,096-unit
terms. -/
def CostModel.FragmentPriced (CM : CostModel Q R S) {sem : PearlCSem Q R S} {p : Pipeline} {G : ℕ} {U : Workload}
    {u : ℕ} (e : SkipEmbed CM sem p G U u) : Prop :=
  ∀ (sp : SkipProgram) (H : Q → R) (act : ℕ → Codes) (σ : State) (s : S),
    CM.cost e.honest H U act σ s ≤ CM.cost (e.embed sp) H U act σ s + 4096 * sp.slots.card

/-- Row `i`'s words in a `k`-column operand. -/
def rowWords (k : ℕ) (act : Codes) (i : ℕ) : Fin k → ℕ := fun l => act i l

/-- **(C6) Each row's noise is one oracle draw**, a condition for the statement reviewer. The ruling's item 4 takes it
from C1, but C1 relates `sem'` to `sem` on the split workload and leaves which oracle inputs `sem.noise` reads
abstract. At salt `s` on unit `u` of `U`, `sem` reads the per-job noise (`F_A`, `E_B`, `F_B`) from the oracle at the
job's input `xJ` only, and row `i`'s `E_A` line from the oracle at `key i w` and `xJ` only, `w` the row's `k` words.
On the unit's rows and their 32-bit words, the keys a row leaf commits, `key` is injective (commitments are ideal here:
it abstracts the row leaf's collision resistance) and never `xJ`. Over every `ℕ` index and word no injective `key` into
a finite `Q` exists; `rowDrawn_satisfiable` shows this form can hold. -/
def PearlCSem.RowDrawn (sem : PearlCSem Q R S) (s : S) (U : Workload) (u : ℕ) (xJ : Q)
    (key : ℕ → (Fin (U.layout.shape u).k → ℕ) → Q) : Prop :=
  (∀ (i i' : Fin (U.layout.shape u).m) (a a' : Fin (U.layout.shape u).k → ℕ), (∀ l, a l < 2 ^ 32) →
      (∀ l, a' l < 2 ^ 32) → key i a = key i' a' → i = i' ∧ a = a') ∧
    (∀ (i : Fin (U.layout.shape u).m) (a : Fin (U.layout.shape u).k → ℕ), (∀ l, a l < 2 ^ 32) → key i a ≠ xJ) ∧
    (∀ (H H' : Q → R) (act act' : Codes), H xJ = H' xJ →
      (sem.noise H s U u act).FA = (sem.noise H' s U u act').FA ∧
        (sem.noise H s U u act).EB = (sem.noise H' s U u act').EB ∧
        (sem.noise H s U u act).FB = (sem.noise H' s U u act').FB) ∧
    ∀ (H H' : Q → R) (act act' : Codes) (i : Fin (U.layout.shape u).m),
      rowWords (U.layout.shape u).k act i = rowWords (U.layout.shape u).k act' i → H xJ = H' xJ →
      H (key i (rowWords (U.layout.shape u).k act i)) = H' (key i (rowWords (U.layout.shape u).k act i)) →
      (sem.noise H s U u act).EA i = (sem.noise H' s U u act').EA i

end Pouw.PearlC
