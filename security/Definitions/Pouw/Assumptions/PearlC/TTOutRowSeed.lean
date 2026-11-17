module

public import Definitions.Pouw.PearlC.DeviceRev1

/-! Per-row seeds as the per-unit game on the split layout: the conditions the reduction reads and the row-seeded statements, with nothing assumed. -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

variable {Q R S : Type}

def rowOffset (L : Layout) (u : ℕ) : ℕ := ∑ v ∈ range u, (L.shape v).m

def rowsOf (L : Layout) : ℕ := rowOffset L L.N

def rowIdx (L : Layout) (u i : ℕ) : ℕ := rowOffset L u + i

def unitOfRow (L : Layout) (v : ℕ) : ℕ := Nat.findGreatest (fun u => rowOffset L u ≤ v) (L.N - 1)

def rowOfRow (L : Layout) (v : ℕ) : ℕ := v - rowOffset L (unitOfRow L v)

def splitLayout (L : Layout) : Layout where
  N := rowsOf L
  shape v := ⟨1, (L.shape (unitOfRow L v)).n, (L.shape (unitOfRow L v)).k⟩
  fixedId v := L.fixedId (unitOfRow L v)

def splitWorkload (U : Workload) : Workload := ⟨splitLayout U.layout, U.fixed⟩

def rowCodes (M : Codes) (i : ℕ) : Codes := fun _ l => M i l

def rowCommit (c : Commit) (i : ℕ) : Commit := ⟨rowCodes c.act i, rowCodes c.ticket i, rowCodes c.useful i⟩

def splitTrace (L : Layout) (τ : Trace) : Trace := fun v =>
  if v < rowsOf L then (τ (unitOfRow L v)).map fun c => rowCommit c (rowOfRow L v) else none

/-- **(C0)**: for an abstract record, credits aren't additive over rows without it; every `devAt p G pr` meets it
(`devAt_rowLocal`). -/
def PearlCDevice.RowLocal (d : PearlCDevice) : Prop :=
  (∀ {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code) (B : Matrix (Fin n) (Fin k) Code) (i : Fin m) (j : Fin n),
    d.ticket A B i j = d.ticket (fun _ : Fin 1 => A i) B 0 j ∧
      ∀ t, d.flags A B i j t = d.flags (fun _ : Fin 1 => A i) B 0 j t) ∧
  ∀ {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code) (B : Matrix (Fin n) (Fin k) Code)
    (FA FB : Matrix (Fin k) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) ℕ) (beB : Matrix (Fin n) (Fin 32) ℕ)
    (i : Fin m) (j : Fin n),
    d.useful A B FA FB beA beB i j = d.useful (fun _ : Fin 1 => A i) B FA FB (fun _ => beA i) beB 0 j

noncomputable def PearlCSem.splitNoise (sem : PearlCSem Q R S) (H : Q → R) (s : S) (U : Workload) (u i : ℕ)
    (act : Codes) :=
  sem.noise H s (splitWorkload U) (rowIdx U.layout u i) (rowCodes act i)

/-- **(C1)**: the row seed binds exactly what the one-row unit's seed binds, so no oracle programming is needed. -/
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

/-- **(C2)**: the cap is per row because an aggregate cap doesn't transfer: a row over its own cap is excluded in the
split game but would be credited in the row-seeded one. -/
noncomputable def pearlCProtocolDevRev1RowInertBound (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Protocol Q R S :=
  { pearlCProtocolDevRev1 d sem ρ with
    inertBoundOK := fun H s U u act => ∀ i < (U.layout.shape u).m,
      sem.tileInertWorkRev1 d H s U u {i} (range (U.layout.shape u).n) act ≤
        ρ * creditDevRev1 d (U.layout.shape u) * tileShare (U.layout.shape u) {i} (range (U.layout.shape u).n) }

/-- **(C2)** per tile, the cap checked for every credited row. -/
noncomputable def pearlCTilesDevRev1RowInertBound (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : TileRules Q R S :=
  { pearlCTilesDevRev1 d sem ρ with
    inertBoundOK := fun H s U g act =>
      let tl := auditTiling 64 64 U.layout
      let sh := U.layout.shape (tl.unit g)
      ∀ i ∈ sem.passRows sh act (tl.rows g),
        sem.tileInertWorkRev1 d H s U (tl.unit g) {i} (tl.cols g) act ≤
          ρ * creditDevRev1 d sh * tileShare sh {i} (tl.cols g) }

noncomputable def pearlCProtocolDevRev1RowInertBoundAt (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρr : ℚ) :
    Protocol Q R S :=
  { pearlCProtocolDevRev1 d sem ρ with
    inertBoundOK := fun H s U u act =>
      (pearlCProtocolDevRev1 d sem ρ).inertBoundOK H s U u act ∧ (pearlCProtocolDevRev1RowInertBound d sem ρr).inertBoundOK H s U u act }

noncomputable def pearlCTilesDevRev1RowInertBoundAt (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρr : ℚ) :
    TileRules Q R S :=
  { pearlCTilesDevRev1 d sem ρ with
    inertBoundOK := fun H s U g act =>
      (pearlCTilesDevRev1 d sem ρ).inertBoundOK H s U g act ∧ (pearlCTilesDevRev1RowInertBound d sem ρr).inertBoundOK H s U g act }

/-- **(C3)**: a hypothesis on the abstract `CM`, a property of W1 (which charges instructions, not which rows form a
unit), and no cryptographic assumption. -/
def CostModel.SplitClosed (CM : CostModel Q R S) : Prop :=
  ∀ (prog : CM.Prog) (U : Workload) (act : ℕ → Codes), ∃ (prog' : CM.Prog) (act' : ℕ → Codes),
    ∀ (H : Q → R) (σ : State) (s : S),
      CM.run prog' H (splitWorkload U) act' σ s = splitTrace U.layout (CM.run prog H U act σ s) ∧
        CM.cost prog' H (splitWorkload U) act' σ s ≤ CM.cost prog H U act σ s ∧
        CM.calls prog' H (splitWorkload U) act' σ s ≤ CM.calls prog H U act σ s

/-- **The row-seeded TT_OUT**, a target form proved from rev1's TT_OUT, not assumed. -/
def TTOutRowSeed [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : Layout → Prop) (g₀ : ℝ) : Prop :=
  CM.Queries ∧
  ∀ U : Workload, U.InDomain P D → rowsOf U.layout ≤ 2 ^ 64 → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 →
    ∀ A : Adversary CM, A.Bounded U T q →
      pr (fun x : (Q → R) × S =>
        T / (1 - g₀) < ∑ u ∈ correctSet P U x.1 x.2 (A.trace U x.1 x.2),
          P.credit x.1 x.2 U u (actOf (A.trace U x.1 x.2) u))
        ≤ δPearlC q (rowsOf U.layout)

/-- **The row-seeded TT_OUT per audit tile**, a target form, not assumed. -/
def TTOutTileRowSeed [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : Layout → Prop) (g₀ : ℝ) : Prop :=
  CM.Queries ∧
  ∀ U : Workload, U.InDomain P D → rowsOf U.layout ≤ 2 ^ 64 → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 →
    ∀ A : Adversary CM, A.Bounded U T q →
      pr (fun x : (Q → R) × S =>
        T / (1 - g₀) < ∑ g ∈ goodTiles P TR U x.1 x.2 (A.trace U x.1 x.2),
          TR.credit x.1 x.2 U g (actOf (A.trace U x.1 x.2) ((TR.tiling U.layout).unit g)))
        ≤ δPearlC q (rowsOf U.layout)

/-- **(C1-h3) The code-form seed**: the code's seed is C1's up to a fixed permutation `π L` of the oracle's inputs,
one per layout. -/
def CodeSeedNoise (sem' semC : PearlCSem Q R S) (π : Layout → Q ≃ Q) : Prop :=
  semC.fc = sem'.fc ∧ semC.alpha = sem'.alpha ∧ semC.beta = sem'.beta ∧ semC.floor = sem'.floor ∧
    ∀ (H : Q → R) (s : S) (U : Workload) (u : ℕ) (act : Codes),
      semC.noise H s U u act = sem'.noise (H ∘ π U.layout) s U u act

/-- **(C5) The cost model is closed under relabelling the oracle.** -/
def CostModel.RelabelClosed (CM : CostModel Q R S) : Prop :=
  ∀ (π : Q ≃ Q) (prog : CM.Prog) (U : Workload) (act : ℕ → Codes), ∃ prog' : CM.Prog,
    ∀ (H : Q → R) (σ : State) (s : S),
      CM.run prog' H U act σ s = CM.run prog (H ∘ π) U act σ s ∧
        CM.cost prog' H U act σ s ≤ CM.cost prog (H ∘ π) U act σ s ∧
        CM.calls prog' H U act σ s ≤ CM.calls prog (H ∘ π) U act σ s

def omitMask (atoms : List AtomIn) (S : Finset ℕ) : List Bool :=
  List.ofFn fun t : Fin atoms.length => decide ((t : ℕ) ∈ S)

def chainOmitP (p : Pipeline) (G : ℕ) (atoms : List AtomIn) (S : Finset ℕ) : Option ℚ :=
  ((atoms.toChunks G).zip ((omitMask atoms S).toChunks G)).foldl
    (fun t q => promote t (Skip.foldMask (SkipP.stepP p) 0 q.1 q.2)) (some 0)

def SkipSetU (p : Pipeline) (G : ℕ) (atoms : List AtomIn) (S : Finset ℕ) : Prop :=
  chainOmitP p G atoms S = Accum.chainP p G atoms

def undebitedSkippable (p : Pipeline) (G : ℕ) {k : ℕ} (a b : Fin k → Code) (t : ℕ) : Prop :=
  (∃ S : Finset ℕ, t ∈ S ∧ SkipSetU p G (rowAtoms a b) S) ∧ SkipP.flagsAt p G k a b t = false

open Classical in
/-- **Row `i`'s fragment count** `Y_i`, over aligned 8-column groups `J` and atoms `t`. -/
noncomputable def fragCount (p : Pipeline) (G : ℕ) {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code)
    (B : Matrix (Fin n) (Fin k) Code) (i : Fin m) : ℕ :=
  ((range (n / 8) ×ˢ range (k / 32)).filter fun Jt =>
    ∀ j : Fin n, (j : ℕ) / 8 = Jt.1 → undebitedSkippable p G (A i) (B j) Jt.2).card

def omittedAt (slots : Finset (ℕ × ℕ × ℕ)) (patch : ℕ × ℕ × ℕ → Finset (ℕ × ℕ)) (i j : ℕ) : Finset ℕ :=
  (slots.filter fun s => s.1 = i / 16 ∧ s.2.1 = j / 8 ∧ (i, j) ∉ patch s).image fun s => s.2.2

end Pouw.PearlC

namespace Pouw.QTree

def hits {Q R α : Type} (Bad : Q → R → Prop) (H : Q → R) : QTree Q R α → Prop
  | done _ => False
  | query x k => Bad x (H x) ∨ hits Bad H (k (H x))

end Pouw.QTree

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

variable {Q R S : Type}

/-- **A skip program on a unit**: its slots `(b, J, t)` are omitted `mma.sync m16n8k32` instructions. A patched word
keeps the slot's atom (`SkipProgram.out`), not a free value: free values would let 26 slots over one fragment leave all
128 of its words unconstrained. -/
structure SkipProgram where
  act : Codes
  slots : Finset (ℕ × ℕ × ℕ)
  patch : ℕ × ℕ × ℕ → Finset (ℕ × ℕ)

def SkipProgram.WellFormed (m n k : ℕ) (sp : SkipProgram) : Prop :=
  ∀ s ∈ sp.slots, 16 * s.1 + 16 ≤ m ∧ s.2.1 < n / 8 ∧ s.2.2 < k / 32 ∧ (sp.patch s).card ≤ 5

def SkipProgram.out (p : Pipeline) (G : ℕ) (sp : SkipProgram) {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code)
    (B : Matrix (Fin n) (Fin k) Code) (i : Fin m) (j : Fin n) : Option ℚ :=
  chainOmitP p G (rowAtoms (A i) (B j)) (omittedAt sp.slots sp.patch i j)

def SkipProgram.Correct (p : Pipeline) (G : ℕ) (sp : SkipProgram) {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code)
    (B : Matrix (Fin n) (Fin k) Code) : Prop :=
  ∀ (i : Fin m) (j : Fin n), sp.out p G A B i j = Accum.chainP p G (rowAtoms (A i) (B j))

def SkipProgram.Undebited (p : Pipeline) (G : ℕ) (sp : SkipProgram) {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code)
    (B : Matrix (Fin n) (Fin k) Code) : Prop :=
  ∀ s ∈ sp.slots, ∀ (i : Fin m) (j : Fin n), (i : ℕ) / 16 = s.1 → (j : ℕ) / 8 = s.2.1 →
    ((i : ℕ), (j : ℕ)) ∉ sp.patch s → SkipP.flagsAt p G k (A i) (B j) s.2.2 = false

/-- **The skip class** on unit `u` against `H`. It leaves out a slot whose atom is flagged in only some of its
unpatched words, which saves 4,096 units but is debited `32 + 32/G` a flagged word. -/
def SkipProgram.InClass (sem : PearlCSem Q R S) (p : Pipeline) (G : ℕ) (s : S) (U : Workload) (u : ℕ)
    (sp : SkipProgram) (H : Q → R) : Prop :=
  sp.WellFormed (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k ∧
    sem.OperandOK (U.layout.shape u).m (U.layout.shape u).k sp.act ∧
    (∀ i < (U.layout.shape u).m, ∀ l < (U.layout.shape u).k, sp.act i l < 2 ^ 32) ∧
    sp.Correct p G (sem.unitChain H s U u sp.act).1 (sem.unitChain H s U u sp.act).2 ∧
    sp.Undebited p G (sem.unitChain H s U u sp.act).1 (sem.unitChain H s U u sp.act).2

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

/-- **(C4) Whole-fragment pricing**, a condition on the abstract `CM`: each omitted `m16n8k32` saves at most its whole
fragment. On a per-instruction W1 model it is a lemma. -/
def CostModel.FragmentPriced (CM : CostModel Q R S) {sem : PearlCSem Q R S} {p : Pipeline} {G : ℕ} {U : Workload}
    {u : ℕ} (e : SkipEmbed CM sem p G U u) : Prop :=
  ∀ (sp : SkipProgram) (H : Q → R) (act : ℕ → Codes) (σ : State) (s : S),
    CM.cost e.honest H U act σ s ≤ CM.cost (e.embed sp) H U act σ s + 4096 * sp.slots.card

def rowWords (k : ℕ) (act : Codes) (i : ℕ) : Fin k → ℕ := fun l => act i l

/-- **(C6) Each row's noise is one oracle sample.** `key`'s injectivity idealizes the row leaf's collision resistance;
`rowDrawn_satisfiable` shows this form can hold. -/
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
