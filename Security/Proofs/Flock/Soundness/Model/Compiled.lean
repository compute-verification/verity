import Proofs.Flock.Soundness.Model.Session
import Proofs.Flock.Soundness.Merkle

/-!
# The compiled protocol: Merkle caps, openings after the last coin (PROTOCOL.md §13–§14)

The protocol as run commits each table by a Merkle cap (§14) and opens the queried rows with
their sibling paths in the proof file, which the server receives after every round (S17). The
openings enter no round, so the prover chooses them after all coins of both reps: the table's
last message here (`Opens`, per rep, level and query). The verifier then checks every path
against its level's cap and computes the consistency values from the opened rows.

The model follows `Model.Ligerito` step for step. The running claim and the round polynomial are
affine in the consistency values, so the compiled verifier carries them as functions of the
openings (`LigStateC.s`, `LigStateC.q`) and evaluates them once the openings arrive; every query
records its request (`Req`) for the path checks.

**Salted leaves.** `tableCL` is `tableC` under a leaf scheme (`Merkle.Leaf`): the last message carries the openings
and their salts (`Salts`), and each path verifies from its row's leaf under its salt (`OpensOKL`). At `Leaf.plain`,
`OpensOKL` is `OpensOK` (`opensOKL_plain`).
-/

namespace FlockSoundness.Model

open FlockSoundness.Game FlockSoundness.Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

/-- The openings, sent after the last coin: per rep, level and query, the opened row (its lanes'
elements) and its sibling path (§13.3 steps 9 and the recursive rounds, §14). -/
abbrev Opens (F D : Type) := ℕ → ℕ → ℕ → List F × List D

/-- A Merkle cap, entry by index (§14). -/
abbrev Cap (D : Type) := ℕ → D

/-- A level's cap depth `c = min(⌊log2 Q⌋, d)`, and the path length `d − c` (§14). -/
def capDepth (lv : Level) : ℕ := min (Nat.log 2 lv.queries) lv.logLen

def pathLen (lv : Level) : ℕ := lv.logLen - capDepth lv

/-- The consistency value from opened rows, query `j` reading row `j` (§13.3 step 10). -/
def consValueO (A : Arith F K) (α : List F) (ch : List K) (rows : ℕ → List F) (pos : List ℕ) :
    K :=
  ((pos.zipIdx).map fun pj => A.embed (eqList α pj.2) *
    foldColumn A ch fun ℓ => (rows pj.2).getD ℓ 0).sum

/-- A query request: the level, its positions, the lanes folded, and the path length. -/
structure Req where
  level : ℕ
  positions : List ℕ
  lanes : ℕ
  depth : ℕ

/-- The compiled verifier's state: the claim and the round polynomial as functions of the
openings, the level whose rows the next queries read, the caps received, the requests made. -/
structure LigStateC (F K D : Type) where
  s : Opens F D → K
  q : Opens F D → Quad K
  splitDim : ℕ
  levels : List (List K)
  oods : List (OodCtx F)
  cons : List (ConsCtx F)
  level : ℕ
  caps : List (Cap D)
  reqs : List Req
  prevLevel : Level

/-- A rep's result, pending the openings. -/
structure PendL (F D : Type) where
  verdict : Opens F D → Bool
  caps : List (Cap D)
  reqs : List Req

/-- A rep that rejected before Ligerito. -/
def PendL.reject : PendL F D := ⟨fun _ => false, [], []⟩

/-- `foldRounds` with the claim a function of the openings. -/
def foldRoundsC (A : Arith F K) : ℕ → (Opens F D → K) → (Opens F D → Quad K) →
    Game (List K × (Opens F D → K) × (Opens F D → Quad K))
  | 0, s, q => pure ([], s, q)
  | k + 1, _, q => do
      let r ← drawK A
      let s' := fun o => (q o).eval r
      let msg ← recv (K × K)
      let (rs, s'', q') ← foldRoundsC A k s' (fun o => fromMsg msg (s' o))
      pure (r :: rs, s'', q')

/-- `twoOod` with the claim a function of the openings. -/
def twoOodC (A : Arith F K) (dim : ℕ) (split : ℕ) (s : Opens F D → K) (q : Opens F D → Quad K) :
    Game ((Opens F D → K) × (Opens F D → Quad K) × List (OodCtx F)) := do
  let z₁ ← draw (Fin dim → F)
  let y₁ ← recv F
  let m₁ ← recv (K × K)
  let β₁ ← draw F
  let q := fun o => (q o).addScaled (fromMsg m₁ (A.embed y₁)) (A.embed β₁)
  let s := fun o => s o + A.embed β₁ * A.embed y₁
  let z₂ ← draw (Fin dim → F)
  let y₂ ← recv F
  let m₂ ← recv (K × K)
  let β₂ ← draw F
  let q := fun o => (q o).addScaled (fromMsg m₂ (A.embed y₂)) (A.embed β₂)
  let s := fun o => s o + A.embed β₂ * A.embed y₂
  pure (s, q, [⟨List.ofFn z₁, β₁, some split⟩, ⟨List.ofFn z₂, β₂, some split⟩])

/-- `queryAndBatch` on rep `r`'s current level: the consistency value is a function of the
openings, and the queries are recorded. -/
def queryAndBatchC (A : Arith F K) (r : ℕ) (lv : Level) (ch : List K) (start : ℕ)
    (st : LigStateC F K D) (withMessage : Bool) : Game (LigStateC F K D) := do
  let w ← draw (Fin lv.queries → F)
  let α ← draw (Fin (ceilLog2 lv.queries) → F)
  let pos := positions A lv.queries lv.logLen w
  let e : Opens F D → K := fun o => consValueO A (List.ofFn α) ch (fun j => (o r st.level j).1) pos
  let msg ← if withMessage then do
      let msg ← recv (K × K)
      pure (some msg)
    else pure none
  let β ← draw F
  let ctx : ConsCtx F := ⟨lv.logCols, pos, List.ofFn α, β, start⟩
  pure { st with
    s := fun o => st.s o + A.embed β * e o
    q := match msg with
      | some m => fun o => (st.q o).addScaled (fromMsg m (e o)) (A.embed β)
      | none => st.q
    cons := st.cons ++ [ctx]
    reqs := st.reqs ++ [⟨st.level, pos, ch.length, pathLen lv⟩] }

/-- `recursiveRounds` with caps for the tables. -/
def recursiveRoundsC (A : Arith F K) (r : ℕ) (n : ℕ) (b : ℕ → F) (initial : List K) (ext : ℕ) :
    List Level → LigStateC F K D → Game (PendL F D)
  | [], _ => pure PendL.reject
  | [lv], st => do
      let (ch, s, q) ← foldRoundsC A lv.lanes st.s st.q
      let st := { st with s := s, q := q, levels := st.levels ++ [ch] }
      let yr ← recv (Fin (2 ^ (ext + 1)) → F)
      let st ← queryAndBatchC A r lv ch (st.levels.length) st false
      pure ⟨fun o => finalCheck A n b initial st.levels st.oods st.cons ext yr (st.s o),
        st.caps, st.reqs⟩
  | lv :: lv' :: rest, st => do
      let (ch, s, q) ← foldRoundsC A lv.lanes st.s st.q
      let st := { st with s := s, q := q, levels := st.levels ++ [ch] }
      let dim := lv'.logCols + lv'.lanes
      let C' ← recv (Cap D)
      let (s, q, oods) ← twoOodC A dim st.levels.length st.s st.q
      let st := { st with s := s, q := q, oods := st.oods ++ oods }
      let st ← queryAndBatchC A r lv ch st.levels.length st true
      recursiveRoundsC A r n b initial ext (lv' :: rest)
        { st with level := st.level + 1, caps := st.caps ++ [C'], prevLevel := lv' }

/-- `ligerito` for rep `r` with caps for the tables; the level-0 cap is the table's. -/
def ligeritoC (A : Arith F K) (r : ℕ) (sch : Schedule) (T : F) (b : ℕ → F) :
    Game (PendL F D) := do
  let n := sch.m - 7
  match sch.levels with
  | [] => pure PendL.reject
  | lv₀ :: rest => do
      let z₀ ← draw (Fin n → F)
      let y₀ ← recv F
      let β₀ ← draw F
      let s := A.embed T + A.embed β₀ * A.embed y₀
      let msg ← recv (K × K)
      let (initial, s, q) ← foldRoundsC A sch.k0 (fun _ => s) (fun _ => fromMsg msg s)
      match rest with
      | [] => pure PendL.reject
      | lv₁ :: _ => do
          let C₁ ← recv (Cap D)
          let dim := n - sch.k0 + 1
          let (s, q, oods) ← twoOodC A dim 0 s q
          let st : LigStateC F K D :=
            ⟨s, q, dim, [], ⟨List.ofFn z₀, β₀, none⟩ :: oods, [], 0, [], [], lv₀⟩
          let st ← queryAndBatchC A r lv₀ initial 0 st true
          recursiveRoundsC A r n b initial sch.ext rest
            { st with level := 1, caps := [C₁], prevLevel := lv₁ }

/-- One compiled rep (§9–§13): the PIOP as in `rep`, then compiled Ligerito. -/
def repC (A : Arith F K) (r : ℕ) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m)
    (extra : List (Claim F S.m)) : Game (PendL F D) := do
  match ← zerocheck A S.m with
  | none => pure PendL.reject
  | some zc =>
    match ← lincheck A S zc with
    | none => pure PendL.reject
    | some ab =>
      match ← opening A (by omega) (ab :: cClaim A zc :: extra) with
      | none => pure PendL.reject
      | some op => ligeritoC A r sch op.T op.b

/-- The cap a request's paths verify against: the table's cap at level 0, else the rep's. -/
def capOf (cap₀ : Cap D) (caps : List (Cap D)) (level : ℕ) : Option (Cap D) :=
  if level = 0 then some cap₀ else caps[level - 1]?

open Classical in
/-- Every requested opening of a rep is a row of the level's lane count whose path verifies
against the level's cap (§14). -/
def OpensOK (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (cap₀ : Cap D) (r : ℕ)
    (p : PendL F D) (o : Opens F D) : Prop :=
  ∀ req ∈ p.reqs, ∃ cap, capOf cap₀ p.caps req.level = some cap ∧
    ∀ j < req.positions.length, (o r req.level j).1.length = 2 ^ req.lanes ∧
      Merkle.Verifies H E cap req.depth (req.positions.getD j 0) (o r req.level j).1
        (o r req.level j).2

/-- The compiled table's output: the verdict, the level-0 cap, both reps' results, and the
openings. -/
structure OutC (F D : Type) where
  accepted : Bool
  cap₀ : Cap D
  pend : PendL F D × PendL F D
  opens : Opens F D

open Classical in
/-- **One table's compiled session** (§5, §13–§14): the level-0 cap, the link points, both
reps, then the openings; accepted iff both reps accept on the openings and every path verifies. -/
noncomputable def tableC (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) : Game (OutC F D) := do
  let cap₀ ← recv (Cap D)
  let pts ← draw (Fin 2 → Fin mPts → F)
  let extra := extraClaims S ptLocal mPts pts
  let pend ← interleave (repC A 0 S sch hm extra) (repC A 1 S sch hm extra)
  let o ← recv (Opens F D)
  pure ⟨pend.1.verdict o && pend.2.verdict o &&
      decide (OpensOK H E cap₀ 0 pend.1 o ∧ OpensOK H E cap₀ 1 pend.2 o), cap₀, pend, o⟩

/-! ## Salted leaves (hm96) -/

/-- The salts sent with the openings: per rep, level and query, the opened row's salt (§14). -/
abbrev Salts (Sl : Type) := ℕ → ℕ → ℕ → Sl

open Classical in
/-- `OpensOK` for a leaf scheme `Lf`: every requested opening's path verifies from its row's leaf under its salt. -/
def OpensOKL {Sl : Type} (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (cap₀ : Cap D) (r : ℕ) (p : PendL F D) (o : Opens F D) (sl : Salts Sl) : Prop :=
  ∀ req ∈ p.reqs, ∃ cap, capOf cap₀ p.caps req.level = some cap ∧
    ∀ j < req.positions.length, (o r req.level j).1.length = 2 ^ req.lanes ∧
      Merkle.VerifiesL H E Lf cap req.depth (req.positions.getD j 0) (o r req.level j).1 (sl r req.level j)
        (o r req.level j).2

omit [Field F] [Fintype F] [DecidableEq F] in
theorem opensOKL_plain (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (cap₀ : Cap D) (r : ℕ) (p : PendL F D)
    (o : Opens F D) : OpensOKL H E (Merkle.Leaf.plain H E) cap₀ r p o (fun _ _ _ => ()) ↔ OpensOK H E cap₀ r p o :=
  Iff.rfl

/-- The compiled table's output under a leaf scheme: `OutC`, and the salts sent with the openings. -/
structure OutCL (F D Sl : Type) where
  accepted : Bool
  cap₀ : Cap D
  pend : PendL F D × PendL F D
  opens : Opens F D
  salts : Salts Sl

open Classical in
/-- **One table's compiled session under a leaf scheme `Lf`**: `tableC`, with the salts in the openings' message (one
final message, after the last coin), and every path verified from its row's leaf under its salt. -/
noncomputable def tableCL {Sl : Type} (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) :
    Game (OutCL F D Sl) := do
  let cap₀ ← recv (Cap D)
  let pts ← draw (Fin 2 → Fin mPts → F)
  let extra := extraClaims S ptLocal mPts pts
  let pend ← interleave (repC A 0 S sch hm extra) (repC A 1 S sch hm extra)
  let os ← recv (Opens F D × Salts Sl)
  pure ⟨pend.1.verdict os.1 && pend.2.verdict os.1 &&
      decide (OpensOKL H E Lf cap₀ 0 pend.1 os.1 os.2 ∧ OpensOKL H E Lf cap₀ 1 pend.2 os.1 os.2), cap₀, pend, os.1,
    os.2⟩

end FlockSoundness.Model
