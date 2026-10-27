import Definitions.Flock.Piop
import Definitions.Flock.Accounting.Bound

/-!
# Ligerito (PROTOCOL.md §13), as an interactive oracle proof

The verifier of §13.3 and its final check (§13.6), with one change of model: a level's
commitment is the committed table itself (an oracle, `ℕ → ℕ → F`, lane then position), and the
opened rows are that table's columns at the query positions. The Merkle layer (§14) replaces the
oracle by a cap and authentication paths; `Merkle` relates the two.

Sumcheck messages are `(u₀, u₂) ∈ K²`; with running claim `s` a message defines
`q(X) = u₀ + (s + u₂)·X + u₂·X²`, so `q(0) + q(1) = s` holds by construction (§13.2).
-/

namespace FlockSoundness.Model

open FlockSoundness.Game FlockSoundness.Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- A committed table as an oracle: `(lane, position) ↦ value`. -/
abbrev Oracle (F : Type) := ℕ → ℕ → F

/-- A round polynomial `q₀ + q₁·X + q₂·X²`. -/
abbrev Quad (K : Type) := K × K × K

/-- `from(msg, s)` (§13.2). -/
def fromMsg (msg : K × K) (s : K) : Quad K := (msg.1, s + msg.2, msg.2)

def Quad.eval (q : Quad K) (x : K) : K := q.1 + q.2.1 * x + q.2.2 * x ^ 2

/-- `q ← q + β·q_intro` coefficient-wise. -/
def Quad.addScaled (q p : Quad K) (β : K) : Quad K :=
  (q.1 + β * p.1, q.2.1 + β * p.2.1, q.2.2 + β * p.2.2)

/-- An out-of-domain context (§13.3): the point, the batching coin, and the split it binds
(`none` for the level-0 sample). -/
structure OodCtx (F : Type) where
  z : List F
  β : F
  split : Option ℕ

/-- A consistency context (§13.3 steps 11 and the recursive rounds). -/
structure ConsCtx (F : Type) where
  cols : ℕ
  positions : List ℕ
  α : List F
  β : F
  start : ℕ

/-- The greedy decomposition `Q = Σ 2^{c_i}`, `c_i = min(⌊log2 rem⌋, d)` (§13.5). -/
def strata (Q d : ℕ) : List ℕ :=
  go Q Q
where
  go : ℕ → ℕ → List ℕ
    | 0, _ => []
    | fuel + 1, rem =>
        if rem = 0 then [] else
          let c := min (Nat.log 2 rem) d
          c :: go fuel (rem - 2 ^ c)

/-- Query positions from the query coins (§13.5): summand by summand, stratum by stratum,
`pos = (s << (d − c)) | (lo(coin) mod 2^(d − c))`. -/
def positions (A : Arith F K) (Q d : ℕ) (w : Fin Q → F) : List ℕ :=
  let slots : List (ℕ × ℕ) := (strata Q d).flatMap fun c => (List.range (2 ^ c)).map fun s => (c, s)
  (slots.zipIdx).map fun ((c, s), j) =>
    s * 2 ^ (d - c) + (if h : j < Q then A.lo (w ⟨j, h⟩) else 0) % 2 ^ (d - c)

/-- `eq(α)` over `⌈log2 Q⌉` coordinates, at `j`. -/
def eqList {R : Type} [CommRing R] (α : List R) (j : ℕ) : R :=
  ((α.zipIdx).map fun (a, i) => if j.testBit i then a else 1 + a).prod

/-- The folded value `Σ_ℓ eq_K(challenges)[ℓ]·row[ℓ]` of an opened column. -/
def foldColumn (A : Arith F K) (ch : List K) (row : ℕ → F) : K :=
  ∑ ℓ ∈ Finset.range (2 ^ ch.length), eqList ch ℓ * A.embed (row ℓ)

/-- The batched consistency value `e = Σ_j eq(α)[j]·fold(row_j)` (§13.3 step 10). -/
def consValue (A : Arith F K) (α : List F) (ch : List K) (C : Oracle F) (pos : List ℕ) : K :=
  ((pos.zipIdx).map fun (p, j) =>
    A.embed (eqList α j) * foldColumn A ch fun ℓ => C ℓ p).sum

/-- Draw a `K` coin: one `sample_f128_vec(2)` squeeze `(w₀, w₁) ↦ w₀ + w₁·u` (§2.4). -/
def drawK (A : Arith F K) : Game K := do
  let w ← draw (Fin 2 → F)
  pure (A.ofLimbs (w 0) (w 1))

/-- `k` folding rounds (§13.3 step 4, and each recursive round): draw `r`, set `s ← q(r)`, read
the next message. Returns the challenges, the claim and the round polynomial. -/
def foldRounds (A : Arith F K) : ℕ → K → Quad K → Game (List K × K × Quad K)
  | 0, s, q => pure ([], s, q)
  | k + 1, _, q => do
      let r ← drawK A
      let s' := q.eval r
      let msg ← recv (K × K)
      let (rs, s'', q') ← foldRounds A k s' (fromMsg msg s')
      pure (r :: rs, s'', q')

/-- Two out-of-domain samples on a freshly committed split table (§13.3 step 6). -/
def twoOod (A : Arith F K) (dim : ℕ) (split : ℕ) (s : K) (q : Quad K) :
    Game (K × Quad K × List (OodCtx F)) := do
  let z₁ ← draw (Fin dim → F)
  let y₁ ← recv F
  let m₁ ← recv (K × K)
  let β₁ ← draw F
  let q := q.addScaled (fromMsg m₁ (A.embed y₁)) (A.embed β₁)
  let s := s + A.embed β₁ * A.embed y₁
  let z₂ ← draw (Fin dim → F)
  let y₂ ← recv F
  let m₂ ← recv (K × K)
  let β₂ ← draw F
  let q := q.addScaled (fromMsg m₂ (A.embed y₂)) (A.embed β₂)
  let s := s + A.embed β₂ * A.embed y₂
  pure (s, q, [⟨List.ofFn z₁, β₁, some split⟩, ⟨List.ofFn z₂, β₂, some split⟩])

/-- The coordinate-fold factor `cff(c) = 1 + c·(1 + u)` (§13.6). -/
def cff (A : Arith F K) (c : K) : K := 1 + c * (1 + A.u)

/-- `Π_{s ≥ a} cff(level[s][0])`. -/
def cffFrom (A : Arith F K) (levels : List (List K)) (a : ℕ) : K :=
  ((levels.drop a).map fun l => cff A (l.headD 0)).prod

/-- The multilinear extension of the batched basis over `K` (§12.4). -/
def bHat (A : Arith F K) (n : ℕ) (b : ℕ → F) (p : List K) : K :=
  ∑ y ∈ Finset.range (2 ^ n), A.embed (b y) * eqList p y

/-- The bits of `y` as `0/1` field elements (the clear-residual coordinates). -/
def bitsK (ext : ℕ) (y : ℕ) : List K := (List.range ext).map fun j => if y.testBit j then 1 else 0

/-- The final check (§13.6). -/
def finalCheck (A : Arith F K) (n : ℕ) (b : ℕ → F) (initial : List K)
    (levels : List (List K)) (oods : List (OodCtx F)) (cons : List (ConsCtx F)) (ext : ℕ)
    (yr : Fin (2 ^ (ext + 1)) → F) (s : K) : Bool :=
  let orig := initial ++ (levels.map List.tail).flatten
  let res : ℕ → K := fun y =>
    bHat A n b (orig ++ bitsK ext y) * cffFrom A levels 0
      + (oods.map fun o =>
          let fixed := match o.split with
            | none => orig
            | some a => (levels.getD a []) ++ ((levels.drop (a + 1)).map List.tail).flatten
          let scale := match o.split with
            | none => cffFrom A levels 0
            | some a => cffFrom A levels (a + 1)
          A.embed o.β * scale
            * ((List.zip (o.z.map A.embed) fixed).map fun (zt, ft) => 1 + zt + ft).prod
            * ((List.range ext).map fun j =>
                let zj := A.embed (o.z.getD (fixed.length + j) 0)
                if y.testBit j then zj else 1 + zj).prod).sum
      + (cons.map fun c =>
          let fixed := ((levels.drop c.start).map List.tail).flatten
          A.embed c.β * cffFrom A levels c.start
            * ((c.positions.zipIdx).map fun (p, j) =>
                A.embed (eqList c.α j)
                  * ((fixed.zipIdx).map fun (fi, i) =>
                      1 + fi * (1 + A.embed (A.What i (A.omega p)))).prod
                  * ((List.range ext).map fun i =>
                      if y.testBit i then A.embed (A.What (fixed.length + i) (A.omega p))
                      else 1).prod).sum).sum
  decide (orig.length + ext = n)
    && decide ((∑ y ∈ Finset.range (2 ^ ext),
        res y * (A.embed (yr ⟨(2 * y) % 2 ^ (ext + 1), Nat.mod_lt _ (Nat.two_pow_pos _)⟩)
          + A.u * A.embed (yr ⟨(2 * y + 1) % 2 ^ (ext + 1), Nat.mod_lt _ (Nat.two_pow_pos _)⟩)))
        = s)

/-- The state threaded through the recursive rounds. -/
structure LigState (F K : Type) where
  s : K
  q : Quad K
  splitDim : ℕ
  levels : List (List K)
  oods : List (OodCtx F)
  cons : List (ConsCtx F)
  /-- The previous commitment and its level (queried in the next round). -/
  prev : Oracle F
  prevLevel : Level

/-- Queries and the consistency claim on the previous commitment (§13.3 steps 7–11 and their
recursive counterparts). `ch` are the challenges that folded the previous level's lanes. -/
def queryAndBatch (A : Arith F K) (lv : Level) (C : Oracle F) (ch : List K) (start : ℕ)
    (st : LigState F K) (withMessage : Bool) : Game (LigState F K) := do
  let w ← draw (Fin lv.queries → F)                       -- queries (PoW is a message, R4)
  let α ← draw (Fin (ceilLog2 lv.queries) → F)            -- consistency coins
  let pos := positions A lv.queries lv.logLen w
  let e := consValue A (List.ofFn α) ch C pos
  let q ← if withMessage then do
      let msg ← recv (K × K)
      pure (some (fromMsg msg e))
    else pure none
  let β ← draw F                                          -- claim batching coin
  let ctx : ConsCtx F := ⟨lv.logCols, pos, List.ofFn α, β, start⟩
  pure { st with
    s := st.s + A.embed β * e
    q := match q with | some qi => st.q.addScaled qi (A.embed β) | none => st.q
    cons := st.cons ++ [ctx] }

/-- The recursive rounds (§13.3), one per remaining level. -/
def recursiveRounds (A : Arith F K) (n : ℕ) (b : ℕ → F) (initial : List K) (ext : ℕ) :
    List Level → LigState F K → Game Bool
  | [], _ => pure false
  | [lv], st => do
      -- the last level: fold its lanes, read yr, query it, final batching, final check
      let (ch, s, q) ← foldRounds A lv.lanes st.s st.q
      let st := { st with s := s, q := q, levels := st.levels ++ [ch] }
      let yr ← recv (Fin (2 ^ (ext + 1)) → F)
      let st ← queryAndBatch A lv st.prev ch (st.levels.length) st false
      pure (finalCheck A n b initial st.levels st.oods st.cons ext yr st.s)
  | lv :: lv' :: rest, st => do
      let (ch, s, q) ← foldRounds A lv.lanes st.s st.q
      let st := { st with s := s, q := q, levels := st.levels ++ [ch] }
      let dim := lv'.logCols + lv'.lanes
      let C' ← recv (Oracle F)                               -- the next commitment
      let (s, q, oods) ← twoOod A dim st.levels.length st.s st.q
      let st := { st with s := s, q := q, oods := st.oods ++ oods }
      let st ← queryAndBatch A lv st.prev ch st.levels.length st true
      recursiveRounds A n b initial ext (lv' :: rest) { st with prev := C', prevLevel := lv' }

/-- **Ligerito** (§13.3): decides `Σ_y f(y)·b(y) = T` for the committed level-0 table `C₀`. -/
def ligerito (A : Arith F K) (sch : Schedule) (T : F) (b : ℕ → F) (C₀ : Oracle F) : Game Bool := do
  let n := sch.m - 7
  match sch.levels with
  | [] => pure false
  | lv₀ :: rest => do
      -- step 2: one out-of-domain sample on level 0
      let z₀ ← draw (Fin n → F)
      let y₀ ← recv F
      let β₀ ← draw F
      let s := A.embed T + A.embed β₀ * A.embed y₀
      -- steps 3–4: the first message and the level-0 folds
      let msg ← recv (K × K)
      let (initial, s, q) ← foldRounds A sch.k0 s (fromMsg msg s)
      match rest with
      | [] => pure false
      | lv₁ :: _ => do
          -- step 5: the level-1 commitment; step 6: its two out-of-domain samples
          let C₁ ← recv (Oracle F)
          let dim := n - sch.k0 + 1
          let (s, q, oods) ← twoOod A dim 0 s q
          let st : LigState F K := ⟨s, q, dim, [], ⟨List.ofFn z₀, β₀, none⟩ :: oods, [], C₀, lv₀⟩
          -- steps 7–11: queries on level 0 and its consistency claim
          let st ← queryAndBatch A lv₀ C₀ initial 0 st true
          recursiveRounds A n b initial sch.ext rest { st with prev := C₁, prevLevel := lv₁ }

end FlockSoundness.Model
