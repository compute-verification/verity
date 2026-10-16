import Definitions.Claims

/-!
# Claims: the sampling compiler (draft, not frozen)

The claims block of the sampled-proofs one-stage audit (`verity/core/protocols/sampled_proofs/`), on the shared
definitions of `Definitions.Claims`, in the template's eight items. Parametric in the deployed sampled verifier
(`Deploy`), as Flock's block is in `Flock.Deploy`: binding it to the compiled verifier is the code side's.

1. **Statement** (`Stmt`): the circuit as its regions, each region's check (the public Program it must pass) on the
   wires it reads and produces; the public values; the registered roots and the rows committed under them; the draw's
   law, which fixes how many regions are sampled; and the confidence `δ` the profile is published at. `x` is the
   canonical bytes of what the verifier reads before it draws (its parameters and the registration record), and
   `parse` is its own: its decoding, and R1–R7 against its own copies. The statement holds the roots: without them
   every statement would have a witness, the honest evaluation.
2. **Witness** (`Wit`): the values on the wires that cross region boundaries, which are the rows, with each row's salt
   and path to its root.
3. **Relation** (`R`): fewer than `t` regions are wrong, where a region is right when the witness opens every row it
   reads under the registered roots and its check maps its inputs to its outputs.
4. **Scope** (`InScope`, `ScopeMatches`): the law is well formed over the regions, and `0 < δ < 1`.
5. **Named pieces**: the assemble step (`assemble`, `inner`): a region's boundary values are what Flock's `E^rep`
   recovers in the reruns that sampled it; Flock's decoder (`Flock.ListComplete`); the draw's escape and miss
   (`Law.escape`, `Law.miss`); the expected-time finder (`Bext`); and the parameters (`Deploy`).
6. **Claims**: `KS` at every `t`, with `κ_t = C(N, t)^{1/k}·(miss(t) + κ_F) + 2k/K + e^{−k/8}` and `β = 1` (commit
   strings opened inside Flock proofs are bound); `Complete`; `ZK`; `Nontrivial`; `Bits`; and `Efficient`.
7. **Assumptions used**: `StrictBirthday` (for `B`), `ExpectedCR` (for `Bext`), `CommitHides` (for zero knowledge,
   through Flock's), and `trustedBase`.
8. **Status**: stated, as a draft for review; no claim is proved.

Generic pieces this block adds, which belong with the shared definitions when it freezes: `ownLog`, `ofOC`,
`flatMapRP`, `waitAccC`, `Bext`, `lamLin`, `BitsLin`, `ExtractOwnLe`, `SimAux` and `ZKaux`.
-/

namespace Claims.Sampling

/-! ## 1. Statement -/

/-- Where a region's wire value comes from: public value `i`, or bit `b` of committed row `r`. -/
inductive Src where
  | pub (i : ℕ)
  | row (r b : ℕ)

/-- A region: its check, the public Program it must pass, on the wires it reads (`inp`) and the wires it must produce
(`out`). Regions share a wire by naming the same row. -/
structure Region where
  C : Circuit
  inp : List Src
  out : List Src

/-- A committed row: the root it is committed under (an index into `Stmt.roots`), its position there, and its width in
bits. -/
structure Row where
  root : ℕ
  pos : ℕ
  width : ℕ

/-- The draw's law over the regions `0, …, N − 1`, as the registration fixes it: a uniform `k`-subset; each region
independently at rate `num/den`; or a uniform `k_s`-subset of each stratum. A `work` law is registered expanded, as the
stratified law its rule gives; its closure only draws more. -/
inductive Law where
  | subset (k : ℕ)
  | bernoulli (num den : ℕ)
  | stratified (strata : List (List ℕ × ℕ))

/-- **The sampling statement**: the circuit as its regions, the public values, the registered roots and the rows
committed under them, the draw's law, and the confidence `δ` the profile is published at. -/
structure Stmt where
  regions : List Region
  pub : List Bool
  roots : List Msg
  rows : List Row
  law : Law
  δ : ℚ

/-! ## 2. Witness -/

/-- **The sampling witness**: each row's values, salt and path to its root, by row. -/
structure Wit where
  vals : List (List Bool)
  salts : List Msg
  paths : List Msg

/-- A row's opening: the row, its values, salt and path. -/
structure Opening where
  row : ℕ
  vals : List Bool
  salt : Msg
  path : Msg

/-- The witness's opening of row `r`. -/
def Wit.opening (w : Wit) (r : ℕ) : Opening := ⟨r, w.vals.getD r [], w.salts.getD r [], w.paths.getD r []⟩

/-! ## 3. Relation -/

/-- Whether `o` opens its row: its width is the row's, and `path` shows the commit string `commit(vals; salt)` at the
row's position under its root. `path root pos leaf π` is the registered scheme's path check (frame-v3-sha512). The
relation and the extractor run this one program. -/
def openOC (path : Msg → ℕ → Msg → Msg → OC Bool) (s : Stmt) (o : Opening) : OC Bool :=
  match s.rows[o.row]? with
  | none => .ret false
  | some row =>
    if o.vals.length = row.width then
      .query (packBits o.vals ++ o.salt) fun leaf => path (s.roots.getD row.root []) row.pos leaf o.path
    else .ret false

def val (s : Stmt) (w : Wit) : Src → Bool
  | .pub i => s.pub.getD i false
  | .row r b => (w.vals.getD r []).getD b false

/-- The rows a region reads or produces. -/
def Region.rows (g : Region) : List ℕ :=
  (g.inp ++ g.out).filterMap fun
    | .row r _ => some r
    | .pub _ => none

/-- A region is **right** under `w` when `w` opens every row it reads or produces, and its check maps its inputs to its
outputs. -/
def Right (H : Msg → Msg) (path : Msg → ℕ → Msg → Msg → OC Bool) (s : Stmt) (w : Wit) (g : Region) : Prop :=
  (∀ r ∈ g.rows, ((openOC path s (w.opening r)).eval H).1 = true) ∧
    g.C.eval (g.inp.map (val s w)) = g.out.map (val s w)

instance instDecidableRight (H : Msg → Msg) (path : Msg → ℕ → Msg → Msg → OC Bool) (s : Stmt) (w : Wit)
    (g : Region) : Decidable (Right H path s w g) := by
  unfold Right; infer_instance

/-- The number of regions wrong under `w`. -/
def wrong (H : Msg → Msg) (path : Msg → ℕ → Msg → Msg → OC Bool) (s : Stmt) (w : Wit) : ℕ :=
  (s.regions.filter fun g => !decide (Right H path s w g)).length

/-- **The sampling relation** on a parsed statement: fewer than `t` regions are wrong. -/
def RelS (H : Msg → Msg) (path : Msg → ℕ → Msg → Msg → OC Bool) (t : ℕ) (s : Stmt) (w : Wit) : Prop :=
  wrong H path s w < t

/-- **The sampling relation** on the statement's bytes: it parses (`parse`, the verifier's own) and `RelS` holds. -/
def R (H : Msg → Msg) (path : Msg → ℕ → Msg → Msg → OC Bool) (parse : Msg → Option Stmt) (t : ℕ) : Rel Wit :=
  fun x w => ∃ s, parse x = some s ∧ RelS H path t s w

/-! ## 4. Scope -/

/-- The law is well formed over `N` regions: a subset of `1 ≤ k ≤ N`; a rate `num/den` in `[0, 1]`; strata that
partition the regions, each drawing `1 ≤ k_s ≤ n_s`. -/
def Law.WellFormed (N : ℕ) : Law → Prop
  | .subset k => 1 ≤ k ∧ k ≤ N
  | .bernoulli num den => 0 < den ∧ num ≤ den
  | .stratified ss => (ss.flatMap Prod.fst).Perm (List.range N) ∧ ∀ p ∈ ss, 1 ≤ p.2 ∧ p.2 ≤ p.1.length

/-- **The sampling scope, readable**: the law is well formed over the regions, and `0 < δ < 1`. -/
def InScope (s : Stmt) : Prop := s.law.WellFormed s.regions.length ∧ 0 < s.δ ∧ s.δ < 1

/-! ## 5. Named pieces -/

/-- **The draw's escape**: the probability that the law's draw over `N` regions misses every region of `B`:
`C(N − |B|, k)/C(N, k)`, `((den − num)/den)^|B|`, or `∏_s C(n_s − |B ∩ s|, k_s)/C(n_s, k_s)`. -/
noncomputable def Law.escape (N : ℕ) : Law → Finset ℕ → ℝ
  | .subset k, B => ((N - B.card).choose k : ℝ) / (N.choose k : ℝ)
  | .bernoulli num den, B => (((den - num : ℕ) : ℝ) / den) ^ B.card
  | .stratified ss, B =>
    (ss.map fun p => ((p.1.length - (B.filter (· ∈ p.1)).card).choose p.2 : ℝ) / (p.1.length.choose p.2 : ℝ)).prod

/-- **The draw's miss at `t`**: the largest escape of `t` or more of the `N` regions. -/
noncomputable def Law.miss (L : Law) (N t : ℕ) : ℝ :=
  ⨆ B : {B : Finset ℕ // B ⊆ Finset.range N ∧ t ≤ B.card}, L.escape N B.1

/-- What the deployed sampled verifier brings, read by the claims: the verifier, its parser and scope check, the
registered scheme's path check, the Flock verifier its sessions run (`flock`; the claims read its parser, columns, code,
decoding threshold and count, witness reading and masking, not its numbers), where a run's session is (`session`: the session's statement and what its
run showed), how a session's witness opens rows (`openings`, from the statement, the run and the witness), and the
numbers the errors read. The extractor waits for `k` accepted reruns among `K` (`cap`). The session's `E^rep` at `x`
makes `m′` waits (`innerWaits`) of `K′` reruns (`innerCap`) with `N₀` (`innerN₀`), `ε_table` and the weight `C` of
`γ`, each of them good for every session the draws at `x` make. `openCalls` bounds the SHA-512 calls of checking one
rerun's openings. -/
structure Deploy (Coin Col T : Type) where
  V : Verifier Coin
  parse : Msg → Option Stmt
  scopeCheck : Msg → Bool
  path : Msg → ℕ → Msg → Msg → OC Bool
  flock : Flock.Deploy Coin Col T
  session : Seen → Option (Msg × Seen)
  openings : Stmt → Seen → Flock.Wit → List Opening
  k : Msg → ℕ
  cap : Msg → ℕ
  innerWaits : Msg → ℕ
  innerN₀ : Msg → ℝ
  innerCap : Msg → ℕ
  εtable : Msg → ℝ
  weight : Msg → ℝ
  openCalls : Msg → ℕ

/-! ### Rerun programs -/

/-- An oracle program as a rerun program: its `H` calls are the extractor's own. -/
def ofOC {R α : Type} : OC α → RP R α
  | .ret a => .ret a
  | .query q k => .hash q fun a => ofOC (k a)

/-- The `H` calls a rerun program makes itself, in order, as `(query, answer)`: not its reruns'. -/
def ownLog {R α : Type} (H : Msg → Msg) (run : R → Trace) : RP R α → List (Msg × Msg)
  | .ret _ => []
  | .hash q k => (q, H q) :: ownLog H run (k (H q))
  | .rerun r k => ownLog H run (k (run r).seen)

/-- Run `f` on each element in order, concatenating the results. -/
def flatMapRP {R β γ : Type} (f : β → RP R (List γ)) : List β → RP R (List γ)
  | [] => .ret []
  | b :: bs => (f b).bind fun cs => (flatMapRP f bs).bind fun ds => .ret (cs ++ ds)

/-- `waitAcc`, keeping each accepted rerun's coins with what it showed. -/
def waitAccC {R : Type} (k : ℕ) : List R → List (R × Seen) → RP R (List (R × Seen))
  | [], acc => .ret acc
  | r :: rs, acc =>
    if k ≤ acc.length then .ret acc
    else .rerun r fun v => waitAccC k rs (if v.out = .acc then acc ++ [(r, v)] else acc)

/-- A rerun's coins: the draw's coin (coin 0) from `r`, the rest from `ρ`. -/
def atDraw {Coin : Type} {k : ℕ} (r ρ : Fin k → Coin) : Fin k → Coin := fun i => if i.val = 0 then r i else ρ i

/-! ### The extractor -/

section Extract

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- The session's `E^rep` waits for `k′ = ⌈4·N₀/e⌉` accepted reruns. -/
noncomputable def Deploy.innerK (D : Deploy Coin Col T) (x : Msg) : ℕ := ⌈4 * D.innerN₀ x / Real.exp 1⌉₊

/-- The opened columns of a run's session. -/
def Deploy.sessionCols (D : Deploy Coin Col T) (v : Seen) : List (ℕ × Col) :=
  match D.session v with
  | some (_, vS) => D.flock.cols vS
  | none => []

/-- The openings among `os` that open their rows, each checked by `openOC`, its `H` calls the extractor's own. -/
def checkOpens {R : Type} (D : Deploy Coin Col T) (s : Stmt) (os : List Opening) : RP R (List Opening) :=
  flatMapRP (fun o => (ofOC (openOC D.path s o)).bind fun ok => .ret (if ok then [o] else [])) os

/-- **The session's extraction at an accepted rerun** `(r, v)`: Flock's `E^rep` on the session's statement, at a decoder,
without its replay (the rerun accepted), its reruns keeping `r`'s draw so the prover makes the same session; then the
openings its witness gives that open their rows. -/
noncomputable def inner (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (s : Stmt) (r : Fin (D.V.k x) → Coin) (v : Seen)
    (ρ : Fin (D.innerWaits x) → Fin (D.innerCap x) → Fin (D.V.k x) → Coin) :
    RP (Fin (D.V.k x) → Coin) (List Opening) :=
  match D.session v with
  | none => .ret []
  | some (xS, _) =>
    (firstWait (D.innerK x) D.sessionCols (Flock.assemble H D.flock.parse decode D.flock.read xS)
        (Flock.check H D.flock.parse xS)
        ((List.finRange (D.innerWaits x)).map fun i =>
          (List.finRange (D.innerCap x)).map fun j => atDraw r (ρ i j))).bind fun
      | none => .ret []
      | some wF => checkOpens D s (D.openings s v wF)

/-- **The assemble step**: each row's first opening among those the sessions recovered. -/
def assemble (s : Stmt) (os : List Opening) : Wit :=
  let first (r : ℕ) := os.find? (·.row = r)
  { vals := (List.range s.rows.length).map fun r => ((first r).map (·.vals)).getD []
    salts := (List.range s.rows.length).map fun r => ((first r).map (·.salt)).getD []
    paths := (List.range s.rows.length).map fun r => ((first r).map (·.path)).getD [] }

/-- **The sampling extractor** at a decoder: replay the judged run; if it accepted, rerun on fresh coins until `k`
accepted runs are in hand (giving up after `K`), run the session's extraction at each, and assemble. -/
noncomputable def E (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) :
    Extractor Coin Wit D.V where
  C x := (Fin (D.cap x) → Fin (D.V.k x) → Coin) ×
    (Fin (D.k x) → Fin (D.innerWaits x) → Fin (D.innerCap x) → Fin (D.V.k x) → Coin)
  ext x r₀ c := .rerun r₀ fun v₀ =>
    match D.parse x with
    | none => .ret none
    | some s =>
      if v₀.out = .acc then
        (waitAccC (D.k x) ((List.finRange (D.cap x)).map c.1) []).bind fun rvs =>
          (flatMapRP (fun p => inner H D decode x s p.2.1 p.2.2 (c.2 p.1)) ((List.finRange (D.k x)).zip rvs)).bind
            fun os => .ret (some (assemble s os))
      else .ret none

end Extract

/-! ### The expected-time finder, bits with `β > 0`, and the extractor's own hashing -/

section Finder

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **`B_ext`** (expected time): `B`'s fork, the extractor run in each branch on that branch's run, and the first
collision among the `H` calls the two extractions make themselves (the openings they check). Its cost is both
extractions' SHA-512 calls, their reruns' included. -/
noncomputable def Bext (H : Msg → Msg) (V : Verifier Coin) (E : Extractor Coin W V) : FinderFamily :=
  fun Ω _ _ P x =>
    if hk : 0 < V.k x then
      { S := BCoins Ω Coin (V.k x) × E.C x × E.C x
        ne := ⟨(⟨⟨0, hk⟩, Classical.arbitrary _, fun _ => default, fun _ => default⟩, Classical.arbitrary _,
          Classical.arbitrary _)⟩
        out := fun s =>
          findColl (ownLog H (run H P V x s.1.2.1) (E.ext x s.1.2.2.1 s.2.1) ++
            ownLog H (run H P V x s.1.2.1) (E.ext x (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2) s.2.2))
        cost := fun s =>
          (E.eval H P x s.1.2.1 s.1.2.2.1 s.2.1).2.1 +
            (E.eval H P x s.1.2.1 (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2) s.2.2).2.1 }
    else { S := Unit, out := fun _ => none, cost := fun _ => 0 }

/-- `λ` with `β > 0`: `min(log₂(1/κ), 255.5 − ½ log₂ γ, 256 − log₂(2βc)) − log₂ 3`, where `c` is the extractor's expected
work in runs (`1 + mk` for `E^rep`). -/
noncomputable def lamLin (κ γ β c : ℝ) : ℝ :=
  min (min (Real.logb 2 (1 / κ)) (255.5 - Real.logb 2 γ / 2)) (256 - Real.logb 2 (2 * β * c)) - Real.logb 2 3

/-- `λ` bits with an extraction-based finder `Fe`: as `Bits`, where `Fe` also obeys the expected-time bound. -/
def BitsLin (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (E : Extractor Coin W V) (Fe : FinderFamily)
    (lam : Msg → ℝ) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (t : ℕ), 1 ≤ t →
    (∀ ω r, (run H P V x ω r).pLog.length + (run H P V x ω r).vLog.length ≤ t) →
    (∀ hk : 0 < V.k x, StrictBirthday H (B H V P x hk)) →
    ExpectedCR H (Fe Ω P x) →
    PrBad H V R E x P ≤ t * (2 : ℝ) ^ (-lam x)

/-- **Cost, the extractor's own hashing**: `E` makes at most `calls(x)` SHA-512 calls itself, not counting its reruns',
on every prover, tape and coin. -/
def ExtractOwnLe (H : Msg → Msg) {V : Verifier Coin} (E : Extractor Coin W V) (calls : Msg → ℕ) : Prop :=
  ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r₀ : Fin (V.k x) → Coin) (c : E.C x),
    (ownLog H (run H P V x ω) (E.ext x r₀ c)).length ≤ calls x

end Finder

/-! ### Zero knowledge given the leaf layer -/

/-- A simulator that also receives the leaf layer, every row's commit string. -/
structure SimAux where
  C : Msg → Type
  [fin : ∀ x, Fintype (C x)]
  [ne : ∀ x, Nonempty (C x)]
  prog : (x : Msg) → List Msg → C x → SP (List (Msg × Out))

attribute [instance] SimAux.fin SimAux.ne

/-- The simulator's transcript against `V*` at `θ`, given the leaf layer `aux`. -/
def SimAux.out {Θ : Type} (H : Msg → Msg) (S : SimAux) (Vs : VStar Θ) (θ : Θ) (x : Msg) (aux : List Msg)
    (c : S.C x) : List (Msg × Out) :=
  ((S.prog x aux c).eval H fun as => ((Vs.next x as θ).eval H).1).1

/-- `ZKmal`, where the simulator also receives `aux(x, w)`: for `(x, w) ∈ R`, every verifier `V*` and tape `θ`, the
transcript of `P(w)` against `V*` is within `ε_zk(x)` of `S`'s, on every event. -/
def ZKaux {Ω W : Type} [Fintype Ω] (H : Msg → Msg) (P : W → Prover Ω) (R : Rel W) (aux : Msg → W → List Msg)
    (S : SimAux) (εzk : Msg → ℝ) : Prop :=
  ∀ x w, R x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ) (Ev : List (Msg × Out) → Prop),
    |Pr (fun ω : Ω => Ev (runStar H (P w) Vs x ω θ).msgs) - Pr (fun c : S.C x => Ev (S.out H Vs θ x (aux x w) c))| ≤
      εzk x

/-! ## 6. Claims -/

section Claims

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- The regions at `x`, `0` on a statement that doesn't parse. -/
def nRegions (D : Deploy Coin Col T) (x : Msg) : ℕ :=
  match D.parse x with
  | some s => s.regions.length
  | none => 0

/-- The law's miss at `t` at `x`, `0` on a statement that doesn't parse. -/
noncomputable def missAt (D : Deploy Coin Col T) (t : ℕ) (x : Msg) : ℝ :=
  match D.parse x with
  | some s => s.law.miss s.regions.length t
  | none => 0

/-- The session's knowledge error, Flock's at the inner numbers: `κ_F = 4·ε_table + 2^{−m′} + 2k′/K′ + e^{−k′/8}`. -/
noncomputable def κF (D : Deploy Coin Col T) (x : Msg) : ℝ :=
  4 * D.εtable x + 2 ^ (-(D.innerWaits x : ℝ)) + 2 * D.innerK x / D.innerCap x + Real.exp (-(D.innerK x : ℝ) / 8)

/-- The cover factor `C(N, t)^{1/k}`: a union over the `C(N, t)` sets of `t` regions, spread over `k` accepted
reruns. -/
noncomputable def cover (D : Deploy Coin Col T) (t : ℕ) (x : Msg) : ℝ :=
  ((nRegions D x).choose t : ℝ) ^ (1 / (D.k x : ℝ))

/-- **The knowledge error at `t`**: `κ_t = C(N, t)^{1/k}·(miss(t) + κ_F) + 2k/K + e^{−k/8}`. -/
noncomputable def κ (D : Deploy Coin Col T) (t : ℕ) (x : Msg) : ℝ :=
  cover D t x * (missAt D t x + κF D x) + 2 * D.k x / D.cap x + Real.exp (-(D.k x : ℝ) / 8)

/-- **`γ` at `t`**: `C(N, t)^{2/k}·16·(√(2·k(x)·C) + k(x))²`, with `k(x)` the sampled verifier's rounds. -/
noncomputable def γ (D : Deploy Coin Col T) (t : ℕ) (x : Msg) : ℝ :=
  cover D t x ^ 2 * (16 * (Real.sqrt (2 * D.V.k x * D.weight x) + D.V.k x) ^ 2)

/-- The extractor's expected work in runs, `c = 1 + k·(1 + m′·k′) + k·openCalls`: its expected reruns, and its own
SHA-512 calls. -/
noncomputable def work (D : Deploy Coin Col T) (x : Msg) : ℝ :=
  1 + D.k x * (1 + D.innerWaits x * D.innerK x) + D.k x * D.openCalls x

/-- The leaf layer under `w`: every row's commit string `commit(vals; salt)`. -/
def leaves (H : Msg → Msg) (D : Deploy Coin Col T) (x : Msg) (w : Wit) : List Msg :=
  match D.parse x with
  | some s => (List.range s.rows.length).map fun r => commit H (packBits (w.vals.getD r [])) (w.salts.getD r [])
  | none => []

/-- The sampling scope, on the statement's bytes. -/
def Scope (D : Deploy Coin Col T) (x : Msg) : Prop := ∃ s, D.parse x = some s ∧ InScope s

/-- **Sampling: KS**, at every `t`, at the sampling extractor and every decoder that meets Flock's list-decoding spec:
`Pr[acc ∧ ¬R_t] ≤ κ_t + √(γ_t·Pr[B]) + Pr[B_ext]`. -/
def KS (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ t decode, Flock.ListComplete D.flock.enc D.flock.agree D.flock.nMin decode →
    Claims.KSLin H D.V (R H D.path D.parse t) (E H D decode) (κ D t) (γ D t) (fun _ => 1)
      (Bext H D.V (E H D decode))

/-- **Sampling: Complete**, on the statements in scope, when every region is right. -/
def Complete {Ω : Type} (H : Msg → Msg) (D : Deploy Coin Col T) (P : Wit → Prover Ω) (δc : Msg → ℝ) : Prop :=
  Claims.Complete H D.V P (R H D.path D.parse 1) (Scope D) δc

/-- **Sampling: the readable scope is the verifier's.** -/
def ScopeMatches (D : Deploy Coin Col T) : Prop := Claims.ScopeMatches (Scope D) D.scopeCheck

/-- **Sampling: zero knowledge** of the honest prover with masked sessions, when every region is right, against every
malicious verifier, the simulator given the leaf layer: the transcript shows nothing but hiding commit strings. -/
def ZK {Ω : Type} [Fintype Ω] (H : Msg → Msg) (D : Deploy Coin Col T) (P : Wit → Prover Ω) (S : SimAux)
    (εzk : Msg → ℝ) : Prop :=
  D.flock.zk = true ∧ ZKaux H P (R H D.path D.parse 1) (leaves H D) S εzk

/-- **Sampling: bits** at every `t`, `λ_t = min(log₂(1/κ_t), 255.5 − ½ log₂ γ_t, 256 − log₂(2c)) − log₂ 3`, at every
decoder that meets the spec. -/
def Bits (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ t decode, Flock.ListComplete D.flock.enc D.flock.agree D.flock.nMin decode →
    BitsLin H D.V (R H D.path D.parse t) (E H D decode) (Bext H D.V (E H D decode))
      fun x => lamLin (κ D t x) (γ D t x) 1 (work D x)

/-- **Sampling: efficiency**: the extractor's reruns, at most `1 + K + k·m′·K′` and `1 + k·(1 + m′·k′)` expected; its
own SHA-512 calls, at most `k·openCalls`; and a decoder that meets Flock's list-decoding spec, with its stated
complexity. -/
def Efficient (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T)
    (counted : List (ℕ × Col) → List T × ℕ) (size : List (ℕ × Col) → ℕ) (p : Polynomial ℕ) : Prop :=
  ExtractRunsLe H (E H D decode) (fun x => 1 + D.cap x + D.k x * (D.innerWaits x * D.innerCap x)) ∧
    ExtractRunsExpLe H (E H D decode) (fun x => 1 + D.k x * (1 + D.innerWaits x * D.innerK x)) ∧
    ExtractOwnLe H (E H D decode) (fun x => D.k x * D.openCalls x) ∧
    Flock.ListComplete D.flock.enc D.flock.agree D.flock.nMin decode ∧ DecoderComplexity decode counted size p

/-- **Sampling: nontriviality**: recorded honest runs, each of a statement the scope check takes, that the verifier
accepts. -/
def Nontrivial (H : Msg → Msg) (D : Deploy Coin Col T) (recs : List (Recorded D.V)) : Prop :=
  Claims.Nontrivial H D.V recs ∧ ∀ e ∈ recs, D.scopeCheck e.x = true

end Claims

end Claims.Sampling
