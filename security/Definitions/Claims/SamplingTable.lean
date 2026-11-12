module

public import Definitions.Claims.SamplingGeneric

/-!
# Claims: Flock's extractor at one table of a session

`Flock.E` extracts a session whole: each wait of `E^rep` stops at `k = ⌈4·N₀/e⌉` accepted reruns, with `N₀` the level-0
codeword lengths of the session's tables summed, and `Flock.assemble` returns a witness of every table or none. So a
session's expected reruns, `1 + m′·k`, grow with its tables, and so does `Compiled.es` at `Arg.ofFlock`.

This file states the extractor at one table of a session, beside `Flock.E`, `Flock.KS`, `Arg.ofFlock` and
`FlockArgKS`, which it leaves as they stand:

1. **The extractor** (`Flock.EAt D N decode j`): `E^rep` at the session's verifier, so a rerun is the whole session's,
   on fresh coins at every round, against the prover at the judged run's tape, and an accepted rerun is one the whole
   session accepts. Each wait stops at `k_j = ⌈4·N_j/e⌉` accepted reruns (`Flock.kAt`), with `N_j` table `j`'s level-0
   codeword length alone. Its assemble step is table `j`'s step of `Flock.assemble` (`Flock.assembleAt`): table `j`'s
   opened columns of the reruns, decoded by the one decoder, and the first candidate that, read alone, is a witness of
   table `j`'s own statement. Nothing of the session's other tables is read or held.
2. **Its relation** (`Flock.RAt`): the statement has a table `j`, and the witness is in the relation of that table's
   own statement.
3. **Its knowledge error** (`Flock.κAt`): `Flock.κ` with `k_j` for `k` and an error for each table (`ε x j`) for
   `ε_table`. At `ε x j = ε_table x` it is `Flock.κ` but for the count; at table `j`'s own error it no longer reads
   the session's other tables.
4. **The claim** (`Flock.KSAt`): `Claims.KS` of `Flock.EAt` at every table, at every decoder that meets the
   list-decoding spec.
5. **The proof system for proof blocks** (`Arg.ofFlockAt`): Flock at one table of a session, a second `Arg` beside
   `Arg.ofFlock`. Its statement names a table and a Flock statement (`names`); its verifier is Flock's at the named
   statement, its extractor `Flock.EAt` at the named table and the deployed decoder (`Arg.ext` at it), and its shape
   holds the named table's length, so `Arg.k` there is `k_j`. `Compiled.es` reads `Arg.waits` and `Arg.k` at each proof
   block's shape, so at this `Arg` it is the largest of `1 + m′·k_j` over the proof blocks' tables, with no change to
   `Compiled` (`FlockRerunsAt` states what its numbers are).
6. **Flock's statements at it** (`FlockArgKSAt`, `FlockKSAt`, `FlockKSRegAt`): `FlockArgKS`, `FlockKS` and
   `FlockKSReg` at `Arg.ofFlockAt`, from `Flock.KSAt`.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.Flock

/-- **Table `j`'s relation**: the statement has a table `j`, and the witness is in the relation of that table's own
statement (`table x j`). -/
def RAt (table : Msg → ℕ → Option Stmt) (j : ℕ) : Rel Wit := fun x w => ∃ t, table x j = some t ∧ RelS t w

/-- The decidable relation at table `j`, as `E^rep`'s check. -/
def checkAt (table : Msg → ℕ → Option Stmt) (j : ℕ) (x : Msg) (w : Wit) : Bool :=
  match table x j with
  | some t => decide (RelS t w)
  | none => false

/-- **Flock's assemble step at table `j`**, table `j`'s step of `assemble`: table `j`'s opened columns (`o j` of every
opening `o`) as they were opened, decoded by the one decoder to candidate traces, and the first candidate that, read
alone, is in the relation of table `j`'s own statement, read as a witness. The list is checked once, candidate by
candidate. -/
def assembleAt {Col T : Type} (table : Msg → ℕ → Option Stmt) (decode : List (ℕ × Col) → List T)
    (read : List T → Wit) (j : ℕ) (x : Msg) (opens : List (ℕ → List (ℕ × Col))) : Option Wit :=
  ((table x j).bind fun t =>
    (decode (opens.flatMap fun o => o j)).find? fun M => decide (RelS t (read [M]))).map fun M => read [M]

/-- `E^rep`'s accepted reruns per wait at table `j`, `k_j = ⌈4·N_j/e⌉`, with `N x j` table `j`'s level-0 codeword
length. -/
noncomputable def kAt (N : Msg → ℕ → ℕ) (j : ℕ) (x : Msg) : ℕ := ⌈4 * (N x j : ℝ) / Real.exp 1⌉₊

section Table

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **Flock's extractor at table `j`**, `E^rep` at a decoder: the reruns are the session's, each wait stops at `k_j`
accepted reruns (`kAt`), an accepted rerun's opening is its tables' columns (`D.cols v`), of which `assembleAt` decodes
and checks table `j`'s against the table's own statement (`D.table`). -/
noncomputable def EAt (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (decode : List (ℕ × Col) → List T) (j : ℕ) :
    Extractor Coin Wit D.V :=
  ERep D.V D.waits (kAt N j) D.cap (fun v => [D.cols v]) (assembleAt D.table decode D.read j) (checkAt D.table j)

/-- **Flock's knowledge error at table `j`**: `κ_j = 4·ε_j + 2^{−m′} + 2k_j/K + e^{−k_j/8}`, with `ε x j` the error
table `j` is bounded at. -/
noncomputable def κAt (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ) (j : ℕ) (x : Msg) : ℝ :=
  4 * ε x j + 2 ^ (-(D.waits x : ℝ)) + 2 * kAt N j x / D.cap x + Real.exp (-(kAt N j x : ℝ) / 8)

/-- **Flock: KS at each table of a session**, at `E^rep` at the table (`EAt`) and every decoder that meets the
list-decoding spec: at every table `j`, a session accepts while `EAt` returns no witness of table `j`'s own statement
with probability at most `κ_j + √(γ·Pr[B])`, `γ` and `B` the session's. -/
def KSAt [Zero Col] (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ) : Prop :=
  ∀ decode, ListWithin D.len D.enc D.θ decode → ∀ j,
    Claims.KS H D.V (RAt D.table j) (EAt D N decode j) (κAt D N ε j) (γ D)

end Table

end Claims.Flock

namespace Claims.SamplingGeneric

open Claims

section FlockTable

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **Flock's assemble step at table `j`, counted**: `Flock.assembleAt` at the deployed decoder (`D.decode`), with its
steps: the decoder's on table `j`'s opened columns (`D.dec`'s count) and one check per candidate it lists. -/
def flockAssembleAt (D : Claims.Flock.Deploy Coin Col T) (j : ℕ) (x : Msg) (os : List (ℕ → List (ℕ × Col))) :
    Option Claims.Flock.Wit × ℕ :=
  (Claims.Flock.assembleAt D.table D.decode D.read j x os,
    (D.dec (os.flatMap fun o => o j)).2 + (D.decode (os.flatMap fun o => o j)).length)

/-- **Flock at one table of a session, as a proof system for proof blocks**: a statement `y` names a table and a Flock
statement (`names y`). The verifier is Flock's at the named statement, so a session's run is the session's whatever
table is named; the extractor is `E^rep` at the named table and the deployed decoder (`Flock.EAt D N D.decode`), its
assemble step counted (`flockAssembleAt`); the shape is the numbers it reads (waits `m′`, truncation `K`, rounds, and
the named table's level-0 codeword length, which fixes each wait's `k_j = ⌈4·N_j/e⌉`, `Flock.kAt`, and so its expected
reruns); and its errors are Flock's at the table (`Flock.κAt`) and the session's `γ` (`Flock.γ`). -/
noncomputable def Arg.ofFlockAt (D : Claims.Flock.Deploy Coin Col T) (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ)
    (names : Msg → ℕ × Msg) : Arg Coin where
  Sh := ℕ × ℕ × ℕ × ℕ
  shape y := (D.waits (names y).2, D.cap (names y).2, D.V.k (names y).2, N (names y).2 (names y).1)
  V := ⟨fun y => D.V.k (names y).2, fun y => D.V.next (names y).2⟩
  rounds s := s.2.2.1
  wf _ := rfl
  W := Claims.Flock.Wit
  R y := Claims.Flock.RAt D.table (names y).1 (names y).2
  Op := ℕ → List (ℕ × Col)
  opens v := [D.cols v]
  assemble y := flockAssembleAt D (names y).1 (names y).2
  check y := Claims.Flock.checkAt D.table (names y).1 (names y).2
  waits s := s.1
  k s := ⌈4 * (s.2.2.2 : ℝ) / Real.exp 1⌉₊
  cap s := s.2.1
  κ y := Claims.Flock.κAt D N ε (names y).1 (names y).2
  γ y := Claims.Flock.γ D (names y).2

/-- **The reruns at one table are that table's**: at each statement, `Arg.ofFlockAt`'s waits are Flock's and each stops
at the named table's `k_j` (`Flock.kAt`), so `Compiled.es` over it, which reads `Arg.waits` and `Arg.k` at each proof
block's shape, is the largest of `1 + m′·k_j` over the tables the proof blocks' sessions name. -/
def FlockRerunsAt : Prop :=
  ∀ (D : Claims.Flock.Deploy Coin Col T) (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ) (names : Msg → ℕ × Msg) (y : Msg),
    (Arg.ofFlockAt D N ε names).waits ((Arg.ofFlockAt D N ε names).shape y) = D.waits (names y).2 ∧
      (Arg.ofFlockAt D N ε names).k ((Arg.ofFlockAt D N ε names).shape y) =
        Claims.Flock.kAt N (names y).1 (names y).2

/-- **Flock's KS at each table is `Arg.ofFlockAt`'s**: `Flock.KSAt` at the deployed decoder, which is within Flock's
radius, at every way a statement names a table and a Flock statement. -/
def FlockArgKSAt : Prop :=
  ∀ (H : Msg → Msg) (D : Claims.Flock.Deploy Coin Col T) [Zero Col] (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ)
    (names : Msg → ℕ × Msg),
    Claims.Flock.KSAt H D N ε → Claims.Flock.ListWithin D.len D.enc D.θ D.decode →
      (Arg.ofFlockAt D N ε names).KS H

/-- **The sampled compiler at Flock at one table of a session**: every compiled sampled verifier over
`Arg.ofFlockAt` at the deployed decoder, within its radius, that meets the contract, with a binding commitment, meets
`KS`, from `Flock.KSAt` (`CompilerKS` and `FlockArgKSAt`). `Flock.KSAt`, the binding and the contract are its
hypotheses. -/
def FlockKSAt : Prop :=
  ∀ (H : Msg → Msg) (D : Claims.Flock.Deploy Coin Col T) [Zero Col] (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ)
    (names : Msg → ℕ × Msg) (L : Compiled Coin (Arg.ofFlockAt D N ε names)),
    Claims.Flock.KSAt H D N ε → Claims.Flock.ListWithin D.len D.enc D.θ D.decode → L.commit.Binds H →
      L.Sessions H → L.KS H

/-- **The committed form at Flock at one table of a session**: as `FlockKSAt`, `KSReg` (`CompilerKSReg` and
`FlockArgKSAt`). -/
def FlockKSRegAt : Prop :=
  ∀ (H : Msg → Msg) (D : Claims.Flock.Deploy Coin Col T) [Zero Col] (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ)
    (names : Msg → ℕ × Msg) (L : Compiled Coin (Arg.ofFlockAt D N ε names)),
    Claims.Flock.KSAt H D N ε → Claims.Flock.ListWithin D.len D.enc D.θ D.decode → L.commit.Binds H →
      L.Sessions H → L.KSReg H

end FlockTable

end Claims.SamplingGeneric
