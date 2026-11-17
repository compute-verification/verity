module

public import Definitions.Claims.SamplingGeneric

/-! Claims: Flock's extractor at one table of a session, whose waits read only that table's level-0 length. -/

@[expose] public section

namespace Claims.Flock

/-- **Table `j`'s relation**: `table` gives a statement at index `j` of `x`, and the witness is in that statement's
relation. It reads `table` as given, and does not ask that `j` is one of the statement's tables: a record whose `table`
gives a statement at an index past the tables (Flock's deployed records give the default statement there) has that
statement's relation at the index, and the bound of `KSTable` there is at least `1`, where the word is empty. -/
def RTable (table : Msg → ℕ → Option Stmt) (j : ℕ) : Rel Wit := fun x w => ∃ t, table x j = some t ∧ RelS t w

/-- The decidable relation at table `j`, as `E^rep`'s check. -/
def checkTable (table : Msg → ℕ → Option Stmt) (j : ℕ) (x : Msg) (w : Wit) : Bool :=
  match table x j with
  | some t => decide (RelS t w)
  | none => false

/-- **Flock's assemble step at table `j`**, table `j`'s step of `assemble`: table `j`'s opened columns (`o j` of every
opening `o`) as they were opened, decoded by the one decoder to candidate traces, and the first candidate that, read
alone, is in the relation of table `j`'s own statement, read as a witness. The list is checked once, candidate by
candidate. -/
def assembleTable {Col T : Type} (table : Msg → ℕ → Option Stmt) (decode : List (ℕ × Col) → List T)
    (read : List T → Wit) (j : ℕ) (x : Msg) (opens : List (ℕ → List (ℕ × Col))) : Option Wit :=
  ((table x j).bind fun t =>
    (decode (opens.flatMap fun o => o j)).find? fun M => decide (RelS t (read [M]))).map fun M => read [M]

/-- `E^rep`'s accepted reruns per wait at table `j`, `k_j = ⌈4·N_j/e⌉`, with `N x j` table `j`'s level-0 codeword
length. -/
noncomputable def kTable (N : Msg → ℕ → ℕ) (j : ℕ) (x : Msg) : ℕ := ⌈4 * (N x j : ℝ) / Real.exp 1⌉₊

section Table

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **Flock's extractor at table `j`**, `E^rep` at a decoder: the reruns are the session's, each wait stops at `k_j`
accepted reruns (`kTable`), an accepted rerun's opening is its tables' columns (`D.cols v`), of which `assembleTable`
decodes and checks table `j`'s against the table's own statement (`D.table`). -/
noncomputable def ETable (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (decode : List (ℕ × Col) → List T) (j : ℕ) :
    Extractor Coin Wit D.V :=
  ERep D.V D.waits (kTable N j) D.cap (fun v => [D.cols v]) (assembleTable D.table decode D.read j)
    (checkTable D.table j)

/-- **Flock's knowledge error at table `j`**: `κ_j = 4·ε_j + 2^{−m′} + 2k_j/K + e^{−k_j/8}`, with `ε x j` the error
table `j` is bounded at. -/
noncomputable def κTable (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ) (j : ℕ) (x : Msg) : ℝ :=
  4 * ε x j + 2 ^ (-(D.waits x : ℝ)) + 2 * kTable N j x / D.cap x + Real.exp (-(kTable N j x : ℝ) / 8)

/-- **Flock: KS at each table of a session**, at `E^rep` at the table (`ETable`) and every decoder that meets the
list-decoding spec: at every table `j`, a session accepts while `ETable` returns no witness of table `j`'s own statement
with probability at most `κ_j + √(γ·Pr[B])`, `γ` and `B` the session's. -/
def KSTable [Zero Col] (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ) : Prop :=
  ∀ decode, ListWithin D.len D.enc D.θ decode → ∀ j,
    Claims.KS H D.V (RTable D.table j) (ETable D N decode j) (κTable D N ε j) (γ D)

end Table

end Claims.Flock

namespace Claims.SamplingGeneric

open Claims

section FlockTable

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

def flockAssembleTable (D : Claims.Flock.Deploy Coin Col T) (j : ℕ) (x : Msg) (os : List (ℕ → List (ℕ × Col))) :
    Option Claims.Flock.Wit × ℕ :=
  (Claims.Flock.assembleTable D.table D.decode D.read j x os,
    (D.dec (os.flatMap fun o => o j)).2 + (D.decode (os.flatMap fun o => o j)).length)

/-- The verifier is Flock's at the named statement, so a session's run is the whole session's whatever table is named. -/
noncomputable def Arg.ofFlockTable (D : Claims.Flock.Deploy Coin Col T) (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ)
    (names : Msg → ℕ × Msg) : Arg Coin where
  Sh := ℕ × ℕ × ℕ × ℕ
  shape y := (D.waits (names y).2, D.cap (names y).2, D.V.k (names y).2, N (names y).2 (names y).1)
  V := ⟨fun y => D.V.k (names y).2, fun y => D.V.next (names y).2⟩
  rounds s := s.2.2.1
  wf _ := rfl
  W := Claims.Flock.Wit
  R y := Claims.Flock.RTable D.table (names y).1 (names y).2
  Op := ℕ → List (ℕ × Col)
  opens v := [D.cols v]
  assemble y := flockAssembleTable D (names y).1 (names y).2
  check y := Claims.Flock.checkTable D.table (names y).1 (names y).2
  waits s := s.1
  k s := ⌈4 * (s.2.2.2 : ℝ) / Real.exp 1⌉₊
  cap s := s.2.1
  κ y := Claims.Flock.κTable D N ε (names y).1 (names y).2
  γ y := Claims.Flock.γ D (names y).2

/-- A statement about the proof system at a statement, not about a compiled verifier. -/
def FlockRerunsTable : Prop :=
  ∀ (D : Claims.Flock.Deploy Coin Col T) (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ) (names : Msg → ℕ × Msg) (y : Msg),
    (Arg.ofFlockTable D N ε names).waits ((Arg.ofFlockTable D N ε names).shape y) = D.waits (names y).2 ∧
      (Arg.ofFlockTable D N ε names).k ((Arg.ofFlockTable D N ε names).shape y) =
        Claims.Flock.kTable N (names y).1 (names y).2

def FlockArgKSTable : Prop :=
  ∀ (H : Msg → Msg) (D : Claims.Flock.Deploy Coin Col T) [Zero Col] (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ)
    (names : Msg → ℕ × Msg),
    Claims.Flock.KSTable H D N ε → Claims.Flock.ListWithin D.len D.enc D.θ D.decode →
      (Arg.ofFlockTable D N ε names).KS H

def FlockKSTable : Prop :=
  ∀ (H : Msg → Msg) (D : Claims.Flock.Deploy Coin Col T) [Zero Col] (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ)
    (names : Msg → ℕ × Msg) (L : Compiled Coin (Arg.ofFlockTable D N ε names)),
    Claims.Flock.KSTable H D N ε → Claims.Flock.ListWithin D.len D.enc D.θ D.decode → L.commit.Binds H →
      L.Sessions H → L.KS H

def FlockKSRegTable : Prop :=
  ∀ (H : Msg → Msg) (D : Claims.Flock.Deploy Coin Col T) [Zero Col] (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ)
    (names : Msg → ℕ × Msg) (L : Compiled Coin (Arg.ofFlockTable D N ε names)),
    Claims.Flock.KSTable H D N ε → Claims.Flock.ListWithin D.len D.enc D.θ D.decode → L.commit.Binds H →
      L.Sessions H → L.KSReg H

end FlockTable

end Claims.SamplingGeneric
