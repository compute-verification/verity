import Mathlib
import Definitions.Core.Interaction
import Definitions.Claims.Hm96
import Definitions.Claims.Json

/-!
# Claims: Flock's claim statements (frozen before any proof work)

Flock's claims, in the condensed spec's names, over Core's interaction model (`Definitions/Core/Interaction.lean`: the
runs, extractors, finders, simulators and the shared claim forms `KS`, `Complete`, `ZKmalR`, `Bits`), in these forms:

* Flock's extractor is `E^rep` (`ERep`): `κ = 4·ε_table + 2^{−m′}`, plus its truncation's `2k/K + e^{−k/8}`. Flock needs
  only `KS`, not the linear form `KSLin`.
* The scope is a hypothesis of completeness, in readable form (`Flock.InScope`), never part of `R`; `ScopeMatches`
  states that it is the verifier's own scope check. KS needs no scope: the verifier refuses what is out of scope.
* Zero knowledge (`ZKmalR`) binds the coin commitment only computationally, so its bound adds `γ` times `Fork`'s
  probability. The form first frozen, `ZKmal`, held only at `ε_zk ≥ 1`; the next charged the simulator's own calls,
  which a simulator that hashes a collision makes `1`, so it held at `ε_zk = 0`. Flock's rounds are the masked
  verifier's `D.V.k` plus the `Hello` that commits every coin (`Flock.ZK`), since `D.V` models the coins in the clear.
* Efficiency is the structural cost claims (`ERepCost`) and the decoder's steps. The steps are the deployed decoder's
  own count (`Flock.Deploy.dec`, a named, counted program), at the encoded length of the word it decodes
  (`Flock.wordLen`). The form first frozen (`DecoderComplexity`) took the decoder, its count and its size as arguments,
  so a count of `0` met it.
* Nontriviality (`Nontrivial`) is established by a test against recorded sessions, and no Lean proof of it is expected:
  Lean's kernel can't evaluate SHA-512 over real proofs. The test is `check`'s `lean-agreement` job
  (`tools/verity/check/ci.toml`), which runs the compiled Lean verifier on every recorded honest session
  (`verity/ml/flock/verifier/agree.py`) and fails unless it accepts each. `Flock.Nontrivial` states what that test
  checks.
* List decoding is in its radius form (`Flock.ListWithin`): the opened columns are written into the
  zero-filled word of the trace's codeword length (`Flock.fill`: at each position the first opened column there, the
  positions from the length on ignored), and the decoder lists every trace whose encoding agrees with that word on at
  least `(1 − θ)·len` positions, so leaving positions unopened or opening one twice gains a prover nothing. The radius
  is read per trace, like the length (`Flock.Deploy.len`, `Flock.Deploy.θ`): the plain level-0 rate is `1/2`
  at every schedule, but the masked code's varies with `m`, so one `θ` would cost bits. It replaces the form that
  listed a trace from `nMin` opened columns above the Johnson bound, under which a prover that avoids positions was
  accepted at about `2^{−8}`, so `KS` was false. `Flock.Efficient` asks it of the decoder it costs.
* Flock's relation is the block circuit in every block, and nothing else (`Flock.RelS`). The commitments are
  hm96 rows under frame-v3-sha512 trees that the circuit itself computes and outputs, so the relation reads them as
  the circuit's outputs, and their salts are among the hidden values (`Flock.Wit.hid`). The form first frozen also
  asked the witness to open commit strings `H(u ‖ s)` (`commit`, `Flock.Stmt.cs`), a form the deployed verifier never
  accepts: its commitments are those rows.
* Flock's claims read every table of a session: each of the statement's tables (`Flock.Stmt.tables`) commits
  its own level-0 codeword, so a run's openings are per table (`Flock.Deploy.cols`), `Flock.assemble` decodes each
  table's with the one decoder, takes each table's first candidate in that table's own statement
  (`Flock.Deploy.table`), and reads the tables' candidates as one witness (`Flock.Deploy.read`), and `N₀`, like
  `ε_table`, is summed over the tables: one wait's `k` accepted reruns open every table, so `k` counts all the tables'
  positions. The relation splits by table, each table's units their own statement, so `assemble` checks each table's
  list once, at most `J·L` checks where every combination would be `L^J`, and `Flock.Efficient` bounds each list by
  the decoder's polynomial, which counts them. At one table this is the text it replaces.
* Completeness (`Complete`) is at a hash whose outputs are 64 bytes (`Out64`), as the deployed oracle's are, SHA-512
  with its whole output, and as the random-oracle model draws them; it fixes the length only, so `H` stays a parameter.
  Flock's is of the deployed record's reference prover (`Flock.Deploy.P`), not of a prover the claim takes. The form
  first frozen held at an empty tape type for every verifier, and its prover was whatever its discharge chose, so an
  assembled claim's completeness premise was met for free.
* Zero knowledge is of the masked record's own reference prover and simulator (`Flock.ZkDeploy`): its prover
  is `Flock.Deploy.P`, the prover completeness reads, behind the `Hello` and the opening checks of the coins it
  commits, on a finite nonce and rest (`Flock.ZkDeploy.prover`), and its simulator is the record's
  (`Flock.ZkDeploy.S`). The form before took the prover and the simulator as arguments, so a prover that ignores its
  witness, with the simulator that runs it, met it at every masked record at `ε_zk = 0`.

Mathlib only, as `Definitions/` is. So every claim is parametric in the deployed verifier, its parser and the hash:
binding them to the compiled verifier (`Flock.verifyWith Sha512.hash`, `Flock.ProvedScope.check`) is the code side's.

The named assumptions: `StrictBirthday` and `ExpectedCR` (Core's), hm96's hiding, `Hm96Hiding`
(`Definitions/Claims/Hm96.lean`, which the soundness proofs read), and the JSON library's specification, `JsonSpec`
(`Definitions/Claims/Json.lean`, which only completeness reads). What the claims rest on and no Lean statement can say
is `trustedBase`.

Frozen by `Flock.Guarantees.ClaimsStated` (`Definitions/Flock/Guarantees/Claims.lean`), whose statement reads every claim
here but `Flock.ZK`, and the shared forms of `Definitions/Core/Interaction.lean` they instantiate, and by
`Flock.Guarantees.MaskedZK`, which states `Flock.ZK` at the deployed masked record: the lock records each one's text
under `reads`, so a changed claim is a changed record.
-/

namespace Claims

/-! ## Flock -/

namespace Flock

/-- Where a block's wire value comes from: a public value or a hidden one. -/
inductive Src where
  | pub (i : ℕ)
  | hid (i : ℕ)

/-- The form of a statement, as the verifier's scope check reads it. -/
structure Form where
  hm96Rows : Bool
  typed : Bool
  retainedRounds : Bool
  leaf : String
  tables : ℕ
  publicInputs : Bool
  sharedRows : Bool
  registeredReads : Bool
  hiddenOutputs : Bool
  unitInputGroups : ℕ
  tail : Bool
  unitLog : ℕ
  readsOnlyPortBits : Bool
  rowsV1 : Bool
  rowsV2OrSeg : Bool
  wiresAcyclic : Bool

/-- **Flock's statement**: `2^m` trace bits in blocks of `2^kLog`, the block circuit `C`, the blocks' inputs and outputs
(`inp b`, `out b`, reading the public or the hidden values), the public values, and the statement's form. The
commitments are no field of their own: they are hm96 rows under frame-v3-sha512 trees, which the circuit computes and
outputs (the forms `hm96Rows`, `rowsV1`, `rowsV2OrSeg`), so they are among `C`'s outputs, which the verifier checks
against the registered rows; their salts are among the hidden values. -/
structure Stmt where
  m : ℕ
  kLog : ℕ
  C : Circuit
  inp : ℕ → List Src
  out : ℕ → List Src
  pub : List Bool
  form : Form

def Stmt.nBlocks (s : Stmt) : ℕ := 2 ^ (s.m - s.kLog)

/-- **The statement's tables**: the session's table count (`form.tables`), at least one. Each table commits its own
level-0 codeword; a count of `0` or `1` is one table. -/
def Stmt.tables (s : Stmt) : ℕ := max 1 s.form.tables

/-- **Flock's witness**: the hidden values, the salts of the statement's hm96 rows among them. -/
structure Wit where
  hid : List Bool

def val (s : Stmt) (w : Wit) : Src → Bool
  | .pub i => s.pub.getD i false
  | .hid i => w.hid.getD i false

/-- **Flock's relation** on a parsed statement: `C` maps inputs to outputs in every block. That is the whole relation:
the commitments are hm96 rows under frame-v3-sha512 trees that `C` computes from the hidden values, salts included,
and outputs, so this clause reads them as `C`'s outputs. -/
def RelS (s : Stmt) (w : Wit) : Prop :=
  ∀ b < s.nBlocks, s.C.eval ((s.inp b).map (val s w)) = (s.out b).map (val s w)

instance instDecidableRelS (s : Stmt) (w : Wit) : Decidable (RelS s w) := by
  unfold RelS; infer_instance

/-- **Flock's relation** on the statement's bytes: it parses (`parse`, the verifier's own parser) and `RelS` holds. No
scope: the verifier refuses what is out of scope, and completeness carries the scope. -/
def R (parse : Msg → Option Stmt) : Rel Wit := fun x w => ∃ s, parse x = some s ∧ RelS s w

/-- The decidable relation, as `E^rep`'s check. -/
def check (parse : Msg → Option Stmt) (x : Msg) (w : Wit) : Bool :=
  match parse x with
  | some s => decide (RelS s w)
  | none => false

/-- **The most tables a `--zk` session may have**, `2^16`: the masked verifier refuses a session past it. -/
def zkTableCap : ℕ := 65536

/-- **Flock's scope, readable**: the deployed parameters (`22 ≤ m ≤ 35`, `25 ≤ m` under `--zk`, `6 ≤ kLog ≤ min(m, 27)`)
and the forms the soundness proof covers: hm96-sha512 rows, untyped, retained rounds, `hm96-sha512/v1` leaves or, with
hidden outputs and without `--zk`, plain SHA-512 leaves (`flock-leaf/sha512-unsalted`), at most
one table without `--zk` and `zkTableCap` with it, no public input ports, shared rows only with hidden outputs,
registered reads and `hm96-sha512/v1` leaves (with or without `--zk`); a unit of at most `2^32` rows reading only port bits, which reads one input group and returns the outputs, or,
with hidden outputs, the general instance form: the unit repeated, reading any number of committed input groups, and
any number of stages the verifier evaluates, reading wires from earlier outputs, with an order of the wires that
computes every slot from the leaves (`wiresAcyclic`); every row a `sha512/row/v1` row, or, with hidden outputs, every
row a `sha512/row/v2` or `sha512/row-seg/v1` row. -/
def InScope (zk : Bool) (s : Stmt) : Prop :=
  (if zk then 25 else 22) ≤ s.m ∧ s.m ≤ 35 ∧ 6 ≤ s.kLog ∧ s.kLog ≤ min s.m 27 ∧
    s.form.hm96Rows ∧ ¬ s.form.typed ∧ s.form.retainedRounds ∧
    (s.form.leaf = "hm96-sha512/v1" ∨ (¬ zk ∧ s.form.hiddenOutputs ∧ s.form.leaf = "flock-leaf/sha512-unsalted")) ∧
    s.form.tables ≤ (if zk then zkTableCap else 1) ∧ ¬ s.form.publicInputs ∧
    (s.form.sharedRows → s.form.hiddenOutputs ∧ s.form.registeredReads ∧ s.form.leaf = "hm96-sha512/v1") ∧
    ((s.form.unitInputGroups = 1 ∧ ¬ s.form.tail) ∨ (s.form.hiddenOutputs ∧ s.form.wiresAcyclic)) ∧
    s.form.unitLog ≤ 32 ∧ s.form.readsOnlyPortBits ∧
    (s.form.rowsV1 ∨ (s.form.hiddenOutputs ∧ s.form.rowsV2OrSeg))

/-- **The zero-filled word** of length `n`: at each position the first opened column there, zero where none was opened.
Positions from `n` on are ignored. -/
def fill {Col : Type} [Zero Col] (n : ℕ) (os : List (ℕ × Col)) : Fin n → Col :=
  fun i => ((os.find? fun p => p.1 == i.val).map Prod.snd).getD 0

/-- **List decoding within radius `θ`, the spec a decoder meets**: `decode` lists every trace `M` whose encoding agrees
with the zero-filled word of its codeword length `len M` (`fill`) on at least `(1 − θ M)·len M` positions. Agreement is
counted over all `len M` positions, so a prover gains nothing by leaving positions unopened. The radius is read per
trace, like the length. -/
def ListWithin {Col T : Type} [DecidableEq Col] [Zero Col] (len : T → ℕ) (enc : T → ℕ → Col) (θ : T → ℝ)
    (decode : List (ℕ × Col) → List T) : Prop :=
  ∀ (os : List (ℕ × Col)) (M : T),
    (1 - θ M) * len M ≤ ((Finset.univ : Finset (Fin (len M))).filter fun i => enc M i.val = fill (len M) os i).card →
    M ∈ decode os

/-- **The encoded length of an opened word**, in bits, the size the decoder's steps are bounded at: per opened column,
its position in binary (`Nat.size`), one bit that ends the entry, and each of its field elements in `⌈log₂ |F|⌉`
bits. -/
def wordLen {F : Type} [Fintype F] (os : List (ℕ × List F)) : ℕ :=
  (os.map fun o => Nat.size o.1 + 1 + o.2.length * Nat.clog 2 (Fintype.card F)).sum

/-- **Flock's assemble step**: each table's opened columns (`o j` of every opening `o`, for table `j` below the
statement's `Stmt.tables`) as they were opened, decoded by the one decoder to candidate traces; for each table, its
first candidate that, read alone, is in the relation of that table's own statement (`table x j`); the tables' choices,
in table order, read as one witness. Each table's list is checked once, candidate by candidate, so `assemble` makes at
most the lists' lengths summed in checks. At one table, whose own statement is the statement, this is the first
decoded trace of table `0`'s columns whose witness is in the relation. A position opened twice needs no step here:
`fill` reads the first. -/
def assemble {Col T : Type} (parse : Msg → Option Stmt) (table : Msg → ℕ → Option Stmt)
    (decode : List (ℕ × Col) → List T) (read : List T → Wit) (x : Msg) (opens : List (ℕ → List (ℕ × Col))) :
    Option Wit :=
  match parse x with
  | none => none
  | some s =>
    ((List.range s.tables).mapM fun j => (table x j).bind fun t =>
      (decode (opens.flatMap fun o => o j)).find? fun M => decide (RelS t (read [M]))).map read

/-- What the deployed verifier brings, read by Flock's claims: the verifier and whether it is the masked (`--zk`) one,
its parser and scope check, each table's own statement (`table x j`: the statement of table `j`'s units alone, one
table, as the verifier's parser reads it), each table's opened columns of its first committed codeword in a run
(`cols v j` for table `j`), the code (`enc`), each trace's codeword length (`len`) and decoding radius (`θ`), functions
of the trace because one `T` holds the traces of every statement's code, the deployed decoder with its step count
(`dec`), how the tables' traces, one per table in table order (the units' order), are read as one witness (`read`), and
the numbers the errors read: `ε_table`, the tables' errors summed; `N₀`, the tables' level-0 codeword lengths summed;
the weight `C` of `γ`; `E^rep`'s waits `m′` and its truncation `K`. Every table is decoded by the one decoder at the one
code. A single-table statement is its own table `0`, and reads table `0`'s columns and one trace (`read [M]`), its
`ε_table` and `N₀` that table's.

`dec` is a named, counted program: on opened columns it returns its list and the steps it took, counted along the same
computation, one step per field operation (arithmetic or a comparison of field elements), per position compared and
per column entry read, moving data free. `Flock.Efficient` reads that count; that it is the cost of running the
decoder is `trustedBase`'s.

`P` is the reference prover, the honest prover the deployment runs, on its tapes (`Tape`, a nonempty type):
completeness is its (`Complete`), so the claim is about the deployed prover and no other. `δc` is its completeness
error as the deployment states it, at each hash (`δc H x` at `H`): `Complete` is at that error and no other. -/
structure Deploy (Coin Col T : Type) where
  zk : Bool
  V : Verifier Coin
  parse : Msg → Option Stmt
  scopeCheck : Msg → Bool
  table : Msg → ℕ → Option Stmt
  cols : Seen → ℕ → List (ℕ × Col)
  enc : T → ℕ → Col
  len : T → ℕ
  θ : T → ℝ
  dec : List (ℕ × Col) → List T × ℕ
  read : List T → Wit
  εtable : Msg → ℝ
  N₀ : Msg → ℝ
  weight : Msg → ℝ
  waits : Msg → ℕ
  cap : Msg → ℕ
  Tape : Type
  [tapeNonempty : Nonempty Tape]
  P : Wit → Prover Tape
  δc : (Msg → Msg) → Msg → ℝ

instance Deploy.instNonemptyTape {Coin Col T : Type} (D : Deploy Coin Col T) : Nonempty D.Tape := D.tapeNonempty

/-- The deployed decoder's list, without its count. -/
def Deploy.decode {Coin Col T : Type} (D : Deploy Coin Col T) (os : List (ℕ × Col)) : List T := (D.dec os).1

/-- **A masked record**: a record with what its masked interaction (`--zk`) adds, which commits every coin at `Hello`
(`coin-tree/hm96-sha512/v2`) and opens them round by round. The reference prover's tape there is a nonce, which its
`Hello` sends, and the rest (`Nonce`, `Rest`, finite types), read as its tape `tapeOf (ν, r)`; `hello` is its `Hello` at
a nonce and `abort` its message once an opening fails; `chk x ν ρ j m` is its check of the verifier's opening `m` of coin
`j` against the commitment `ρ`, an oracle program whose `H` calls are the prover's, and `coinOf m` the coin `m` opens, as
the public-coin verifier `V` sends it. `S` is the simulator `ZK` is at. -/
structure ZkDeploy (Coin Col T : Type) extends Deploy Coin Col T where
  Nonce : Type
  Rest : Type
  [nonceFin : Fintype Nonce]
  [nonceNe : Nonempty Nonce]
  [restFin : Fintype Rest]
  [restNe : Nonempty Rest]
  tapeOf : Nonce × Rest → Tape
  hello : Msg → Nonce → Msg
  abort : Msg
  chk : Msg → Nonce → Msg → ℕ → Msg → OC Bool
  coinOf : Msg → Msg
  S : Simulator

attribute [instance] ZkDeploy.nonceFin ZkDeploy.nonceNe ZkDeploy.restFin ZkDeploy.restNe

/-- Whether every opening from coin `j` on passes `chk`, checked in order up to the first that fails. -/
def allPass (chk : ℕ → Msg → OC Bool) : ℕ → List Msg → OC Bool
  | _, [] => .ret true
  | j, m :: ms => (chk j m).bind fun b => if b then allPass chk (j + 1) ms else .ret false

/-- **The masked record's reference prover** in its masked interaction, at the witness `w` and the tape `(ν, r)`:
`Hello` at `ν` first; then, if every opening so far passes `chk` against the commitment `ρ` (the verifier's answer to
`Hello`), the record's reference prover `D.P w` at the history with each opening read as its coin (`coinOf`), at the tape
`tapeOf (ν, r)`; else `abort`. -/
def ZkDeploy.prover {Coin Col T : Type} (D : ZkDeploy Coin Col T) (w : Wit) : Prover (D.Nonce × D.Rest) :=
  ⟨fun x hist ω => match hist with
    | [] => .ret (D.hello x ω.1)
    | (_, ρ) :: ops => (allPass (D.chk x ω.1 ρ) 0 (ops.map Prod.snd)).bind fun ok =>
        if ok then (D.P w).next x (ops.map fun p => (p.1, D.coinOf p.2)) (D.tapeOf ω) else .ret D.abort⟩

section Claims

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- `E^rep`'s accepted reruns per wait, `k = ⌈4·N₀/e⌉`. -/
noncomputable def Deploy.k (D : Deploy Coin Col T) (x : Msg) : ℕ := ⌈4 * D.N₀ x / Real.exp 1⌉₊

/-- **Flock's extractor**, `E^rep` at a decoder: an accepted rerun's opening is its tables' columns (`D.cols v`), which
`assemble` decodes and checks table by table, each against the table's own statement (`D.table`). -/
noncomputable def E (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) : Extractor Coin Wit D.V :=
  ERep D.V D.waits D.k D.cap (fun v => [D.cols v]) (assemble D.parse D.table decode D.read) (check D.parse)

/-- **Flock's knowledge error**: `κ = 4·ε_table + 2^{−m′} + 2k/K + e^{−k/8}`. -/
noncomputable def κ (D : Deploy Coin Col T) (x : Msg) : ℝ :=
  4 * D.εtable x + 2 ^ (-(D.waits x : ℝ)) + 2 * D.k x / D.cap x + Real.exp (-(D.k x : ℝ) / 8)

/-- **Flock's `γ`**: `16·(√(2·k(x)·C) + k(x))²`, with `k(x)` the verifier's rounds. -/
noncomputable def γ (D : Deploy Coin Col T) (x : Msg) : ℝ :=
  16 * (Real.sqrt (2 * D.V.k x * D.weight x) + D.V.k x) ^ 2

/-- Flock's scope, on the statement's bytes. -/
def Scope (D : Deploy Coin Col T) (x : Msg) : Prop := ∃ s, D.parse x = some s ∧ InScope D.zk s

/-- **Flock: KS**, at `E^rep` and every decoder that meets the list-decoding spec, the one decoder of every table. -/
def KS [Zero Col] (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ decode, ListWithin D.len D.enc D.θ decode → Claims.KS H D.V (R D.parse) (E D decode) (κ D) (γ D)

/-- **Flock: Complete**, on the statements in scope, of the deployed record's reference prover (`D.P`), at the
deployed error (`D.δc H`). -/
def Complete (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  Claims.Complete H D.V D.P (R D.parse) (Scope D) (D.δc H)

/-- **Flock: the readable scope is the verifier's.** -/
def ScopeMatches (D : Deploy Coin Col T) : Prop := Claims.ScopeMatches (Scope D) D.scopeCheck

/-- **Flock: zero knowledge** of the masked record's reference prover (`D.prover`, the record's `D.P` in the masked
interaction) at the record's simulator (`D.S`), against every malicious verifier of the deployed interaction's
`D.V.k + 1` rounds: the extra round is the `Hello`, at which the masked verifier sends its commitment to every coin, and
then come `D.V`'s rounds, which open the coins. `D.V` models the coins in the clear; the commitment's hiding is the
live-coins assumption of `trustedBase`. The prover's tape is the nonce its `Hello` sends and the rest. -/
def ZK (H : Msg → Msg) (D : ZkDeploy Coin Col T) (εzk γ : Msg → ℝ) : Prop :=
  D.zk = true ∧ ZKmalR H (fun x => D.V.k x + 1) D.prover (R D.parse) D.S εzk γ

/-- **Flock: bits**, `λ = min(log₂(1/κ), 255.5 − ½ log₂ γ) − 1`, at every decoder that meets the spec, the one decoder
of every table. -/
def Bits [Zero Col] (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ decode, ListWithin D.len D.enc D.θ decode →
    Claims.Bits H D.V (R D.parse) (E D decode) fun x => lamOf (κ D x) (γ D x)

/-- **Flock: efficiency**: `E^rep`'s reruns at the deployed decoder, which meets the list-decoding spec: the one decoder
of every table. On every opened word its steps (`D.dec`'s count) are at most `p` at the word's encoded length
(`wordLen`), and so are its lists' lengths, so `assemble`, which checks each table's list once, makes at most that many
checks per table. -/
def Efficient {F : Type} [Fintype F] [DecidableEq F] [Zero (List F)] (H : Msg → Msg) (D : Deploy Coin (List F) T)
    (p : Polynomial ℕ) : Prop :=
  ERepCost H D.V D.waits D.k D.cap (fun v => [D.cols v]) (assemble D.parse D.table D.decode D.read) (check D.parse) ∧
    ListWithin D.len D.enc D.θ D.decode ∧ (∀ os, (D.dec os).2 ≤ p.eval (wordLen os)) ∧
    ∀ os, (D.decode os).length ≤ p.eval (wordLen os)

/-- **Flock: nontriviality**: recorded honest runs, each of a statement the scope check takes, that the verifier
accepts. It is established by a test against recorded sessions, not by a Lean proof, and none is expected:
`check`'s `lean-agreement` job (`tools/verity/check/ci.toml`) runs the compiled Lean verifier on recorded honest
sessions (`verity/ml/flock/verifier/agree.py`) and fails unless it accepts each. This states what that test checks. -/
def Nontrivial (H : Msg → Msg) (D : Deploy Coin Col T) (recs : List (Recorded D.V)) : Prop :=
  Claims.Nontrivial H D.V recs ∧ ∀ e ∈ recs, D.scopeCheck e.x = true

end Claims

end Flock

end Claims
