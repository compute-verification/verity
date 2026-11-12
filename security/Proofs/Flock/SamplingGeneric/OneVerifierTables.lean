module

public import Proofs.Flock.SamplingGeneric.MaskedKSTable
public import Proofs.Flock.SamplingGeneric.Deployed
public import Proofs.Flock.Claims.DeployTables
import Mathlib.Tactic.Linarith

/-!
# One compiled verifier, read at sessions and at tables

`Arg.ofFlock` and `Arg.ofFlockTable` are two proof systems for proof blocks over one Flock verifier: the first extracts
a session whole, at the session's relation, and the second one table of it, at that table's relation. Completeness and
zero knowledge hold at the session's relation only: a statement whose table `0` has a witness and whose table `1` has
none is in table `0`'s relation, and no prover is accepted at it. So a compiled sampled verifier keeps its
completeness and its zero knowledge over `Arg.ofFlock` and takes its knowledge soundness over `Arg.ofFlockTable`.

* **One verifier** (`Compiled.atTables`, `Compiled.atTables_complete`, `Compiled.atTables_ks`,
  `Compiled.atTables_ks_complete`, and `Compiled.atTables_ks_complete_masked` at Flock's deployed masked record and
  SHA-512): a compiled verifier over `Arg.ofFlock`, read at tables, is a compiled verifier over
  `Arg.ofFlockTable` with the same verifier, parser, partition, commitment, sample, rounds and reference prover; each
  proof block's session is the same session with its statement named, and only what the table's extractor reads is new.
  Its completeness statement is the session one's, and its `KS` is a statement about the session one's verifier, scope
  and relation, at the table extractor and its errors.
* **A named session runs Flock's verifier** (`ofFlockTable_verifier_named`): where the naming reads back, the table
  proof system's verifier at a named statement is Flock's at the session's statement.
* **Its reruns** (`Compiled.es_ofFlockTable`, `Compiled.es_ofFlockTable_le`): where every proof block's shape is a
  statement's, `Compiled.es` is the largest of `1 + m′·k_j` over the tables the proof blocks' statements name, so at
  most `1 + m·k` where each has at most `m` waits and each named table's `k_j` is at most `k`.
* **At one table the table's relation is the session's** (`FlockDeploy.rTable_zero_of_tables_eq_one`): at a statement of
  one table, table `0`'s relation is the statement's. A verifier that holds every session to one table gains nothing
  from the table statement.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims
open Claims.Sampling (AuditStatement Opening KSLinOn Bext Rhon)

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- `maxBelow` reads its function below its bound only. -/
theorem maxBelow_congr {n : ℕ} {f g : ℕ → ℝ} (h : ∀ u < n, f u = g u) : maxBelow n f = maxBelow n g := by
  unfold maxBelow
  exact congrArg (List.foldr max 0) (List.map_congr_left fun u hu => h u (List.mem_range.1 hu))

section Reruns

variable {D : Claims.Flock.Deploy Coin Col T} {N : Msg → ℕ → ℕ} {ε : Msg → ℕ → ℝ} {names : Msg → ℕ × Msg}

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
/-- **The sessions' expected reruns at one table of a session**: over `Arg.ofFlockTable`, where each proof block's shape
is a statement's (`hsh`: block `u`'s is `y u`'s, as the contract's `runs` gives each sampled block), `Compiled.es` is
the largest, over the proof blocks, of `1 + m′·k_j` at the table and the Flock statement the block's statement names. -/
theorem Compiled.es_ofFlockTable (L : Compiled Coin (Arg.ofFlockTable D N ε names)) (x : Msg) (y : ℕ → Msg)
    (hsh : ∀ u < L.nPU x, (Arg.ofFlockTable D N ε names).shape (y u) = L.sh x u) :
    L.es x = maxBelow (L.nPU x) fun u =>
      1 + (D.waits (names (y u)).2 : ℝ) * (Claims.Flock.kTable N (names (y u)).1 (names (y u)).2 : ℝ) :=
  maxBelow_congr fun u hu =>
    (congrArg (fun s : (Arg.ofFlockTable D N ε names).Sh =>
      (1 : ℝ) + ((Arg.ofFlockTable D N ε names).waits s : ℝ) * ((Arg.ofFlockTable D N ε names).k s : ℝ))
      (hsh u hu)).symm

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
/-- **They are one table's**: where each proof block's statement has at most `m` waits and names a table whose `k_j` is
at most `k`, the sessions' expected reruns are at most `1 + m·k`. -/
theorem Compiled.es_ofFlockTable_le (L : Compiled Coin (Arg.ofFlockTable D N ε names)) (x : Msg) (y : ℕ → Msg)
    (hsh : ∀ u < L.nPU x, (Arg.ofFlockTable D N ε names).shape (y u) = L.sh x u) {m k : ℕ}
    (hm : ∀ u < L.nPU x, D.waits (names (y u)).2 ≤ m)
    (hk : ∀ u < L.nPU x, Claims.Flock.kTable N (names (y u)).1 (names (y u)).2 ≤ k) :
    L.es x ≤ 1 + (m : ℝ) * (k : ℝ) := by
  rw [Compiled.es_ofFlockTable L x y hsh]
  have hm0 : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg m
  have hk0 : (0 : ℝ) ≤ (k : ℝ) := Nat.cast_nonneg k
  have hmk : (0 : ℝ) ≤ (m : ℝ) * (k : ℝ) := mul_nonneg hm0 hk0
  refine maxBelow_le (by linarith) fun u hu => ?_
  have h1 : (D.waits (names (y u)).2 : ℝ) ≤ (m : ℝ) := Nat.cast_le.2 (hm u hu)
  have h2 : (Claims.Flock.kTable N (names (y u)).1 (names (y u)).2 : ℝ) ≤ (k : ℝ) := Nat.cast_le.2 (hk u hu)
  have h3 : (D.waits (names (y u)).2 : ℝ) * (Claims.Flock.kTable N (names (y u)).1 (names (y u)).2 : ℝ) ≤
      (m : ℝ) * (k : ℝ) := mul_le_mul h1 h2 (Nat.cast_nonneg _) hm0
  show (1 : ℝ) + (D.waits (names (y u)).2 : ℝ) * (Claims.Flock.kTable N (names (y u)).1 (names (y u)).2 : ℝ) ≤
    1 + (m : ℝ) * (k : ℝ)
  linarith

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
/-- **A named session runs Flock's verifier**: where the naming reads back (`names (enc j xS) = (j, xS)`), the table
proof system's verifier at the statement naming table `j` of `xS` has Flock's rounds and next message at `xS`. -/
theorem ofFlockTable_verifier_named (D : Claims.Flock.Deploy Coin Col T) (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ)
    (names : Msg → ℕ × Msg) (enc : ℕ → Msg → Msg) (hn : ∀ j xS, names (enc j xS) = (j, xS)) (j : ℕ) (xS : Msg) :
    (Arg.ofFlockTable D N ε names).V.k (enc j xS) = D.V.k xS ∧
      (Arg.ofFlockTable D N ε names).V.next (enc j xS) = D.V.next xS := by
  show D.V.k (names (enc j xS)).2 = D.V.k xS ∧ D.V.next (names (enc j xS)).2 = D.V.next xS
  rw [hn j xS]
  exact ⟨rfl, rfl⟩

end Reruns

section OneVerifier

variable {D : Claims.Flock.Deploy Coin Col T}

/-- **A compiled verifier over Flock, read at tables**: the compiled verifier over Flock at one table of a session
(`Arg.ofFlockTable`) with `L`'s verifier, parser, scope check, partition, commitment, second stage, sample, rounds,
session keys, interior roots, opening calls, wait sizes, reference prover and completeness error. Each proof block's
session is `L`'s, its statement named with the block's table (`enc (tab x u)`). What the table's extractor reads is
given: the proof blocks' shapes and the bounds on its errors (`sh`, `κb`, `γb`), and what a table's witness reveals
(`reveal`). It takes any `tab`: nothing here holds `tab x u` to one of the statement's tables, and at an index past
them the deployed masked record's error is at least `1` (`FlockDeploy.KS.one_le_κTable_past`), so the bound of a
verifier that names one says nothing. -/
noncomputable def Compiled.atTables (L : Compiled Coin (Arg.ofFlock D)) (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ)
    (names : Msg → ℕ × Msg) (enc : ℕ → Msg → Msg) (tab : Msg → ℕ → ℕ) (sh : Msg → ℕ → ℕ × ℕ × ℕ × ℕ)
    (κb γb : Msg → ℕ → ℝ)
    (reveal : AuditStatement → ℕ → Seen → Claims.Flock.Wit → List (List Msg × Opening Bool)) :
    Compiled Coin (Arg.ofFlockTable D N ε names) where
  V := L.V
  parse := L.parse
  scopeCheck := L.scopeCheck
  partition := L.partition
  commit := L.commit
  stage2 := L.stage2
  checked := L.checked
  sh := sh
  κb := κb
  γb := γb
  rounds := L.rounds
  session x u v := (L.session x u v).map fun p => (enc (tab x u) p.1, p.2)
  skey := L.skey
  nSess := L.nSess
  iroots := L.iroots
  reveal := reveal
  openCalls := L.openCalls
  nOut := L.nOut
  nIn := L.nIn
  Tape := L.Tape
  tapeNonempty := L.tapeNonempty
  P := L.P
  δc := L.δc

variable (L : Compiled Coin (Arg.ofFlock D)) (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ) (names : Msg → ℕ × Msg)
  (enc : ℕ → Msg → Msg) (tab : Msg → ℕ → ℕ) (sh : Msg → ℕ → ℕ × ℕ × ℕ × ℕ) (κb γb : Msg → ℕ → ℝ)
  (reveal : AuditStatement → ℕ → Seen → Claims.Flock.Wit → List (List Msg × Opening Bool))

omit [Nonempty Coin] [DecidableEq Col] in
/-- **Completeness is the session one's**: `Compiled.Complete` reads the verifier, the reference prover, the
commitment, the parser, the partition, the scope and the completeness error, which reading at tables keeps. -/
theorem Compiled.atTables_complete (H : Msg → Msg) :
    (L.atTables N ε names enc tab sh κb γb reveal).Complete H ↔ L.Complete H := Iff.rfl

omit [DecidableEq Col] in
/-- **`KS` at tables is about the session one's verifier**: `L`'s verifier, scope and relation, at the table
extractor, its errors and its finder. -/
theorem Compiled.atTables_ks (H : Msg → Msg) :
    (L.atTables N ε names enc tab sh κb γb reveal).KS H ↔
      ∀ δ : ℝ, 0 ≤ δ →
        KSLinOn H L.V L.Scope (Sampling.R H L.commit L.parse L.partition δ)
          (L.atTables N ε names enc tab sh κb γb reveal).E ((L.atTables N ε names enc tab sh κb γb reveal).κ δ)
          (L.atTables N ε names enc tab sh κb γb reveal).γ (fun _ => 1)
          (Bext H L.V (L.atTables N ε names enc tab sh κb γb reveal).E) := Iff.rfl

/-- **One verifier, knowledge sound at tables and complete at sessions**: a compiled verifier over Flock that is
complete, with a binding commitment, whose reading at tables meets the contract, is knowledge sound at the table
extractor from `Flock.KSTable` (`flockKSTable`), and complete: both of the one verifier. The contract at tables (`hS`)
is the hypothesis no compiled verifier is shown to meet here. -/
theorem Compiled.atTables_ks_complete [Zero Col] (H : Msg → Msg) (hKS : Claims.Flock.KSTable H D N ε)
    (hdec : Claims.Flock.ListWithin D.len D.enc D.θ D.decode) (hB : L.commit.Binds H)
    (hS : (L.atTables N ε names enc tab sh κb γb reveal).Sessions H) (hC : L.Complete H) :
    (L.atTables N ε names enc tab sh κb γb reveal).KS H ∧
      (L.atTables N ε names enc tab sh κb γb reveal).Complete H :=
  ⟨flockKSTable H D N ε names (L.atTables N ε names enc tab sh κb γb reveal) hKS hdec hB hS, hC⟩

end OneVerifier

section Masked

open Claims.Sampling (pathCommit hm96Leaf regPath RowPlaces)

variable (names : Msg → ℕ × Msg) (enc : ℕ → Msg → Msg) (tab : Msg → ℕ → ℕ) (sh : Msg → ℕ → ℕ × ℕ × ℕ × ℕ)
  (κb γb : Msg → ℕ → ℝ)
  (reveal : AuditStatement → ℕ → Seen → Claims.Flock.Wit → List (List Msg × Opening Bool))
  (L : Compiled FlockDeploy.Coin (Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation)))
  {own : Array _root_.Flock.Registered.Port} {at_ : List Msg → ℕ → Option (Msg × ℕ)}
  {row salt π : Opening Bool → Msg} {layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg}

/-- **At Flock's deployed masked record and SHA-512**: a complete compiled verifier over it, committing by hm96 leaves
of rows that place the value under the registered path check, whose reading at tables (at its tables' words' lengths
and each table's own error) meets the contract, is knowledge sound at the table extractor (`ksTable_masked`) and
complete. The contract at tables (`hS`) and completeness (`hC`) are its hypotheses. -/
theorem Compiled.atTables_ks_complete_masked
    (hcommit : L.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer) (hrow : RowPlaces row)
    (hS : (L.atTables (FlockDeploy.KS.wordLenJ true) (FlockDeploy.KS.tableErrJ true) names enc tab sh κb γb
      reveal).Sessions FlockDeploy.sha512)
    (hC : L.Complete FlockDeploy.sha512) :
    (L.atTables (FlockDeploy.KS.wordLenJ true) (FlockDeploy.KS.tableErrJ true) names enc tab sh κb γb
        reveal).KS FlockDeploy.sha512 ∧
      (L.atTables (FlockDeploy.KS.wordLenJ true) (FlockDeploy.KS.tableErrJ true) names enc tab sh κb γb
        reveal).Complete FlockDeploy.sha512 :=
  ⟨ksTable_masked names
      (L.atTables (FlockDeploy.KS.wordLenJ true) (FlockDeploy.KS.tableErrJ true) names enc tab sh κb γb reveal)
      hcommit hrow hS, hC⟩

end Masked

end Claims.SamplingGeneric

namespace FlockDeploy

open Flock

/-- **At one table the table's relation is the session's**: at a statement the claims' parser accepts with one table,
a witness is in table `0`'s relation exactly when it is in the statement's. -/
theorem rTable_zero_of_tables_eq_one {zk : Bool} {rel : Readable.Parsed → Readable.Relation}
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse zk rel x = some s) (h1 : s.tables = 1)
    (w : Claims.Flock.Wit) : Claims.Flock.RTable (table zk rel) 0 x w ↔ Claims.Flock.RelS s w := by
  unfold Claims.Flock.RTable
  rw [table_zero_of_tables_eq_one hs h1]
  exact ⟨fun ⟨_, ht, hw⟩ => by cases ht; exact hw, fun hw => ⟨s, rfl, hw⟩⟩

end FlockDeploy
