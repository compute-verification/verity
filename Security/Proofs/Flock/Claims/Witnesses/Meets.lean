import Proofs.Flock.Claims.DeployEfficient
import Definitions.Claims.Sampling

/-!
# What the claims ask of a record holds at the deployed ones

The lock's `claims` section names, for each claim, theorems `meets`, one per deployed record, that state the claim's
checkable antecedents there, so that the claim is not true at the deployment only because nothing meets them. The
deployed records are Flock's, `deployPlain` and `deployZk` (`DeployCode`). The sampling compiler's claims read a Flock
record, the sampling record's `flock`, and their antecedents are of that record (the recursion's are `RecursionMeets`):

* `Flock.decoder_meets_plain`, `_zk`: the record's own decoder meets the list-decoding spec at the record's code,
  length and radius (`decode_plain`, `listWithin_plain`). That is the antecedent of Flock's `KS` and `Bits` (every
  decoder that meets the spec) and a conjunct of its `Efficient`.
* `Sampling.decoder_meets_plain`, `_zk`: the same at a sampling record whose `flock` is the deployed record, the
  antecedent of the sampling `KS`, `KSCount` and `Bits` and a conjunct of its `Efficient`.
* `Flock.masked_meets_zk`: `deployZk` is the masked verifier, the first conjunct of Flock's `ZK`;
  `Sampling.masked_meets_zk`: so is the `flock` of a sampling record over it, the first conjunct of the sampling `ZK`.
* `Flock.scope_meets_plain`, `_zk`: the record's readable scope (`Flock.InScope` at the record's `zk`) holds a statement
  with a witness in its relation (`RelS`), the readable part of `Complete`'s antecedent (`Scope x`, `R x w`), what
  `ScopeMatches` equates with the scope check, and what each of `Nontrivial`'s recorded runs is of. That some bytes
  parse to it is beyond Lean: the verifier's parser reads the circuit file's header with `Lean.Json.parse`, a
  `partial` function.
* `Sampling.scope_meets`: the sampling scope's readable form (`Sampling.InScope`) holds a program, partition and
  design; it reads no field of the record, of which the tree has no concrete one. It is the readable part of the
  sampling `Complete`'s antecedent, of what its `ScopeMatches` equates, and of `Nontrivial`'s runs.
* `Sampling.universal_meets`: some netlist fits its size class, on inputs of the class's width, the antecedent of
  `Universal`, which reads no field of the record either.
-/

open FlockDeploy

local notation "𝔽" => FlockLevel3.GF128

namespace Claims

/-- **The plain record's decoder meets the list-decoding spec** at the record's code, length and radius. -/
theorem Flock.decoder_meets_plain (rel : Flock.Readable.Parsed → Flock.Readable.Relation) :
    Claims.Flock.ListWithin (deployPlain rel).len (deployPlain rel).enc (deployPlain rel).θ
      (deployPlain rel).decode := by
  rw [show (deployPlain rel).decode = Claims.GS.plainDecodeRD from decode_plain rel]
  exact listWithin_plain rel

/-- **The masked record's decoder meets the list-decoding spec** at the record's code, length and radius. -/
theorem Flock.decoder_meets_zk (rel : Flock.Readable.Parsed → Flock.Readable.Relation) :
    Claims.Flock.ListWithin (deployZk rel).len (deployZk rel).enc (deployZk rel).θ (deployZk rel).decode := by
  rw [show (deployZk rel).decode = Claims.GS.zkDecodeRD from decode_zk rel]
  exact listWithin_zk rel

/-- **The masked record is the masked verifier.** -/
theorem Flock.masked_meets_zk (rel : Flock.Readable.Parsed → Flock.Readable.Relation) : (deployZk rel).zk = true :=
  rfl

/-- **A statement of a record's readable scope**: `2^m` trace bits in blocks of `2^6`, the empty circuit, one table,
hm96 rows (`rows-v1`) under SHA-512 leaves. -/
def Flock.scopeStmt (m : ℕ) : Claims.Flock.Stmt :=
  ⟨m, 6, ⟨0, [], []⟩, fun _ => [], fun _ => [], [],
    ⟨true, false, true, "hm96-sha512/v1", 1, false, false, false, false, 1, false, 0, true, true, false, false⟩⟩

/-- **The plain record's readable scope holds a statement with a witness in its relation.** -/
theorem Flock.scope_meets_plain (rel : Flock.Readable.Parsed → Flock.Readable.Relation) :
    ∃ (s : Claims.Flock.Stmt) (w : Claims.Flock.Wit), Claims.Flock.InScope (deployPlain rel).zk s ∧
      Claims.Flock.RelS s w :=
  ⟨Flock.scopeStmt 22, ⟨[]⟩, by
    show Claims.Flock.InScope false _
    unfold Claims.Flock.InScope
    decide, fun _ _ => rfl⟩

/-- **The masked record's readable scope holds a statement with a witness in its relation**, at `2^25` trace bits. -/
theorem Flock.scope_meets_zk (rel : Flock.Readable.Parsed → Flock.Readable.Relation) :
    ∃ (s : Claims.Flock.Stmt) (w : Claims.Flock.Wit), Claims.Flock.InScope (deployZk rel).zk s ∧
      Claims.Flock.RelS s w :=
  ⟨Flock.scopeStmt 25, ⟨[]⟩, by
    show Claims.Flock.InScope true _
    unfold Claims.Flock.InScope
    decide, fun _ _ => rfl⟩

/-- **A statement of the sampling scope**: the one-gate program that returns its input, under `Q_word` v1, with the
design that checks every unit of its one class. -/
def Sampling.scopeStmt : Claims.Sampling.AuditStatement := ⟨[], ⟨[.input], [0]⟩, ⟨"Q_word", 1, []⟩, ⟨[.all]⟩, []⟩

/-- Its partition: one class of one stratum of one proof block, `{0}`. -/
def Sampling.scopePartition : Claims.Sampling.Partition := ⟨[[[[0]]]]⟩

/-- **The sampling scope holds a statement**: its readable form reads no field of the record. -/
theorem Sampling.scope_meets :
    ∃ (P : Claims.Sampling.Partition) (s : Claims.Sampling.AuditStatement), Claims.Sampling.InScope P s := by
  refine ⟨Sampling.scopePartition, Sampling.scopeStmt, ⟨?_, ?_⟩,
    ⟨by decide, by decide, by decide, by decide, by decide⟩, rfl, ?_⟩
  · intro g x hg i hi
    cases g with
    | zero =>
      simp only [Sampling.scopeStmt, List.getElem?_cons_zero, Option.some.injEq] at hg
      subst hg
      simp [Claims.Sampling.BGate.reads] at hi
    | succ n => simp [Sampling.scopeStmt] at hg
  · intro o ho
    simp [Sampling.scopeStmt] at ho ⊢
    exact ho
  · simp [Sampling.scopeStmt, Sampling.scopePartition, Claims.Sampling.OneStage.WellFormed]

/-- **A private circuit of a size class**: the class of one gate over two inputs and one output, and its netlist, the
AND of the two inputs, on inputs of its width. -/
theorem Sampling.universal_meets :
    ∃ (c : Claims.Sampling.SizeClass) (n : Claims.Sampling.Netlist) (x : List Bool), n.Fits c ∧ x.length = c.nIn :=
  ⟨⟨1, 2, 1⟩, ⟨2, [(1, 0, 1)], [2]⟩, [true, false], by unfold Claims.Sampling.Netlist.Fits; decide, rfl⟩

/-- **At a sampling record over the plain record**, its Flock record's decoder meets the list-decoding spec. -/
theorem Sampling.decoder_meets_plain (rel : Flock.Readable.Parsed → Flock.Readable.Relation)
    (D : Claims.Sampling.Deploy Coin (List 𝔽) Claims.Flock.PlainTrace) (hD : D.flock = deployPlain rel) :
    Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ D.flock.decode := by
  rw [hD]
  exact Claims.Flock.decoder_meets_plain rel

/-- **At a sampling record over the masked record**, its Flock record's decoder meets the list-decoding spec. -/
theorem Sampling.decoder_meets_zk (rel : Flock.Readable.Parsed → Flock.Readable.Relation)
    (D : Claims.Sampling.Deploy Coin (List 𝔽) Claims.Flock.ZkTrace) (hD : D.flock = deployZk rel) :
    Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ D.flock.decode := by
  rw [hD]
  exact Claims.Flock.decoder_meets_zk rel

/-- **At a sampling record over the masked record**, its Flock record is the masked verifier. -/
theorem Sampling.masked_meets_zk (rel : Flock.Readable.Parsed → Flock.Readable.Relation)
    (D : Claims.Sampling.Deploy Coin (List 𝔽) Claims.Flock.ZkTrace) (hD : D.flock = deployZk rel) :
    D.flock.zk = true := by
  rw [hD]
  rfl

end Claims
