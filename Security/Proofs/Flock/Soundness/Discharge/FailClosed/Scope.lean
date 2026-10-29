import Proofs.Flock.Soundness.Discharge.Layout.Scope
import Proofs.Flock.Soundness.Discharge.ZkExec.Event
import Proofs.Flock.Soundness.Discharge.Hidden.Facts
import Proofs.Flock.Soundness.Discharge.FailClosed.TmplStmt
import Proofs.Flock.Soundness.ExecSetup
import Flock.ProvedScope

/-!
# What the verifier accepts: `Accepts` and its statement check

`flock-verify verify` refuses every session of a statement whose form the soundness proof does not cover:
`Main.verifyCmd` runs `Flock.ProvedScope.check` on the statement in every session's verdict, after every other check but
the draw-less session's refusal, and refuses the session when it fails. So what the verifier accepts is `Verifies`
(under `--zk`, `VerifiesZK`; with `--registered F`, `VerifiesR` and under `--zk` `VerifiesZKR`): the setup and verdict
`Accepts` models, and the check. Every condition on the statement's form that the headlines took as a hypothesis follows
from the check: an untyped statement (`Checks.typed`), retained rounds, `hm96-sha512/v1` leaves or, at hidden outputs
without `--zk`, plain SHA-512 leaves (`Checks.merkleLeaf`), at most one table without `--zk`, the circuit and public file as the headlines read them (`circ`, `pubOf`), no shared-row public file
unless the statement has hidden outputs and `hm96-sha512/v1` leaves and the verifier holds its own registered reads
(`Checks.tables`; without them, `Checks.tables_none`) and `Layout.scopeOk` (`Checks.scope`, by `scopeOk_eq`: the
executable's `ProvedScope.scopeOk` is `Layout.scopeOk`, its port bits a mask), or under hidden outputs `Layout.scopeOkI`
(`Checks.scopeI`), and wherever the wires feed the unit or the outputs, an order of them that computes
(`Checks.wiresAcyclic`). So a circuit the wires feed neither way is in `Layout.scopeOkH` (`Checks.scopeH`), and one they
feed, under hidden outputs, in the general form `Layout.scopeGenH` (`Checks.scopeGen`). `check_ok` gives exactly what the
check runs: the executable's `ProvedScope.scopeOkI` is `Layout.scopeOkI` (`scopeOkI_eq`). The
unconditional headlines (`FailClosed.Composed`, `FailClosed.ZkHidden`,
`FailClosed.Exec`) are over `Verifies`, with no hypothesis on the statement's form, and so are the registered ones
(`FailClosed.PlainRegistered` over `VerifiesR`, `FailClosed.ZkRegistered` over `VerifiesZKR`). Under `--plain-tables` the
check is `ProvedScope.checkP` (`ScopeP`, `VerifiesP`): it refuses hidden outputs with plain SHA-512 leaves and more than
`Tags.MAX_TABLES` tables, and is otherwise the check at one table (`ScopeP.checks`).

`Checks` is `verifyCmd`'s call `ProvedScope.check tags zk circuitB pubB tables count publicInputs.isSome
registered.isSome` argument for argument. `tags` is `Inputs.tags` without `--zk` and `zkTags` under it (`--statement`'s,
after `withCoins`, then `Zk.tags` at `maskLogOf circuitB` and `count`, on the coin tree of `tableNames count`). The
circuit, public file and tables are the bytes `verifyCmd` sets the session up from, and `count` is `--session-tables`.
The public-inputs flag is `false`: with `--public-inputs` the check refuses every session, and without it `buildSession`
passes `setupTables` no public input words, the setup `Accepts` models. The registered flag `reg` is whether the verifier
holds `--registered F`: `Scope` and `ScopeZ` are the check without it, `ScopeR` and `ScopeZR` the check with it, which
`VerifiesR` and `VerifiesZKR` pair with the registered check `AcceptsR` and `AcceptsZKR` model. A session's verdict is `verifyCmd`'s `setupOf` at the record's `recordDraw` (the check
requires `hm96Rows`): `setupTables`, under `--zk` `Zk.shapeOk` on every table, then `ProvedScope.drawLaw` on a drawn
session's draw, `verify` or `Zk.verify`, the check, and last `ProvedScope.drawLaw` on a session without a draw. So a session
`verifyCmd` accepts satisfies `Verifies` without `--registered` and `VerifiesR` with it (under `--zk`, `VerifiesZK` and
`VerifiesZKR`). `verifyCmd` refuses more, which only shrinks the
event: a draw by a law no soundness claim draws by (`ProvedScope.drawLaw`: a bernoulli or work draw, or a subset draw
without `--zk` or `--plain-tables` at public outputs, or under `--plain-tables` a stratified draw), a session without a draw (every claim's record hypothesis reads the record's draw), registered reads (`checkRegistered`:
a header naming registered values without `--registered`; with it, `Registered.check` needs a shared-row public file,
which the check passes only at hidden outputs and `hm96-sha512/v1` leaves, where `VerifiesR` and `VerifiesZKR` model the
registered check), a coin-tree
key an earlier session of the run used, and a session file it cannot read. Each drawn session of a run is set up at its own draw,
and the draw-less setup, one statement, once.
-/

namespace FlockSoundness.Discharge.FailClosed

open Flock Exec ZkExec

/-! ## The executable's `scopeOk` is `Layout.scopeOk` -/

/-- **The executable's `scopeOk` is `Layout.scopeOk`.** -/
theorem scopeOk_eq (c : Circuit) : ProvedScope.scopeOk c = Layout.scopeOk c := by
  simp only [ProvedScope.scopeOk, Layout.scopeOk, portReadsB_eq, ProvedScope.v1Ports, Layout.v1Ports,
    ProvedScope.v1Rows, Layout.v1Rows, ProvedScope.unpinned, Layout.unpinned]

/-- **The executable's `scopeOkH` is `Layout.scopeOkH`.** -/
theorem scopeOkH_eq (c : Circuit) : ProvedScope.scopeOkH c = Layout.scopeOkH c := by
  have e : ProvedScope.v2Row = Layout.v2Row := rfl
  have e' : ProvedScope.segRow = Layout.segRow := rfl
  simp only [ProvedScope.scopeOkH, Layout.scopeOkH, portReadsB_eq, ProvedScope.v1Ports, Layout.v1Ports,
    ProvedScope.v1Rows, Layout.v1Rows, ProvedScope.unpinned, Layout.unpinned,
    ProvedScope.bitRows, Layout.bitRows, ProvedScope.bitRowsU, Layout.bitRowsU, e, e']

/-- **The executable's `scopeOkI` is `Layout.scopeOkI`.** -/
theorem scopeOkI_eq (c : Circuit) : ProvedScope.scopeOkI c = Layout.scopeOkI c := by
  have e : ProvedScope.v2Row = Layout.v2Row := rfl
  have e' : ProvedScope.segRow = Layout.segRow := rfl
  simp only [ProvedScope.scopeOkI, Layout.scopeOkI, portReadsB_eq, ProvedScope.v1Ports, Layout.v1Ports,
    ProvedScope.v1Rows, Layout.v1Rows, ProvedScope.unpinned, Layout.unpinned,
    ProvedScope.bitRows, Layout.bitRows, ProvedScope.bitRowsU, Layout.bitRowsU, e, e']

/-! ## The check, and what it gives -/

/-- **The verifier's statement check passes** at tags `t` (the session's: under `--zk`, `zkTags`'s), with no public input
words of the verifier's own (the setup `Accepts` models takes none), and with its own registered reads (`--registered F`)
when `reg`. -/
def Checks (t : Tags) (zk : Bool) (I : Inputs) (reg : Bool) : Prop :=
  ProvedScope.check t zk I.circuitFile I.publicFile I.tables I.count false reg = .ok ()

/-- The circuit as the check parses it (`ProvedScope.circuitOf`'s: at hidden outputs the one the session commits, a
templated statement's carrying its template), `default` when it does not parse, and then the check fails. -/
def circC (t : Tags) (I : Inputs) : Circuit :=
  match ProvedScope.circuitOf t I.circuitFile I.tables with
  | .ok c => c
  | .error _ => default

/-- **The statement circuit** (ASK-112 option A): the circuit the check parses, a templated statement's unrolled to the
flat circuit its template unrolls to (`HmOut.unrolled`); a flat statement's is its own. -/
def circ (t : Tags) (I : Inputs) : Circuit := HmOut.unrolled (circC t I)

/-- The public file as the check loads it (`default` when it does not load, and then the check fails). -/
def pubOf (t : Tags) (I : Inputs) : Public :=
  match ProvedScope.publicOf t (circ t I) I.publicFile with
  | .ok p => p
  | .error _ => default

theorem check_ok {t : Tags} {zk : Bool} {cf pf : ByteArray} {tables : List (String × ByteArray)} {count : ℕ}
    {reg : Bool} (h : ProvedScope.check t zk cf pf tables count false reg = .ok ()) :
    (t.hm96Rows = true ∧ t.typed = false ∧ t.retainRounds = true ∧
      (t.merkleLeaf = "hm96-sha512/v1" ∨
        (zk = false ∧ t.hiddenOutputs = true ∧ t.merkleLeaf = "flock-leaf/sha512-unsalted")) ∧
      (zk = true ∨ count ≤ 1)) ∧
    ∃ c pub, ProvedScope.circuitOf t cf tables = .ok c ∧ ProvedScope.publicOf t c pf = .ok pub ∧
      (pub.tables = none ∨ ProvedScope.sharedRowsOk t zk reg = true) ∧
      ((c.tmpl = none ∧ (if t.hiddenOutputs then Layout.scopeOkI c else ProvedScope.scopeOk c) = true ∧
          (ProvedScope.wiresFeed c = true → ProvedScope.wiresAcyclic c = true)) ∨
        ∃ tu, c.tmpl = some tu ∧ ProvedScope.scopeOkT t.hiddenOutputs zk count c tu = true) := by
  unfold ProvedScope.check at h
  split at h
  · cases h
  rename_i hf
  have hall : ∀ r ∈ ProvedScope.statementRules t zk count false, r.1 = true := by
    unfold ProvedScope.firstFailure at hf
    rw [Option.map_eq_none_iff, List.find?_eq_none] at hf
    intro r hr
    simpa using hf r hr
  simp only [ProvedScope.statementRules, List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq,
    Bool.not_eq_true', Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq, Bool.not_false] at hall
  obtain ⟨h1, h2, h3, h4, h5, -⟩ := hall
  have h4' : t.merkleLeaf = "hm96-sha512/v1" ∨
      (zk = false ∧ t.hiddenOutputs = true ∧ t.merkleLeaf = "flock-leaf/sha512-unsalted") := by
    revert h4
    unfold ProvedScope.leafOk
    cases zk <;> cases t.hiddenOutputs <;> simp
  refine ⟨⟨h1, h2, h3, h4', (Bool.eq_false_or_eq_true zk).imp id fun hz => by simpa [hz] using h5⟩, ?_⟩
  split at h
  · cases h
  rename_i c hc
  split at h
  · cases h
  rename_i pub hp
  unfold ProvedScope.circuitRules at h
  split at h
  · cases h
  rename_i ht
  have htab : pub.tables = none ∨ ProvedScope.sharedRowsOk t zk reg = true := by
    cases hpt : pub.tables with
    | none => exact Or.inl rfl
    | some _ =>
      refine Or.inr ?_
      rw [hpt] at ht
      revert ht
      generalize ProvedScope.sharedRowsOk t zk reg = b
      cases b <;> simp
  refine ⟨c, pub, hc, hp, htab, ?_⟩
  cases hct : c.tmpl with
  | some tu =>
    simp only [hct] at h
    refine .inr ⟨tu, rfl, ?_⟩
    revert h
    generalize ProvedScope.scopeOkT t.hiddenOutputs zk count c tu = b
    cases b <;> simp
  | none =>
    simp only [hct] at h
    have hw : ¬ (ProvedScope.wiresFeed c && !ProvedScope.wiresAcyclic c) = true := by
      intro hx
      simp only [hx, ↓reduceIte] at h
      cases h
    refine .inl ⟨rfl, ?_, fun hf => ?_⟩
    · simp only [hw] at h
      have hs : (if t.hiddenOutputs then ProvedScope.scopeOkI c else ProvedScope.scopeOk c) = true := by
        generalize (if t.hiddenOutputs then ProvedScope.scopeOkI c else ProvedScope.scopeOk c) = b at h
        cases b
        · cases h
        · rfl
      cases hh : t.hiddenOutputs
      · simpa [hh] using hs
      · simp only [hh, ↓reduceIte] at hs ⊢
        rw [← scopeOkI_eq, hs]
    · cases ha : ProvedScope.wiresAcyclic c
      · simp [hf, ha] at hw
      · rfl

/-- At public outputs the circuit rules are the same with the verifier's own registered reads and without them:
`sharedRowsOk` and `scopeOkT` both need hidden outputs. -/
theorem circuitRules_reg {t : Tags} {zk : Bool} {count : ℕ} {c : Circuit} {pub : Public}
    (hh : t.hiddenOutputs = false) :
    ProvedScope.circuitRules t zk count true c pub = ProvedScope.circuitRules t zk count false c pub := by
  simp [ProvedScope.circuitRules, ProvedScope.sharedRowsOk, ProvedScope.scopeOkT, ProvedScope.scopeWhyT, hh]

theorem circC_of {t : Tags} {I : Inputs} {c : Circuit}
    (hc : ProvedScope.circuitOf t I.circuitFile I.tables = .ok c) : circC t I = c := by
  unfold circC
  rw [hc]

theorem circ_of {t : Tags} {I : Inputs} {c : Circuit}
    (hc : ProvedScope.circuitOf t I.circuitFile I.tables = .ok c) : circ t I = HmOut.unrolled c := by
  unfold circ
  rw [circC_of hc]

namespace Checks

variable {t : Tags} {zk : Bool} {I : Inputs} {reg : Bool} (h : Checks t zk I reg)
include h

theorem typed : t.typed = false := (check_ok h).1.2.1

theorem hm96Rows : t.hm96Rows = true := (check_ok h).1.1

theorem retainRounds : t.retainRounds = true := (check_ok h).1.2.2.1

/-- `hm96-sha512/v1` leaves, or plain SHA-512 leaves at hidden outputs without `--zk` (`ProvedScope.leafOk`). -/
theorem merkleLeaf : t.merkleLeaf = "hm96-sha512/v1" ∨
    (zk = false ∧ t.hiddenOutputs = true ∧ t.merkleLeaf = "flock-leaf/sha512-unsalted") :=
  (check_ok h).1.2.2.2.1

/-- **`hm96-sha512/v1` leaves** under `--zk` and at public outputs. -/
theorem merkleLeaf_hm96 (hz : zk = true ∨ t.hiddenOutputs = false) : t.merkleLeaf = "hm96-sha512/v1" :=
  h.merkleLeaf.resolve_right fun ⟨h1, h2, _⟩ => by rcases hz with hz | hz <;> simp_all

theorem circuit_ok : ProvedScope.circuitOf t I.circuitFile I.tables = .ok (circC t I) := by
  obtain ⟨c, -, hc, -⟩ := (check_ok h).2
  rw [circC_of hc, hc]

/-- **The circuit the check parses is flat, or templated in the form `ProvedScope.scopeOkT` passes.** -/
theorem form : ((circC t I).tmpl = none ∧
      (if t.hiddenOutputs then Layout.scopeOkI (circC t I) else ProvedScope.scopeOk (circC t I)) = true ∧
      (ProvedScope.wiresFeed (circC t I) = true → ProvedScope.wiresAcyclic (circC t I) = true)) ∨
    ∃ tu, (circC t I).tmpl = some tu ∧ ProvedScope.scopeOkT t.hiddenOutputs zk I.count (circC t I) tu = true := by
  obtain ⟨c, -, hc, -, -, hs⟩ := (check_ok h).2
  rw [circC_of hc]
  exact hs

/-- A flat statement's circuit is the one the check parses. -/
theorem circ_flat (hf : (circC t I).tmpl = none) : circ t I = circC t I := unrolled_flat hf

/-- **A templated statement the check passes**: hidden outputs, `--zk`, at most one table and a templated statement's tags; `HmOut.parseStmt` reads its statement circuit, which carries no
template, has a `Built` unit, is in `Layout.scopeOkI`, and whose wires feed neither the unit nor the outputs
(`unrolled_scope`). -/
theorem templated {tu : TmplUnit} (ht : (circC t I).tmpl = some tu) :
    t.hiddenOutputs = true ∧ zk = true ∧ I.count ≤ 1 ∧ t.isTmpl = true ∧
      HmOut.parseStmt t I.circuitFile I.tables = .ok (circ t I) ∧ (circ t I).tmpl = none ∧
      Built (circ t I).unit ∧ Layout.scopeOkI (circ t I) = true ∧ ProvedScope.wiresFeed (circ t I) = false := by
  rcases h.form with ⟨hf, -⟩ | ⟨tu', ht', hs⟩
  · rw [hf] at ht
    cases ht
  · rw [ht] at ht'
    cases ht'
    exact unrolled_scope h.typed h.circuit_ok ht hs

/-- **Outside the templated form the check parses a flat circuit**: at public outputs, without `--zk`, or at more than
one table. -/
theorem flat (hn : t.hiddenOutputs = false ∨ zk = false ∨ 1 < I.count) : (circC t I).tmpl = none := by
  cases ht : (circC t I).tmpl with
  | none => rfl
  | some tu =>
    exfalso
    obtain ⟨hh, hz, hc, -⟩ := h.templated ht
    rcases hn with e | e | e
    · rw [hh] at e; cases e
    · rw [hz] at e; cases e
    · omega

/-- **A statement whose tags are not a templated statement's is flat** (`Tags.isTmpl`): the check passes a templated
circuit only at a templated statement's tags. -/
theorem flat_of_isTmpl (hi : t.isTmpl = false) : (circC t I).tmpl = none := by
  cases ht : (circC t I).tmpl with
  | none => rfl
  | some tu =>
    obtain ⟨-, -, -, hi', -⟩ := h.templated ht
    rw [hi] at hi'
    cases hi'

/-- The statement circuit carries no template. -/
theorem circ_tmpl : (circ t I).tmpl = none := by
  cases ht : (circC t I).tmpl with
  | none => rw [h.circ_flat ht]; exact ht
  | some tu =>
    obtain ⟨-, -, -, -, -, hn, -⟩ := h.templated ht
    exact hn

theorem public_ok : ProvedScope.publicOf t (circ t I) I.publicFile = .ok (pubOf t I) := by
  obtain ⟨c, pub, hc, hp, -⟩ := (check_ok h).2
  unfold pubOf
  rw [circ_of hc, publicOf_unrolled, hp]

/-- No shared-row public file, or one where the check covers it: a statement with hidden outputs and `hm96-sha512/v1`
leaves whose verifier holds its own registered reads, with or without `--zk`. -/
theorem tables : (pubOf t I).tables = none ∨
    (t.hiddenOutputs = true ∧ reg = true ∧ t.merkleLeaf = "hm96-sha512/v1") := by
  obtain ⟨c, pub, hc, hp, ht, -⟩ := (check_ok h).2
  have hpub : pubOf t I = pub := by unfold pubOf; rw [circ_of hc, publicOf_unrolled, hp]
  rw [hpub]
  refine ht.imp id fun hs => ?_
  simp only [ProvedScope.sharedRowsOk, Bool.and_eq_true, beq_iff_eq] at hs
  exact ⟨hs.1.1, hs.1.2, hs.2⟩

/-- Without the verifier's own registered reads, no shared-row public file. -/
theorem tables_none (hr : reg = false) : (pubOf t I).tables = none :=
  h.tables.resolve_right fun ⟨_, h', _⟩ => by rw [hr] at h'; cases h'

theorem scope (hh : t.hiddenOutputs = false) : Layout.scopeOk (circ t I) = true := by
  have hf := h.flat (.inl hh)
  rcases h.form with ⟨-, hs, -⟩ | ⟨tu, ht, -⟩
  · simp only [hh, Bool.false_eq_true, ↓reduceIte] at hs
    rw [h.circ_flat hf, ← scopeOk_eq, hs]
  · rw [hf] at ht
    cases ht

/-- **`Layout.scopeOkI`** under hidden outputs. -/
theorem scopeI (hh : t.hiddenOutputs = true) : Layout.scopeOkI (circ t I) = true := by
  rcases h.form with ⟨hf, hs, -⟩ | ⟨tu, ht, -⟩
  · simp only [hh, ↓reduceIte] at hs
    rw [h.circ_flat hf, hs]
  · obtain ⟨-, -, -, -, -, -, -, hs, -⟩ := h.templated ht
    exact hs

/-- **The wires compute in some order** wherever they feed the unit or the outputs. -/
theorem wiresAcyclic (hf : ProvedScope.wiresFeed (circ t I) = true) : ProvedScope.wiresAcyclic (circ t I) = true := by
  rcases h.form with ⟨hfl, -, hw⟩ | ⟨tu, ht, -⟩
  · rw [h.circ_flat hfl] at hf ⊢
    exact hw hf
  · obtain ⟨-, -, -, -, -, -, -, -, hw⟩ := h.templated ht
    rw [hw] at hf
    cases hf

/-- **`Layout.scopeOkH`** where the wires feed neither the unit nor the outputs, with or without hidden outputs
(`scopeOk` gives it). -/
theorem scopeH (hf : ProvedScope.wiresFeed (circ t I) = false) : Layout.scopeOkH (circ t I) = true := by
  cases hh : t.hiddenOutputs
  · exact Layout.scopeOkH_of_scopeOk (h.scope hh)
  · exact Layout.scopeOkH_of_scopeOkI (h.scopeI hh) hf

/-- **`Layout.scopeGenH`**, the general form, under hidden outputs where the wires feed the unit or the outputs. -/
theorem scopeGen (hh : t.hiddenOutputs = true) (hf : ProvedScope.wiresFeed (circ t I) = true) :
    Layout.scopeGenH (circ t I) = true :=
  Layout.scopeGenH_of_scopeOkI (h.scopeI hh) (h.wiresAcyclic hf)

theorem parse (hh : t.hiddenOutputs = false) : HmRow.parse t I.circuitFile I.tables = .ok (circ t I) := by
  have := h.circuit_ok
  rw [h.circ_flat (h.flat (.inl hh))]
  simpa [ProvedScope.circuitOf, hh] using this

/-- **The statement circuit is the one `HmOut.parseStmt` reads**, at hidden outputs: a flat statement's `HmOut.parse`'s,
a templated statement's committed circuit unrolled. -/
theorem parseStmt (hh : t.hiddenOutputs = true) :
    HmOut.parseStmt t I.circuitFile I.tables = .ok (circ t I) := by
  have hc := h.circuit_ok
  rw [ProvedScope.circuitOf_hidden hh] at hc
  exact parseStmt_of_commit h.typed hc

/-- At hidden outputs, a flat statement's circuit is the one `HmOut.parse` reads. -/
theorem parseH (hh : t.hiddenOutputs = true) (hf : (circC t I).tmpl = none) :
    HmOut.parse t I.circuitFile I.tables = .ok (circ t I) := by
  rw [h.circ_flat hf]
  refine ProvedScope.parse_of_circuitOf hh ?_ h.circuit_ok
  cases hi : t.isTmpl
  · rfl
  · have hc := h.circuit_ok
    rw [ProvedScope.circuitOf_hidden hh, parseCommit_tmpl hi] at hc
    obtain ⟨tu, -, ht, -⟩ := unrolled_of_parseTmpl hc
    rw [hf] at ht
    cases ht

theorem load (hh : t.hiddenOutputs = false) : HmRow.loadPublic t (circ t I) I.publicFile = .ok (pubOf t I) := by
  have := h.public_ok
  simpa [ProvedScope.publicOf, hh] using this

theorem loadH (hh : t.hiddenOutputs = true) : HmOut.loadPublic t (circ t I) I.publicFile = .ok (pubOf t I) := by
  have := h.public_ok
  simpa [ProvedScope.publicOf, hh] using this

/-- At hidden outputs, the circuit the check parses is the one the session commits (`HmOut.parseCommit`). -/
theorem parseC (hh : t.hiddenOutputs = true) : HmOut.parseCommit t I.circuitFile I.tables = .ok (circC t I) := by
  have hc := h.circuit_ok
  rwa [ProvedScope.circuitOf_hidden hh] at hc

theorem loadHC (hh : t.hiddenOutputs = true) :
    HmOut.loadPublic t (circC t I) I.publicFile = .ok (pubOf t I) := by
  have := h.public_ok
  rw [show circ t I = HmOut.unrolled (circC t I) from rfl, publicOf_unrolled] at this
  simpa [ProvedScope.publicOf, hh] using this

/-- A flat statement's circuit parses as `HmRow.parse` reads it, with or without hidden outputs (`HmOut.parse` of an
untyped statement is `HmRow.parse`'s, then its rows' check). -/
theorem parseRow (hf : (circC t I).tmpl = none) : HmRow.parse t I.circuitFile I.tables = .ok (circ t I) := by
  cases hh : t.hiddenOutputs
  · exact h.parse hh
  · exact (Hidden.parse_untyped h.typed (h.parseH hh hf)).1

/-- The statement circuit's unit is `Built`: a flat statement's as `HmRow.parse` builds it, a templated statement's
`Tmpl.unroll`'s net (`unrolled_built`). -/
theorem built : Built (circ t I).unit := by
  cases hf : (circC t I).tmpl with
  | none =>
    have P := parse_facts (h.parseRow hf)
    obtain ⟨ls, hls⟩ := P.check
    exact unit_built P (check_facts P.typed hls)
  | some tu =>
    obtain ⟨-, -, -, -, -, -, hb, -⟩ := h.templated hf
    exact hb

/-- **The unit returns the outputs** under public outputs (`scopeOk`); under hidden outputs, wherever the wires feed
neither the unit nor the outputs (`Layout.outNet_of_wiresFeed`). -/
theorem outNet (hh : t.hiddenOutputs = false) : (circ t I).outNet = (circ t I).unitNet :=
  (Layout.scopeOk_spec h.built (h.scope hh)).2.1

end Checks

/-! ## What the verifier accepts -/

variable {n : ℕ} {Reg : Type}

/-- **`flock-verify verify`'s statement check** (no `--zk`), without the verifier's own registered reads. -/
def Scope (I : Inputs) : Prop := Checks I.tags false I false

/-- **`flock-verify verify --registered F`'s statement check** (no `--zk`): the check with the verifier's own registered
reads, which also passes a shared-row public file of a statement with hidden outputs and `hm96-sha512/v1` leaves. -/
def ScopeR (I : Inputs) : Prop := Checks I.tags false I true

/-- **What `flock-verify verify` accepts** (no `--zk`): the setup and verdict `Accepts` models, and the statement check. -/
def Verifies (I : Inputs) (t : Exec.Transcript) : Prop := Accepts I t ∧ Scope I

/-- `Verifies` at the live session's transcript. -/
def LiveVerifies (I : Inputs) (o : Reg × Finset (Fin n) × Exec.Transcript) : Prop := Verifies I o.2.2

/-- **`flock-verify verify --zk`'s statement check**, at the session's tags (`zkTags`), without the verifier's own
registered reads. -/
def ScopeZ (I : Inputs) : Prop := Checks (zkTags I) true I false

/-- **`flock-verify verify --zk --registered F`'s statement check**, at the session's tags (`zkTags`): the check with the
verifier's own registered reads, which also passes a shared-row public file of a statement with hidden outputs. -/
def ScopeZR (I : Inputs) : Prop := Checks (zkTags I) true I true

/-- **What `flock-verify verify --zk` accepts**: the setup and verdict `AcceptsZK` models, and the statement check. -/
def VerifiesZK (I : Inputs) (t : Exec.Transcript) : Prop := AcceptsZK I t ∧ ScopeZ I

/-- `VerifiesZK` at the live session's transcript. -/
def LiveVerifiesZK (I : Inputs) (o : Reg × Finset (Fin n) × Exec.Transcript) : Prop := VerifiesZK I o.2.2

/-- The session's circuit (no `--zk`). -/
abbrev circS (I : Inputs) : Circuit := circ I.tags I

/-- The session's public file (no `--zk`). -/
abbrev pubS (I : Inputs) : Public := pubOf I.tags I

/-- The session's circuit under `--zk`. -/
abbrev circZ (I : Inputs) : Circuit := circ (zkTags I) I

/-- The session's public file under `--zk`. -/
abbrev pubZ (I : Inputs) : Public := pubOf (zkTags I) I

theorem ScopeZ.typed {I : Inputs} (h : ScopeZ I) : I.tags.typed = false := by
  rw [← zkTags_typed]; exact Checks.typed h

theorem ScopeZ.retainRounds {I : Inputs} (h : ScopeZ I) : I.tags.retainRounds = true := by
  rw [← zkTags_retainRounds]; exact Checks.retainRounds h

theorem ScopeZ.merkleLeaf {I : Inputs} (h : ScopeZ I) : I.tags.merkleLeaf = "hm96-sha512/v1" := by
  rw [← zkTags_merkleLeaf]; exact Checks.merkleLeaf_hm96 h (Or.inl rfl)

theorem ScopeZ.parse {I : Inputs} (h : ScopeZ I) (hh : I.tags.hiddenOutputs = false) :
    HmRow.parse (zkTags I) I.circuitFile I.tables = .ok (circZ I) :=
  Checks.parse h (by rw [zkTags_hiddenOutputs]; exact hh)

/-- **The session's statement circuit under `--zk`**, at hidden outputs: the one `HmOut.parseStmt` reads, a templated
statement's committed circuit unrolled (ASK-112 option A). -/
theorem ScopeZ.parseH {I : Inputs} (h : ScopeZ I) (hh : I.tags.hiddenOutputs = true) :
    HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok (circZ I) :=
  Checks.parseStmt h (by rw [zkTags_hiddenOutputs]; exact hh)

/-- At public outputs the circuit parses as `HmRow.parse` reads it. -/
theorem ScopeZ.parseRow {I : Inputs} (h : ScopeZ I) (hh : I.tags.hiddenOutputs = false) :
    HmRow.parse (zkTags I) I.circuitFile I.tables = .ok (circZ I) :=
  Checks.parseRow h (Checks.flat h (.inl (by rw [zkTags_hiddenOutputs]; exact hh)))

theorem ScopeZ.load {I : Inputs} (h : ScopeZ I) (hh : I.tags.hiddenOutputs = false) :
    HmRow.loadPublic (zkTags I) (circZ I) I.publicFile = .ok (pubZ I) :=
  Checks.load h (by rw [zkTags_hiddenOutputs]; exact hh)

theorem ScopeZ.loadH {I : Inputs} (h : ScopeZ I) (hh : I.tags.hiddenOutputs = true) :
    HmOut.loadPublic (zkTags I) (circZ I) I.publicFile = .ok (pubZ I) :=
  Checks.loadH h (by rw [zkTags_hiddenOutputs]; exact hh)

theorem ScopeZ.built {I : Inputs} (h : ScopeZ I) : Built (circZ I).unit := Checks.built h

theorem ScopeZ.tables {I : Inputs} (h : ScopeZ I) : (pubZ I).tables = none := Checks.tables_none h rfl

theorem ScopeZ.scope {I : Inputs} (h : ScopeZ I) (hh : I.tags.hiddenOutputs = false) :
    Layout.scopeOk (circZ I) = true := Checks.scope h (by rw [zkTags_hiddenOutputs]; exact hh)

theorem ScopeZ.scopeH {I : Inputs} (h : ScopeZ I) (hf : ProvedScope.wiresFeed (circZ I) = false) :
    Layout.scopeOkH (circZ I) = true := Checks.scopeH h hf

theorem ScopeZ.scopeGen {I : Inputs} (h : ScopeZ I) (hh : I.tags.hiddenOutputs = true)
    (hf : ProvedScope.wiresFeed (circZ I) = true) : Layout.scopeGenH (circZ I) = true :=
  Checks.scopeGen h (by rw [zkTags_hiddenOutputs]; exact hh) hf

theorem ScopeZ.outNet {I : Inputs} (h : ScopeZ I) (hh : I.tags.hiddenOutputs = false) :
    (circZ I).outNet = (circZ I).unitNet := Checks.outNet h (by rw [zkTags_hiddenOutputs]; exact hh)

theorem ScopeZR.typed {I : Inputs} (h : ScopeZR I) : I.tags.typed = false := by
  rw [← zkTags_typed]; exact Checks.typed h

theorem ScopeZR.retainRounds {I : Inputs} (h : ScopeZR I) : I.tags.retainRounds = true := by
  rw [← zkTags_retainRounds]; exact Checks.retainRounds h

theorem ScopeZR.merkleLeaf {I : Inputs} (h : ScopeZR I) : I.tags.merkleLeaf = "hm96-sha512/v1" := by
  rw [← zkTags_merkleLeaf]; exact Checks.merkleLeaf_hm96 h (Or.inl rfl)

/-- **The session's statement circuit under `--zk --registered`**, at hidden outputs: the one `HmOut.parseStmt` reads,
a templated statement's committed circuit unrolled. -/
theorem ScopeZR.parseH {I : Inputs} (h : ScopeZR I) (hh : I.tags.hiddenOutputs = true) :
    HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok (circZ I) :=
  Checks.parseStmt h (by rw [zkTags_hiddenOutputs]; exact hh)

/-- **At tags that are not a templated statement's, `HmOut.parse` reads the session's circuit** under
`--zk --registered`, at hidden outputs (`Checks.flat_of_isTmpl`). -/
theorem ScopeZR.parseHF {I : Inputs} (h : ScopeZR I) (hh : I.tags.hiddenOutputs = true)
    (hi : I.tags.isTmpl = false) :
    HmOut.parse (zkTags I) I.circuitFile I.tables = .ok (circZ I) :=
  Checks.parseH h (by rw [zkTags_hiddenOutputs]; exact hh) (Checks.flat_of_isTmpl h (by rw [zkTags_eq]; exact hi))

theorem ScopeZR.loadH {I : Inputs} (h : ScopeZR I) (hh : I.tags.hiddenOutputs = true) :
    HmOut.loadPublic (zkTags I) (circZ I) I.publicFile = .ok (pubZ I) :=
  Checks.loadH h (by rw [zkTags_hiddenOutputs]; exact hh)

theorem ScopeZR.built {I : Inputs} (h : ScopeZR I) : Built (circZ I).unit := Checks.built h

theorem ScopeZR.scope {I : Inputs} (h : ScopeZR I) (hh : I.tags.hiddenOutputs = false) :
    Layout.scopeOk (circZ I) = true := Checks.scope h (by rw [zkTags_hiddenOutputs]; exact hh)

theorem ScopeZR.scopeH {I : Inputs} (h : ScopeZR I) (hf : ProvedScope.wiresFeed (circZ I) = false) :
    Layout.scopeOkH (circZ I) = true := Checks.scopeH h hf

theorem ScopeZR.scopeGen {I : Inputs} (h : ScopeZR I) (hh : I.tags.hiddenOutputs = true)
    (hf : ProvedScope.wiresFeed (circZ I) = true) : Layout.scopeGenH (circZ I) = true :=
  Checks.scopeGen h (by rw [zkTags_hiddenOutputs]; exact hh) hf

/-- **The check with the verifier's registered reads is the check without them at public outputs**: `sharedRowsOk`
needs hidden outputs. -/
theorem ScopeZR.scopeZ {I : Inputs} (h : ScopeZR I) (hh : I.tags.hiddenOutputs = false) : ScopeZ I := by
  unfold ScopeZ
  unfold ScopeZR Checks ProvedScope.check at h
  unfold Checks ProvedScope.check
  simp only [circuitRules_reg (by rw [zkTags_hiddenOutputs]; exact hh)] at h
  exact h

theorem Scope.typed {I : Inputs} (h : Scope I) : I.tags.typed = false := Checks.typed h

theorem Scope.retainRounds {I : Inputs} (h : Scope I) : I.tags.retainRounds = true := Checks.retainRounds h

/-- At public outputs, `hm96-sha512/v1` leaves. -/
theorem Scope.merkleLeaf {I : Inputs} (h : Scope I) (hh : I.tags.hiddenOutputs = false) :
    I.tags.merkleLeaf = "hm96-sha512/v1" :=
  Checks.merkleLeaf_hm96 h (Or.inr hh)

/-- The leaves the non-ZK headlines cover: `hm96-sha512/v1`, or plain SHA-512 leaves (`flock-leaf/sha512-unsalted`). -/
theorem Scope.leaf {I : Inputs} (h : Scope I) :
    I.tags.merkleLeaf = "hm96-sha512/v1" ∨ I.tags.merkleLeaf = "flock-leaf/sha512-unsalted" :=
  (Checks.merkleLeaf h).imp id fun h' => h'.2.2

theorem Scope.parse {I : Inputs} (h : Scope I) (hh : I.tags.hiddenOutputs = false) :
    HmRow.parse I.tags I.circuitFile I.tables = .ok (circS I) := Checks.parse h hh

theorem Scope.parseH {I : Inputs} (h : Scope I) (hh : I.tags.hiddenOutputs = true) :
    HmOut.parse I.tags I.circuitFile I.tables = .ok (circS I) :=
  Checks.parseH h hh (Checks.flat h (.inr (.inl rfl)))

theorem Scope.load {I : Inputs} (h : Scope I) (hh : I.tags.hiddenOutputs = false) :
    HmRow.loadPublic I.tags (circS I) I.publicFile = .ok (pubS I) := Checks.load h hh

theorem Scope.loadH {I : Inputs} (h : Scope I) (hh : I.tags.hiddenOutputs = true) :
    HmOut.loadPublic I.tags (circS I) I.publicFile = .ok (pubS I) := Checks.loadH h hh

theorem Scope.built {I : Inputs} (h : Scope I) : Built (circS I).unit := Checks.built h

theorem Scope.tables {I : Inputs} (h : Scope I) : (pubS I).tables = none := Checks.tables_none h rfl

theorem Scope.scope {I : Inputs} (h : Scope I) (hh : I.tags.hiddenOutputs = false) : Layout.scopeOk (circS I) = true :=
  Checks.scope h hh

theorem Scope.scopeH {I : Inputs} (h : Scope I) (hf : ProvedScope.wiresFeed (circS I) = false) :
    Layout.scopeOkH (circS I) = true := Checks.scopeH h hf

theorem Scope.scopeGen {I : Inputs} (h : Scope I) (hh : I.tags.hiddenOutputs = true)
    (hf : ProvedScope.wiresFeed (circS I) = true) : Layout.scopeGenH (circS I) = true := Checks.scopeGen h hh hf

theorem Scope.outNet {I : Inputs} (h : Scope I) (hh : I.tags.hiddenOutputs = false) :
    (circS I).outNet = (circS I).unitNet := Checks.outNet h hh

theorem Scope.count {I : Inputs} (h : Scope I) : I.count ≤ 1 := by
  rcases (check_ok h).1.2.2.2.2 with h' | h'
  · cases h'
  · exact h'

/-- **`flock-verify verify --plain-tables`'s statement check** (no `--zk`, without the verifier's own registered reads,
which `verifyCmd` passes as `false` under the flag, so a shared-row public file is refused): `ProvedScope.checkP`, which
refuses hidden outputs with leaves other than `hm96-sha512/v1` (`ScopeP.merkleLeaf`) and more than `Tags.MAX_TABLES`
tables, is otherwise the check at one table, and holds the verifier's own program (`--program`) to the one the circuit's
META binds (`ScopeP.program`). -/
def ScopeP (I : Inputs) : Prop :=
  ProvedScope.checkP I.tags I.circuitFile I.publicFile I.tables I.count I.program false false = .ok ()

/-- **What `flock-verify verify --plain-tables` accepts**: the setup and verdict `Accepts` models, and its statement
check. -/
def VerifiesP (I : Inputs) (t : Exec.Transcript) : Prop := Accepts I t ∧ ScopeP I

/-- **The statement's circuit is the verifier's program's**: the verifier holds its own program's bytes `b` (`--program`),
the circuit's META binds a program (`ProvedScope.metaProgram`), and `b`'s digest (`ProvedScope.programDigest`, the
`program_sha512` the verdict names) is it. The circuit's gates are not compiled from `b` here: the binding is META's. -/
def ProgramOfCircuit (I : Inputs) : Prop :=
  ∃ b, I.program = some b ∧ (ProvedScope.metaProgram (circS I)).isSome = true ∧
    ProvedScope.programDigest b = ProvedScope.metaProgram (circS I)

namespace ScopeP

variable {I : Inputs} (h : ScopeP I)
include h

theorem why : ProvedScope.plainTablesWhy I.tags I.count = none := (ProvedScope.checkP_ok h).1

/-- **The check at one table**: `checkP` passes only where `check` passes the statement at one table. The circuit and
public file it reads are the session's (`circ` and `pubOf` read neither the count nor anything else of `I`). -/
theorem checks : Checks I.tags false { I with count := 1 } false := (ProvedScope.checkP_ok h).2.1

theorem count : I.count ≤ Tags.MAX_TABLES := by
  have hw := h.why
  unfold ProvedScope.plainTablesWhy at hw
  split at hw
  · cases hw
  · split at hw
    · assumption
    · cases hw

theorem typed : I.tags.typed = false := Checks.typed h.checks

theorem retainRounds : I.tags.retainRounds = true := Checks.retainRounds h.checks

/-- **`hm96-sha512/v1` leaves**, at public outputs by the check at one table, and at hidden outputs by `plainTablesWhy`,
which refuses the plain SHA-512 leaves the check at one table also passes there. -/
theorem merkleLeaf : I.tags.merkleLeaf = "hm96-sha512/v1" := by
  cases hh : I.tags.hiddenOutputs
  · exact Checks.merkleLeaf_hm96 (I := { I with count := 1 }) h.checks (Or.inr hh)
  · have hw := h.why
    unfold ProvedScope.plainTablesWhy at hw
    by_contra hl
    simp [hh, hl] at hw

theorem parse (hh : I.tags.hiddenOutputs = false) : HmRow.parse I.tags I.circuitFile I.tables = .ok (circS I) :=
  Checks.parse (I := { I with count := 1 }) h.checks hh

theorem parseH (hh : I.tags.hiddenOutputs = true) : HmOut.parse I.tags I.circuitFile I.tables = .ok (circS I) :=
  Checks.parseH (I := { I with count := 1 }) h.checks hh (Checks.flat h.checks (.inr (.inl rfl)))

theorem load (hh : I.tags.hiddenOutputs = false) : HmRow.loadPublic I.tags (circS I) I.publicFile = .ok (pubS I) :=
  Checks.load (I := { I with count := 1 }) h.checks hh

theorem loadH (hh : I.tags.hiddenOutputs = true) : HmOut.loadPublic I.tags (circS I) I.publicFile = .ok (pubS I) :=
  Checks.loadH (I := { I with count := 1 }) h.checks hh

theorem built : Built (circS I).unit := Checks.built (I := { I with count := 1 }) h.checks

theorem tables : (pubS I).tables = none := Checks.tables_none (I := { I with count := 1 }) h.checks rfl

theorem scope (hh : I.tags.hiddenOutputs = false) : Layout.scopeOk (circS I) = true :=
  Checks.scope (I := { I with count := 1 }) h.checks hh

theorem scopeH (hf : ProvedScope.wiresFeed (circS I) = false) : Layout.scopeOkH (circS I) = true :=
  Checks.scopeH (I := { I with count := 1 }) h.checks hf

theorem scopeGen (hh : I.tags.hiddenOutputs = true) (hf : ProvedScope.wiresFeed (circS I) = true) :
    Layout.scopeGenH (circS I) = true :=
  Checks.scopeGen (I := { I with count := 1 }) h.checks hh hf

/-- **The verifier's own program is the circuit's** (`ProgramOfCircuit`). -/
theorem program : ProgramOfCircuit I := by
  obtain ⟨-, -, hp⟩ := ProvedScope.checkP_ok h
  obtain ⟨c, hc, hw⟩ := ProvedScope.programP_ok hp
  have hpc : ProvedScope.circuitOf I.tags I.circuitFile I.tables = .ok (circS I) := by
    show ProvedScope.circuitOf I.tags ({ I with count := 1 } : Inputs).circuitFile ({ I with count := 1 } : Inputs).tables
      = .ok (circ I.tags { I with count := 1 })
    rw [Checks.circ_flat (I := { I with count := 1 }) h.checks (Checks.flat h.checks (.inr (.inl rfl)))]
    exact Checks.circuit_ok (I := { I with count := 1 }) h.checks
  rw [hpc] at hc
  injection hc with hc
  subst hc
  exact ProvedScope.programWhyP_none hw

/-- The tags the plain claim at `J` tables of public outputs reads: public outputs, retained rounds, `hm96-sha512/v1`
leaves. -/
theorem tagsOk (hh : I.tags.hiddenOutputs = false) : I.tags.hiddenOutputs = false ∧ I.tags.retainRounds = true ∧
    I.tags.merkleLeaf = "hm96-sha512/v1" :=
  ⟨hh, h.retainRounds, h.merkleLeaf⟩

end ScopeP

theorem ScopeR.typed {I : Inputs} (h : ScopeR I) : I.tags.typed = false := Checks.typed h

theorem ScopeR.retainRounds {I : Inputs} (h : ScopeR I) : I.tags.retainRounds = true := Checks.retainRounds h

theorem ScopeR.leaf {I : Inputs} (h : ScopeR I) :
    I.tags.merkleLeaf = "hm96-sha512/v1" ∨ I.tags.merkleLeaf = "flock-leaf/sha512-unsalted" :=
  (Checks.merkleLeaf h).imp id fun h' => h'.2.2

theorem ScopeR.parseH {I : Inputs} (h : ScopeR I) (hh : I.tags.hiddenOutputs = true) :
    HmOut.parse I.tags I.circuitFile I.tables = .ok (circS I) :=
  Checks.parseH h hh (Checks.flat h (.inr (.inl rfl)))

theorem ScopeR.loadH {I : Inputs} (h : ScopeR I) (hh : I.tags.hiddenOutputs = true) :
    HmOut.loadPublic I.tags (circS I) I.publicFile = .ok (pubS I) := Checks.loadH h hh

theorem ScopeR.built {I : Inputs} (h : ScopeR I) : Built (circS I).unit := Checks.built h

theorem ScopeR.scopeH {I : Inputs} (h : ScopeR I) (hf : ProvedScope.wiresFeed (circS I) = false) :
    Layout.scopeOkH (circS I) = true := Checks.scopeH h hf

theorem ScopeR.scopeGen {I : Inputs} (h : ScopeR I) (hh : I.tags.hiddenOutputs = true)
    (hf : ProvedScope.wiresFeed (circS I) = true) : Layout.scopeGenH (circS I) = true := Checks.scopeGen h hh hf

theorem ScopeR.count {I : Inputs} (h : ScopeR I) : I.count ≤ 1 := by
  rcases (check_ok h).1.2.2.2.2 with h' | h'
  · cases h'
  · exact h'

/-- **The check with the verifier's registered reads is the check without them at public outputs**: `sharedRowsOk`
needs hidden outputs. -/
theorem ScopeR.scope {I : Inputs} (h : ScopeR I) (hh : I.tags.hiddenOutputs = false) : Scope I := by
  unfold Scope
  unfold ScopeR Checks ProvedScope.check at h
  unfold Checks ProvedScope.check
  simp only [circuitRules_reg hh] at h
  exact h

/-! ## A bound under the statement check -/

/-- **A bound at every statement the check passes, lifted to the event that implies the check.** When the event `V`
implies `A` and the statement fact `Q`, and `A ∧ E q` has probability at most `B q` for every proof `q` of `Q`, then
`V ∧ ∀ q, E q` has probability at most `⨆ q, B q`: at a statement without `Q`, `V` is empty and the bound is `0`. -/
theorem prob_scope {α : Type} {V A : α → Prop} {Q : Prop} (hA : ∀ o, V o → A o) (hQ : ∀ o, V o → Q)
    (E : Q → α → Prop) (B : Q → ENNReal) (g : FlockSoundness.Game α) (s : Game.Strategy g)
    (h : ∀ q, Game.prob (fun o => A o ∧ E q o) g s ≤ B q) :
    Game.prob (fun o => V o ∧ ∀ q, E q o) g s ≤ ⨆ q, B q := by
  by_cases hq : Q
  · calc Game.prob (fun o => V o ∧ ∀ q, E q o) g s ≤ Game.prob (fun o => A o ∧ E hq o) g s :=
          Game.prob_mono (fun o ho => ⟨hA o ho.1, ho.2 hq⟩) g s
      _ ≤ B hq := h hq
      _ ≤ ⨆ q, B q := le_iSup B hq
  · calc Game.prob (fun o => V o ∧ ∀ q, E q o) g s ≤ Game.prob (fun _ => False) g s :=
          Game.prob_mono (fun o ho => hq (hQ o ho.1)) g s
      _ = 0 := Game.prob_false g s
      _ ≤ ⨆ q, B q := zero_le

end FlockSoundness.Discharge.FailClosed
