# SPOT: design

Which form of SPOT each configuration's verifier runs, the verdicts of record and the guards beside them, and the
docket with its registered values. `README.md` says what SPOT is, what it is proved to do, and what is not yet in Lean.

## The configurations

SPOT (`verity/sampled-proofs/v1`) is a protocol of both configurations: remote (`verity/remote/v1`,
`verity/protocols/compose/verifier/remote/`) composes it with Pearl+'s run (`verity/pouw/pearl-plus/v1`; H1's development executable
`verity/remote/v0`, with PoUW's period `verity/pouw/period/v1`) and the inference-only policy, and onsite
(`verity/protocols/compose/verifier/onsite/`) with the other protocols. Each configuration's verifier runs
`VStar.Recursive.Sampled`'s Lean check: one-stage v2 (`verity/sampled-proofs/one-stage/v2`, the audit on the proofs).
One-stage v1, whose verdicts are the evidence's, is refused by name (`V1_REFUSED`: a one-stage form whose params or
evidence name no `proofs`), and so is the recursive audit (`verity/sampled-proofs/recursive/v1`, `RECURSIVE_REFUSED` at
its params), which SPOT's soundness doesn't cover. The composite accepts only when every subprotocol accepts.

## Verdicts of record and guards

### Verdicts of record

No function here returns a verdict of its own. Each verdict SPOT acts on is the Lean's: `one_stage.verifier` asks
`spot-verifier` and computes the Python twin beside it, and a disagreement, or no answer from the Lean, is a
refusal (`DISAGREE`). A verdict whose Lean no interface here reaches yet raises
`NotImplementedError("proofs/spot-two-stage-sound")` (`verifier.not_written`), and its twin answers the difftest only.

| Verdict | Entry | Lean of record | Twin |
| --- | --- | --- | --- |
| R1-R8 on a docket | `verifier.check` | `registration` (`OneStage`) | `registration.check` |
| The audit record | `verifier.audit_record` | `Record.record` | `audit.audit_record` |
| A docket's whole run | `verifier.run` | `Run.run` | the stream's own outcome |
| A two-stage outcome | `Stream.outcome` (`_audit_two`) | `Run.run`'s symbol; it refuses every two-stage run while `SecondStage.FORM` is off | the two records' join |
| Round 2's statements | `second_stage.check` | `SecondStage.checkAny`, which only `Run.run` reaches | `check_twin` |
| A two-stage outcome's completeness | `second_stage.complete` | none yet | `complete_twin` |

A two-stage run whose stages read no coin bytes gives `Run.run` nothing to read, so `_audit_two` raises rather than
join its records.

### Guards

Each takes a verdict of record or files Lean then checks, and can only refuse: none returns an accept a Lean verdict did
not give.

| Guard | Why it can only refuse |
| --- | --- |
| `parties`' `_held` | returns a reject answer unchanged, and turns an accept into a reject when its commitment or interior is not the docket sent; otherwise returns the answer it was given |
| `one_stage.proofs.ks_form_why` | a reason or None for a circuit outside the form `verify --zk` covers (`Flock.KsScope.filesWhy`); a reason refuses a session, None leaves the binary's verdict |
| `one_stage.proofs.template_why` | a reason or None for a circuit the template refuses, read only before `members.files_why`'s other refusals; None decides nothing |
| `one_stage.proofs.population_why` | a reason or None for a public file whose proof blocks are not the docket's population in order (`Spot.FilePart.why`); None leaves `verify`'s verdict |

## The docket

### The docket

A docket fixes everything a sampled proof block's statement can depend on before the verifier samples. Its record,
`verity/registration/v1` (`one_stage.docket.record`), names the program, query and partition digests, the
population, the sampling design expanded over the verifier's program and query, the commitment scheme, each port's root
with its leaf count and domain, the leaf layer, the anchors, the run (`window`: its provenance, `served` or `stand-in`, the
stream and prover it is bound to, and its seed salt) and `previous`, the digest of this prover's last record. A record
that registers values is `verity/registration/v2`: the same keys and `registered` (below), so v1's bytes are unchanged.
Its digest is `SHA-512(format ‖ 0 ‖ canon(record))`, canon being canonical JSON (RFC 8785,
`verity.circuits.codec.canonical_json`).

The verifier holds the record to its own copies with R1 to R8 (`check`). The Lean verifier runs the same check
(`Spot.Docket.check`), `Properties.Spot.R1` to `R8` state what each accepts, and
`tests/verity/protocols/proofs/spot/verifier/lean/test_difftest.py` holds the Python to the Lean.

It then writes a receipt (`receipt`, `verity/registration-receipt/v1`), and only then samples. It also holds its own public
file to the record (`matches_public`), which `FailClosed.flock_verify_public_ties` ties to the proved rows. The
challenger's `service.ExecutionPhaseCommitmentRecord` is a record it admitted, with its receipt and its position in the
stream. The format, each check's rule and codes and the window domains (`window_domain`) are in
`one_stage.docket`'s docstring.

Pinned by `tests/verity/protocols/proofs/spot/verifier/test_one_stage.py`, and the window domain also by `tests/adapters/vllm/tests/commit/test_serving_rows.py`.

### Registered values

A program's weights, constants and tables are values: committed once, under hiding leaves, and read only inside a proof
(`one_stage.registered`, the Lean `Spot.Registered`, and Flock's `Flock.Registered`).

- **The commitment.** Row r of a registered value is `words` leaves of `word_bits` bits (16 or 32). It is committed as
  the circuit statement commits that port's row, in the port's row format (`hm96-sha512/row/v1`, or the bit rows
  `hm96-sha512/row/v2` and `hm96-sha512/row-seg/v1`), with a salt sampled once. The root is the frame-v3-sha512 root over
  the rows under the value's registered-value domain (`registered.domain`, on the SHA-512 identity frame of
  `verity/one-stage/registered-domain/v1`). The prover keeps the rows and salts
  (`Registered.commit`). The record's `registered` entry is `{"value", "root", "leaves", "words", "domain"}`, with
  `"word_bits"` when it is not 16 (`Registered.entry`).
- **R7** holds the entries to the verifier's own (`registered_entries`, from its own program). It holds each value to a
  root it already has, when it has one: its pinned root (`pinned`: the weights' root recorded before the salt), the root
  an earlier record of the log registered under the same domain (`held_roots`), or, for an output or reused value, the
  earlier record's port or entry (`outputs`). `Stream.register` and `register_interiors` take `outputs`, keep the record
  each one names in the log R7 reads, and pass them to the run of record: this is how a record reads the state an
  earlier one left, such as a request carried from one run's serving into the next.
- **Per-run values.** Given the run's open root R, from the challenger's signed open of it, every
  value's domain binds R, except the ones the verifier holds across runs (`held`, and its pinned, supplied and output
  values). So a run's values are held to no earlier run's roots. A verifier that names no R refuses its own inputs
  when a value it doesn't hold across runs would be held to an earlier run's root (`unwindowed`).
- **Supplied values.** These are values the verifier fills itself, from the record's roots, positions after the log's,
  its run's seed salt (bound to its stream and prover) and its own bytes. They are committed under public salts, and R7
  recomputes their root (`R7-supplied`). PoUW's seed inputs are such values
  (`tests/verity/protocols/resource_accounting/compute/pouw/verifier/test_pouw_supplied_inputs.py`).
- **Certified values.** R7 holds a registered value no parameter reads to the verifier's `certified` entry of it.
- **Reads.** Every port of a statement is either a registered port of the `--registered` file
  (`Registered.checkComplete`) or bound to the window's committed values: the file's `bound` names its rows' root and
  domain as the docket holds them, and the public header's must be the same (`bind_ports`; `Flock.RegBound`). No
  port that reads a registered input of the verifier's own program is ever bound, and nor is any port the window's
  registered circuit record declares registered for the statement's template (its registered reads, table reads and
  openings' verdicts, outputs included: `RegBound.declaredOf`), which must be a registered port of the file. A file
  that binds ports is verified with that record (`verity-verify-proof --interface`), and refused without it.
- **Served trees.** A served path registers each tree a proof block reads as a registered value. `served.register_served`
  returns a `ServedDocket`: the record's entries, R7's copies and the `--registered` file. That docket is the
  premise `RowsSession.ServedRegistered` of `Flock.Guarantees.ServedRowsLinked`.
- **The audit record** holds the roots the verifier opened the reads against (`audit.session_registered`, Lean
  `Record.sessionRegistered`) to the docket's own (`Record.ownRegistered`), taking a read named by one of its
  own ports at that port's root (`audit.less_own_reads`, `Record.lessOwnReads`) and refusing it at any other. It holds
  a read of a window value its caller names (the parameters' `window_values`, none by default; never a port or a
  registered value of the verifier's) to the root of that value's entry in the record's `window.activations.values`,
  refusing it at any other root, under two roots, or with no such entry (`audit.window_reads`, `Record.windowReads`).

Pinned by `tests/verity/protocols/proofs/spot/verifier/test_registered.py`, `tests/verity/protocols/proofs/spot/verifier/test_served.py` and `tests/verity/protocols/proofs/flock/prover/test_lean_registered_reads.py`.

