[2026-11-12T06:22Z] 1d9733a9e52c
SCANLEAN to network-accounting (coordinator), 10:22 PM PST Wed 11 Nov. Item 0 of job 3 (NET-357): the scan's shared trees under the
per-member form. Read only; nothing run. Marks: [read] a file on `origin/main` `d6075b75d289`, [argued] my inference.

**The line you asked for.** I did not meet it. My NET-352 plan read `rootsOf` only for the `<member>/o0` naming and
never held the scan's other root names against it. proofs is right.

**Short answer.** (i) as written (name each shared tree as one member's port) holds for the view, perhaps for the
state, and fails for the slot entries and for the network certifier's rows. A different (i) that is still ours alone
does hold [argued]: take the shared roots out of the registration's `roots` and have the period check read them
where the per-member form already carries them, each member's registered file. One point in it is soundness's to
rule (what fixes those files before the coins). If that is refused, (ii) is one sentence, at the end.

**Correction to the count: four kinds, not three.** The scan's registration also names `scan/view`
[read: `scan_zk.py:643`, `:1260-1264`, `:1674`: the honest round names state, prog and view; the egress-only round
view; the block round prog and state].

**Which form today's scan tests stage.** Not `Spot.OneStageZk` at all. Each sampled member runs one `--zk` session
behind the ZK server, held by `flock-verify verify --zk --archive --partition --program --registered` with the
verifier's own registered file (`a0.lean_verify`); the registration is a `verity/registration/v1`-form record checked
by one-stage's Python (`V.check`, R1 to R8) and the audit by `V.audit_record` [read: `scan_zk.py` `audit`]. So the
probe for item 1 meets no per-member refusal, and proves nothing about it.

## 1. Each kind: who reads it, through which port, how today's form holds it [read: `scan_zk.py` `member`, header]

| Root | Leaves | Read by | Held today by |
|---|---|---|---|
| `scan/state` | G·T; leaf 0 the empty state, leaf t+1 = S_t | input `sin` of `ingress-w`, `last`; output `o1` of `first`, `ingress-w` (the last unit's `o1` opens none) | each session's registered read at the tree's root (`Flock.Registered.checkPort`); R3 on leaves and domain; the period check's `stateTied`: the scan's `scan/state` entry and a wired audit's state entry (`roots` port `scan/state` or `registered` value `verity/certifier/scan-state/v1`, the anchored Program's `SCAN_STATE`) at one root and domain |
| `scan/prog` | NR·nb slot entries | inputs `prog<j>` of `first`, `ingress-w`, `last`, one table row a boundary | the same registered read; it is the block member's `o0` frame tree (`block_binding`: committed under that port's binding and salts); R8 lets `prog` through as tied; the period check's `progTied`, `wired`: the `scan/prog` entry at the root and domain of an audit's entries port `SlotEntry_v1{…}/c0/o0` |
| `scan/view` | G·T | output `o2` of `first`, `ingress-w`, `last`; input `state` of `egress-w` | the registered read; R3. The period check reads nothing of it |
| `rows/<direction>/<w>` | T, the network certifier's signed rows | ingress: input `row`; egress: output `o1` | the registered read at the head the verifier read under the deployment's key; R3; the period check's `holds`: some entry of `roots` or `registered` at the signed rows registration's root and domain (by root and domain, not by name) |

Nothing in one-stage's Lean ties a member's registered-file root to these roots of the record. The names are chosen
so it does not look: a session reports its reads under the *values* (`verity/certifier/scan-<kind>/v1`,
`warden/<direction>/<w>`), the record names the roots `scan/<kind>`, `rows/…`, and `Record.ownRegistered` keeps only
a finding whose value is a root's port or a `registered` value of the record [read: `Spot/Record.lean:99-140`,
`scan_zk.py` `warden_roots`' docstring]. Today the tie is the driver's: it writes both from one tree.

## 2. What the per-member form lets a member read of a tree that is not its own

- **Of the registration's `roots`: only its own, and they must be its circuit's ports** [read]. `Member.slice` hands a
  member the roots named `<member>/…`; `headerHeld` (`Record.matchesPublic`, `sameDict`) requires the member's public
  file to name exactly those roots; `HmOut.checkPublic` recomputes each from the file's `b‖c` at the header's domain.
  So a root named `<member>/<port>` is the frame root of a real port of that member's circuit, under that port's
  frame domain. A name alone cannot make a shared tree a member's.
- **Through `registered`: yes** [read: `OneStageZk.lean:89-91`, `:463-520`, `:1037-1066`]. Each member's own registered
  file (`proofs.members.<name>.registered`, the verifier's, in the parameters) is its statement's as the one
  statement's is: Flock holds every table row of a sampled session to open the root that file names
  (`Registered.checkPort`), and `classHeld` refuses two members' ports of one value that differ in root, leaves, row
  width, domain or schema, sampled or not. So members already share a tree: by value, at one root.
- **Known content rides the same path**: `readsOf` runs `knownOf` and `checkKnownAll`, the setup `KnownRead.check`.
  Form 1a's `known` entry on `o0` goes through the per-member form by the code item 1 will run [read, not run].
- **`supplied`**: a record-level form (R7: the verifier fills a registered parameter of the Program), not per member.
  Not read further.
- **What no one checks in Lean:** that a reader's registered root is the root the registration holds for the owner's
  port. If the reader's value is named as a root's port, one-stage's `recordChecks` refuses the audit unless the
  record lists it under `registered` (`Record.checks`: the kept findings must equal the record's `registered`). The
  two-stage run has the rule that would let it through (`Record.lessOwnReads`: "reads of the registration's own
  ports at their own roots", `own_reads`, `Spot/Run.lean:175`); one-stage does not apply it.

**So, kind by kind, can the tree be one member's port (the `block_binding` construction: commit the tree under
that port's frame binding and salts, the others read it through their registered files)?** [argued, none run]
- `view`: yes. An egress member's input `state` is exactly the view rows of its execution phase; the view becomes
  one tree an execution phase (`egress-w/state`), read by that phase's ingress units' `o2`.
- `state`: perhaps. No member's port holds all G·T rows today (leaf 0 is read by no scan unit; three members share
  the rest). An ingress member's `sin` table would have to carry every row, which needs Flock to take a table row no
  unit reads [not checked], and the tied audit's `SCAN_STATE` entry would have to take that port's domain, since
  `stateTied` compares root and domain and R7 holds the audit's entry to the audit verifier's own domain.
- `prog`: no. It is already a member's port, the block member's `o0`, in the *audit's* registration. A scan member's
  `prog<j>` port has another frame domain (Flock binds the set, the range, the port's name and the schema:
  `prove_leg.frame_ports`), so `wired`'s equal root and domain would fail.
- the certifier's rows: no. Their domain is the certifier's signed one; a frame port's is Flock's binding.

**What (i) would have to be instead (still ours, no accepted form added):** the shared roots leave the
registration's `roots`. The period check, which under NET-352's plan hands each member its registered file when it
calls the per-member form, makes its ties against those files instead of against `scan/*` and `rows/*` entries:
`holds`: the members' ports of the certifier's value are at the signed rows registration's root and domain (the
period check verifies that registration itself, `rowsCheck`); `progTied` and `wired`: the members' `prog<j>` ports
are at the audit's entries port; `stateTied`: the members' state ports are at the audit's state entry. `classHeld`
already gives "one tree a value across members". That is a walk over members and ports, never over units.
**The point for soundness:** today those roots are under the registration's digest, which the challenger's coins are
released after. In this form they are fixed before the coins only as every v2 registered file is: the auditor's
parameters. For prog, state and the certifier's rows another record that predates the coins also holds the root
(the audit's registration, the certifier's signed head). For the view nothing else does, so the view should be the
egress member's own port, which is in the registration.
A second way I did not finish reading: list them under the registration's `registered` (a v2 record). The period
check's four readers already accept a `registered` entry, and `recordChecks` would then hold every accepted session's
reported root to the record's. But R7 holds `registered` to the Program's registered parameters at the verifier's own
domains, and `sameDict` needs every listed value opened by some accepted session, which a sample that misses a
member breaks. I would not choose it.

## 3. What that (i) changes [read for the sites; line counts rough]

- **The registration**: 4 kinds of root gone, so its digest; `scan_zk.py` `audit` `:1260-1268` (9 lines),
  `warden_roots` (17), the header's text (about 10). New: the driver writes the per-member parameters (each member's
  registered file exists already as `stage-<member>/registered.json`); that writer is S1 of the plan either way.
- **The period check** (`NetworkCheck/Period.lean`): `holds` (4 lines), `tiedAudits` (11), `wired` (11),
  `stateEntries` and `stateTied` (15), their source moved from `reg`'s entries to the members' registered ports,
  plus a reader of those ports (about 30 new) and the module docstring (about 20). `rowsWindowCheck` calls them; its
  own lines change by the argument. The audit's side of each tie (its `<unit>/o<j>` output ports, its entries port)
  is members' ports of the *audit's* registration and does not change.
- **NET-304's rule** (`tiedInputWired`, `tiedInputCheck`; on `origin/cursor/na-tied-need-crossed-37ad`, not on
  `main`): it reads the answer's `rests_on_period_check` names (`prog<j>`), not a root, so it does not change. On
  that tree the scan's registration names one `scan/prog<j>` root a lane (`progName j`), each chunk `j`'s entries
  port, so there the kinds are the lanes' progs, the state, the view and the rows; the same move, a lane at a time
  [read: that branch's `Period.lean:1076`, `:1141`, `:1306`].
- **Its properties**: `security/Properties/NetworkPeriod.lean` states two of these ties in the registration's terms
  (`:95`, `:113`) and `security/Proofs/Properties/NetworkPeriod.lean` proves them (`:206`, `:355`). Two statements
  change, so two lock records. That is a Lean pull request with changed statements, ours.
- **The mirror**: `period_case.py` `_holds`, `_prog`, `_tied_audits`, `_wired`, `_states`, `_state_tied`
  (`:791-875`, about 80 lines), the negatives that move a name between `roots` and `registered` (`:1348-1400`, about
  50), the case builder (`:1592`, `:1718-1723`).
- **Tests**: `test_scan_zk.py`, 14 sites in about 6 tests; `test_network_period_protocol.py`, the cases follow the
  mirror. `scan_period.py` passes registrations whole (2 docstring lines).
- **The records audit**: nothing. It reads the scan's state as a `registered` value (`SCAN_STATE`), never a root.
- **Fixtures**: none names these roots [ran: `rg` over `*.json`, `*.toml` outside `tools/`].
- **Onsite's composite**: nothing. `live_challenger`'s test names `warden/egress/0` as the certifier's registration
  value, not a root of the scan's; `verity/protocols/onsite` and `remote` name none.

## 4. If that is refused: the smallest (ii), for proofs

"A root of the registration whose port is no member's is the registration's own: `rootsOf` gives it to no member, and
the record check holds a session's registered read whose value is that port as the two-stage run's does
(`Record.lessOwnReads`), so every accepted session that reads the value reads it at that root, and R3 holds its
leaves and domain as today." Our driver then names each shared value by its root's port. Roots: `scan/state`,
`scan/prog`, `scan/view`, `rows/<direction>/<w>`.

## 5. Does the same refusal meet the tied audits' registrations?

- **Served windows** (`rows.registration`, what the records audit reads): every root is `<member>/<port>`, except
  `row-table` at a window with row tables, which the form refuses by name (`ROW_TABLES`) and the records audit
  refuses already; the program's values, the scan's state among them, are `registered` entries, not roots
  [read: `rows.py:773`, `:795`, `:843`; `records_audit.py:665-709`]. The ties' proof blocks are templates of that
  Program, so their ports are members' [argued: not read end to end for the response tie].
- **The ingress tie on its own stream** (`tie_service.IngressTie.registration`): one root, port `cells`, the network
  certifier's root. It is one statement, not members, so `rootsOf` never runs; whether the one-statement form's
  `publicHeld` passes a public file against a root named `cells` I did not check.
- **The certifier's other drivers** (`warden_zk`, `egress_zk`, `served_tie_zk`, `compile_slice_zk`,
  `served_statics_zk`): `<name>/<port>` only [ran: `rg "roots.append"` over `benchmarks/`]. Only the scan (and one
  benchmark, `a7.py:962`, `canon/<kind>`) names a root that is no member's.

## One fact for the choice on NET-352, met on the way [read]

`security/Proofs/Flock/SoundAtBridge/Served.lean`: Flock's proved link ties the value layer to a committed row "only
at a registered read: a row of a port of `own`, the roots registered before the session's statement and coin … At a
hiding leaf nothing else is bound, so at a served root that is not a port of `own` the value layer is the
extractor's and not the committed tree's". A member's `o0` is no port of `own` today. Form 1a makes it one. That is
the fact I asked proofs for, read from the trusted statement's own docstring; proofs should confirm it is the
statement PRF-371 instantiates.

**Not done:** nothing run; the Rust prover and `KsScope` not read for a table row no unit reads; R7's held, pinned
and output forms not read for the `registered` way; the response tie not read end to end. Items 1 to 6 are next;
the probe's driver change is designed and not yet written.

