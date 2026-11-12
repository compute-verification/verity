[2026-11-12T07:12Z] 56222b3a1e16
SCANLEAN to network-accounting (coordinator), 11:12 PM PST Wed 11 Nov. Job 3 (NET-357) is done. The note:
`note:network-accounting/20261112T0706Z-finding-net357-form-1a-probe` (synced, 239e81b493). Its first lines say what
you asked: today's scan tests do not stage the per-member form, so the probe shows Flock taking known content on a
member's own port and nothing about that form.

**Form 1a runs, and my recommendation stands**, on three conditions: proofs' fact (no sentence yet), a fixed set
name for the member, and the period check checking the carried constants once itself.

1. **Flock accepts it** [ran: r20261112-063024-1db8, rc 0]. The Lean verifier `flock-verify verify --zk
   --registered`, on each member's session behind the ZK server's send check: `honest/first: 1 units drawn, Lean
   True`, `ingress-0: 14 … True`, `last: 1 … True`, `egress-0: 16 … True`; the audit accepts, bound 0 of 32.
2. **A unit whose bit is 0** [ran: same run]: `Lean False no verdict (rc 2): registered reads: port o0:
   committed/ingress-0/o0's registered root is not the root of its 1 known rows` (Flock's `checkKnown`). With no
   `known` it is accepted, as on main. With the constant root named: `table row 0 does not open … at position 0`
   (`checkPort`). A forged prover that commits `[1]` and computes 0: the proof fails (`zk inner: the batched
   constraint fails`). **One thing to know: a file naming known rows `[0]` is accepted.** Flock holds the content
   the verifier's file names; the content `[1]` is fixed only by who writes that file and by the period check's rule.
3. **The rule in Lean** [ran: r20261112-065557-5fb0, `lean exit 0`], on that run's own registrations: `TRY
   run-honest-T16 T=16 walk: ACCEPT`, `carried, the comparison: ACCEPT`; the unread registration's entry: `REFUSE …
   is not the root of its 1 known rows`. The constants can be a key of the rows form's parameters beside `scan` and
   `schedule`; the period check checks them once when it reads them (`checkCarried`) [argued].
4. **Cost** [ran: r20261112-064013-8b2b, r20261112-065747-abe5; a 192-thread AMD EPYC host, one thread]. As ruled,
   T = 32: 512 SHA-512 calls once, 2.4 to 2.7 ms; then 0 calls and 75 ns an execution phase: 14 ms a week, 61 ms a
   month, 0.74 s a year on one link. The walk it replaces: 512 calls and 2.5 to 2.9 ms an execution phase, 96.8
   million calls and 476 to 554 s a week, 5.05 billion and 6.9 to 8.0 hours a year. T = 16: 256 calls once, 1.2 to
   1.5 ms. Flock's own verify beside it: 29.4 s for the four sessions of one execution phase.
5. **Salt rule**: `SHA-512(tag ‖ u64be(i) ‖ u32be(c))`, c = 0, 1, 2; the tag and the leaf's index only. **The
   frame domain is what varies today**: its set name carries the round's name and the member's index, so one
   round's constants refuse another's registration [ran: the crossed comparison]. The set name is the driver's
   choice, so it can be fixed [read]; not run.
6. **Fewer than `leaves` hashes: no** [read: `FV3.lean`]. Every leaf, pad and node frames its index.

**proofs' statement:** none yet in PRF-371 or PRF-490 (newest comment Tue 11:40 PM PST). #8569's `opens_rowLeaf` is
one member's. **Apart from form 1a:** its premise `OpensReady` asks at least two leaves a port, and the scan's
`first` and `last` members have one [read, not checked further]. That is for soundness to hear from you.

**`tests/test_verifier_cost.py` passes with the walk in the tree** [ran: local]: its scan does not name this loop,
so its passing is no evidence either way.

Scratch branch `cursor/na-probe-okbit-known-7c1e`, head `bfd80a7a7c`, pushed, no pull request. Not run: the
per-member form, T = 32 through Flock, more than one link, the value-salt service, the scan's suite on the branch.

