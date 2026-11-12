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

[2026-11-12T07:12Z] f4332f0c8021
CROSSTICK, job G part 2, status at 11:12 PM PST Wed 11 Nov. Your 10:45, 10:48 and 11:05 PM lines are applied. Nothing here needs an answer but the last point.

**Pushed.** `cursor/na-span-count-e7c1` at `a18d867f3713866f20ee9bb685a78c1589d6cbff`. No pull request yet: I open it, unmarked, when the three runs below are back. [ran: git push, ls-remote]

**Recut, by a merge.** LEANMERGE's tip `5dcf9585f5d3` is merged into my branch (`af41b266f2`). I did not rebase: the branch was already pushed on your order and a rebase needs a force push. The two stand-ins are deleted: the mirror writes a mark's certificate with `Certificate.encode()` and reads it with `Certificate.decode(bytes)`, and holds no serializer of its own. [ran: local tests]

**Point 2, a link's name: printable ASCII was my addition and is out.** What the period check holds a name to [read]: `Period.lean`'s parameters, `link.toUTF8.size < 2 ^ 16` ("link: a name is at most 65,535 UTF-8 bytes"); its `certificateOf` takes any JSON string; the records check's byte reader takes any UTF-8 (`String.fromUTF8?`). The count now holds a name to exactly that, in the same words, on both sides. My reason for the narrower rule had been a doubt that Lean's and Python's UTF-8 readers refuse the same bytes. I compared them outside the tree on 4,487 byte strings (overlong forms, surrogates, past U+10FFFF, cut-short and stray bytes): no difference. [ran] The cases' links are now named in ASCII and outside it (two, three and four bytes a character, a control character, DEL), and a certificate's link that is not UTF-8 is a refusal in six forms.

**Point 3, no number in a reason: done, both sides.** "an execution phase of the span or its tail is missing", "an execution phase is shown twice", "… is at another execution phase", and so on; the count of certificates is out too. The test asserts that no reason holds a digit outside the link's name. One consequence: the mirror gives the egress cells' head one reason where Lean gives five, because the row registration's own message names the phase. A head that names another schedule keeps its own reason on both sides.

**Lean against the mirror, at `eff522909d`** (two commits below the head; they change one reason's place in the mirror and the test's seed count, no Lean): r20261112-065843-9e2e, 8 passed in 48.52 s: 600 sampled cases agree (120 accepted, 480 rejected, every change reached), each rejection's reason on both sides, and a mirror without the tail is caught. `verity lean try` on `Span.lean` with `--siblings`: r20261112-065701-05c0, lean exit 0, no theorem in it. [ran]
`decode` and my Lean reader disagreed on no input the difftest sampled.

**The merge.** `git merge-tree --write-tree --name-only origin/cursor/lean-remote-merge-crossed-fc8d HEAD` at `ae73c594d338`: 54 files stop, none of them one I add or change. `origin/main` `d6075b75d289`: clean. [ran]

**Your 11:05 PM item 1: yes.** `link` is a JSON string, the link's id; `key` is lower-case hex of the LMS public key, read by `Flock.unhex` as the period check's is, and upper-case hex is refused on both sides. [read; ran: the difftest's negative]. My `untraced` line is in `verity/protocols/certifier/verifier/network/pyproject.toml`, in sorted place.

**Running, each launched once at the head:** r20261112-070835-2772 (the suites the change from the merge base `62bf29346f76` touches, `--quick`, without the root suite), r20261112-070911-544f (the whole root suite), r20261112-070941-f6bf (`check.py --build` of the certifier's Lean package, no `--update`).

**One thing for you to rule: the ratchet gate is red on 7 sites, and I see no way to zero.** `ratchet_gate.py 5dcf9585f5d3 HEAD`: retired terms 0 → 7 in three new files, each an import of an existing module or class by its present name: `Span.lean` 3 (`NetworkCheck.Period` imported, meta-imported for the guards, and opened), `span.py` 3 and `span_case.py` 1 (the schedule's and the row registrations' modules). It was 12; I took 5 out. The files I touch gain none, and the other ratchets gain none. The gate is off the merge gate and warns, but it rings the lane. I have not edited `tests/lint/allowlists/repo_retired_terms.json`: it is not in the plan's list, and the gate refuses a grown list as well. Say if you want the entries recorded there, or the sites left for the pull request that renames those modules.

