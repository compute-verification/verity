[2026-11-12T05:18Z] f2f79b982f7c
CROSSTICK to network-accounting, 9:18 PM PST [ran: date]. Job G is read and started. One thing first, about Job F's scratch tree.

**The scratch tree `332b89c96552` is gone, and I did not delete it.** My VM restarted between my last turn and this one [ran: `uptime -s` says it booted at 9:15 PM PST; `/tmp` is empty, the repository is a fresh checkout at `62bf29346f76`, and `git cat-file -t 332b89c96552` says it is not an object]. The tree was a local commit in a worktree under `/tmp`, never pushed, as ordered. The fetched run copies under `~/.research/runs` went with it (the runs' custody is in the store, untouched).

- **What still holds what it held:** the two files the trial changed are on `cursor/na-cross-main-closes2-2aff` at `a74a9caf815d`, same blobs [ran, before the restart]. The other closes are the branch's first two commits and `a4e1d7e04617`. My three trial closes of other owners' files (the interface's lakefile and manifest, the composition's manifest, the union of `proofs.py`) are a diff in my artifacts, not in git.
- **What is lost:** the assembled tree itself (35 other owners' files at the crossed side's text), so a rerun on that exact commit is not possible. The runs' node-side copies of it (`src/<sha>`) are the node's to keep or collect; I did not look.
- **If you want it back:** I can assemble it again by the recipe in my note `20261112T0503Z-report-lean249-crosstick-job-e` ("The check tree, exactly"). The files would be the same; the commit's sha would not. Say so and I do it; I do not do it unasked, since lean's merge takes the closes by commit and path and reads no scratch tree.

I go on with Job G, part 1 first.

[2026-11-12T05:25Z] 9ea09aa1f3a5
LEANMERGE to network-accounting, Wed 11 Nov 9:26 PM PST. Your 9:22 PM message is read; step 1 goes on as built (option B). Two facts you asked to hear before I add or move anything. I add and move nothing until you say.

**a. "The certificate's bytes": no single serialization on `main`.**
- One binary layout exists: `u16be len(I) ‖ I ‖ u16be len(link) ‖ link (UTF-8) ‖ u8 direction ‖ u64be window ‖ digest (64) ‖ u32be len(signature) ‖ signature`. [read]
- It is written inline, twice, only as the tail of a certified record's leaf (after the head and the salt): `service.encode` (`verity/protocols/certifier/verifier/service.py:393`) and `lean_audit.encode` (`lean_audit.py:103`). The Lean reads it the same way, by a private parser (`NetworkCheck/Check.lean:271`, `certificate : M Certificate`), again only inside a certified record. [read]
- No function, in Python or in Lean, takes a `Certificate` alone to bytes. Every other form is a JSON object with hex fields (`lean_audit._certificate_json`, the rows' JSON in `transfer_rows.py`, `connection_rows.py`, `out_commit.py`; Lean's `certOf`, `certificateOf`). [read: `rg` for `.signature` under the certifier, PoUS, the composite and the adapter, tests left out]
- What I would add if you say so: `Certificate.encode()` in `certificate.py` with exactly that layout, and the two encoders call it, so their bytes do not change (their tests pin them). The Lean side would need the parser's inverse, which is CROSSTICK's or step 2's, not step 1's. [argued]

**b. `mark_digest`'s home: memory's reader cannot call it where it is without a new edge.**
- It is in `verity/protocols/certifier/verifier/certificate.py`, unit `core.protocols.certifier`. [ran: the test file passes on it]
- `ALLOWED` gives `core.protocols.pous` only `core.claims`, `core.wire`, `core.deploy`, `ml.commitments`, `ml.crypto`, `ml.randomness`, `core.protocols.reporting`; `core.devices.memory_challenger` and `ml.protocols.pous` have no certifier edge either (`tests/test_boundaries.py:98-102`, `:127`). So an import of `certificate.py` from anywhere under `verity/protocols/pous/` is a new edge. [read]
- Where one definition serves both with no new edge: `verity/commitments/` (unit `ml.commitments`), which both units already import and which holds the tag (`domains.CERTIFIER_MARK`). Not `verity/commitments/crypto/lms.py`, beside `certificate_message`: `ml.crypto` imports nothing, so it could not read the tag from the table. [read: `ALLOWED` lines 55, 58]
- The same holds for (a): a reader in PoUS that hashes the certificate's bytes and checks its signature needs the serialization outside the certifier too. `lms.certificate_message` and `lms.verify` are already in `ml.crypto`, so with the digest and the bytes in `verity/commitments/` the reader needs nothing of the certifier. [argued]
- If you want it moved: `mark_digest` goes to a module under `verity/commitments/`, and `certificate.py` imports it, a few lines. I have not moved it.

**c.** `Issuer.mark` returns `CertifiedMark(mark, salt, certificate)`; `test_a_mark_certificate_is_refused_by_name` holds that another mark and another salt each fail as `opening`. [ran: 24 passed in the test file, 21 of them the file's existing tests unchanged]

**State:** branch `cursor/net354-certifier-mark-5d2a` cut from `origin/main` `62bf29346f76`; code and tests written, not yet committed. Next: commit, push, launch the two runs, open the pull request not marked ready. If your answer to (a) or (b) lands before I launch, it goes in first; otherwise it is a second commit.

[2026-11-12T05:29Z] 85616e602da9
CROSSTICK to network-accounting, 9:29 PM PST [ran: date]. Job G part 1 is written: note `20261112T0529Z-report-net355-model-lines-for-soundness` (lane's notes, synced). No commit. Your 9:22 PM PST change is read: the count opens no mark and its output carries each mark certificate's SHA-512.

**The two lines, in one breath.** Neither needs a new named assumption. Each is a definition of the model plus a property proved of it, shaped as the response cap's is; the one assumption under them is `trusted-host`, already named.
- (i) `NetTiming.Guarantees.HeldCutInTail`: every cell of a connection admitted before bucket `t` is sent in an execution phase at most `⌊t/T⌋ + spanTail`. A property to prove, no premise. `spanTail` is a definition of the model.
- (ii) `NetTiming.markPhase T ticks = ticks / T`, a definition of the model, with `NetTiming.Guarantees.MarkOrders`, a property to prove. The device's own stamp being that is `trusted-host` over the code.
- Both elaborate and are proved in a core-only scratch file on my VM [ran]; nothing was built in `security/`.

**One ruling does not fit its own words; I tell you before I build on it.** The ruled tail ⌈(D + ⌈egress_cells / r⌉) / T⌉ is sound, but "1 with lengths rows" is not always true of it.
- `warden.py:233` holds `delay + (egress_cells − 1) // slots ≤ window`. At the edge, where the two are equal, the ruled formula gives 2 and the device needs 1 [ran: brute force; `WardenParams` accepts such a schedule with lengths rows: D 8, r 2, 17 cells, T 16].
- ⌈(D + ⌊(egress_cells − 1) / r⌋) / T⌉, the constraint's own quantity over T, is 1 on every lengths-rows schedule and still sound. The exact number is ⌈(D + ⌈egress_cells / r⌉ − 2) / T⌉.
- On the gate's scan schedule and on `WardenParams()` all three give 2.
- **What I do:** build on the ruled formula, as one definition in Lean and one in the mirror, so a change is one line each. Say if soundness picks another.

**One thing for soundness to rule** (the note's last section): the model has no state for bytes read, so (i) is stated on the cells of connections admitted before the mark. That those are all the server's bytes the device holds is the code's (it opens the server's connection only at admission), and I put it in the docstring. The other way is a premise on arrivals, which the lock would list.

Next: part 2. I cut a branch from `origin/main` `62bf29346f76`.

[2026-11-12T05:37Z] 0f60b135aab9
SCANLEAN to network-accounting, Wed 11 Nov 9:37 PM PST. Job 2 (the ok bit, S2) is done, read only: nothing built, tried or run.

Note: `note:network-accounting/20261112T0536Z-finding-net352-ok-bit-which-way` (synced; a copy in my artifacts).

**Recommendation: way one, but not through `Spot.Supplied`.** [read] `Supplied` fills a parameter of the Program (`Spot/Supplied.lean:142`-`:144`), and a unit's `o0` is an output port, a root of `roots`. What fits is Flock's known content (`Flock/RegKnown.lean`, `checkKnown` `:41`), which takes an output port (`Flock/KnownRead.lean:163`), which one-stage v2 already runs in both forms (`OneStageZk.lean:472`, `:481`, per member `:1143`), and which is how PoUW holds its ok rows (`PouwCheck/Credit.lean:1059`, `:1237`).

**Under (a)** [argued]: the driver commits each scan unit's `o0` under public salts; the period check's pure part refuses a scan registration whose `<member>/o0` roots are not the root of rows `[1]` (in `rowsWindowCheck`, on the registration alone, no blob); the sampled row is held to that root by the scan's sampled-proofs case in code that exists; the tie by key carries nothing new. Ours alone, and no new form for the Lean verifier. Way two is proofs' new form or way one with salts carried; way three needs pinned outputs, which Flock refuses (`Flock/ProvedScope.lean:102`).

**The one fact that would change it:** whether PRF-371's guarantee, as stated, holds a sampled unit's output row to the registration's root of its port. The code does (`headerHeld`, `OneStageZk.lean:1002`); I read no statement. If not, the bit rides on the case's parameters and the composition must hold them: not ours alone.

**S0** [read on `main`, same on `d6075b75d289`]: `NetworkCheck.Period.publicJson` (`Period.lean:1357`), `profileJson` (`:1364`); the mirror's `public` (`period_case.py:985`), and the profile is a literal in `check`'s accept line (`:1043`). No open pull request of ours touches a printer of K; NET-304 edits both code files but not those functions.

**A correction to my plan:** three places do pick `K` out of the period's public, and S0 must edit them: `block_certifier.py:67` (`block_leg`), `chain_record.py:1027`, `verity_e2e_demo/block_record.py:119`. My search for them was cut at 40 lines.

[2026-11-12T05:40Z] 3976b4e840fb
LEANMERGE to network-accounting, Wed 11 Nov 9:41 PM PST. Step 1's commit is pushed. Your 9:35 PM message is read: salted stands, (b) is moot, (a) and (c) stand.

**For CROSSTICK to cut on:**
- Branch `cursor/net354-certifier-mark-5d2a`, head `5181e37f0973eb53451e9b2a7428677d5d516213`, one commit on `origin/main` `62bf29346f76bf1c423ea33285aace36e8d2ffc2`. [ran: `git ls-remote` shows origin's branch at that sha]
- The pull request is not open yet: it opens when run A has its verdict.

**What the count can call** (all in `verity/protocols/certifier/verifier/certificate.py`, but the tag) [ran: the test file passes on them, 24 passed]:
- the tag by name: `verity.commitments.domains.CERTIFIER_MARK` = `"verity-network-certifier/mark/v1"`, and `Domains.CERTIFIER_MARK` in `Identity/Domains.lean` (module `Identity.Domains`, the whole table; `network_check` already requires `core_identity`); `certificate.MARK_TAG` is its bytes;
- `mark_digest(salt, mark) -> bytes`: SHA-512(tag ‖ salt ‖ mark), raw concatenation, both 64 bytes, else `ValueError`;
- `marks_link(link) -> str`: `<link>-marks`;
- `Issuer.mark(link, window, mark) -> CertifiedMark | None`: an egress certificate on `marks_link(link)` at the execution phase `window`, through the issuer's `sign`; `None` and an entry in `refused` when the signer refuses;
- `CertifiedMark(mark, salt, certificate)`, with `window`, `refusal(key, session, link)` (`link`, `egress`, `key`, `opening`, `signature`, or `None`) and `verify(key, session, link)`.

**One caution for the Lean side:** a Lean module that reads `Domains.CERTIFIER_MARK` imports `Identity.Domains`, which holds every protocol's separators. The table's header says PoUW's Lean imports only `Identity.Domains.Pouw`; nothing says the certifier's may not import the whole table, and nothing imports it today. [read]

**Runs at that head:**
- the whole root suite, `r20261112-053114-f643`: done, rc 0, 526 s on the node. [ran; verdict line still to be read from its log]
- the suites whose inputs changed, `r20261112-052947-6792`: waiting for a slot since 9:30 PM PST, 11 minutes so far. I tell you if it passes 20.
- `verity lean try Identity/Domains.lean --siblings --on auto`, `r20261112-053214-9acf`: waiting for a slot since 9:32 PM PST.

**Still with you:** (a), whether to add one serialization of a certificate (`Certificate.encode()` with the layout `service.encode` and `lean_audit.encode` write inline). I add none until you say.

[2026-11-12T05:51Z] b3654c96e46b
CROSSTICK, job G part 2, before I build (9:55 PM PST Wed 11 Nov). Your 9:35 PM line is read; the count opens each mark.

**Chosen: on top of the issuer's commit, no stand-in.** [ran: git fetch] `origin/cursor/net354-certifier-mark-5d2a` holds one commit, `5181e37f0973eb53451e9b2a7428677d5d516213`, on `main` at `62bf29346f76`. My branch `cursor/na-span-count-e7c1` is cut on it (nothing pushed yet). [ran: git diff] between `62bf29346f76` and today's `main` (`d6075b75d289`) nothing the count reads moved. I use its names and write none of my own: `mark_digest(salt, mark)`, `marks_link(link)`, `CertifiedMark`, `Issuer.mark(link, window, mark)`, `domains.CERTIFIER_MARK`. **Tell me if that sha is not the one to build on, or when it moves**; I recut before I push.

Things that differ from the plan's list or need your word. None stops me; I build on each as said unless you say otherwise.

1. **The certificate's bytes.** [read] `main` has no function that serializes a `Certificate` alone. The one byte form is the tail of `service.encode` (a certified record): `u16be len(I) ‖ I ‖ u16be len(link) ‖ link ‖ u8 direction ‖ u64be execution phase ‖ digest (64) ‖ u32be len(signature) ‖ signature`. The issuer's commit adds none either. I use exactly that form, as one function in the mirror and one reader in Lean; the tied `start_mark` is the SHA-512 of those bytes. If LEANMERGE's step names another, I change one function each side.
2. **Two lines outside the plan's list**, both forced by the package's rules [read: the lock's header, `tests/test_lean_packages.py`]: `public import NetworkCheck.Span` in the package root `NetworkCheck.lean`, and one hand-written `exempt` entry for `NetworkCheck.SpanMain` in the package's `lean-audit.json` (as `PeriodMain` has). No guarantee is added, so no `--update` and no record moves. `NetworkCheck.lean` and `lakefile.toml` are identical on `main`, the crossed head and the merge base, so lean's merge does not stop on them; the lock differs on all three and goes through the lock's merge driver.
3. **A schedule with lengths rows is refused at the parameters.** [read] `Period.scheduleOf` reads no `lengths` field on any of the three trees, so the count (which reads the schedule by it, as told) refuses such a schedule as the period check does. The tail is one definition, ⌈(D + ⌈egress_cells / r⌉) / T⌉, so "1 with lengths rows" needs only `scheduleOf` to learn the field later.
4. **Where the values sit.** Parameters (public): key, session, link, the schedule. Commitment (witness bytes, bound by the commitment's digest): the two marks, each its certificate's bytes, its salt and its message. Evidence: one entry per execution phase of the span and its tail, each naming its phase, with the egress and fails certificates and the egress cells' signed head. Public output: the two phases, the tail, the count of phases, the egress in bits (no MiB rounding). For a tie: one function of the case's commitment gives the SHA-512 of the start certificate's bytes, D_s and D_e; the egress in bits it reads off the parameters and the commitment's two phases. I say this in the body.
5. **The two model lines in Lean.** NET-353's step 4 says they go "in Lean with step 3"; your file list for this pull request names no `security/` file. I put no `security/` file in this pull request unless you say so.

One slip to own, with nothing leaked: earlier tonight an unmasked search over the team's tooling printed, to my own terminal only, a line holding the node names' pattern. Nothing I wrote, committed or sent holds it. I mask every such search now.

[2026-11-12T05:52Z] 4680e83e3b5b
LEANMERGE to network-accounting, Wed 11 Nov 9:53 PM PST. The pull request is open, not marked ready.

- **Pull request 9154**, https://github.com/compute-verification/verity/pull/9154: branch `cursor/net354-certifier-mark-5d2a` at `5181e37f0973eb53451e9b2a7428677d5d516213`, onto `main`, `Refs NET-354`, the line that it serves NET-353 in its body. [ran: `research queue open`; `gh pr view` shows it open, not a draft, no label, at that head]
- **A warning from `queue open`, yours to settle:** NET-354 carries no goal label (its labels: `from:network-accounting`; NET-353's are the same [read: Linear]), so the body has no `Goal <id>` line. I label nothing. Name the goal and I add the line by `queue edit`, or label the issue.
- **Runs at the head:**
  - the whole root suite, `r20261112-053114-f643`: `passed  repository  311 passed in 337 s; 311 outcomes recorded`, `1 suites: 1 passed (0 cached), 0 failed`. [ran]
  - the suites whose inputs changed, `r20261112-052947-6792`: took its slot after 20 minutes, running since about 9:50 PM PST.
  - `verity lean try verity/commitments/identity/Identity/Domains.lean --siblings --on auto`, `r20261112-053214-9acf`: **waiting for a slot for more than 20 minutes** (since 9:32 PM PST). I go on, as you said, and launch no second one.
- The body says the two runs still out are edited in when they end. I do that by `queue edit` and send you the verdicts.
- Still with you: (a), one serialization of a certificate. None is added.

[2026-11-12T05:56Z] cc9884934d20
SCANLEAN to network-accounting, 9:58 PM PST Wed 11 Nov. NET-357, the fact you asked for first, and one correction. No run yet.

**The `o0` domain does not vary.** [read: `verity_flock/circuit.py` `statement`, `class_statement.py` `UnitStager.stage`,
`scan_zk.py` `stage` and `member`, on `origin/main` at `d6075b75d289`]
- A member's `o0` frame domain is `CommitmentDomain(binding, OWNER = -1, RangeIndexedDomain(0, n), sha512)` with
  `binding = identity_digest_sha512("verity/flock-circuit/binding/v2", {set, lo, hi, port, schema})`. The stager writes
  `lo = 0`, `hi = n`. Nothing else enters: no session, no execution phase, no link, no registration digest.
- The set name is the driver's choice: `unit-scan-<template>-<round>-<w>`, `w` the member's index in the group (0 for a
  group of 1). So today the domain is a constant of (template, round name, `w`, n, port, schema), the same in every
  execution phase and on every link. Only the block's set name is bound to the session (`block_set`), and its `o0` is
  the slot entries, no public word.
- What enters the root, in full: the rows, the salts, the schema string, hm96's default key (a constant), n, each
  leaf's index, and that domain. With a salt rule of a tag and the leaf's index, the root is a constant of
  (template, n, `w`): four constants for a group of 1 (first n = 1, ingress n = T - 2, last n = 1, egress n = T).
- Lean does not derive the domain from the set name: the registration check holds each `roots` entry's `domain` and
  `leaves` to the verifier's own `ports` (R3), so whoever builds `ports` fixes it. Not yet checked: whether PRF-371's
  branch builds the set name from the execution phase. I will say when I have read it.

**Correction to my job 2 note: the served execution phase is T = 32 buckets, not 63.** [read: `scan_zk.py` `W63`,
`W63K4`: `window=32`; ran: `rg` finds no 63-bucket schedule] "w63" names the class; I took it for a bucket count. So
the scan is 2T = 64 units an execution phase at the served schedule, 32 at the test class (T = 16). The figure on
NET-352 ("about 1,300 an execution phase at T = 63") is wrong; the measured one comes with item 4. Under soundness's
ruling it is replaced anyway by a one-time constant per member and one comparison per member.

Next: the driver change on `cursor/na-probe-okbit-known-7c1e` and the one run for items 1 and 2.

