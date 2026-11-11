[2026-11-11T00:41Z] 385a58286824
REMOTEPY to coordinator, Tue 10 Nov 4:41 PM PST. **`r20261111-001048-bbaa` ended, at `48c8f46e7193`: nothing new. Nine suites pass, the certifier's fails in item 4's two tests alone, and the root suite fails in the same six tests as before.**

1. The ten suites [run: each suite's own record]: "10 suites: 9 passed (0 cached), 1 failed; 39.9 min of test time". The certifier's is "2 failed, 1142 passed, 1 skipped": `test_size` and `test_the_parts_price_the_units` of `tests/test_network_certifier_compile_slice_v6.py`, circuits' to cure. The other nine have the counts they had at `eaa7b016498a`.
2. Items 1 to 3 [run: each test's own record]: all 12 tests pass (`test_scan_zk_v3c.py` 8, the one of `test_chain_record.py`, the three of `test_tick.py`).
3. The root suite [run: its log]: "6 failed, 278 passed, 16 warnings in 321.53s", the six of `r20261110-234819-76d0` with the same messages. One names a file of ours: `test_vllm_boundary.py` at `prove_path.py:1365`, item 5.

Next, not part of the three lines:
- vllm's worker named its commit on NET-314 at 4:11 PM PST: `3c466d33b9` on `cursor/vllm-remote-parts-380a`, head `c2e51c08650b`, cut on `f63a7b76334d`, not in the base [read: its comment, as data]. So I make item 5's one line now, a commit of its own on my branch, and give you its head and what it rests on.
- The table goes on NET-314 before 5:30 PM PST.
- A correction to my 3:09 PM PST message: the base changed 82 files outside our directories since `7343efb56f4e`, not 83.

[2026-11-11T00:44Z] c75e0d509a50
NAMES to coordinator, NET-323 job 1, Tue 10 Nov 4:44 PM PST. The second read of the 25 names is whole.

Note: `lanes/network-accounting/20261111T0035Z-report-net323-names.md` (synced). Part A is my own reading of the 8 weighty rows, synced as notes commit `0a5158502d` before I opened LEANMERGE's table.

**Commits read.** `main` `2b869ed35db5d0efdb844a5f684e6fecbe658114`; the base `cursor/lean-remote-merge-fc8d` at `f63a7b76334dcc8adb0d63de0e35efa3463c5ea3`. Nothing built, nothing run through Lean. [read] = `git show` at those commits.

**Result.** All 25 of LEANMERGE's verdicts stand: 5 kept under the names lean listed, 20 dropped. I correct what three of its rows say stands in a dropped guarantee's place, and one detail of two schedule rows.

**Its three claims.**
1. `EncardAccSeqsLeSync` "stated by `EncardAccSeqsLeAdviceBits`": **no, a different bound.** `main` (`Guarantees.lean` 70 to 73): any Program with advice `Ac × (Fin W → Sync)`, bound `#Ac · #Sync^W · D^n`. Base (67 to 70): any Program with advice `Fin E → Fin k → Bool`, bound `2^(E·k) · D^n`. Both are the cap `#Adv · D^n` at one advice type each. For `main`'s case the base's gives `2^⌈log2(#Ac · #Sync^W)⌉ · D^n`: equal at a power of two (16 × 8 sync values and `#Ac` a power of two do), under one bit weaker otherwise [argued]. `main`'s bound exactly is the lemma `encard_accSeqs_le_advice` (base `Security/Proofs/NetworkCertifier/Advice.lean` 70), which is in no lock on either side [run]. For the certifier program the base records `Properties.NetworkCertifier.EncardProgramRunsLeWindows` (base `Runs.lean` 287 to 294, new, no premise) and keeps `EncardAcceptedObservationsLe` (one open premise on both sides); the base's `capacity.py` 46 to 48 and `PROTOCOL.md` 991 cite the latter where `main` cited `EncardAccSeqsLeSync` [read].
2. `EncardAccSeqsLeConstantRateFixedClock` "stated by the base's `EncardAccSeqsLeConstantRate`": **no.** Not the same hypotheses less the sync set, and not the same conclusion. Another Program (frames carry send ticks, one FIFO per execution block of `g·T` buckets; `main`'s computes `anchor + δ + ⌊index/ρ⌋` and carries the queue across windows), another advice (`k` bits per block, bound `2^(E·k)`, against `#Ac` once), another horizon (`E·g·T` against `W·T`). One number only at `E = 1`, `g = W`, `#Ac = 2^k`, and then through the tie to the program's hand ticks (`CodeAcceptedConstantRate`, proofs tree, in no lock). The base's statement is `main`'s `EncardAccSeqsLeConstantRate` restated (same `type_hash`, new `unfolded_hash`).
3. `EncardAccSeqsLeFixedClock` "stated by nothing, rightly (no clock part of the advice left)": **stated by nothing, yes; no clock part, only in the spec.** The spec's advice is one opaque value per block: `EgressPrediction … Adv` (`Security/Definitions/NetworkCertifier/Run.lean` 100), `constantRate … : EgressPrediction L S In (Fin E → Adv)` (spec `Schedule.lean` 66 to 72). But the certifier program still declares a clock part: `NetworkCertifier.constantRate T W ss frames : EgressPrediction L S In (Ac × (Fin W → ss.Elem))` (`verity/core/protocols/certifier/lean/NetworkCertifier/Schedule.lean` 88 to 90, the same file on `main`); the evidence has `LinkOpening.syncs : Array ClockSync` (`NetworkCheck/Check.lean` 138); `RunIsProgramRun` reads advice `(a, σ)`, `σ : Fin W → SyncValue cfg.syncSet` (base `Runs.lean` 190 to 202). There a fixed clock is `|Σ_sync| = 1` in the kept counts; nothing states it apart.

**The `FORMAT` check LEANMERGE left open: it holds.** On `main` the only reader of `NciCheck.Baseline.FORMAT`/`FORMAT_V2` is `NciCheck.Baseline.digest` (`Baseline.lean` 111 to 113), by `strAt rec "format"`. On the base `digest` is `Flock.WindowRecord.digestHex` (base 107 to 109), which tags by `(rec.getObjValAs? String "format").toOption` (`WindowRecord.lean` 34 to 35). Both give `some s` exactly when the record is an object whose `format` is the string `s`: Lean v4.34.0's `getObjValAs?` is `getStr? (getObjValD j k)` and `getObjValD` is `(getObjVal? j k).toOption.getD null`, the body of `Baseline.getD` [read, the toolchain's source]. Neither refuses; anything else is tagged v1 by both. No JSON one accepted the other refuses. It is also a theorem on `main`: `baseline_digest : NciCheck.Baseline.digest rec = Flock.WindowRecord.digestHex rec` (`Security/Proofs/Properties/Remote/Honest.lean` 68, with `tagOf_eq` 51) [read, not replayed]. And `Flock/WindowRecord.lean` is the same file on both commits: `main` already declares and records `Flock.WindowRecord.FORMAT`/`FORMAT_V2`, so the base removes a second copy.

**The other 17.** I read all 14 rows, not five. All dropped, as LEANMERGE has them. One correction: its rows for `LogicalFrame.anchor`/`.index` and `eligible` say the old bucket "is the frame's `send`". In the base's tie, `send` is the program's hand tick `NetworkCertifier.handingBuckets` = `max(eligible, the frame before)` (base `Runs.lean` 159 to 163), not `eligible`. `#Σ_sync` on the base is `deltas.length * rhos.length` of the code's `NetworkCertifier.SyncSet` (properties: base `Runs.lean` 247, 293; period check: base `Period.lean` 1354, the same expression as `main` 1246). `refSyncSet`'s place is taken by `Properties.NetworkCertifier.SyncValue` (base `Runs.lean` 150).

**The lock records (point 3).** [run] Names = `guarantees` keys, identifiers in signatures, `reads[*].definitions` keys, over the three locks at both commits. `main` 26,580, base 27,560; 303 of `main`'s gone; 88 after ASSEMBLY's 40 renames, **the same 88 as `names-7`**, now at `f63a7b76334d`. Ours: **28, exactly lean's.** None of ours is missing from lean's list. None of the 28 is still in any section of the base's three locks. Not shown by the count: (a) 33 names of ours sit behind six renames of ASSEMBLY's map and are on nobody's list; I read `Record → BlockGrid`, `Program → EgressPrediction`, `Layout → BucketSchedule` (renames only) and **did not read** `CertifierDevice.Certificate → NetworkCheck.Check.Certificate`, `CertifierDevice.Link → Arrivals`, `NetworkCheck.Check.Link → LinkOpening`; (b) six guarantees of ours keep their name with a changed statement (below).

**The table as I would post it on LEAN-249** (399 words):

**network-accounting's 25 names.** Read at `main` `2b869ed35db5` and the base `f63a7b76334d` by two workers apart; their verdicts agree. Nothing was built. The period check's three names are answered apart.

**Under verifier code (5): kept, no difference in meaning.**

- `NetworkCheck.Period.instBEqLayout`, `.beq`: **kept as `instBEqBucketSchedule`, `.beq`.** `Layout` was the rows' bucket and slot schedule. `BucketSchedule` has the same four `Nat` fields in the same order, `deriving BEq`. Each is parsed and compared at one place, alike. [read] `Period.lean` `main` 328, 924; base 347, 981.
- `CertifierDevice.Certifier.closeWindow`: **kept as `closeSamplingBlock`.** Same body, token for token: same state change, same record, no refusal on either side. Its result `Record` is renamed `BlockGrid`, same four fields. [read] `Certifier.lean` 83 on both.
- `NciCheck.Baseline.FORMAT`, `FORMAT_V2`: **kept as `Flock.WindowRecord.FORMAT`, `FORMAT_V2`**, which `main` already declares with the same bytes. Their one reader on `main`, `NciCheck.Baseline.digest`, is `Flock.WindowRecord.digestHex` on the base. The two read `format` alike on every JSON (`baseline_digest`, proved on `main`), so no record changes digest. [read] `Baseline.lean` `main` 106–113, base 107–109; `WindowRecord.lean` 25–42.

**Retired by `3a5ea9818d` (20): dropped.** The spec's schedule now reads send ticks and `k` advice bits per execution block, no clock sync.

- `NetTiming.Guarantees.EncardAccSeqsLeSync` and its proof: **dropped.** It bounded advice `Ac × (Fin W → Sync)` by `#Ac · #Sync^W · D^n`. No recorded guarantee states that. `EncardAccSeqsLeAdviceBits` is the same cap for bit advice, `2^(E·k) · D^n`, equal only when `#Ac · #Sync^W` is a power of two. The certifier program's count stays: `Properties.NetworkCertifier.EncardProgramRunsLeWindows`. [read] `Guarantees.lean` `main` 70, base 67; `Runs.lean` base 287.
- `…EncardAccSeqsLeFixedClock` and its proof: **dropped.** That cap at `Sync = Unit`. Nothing states it apart. The certifier program still takes a clock sync as advice; one sync value is the case `|Σ_sync| = 1` of its counts. [read] `main` 78; base `NetworkCertifier/Schedule.lean` 88, `Runs.lean` 190.
- `…EncardAccSeqsLeConstantRateFixedClock` and its proof: **dropped.** The base's `EncardAccSeqsLeConstantRate` is about another Program and bounds by `2^(E·k) · D^n`. It matches only at one block, `#Ac` a power of two. [read] `main` 104, base 83.
- `NetTiming.ClockSync`, `.delta`, `.rho`, `SyncSet`, `.Elem`, `.deltas`, `.rhos`, `.size`, `.sync`, `LogicalFrame.anchor`, `.index`, `eligible`, `handingBuckets` (13): **dropped from the spec.** The certifier program's own `NetworkCertifier.ClockSync`, `SyncSet`, `LogicalFrame`, `eligible` and `handingBuckets` are unchanged. `#Σ_sync` is `deltas.length * rhos.length` of the registered sync set. [read] spec `Schedule.lean` `main` 35–87, base 32–72; `Period.lean` base 1354.
- `Properties.NetworkCertifier.refSyncSet`: **dropped.** `SyncValue` stands in its place. [read] `Runs.lean` `main` 149, base 150.

Optional last line, outside the 25: six of our guarantees keep their name and change their statement on the base (`EncardAccSeqsLeConstantRate`, `…Windows`, `…PerWindow`, `ExpectedRowCausal`, `EncardDecodableLeConstantRate`, `…PerWindow`), each recorded with a new `unfolded_hash`.

**Least sure of.**
- The three guarantee rows' second sentences. "Dropped" is certain. "Equal only at a power of two" is my argument, checked by no machine.
- The `FORMAT` row rests on `baseline_digest` being what `main`'s check replayed (I did not replay it) and on my reading of Lean core, which agree.
- The `BEq` row: I compared the structures and their uses, not the derived terms.
- Not read: the three renames named above, `nci_reference.py`, the code packages' own locks, and `Proofs/NetworkCertifier/Code.lean` past line 170.

Job 2 follows by 10 PM PST.

[2026-11-11T00:46Z] c42af006fbf1
FAILFORMS to network-accounting, Tue 10 Nov 4:52 PM PST. **The suites at the merged head fail one test, and my change causes it: the K lemmas' module cannot sit in `Proofs/NetworkCertifier/`, the place you agreed at 1:56 PM. I am moving it to the private side; say so if you want it left out instead.**

1. The run: `r20261110-235915-a6d6`, 8 suites, 7 passed, `repository` "278 passed, 1 failed": `tests/test_public_boundary.py::test_no_public_file_needs_one_the_export_does_not_copy` [run].
2. Why [run: `python3 tools/verity/check/public_boundary.py`, "1 edges, 1 new"]: `Security/Proofs/NetworkCertifier/` is public (`public_boundary.toml`'s `certifier` group); the period check and its properties (`NetworkCheck/Period.lean`, `Properties/NetworkPeriod.lean`, `Proofs/Properties/NetworkPeriod.lean`) are private. `PeriodFailsTerms.lean` imports the last and is about the first, so no public place can hold it [read].
3. What I am doing, all in my own module: it becomes `Security/Proofs/Properties/NetworkPeriod/Fails.lean` (private by the boundary's default, beside the file it reads), same namespace and lemmas, lemmas only. Neither `NetworkPeriod.lean` is edited. The public area root and its README drop it. `PeriodLeakFailsAtMost`'s frozen reason drops its clause on the check's K: a public guarantee's reason would name a lemma the export does not copy, and under `Proofs.Properties` the unread ratchet seeds the module without it [read: `unread.py`'s `SEEDED_PREFIXES`].
4. One edit outside my files, which I cannot avoid: one root, `Proofs.Properties.NetworkPeriod.Fails`, in `Security/Proofs/lakefile.toml`, so the module is built. The export drops a private root by itself [read: `public_boundary.py`'s docstring]. `Security/Proofs/lean-audit.json` needs none: its root `Proofs.Properties.NetworkPeriod` covers the module by prefix [read: `check.py`'s `under`].
5. Nothing under `tools/` changes; no line goes into `public_boundary_allowed.txt`.
6. The other choice is to leave the K lemmas off this branch until the check's change, which is private too. I did not take it because item 4 of the brief asks for them.
7. Then one lock run and the suites again at the new head. The lock run at the merged head, `r20261110-235911-82a4`, is still running; its verdict comes in the 8 PM report.
8. The same suite prints two bookkeeping warnings that are not mine: retired words in Lean names ('law': 148 > 133) and SPOT's older name. My six Lean files hold no such name [run: `retired_names` on them, empty]. I did not look at where they come from.

