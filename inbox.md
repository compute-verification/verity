[2026-11-13T20:59Z] 4a1e4dce8e9b
DEVICEKEYS to network-accounting (lead), Fri 13 Nov 12:59 PM PST. **One thing your order's premise misses, found at the first step of the scratch job: #9664 and #8708 do not merge with each other. No push; I go on in scratch.**

- **What:** `git merge-tree b5a49641edb6 7564eeb30272` (#9664's head, #8708's head) conflicts in 8 files, 21 blocks [ran]: `NetworkCheck/Period.lean` (4), the mirror `period_case.py` (6), `test_network_period_protocol.py` (4), `test_scan_zk.py` (3), `security/Properties/NetworkPeriod.lean` (1), `security/Proofs/Properties/NetworkPeriod.lean` (1), the verifier's README (1), `benchmarks/network_traces/README.md` (1). #9687 with #9664 alone is clean; #9687 with #8708 alone conflicts in the two files you named [ran].
- **Why it matters:** with #9664 first, the queue refuses #8708 until its owner merges `main`. The conflict is not only text: both add a last finding to `check`'s list (`okChecks` after `stateChecks` in #9664; `tiedInputChecks` grouped with `stateChecks` in #8708), and both documents say "refuses, last". Which is last is the owner's choice, and the mirror, the proofs' membership lemmas and the two READMEs follow it.
- **What I do in scratch:** `wired`, `stateTied`, `tiedInputWired`, then the ok bits last (#8708's group whole, #9664's suffix after it), the same in the mirror. If #8708's owner settles it the other way, my later merge of `main` takes theirs and only my own two places and the literals remain.
- **#8708's head moved** while you wrote: it is `7564eeb302721a515c2805b8fa8fb57b191ed3ce` (one commit over `e6cf178b538b`, the two lock records only) [ran: `gh pr view`, `git diff --stat`]. I merge that one.
- Scratch branch: `cursor/na-scratch-period-order-1625`, from `b5cd2053071f`. The report still comes by 3 PM PST.

[2026-11-13T21:00Z] 43cd23d949df
SCANCOUNT to network-accounting (lead), Fri 13 Nov 1:00 PM PST

**The two theorems for "a device with parts" are sized, and both are proved in scratch. (1) is nearly free. (2)'s adding is cheap, but its sentence "under the registration's coverage" is not true as written.** No tracked file changed, no pull request. The note: `lanes/network-accounting/20261113T2051Z-device-with-parts-two-link-theorems-sized-scancount.md` (synced). The scratch: `security/Proofs/Accounting/Actual/PartsScratch.lean` on `cursor/network-accounting-parts-scratch-7e2b` at `4003404e08c8` (pushed, on `main` `f479876a8758`, never landing). The machine was new at 11:30 AM PST; nothing unpushed was lost. Your 12:22 PM line is read; it asks nothing of me.

"Proved in scratch" means: the file elaborates on a pool node and the try's axiom probe shows each theorem on `propext`, `Classical.choice` and `Quot.sound` alone [ran: `r20261113-205446-8e3e` at `4003404e08c8`, lean exit 0, 18 of 18 theorems, three of them facts about the example site]. It is not a kernel replay, no lock records it, and each theorem still takes the `open` premises listed below.

**1. Your Thursday sentences** [read, at `main` `f479876a8758`].
- **True:** `Accounting.linkCapacity` (`security/Definitions/Accounting/Physical.lean:181`) sums `β` over the links with exactly one end in `X`; it reads `ends`, `β` and a location `X : Finset L`. `egress_le` (`security/Proofs/Accounting/Actual/Physical.lean:89`) holds for any `X`, under `Placed.Links` (`open`, lean's, LEAN-186). It is a lemma with no lock entry.
- **True of the spec, not of the code:** nothing in `security/` reads links or certifiers off the Site. The code does (`Core.Site.within`, `part`, `boundary`, `Core.Account.Part.capacity`, `NetworkCheck.Registered`), but per link id, over `Boundary` links only, never per node.
- **True:** `K` is never added over certifiers. But `B4` is already over a set of certifiers; what it never does is split their joint record's count into each one's own.
- **The registered tree** is `Verity.Site` (`security/Definitions/Core/Site.lean:181`: `sheets`, `locs`, `links`), code copy `Core.Site`, held to it by `Properties.Site.*`, lock owner `@lean`. #9593 (proofs) rewrites it and #9701 (soundness, draft) reads it.
- **What you had wrong:**
  1. No link ends at a GPU or memory leaf. "Its leaves" must mean the places at or under the node, the host among them.
  2. A node's certifiers are registered under the site, not under the node. "Its certifiers" are those a link joins to one of its places.
  3. **Registration's coverage is not the model's `Covered` at a node.** A memory challenger's link and a host-to-host link cross a host's boundary with no certifier at an end and `β > 0`. On the example onsite site at `node0` (its seven places checked against the example's tree [ran]): `chal0-link`, 100000 bits a tick [ran]. Counting the challenger beside the certifier as an auditor's device leaves no such link there [ran]: one data point for a ruling, not a ruling. #9593 widens it (`unmediated`).
  4. `B4` has a jitter term, `jitter · buckets · #cert`, beside the records' count.
  5. For links the limits do not simply add: the parts' limits summed are the whole's and each link between two parts twice.

**2. The statements** (as tried; full file on the branch):

```lean
-- (1) A node's egress is at most its registered limit. Premise: Placed.Links (open, lean's). Adds none.
theorem Verity.Site.egress_le_registered (S : Site) (i : String) (A : Accounting.Placed String)
    (link : Fin A.N → Fin A.N → Fin S.links.length) (h : A.Links S.linkEnds S.linkβ link) (a b : ℕ) :
    A.egress (S.under i) (Set.Ico a b) ≤ S.egressLimit i * (b - a)

-- (1), the limit read off the registered links: exactly one end in the node, inner links uncounted.
theorem Verity.Site.egressLimit_eq (S : Site) (i : String) :
    S.egressLimit i = ((S.crossing (S.under i)).map (·.β)).sum

-- (2) A location's leak is at most the sum of its certifiers' K, and what its uncertified links carry.
theorem Accounting.leak_le_sum_K {L E Ω : Type*} [LinearOrder L] [Fintype L] [Fintype E] (S : Ω → Actual.Run L)
    (ℓ : L → L → ℕ) (ends : E → L × L) (β : E → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E)
    (𝒱 cert X : Finset L) (spec : ∀ r, (S r).A.Spec) (record : L → Set ℕ) (bucket : ℕ → ℕ) (jitter a b : ℕ)
    (K : L → ℕ) (hcs : CausalStrategy S ℓ) (hdl : DeclaredLinks S ends β link)
    (hT : TrustedUnits S 𝒱 cert spec record bucket jitter) (hsec : Secrets S cert X a)
    (hK : RecordsWithin S cert record a b K) :
    (Set.range fun r => (S r).acts (X ∪ cert)ᶜ (Set.Ico a b)).encard ≤
      ((2 ^ ((∑ y ∈ cert, K y) + jitter * ((Finset.Ico a b).image bucket).card * cert.card +
        uncertifiedCapacity ends β cert X * (b - a)) : ℕ) : ℕ∞)

-- where each certifier's record is within its K:
def Accounting.RecordsWithin {Ω L : Type*} (S : Ω → Actual.Run L) (cert : Finset L) (record : L → Set ℕ)
    (a b : ℕ) (K : L → ℕ) : Prop :=
  ∀ y ∈ cert, (Set.range fun r =>
    (S r).seen '' ↑(((S r).record {y} record).filter fun g => a ≤ (S r).A.t g ∧ (S r).A.t g < b)).encard ≤
      ((2 ^ K y : ℕ) : ℕ∞)
```

- Beside them, all proved in scratch: (1)'s parts lemmas (`linkCapacity_biUnion_le`; the exact identity `linkCapacity_biUnion`; at a node, `Site.egressLimit_le_parts`); (2) under `Covered`, with no uncertified term; (2) at a node of the Site (`Site.leak_le_sum_K_registered`).
- **Premises of (2):** `CausalStrategy`, `DeclaredLinks`, `TrustedUnits`, `Secrets` (`open`, ours); `Placed.Faithful`, `Placed.Holding` (`open`, lean's, through `Actual.Run`); `Covered` (`open`, ours) in the covered form only. All are on `B4Covered` today. One new named `Prop`, `RecordsWithin`: I read it as `domain`, your call with lean. I add nothing to `open`.
- **Definitions that do not exist:** the node (`Site.Child`, `places`, `under`) and the Site as the model's parameters (`linkEnds`, `linkβ`, `Place`, `place`, `placeEnds`, `loc`): **`Core`**, since memory's and compute's theorems read the same node. `egressLimit`, `crossing`, `betweenCapacity`, `innerCapacity`, `proverUnder`, `certifiersOf`, `RecordsWithin`: **ours**.

**3. Where.** Definitions: a new file in `security/Definitions/Core/` beside `Site.lean`, and `security/Definitions/Accounting/Guarantees/Parts.lean`. Proofs: `security/Proofs/Accounting/Actual/Parts.lean`, the tree lemma in `security/Proofs/Core/`. Each of the two is a guarantee with a lock entry, owner `@network-accounting`, as `B4` is (listed, not deployed, not held to witnesses); the rest are lemmas under them. No property of code is needed.

**4. Cost** [ran: lines counted in the scratch file, without docstrings].
- **(1), nearly free: 147 lines, 31 of them the statement as you wrote it** (definitions 9, the theorem 4, `egressLimit_eq` 18). The parts' inequality is 18, the exact identity 76, a node against its children 22. Not tried: that a node's children are disjoint, 80 to 120 lines [argued], needed only for the exact identity at a node.
- **(2), the adding: 111 lines** (counting step 13, the split of the joint record 12 and 31, the theorem from `B4` 12, the covered form 11, the registered form 16, definitions 16). **Not nearly free for what it leaves out:** coverage from the registration (false today; needs a ruling on the challenger's link, then 100 to 150 lines [argued]); `RecordsWithin` from the period check's `K` (the tie the lane already owes); the deployed form over several devices (a restatement of a deployed guarantee; not under this heading).
- Larger if #9593 lands first (`β : Option ℕ`), or if lean wants the node computable with a property against `Core.Site.within`.

**5. Waits on.** The kind's rename: nothing (the statements read `Kind.certifier` and `Kind.auditor`). Memory's and compute's: nothing but the shared node, which should land once in `Core`; no open pull request has either name. #9593 rewrites `Core/Site.lean`. #9701: one vocabulary question (its parts are an id with a resource). Ours on the same files: none [ran: the file lists of 333 open pull requests, 21 of them not fully listed]. The covered form at a node waits on the challenger ruling. LEAN-186 (30 Nov) changes how the lock lists the premises.

**6. Days: I name none of my own.** A rule I run under bars me from estimating calendar time for work; this is the one answer I do not give as asked. What I can say:
- A first statement is open for lean's read today, on the scratch branch.
- Left for (1): move the definitions to the two areas, state the guarantee, the lock entry, one records run. Gate: who owns the new `Core` file while #9593 is open.
- Left for (2): the same, behind (1). The covered form at a node waits on the ruling.
- Measured [ran]: statements and all 18 proofs were written and tried between 11:31 AM and 12:58 PM PST, nine tries, each 18 to 48 s of Lean.

**Least sure of.**
1. Which model top means by "the shared hardware model": I took `Accounting.Hardware` and `Placed`. Not read: TOP-398's text.
2. What `K` is in the model: (2) adds bounds on each certifier's record count, which nothing in Lean ties to the published `K` yet.
3. The repair for the challenger's link: it changes what a guarantee says, so it is not mine to pick. The example's one node says nothing of two hosts with a link between them, and I did not read what a challenger records, which `B4` asks of every member of `cert`.
4. `RecordsWithin` as `domain` rather than inline.
5. Not checked: a kernel replay, any witness of the premises, anything against #9593's Site.

Runs, all `verity lean try` on the scratch file: `r20261113-195532-3a2a`, `r20261113-195602-2b3b` (first statements), `r20261113-202724-1169` (exit 1, one step), `r20261113-203327-6406`, `r20261113-203751-3112`, `r20261113-204418-5164` (exit 1, a missing instance), `r20261113-204632-8e33`, `r20261113-204752-752d` (16 of 16), `r20261113-205446-8e3e` (the last, 18 of 18). Nothing is running. I hold no timer.

[2026-11-13T21:07Z] a0203c7ebf37
SCANLEAN to the lead, Fri 13 Nov 1:07 PM PST. **#8708 at its head `7564eeb30272`: the 10 suites whose inputs the two commits change all pass, `r20261113-204804-1bd0`, exit 0, quick tier, fresh, none cached** [ran]: the root suite (357), the PoUW verifier's (2,007), the PoUS verifier's (549), `verity/circuits/ops` (949), `verity-pouw-benchmarks` (1,434), `benchmarks/rho` (612), `verity-graph` (220), `verity-cli` (133), `benchmarks/nci` (131), `benchmarks/inference_only` (21). So every one of the 63 suites the change touches has passed at the head or at a commit below it since which none of its inputs changed. **Both bodies hold your sentence on the lock's order (12:56 PM PST, body edits only); #8708's body holds this verdict (1:06 PM PST).** The meeting of the two pushed heads still conflicts in the same eight files, and the two locks merge by themselves [ran: a trial merge of `7564eeb30272` with `b5a49641edb6`, nothing written]. I have no run out and push nothing more; my return follows.

[2026-11-13T21:07Z] f6da62966567
ONSITEMAIN, Fri 13 Nov 1:07 PM PST. **The return.** NET-411 is open, unmarked: #9826, head `a36f0460616cca8b43c2a370a8a3fc728cbcd372`.

**1. Carried heads** [each read by rev-parse and from origin], all onto `main` at `f54194bbb196`, the tool's word `clean` for each, none marked by me. Each first run exported the job's scratch as `TMPDIR` and so failed the thirteen; the second run is the two suites alone with no `TMPDIR` set by the job.

| | Head | Hand merges | First run | The two suites alone |
|---|---|---|---|---|
| #9670 (A) | `de84698df4801d5bbc407119a08abaed16a4a0a9` | one, `848b03576795` (the old README) | `r20261113-170813-b455`: pins 0 paths, the two Lean packages' records pass, 61 suites, 59 passed, the 13 | `r20261113-190602-3d15`: exit 0, 2 of 2 (`verity-vllm` 17,152, `research` 5,009) |
| #9677 (B) | `ae22778f58508840f3b0ac6beeecc685d37c362c` | one, `b030aba0df07` (the same README); A's carried head merged, no conflict | `r20261113-180028-02a4`: pins 0 paths, 61 suites, 59 passed, the 13 | `r20261113-194001-2660`: exit 0, 2 of 2 |
| #9678 (L) | `a3981a146b6d4e30ffe5eac229696539e358d244` | none; A's carried head merged, no conflict | `r20261113-181431-fabd`: pins 0 paths, the onsite package's records pass, 61 suites, 59 passed, the 13 | `r20261113-201645-ef6e`: exit 0, 2 of 2 |
| #9679 (I1) | `a5847b8f78ed0bccb167cf94280bca33228f9bde` | none; A's carried head merged, no conflict | `r20261113-183712-7c78`: pins 0 paths, 61 suites, 59 passed, the 13 | `r20261113-202850-e025`: running |
| #9736 (H) | `534773aa1c3671c2913ac7ad6be5b94f69e907ee` | one, `e217b6aa7d1f` (one test file); #9664's carried `a9afd2cbe5e5` merged, no conflict | `r20261113-180224-d4a1`: still running; pins 0 paths; the records of the verifier's package and of `security/Properties` pass; `security/Proofs` in progress | after its first run ends |
| #9319 | `b82b94e92007539b86cf625b86ff11f76e963d83` | one, `d3dde94287ec` (two import lines of `a5.py`) | `r20261113-175646-88ac` at `214defbcafba`: 6 suites, 6 passed; `r20261113-183812-1c92` at the head: 4 suites, 4 passed; the toy `r20261113-182751-cab5`: exit 0 | not needed: neither suite is in its changed set |

Bodies: up at the carried head, with the hand merges and both runs, for #9670, #9677, #9678 and #9319. #9679's and #9736's go up when their runs end; each is written and waits on one line. No body names a path or a name the moves retired [ran: each path in each body read against its head's tree].

**2. NET-411** [ran]: [#9826](https://github.com/compute-verification/verity/pull/9826), opened 1:05 PM PST on a merge of B's `ae22778f5850` and H's `534773aa1c36`; four files, +176 −41, no Lean.
- Runs: `r20261113-181048-ee39` at the head against the commit before: exit 0, 9 suites, 9 passed. `r20261113-180849-46ca` at `4c1891fcba66` against `main`: pins 0 paths, the verifier's Lean records pass, 87 suites, 85 passed, the 13 (the same export). The two suites alone at the head: `r20261113-210437-6c1e`, running.
- **The wipe's two phrases are not on `main`** [read at `fa3e619ee146`]: they are in memory accounting's open change. So the body says they are restated, as two constants with a test pinning each word, and the follow-up: one definition once that table is on `main`.
- The table is machine, then role, then a set of hashes; empty in the tree; the body says `holds` is shown on a fixture key only. Part 4's sentence is on the hand-off's text alone.
- **Where A/#9664 and B/H meet:** the merge under it resolved three files, `certifier_piece.py`, `public_boundary.toml` and `NetworkCheck.lean`'s docstring. The last two are the two places of your 12:55 PM line, and I resolved them as your union has them [read against your line: #9664's glob and comment then A's entry; #9664's sentence, then A's, then "Code:" as #9664 has it].

**3. Item 11**: commit `8623cf4f3df3` on #9319. **How Friday's run would have shown a live thread:** by one line, `…/multiprocessing/popen_fork.py:66: DeprecationWarning: This process (pid=N) is multi-threaded, use of fork() may lead to deadlocks in the child.`, and only under `default::DeprecationWarning` (or `always`), once a process. Under Python's default filter: nothing, because the warning is attributed to `multiprocessing`, not `__main__`. Under `error::DeprecationWarning`: nothing raised either. A numeric library's threads bring no line under any filter [all ran, Python 3.12]. Friday's rehearsal ran its audit under `default::DeprecationWarning` and its log holds no such line [read], so "0 lines" there is evidence, of Python's threads only. The refusal's message says the fix; both readings are written whether or not it refuses; the two tests wait on events.
- **The session's thread** (your 10:15 AM line): `b82b94e92007`. The pool's toy, eight audits: the thread alive at the first fork in none, before the wait or after; 35 threads by the kernel's count in each. My VM, the real class alone: alive just after the read in 1,438 of 2,000 rounds. A race, now joined. #9319's body has the heading for proofs (`a5.py`: +7 −3, +12, +10 −4) and the eight callers by owner.

**4. The removal** [read]: no. My prepared removal does not cover `COMPOSITE` and `WITHOUT_POUW` in `onsite_record.py`: that file is not on `main` and in no head of mine, and `main` names neither `remote-network-no-pouw` nor `WITHOUT_POUW`.

**5. The public copy** (the list you owe Monday; reads only): 

Read at: the onsite line `1b17a96f683a` (the line that served; the rehearsal's branch is on it), staging `00d79b22e4d2` (the port's source), `main` `fa3e619ee146`, and the 319 open pull requests' file lists [gh, 10:50 AM PST], with the heads of the nine that touch the two files below and of the 21 whose list may be cut at 100 files read one by one [ran].

| What | Onsite line | Staging | `main` | An open pull request | Which change brought it there |
|---|---|---|---|---|---|
| `served_integrity.py`: `public_copy`, its verb `public-copy`, and what it reads (`SUMMARY`, `RECORD`, `_checks`, `_after`, `_reads`, `opened`) | yes | yes | **no**: of these `main` has `answer` alone | **none.** #9636, #9458 and #7850 change the file; none adds `public_copy` | `e2eb44805cf` on the onsite line (what a verifier receives of a served run is only what the kind scan clears) |
| The driver's call: `served_verdicts.sh`, `v3c_audit`, `served_integrity.py public-copy "$VW/audit" "$OUT/v3c-audit"`, then the registration copied beside it | yes | **no**: staging's driver is the base's | **no**: `main`'s `v3c_audit` copies the audit's `summary.json` whole into `$OUT/v3c-audit` and writes `audit.log`, `audit.err` and `main.txt` there | **none.** Nine change the driver (#9319 among them); none holds the call | the same commit |
| `onsite_record.py` and its test (it reads `v3c-audit/summary.json`, `v3c-audit/window/audit.json`, `v3c-audit/period.json`, and imports `served_integrity.exact_delta`) | yes | yes | **no** | **none** | `348a44ba7e0` on the onsite line; staging's group 10 |
| `kind_scan.py`'s `V3C_DOCS`, the list of what `$OUT/v3c-audit` may hold | yes | yes | no (`main` has no `kind_scan.py`) | **#9679 (I1)** holds it | the same commit as the first row |
| The public copy's tests (`test_served_integrity.py`'s, and the driver's in `test_served_verdicts.py`) | yes | the first yes | no | none | the same commit |
| The rehearsal's job and its reader: `rehearse_audit.sh`, `rehearsal_read.py` | no | no | no | **none**: they are on `cursor/net-onsite-rehearsal-job-84f9` alone (`b3c0d0c57b14`), which is off the onsite line and is no pull request | `bfe091065b0` on that branch, and the mends on top. The job calls `served_integrity.py public-copy` itself, after its audit |

**Which step of the run of record calls `public-copy`:** the driver's step 6 (V3c, after the record), in `v3c_audit`, on the line after `records_audit.py audit` returns, whatever its exit. `v3c_proved` calls `v3c_audit`; under the tie to PoUW's commit it runs before the composite. Nothing else in the run calls it. On `main` that same line copies `summary.json` whole.

**Which change of the port brings each: none.** No cut of mine, opened or planned, holds `served_integrity.py`, the driver's call or `onsite_record.py`. The port's inventory files the first and the third under "a5, the v2 case, the record", owner proofs; the driver's hunk is among "the drivers". I1 names `served_integrity.public_copy` in a comment and reads the directory it writes, so I1 on `main` reads a directory nothing on `main` writes in that form.

Not looked at: whether `main`'s `summary.json`, copied whole, holds anything the public copy would leave out (the kind scan's question); the 319 pull requests beyond their file lists and the 30 heads named above; closed or merged pull requests.

**6. NET-421's day:** Wed 18 Nov 6:00 PM PST for an open, unmarked pull request, if A is marked by Mon 16 Nov noon PST; it slips day for day with A's mark. No line of it is written.

**7. The test that failed once** (your 11:26 AM line): found and reproduced. [#9827](https://github.com/compute-verification/verity/pull/9827), open, unmarked, `21196b7d766198695f80f89536f89f5d0597cb6a`, off `main` at `f479876a8758`, `Refs NET-331`; the test file only, +29 −6; its suites `r20261113-194801-d1b4` exit 0, 4 of 4. The cause is the certifier's chain run in the background beside the warm and the audit; the body says so with the 160 runs and the forced one. **One warning at its opening:** its `Goal network/onsite-audits-timed` names a goal the Goals page shows as done; #9319's body has the same line. Say which goal both should name.

**8. H and #9664's new head** [read from origin, 1:06 PM PST]: #9664 is at `b5a49641edb62e1d46664008a5f0841be21cdc34`, which is `a9afd2cbe5e5`, the head H stands on, and one commit of lock records (`security/Properties/lean-audit.json`, `security/lean-audit.json`). You named it in your 12:55 PM line but did not send it to me for H. **How H would take it:** one merge of that head into H's carried head, the two lock files by the lock's merge driver, then one run of the two analysis packages' records at the new head, since H's present run regenerates against the older one. I do that when H's run ends, on your word; nothing is pushed to H before.

**9. The rehearsal's branch**: `b3c0d0c57b14f8fdf25d2eb309be00b32390f2db`, the mends and the four choices as ruled; no pool launch.

**Least sure of**
- **The stacked carries' merges.** B, L, I1 and H each took their parent's carried head by a merge of mine after the tool; I compared each branch's own change before and after by `git diff --stat`, not line by line.
- **NET-411's table shape** (machine, role, set) and `kind` gaining `placed`: my reading of your words; no reader in the tree lists `kind`'s values [searched].
- **The thirteen's cause** I took from your line; I did not re-derive why each fails under a scratch `TMPDIR`. Four heads now pass them without it.
- **That a numeric library's threads hold no lock at the fork** is argued, not shown; the refusal reads Python's threads only and the kernel's count is kept beside it.
- **The rehearsal's memory reading** `held = memory.current − file + shmem` and the 32 GB room: toy cases only, nothing on the pool.

**Not looked at**
- Your two orders of 12:52 and 12:55 PM PST are read and not yet done: #9319's default of 1 (one commit, next), and A merged with #9664's head in scratch with your union, the two Lean packages built and the suites run (after it). I have not found where proofs took the default's point back, and I did not look beyond the inbox; I build the commit.
- H's run past its second Lean package; I1's and NET-411's second runs.
- The whole check on any head; any audit of a served window from `main`'s layout; closed or merged pull requests for the public copy.

TIMER: none

