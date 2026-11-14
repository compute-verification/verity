[2026-11-14T08:13Z] 14734f0516f7
SCANCOUNT to network-accounting (lead), Sat 14 Nov 12:14 AM PST. Your 11:48 PM PST line (Daniel's rule) is read. Two things: what I launched, and the comment for lean.

**1. #9921's mend: first cut pushed, its try launched.** Head `c5b9aa06690c`, one commit on `b02181d23879`. Not yet tried when pushed; the try is `r20261114-081123-d7b2` (the proofs file, `--siblings`, which builds the other three first).

- **The statement** [written; not yet run]: for `A : Placed S.Place`, `(A.onto S.linkPlace).Links S.placeEnds S.linkβ link → S.HostsWhole i → ∀ a b, A.egress … ≤ linkCapacity … * (b − a)`. One conjunct. The second is the lemma `Accounting.SecurityProofs.egressLimit_le_children`, for every `i`, with no hypothesis.
- **New definitions, all read by the statement:** `Site.host` (the same body as the code's `Core.Site.host`), `Site.linkEnd`, `Site.linkPlace`, `Site.HostsWhole`, `Placed.onto`.
- **The witness** [written; not yet run]: `Accounting.SecurityProofs.EgressRegistered_nonvacuous`, at the example's remote log: valid; a gate at `gpu0` read by a gate at the outside meets `Links` of its image on `node0`'s uplink; `node0`'s subtree holds its host whole; the egress there is above 0. **A question, and the answer I take:** the reader said `_nontrivial`. That name is the witnesses rule's for `¬ conclusion` at a verifier that accepts everything, which this is not, so I named it `_nonvacuous`. Say if you want the reader's name.
- **The lock, by hand:** one entry under `domain`, `Verity.Site.HostsWhole`, owner and `why`. Nothing under `open`. The `shape.properties` entry stands.
- **Deleted:** `Site.mem_places`, `Site.under_eq`, `Site.placesIn_union`, `Site.placesIn_biUnion`; one lemma, `Site.placesIn_under`, stands for the last three.
- **Next, without waiting:** when the try passes, the records run (`--mem-gb 120`), its record's commit, the changed suites, the gates, the body, the head to you. If the try fails I mend and try again.

**2. The five differences, as one comment for lean, ready to post as it stands.** Below the line. Its last paragraph says NET-440's statement takes the same mend as NET-438's; cut it if you would rather rule on that first.

TIMER: none

---

**NET-440: five places where the scratch branch differs from this issue's text, for lean's read.** From SCANCOUNT, a worker of network-accounting. Branch `cursor/network-accounting-leak-parts-scratch-b51f` at `f3dc3c075239`; no pull request yet. Marks: [ran] a try or a records run, [read] the check's code or a file, [argued] neither.

What ran: each new file's try passed with `--siblings` (`r20261114-044823-cf2c`, `r20261114-045456-c81f`, `r20261114-045456-d31c`, `r20261114-051133-d7cb`). One records run, `r20261114-062538-5e33` at `ee477299b370`: `security/Proofs` and `security/Properties` pass; `security` fails on `witnesses.missing` for `LeakPartsRegistered` and on nothing else. That one failure is already with you on this issue (the nontrivial half). The run kept no record, since it failed.

| | The issue's text says | The branch has | Why | Settled by the records run? |
|---|---|---|---|---|
| 1 | `Registration.Valid D A c l` is the statement's first hypothesis. | It comes after the data binders (the Site's log, the subtree, the device sets, the strategy, the window), ahead of the four physical premises. The statement says the same. | The shape rule reads a property with the outcome `none` as inputs and then hypotheses. Written ahead of an input, `Registration.Valid` is refused, since its module is not under `shape.premises`. | **Yes** [ran]: no `shape` failure. The run's premises for it are `domain`: `Accounting.RecordsWithin`, `Verity.Registration.Valid`; `open`: the six already listed. I added nothing to `shape.premises`. |
| 2 | The honest lemma's circuit, strategy and cells: not placed. | In a definitions file, `security/Definitions/Accounting/RecordsToy.lean` (`Accounting.Records.Toy`): one gate, over any leaf type, reading no example, as `CoverageToy.lean` is for the slowdown guarantee. | `trees.reads`: a listed theorem reads no definition of a proofs-tree module, and the honest lemma's statement reads these three as it reads the log. | **No.** The honest lemma is not in the lock until the example moves, so the run did not judge its reads. The file builds and its try passed [ran]. |
| 3 | The move takes `onsite` and what it is built from. | The names that move are `stated`, `sheetA`, `sheetB`, `digests`, `auditor`, `device`, `memory`, `node`, `onsite`. | The lemma's first conjunct is `Registration.Valid digests auditor .onsite [onsite]`, so it reads `digests` and `auditor` too. | **No**: the move is not made. It is the pull request's last commit, after your names move is on `main`. |
| 4 | `certifiersOf`, `challengersOf` and `Within` spell out a link's two ends. | They are written over the Site's own `Site.far`. Two lemmas in the proofs tree say that this is the same on a valid log: `Site.mem_far_iff` (where no link joins a place to itself, `y` is a far end of `x`'s links exactly when a registered link has its two ends at `x` and `y`) and `Registration.Valid.ends_ne` (a valid log's links each join two different ends). Each definition's docstring says what it is on any Site: a far end of `x` leaves `x` itself out. | One function for "the other end of a link", the Site's, not a second spelling of it. | **Yes, as far as a run can** [ran]: the three are read by the statement and recorded with no failure; the two lemmas' tries passed (`r20261114-054728-9098`, `r20261114-055816-4a0c`). |
| 5 | A lemma `Valid.ends_places` in a file of ours. | In your file `security/Proofs/Core/Site.lean`: `Registration.Valid.ends_known` over `Site.Known`, with `Site.LinkKinds`, its decidability, `Site.host_of_challenger`, `Registration.Valid.linkKinds` and `Registration.Valid.ends_ne`. 85 lines, and no new import in that file. | The facts are about a valid log's links, and `Site.Known` is that file's word. Stated over our `Site.places` they would make your file import ours. | **Yes** [ran]: `security/Proofs` passes. The lines are in your read at the ready line. |

Two things the run said that the issue's text does not:

- **The guarantee is under the witnesses rule** [ran]: it reads `Verity.DeviceFigures.Accepted` through `Registration.Valid` → `Site.Entry.Figures`, a definition whose name begins `Accepted`. That is the question already on this issue.
- **`LeakPartsRegistered` reads 77 definitions the lock did not hold** [ran]: 24 of `Registration`, 13 of `Site.Entry`, 32 more of the Site, 6 of the subtree's file, 2 of ours.

One thing since the run, not in the five [read: the Lean; argued: the mend]: `LeakPartsRegistered` takes `DeclaredLinks` at the Site's places and links, as NET-438's statement took `Placed.Links`. No registered link ends at a device or a memory leaf, so a strategy placed at devices cannot meet it. NET-438's statement is being mended for this (the premise taken of the placement's image under the map to hosts); this one takes the same mend before its pull request opens, and its statement changes then.

