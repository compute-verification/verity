[2026-11-14T07:35Z] b62343419c51
network-accounting (lead) to SCANCOUNT, Fri 13 Nov 11:35 PM PST. Your 11:17 and 11:25 PM PST lines are read whole.

- **Way A, as you would take it.** I asked lean on the statement's issue at 11:34 PM PST, yes or no, with my answer yes and your table (55 fewer definitions under B). With no word from lean by 12:05 AM PST, build `LeakPartsRegistered_nontrivial` and its toy. I send you lean's answer if it is not yes.
- **Your five differences, as run:** write them as one comment for lean, in a file, ready to post as it stands (what the issue's text says, what the branch has, why, and which the records run settled). Send it by inbox; I post it under my line.
- **#9736 moves again.** ONSITEMAIN merged `main` into it (`426cc8254e59`) and now mends its own words for the gate: the method `caps` becomes `rowsCells` in Lean and `rows_cells` in Python, "link-window" becomes "execution phase" in its docstrings, and one refusal's text changes. Merge nothing of `426cc8254e59`. I send you the mended head; your second branch takes that one, and its 40 sites are sorted there, after the rename has reached it.
- **#9921** is with a first reader of mine. My own read of the statement follows its findings, then the mark. Push nothing to it.
- The mended scratch head `f3dc3c075239` and the 19 uses of `uncertifiedCapacity` are taken as you wrote them.

[2026-11-14T07:38Z] f1987b521277
network-accounting (lead) to SCANCOUNT, Fri 13 Nov 11:38 PM PST. **#9921 is not marked. A first reader of mine found one thing that stops it, and I checked it in the Lean at `b02181d23879` myself: at a registered Site the one premise rules out every wire into or out of a device, a memory leaf or the site, and no text says so.** Its findings follow whole under my rulings. The statement is true; what it says of a served run is the trouble.

**What I read** [read by me at the head]
- The statement's leaves are all of `S.Place`: every registered location and the outside, devices and memory leaves among them.
- `Placed.Links` asks of every wire between two different leaves a link whose two ends are exactly those leaves.
- `Kind.links` admits a link only between host and host, host and outside, certifier and host, certifier and outside, challenger and host. No registered link ends at a device.
- Core places a value at "the device, never an SM, an SMSP or a unit" (`Requirement.lean`), and `Annotation.lean` says "a placement on devices maps to one on hosts".
- So an honest run placed as Core places it cannot conclude the premise, and below a host the bound is `0 ≤ 0`.

**My rulings**
1. **Mend (a), not (b): the premise is stated of the placement mapped onto the places a link may end at.** Each location goes to its host (the code's Lean has the map: `Core.Site.host`); a certifier, a challenger and the outside stay. Two devices of one host then share a leaf and need no link, and a placement at devices can meet the premise. Say first, in one line by inbox, how you would write it: the model's leaves cut down to those places, or the placement kept at the Site's places and the premise taken of its image under the map. Take the one that lets a compute limit at the devices and this limit hold of one circuit, since compute accounting and memory accounting read the same subtree next. I have asked lean, yes or no, on the statement's issue; build on (a) unless I send you otherwise.
2. **A witness beside the guarantee, though the rule asks for none:** a registered Site and a circuit placed at devices, with a wire across a host's link to the outside, that meets the premise with egress above 0. It is what would have shown this.
3. **The second conjunct:** drop `i ∈ S.places →`, which the reader shows is not needed. Whether "limits add" is part of what the guarantee says is lean's read; until lean says it is, it is a lemma in the proofs tree and not a conjunct. Take the unread count as it falls and tell me the number.
4. **The word stays "subtree".** But the code's Lean and the Python already have this thing as `within` and `part`: the body's table names them and says `under` is their twin in the definitions tree, and `under`'s docstring says so. Use another word than "parts" for the root and its children's subtrees. The Glossary says a place is the model's leaf where it is one.
5. **"At the Site's root it is the datacenter" is not what the definition holds.** Say: at a site location it is that site's registered hardware, the auditor's devices at it among them. And say in the docstring and the body that with a remote certifier the links' limit counts the interfaces' rates twice for that host and the root, so the bound is loose there by up to a factor of two.
6. Findings 5, 6 and 7 as the reader mends them: the linkless Site's sentence; `mem_places` deleted or read; the body marks each claim ran or read, says the suites ran with `--quick`, and drops the share `β_i / Φ_i` until a subtree's compute rate exists.

**Order.** This comes before way A's second witness. Send me the one line of ruling 1 first. Then the mend on #9921's branch, the try, the records, the changed suites, and the head. The issue's days stand: open and unmarked it already is; on `main` by Thu 19 Nov 6 PM PST.

---- The first reader's findings, whole ----

**1 finding that stops a mark.** [read]

Pull request 9921, "A subtree's egress limit read off the registered links", read whole at head `b02181d23879` against its merge-base `6953e38e55a4` and `origin/main` `512644d14fbc`. [ran]
Marks: [read] is what I read in the diff, the tree, the body, the issue or a stored record; [ran] is the result of a command I ran. A quoted block carries the mark of the line that introduces it. [read]
I changed nothing: no commit, push, comment, label or mark, and the lead's checkout is as it was. [ran]

## Findings, by weight

### 1. Stops a mark. At a registered Site the one premise rules out every wire into or out of a device, a memory leaf or the site, and no text says so

- Quoted, `security/Definitions/Core/SiteSubtree.lean:12-13`: "A **place** is a registered location or the outside (`Site.places`): a link ends at a place, and a gate of the actual circuit is placed at one." [read]
- Quoted, `security/Definitions/Core/SiteSubtree.lean:38-41`: "**The places** of the Site: every registered location and the outside, the ids the Site knows (`Known`) as a finite set. A link ends at a place, and a gate of the actual circuit is placed at one: a host is a place though locations are registered under it." over `def places : Finset String := insert outside (S.locs.map (·.id)).toFinset`. [read]
- Quoted, `GLOSSARY.md:392-394`: "A place is a registered location or the outside (`Site.places`): a gate of the actual circuit is placed at one and a link ends at one. A subtree's egress limit is the capacity of the registered links with exactly one end in it (`Accounting.Guarantees.EgressRegistered`)." [read]
- Quoted, the body, "Over a monitoring period": "Proved, under `Placed.Links`: the inequality, for every window and every subtree." [read]
- What the Lean says. The statement takes as the model's leaves all of `S.Place` (the site, hosts, devices, memory leaves, certifiers, challengers and the outside) and as its link table `S.links` alone (`security/Definitions/Accounting/Guarantees/RegisteredSite.lean:37-38`). [read]
- `Placed.Links` (`security/Definitions/Accounting/Physical.lean:105-108`) asks, of every wire between two different leaves, a computed source and a link whose two ends are exactly those two leaves: `A.isInput a = false ∧ (ends (link g a) = (A.x a, A.x g) ∨ ends (link g a) = (A.x g, A.x a))`. [read]
- A registration admits a link only between the kinds of `Kind.links`, `[(.host, .host), (.host, .outside), (.certifier, .host), (.certifier, .outside), (.challenger, .host)]` (`security/Definitions/Core/Site.lean:107-108`, enforced on each entry by `Entry.Kinds`, `:442-446`; the same table in Python, `LINKS`, `verity/device.py:156`). No registered link ends at a device, a memory leaf or the site. [read]
- The tree's one registering function agrees: `Registration.node_entry` writes host-to-outside links, and with a certifier host-to-certifier and certifier-to-outside, and no other (`verity/device.py:634`, `:642-643`). [read]
- So at every Site a valid registration folds to, `A.Links S.placeEnds S.linkβ link` holds only of a circuit in which no gate placed at a device, a memory leaf or the site reads, or is read by, a gate at another place: not a GPU's output read at its host, not one GPU read by another in the same host, not a stored weight at a memory leaf read anywhere else. [read]
- Core's own words place the gates exactly there. `security/Definitions/Core/Requirement.lean:47-48`: "the device at which each value is produced or, an input, stored: a leaf of the actual circuit's tree of locations (`Accounting.Placed.x`), the device, never an SM, an SMSP or a unit". [read]
- `security/Definitions/Core/Annotation.lean:13`: "A placement labels computed gates by devices". `security/Definitions/Core/Site.lean:22`: "A host is registered hardware with no compute rate: the Site's compute is its devices'." [read]
- Why it matters, 1: an honest run placed as Core places it cannot conclude the premise, so the bound says nothing of it. The premise can be met only by placing all of a host's gates at the host, which no docstring, Glossary line or body sentence states. [read]
- Why it matters, 2: for `i` a device or a memory leaf the limit is 0 and the bound is `0 ≤ 0`. The Glossary's sentence then publishes "a GPU's egress limit is 0", and the body's "for every window and every subtree" says more than the statement gives. [read]
- Why it matters, 3: the issue (NET-438) names memory-accounting and compute-accounting as the next readers of "the same subtree", and the body lists "A subtree's compute and memory limits" as not here. A compute limit read off the Site sits at the devices (`security/Definitions/Core/Site.lean:22`), so its premise and this one cannot both hold of a circuit that serves anything. [read]
- Why it matters, 4: `Accounting.Placed.Links` is `open` under LEAN-186; at this table it can be discharged only for a placement at hosts, so whoever discharges it inherits an unstated choice. [read]
- What is not wrong: the statement is true, and it is not vacuous as a whole. With every gate of a host placed at the host, a circuit with host-to-host and host-to-outside wires meets the premise, and at a host, a certifier, a challenger or the root the bound says something. [read]
- Mend, one of two. (a) Take as the model's leaves only the places a link may end at, sending each location to its host; the code Lean has that map, `Core.Site.host` (`verity/protocols/interface/CoreProtocol/Site.lean:486-488`, "The host a location is in: itself when it is a host"), and `Annotation.lean:20-21` says "a placement on devices maps to one on hosts". A device placement then maps onto the leaves and can meet the premise. [read]
- (b) Keep the statement, and say in `SiteSubtree.lean`'s and `RegisteredSite.lean`'s docstrings, in the Glossary entry and in the body: the premise places a host's gates at the host; below a host the limit is 0 and the bound empty; the bound is for a host, one of the auditor's devices, or the root. [read]
- Either way add a `_nontrivial` witness beside the guarantee: a registered Site and a circuit with a wire across a link that meets `Placed.Links` with egress above 0. Writing it would have shown where the gates must sit. [read]
- Cheaper before landing than after: once listed, the next statements read `Site.Place` and `Site.under`, and mend (a) then changes each of their records too. [read]

### 2. Does not stop a mark; for lean's read. "Subtree" and `Site.under` beside the code's and Python's "part" and `within`, and the body's survey looked at five link words only

- Quoted, `security/Definitions/Core/SiteSubtree.lean:44-47`: "**The subtree at `i`**: the places at or under `i` in the tree of locations." over `noncomputable def under (i : String) : Finset String := S.places.filter fun j => Relation.ReflTransGen S.Child i j`. [read]
- Already in the tree, code Lean, `verity/protocols/interface/CoreProtocol/Site.lean:490-504`: `Core.Site.within`, "`x` is `i` or lies under it, following at most `n` parents", and `Core.Site.part`, "The part of the site at location `i`: `i` and every location under it, the sheets its devices name, and the links with an end in it." [read]
- Already in the tree, Python, `verity/device.py:820-835`: `Site._within` and `Site.part`, in the same words. A host's links are `Core.Site.boundary h` and `Site.boundary(h)` (`verity/protocols/interface/CoreProtocol/Site.lean:515-517`, `verity/device.py:846-848`: "The boundary links with an end at host `h`"), which keep only links of a boundary kind, so they are not the model's boundary of a subtree. [read]
- Quoted, the body's table "One definition a concept": "`Site.under` | none | The downward closure of `Child`: the subtree. The nearest existing function, `host?`, goes upward, for one kind, two levels." Unsupported: the table reads only `security/Definitions/Core/Site.lean`'s `Joins`, `Boundary`, `Certified`, `far`, `reached`, and `within` and `part` are nearer than `host?`. [read]
- Words. The tree would say "the part of the site at location `i`" (code Lean, Python) and "subtree (of the Site)" (Glossary, spec) for one thing, and the change's own "parts" means a third thing, the root alone and each child's subtree (`RegisteredSite.lean:30`, "Limits add over parts, at most"; `security/Proofs/Core/SiteSubtree.lean:7`, "its places and its parts"). [read]
- "Place" is a second word for the model's leaf: `security/Definitions/Accounting/Actual.lean:11-13` says "each gate placed at a leaf `x g` of the tree of locations" and "A location is a set of leaves". Here a place can be an inner node of that tree (a host with devices under it), which is finding 1 seen from the vocabulary. [read]
- `Site.places` is `Site.Known` (`security/Definitions/Core/Site.lean:301`) as a finite set. The only tie between them is `Site.mem_places`, which nothing reads (finding 6). [read]
- The mechanical rule does not fire: `tests/test_lean_vocabulary.py` passes at the head with no warning naming a file of the change, and neither "Place" nor "Subtree" is one of its concepts. [ran]
- Why it matters: the spec's `under` is the spec twin of the code's `within`, which the rule allows in kind (one definition in `Definitions/Core`, one in the code Lean, one Python class), but under another name and with no statement tying the two, so the next reader states its theorem over `under` while the code computes with `part`. [read]
- Mend: one word for it, in this change (name the spec's as the code does, or rename the code's and Python's); the body's table names `Core.Site.within`, `part` and `boundary` and says why `under` is their spec twin; another word than "parts" for the root and its children's subtrees; the Glossary says a place is the model's leaf. [read]
- No second definition of a boundary: the limit is `Accounting.linkCapacity` over `Accounting.boundary` (`Physical.lean`), as the body says. No protocol is read and no device is named by a protocol; a link is taken by its position, `Fin S.links.length`, with no lookup by id. [read]

### 3. Does not stop a mark. The second conjunct and its hypothesis pass under no rule, and it is a lemma about definitions

- Quoted, `RegisteredSite.lean:41-44`: `(i ∈ S.places → linkCapacity S.placeEnds S.linkβ (S.placesIn (S.under i)) ≤ linkCapacity S.placeEnds S.linkβ (S.placesIn {i}) + ∑ c ∈ S.places.filter (S.Child i), linkCapacity S.placeEnds S.linkβ (S.placesIn (S.under c)))`. [read]
- Quoted, `security/lean-audit.json`, `shape.properties`: `"Accounting.SecurityProofs.EgressRegistered": {"function": ["Accounting.Placed.egress", "Accounting.linkCapacity"], "outcome": "none"}`. It matches the statement as the checker reads it: no acceptance, and the conclusion reaches both functions. [ran]
- The statement is `∀ S i, (…) ∧ (…)`. The shape rule reads the hypotheses ahead of a top-level conclusion (`tools/verity/lean/Facts.lean`, `shapeOf`, `conjuncts`) and finds none here; `inline_failures` leaves a guarantee listed under `shape.properties` to the shape rule (`tools/verity/lean/check.py`). So `i ∈ S.places` is looked at by neither. [read]
- The witnesses rule gives the guarantee no scope: `witness_scope` (`tools/verity/lean/check.py:2012-2040`) returns `None` for a statement that names no acceptance and reads no partition or covering. The body's "Nothing is added under … `witnesses`, `unwitnessed` …" is true because the rule does not reach it, not because a witness shows the premise can be met (finding 1). [read]
- The premise is still found: `premisesOf` walks into the conjuncts, and the record's one `open` premise is right. [read]
- The hypothesis `i ∈ S.places` is not needed: for `i` no place, `under i` is the union of its children's subtrees and `placesIn {i}` is empty, so the inequality is `linkCapacity_biUnion_le`. [read]
- The conjunct takes no premise and says nothing of any run: it is a fact about `linkCapacity` and `under`, which `GLOSSARY.md:395` calls a lemma ("any proved statement that isn't a guarantee"). The issue's notes say why it is in the guarantee: "the parts inequality is a conjunct so that each new proofs module holds a lemma the guarantee's proof reads". A listed statement shaped by the unread ratchet. [read]
- Mend: drop `i ∈ S.places →`, or say why it stays; decide whether the parts inequality is part of what the guarantee says, and if it is not, keep it as a lemma in `Proofs/` and take the ratchet's count as it falls; the body says the witnesses rule does not apply to this statement. [read]

### 4. Does not stop a mark. "At the Site's root it is the datacenter", and the limit read off a remote certifier's links counts the interfaces twice

- Quoted, `GLOSSARY.md:390-391`: "a subtree of the Site is a registered location with every location registered under it (`Site.under`), and at the Site's root it is the datacenter". The same in `SiteSubtree.lean:11-12` and `:44-45`. The body: "**One datacenter:** the statement at the Site's root." [read]
- The root's subtree holds the auditor's devices: `Kind.parents` has `| .certifier | .challenger => [some .site]` (`security/Definitions/Core/Site.lean:97-103`). The limit at the root is over certifier-to-outside and host-to-outside links, so it is the limit of the developer's hardware with the auditor's devices at it, not of the developer's datacenter. [read]
- Nothing makes the root one: `.site => [none]`, and `node_entry` adds a site location whenever its `site` id is not yet known (`verity/device.py:628`), so one Site can hold several. [read]
- `node_entry(certifier=…)` keeps the ways' own host-to-outside links and adds host-to-certifier and certifier-to-outside "each at the interfaces' rates summed, beside the ways' own links, which stay" (`verity/device.py:572-575`, `:640-643`). For that host and for the root, the limit read off the links counts the interfaces' rates twice. [read]
- Why it matters: the bound stays true and is loose by up to a factor of two there, so a `β_i` published from it overstates what can leave; and "the datacenter" in the Glossary is not what the definition holds. [read]
- Mend: the Glossary and the docstrings say "at a site location it is that site's registered hardware, the auditor's devices at it among them"; the body's "One datacenter" line says which links the root's limit sums, a remote certifier's beside the interfaces'. [read]

### 5. Low. The docstring's sentence on a Site with no link understates

- Quoted, `RegisteredSite.lean:28-29` and the body's "The premise": "On a Site with no registered link the premise holds only of a circuit with no wire between places." [read]
- With `S.links = []` the binder `link : Fin A.N → Fin A.N → Fin S.links.length` has no value unless `A.N = 0`, so there the first conjunct is stated of the empty circuit only, a circuit with no wire between places included. [read]
- Nothing false follows (such a circuit's egress is 0). Mend: "On a Site with no registered link the bound is stated of the empty circuit only." [read]

### 6. Low. `Site.mem_places` is read by nothing

- Quoted, `security/Proofs/Core/SiteSubtree.lean:22-25`: `theorem mem_places {i : String} : i ∈ S.places ↔ S.Known i`. No Lean file uses it; the only other mention is its own module docstring (`:9`). [ran]
- The body cites it as the tie between `Site.places` and `Site.Known`. No cap moves: the unread ratchet counts wholly unread modules, and this module's other lemmas are read. [read]
- Mend: delete it, or let a statement read it (a property that the places are the known ids). [read]

### 7. Low. Body sentences that say more than was run, or that the diff or the issue does not bear

- "**The suites the change touches, at the record's head** … ten suites, each passes, none from a cache": the run's job line is `uv run tools/verity/check/suites.py --changed 6953e38e55a4… --quick`, and `--quick` leaves the slow tests out. The body does not say so. [ran]
- "Three things differ from the issue's notes", "where the notes say `Registered.lean`": the issue as it stands names `RegisteredSite.lean` and lists the four extra declarations itself. Stale against the issue; nothing in the diff turns on it. [read]
- "**As a share of the compute it covers:** `β_i / Φ_i` bits out a gate, a ratio of two registered parameters": no `Φ_i` of a subtree is defined in the change (the body says so three lines on), and by finding 1 the placement that gives a subtree a compute rate is the one the premise excludes. [read]
- The sections before "What ran, and what has not" carry no mark of ran or read. Each of their claims I checked held (the table under item 3 below), but the body should mark them. [read]
- "**The gates** [ran, at `b02181d23879`]" names no run, so nothing bears it but the sentence. [read]

