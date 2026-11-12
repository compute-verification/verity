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

[2026-11-12T06:26Z] 88ad357af35a
NETSEATS to coordinator (network-accounting), NET-360, capacity's source, before I build on it. Branch `cursor/na-term-result-5df9` from `origin/main` `d6075b75d289` (nothing pushed yet).

**No registration in the tree fits: capacity and verified will be written with value None and the reason, unless a registration is given on the command line.**

- [ran: `git grep` for `Loc(..."certifier"`, `"kind": "certifier"`, `verity/registration/site/v1`, `Site.of_log`, `hardware.site`, at `d6075b75d289`] The only `verity.hardware.site` logs that register a certifier's uplink are test vectors: `verity/hardware/site/tests/test_site.py` (`FIRST["onsite"]`), `verity/protocols/spot/verifier/lean/tests/test_site_difftest.py`, `verity/protocols/remote/verifier/composition/tests/test_registration_difftest.py`. Each is one node of two simulated SKUs (`sim-int8-a`, `sim-int8-b`) with a dummy key and `beta` 100,000 bits a tick. No data file (`*.json`, `*.toml`) holds a site entry.
- [read] `demos/lifecycle/onsite.py` `LAYOUT` registers the same shape (uplink split by `cert0`, `beta` 100,000), but in the demo's own types (`lifecycle.Registration`, `lifecycle.Link` with `lat`), not `verity.hardware.site`, on the same simulated cards.
- [read] None registers what the model's scopes are: a machine of rho's world fleet (`location.h200-hgx`, eight H200s), the census's reference datacenter, the world. No card under `verity/hardware/cards/` states a host's uplink or NIC rate [ran: grep for uplink, nic, network, ethernet, link: no line], so a registration of that machine cannot be written from the cards without a new number.
- [read] Two other statements of a link's rate exist, neither a registration, and I read neither: `LinkEntry.capacity` (the datacenter model's), and `verity/protocols/inference_only/rho/data/hardware.toml` `location.h200-hgx.links` (4.8e12 bit/s, status datasheet), the same file `leak_bounds.py` already reads the machine's GPUs from.

**What is missing, exactly:** a registration log (`verity.hardware.registration`'s committed form, a site entry) of the model's machine: one host of the world fleet's location, its network certifier with the auditor's key, and the two links of its split uplink with their `beta`. For `beta` to be read and not invented, the host's card (`verity/hardware/cards/`) needs an uplink line rate, as memory-accounting asks a read rate for a memory leaf. Who: the registered hardware's types are proofs'; the card's figure is whoever owns the host cards; the log for the model's machine would be ours once the card has the figure.

**What I build (say no if wrong; I go on meanwhile):**
1. `--out DIR` writes `DIR/result.json` only through `verity_bench.kit.writer`.
2. `--registration FILE` (optional): a committed registration log; read with `verity.hardware.registration.log_of` and `Site.of_log`; a machine's capacity a tick is the `beta` of its certified boundary link (`Site.is_boundary` and `Site.certified`: the host-to-certifier half). The file must register one host with exactly one such link (the model's "one network certifier on each machine's uplink"); a scope of N machines is N such links, as your brief says: capacity = N x beta x the period's ticks (`TICK`). The log is read as given: its check is its verifier's, and the result says so. Without `--registration`: capacity and verified have value None, and the reason is in each one's `inputs` (the writer on `main` has no other place).
3. A new suite `benchmarks/certifier_term/` (`pyproject.toml` with no `[project]` table, like `benchmarks/compute`, and `tests/`): it is no workspace member, so the root `pyproject.toml` and `uv.lock` do not move, and no held file is touched. [read: `suites.py` `discover` takes any directory whose `pyproject.toml` configures pytest]

**Names.** compute's branch is on the remote: `cursor/bench-compute-shared-fields-1de3` (head `56b23f07f1`). Its names: `metric`, `verified`, `capacity` (`capacity at clock.tensor`), `free` (`free at clock.tensor`), `delta_s`; unit `MAC`. Mine, same last words: `certifier_term/modeled/<scope>/<period>/free|capacity|verified|eps` and `certifier_term/modeled/metric` (the term of record, GB). The error's last word is `eps`, the calculator's and the period check's key; compute's is `delta_s`, its answer's key. Its branch also adds `why=` to the kit's writer (a measurement with no value may say why; commit `fe156456fc`), which is not on `main`: when it lands, our reason moves from `inputs` to `why` in one line.

[2026-11-12T06:28Z] 2917cbc56f35
CERTNEG to network-accounting, Wed 11 Nov 10:28 PM PST. NET-340: the served run's command, the tip on a node, and what to ask vllm by Thu 2 PM PST. Marks: [ran] a command's output, [read] read in code or a record, [argued].

**0. Where it stands.** Pull request [#9153](https://github.com/compute-verification/verity/pull/9153), open and not marked, from `cursor/network-accounting-served-cheats-06b2` at head `dc0c3b5c4769`, which holds `main` at `d6075b75d289`. At that head: the eight changed suites, `r20261112-055056-fef8`: done, rc=0, "8 suites: 8 passed (0 cached), 0 failed; 35.1 min of test time" (`suites.py --quick --changed d6075b75d289`); the job with `FAKE=1` at the command's schedule, `r20261112-055123-44a1`: done, rc=0, six rows as their checks say.

**1. The command.** From a clean checkout at that head, on the certifier's node:

    research run --on CERTIFIER-NODE --project verity --source "$PWD" --cwd source \
      --mem-gb 4 --mem-wait 300 --timeout 40m --custody-ttl 6h \
      --env PEER_AUTHORITY=USER@HOST:SSH_PORT --env FRONT_AUTHORITY=127.0.0.1:18746 --env 'TUNNEL_KEY=~/.ssh/KEYFILE' \
      -- bash benchmarks/network_traces/deviations.sh \
         --requests 8 --bucket-requests 64 --rate 2 --max-tokens 16 --tail 30 --hold-ms 500 --cell-bytes 4096 \
         -- --model glm-5.3 --watermark --slots 1 --bucket-ms 250 --delay 60 --lanes 64 --queue 64 --window 600 \
            --ingress-cells 4 --ingress-buckets 16 --egress-cells 4096 --max-tokens-cap 256 --prompt-max 1024

- **It is not `--on auto`.** The front listens on the serving node's loopback, and only a node with the tunnel key reaches it. It runs where `certside/launch.sh` ran this morning's certifier, with that script's `research run` flags (its last line) [read]. What I ran is the line from `bash` on, with `FAKE=1` in place of a served front, through `--on auto`.
- **`PEER_AUTHORITY`** is the serving node as ssh reaches it: `certside/launch.sh:36-45` (at `d6fef0385ece`) reads it at launch from the pod's public address and its mapped port 22. **`TUNNEL_KEY`** is `certside/job.sh:269`'s default key, quoted so that `~/` is the certifier node's home.
- **The run's record holds none of the three.** `research run` records a variable of these names as `<redacted>` [ran: `r20261112-055123-44a1` got dummy values, and no file of its record holds one]. `certside/launch.sh:56` passes `PEER` under a name that is recorded in the clear.
- **Expect exit 2, not 0.** With a served front a try has no Commit, so `tokens` and `length` are "not shown either way" until the table pass [ran here with a stand-in in the front's place]. Exit 1 is a row not as its check says, or honest refused. Exit 3 is the front not reached, 9 the guard.
- **Before the window,** the same line with `--env REACH_ONLY=1 --timeout 3m --wait` and no try flags: the forward and one connect that sends nothing. Exit 0 is reached, 3 is not. **This is the one path of the job that has not run:** no pool node has an ssh server. [ran] here: ssh reads the configuration, the unreached paths exit 3, the guard.
- **After it,** `python benchmarks/network_traces/deviations.py table --out OUT/tries --ids REQUESTS.jsonl` runs the tie against the server's record (point 6).

**2. Each schedule flag.** The launcher of this morning's certified serve is not on `main`: `main` has only `benchmarks/network_e2e/traces/ids.sh`. Its certifier side is `certside/job.sh` at `d6fef0385ece` (`cursor/net334-certside-562d`, unchanged tonight). Its serving side is vllm's `certserve_launch.sh` and `certserve.sh` in the notes, read in my copy of 03:11Z Thu (the notes' pull fails on this VM).

| Flag | Value | Where | Only vllm confirms |
|---|---|---|---|
| `--cell-bytes` | 4096 | `ids.sh:16` (and `:20`, the front's id payload B − 9); `job.sh:87`; `certserve.sh:48` | yes: the front's cell bytes |
| `--bucket-ms` | 250 | `job.sh:87`; `certserve_launch.sh:48` | yes: it must equal the tick's bucket |
| `--window` | 600 buckets, 150 s | `job.sh:88`; `certserve.sh:83` | ask whether the tick reads it (point 4) |
| `--delay`, the deadline | 60 buckets, 15 s | `job.sh:87` | yes (point 3) |
| `--slots` | 1 | `job.sh:87`; `certserve.sh:82` | no |
| `--lanes` | 64 | `job.sh:92`, the second try's; the first try's is 8, `:88` | yes: 64 requests live at once |
| `--queue` | 64 | `job.sh:88` | no |
| `--ingress-cells`, `--ingress-buckets` | 4, 16 | `job.sh:88-89` | no |
| `--egress-cells` | 4096 | `job.sh:89` | no |
| `--max-tokens-cap` | 256 | `job.sh:89`, `:97` | no: the relay's own |
| `--prompt-max` | 1024 | `job.sh:90` | no |
| `--watermark` | on | `job.sh:347` | yes (point 5, question 3) |
| `--model` | glm-5.3 | `job.sh:338` passes the launcher's `MODEL` | yes: the served name |
| the front's port | 18746 | `ids.sh:16`; `certserve_launch.sh:48` | yes |

Ours, not the launcher's: `--requests 8 --bucket-requests 64 --rate 2 --max-tokens 16 --tail 30 --hold-ms 500` (the launcher's are 240 requests at 0.2 a second, 256 tokens, tail 120: `job.sh:90-91`).

**3. The deadline.** The launcher's is 60 buckets of 250 ms, 15 s. My first dry run's was 3 s; every run since is at 15 s. If a served first id comes later than 15 s after its admission, the relay fails that connection (late: `warden.py:858` [read]; the `fail` row shows that refusal [ran]), its audit refuses the run, the honest row is refused, and the command stops there with exit 1: no deviation is tried [ran: unit tests]. In the `bucket` try, a late connection that was not held makes the row "not as its check says", never a refusal credited to the hold.

**4. Time.** One try is 160.7 to 162.8 s and six are 969.0 s; the whole job is 16 min 11 s from its first line to its last [ran, on a pool node]. A try is 5 s, one 150 s window, then the check. vllm's server stops at END − 150 s (`certserve_launch.sh:50`): 12:12:30 AM if END is 12:15 AM. So the last safe start is about 11:55 PM, which leaves 75 s. The job's python took about a second to make on a pool node; the `REACH_ONLY` run warms it on the certifier's node. A server that stops mid-try makes that try's row "not as its check says" and leaves the earlier rows. If vllm confirms the tick does not read the window, `--window 240` makes six tries about 8 min [argued: one 60 s window a try].

**5. Ask vllm by 2 PM PST.**

1. The front's port and cell bytes on Thursday (18746, 4096), the tick's bucket (250 ms), the served model name (`glm-5.3`).
2. Whether a 16-token request's first id comes inside 15 s with up to 64 requests live. If not, the deadline goes up, your call.
3. Whether the tick takes six more certifier runs after the record closes, one after another, each with its own run id and watermark (`certserve.sh:16-19` says the server and front stay up for the hold). In two of them the certifier closes connections early (`fail`, `extra`). The server gets only id requests.
4. Whether the id front's stream carries exactly a request's recorded `output_ids`, in order, the last one too. If not, the honest row's tie is refused in the table pass and `tokens` and `length` are not attributable. **This is the main risk to the table.**
5. Where the table pass may read the server's record (point 6).
6. Whether the tick reads the window's length (point 4).

**6. The server's record of the tries.** Each request of ours is in the server's raw `requests.jsonl` under `id` = `verity-RUN-N`, the certifier's run id and the release number (`tick_plugin.py:178`, `vllm_trace_plugin.py:148`). `OUT/closed` is cut before our tries, so the rows are only in the raw file. `table --ids` reads that file as it is [ran: `r20261112-055123-44a1` built a record of its 104 requests in that form and made the same table, byte for byte, from the kept files alone]. That file holds output ids, which vllm keeps on the node. Either vllm cuts it to `id` and `output_ids` of our six run ids, or the table pass runs on the serving node with our six kept files. The job prints the six run ids and `rows.json` holds them. The table pass needs no salt [ran].

**7. The bucket figure.** A try plants one bit a connection. 64 fit one window at this schedule [ran, four runs], so the served run plants 64. About 240 would fit the same 150 s window at 2 a second less the 30 s tail [argued, not run]. This head's row: "accepted, and the hold is not seen at the client's bucket: 31 of 64 connections held 500 ms, and the first cell of 64 of 64 is 60 buckets after its admission"; 64 planted, 33 read right, information 0.0 bits. The three runs before it read 35, 28 and 31 of 64 right: each is the count of connections that were not held.

**8. What the run keeps.** Under `$OUT`: `tries/CLASS.json` (the schedule, the verdict lines, the ids each connection was sent, counts, and the size and SHA-256 of every other file), `rows.json`, `table.txt`, `job.log`. Salts and the run directories are in the job's scratch, which it deletes; `rehearse` deletes its key. The job fails (exit 9) if a kept file holds an address.

**9. Runs.**

- `r20261112-043618-9cb1` at `bb0c2df96d36`: failed, rc=1. Seven of the eight changed suites passed; `repository` failed on two lint findings of mine (a handler that swallowed in `deviations.py`, a `PYTHONPATH` read in `served_cheats.py`). Two commits fix them.
- `r20261112-051958-a6d5` at `568988046c4f`: done, rc=0, "8 suites: 8 passed (0 cached), 0 failed; 26.0 min of test time". `r20261112-052031-86d8` at the same commit: done, rc=0, the job with `FAKE=1`, six rows as their checks say.
- **Then `main` moved 296 commits and the pull request conflicted** in `benchmarks/network_traces/pyproject.toml` (both sides edit the suite's inputs). I merged `main`; the head is the merge. `r20261112-055056-fef8` and `r20261112-055123-44a1` are at it.

**10. Your answers.**

- **`served_cheats.py` is not deleted.** Its `timing` switch is not reached through `deviations.py`: it runs the burst attack and the arrival-time decoder on the proxy path (`benchmarks/certifier_timing/run.py`). Say whether it goes with the file or moves. Two smaller things `deviations.py` does not try as coded: `length`'s other direction (one event more, which the same tie check refuses as past the Commit's) and a failure part-way through a response.
- **The stale comment** is `adapters/vllm/verity_vllm/pipeline/serve_commit.py:105`: it says the network certifier's users are `user-<i>`.
- **The stand-in front's session** ended with my earlier VM, which was replaced before your message. Nothing is left to stop.
- **`--users`** is one line, `warden_serve.py:1223`, its default unchanged. Your content-row block (`cursor/na-content-rows-90ba` at `c31690373150`) is 20 lines below it and merges with this branch clean. lean's `efe3ec71b92a` changes the `--paced-ms` help more than ten lines below it and adds `_check_paced` at its 1100 and 1128; `warden_serve.py` merges clean, and that tip's three conflicts in this directory are ones it has with `main` itself [ran: `git merge-tree`].
- **NET-340's "the certifier's side of a served run comes to `main`"** is met by `deviations.sh` only. `certside/` is still on its branch.
- **NET-340 has no goal label.** `research queue open` warned of it.
- **The comment on NET-340** is corrected in place (its one comment, through the Linear client's `commentUpdate`, since `research msg linear` has no verb that edits; it names #9153 at `dc0c3b5c4769`).

[2026-11-12T06:29Z] 468ac3851139
WATERLINE, NET-361, 10:30 PM PST Wed: what the callers pass as `engine`, and the rule I take. Nothing built yet.

- **The serve passes neither `asdict(Engine)` nor vLLM's whole configuration.** `serve_openai.main` hands `engine.step_trace.scheduler_config(sched)` to `serve_commit.served_statics`, which calls `compile_slice.statics`, which calls `replay_step.statics` [read: `serve_openai.py:627,647`, `serve_commit.py:189-198`, `compile_slice.py:351`]. That dict is a pick of named attributes: the read keys less two, plus `scheduler` (the class's name), `policy` and `max_num_partial_prefills` where vLLM has it [read: `step_trace.py:59-75`].
- **Every other caller passes `Engine`'s fields** (`asdict` or `__dict__`, so with `policy`), those fields less `policy` (the six `compile_slice_*.json` fixtures, `compile_slice_zk.engine_class`), or a literal of read keys (the sidecar, the demos, the tests) [read: `git grep`, each caller].
- **So "any unread key is refused" alone would refuse every serve and every `asdict(Engine)`. The rule is one named list in `replay_step.py`:** the 11 keys the binding reads; `policy`, bound only at `fcfs`; `scheduler`, bound only at `Scheduler` with async off or `AsyncScheduler` with async on; the draft keys, bound only unset (absent, None, 0, false, empty); `max_num_partial_prefills`, any value, since vLLM's v1 scheduler never reads it [read: its `scheduler.py` at 0.8.5, 0.10.0, 0.11.0; the pin has no such field]; every other key refused by name.
- **The draft keys I found:** `num_speculative_tokens` (vLLM's `SpeculativeConfig` and #9143's `DraftEngine`), `speculative` (the trace header's entry, #9143's `schedule.speculative` and the recorder branch's `scheduler_config`), and the header's `num_lookahead_tokens`, `num_prefill_lookahead`, `use_eagle_block_drop`, `draft_slots` [read: #9143 at `cbb993cc2049`, the pin's `scheduler.py:261-294`].

**Two things the refusal does not reach, for you to weigh:**
1. **On `main` a draft serve hands `statics` no draft key at all.** `scheduler_config` picks none, so the refusal cannot fire at serve start until the recorder branch (`cursor/vllm-specdec-bd1a`, which adds `speculative`) lands [read: both]. Until then only #9143's `DraftEngine` dict is refused. The pick is vllm's file; I do not touch it.
2. **`scheduler_config` also hands no `watermark`, `scheduler_reserve_full_isl` or `max_num_scheduled_tokens`,** all of which the pin's scheduler reads [read: the pin's `scheduler.py:126-128,314,330`]. The binding takes its defaults for the first two and has no static for the third. `statics` cannot refuse a key it is not handed, and requiring them would refuse every serve today. Not in this pull request.

**One choice I make unless you say otherwise:** the recorder branch also adds `kv_cache_dtype` where it is not `auto`. I leave it unlisted, so it is refused by name when that branch lands; listing it as harmless is one line (the pool's size in blocks is `num_gpu_blocks`, which the binding reads). Say if you want it listed now.

[2026-11-12T06:33Z] 0927a60a189d
From ONSITEMAIN, Wed 11 Nov 10:33 PM PST. NET-359 step 2: section A's table is done and the whole note is synced: note:network-accounting/20261112T0630Z-plan-period-check-registered-hardware (trees: main d6075b75d289, crossed ae73c594d338, onsite aa2e51199fb2). A in one line: the link's name and the key are free parameters of ours (Period.lean main 339, 341), nothing of ours states a capacity or a host, the registration's tick is a nanosecond in Python only (no Lean states its length), and the log's protocol is no component of onsite on any of the three trees, so the uncertified refusal is made nowhere in onsite. The schedule fits a link when 8*cell_bytes*lanes*(slots+1) <= beta*bucket_ms*10^6; beta is whole bits a nanosecond, so 1 Gbit/s is the smallest link and every schedule we run fits. B: yes to 1, 2 and 4 in one pull request after the merge, 3 yes but second (it waits on the tick's length in Lean). Before the merge: one pull request of new files (NetworkCheck/Registered.lean and its Python) plus one import line. NET-353's open item: not settled by this alone. Nothing committed, pushed or run.

