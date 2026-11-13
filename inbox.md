[2026-11-13T05:20Z] aa39a91831aa
NETSEATS to network-accounting (lead), Thu 12 Nov 9:20 PM PST. **The read: what of a registration and of the gate's output is public or private, by the design rule, and what reads it.** No code changed. Every line number is at #9656's head `fce58a7ec8ca`. `V` = `verity/protocols/certifier/verifier`, `N` = `benchmarks/network_traces`, `G` = `verity/composite/device/live_challenger/auditor.py`. Tags: [read] the code says so, [ran] a command showed it, [argued] my reasoning from the rule. "Ruled" = soundness's three rulings as you gave them.

**The rule's words I hold each row to:** "Only four kinds of things are public, each with its reason: protocol structure, the protocol's outputs, the verifier's coins and samples, and hiding commitments"; "Commitments are hiding commitments (`hm96` rows, or keyed leaves ...) in frame-v3-sha512 trees"; of the auditor's devices, "what they sign is witness and never reaches the auditor in the clear".

**What comes out of it, first:**

1. **All three roots are `hm96` rows in a frame-v3-sha512 tree, each row under its own 192-byte salt that stays witness** [read]. By the rule each is a hiding commitment, so public [argued]. Yours to rule.
2. **Only the cells' registration holds a field the rule keeps private: `wiring`.** A connection rows' registration and a load's hold structure, a hiding root and the certificate, and nothing else [read]. So if `root` is ruled public, those two could be published whole and the cells' whole less `wiring` [argued]. No reader on the auditor's side asks for either today [read], so the move stays as it is.
3. **A cells' certificate can't be checked against its root without `wiring`**: the signed digest is over `wiring ‖ domain ‖ root` [read: `V/warden_rows.py:42-44`, Lean `NetworkCheck/Period.lean:982-986`]. Every check of a cells' head today reads private text.
4. **One leak the list did not have: `out/rehearse.log`.** The gate publishes the rehearsal's stdout, and its last line prints the two times' quantiles and the tick lateness [read: `G:2607-2610`, `N/warden_serve.py:1346, 1204-1207`]. The end itself cuts them (`CELL_ROWS`, `G:460`). Not changed; it is a row below.

**The registration's fields**

| What | Public or private | The reason, in the rule's words | What reads it |
|---|---|---|---|
| cells' `root` | **public** [argued]; not ruled | A hiding commitment. Its leaves [read: `V/warden_rows.py:27-38, 337-346, 376-384`]: T leaves, T the schedule's buckets; `leaf_k = row_leaf(domain, k, hm96.SHA512.tree_leaf(default_key, b‖c_k), "hm96-sha512/row/v2")`, `b‖c_k = hm96.SHA512.commit_string(default_key, x_k, salt_k)`, `x_k` the row's `sha512/row/v2` digest, `salt_k` 192 bytes fresh from the operating system for that row alone. The salts are `cells.salts`, which the gate keeps in `witness/` [read: not in `WARDEN_PUBLIC`, `G:449`]. | `check_registration` (`V/warden_rows.py:448`); the registered port (`_port` 486, `port` 493, `bucket_port` 509), which Lean's `Flock.Registered.checkPort` holds the table rows' paths to (`verity/protocols/flock/verifier/Flock/Registered.lean:270`); the Lean period check (`Period.lean:863-866` reads it, `982-986` hashes it into the head); the scan's registered roots (`N/scan_zk.py:1405-1420`); the session's open root, a SHA-512 over each registration whole (`N/scan_zk.py:1273-1280`). Callers of the check and the port: `V/tie_service.py:448`; `N/scan_zk.py:581, 643, 701, 968`; `N/egress_zk.py:107, 156, 184`; `N/n2_tie_zk.py:208, 350`; `N/warden_zk.py:133, 203, 306`; `benchmarks/one_stage/a5_warden.py:142, 230, 268`; `N/negatives.py:239`; `N/warden_serve.py:1042`. |
| a connection rows' table's `root` | **public** [argued]; not ruled | A hiding commitment, the same form [read: `V/connection_rows.py:40-50, 343, 369-381`]: the table's rows' leaves, `hm96-sha512/row/v2`, "salt_k is fresh for each row of each table"; the salts are `rows.salts`, in `witness/`. | `check_registration` (`V/connection_rows.py:431`), `port` (478), `verity/protocols/certifier/prover/lengths_row.py:11`, `N/connection_rows_run.py:215`. No Lean reads it [ran: a grep of the Lean for connection rows finds no file]. |
| a load's `root` | **public** [argued]; not ruled | A hiding commitment [read: `V/transfer_rows.py:12-23, 158-166`]: the value's rows' leaves, `hm96-sha512/row/v2` or `hm96-sha512/row-seg/v1`; `b‖c_r = hm96.SHA512.commit_string(default_key, registered.row_digest(row r, word_bits, schema), salt_r)`, `salt_r` 192 bytes for row r and no other. The salts go to the prover alone. | `check_registration` (`V/transfer_rows.py:222`), `port` (262), `V/blocks.py:264` (`certified`) and `:273` (`pins`, which only its tests call), `N/load_relay.py:1126` (`check_load`, from `N/warden_serve.py:1068`, in the run's own directory). No Lean reads it: the Lean names its tag as a constant alone (`verity/commitments/identity/Identity/Domains.lean:208`) [ran: a grep]. |
| `domain` | **public** [argued] | Protocol structure: a digest of the schedule's parameters, the session, the link, the direction, the execution phase and the rows' shape (`V/warden_rows.py:265-271`; a table's adds its name and row bytes, `V/connection_rows.py:255-261`; a load's is `registered.domain(program, value, words, rows, word_bits)`, `V/transfer_rows.py:100-104`). The session is in `deployment.json`, which the gate publishes (`N/warden_serve.py:776-777`). | The verifier derives it and compares, never trusts it: `check_registration` (13 of the cells' 16 keys are derived). The port and Lean's `checkPort` open paths under it. |
| `wiring` | **private** (ruled) until committed as an `hm96` row or a keyed leaf | Not a hiding commitment in the rule's two forms: `tagged_sha512(WIRING_TAG, salt(64) ‖ u32be count ‖ (q, first, cells) by q)` (`V/warden_rows.py:141-153`). Its opening is each connection's first row and how many cells it has, which the module says "is opened to the verifier for now" (`:81`). | The signed head (`:42-44`); `check_registration` (448); Lean `rowsOf` and `headDigest` (`Period.lean:863-866, 982-986`), through the case. |
| `rows` | cells': **public** [argued]. A table's: **public** (ruled). A load's `rows`, `words`, `word_bits`: **public** [argued] | Protocol structure. Cells': T, the schedule's (`rows_of`, `V/warden_rows.py:121`). A table's: the schedule's whatever the load (`tables_of`, `V/connection_rows.py:143-150`); how many rows are in use is a value and is not in the registration. A load's: the value's shape, which the public program's port fixes ("the model's size is public"). | The verifier derives them; `port` gives them to `checkPort`, which refuses a port whose rows don't fit (`Registered.lean:276-277`). |
| the program (a load's `program`) | **public** [argued] | Protocol structure: "the circuit's program digest (`circuit.json`'s `program`), public" (`V/transfer_rows.py:21-22`); "a circuit holds only gates", registered by digest. The gate's `request.json`, which every fetch carries, already names it. | `check_registration` takes it as an argument and compares (`V/transfer_rows.py:232-240`); the load's domain. |
| the two times, `commit_ms` and `sign_ms` (connection rows: `build_ms` too) | **private** (ruled) | None of the four kinds: a time the device measured over its witness (`time.perf_counter`, `N/warden_serve.py:533-540`; `N/connection_rows_run.py:131-142`). | Each line's `cost` in `witness/cells.jsonl` and `witness/rows.jsonl`; the check's summary takes their quantiles (`N/warden_serve.py:1049-1052`; `N/connection_rows_run.py:224-226`), which then leave by the two rows marked * below. |

**Point 6's four, and two more**

| What | Public or private | The reason | What reads it |
|---|---|---|---|
| `out/load.json` | **moved to `witness/`** in #9656 [ran]. By the rows above every field of a load's registration is public [argued]. | Each value's registration whole: structure, the program, a hiding root, the certificate. No `wiring`, no time. | The run's own check (`check_load`) and `blocks.pins`, which takes registrations as an argument; nothing opens a gate's `out/load.json` [read]. |
| the three jobs' copies into `$OUT` | **private** as a whole [argued]; not changed | Each copies a rehearsal's own files, not a gate's: `N/scan_zk.sh:91-93` (`deployment.json cells.jsonl certified.jsonl check.json warden.json serve.log`), `N/warden_zk.sh:37` (the first five), `benchmarks/one_stage/a5_tp.sh:257, 263` (`deployment.json cells.jsonl check.json`). `cells.jsonl` holds `wiring` and the times; `certified.jsonl`, `check.json`, `warden.json` and `serve.log` are what the device signed and saw. `$OUT` is the run's record in the store, not the auditor. | `N/tests/test_out_request_values.py:117-119` pins the scan's list and calls them "a warden rehearsal's public files". |
| the chain's certificates step * | **private** in part [argued]; not changed | `N/warden_chain.py:133` writes the check's summary whole: the two times' quantiles, `arrival_lag_ms`, `tick_lateness_ms`, `clocks`, `ticker_stalls`, and counts that follow the traffic (`data_bytes`, `completion_tokens`, `prompt_tokens`, `finish_reasons`), with up to 20 of the check's problems as text. | Nothing in the tree reads `warden_check` back [ran: a grep finds the one writer]. |
| the end's `tick_lateness_ms` | **private** [argued], by the ruling on the two times; not changed | A time the device measured (its ticker's lateness, from `warden.json`), in `WARDEN_COUNTS` (`G:2492-2494`). | Nothing in the tree reads the end's copy by name [ran: a grep]. |
| the end's `witness` | names **public**, sizes **not one of the four kinds** [argued]; not changed | Each witness file's name and byte count (`G:2540`). A file whose size the schedule fixes (`cells.salts`) is structure; one whose size follows the traffic (`certified.jsonl`, `serve.log`) is a function of values. | `N/leak_guard_cost.py:68`. |
| `out/load-relay.json`'s `seconds` | **private** [argued], by the ruling on the two times; not changed | The relay's measured wall time (`N/load_relay.py:1161-1163`). The rest of the file is counts and the sink's acknowledgements. | The gate, which puts the file whole in the end's `load` (`G:2541-2545`): so `seconds` is in `end.json` too. Nothing else reads it [ran: a grep]. |
| `out/rehearse.log` * (new) | **private** in part [argued]; not changed | The rehearsal's stdout and stderr whole (`G:2607-2610`). Its last line (`table`, `N/warden_serve.py:1196-1207`) prints the two times' quantiles, the tick lateness, the clocks' reading and the start's `base_ms`; stderr is whatever the rehearsal says when it fails. | A person reading a failed run; `N/deviations.py:757` reads its own copy's last lines. |

**Stale statements this leaves, for whoever takes the ruling to the tree** (I changed none: the right text is the ruling's):

- The three verifier modules say the head "is public, as a hiding root is" and the registration is "one public JSON object": `V/warden_rows.py:40, 50`, `V/connection_rows.py:52, 65`, `V/transfer_rows.py:26, 41`. For the cells' this is false of `wiring` under the ruling.
- `N/tests/test_out_request_values.py:117` names the scan's copies "public files".

**What I did not check:** whether `hm96` with the default key and a 192-byte salt is hiding (I read the construction, not its proof); whether any reader outside this repository opens a gate's `out/`; the end's `cells`, `wire_bits` and `cells_in_their_bucket`, which are totals over the run and which I did not hold to the rule; what `tunnel.log` holds.

**The runs:** the chain record `r20261113-050508-9400` and the suites `r20261113-045547-4f11` are both still running at this hour. #9656's body has `Goal demos/onsite-demo` above `Refs NET-417` since 9:14 PM PST [ran].

[2026-11-13T05:20Z] 863bafe54dda
TERMGB to network-accounting (lead), Thu 12 Nov 9:20 PM PST. Your 9:09 and 9:13 PM PST are read whole. **Points 1 to 3 are pushed: #9501 is at `6fba82a0ac32` [ran: 13 reader tests, 50 with the calculator's two, the runner's 15, the five root files 65]. It is not yet the head to name: the removal test, the mutations and the harness's file come first.**

- **1, the reason is held.** `REASONS` beside `CONTROLS`: `stop_word`, `id`, `max_tokens`, `output`. A control refused for another reason leaves its verifier without its control at that commit, whatever other it refused; the line is "control stop_word refused for 'parse', not 'stop_word', on RUN", and the class's line still prints each reason given. Of a superseded run such a control stands. Tests: another word for each of the four (four words each, a case and a trailing space among them), the right word beside a wrong one in either order, two wrong ones.
- **2, the rate's text.** `EVERY_BLOCK` is `"1/1"`. A registration at `2/2`, `rate:1/1`, `1/1 `, `1 / 1`, `01/1` or `1` is refused as another rate, whatever its verdicts state; a verdict that states `rate:1/1` or `2/2` under `1/1` is refused.
- **3, a control with more fields: you read `control_lacks` right, and one thing beside it was not so.** At `44780e30a640`, `read` took a control for a superseded run's where it named that run as its `serve`. It no longer does: of a control the five fields are read and no other. A test hands the four controls with `registration`, `serve`, `class`, `rate` and `piece` and gets the same answer line for line, and another a control whose `serve` names a superseded run, which still counts.
- **The removal test (your 9:13):** building now. I expect it to raise in the cases where the thing left out is itself what refused (a refused or disagreeing record, a control not refused, a second head); I send each kind with its smallest case before changing anything in the reader.
- Not done yet: the mutations at this head, the harness's file, the control's form on NET-403, #9501's body, the comparison (by Fri 8 AM PST).

[2026-11-13T05:21Z] f72bf7c79bf6
CROSSTICK to network-accounting (lead), Thu 12 Nov 9:21 PM PST. **Two things: where my report on #9216 and #9524 is, and events of yours that were delivered to my conversation.**

**1. The report** (also my turn's final message; the whole is the synced note `network-accounting/20261113T0418Z-report-9216-marksonce-9524-amended`)
- #9216: `cursor/na-span-count-e7c1` at `066825c49b25f54a3ecf13cd16bb6a5c9305550b`, pushed, unmarked, body at the head. `Issuer.mark` raises `MarkRefused` on a second mark of a message from `d639a6770e62`; the test is `test_the_issuer_signs_no_second_mark_of_a_message`.
- Runs, all finished [ran]: `r20261113-034148-0837` (verifier suite 1455 passed; network suite 5 failed at `d639a6770e62`, the mirror's break the merge made, mended at the head), `r20261113-040946-f247` (network suite at the head, 115 passed), `r20261113-034230-13aa` (83 suites passed, `--quick`), `r20261113-034302-3f30` (Lean try, exit 0).
- #9524 at `e37a931d814c`, per statement and premise, is section 2 of the note. Two differ from what the count and the device give: `SentWithinCapacity` over every phase is false of the device (a refused connection's cell is outside the lanes, in a phase it certifies no record of), and `capacityBits` is the schedule's, not the registered link's `β`. The three model lines hold as written and my proofs check on that head's text.

**2. Events addressed to you reached my conversation, 9:16 to 9:20 PM PST**
- Your hourly checkpoint timer's firings of 01:10Z, 02:10Z and 03:10Z.
- Fifteen Slack thread events of 03:04Z to 03:28Z from your subscription.
- Four workers' completion notices: NETSEATS, ENTRIESONCE, WATERLINE, PERIODPROOF. PERIODPROOF's last output read as mid-task.
- I acted on none of them: I read no names as the lane, pulled no inbox of yours, resumed no one, ran no Linear audit and wrote no checkpoint line. If they did not also reach you, the lane's sign-of-life line for those hours is missing.
- My own one-off timer's prompt never showed; I did its job on the turn the notices opened.

TIMER: none

[2026-11-13T05:27Z] 61b676bf59ca
ONSITEMAIN, Thu 12 Nov 9:27 PM PST.

**The four reruns on `main` all ended with exit 0** [ran], each launched once, each with `check.py --build` of the three packages named in the launch line and placed by the audit pool, then the suites the change reaches against `main` at `1e4d5c993c08`, quick:

| Cut | Branch | Head | Run | Lean check | Suites |
|---|---|---|---|---|---|
| A | `cursor/na-scratch-port-cut-a-onmain-84f9` | `d95a1ba976f895e1442b7b5f9d48bd6b74ead7a9` | `r20261113-033323-c6aa` | exit 0 | 60 of 60 passed |
| B | `cursor/na-scratch-port-cut-b-onmain-84f9` | `8aa48b780e960d6a4e30835b4500c958dcc73ba4` | `r20261113-033325-f0f9` | exit 0 | 60 of 60 passed |
| I1 | `cursor/na-scratch-port-cut-i1-onmain-84f9` | `f74396a5c3ed014890a4c187e6f676ed2ab93b75` | `r20261113-033326-eaff` | exit 0 | 60 of 60 passed |
| L | `cursor/na-scratch-port-cut-l-onmain-84f9` | `1a2ae84d146a35d96e6d97b20dd24f7cf980d20f` | `r20261113-033327-20c5` | exit 0 | 60 of 60 passed |

- The public boundary's test passes in each with the rows audit and its executable listed private, and the record difftest's two tests that failed on the old base pass on `main`.
- Each body in my note is at its new head: the base, the branch, the run; cut A's holds the sentence that the rows audit and its executable are private because they import the period check, and that making either public is a boundary decision nobody has taken.
- No pull request is open for any of the four and nothing is marked. They are one series (A, then B, I1, L on it), so B, I1 and L open stacked on A unless you say otherwise. Say which open, and in what order against NET-416, which meets B in `certifier_piece.py`'s `decide` and its three steps.

H's check `r20261113-044737-f422` is running; the certifier verifier's package has passed in it.

Least sure of: nothing in these four; the one thing they do not show is a served window, which no cut of the port has run on.

TIMER: none

