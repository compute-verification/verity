[2026-11-12T19:02Z] b621cb436b15
TERMGB to the network-accounting coordinator, Thu 12 Nov 11:04 AM PST. **NET-394, items 1 and 2 as you ruled them at 10:51 AM, are built and pushed: `cursor/na-certifier-term-simulated-b4e6` at `c3e29d66cf7b`, two commits on `main` `0582899c4392`. proofs' worker can build on it now. The harness suite fails at this head, on the mark before any label: 12 of its 14 tests. The count's term is next.**

**What the branch holds** [ran: both test files at the head, 21 passed and 15 passed]
- `OF_RECORD` stays 2. The document's `value`, the tables' column order and the result's metric are as on `main`. No number moves: the document at `--response-ids 1024` differs from `main`'s in the label, the new input, the new `mark` and two legend lines, and in nothing else [ran: a diff of the two documents].
- `SET_TIME` is "with a simulated set-time card"; its short form, the rows `timing` and `all three`, the lines above the tables and the README say the same.
- `MARKS` has a fourth mark, `simulated`. One input carries it: `set_time_card`, value `simulated`, source soundness's clause in your words, with no measured figure. The document's `with_set_time_card` carries `mark: simulated`. The schema stays `certifier-term/v2`: what is new is one input and one key.
- With `--out` the result holds one more measurement, `certifier_term/modeled/simulated/metric`: the scan's row over the world's year at D_j = 1, in GB against the same target. Its `inputs` open with the row it is of and then the card as the document states it, the clause whole. The kit has no mark of its own on a measurement [read: `kit.py`], so nothing was refused: "simulated" is in the name and the inputs.

**The harness suite at this head** [ran: `benchmarks/harness/tests/test_exfiltration.py`; at `main`'s tree all 14 pass]
- 12 fail, 2 pass. 11 fail at `verity_bench/exfiltration.py:394`, its own check of marks: "an input is marked one of measured, modeled, assumed, with its source: set_time_card". The twelfth, `test_verity_bench_exfiltration_is_the_commands_runner`, exits 2 at its line 732, the same refusal through the runner.
- So the first line proofs' worker needs is `exfiltration.py:102`, its `MARKS`, taking `simulated`.
- With that one line given in memory and nothing written [ran]: 13 pass, and one fails on the label's text, `test_what_rests_on_the_set_time_card_is_printed_only_under_the_calculators_label` at its line 523. Lines 523, 524, 529 and 530 of the test spell "with a set-time card (not yet tested)"; each needs "with a simulated set-time card".
- One more reader states the old way and is not mine to touch: `benchmarks/exfiltration/README.md` lines 36 and 106 ("no timing test has shown").
- I changed nothing under `benchmarks/harness/`. My pull request cannot pass that suite alone: it lands with proofs' change or after it. Say if you want it otherwise.

**Other readers** [read: a grep of the tree for the calculator's names]
- `chain_record.py leak` reads `per_link`, `at_link_rate`, `READINGS` and `WEIGHTS`; `deviations.py` reads `terms`. Neither reads the document, the label or the marks, and I changed none of those functions: each prints what it printed.
- The cards (`benchmarks/spot_figures/cards`) are read by the calculator and read nothing of it: unchanged.
- The only reader of the document outside the benchmark is the harness's `exfiltration.py`.

**Next:** the count's term. From #9378 at `1d6ebf70ff8c` the rule is R = lanes × ⌈(T + ingress buckets − 1)/(D + 1)⌉ [read: `WardenParams.phase_releases`]. Its worker's 336 is that rule at 16 lanes, T = 600, D = 30 and 30 ingress buckets [computed]. At the benchmark's 1,280 buckets and the same schedule it gives R = 688, and six classes are 6 × log2(689) = 56.6 bits a link a phase [computed]. The lanes, the delay and the ingress buckets are not inputs of the benchmark today: I make them inputs, assumed, at that schedule's values, unless you say otherwise.

**The suites' run** `r20261112-181342-dd6c` has a slot: `running` at 11:03 AM PST [read: its status].

A slip: my last message said 10:54 AM PST; the clock read 10:52.

TIMER: none

[2026-11-12T19:04Z] e61dec65716d
CERTNEG to the network-accounting coordinator, Thu 12 Nov 11:10 AM PST. **NET-396, step 1: every reader of the two lines in the tree. No line is changed yet; I build once you have this, unless you say stop.**

**Result: no reader in the tree breaks on an added field, with one rule to keep: every new field is a float that is always there, on every `LIVE` line `flock-circuit` prints.** One test compares two whole `LIVE` records key by key (item 4). proofs' candidate is already answered for our sessions by a field the line has (item 7).

**1. How I looked.** [ran] a scan of every tracked file for the two tags, at lean's tip `e209a56847d4` (#9375's head, still open at 11:02 AM PST) and at `main` `0582899c4392`; then [read] every site. Lean's tip: 127 sites in 70 files. `main`: 129 in 72; the two more are ours, `benchmarks/scan_cost/result.py` (reads `e2e_s`, `tp_s`, `prove_s`, `prove_total_s`, `witness_s`, `wait_e2e_s`, three bucket keys, by name) and its test. Then [ran] a second pass for readers the tag scan can't see: files a shell reader writes the record to, and `proof_server`'s session object. It found six more (items 3c, 3d).

**2. Writers.** Ten Rust binaries print the two tags. I change one, `flock-circuit` (`verity/protocols/flock/prover/live/src/bin/flock-circuit.rs`):

- `LIVE`: built in `session_finish` (lines 4694 to 4708), printed at 7561 (`prove_jobs`, the stream) and 7680 (`prove`).
- `SESSION`: built and printed in `serve_on` (7187 to 7198). The same line is appended to `<out>/index.jsonl`: a second sink, with its own readers (3e).
- Untouched: the session's record `session.json` (what the Lean verifier reads), the other nine binaries (`flock-live`, `flock-pure`, `flock-pure-gpu`, `flock-link`, `flock-gpu-link`, `flock-ir-block`, `flock-ir-frame`, `flock-ir-sampling`, `flock-vllm-v1`), and five `.patch` files under `flock/agreement/` that carry the same `println!`.

**3. Readers on lean's tip: 51 files by tag, 6 more by the second pass, 4 frozen fixture files.** Paths under `benchmarks/` unless they start with `flock/` (= `verity/protocols/flock/`).

a. *Parse the JSON and take fields by name* (an added key can't reach them):
- `one_stage/`: `a8.py:229`, `a8_cost.py:47`, `phase_profile.py:130`, `phase_profile.sh:224`, `replay.py:298`, `session_profile.py:33`, `zk_cost.py:396`.
- `pouw/`: `h1_lanes/served.sh:205`, `h1_prove/run.py:844,846` (both lines; of `SESSION`: `accepted`, `aborted`, `why`), `hidden_zk/send_check.sh:81`, `hidden_zk/summarize.py:59`, `served_zk/served_commit.py:710`, `tests/test_hidden_zk_summarize.py:43`, `tests/test_served_zk.py:1275`.
- `flock_s8_block/summary.py:66`, `multi_unit/prove.py:46`, `private_circuit/universal.py:159`, `proof_throughput/verity_proof_throughput/zk.py:85` and `tests/test_zk.py:22`, `recursion_e2e/rec_profile.py:126,131,229` (both lines; of `SESSION`: `accepted`) and `tests/test_recursion_e2e.py:30,39,531`, `served_units/summarize.py:53`.
- `flock/agreement/`: `firewall_controls.sh:54,55,61` (both lines; of `SESSION`: `accepted`, `why`, `rounds`), `firewall_finish_agree.sh:68`.
- `flock/pod/`: `60-circuit.sh:84,88`, `80-rec-inner.sh:92`, `85-rec-reprice.sh:388,389` (both lines), `86-zk-cpu-steps.sh:139,147` (both lines; of `SESSION`: `verify_s`, `accepted`, `session`), `88-rec-c3-inner.sh:128`, `90-tmpl-unit.sh:129`, `gemm_hill.py:246`.
- `flock/prover/python/verity_flock/`: `bench.py:92,199,283`, `circuit_bench.py:141,196`, `class_statement.py:1335`, `ir_bench.py:195,238`, `negatives.py:69`, `register.py:32`.
- `flock/prover/tests/`: `test_class_statement.py:687,720,818`, `test_gemm_hill.py:109`, `test_lean_record_checks.py:49`.

b. *Count the lines, stamp their arrival, or print a prefix*: `one_stage/zk_cost.py:212`, `flock/pod/60-circuit.sh:94`, `flock/pod/86-zk-cpu-steps.sh:103`, `flock/pod/92-zk-sessions.sh:51` and `93-zk-tree-from.sh:62` (the line's first 200 characters to a log; keys print sorted, so a new key can move what those 200 characters show), `one_stage/tests/test_server_rate.py:54` (writes a stand-in line).

c. *Pass the whole record on*, each with who reads it there:
- `one_stage/a0.py:242` returns it to the scan and the one-stage audits, which take fields by name; line 254 adds a key of its own, `challenger`.
- `flock/prover/python/verity_flock/proof_server.py:252,505` keeps both lines on its session object (`.live`, `.served`). Read from there by name: `one_stage/a5.py:993`, `one_stage/prove_leg.py:377`, `one_stage/server_rate.py:444 to 488` (line 488 writes the record less five keys to `sessions.jsonl`), `one_stage/session_split.py:54 to 56, 167 to 169`, `flock/prover/tests/test_proof_server.py:107,136`.
- `pouw/h1_prove/drawn.sh:113` to `verdicts.txt`, read by `pouw/h1_prove/drawn_cost.sh:73` (`accepted`).
- `flock/pod/85-rec-reprice.sh:1036` to `v-sess/*.live`, copied at 1201, read by `flock/pod/lean_verdicts.py:93 to 104` (`prover_stopped`).
- `flock/pod/89-served-registered.sh:111` and `91-plain-hidden.sh:92` to `live.json`, which becomes a fixture: `flock/prover/tests/test_lean_served_registered.py:148` (`public_file_sha512`), `test_lean_plain_tree.py:80`, `test_lean_verify_members.py:137,223` (that it exists).
- `flock/pod/10-link.sh:59` to `sessions.tsv` and `76-read-gadget.sh:112` to `live.tsv`: no reader in the tree.

d. *Frozen lines*: `numerical/tests/bench/data/flock-runs/r20260925-233831-a064/out/cell/p4-16384/{prove,loopback-prove}-s{0,1}.txt`, four files of `flock-live`'s lines. Not regenerated by a test.

e. *`sessions/index.jsonl`* (the `SESSION` line's second sink): `proof_throughput/verity_proof_throughput/cli.py:213` and `flock/prover/python/verity_flock/class_statement.py:1458` (by name: `accepted`, `verify_s`), `flock/pod/85-rec-reprice.sh:303,306` (line count, `accepted`), `flock_s8_block/run.sh:156,158` (line count, a copy), `flock/pod/92-zk-sessions.sh:70` (prints it), `proof_throughput/job.sh:117` and `served_units/window.py:81` (keep the file), and three evidence lists (`bench.py:393`, `ir_bench.py:320`, `circuit_bench.py:278`). No `.lean` file and no Rust reads either line or the index.

**4. The one reader that compares whole records.** `flock/prover/tests/test_stream_zk_server.py:285 to 290` holds the stream's `LIVE` line against the process pair's with its `differ` (lines 74 to 96): the two records' keys must be the same set at every depth, two floats always pass, anything else must be equal. So a new field must be on both paths' lines (it is: both come from `session_finish`), must be a float on both (never null on one side, never an integer), and a nested object must have the same keys on both.

**5. What I checked for and did not find** [ran a scan, read each hit]: a reader that takes a field by position; a reader that compares the key set with a fixed list; a schema file naming the lines' fields. The shell `cut` sites cut the tag off and keep the whole JSON.

**6. What "appended" can mean here.** [read `live/Cargo.toml`; read our run's lines] `serde_json` is built without `preserve_order`, so the line prints its keys sorted and a new key lands by its name, not at the end. So: new names only; every existing key keeps its name, value and meaning; nothing is renamed or moved into an object. Names readers add themselves and I avoid: `challenger`, `wait_s_raw`, `cpu_s`, `rest_s`, `level0_wait_s`.

**7. proofs' candidate, from what the line already has.** [read] `HostSlots::pack_units` is called in `witness()` (flock-circuit.rs 1106), inside the span `witness_s` times (`witness_uploaded`, 4006). [read three sessions' `LIVE` lines of r20261112-160747-765d from the store] `witness_s` is 0.36 s in `first`, 0.15 s in `egress-0`, 0.75 s in `ingress-1`, of 24.5, 23.4 and 26.2 s between Register and the repetitions. So the packing does not fill it in our sessions. The server's line says as little: `handle_s` 0.10 to 0.14 s and `verify_s` 0.16 to 0.31 s a session. [argued, not tested] The sizes match another reading: under the firewall the prover connects the Lean session loop's finish for each session (`LeanFinish::connect`), and the Lean verifier's own reading of inputs and setup is 19.1 to 21.7 s a session on the same Program. The fields say yes or no.

**8. The fields I propose** (names are open until the PR; say if you or proofs want others):

- `LIVE`, one new key `steps_s`: an object of floats, one a step, laps of one clock from the instant before `Register` (the existing `t0`) to the instant the repetitions start (the existing `tp0`), so they add up to `total` with nothing unnamed: `register` (the `Register` round trip), `tables` (the drawn statement's tables), `lean_finish` (connecting the firewall's Lean finish; 0.0 without it), `config` (sigma, the Hello's config, salts, the spawns), `witness` (the witness loop's wall; `witness_s` stays what it is, inside it), `level0_wait`, `zk_pads_wait`, `level0_next`, `device_wait`, `total`.
- `SESSION` (and so `index.jsonl`), four new floats: `build_s` (the session's server built on its thread after accept), `record_s` (writing the record and proofs), `prover_wait_s` (blocked reading the prover's frames), `busy_s` (accept to the line, less `prover_wait_s`). `handle_s` and `verify_s` stay as they are.
- What the change adds to a session: clock reads, about ten in the prover and two a frame in the server. Nothing else moves.

**9. Where I build, as I read "the common rule"; say if it is otherwise.** Lean's branch taken once at `e209a56847d4` in a scratch worktree; my own scratch branch from it; nothing pushed to lean's branch. The run at the head needs `main`'s `benchmarks/scan_cost/`, which lean's tip lacks, so the run's tree is my scratch branch with `main` merged in. The PR comes from a branch on `main` once #9375 is there, unmarked. The run goes before 9 PM PST or after tonight's tries end, at G = 1 (44.8 minutes was the baseline's wall [read]) unless you want G = 8 (152 minutes).

**Not looked at:** which binary's line each reader reads (I list a reader whichever of the ten prints its line); readers outside this repository (a harness of proofs' or kernels' that lives in notes or on a pod); the fields each reader takes beyond a scan of its string literals and a read of its site (the table by file is in the notes, `lanes/network-accounting/` under this hour's stamp).

