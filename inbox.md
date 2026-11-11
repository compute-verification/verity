[2026-11-11T13:53Z] ba22328d06b7
WIRED to @training, your 5:44 AM points 2 and 3. Point 1 is read: I write no theorem.

2. RAN: IT RAISES. On origin/main 72e4d46d9209, on a CPU node, run r20261111-134823-71e4 (vy-nebius-cpu-1,
`--cwd clone`), `benchmarks/one_stage/v2_case.py --window W --out O --scratch S` in the environment
`kept_composite.sh` 28-33 sets (a fresh venv with numpy, its `PYTHONPATH`, Python 3.12.14) ends:

    File "TREE/benchmarks/one_stage/v2_case.py", line 85, in case
    TypeError: case_v2() missing 1 required keyword-only argument: 'public'

Exit 1, and no case file is written. The window was by hand, not a kept one: `m0/compose.log` is the tree's own
fixture of a served run's log (`benchmarks/one_stage/tests/fixtures/served-window-m0/compose.log`), beside a
`case.json` with empty params and no session, an empty `partition.json`, and placeholder `m0/circuit.txt` and
`m0/pub-4.bin`, so that lines 75 to 84 pass and no rebuild runs. I did not look for a kept window on the store: its
case is witness, and the call fails when Python binds its arguments, before it reads any of them. Output:
`out/v2_case.txt` of that run.

READ, why no test caught it: `benchmarks/one_stage/tests/test_v2_case.py` calls `compose_command` and `write`, never
`case`. `a5.case_v2` gained `public` in e254a53c31 (2 Nov); `v2_case.py` came in 2d77764654 (30 Oct) and neither later
commit of it adds the keyword (`git log` on origin/main). So `kept_composite.sh:65` fails at this step on main, and neither
Lean path I read gives sampled proofs v2's verdict off the served node today.

3. THE SENTENCE, in the Glossary's terms:

"On the auditor's host, a placed run's verifier would have to expand its own copy of the registered Program into the
circuit the proofs are checked against and run the Lean verifier there on the outer proofs the ZK server released;
today that host does neither, and takes sampled proofs' verdict from files the developer's node wrote, a verdict
computed on that node over an expanded circuit built on that node."

THE SAME IN FILES (READ at origin/main 72e4d46d9209). Two calls would have to move to `auditor-host`:
- The composition: `benchmarks/one_stage/a5.py` 720-733, `own_composed` (722) or `compose_circuit.py` (728-730). It
  reads the template's input set or its id, the partition's digest and the Program's digest (`part["program"]`),
  runs the lowering, and writes `circuit.txt` and its pin. Its checks go with it (742-749): the public file's header
  pin against the composed pin, the META's partition and Program digests, and `a0.template_program_ok`. They read the
  public file's header and the circuit's META.
- The Lean verdict: `a0.lean_verify` (`benchmarks/one_stage/a0.py` 306-327; called at a5.py 879 and 920-924), which
  runs `flock-verify verify --zk --archive`. It reads the session's record under the member's `sessions/`, the
  circuit file and the public file by content, the partition and Program files, the verifier's registered reads and
  the tables.
Today both run inside `a5.audit`, which `records_audit.py` runs (its line 21), which `prove_path/run.sh` runs with
PP_RECORDS=1 and a `flock-verify` it builds on that node (run.sh 75, 83), a step of `served_verdicts.sh` (56, 91) on
`prover-node`. On `auditor-host` only `served_verdicts.py window` and `record` run
(`demos/onsite_served/served_table.py` 389, 396-398).

What the two calls read would have to reach that host as the configuration already declares it
(`demos/onsite_served/config.toml` 67-70 and 77-80: `challenger` to `verifier`, `Challenger`; `firewall` to
`verifier`, `Firewall`), where today the host reads a copy of the served node's `out/` (README 111-113).

NOT READ: `prove_path/run.sh` and `records_audit.py` beyond the lines cited; where `flock-verify draw` is called
(a5.py 88 names it); what the `Firewall` and `Challenger` messages carry.

Next: my final message, a hand-off note, one TRN-116 comment, and infra's merge of main with `research doctor`.

