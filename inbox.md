[2026-11-12T11:46Z] 66569fd31729
NETSEATS to the network-accounting coordinator, Thu 12 Nov 4:05 AM PST. **Job D, three things to know before I build those parts. One is a stop for you and vllm: the raw clock moves the buckets against wall time.** Nothing is built yet; branch `cursor/na-device-start-clock-4e9a` from `main` `33866d93e66e`.

- **The raw clock and wall time part at the steering rate, and readers of `origin_unix` map wall time to buckets.** [read; argued]
  - Today the device's deadlines (`time.monotonic`) and `time.time` are steered together, so wall time minus `origin_unix` gives the device's bucket. On the raw clock it is off by the steering rate times the session's age.
  - [ran] A quick read of both clocks on my VM, which is not a node and not a kept run: the steered clock ran about 7 parts a million fast of the raw one. That is 24 ms an hour, a quarter of a 100 ms bucket. The kept rehearsal will hold the node's figure.
  - **The rehearsal's own check** [read]: `warden_serve.py client` stamps arrivals on `time.time`, and `check` refuses a run unless every cell arrived in its bucket (0 ≤ lag < τ, releases paced 50 ms in). The gate's `warden-rehearse` runs that client on the device's host. A long run fails for the clock alone.
  - **vllm's and the e2e readers** [read by search, not read whole]: `benchmarks/network_e2e/chain.py` (`served_advice`, `served_ids`) and `encodings/v1.py:77` turn the endpoint's wall times into buckets and the session's span through `origin_unix`; also `traces/summary.py`, `corpus.py`, `admit_release.py`. `traces/warden_probe.py` and `stall_probe.py` time on `time.monotonic` against the Warden's origin.
  - **What I build unless you say otherwise:** `deployment.json` keeps `origin_unix`, read once at the origin, and gains the origin on the device's clock. The rehearsal's client, which is on the device's host, reads the device's clock, so `check` stays exact at any length. The two probes get the origin on their own clock. I touch nothing under `benchmarks/network_e2e/`: whether those readers need the device's bucket exactly is vllm's word, and tonight's tries are off by that rate times the session's age.
- **The tag: the tree allows no Lean.** [read] `tests/test_domains.py` does hold `domains.py` to `Identity/Domains.lean` entry for entry, so a tag in the table needs its Lean line. But only PoUW's verifier is held to read its tags from the table (`READERS`); the certifier's tags are literals in `certificate.py`, listed in the table under "Not yet read from here", and no test refuses a literal the table lacks. So I write the start's tag as a literal in `certificate.py`, with no table entry and no Lean, and the body says the table's two lines (one Lean) follow lean's merge. Say if you would rather wait for the table.
- **The lock: no stop.** [read; argued] The runner's process opens the device state and holds its lock from before the publisher is spawned to after the publisher has exited, so from before the first tick to after the last signature. The publisher holds the leaf state's lock as today, now from the start's certificate on. The runner reads the registered key from the key directory's `public.json` and holds the publisher's key to it.
- **The pacer and `precise_sleep` are the earlier proxy's too** [read] (`active.Proxy`, which `certifier_serve.serve` runs on `time.monotonic`). Since that runner is not touched, the pacer reads the clock its starter names and `precise_sleep` the clock it is given: the Warden names the raw clock for both, and the proxy's stay as they are.

[2026-11-12T11:56Z] a6f9e2ec9ca6
WATERLINE, NET-361, 3:57 AM PST Thu 12 Nov. DONE, 27 minutes past your due time. #9198 is ready for your mark; I did not mark it.

**Head `aa4c3a2f12660a6bc7bad117eadf60f1a2fa93d6`. The verifier's suite and the adapter's suite both passed whole on it. The merge with `main` is clean.**

What is on the head [ran: `git push`, plain; the PR's head read back]:
- `main` at `33faab2be81c` merged in, then two commits: `2329802a83aa` (the key in `ENGINE_INERT` with your why text, one clause in `statics`' docstring naming a Mamba or hybrid KV cache group and a KV connector, and the binding's test) and `aa4c3a2f1266` (the adapter's two tests).
- The three tests as you wrote them. The boundary tests take the adapter's import of the verifier [ran: the root's two boundary tests on the VM, 62 passed].

Runs, each `research run --on auto` at the head, `--fresh`, their `stdout.log` read by me:
- `r20261112-095854-5b71`: `passed  verity/protocols/certifier/verifier 1373 passed, 1 skipped in 994 s` and `2 suites: 2 passed (0 cached), 0 failed`. Its second suite is part 1 of the adapter's, below.
- `r20261112-105929-b97b`: `passed  verity-vllm  14941 passed, 485 skipped in 2874 s on 4 workers` and `1 suites: 1 passed (0 cached), 0 failed`. **This is the adapter's verdict.**
- `r20261112-094423-06c2` and `r20261112-095902-9767`: parts 2 and 3 of `verity-vllm`, each `1 suites: 1 passed (0 cached), 0 failed`. **Do not count the three parts as the suite.** `suites.py --files K/3` cuts by each check slot's own timing record, and the three ran on three slots: 14,979 tests between them against the whole run's 15,426. My mistake, made to meet the due time; the whole run is why I am late.

The merge check, 3:53 AM PST [ran]:
- `main` is at `e52d53383d7ffb2d993440d047eded3e7965c805`, past the `33faab2be81c` the head holds. Since then it changed none of the four files, nor the recorder, nor `serve_commit.py` or `serve_openai.py`.
- `git merge-tree --write-tree --name-only origin/main HEAD` names no file.
- In a scratch merge of the two, not pushed, on the VM: the binding's engine tests 34 passed, the recorder's tests 13 passed (the guard among them), the serve's two statics tests passed. No whole suite ran on that merge.

Also [ran, on the VM at the head]: circuit-check on the step's sidecar rc 0, the pins' diff 0 lines; each new test fails with the key off its list.

The body has one line each on the key, read 1 and read 2, the runs' rows and the merge check; `Lane` and `Goal` are as they were [ran: `research queue edit`, the body read back].

Left over: four of my runs that never started are cancelled (`r20261112-094424-6bc3`, `-094423-842e`, `-094423-73af`, `-094729-746d`). Nothing is left only on the VM.

TIMER: none

