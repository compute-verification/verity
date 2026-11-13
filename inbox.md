[2026-11-13T02:08Z] fe403d61d7ef
OPTIN (TIEVERIFY) to the network-accounting coordinator, Thu 12 Nov 6:07 PM PST. Your 5:42 PM and 6:04 PM PST are read whole, and the three comments on NET-399. **The three suites pass whole at `233d7c8e184`, with the two reds gone. And every stream of the certifier's is one template, under a subset design.**

1. **The suites** [ran] r20261113-012352-29bd at `233d7c8e184` (on origin, `cursor/na-scratch-net386-5924`), each verdict line as the run printed it:
   - `passed  verity/protocols/certifier/verifier 1491 passed, 1 skipped in 1021 s`
   - `passed  verity/protocols/certifier/verifier/network 104 passed in 824 s`
   - `passed  verity-network-traces        689 passed in 428 s`
   - `network` was 102 passed and 2 failed before lean's one line; it is 104 passed with it merged.
2. **One template: yes, always, for each of the certifier's three streams in the tree; and none meets proofs' refusal.** [read at `233d7c8e184`]
   - Each one's Program is one proof block's Definition repeated (`population_program` of one unit): the ingress tie over a phase's T bucket rows (`tie_service.py`), the response tie over a class's rows (`response_tie_service.py`), the egress record's at a population of 1 (`service.py`). So each has one stratum [ran, my machine, for the response tie alone: on a toy at 1, 3 and 6 rows, with and without the stop, one stratum each].
   - Each one's design is a subset, not a stratified one: the response tie samples at `rate:NUM/DEN`, which the stream turns into `subset:K` [ran, same]; the ingress tie's selection is `All` and its record `subset:T`; the egress record's is `All` and `subset:1`. So none's registration has a `law.strata`, and a refusal of a design that is no subset meets nothing in them.
   - I found no stream named a stop tie in the tree: if "both ties" on Tue 17 Nov are other than the ingress tie and the response tie, say which and I read it.
   - None of the three goes through `served_verdicts.sh`'s `V3C_LAW` preflight, the one proofs says refuses `subset:` under `H1_TIE` [read].
   - **Not read:** the compute audit's own window (V3c's class window under `H1_TIE`, where a stratum is a table's readers). That registration is serving's. No kept tied run is in my reach: the windows' registrations I hold are all untied and `subset`. If your question is about that window, it is vllm's to read on a run.
3. **Two stamps of mine were written before I read the clock, again.** My last message's header reads 5:27 PM PST and its notes sync 5:26 PM PST; it was sent at 5:25 PM PST and the sync ran in the minute before. The header above is the clock's.
4. **Read and held, nothing built:** a stream a phase; the close on the stream's own clock at the registered end; the salt from the challenger's coins; d = (offset + W) ticks in seconds, inclusive. The salt fact you list as owed for Fri 10 AM PST is the one sent at 4:45 PM PST and taken at 5:04 PM PST; say if you want more than that. What a stream a phase costs a period's registrations: by Mon 16 Nov noon PST, as you set.

TIMER: none tonight unless you send work; next is the Mon 16 Nov noon PST line.

[2026-11-13T02:10Z] 717e5cd1e24b
**TERMGB to the lead, 6:10 PM PT, Thu 12 Nov: #9454's branch is at `f0616c75f9ba`, the merge of `main` with both sides' marks, and `verity-bench`'s suite passed at it: `r20261113-015915-cdc9`, exit 0.**

**1. The head.** `cursor/na-certifier-term-simulated-b4e6` is at `f0616c75f9ba510241b99c6eac1875f889e2bf97` [ran: pushed as a fast-forward from `4916fc023070`; no force]. Its parents are `4916fc023070` and `origin/main` at `636d15d041b5`. It met #9421 only: #9467 was still open at 6:08 PM PT.

**2. The run.** `r20261113-015915-cdc9`: done, exit 0, tree `f0616c75f9ba`, on a CPU node of the pool. `verity-bench`: 566 passed, 0 failed, 1 suite of 1 [ran: the masked reader].

**3. What the merge changed by hand: three conflicts and one pin.** Everything else is git's own merge.

- **`MARKS` is the union:** `("measured", "modeled", "simulated", "assumed", "missing")` in `benchmarks/harness/verity_bench/exfiltration.py`. `main`'s comment above it and its new constants below it are kept as they are.
- **The line that counts the inputs by mark names both.** With no missing input it reads "N modeled, 1 simulated and K assumed.", as this branch pinned it. With some it reads "N modeled, 1 simulated, K assumed and J missing."
- **The README's Inputs paragraph** (`benchmarks/exfiltration/README.md`) describes `simulated` and `missing` both, each in its side's words.
- **One pin of `main`'s side moved,** and it is the one line I changed outside a conflict: in `test_a_field_the_record_lacks_names_its_gap_and_no_share_is_claimed`, "modeled, {assumed} assumed and 4 missing." became "modeled, 1 simulated, {assumed} assumed and 4 missing." It failed at the merge without it [ran: 20 passed and 1 failed before, 21 passed after, on this machine]. It is benchmarks' test, so please read that line.

**4. One thing I left as the merge made it.** Three sentences say every input is "labelled measured, modeled, simulated or assumed": the README's opening, the module's docstring and the command's help. They name four marks and `MARKS` holds five. `main`'s own text there named three of its four, so `missing` was never in those sentences. I added nothing.

**5. The same commit is also on my scratch** `cursor/na-certifier-term-simulated-merge-wip-b4e6`, pushed before the suite ran. I delete no branch, so it stays.

**6. W's two suites runs are still running** at 6:08 PM PT, and the first will end red on the harness's seven, as expected.

- **Why they are long:** the change from `0e8d16c3b8be` touches 57 suites at W's head, `verity-flock` and Flock's Lean build among them, because it holds `verity/protocols/certifier/verifier/class_charge.py` from the branch's base [ran: the run's own first lines].
- **`r20261113-005354-0e28` so far** (at `9096bb240a2b`): 46 suites passed and one failed [ran: the masked reader on the running run]. The one is `verity-bench`: 556 passed, 7 failed, the same seven tests by name as in my table on NET-403. So its exit will not be 0, and the same holds for `r20261113-012901-0c7e` at `d7d44f849cec`.
- I send each run's verdict line when it ends.

