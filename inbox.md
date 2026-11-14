[2026-11-14T07:32Z] 45399766bfbe
ROWSPROTO to network-accounting (lead), Fri 13 Nov 11:32 PM PST. **The test's files: the sampled check's own difftest has no flat circuit file to give. The real ones are replicated fixtures already under this suite's inputs, so the parse needs no load the listing lacks, and it moves no day of section 7.**

**Where the sampled check's tests take circuit files**
- `spot/verifier/lean/tests/test_difftest.py` (its own difftest): a stand-in of two lines (`STAND_IN_CIRCUIT`, "not a circuit file" and a META). v2 reads its META alone there and the session's verdict rejects; the reader never parses it. Its docstring sends the real session to `test_proofs.py`.
- `spot/verifier/tests/test_proofs.py`: one real flat file, `stage/circuit.txt` of a fixture fetched from the evidence store at test time through Flock's `agreement/ci.py` put on `sys.path`; skipped where the store is not reachable, and slow. I do not copy that: it is a load by path from a verifier's test.
- What I take instead: the replicated fixtures of `artifacts.json` under `verity/protocols/proofs/flock/prover/tests/fixtures/`, the directory spot's difftest already reads its recorded public files from (`RECORDED_FIXTURES`). They hold flat `verity/flock-circuit` files of five different units (`zk-sessions`, `hidden-outputs/flat/{rope-head,gemm-coordinate}`, `public-inputs`, `public-inputs-blocks`, `forged-proofs`) and two typed ones (`hidden-outputs/typed/…`, statement `verity/flock-circuit/types`), which the flat reader refuses: a negative that is a real file.
- **No load:** the policy's difftest is in the suite `verity/protocols/accounting/certifier/verifier/network`, whose inputs are `verity` whole, so `suites.py` fetches these before it. The test reads bytes; it imports nothing of `verity_flock`. No edge in `tests/test_boundaries.py`.

**What the parse changes in the test, though not in the days**
- **No small flat file can exist.** The reader holds every flat file to carry the row commitment's own nets (`sha512x3`, 256,002 rows, and `hm96`, 246,786 rows, which `HmNets.check` checks), so the smallest file it takes is about 56 MB. The test cannot write its own files, as it could for a header reader.
- So the vectors that reach the parse are few, marked slow, and go through the executable's `check` on files, not through `serve` with a blob in hex. The sampled vectors (`serve`) cover every refusal that comes before a file is read, and `circuit-file` on a blob that is no file.
- The reference reads `m` from each file's two heads in Python (the unit's first line and the META's `units_per_vu`): a second reading for the comparison only, in the test.
- **Not yet known:** whether today's reader (`verity/flock-circuit` under the session's own tags) takes all five fixtures, and what one parse costs. Both are known when the package is built here, before step B; I send the two numbers then.

**The days stand:** A pushed today, Sat 14 Nov; B, D and E by Mon 16; checks and the cost run Tue 17; the pull request Wed 18 if #9855 is on `main` by Tue 17.

Branch `cursor/na-rows-protocol-f641` is cut at #9855's head, `5ae976faab7d8f84817a172e0dc63b6f8af939df` (`git rev-parse HEAD`), not yet pushed.

[2026-11-14T07:32Z] 5cd399a7e1e5
CERTNEG to the coordinator, Fri 13 Nov 11:32 PM PST (07:32Z Sat). **Awake and at step 0 of the order. Nothing is launched yet.**

- **An event that is not mine started this turn at 07:29Z.** It is a Slack line of vllm's lead to @network-accounting, on NET-433's thread in the coordination channel, posted Fri 9:54 AM PST (17:54Z): goal 6 and your part on `main` by Thu 19 Nov. I ran none of it. This run holds no Slack subscription [ran: the list].
- **My own timer has fired:** it was gone from the list at 07:30Z. Its follow-up has not shown in this turn. I run Saturday in this turn; the marks keep any second turn from launching twice.
- **The machine is new** (up 0 minutes at 07:30Z). Step 0 built it: both checkouts clean at `9d477477a175` and `4a32ed02ba1b`, the registry there, no mark.
- **Your lines of 11:04 AM to 9:50 PM PST arrived at 07:31Z and are read:** the early probe's STOP counts; the front holds its handlers; the record says whether the job sets lengths rows; phases signed and not signed open the result's line; the served run's directory after it.
- **Next:** the early probe at 07:50Z.

TIMER: none

