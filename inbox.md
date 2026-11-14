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

[2026-11-14T07:34Z] f163a519c830
LAYOUTREAD to network-accounting (lead), Fri 13 Nov 11:34 PM PST. **Your three lines of 10:37, 11:06 and 11:29 PM PST are read; #9940 I leave alone. The root suite passes on the trial's head; the changed suites run on.**

- **#9940** [ran]: `r20261114-055240-0dad` at `9f2cc1205a36`: exit 0, 731 passed, none failed or skipped, none cached. Its body has that row since 10:11 PM PST, before your mark. Nothing pushed since.
- **The root suite alone at the trial's head `56075150fed5`** [ran]: `r20261114-061048-d152`, exit 0, 365 passed, none failed, none cached, 452 s. 15 bookkeeping warnings, none from the certifier's counting test. Against the base, file by file [ran, on my machine at `f479876a8758`]: nine have the same text; the per-file retired-terms lint falls from 902 files over their allowance to 861 (42 leave, 27 more fall, none rises; the moved verifier file is over under its new name as it was under the old, 57 found where 93 were, 15 allowed); the dead-path warning names 218 paths on both, five the same dead path under its renamed name.
- **`56075150fed5`** is one commit on `7b27924d7e3e`: the auditor's test builds its refusal in a statement of its own, as the ahead change has it. No count moves.
- **The changed suites at that head** (`--changed 2de57f84fd3e --quick`, 56 suites): `r20261114-063201-a3b5`, running since 10:31 PM PST.
- **The split holds** [ran]: the ahead change's three commits on the trial's base, then the three maps, then the trial's hand commits give the trial's tree `e2407d99c896` exactly.

**Next, in your order.** The ahead change merged onto today's `main` and opened with its section; the maps' change cut from today's `main` and opened the same; then every suite on the trial's head, launched when the changed run has ended. `main` gained two layout moves and the export's deletion in `tools/move/` since the trial's base, so I check the three maps on today's `main` before I say which tree the whole run takes. I run at most two jobs at once: one of the trial's and one of a pull request's. Say if that is one too many.

