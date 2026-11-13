# Goals round, Fri 13 Nov 2026 (TOP's issue names this file)

Daniel, 8:06 AM PST: "It seems like they're going to need restating and cleaning up. Also, it seems like we've kind of lost
track of what we mean by a tier one goal. I think there's like too many tier one goals here. But also, broadly, yeah, I'm
still mainly interested in like reorganizing the repository around like the new architecture and kind of like implementing
that new architecture rather than like getting other results out. Can we have a round of re-specifying the goals?"

The page at 8:06 AM: 259 open goals; 41 at the top level (15 set there by hand, 26 only because nothing waits on them);
79 due today; 129 with a moved date.

## What I proposed to Daniel (his word is not in yet)

Tier 1: a state of the whole system that Daniel named, that one command or one page shows true or false, and that other
goals wait on. Only his word adds one, and there are at most five. Every other goal sits under one of them, is parked
with the state it stopped in, or is dropped.

| | Tier-1 goal | Owner | What shows it |
| --- | --- | --- | --- |
| 1 | The repository has the architecture's layout | top | `tests/tree.toml`'s not-yet lines (L1 to L10 below) at zero |
| 2 | The architecture is implemented | top holds it, one lead per row | the architecture file's rows of code still to be written (R1 to R24 below) at zero; each area's code at the bar |
| 3 | One dashboard shows every number | console | every benchmark's numbers, the counts of 1 and 2, the proofs' status |
| 4 | A change lands within two hours | ci, with infra's platform under it | ci's daily row |

Parked as one group, nothing deleted: the GLM-5.3 network number, inference-only on a small model, vLLM's serving cost and
the 1% target, proof system performance, the climbs, the SPOT post, "fully proved". The two demos stay only as R6.
Tonight's two builds run as built; nothing new starts for them.

## Your answer: one table, a row for each of your open goals

| goal | K1, K2, K3, K4, P or D | the line or row it closes or serves | new title and its check, if restated | a day you can hold |

- **K1 to K4**: keep, under that goal. Name the L line or R row it closes, or the one it serves.
- **P**: park. A result, a climb, or work the architecture does not need now. Say in a few words what state it stops in.
- **D**: drop. Done, a duplicate, or no longer meant.
- My first sort is in your section below. `P?` means only that the goal sits under a result today: correct it where
  the work is the architecture's (a port, an interface, a verifier moving to Lean).
- Under the table: each L line or R row that is yours and has no goal (write its goal now); each goal of another lead's
  that yours needs and would lose if parked; one line if the frame is wrong.

Restating: a title is a state of `main` someone else can check, and its description opens with the check (a test, a
path, a row deleted, a number on the dashboard). A goal due today that will not be true at 6 PM gets the day you can
hold now, with its reason. You may restate a kept goal's title, description and date at once. Do not change a tier or a
parent, and park nothing on the page, until I write on the issue that Daniel confirmed the list.

## The layout's not-yet lines (`tests/tree.toml` at #9742's head)

| | Path | Goes to | Owner | Day |
| --- | --- | --- | --- | --- |
| L1 | `verity/protocols/reporting.py` | into verity/protocol.py | top | 2026-11-15 |
| L2 | `verity/hardware/registration.py` | into the hardware file | top | 2026-11-15 |
| L3 | `verity/hardware/record.py` | into the hardware file | top | 2026-11-15 |
| L4 | `verity/hardware/lean.py` | deleted, once the Lean takes a card as a parameter | compute-research | 2026-11-24 |
| L5 | `verity/definitions/` | its protocol part to verity/circuits/, its served part to the vLLM adapter | top | 2026-11-15 |
| L6 | `verity/claims/` | not ruled, with soundness: it defines the assumption ids that Lean and the locks name, so it is not deleted | top | 2026-11-17 |
| L7 | `verity/periods/` | verity/protocol.py | top | 2026-11-15 |
| L8 | `fixtures/` | beside its reader, each file | proofs | 2026-11-18 |
| L9 | `verity/protocols/accounting/pouw/clear/` | verity/protocols/accounting/pouw/verifier/ | compute-accounting | 2026-11-19 |
| L10 | `verity/protocols/proofs/flock/sigma/` | its protocol's verifier/, prover/ or device/, or out of verity/; proofs says which | proofs | 2026-11-18 |

## The architecture file's rows of code still to be written (`ARCHITECTURE.md` section 7 at #9702's head)

| | What | Today |
| --- | --- | --- |
| R1 | Every protocol has parties | Only SPOT, PoUS and the remote memory-read audit do |
| R2 | A PyTorch adapter under the vLLM adapter, with one translator from PyTorch to circuits | One adapter, and a second translator inside the training benchmark |
| R3 | The messages between workload and prover as one stated list | Function calls and files; 15 files with no settled name or place |
| R4 | PoUW's serving code and GPU kernels in one package under `protocols/accounting/pouw/`, reached by one call | In the adapter (18,800 lines) and under `benchmarks/` |
| R5 | Benchmarks and demos reach the adapter through its public interface only | 114 modules are reached past it |
| R6 | The two demos start from a configuration, run by the runtime | The remote demo is a shell script under `benchmarks/`; the onsite demo is not built |
| R7 | Python never returns accept or reject | Several protocols still decide in Python. SPOT's two-stage outcome is joined in Python; its Lean check is written and refuses every audit until its pr… |
| R8 | Each accounting protocol takes the registered hardware and says which part it accounts for | None does |
| R9 | The security model has the auditor's devices as parties, each with a key the auditor pins and its whole signing history | Only the challenger is a party, so no onsite memory or network claim can be proved yet |
| R10 | A device's figures have one source: the registration reads them from the auditor's data sheet | Written in two places that disagree (the H200's tensor rate at two clocks) |
| R11 | "Registration" names one thing: SPOT's own record of a window takes another name | SPOT's code calls the record it fixes before any challenge a registration record: the circuit's digest, its pieces, the sampling rule and the committe… |
| R12 | The circuit checker is handed the served circuits and names none | It names them |
| R13 | The compiler cuts a period's circuit into pieces wired to each other | For the demo's model the pieces are wired only where the eight GPUs combine results, so the audit's second round runs over a stand-in |
| R14 | Every kernel the production settings serve has a circuit held to the kernel on an H200 | Five are written. Four are compared with the kernel on an H200 and no word differs: the FP8 cache's write, its two reads, and the accept rule. Two of … |
| R15 | Every kind of piece in a request has a statement the Lean verifier accepts | 18 of 254 kinds, holding 88.1% of the pieces |
| R16 | One form of the network certifier | Two: the remote demo runs an older one, a proxy in front of the server; the onsite demo runs the device on the link |
| R17 | The Lean takes a data sheet as a parameter | Two Lean tables restate their data sheets |
| R18 | Every benchmark on the dashboard | 5 of 69 wired; 3 of the 5 show a value no older than their schedule allows |
| R19 | The proofs' status on the dashboard: guarantees fully proved, open premises per lead | Printed by the Lean check only |
| R20 | The Lean check passes on the public repository | On Tuesday's export 14 of 16 Lean packages pass |
| R21 | Every area of `security/` has a lead | 826 of its 4,816 files fall to lean because no lead claims them |
| R22 | A memory-read session is proved in zero knowledge | Each span's openings go to the verifier as they are |
| R23 | PoUS's setup is a proof about the store | The check reads the whole store |
| R24 | PoUS's session is checked inside a proof | The check reads the memory challenger's signed record as it is: for each round, which block was asked, when it was sent and answered, and the block's … |

# Each lead's open goals, with my first sort

## benchmarks: 6 open goals (? 4, P? 2)

| goal | title | due | state | at the top today | first sort | because it is under |
| --- | --- | --- | --- | --- | --- | --- |
| `benchmarks/one-harness` | Every benchmark uses one harness | 11/17 | on-track |  | P? | `benchmarks/proof-system-performance` |
| `benchmarks/proof-system-performance` | Proof system performance benchmarked | 11/17 | on-track | set by hand | P? |  |
| `benchmarks/head-key-blocks-as-units` | Head key blocks priced as served units | 11/13 | on-track |  | ? |  |
| `benchmarks/every-number-marked` | Every benchmark number carries its mark | 11/13 | on-track |  | ? |  |
| `benchmarks/one-stage-line-dropped` | One-stage every-block line out of the print | 11/16 | on-track |  | ? |  |
| `benchmarks/six-directories-wired` | Benchmarks' six directories declare their numbers | 11/17 | on-track |  | ? |  |

## ci: 13 open goals (K4 13)

| goal | title | due | state | at the top today | first sort | because it is under |
| --- | --- | --- | --- | --- | --- | --- |
| `ci/night-lean-batches` | Tonight's Lean batches reach main | 11/13 | on-track |  | K4 | `ci/lean-landing-two-hours` |
| `ci/check-shards-balanced` | Check runs in five balanced parts | 11/13 | on-track |  | K4 | `ci/landing-two-hours` |
| `ci/review-verdicts-signed` | Review verdicts signed by their readers | 11/13 | on-track |  | K4 | `infra/platform` |
| `ci/slow-part-spares` | Slow check parts get spares | 11/14 | on-track |  | K4 | `ci/landing-two-hours` |
| `ci/reviewed-landing` | Only reviewed changes can land | 11/16 | on-track |  | K4 | `infra/platform` |
| `infra/full-check-45-min` | Check finishes within 45 minutes | 11/16 | off-track |  | K4 | `ci/landing-two-hours` |
| `ci/landing-two-hours` | Plain changes land within two hours | 11/17 | on-track | set by hand | K4 |  |
| `ci/old-queue-deleted` | Old queue code deleted | 11/17 | on-track |  | K4 | `ci/check-in-tools-check` |
| `ci/tools-start-fresh` | Every tool starts from fresh copy | 11/17 | on-track | set by hand | K4 |  |
| `ci/check-in-tools-check` | Check lives in tools/check | 11/18 | on-track | nothing waits on it | K4 |  |
| `ci/no-cold-lean-build` | No batch builds Lean from cold | 11/18 | off-track |  | K4 | `ci/lean-landing-two-hours` |
| `ci/lean-landing-two-hours` | Lean changes land within two hours | 11/19 | off-track | set by hand | K4 |  |
| `ci/platform-repo-queue` | Platform repository lands through the queue | 11/19 | on-track | set by hand | K4 |  |

## circuits: 9 open goals (? 6, K1/K2 3)

| goal | title | due | state | at the top today | first sort | because it is under |
| --- | --- | --- | --- | --- | --- | --- |
| `circuits/checker-handover-in-review` | Checker's hand-over changes in review | 11/13 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `circuits/checker-names-no-served-circuit` | Circuit checker names no served circuit | 11/14 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `circuits/slice-clean` | Circuit code cleaned up for review | 11/20 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `circuits/request-blocks-lean-accepted` | Request's block types accepted by Lean | 11/13 | on-track |  | ? |  |
| `circuits/long-row-serving-waiting-to-merge` | Long-row serving change waiting to merge | 11/13 | on-track |  | ? |  |
| `circuits/production-settings-as-circuits` | FP8 cache and speculative decoding written as circuits | 11/16 | on-track |  | ? |  |
| `circuits/request-builder-one-gpu` | One GPU's share of the request built | 11/16 | on-track |  | ? |  |
| `circuits/programs-not-unrolled` | Verifier reads compact program files | 11/17 | on-track | nothing waits on it | ? |  |
| `circuits/request-built-as-proof-blocks` | GLM-5.3 request built as proof blocks | 11/19 | on-track | nothing waits on it | ? |  |

## comms: 5 open goals (? 1, K4 3, P? 1)

| goal | title | due | state | at the top today | first sort | because it is under |
| --- | --- | --- | --- | --- | --- | --- |
| `comms/review-fri` | Friday's messaging review published | 11/13 | on-track |  | K4 | `infra/platform` |
| `comms/no-personal-tokens` | Messaging holds none of Daniel's tokens | 11/13 | on-track |  | K4 | `infra/platform` |
| `comms/agents-coordinate` | Agents coordinate without Daniel | 11/13 | on-track |  | K4 | `infra/platform` |
| `comms/spot-differences-placed` | SPOT draft's differences on the page | 11/13 | on-track |  | P? | `proofs/spot-post-ready` |
| `comms/device-term-rolled-out` | Daniel's term device used everywhere | 11/14 | on-track |  | ? |  |

## compute-accounting: 17 open goals (K1/K2 1, K2 5, P? 8, parked 3)

| goal | title | due | state | at the top today | first sort | because it is under |
| --- | --- | --- | --- | --- | --- | --- |
| `compute-accounting/pouw-package-holds-kernels` | PoUW package holds its GPU code | 11/19 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `compute-accounting/pouw-code-cleaned` | PoUW's code meets the code rules | 11/17 | on-track |  | K2 | `compute-accounting/pouw-in-page-shape` |
| `compute-accounting/pearlplus-in-remote-theorem` | Remote demo's proof covers Pearl+ | 11/18 | on-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `compute-accounting/pouw-parties-and-files` | PoUW's parties and files in place | 11/19 | on-track |  | K2 | `compute-accounting/pouw-in-page-shape` |
| `compute-accounting/onsite-proof-covers-pouw` | Onsite demo's proof covers Pearl+ | 11/20 | on-track |  | K2 | `compute-accounting/pouw-in-page-shape` |
| `compute-accounting/pouw-in-page-shape` | PoUW in the page's shape | 11/20 | on-track | set by hand | K2 |  |
| `compute-accounting/hardware-sheets-as-parameters` | Lean reads hardware data sheets as parameters | 11/16 | on-track |  | P? | `compute-accounting/compute-share-climb` |
| `compute-accounting/twice-the-tiles` | Twice the sampled tiles a window | 11/17 | on-track |  | P? | `compute-accounting/compute-share-climb` |
| `compute-accounting/pouw-verdict-from-main` | PoUW verdict accepts from main | 11/18 | on-track |  | P? | `compute-accounting/compute-share-climb` |
| `compute-accounting/excluded-rows-credited` | Massive-activation rows credited | 11/19 | on-track |  | P? | `compute-accounting/compute-share-climb` |
| `compute-accounting/pearlplus-tiles-proved` | Pearl+ check runs on the auditor's machine | 11/20 | on-track |  | P? | `compute-accounting/compute-share-climb` |
| `compute-accounting/lm-head-credited` | LM head credited by Pearl+ | 11/30 | on-track |  | P? | `compute-accounting/compute-share-climb` |
| `compute-accounting/compute-share-97` | 97% of the small model's work verified | 12/02 | on-track |  | P? | `compute-accounting/compute-share-climb` |
| `compute-accounting/compute-share-climb` | Small model's verified compute share reaches 97% | 12/02 | on-track |  | P? |  |
| `compute-accounting/h200-accounted-page` | Daniel's page on H200 accounting | 11/13 | on-track |  | parked | `top/parked` |
| `compute-accounting/h100-debit-mended` | Pearl-C's debit closes red team's breaks | 11/20 | off-track |  | parked | `top/parked` |
| `accounting/h200-97-percent` | 97% of H200 capacity accounted for | 11/30 | off-track |  | parked | `top/parked` |

## compute-research: 17 open goals (K1/K2 5, K2 6, parked 6)

| goal | title | due | state | at the top today | first sort | because it is under |
| --- | --- | --- | --- | --- | --- | --- |
| `compute-research/data-sheets-moved` | Hardware data sheets moved to a top-level directory | 11/13 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `compute-research/data-sheet-reader-folded` | One reader for hardware data sheets | 11/13 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `compute-research/dead-lean-deletion-built` | Dead PoUW Lean deletion in review | 11/13 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `compute-research/lead-leftovers-deleted` | Dead PoUW Lean deleted from main | 11/13 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `compute-research/code-at-overhaul-bar` | Compute-research code cleaned up for review | 11/20 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `compute-research/machine-code-check-moved` | Kernel machine-code check moved beside PoUW's prover | 11/13 | on-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `compute-research/merge-held-files-cleaned` | Compute files held by merge cleaned | 11/13 | on-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `compute-research/pearl-names-renamed` | Pearl's cap and debit renamed to inert work | 11/13 | on-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `compute-research/compute-code-consolidated` | Compute accounting code consolidated and cleaned up | 11/18 | on-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `compute-research/lean-takes-data-sheet` | Machine model takes data sheet as parameter | 11/18 | on-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `compute-research/accounting-code-vllm-quality` | Accounting code brought to vLLM's quality | 11/27 | off-track | nothing waits on it | K2 |  |
| `accounting/silicon-logs-all-compute` | Compute-logging chip specified | 11/30 | off-track |  | parked | `top/parked` |
| `accounting/useful-weights-hard` | Useful weights resist matmul shortcuts | 11/30 | off-track |  | parked | `top/parked` |
| `compute-research/chain-premise-probed` | Logging hash's security assumption tested | 11/30 | off-track |  | parked | `top/parked` |
| `compute-research/chain-round-pinned` | Logging hash written down exactly | 11/30 | off-track |  | parked | `top/parked` |
| `compute-research/first-four-layers-numbers` | First training results on 70B | 11/30 | off-track |  | parked | `top/parked` |
| `compute-research/hardness-train-at-point` | Quality loss recovered by training | 11/30 | off-track |  | parked | `top/parked` |

## console: 7 open goals (K3 6, K4 1)

| goal | title | due | state | at the top today | first sort | because it is under |
| --- | --- | --- | --- | --- | --- | --- |
| `console/eleven-numbers-live` | The eleven's numbers live on the console | 11/13 | off-track |  | K3 | `console/one-dashboard` |
| `console/numbers-format-written` | How a benchmark's numbers are declared, written down | 11/13 | on-track |  | K3 | `console/one-dashboard` |
| `console/numbers-listed-per-benchmark` | Each benchmark lists its numbers in its directory | 11/17 | on-track |  | K3 | `console/one-dashboard` |
| `console/bounds-per-period` | Each bound shows per week, month and year | 11/18 | on-track |  | K3 | `console/one-dashboard` |
| `console/one-dashboard` | One dashboard shows every benchmark's numbers | 11/18 | on-track | set by hand | K3 |  |
| `console/proofs-status-on-dashboard` | Dashboard shows what each guarantee rests on | 11/18 | on-track |  | K3 | `console/one-dashboard` |
| `console/routes-check-caller` | Every docs-site route checks who calls it | 11/13 | on-track |  | K4 | `infra/platform` |

## infra: 15 open goals (? 1, K1/K2 1, K4 13)

| goal | title | due | state | at the top today | first sort | because it is under |
| --- | --- | --- | --- | --- | --- | --- |
| `infra/slice-clean` | Infra code cleaned up for review | 11/13 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `infra/nightly-second-night` | Second night's check names no false disagreement | 11/13 | on-track |  | K4 | `infra/nightly-uncached-check` |
| `infra/quick-runs-start-in-five-minutes` | Quick runs start within five minutes | 11/13 | off-track |  | K4 | `infra/platform` |
| `infra/disk-remake-procedure-written` | Disk remake procedure written, first hour agreed | 11/13 | on-track |  | K4 | `infra/platform` |
| `infra/nightly-uncached-check` | Nightly uncached check confirms cached results | 11/13 | on-track | nothing waits on it | K4 |  |
| `infra/queue-service-dry-run` | Queue service dry-runs on its new machine | 11/13 | on-track |  | K4 | `infra/platform` |
| `infra/runs-start-or-say-why` | Runs start within ten minutes | 11/13 | on-track |  | K4 | `infra/platform` |
| `infra/old-runs-deleted-hourly` | Old run files deleted hourly everywhere | 11/16 | off-track |  | K4 | `infra/platform` |
| `infra/four-disks-remade` | More file slots on four check machines | 11/17 | off-track |  | K4 | `infra/platform` |
| `infra/infra-repo-lands` | Infra repository merges without Daniel | 11/17 | off-track |  | K4 | `infra/platform` |
| `infra/launchers-off-control-pod` | Check launches leave the control pod | 11/17 | off-track |  | K4 | `infra/platform` |
| `infra/message-reaches-agents` | Messages reach every running agent | 11/17 | off-track |  | K4 | `infra/platform` |
| `infra/platform` | Platform reliable and secure | 11/20 | off-track | set by hand | K4 |  |
| `infra/security-posture-finalized` | Security posture finalized | 11/20 | off-track |  | K4 | `infra/platform` |
| `infra/executor-slot-stays-busy` | Checker slot starts its next task at once | 11/13 | on-track |  | ? |  |

## kernels: 13 open goals (K1/K2 7, P? 6)

| goal | title | due | state | at the top today | first sort | because it is under |
| --- | --- | --- | --- | --- | --- | --- |
| `kernels/gpu-code-move-planned` | GPU code move planned with dates | 11/13 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `kernels/int8-check-on-served-kernel` | Exact int8 check runs the served kernel | 11/14 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `kernels/builds-unchanged-by-moves` | Kernel builds unchanged by file moves | 11/17 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `kernels/code-at-overhaul-bar` | Kernel code cleaned up for review | 11/20 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `kernels/gpu-code-out-of-benchmarks` | GPU code moved out of benchmarks | 11/20 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `kernels/timed-rows-from-harness` | Timed GPU runs from shared harness | 11/20 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `kernels/unused-benchmarks-deleted` | Unused kernel benchmarks deleted | 11/20 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `proofs/spot-proof-server-steady-state` | Steady-state proving cost measured | 11/13 | on-track |  | P? | `proofs/exfiltration-under-300gb` |
| `kernels/overhead-benchmark-gpu-check` | Overhead benchmark passes its GPU check | 11/13 | on-track |  | P? | `compute-accounting/compute-share-climb` |
| `kernels/overhead-counts-output-commitment` | PoUW overhead counts the output's commitment | 11/17 | on-track |  | P? | `compute-accounting/compute-share-climb` |
| `kernels/wide-leaf-measured` | Flock's wide leaf measured on a GPU | 11/17 | on-track |  | P? | `proofs/exfiltration-under-300gb` |
| `kernels/witness-on-the-card` | Deep blocks' witness built on GPU | 11/18 | on-track |  | P? | `proofs/exfiltration-under-300gb` |
| `kernels/row-digest-takes-head-rows` | Row commitment takes LM-head rows | 11/24 | on-track |  | P? | `compute-accounting/compute-share-climb` |

## lean: 16 open goals (? 6, K1/K2 2, K4 4, P? 4)

| goal | title | due | state | at the top today | first sort | because it is under |
| --- | --- | --- | --- | --- | --- | --- |
| `lean/export-change-finished` | Public export lists statements without proofs | 11/14 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `lean/public-locks-clean` | Lean check passes on public repository | 11/17 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `lean/prs-carry-no-records` | Lean PRs carry no lock records | 11/13 | on-track |  | K4 | `proofs/proof-check-fast` |
| `lean/check-reuses-past-results` | Lean check reuses more past results | 11/16 | on-track |  | K4 | `proofs/proof-check-fast` |
| `lean/build-cannot-rewrite-checked-files` | Lean build cannot rewrite checked files | 11/17 | on-track | nothing waits on it | K4 |  |
| `proofs/proof-check-fast` | Lean check finishes within 15 minutes | 11/17 | off-track | nothing waits on it | K4 |  |
| `lean/refused-lemma-answer-read` | Refused lemma's answer read | 11/13 | on-track |  | P? | `compute-accounting/compute-share-climb` |
| `lean/policy-proved-and-open-listed` | Policy's proved and open parts listed | 11/14 | on-track |  | P? | `training/small-model-slowdown-1000` |
| `lean/witness-held-to-its-guarantee` | Each guarantee's witness is checked against that guarantee | 11/17 | on-track |  | P? | `compute-accounting/compute-share-climb` |
| `lean/inference-only-on-served-traffic` | Inference-only policy proved for served traffic | 11/25 | off-track |  | P? | `training/small-model-slowdown-1000` |
| `lean/two-stage-read` | Lean verifier reads and refuses two-stage audits | 11/14 | on-track |  | ? |  |
| `lean/verdict-verifiers-page` | Page lists which verifier decides each verdict | 11/16 | on-track | nothing waits on it | ? |  |
| `lean/honest-session-proof-sized` | Honest signed session's proof sized | 11/16 | on-track | nothing waits on it | ? |  |
| `lean/security-owners-file` | Owners file names each security area's lead | 11/17 | on-track |  | ? |  |
| `lean/no-guarantee-without-accepted-run` | No new guarantee without an accepted run | 11/18 | on-track | nothing waits on it | ? |  |
| `lean/security-areas-protocol-names` | Security areas named after their protocols | 11/18 | on-track | nothing waits on it | ? |  |

## memory-accounting: 15 open goals (? 4, K1/K2 4, K2 4, P? 3)

| goal | title | due | state | at the top today | first sort | because it is under |
| --- | --- | --- | --- | --- | --- | --- |
| `memory-accounting/provers-out-of-benchmarks` | Both memory protocols' provers out of the benchmarks | 11/13 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `memory-accounting/verifier-tests-off-benchmarks` | Verifier tests stop loading benchmark code | 11/13 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `memory-accounting/code-consolidated` | Memory accounting's code consolidated, lists at zero | 11/17 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `memory-accounting/held-cleanup-after-remote` | Memory accounting code cleaned up | 11/24 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `memory-accounting/storage-audit-reads-bf16` | Storage audit reads the served checkpoint | 11/16 | on-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `memory-accounting/verdicts-tied-to-auditor-device` | Memory verdicts tied to auditor's device | 11/17 | on-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `memory-accounting/wipe-signs-start-and-end` | Wipe's start and end each signed once | 11/17 | on-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `memory-accounting/onsite-memory-verdicts` | Memory verdicts on the onsite record | 11/24 | off-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `memory-accounting/audit-cost-at-100-bits` | Cost of a 100-bit memory audit measured | 11/17 | on-track |  | P? | `memory-accounting/share-to-target` |
| `memory-accounting/served-share-with-host-memory` | Uncertified memory share at most 0.17 | 11/18 | on-track |  | P? | `memory-accounting/share-to-target` |
| `memory-accounting/share-to-target` | Uncertified memory share reaches its target | 11/18 | on-track |  | P? |  |
| `memory-accounting/grader-out-of-verity` | PoUS grader moved out or deleted | 11/13 | on-track |  | ? |  |
| `memory-accounting/halves-out-of-verifiers` | Provers' halves leave the verifiers | 11/14 | on-track | nothing waits on it | ? |  |
| `memory-accounting/runtime-passes-to-proofs` | Runtime and shared types pass to proofs | 11/15 | on-track |  | ? |  |
| `memory-accounting/onsite-poub-parties` | Onsite PoUB states its parties | 11/17 | on-track |  | ? |  |

## network-accounting: 24 open goals (? 6, K1/K2 6, K2 5, P? 5, parked 2)

| goal | title | due | state | at the top today | first sort | because it is under |
| --- | --- | --- | --- | --- | --- | --- |
| `network/policy-calculators-moved` | Inference-only policy's calculators moved to benchmarks | 11/13 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `network/parties-stated` | Certifier and inference-only policy state their parties | 11/16 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `network/certifier-leaves-benchmarks-plan` | Plan for one network certifier outside benchmarks | 11/16 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `network/certifier-programs-leave-benchmarks` | Served certifier's programs out of benchmarks | 11/17 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `network/certifier-public` | Network certifier published under one name | 11/20 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `network/slice-clean` | Network accounting code cleaned up for review | 11/27 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `network/view-charge-prints-ruled-value` | Check prints the ruled value on Monday's data | 11/13 | on-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `network/port-recount-and-first-cuts` | Onsite port recounted and first changes built | 11/13 | on-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `network/record-at-tight-figure` | Run record reports the tighter exfiltration capacity | 11/16 | off-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `network/onsite-port-staged` | Onsite network files on main | 11/20 | off-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `network/egress-tied-to-trace` | Network check ties responses to committed outputs | 11/24 | off-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `network/term-in-the-number` | Benchmark reports network certification's exfiltration capacity | 11/13 | on-track |  | P? | `accounting/network-glm53` |
| `network/run-end-arrivals-turned-away` | Certifier turns away arrivals at a run's end | 11/16 | on-track |  | P? | `proofs/exfiltration-under-300gb` |
| `network/policy-check-counts-rows` | Inference-only check counts blocks by circuit rows | 11/16 | on-track |  | P? | `training/small-model-slowdown-1000` |
| `network/certifier-term-zero` | Network certification adds no exfiltration capacity | 11/23 | off-track |  | P? | `proofs/exfiltration-under-300gb` |
| `network/period-bound-checked` | Network bound counts only unchecked inputs | 11/23 | on-track |  | P? | `proofs/exfiltration-under-300gb` |
| `network/scan-count-statements-restated` | Scan count statements restated as soundness amended | 11/13 | on-track |  | parked | `top/parked` |
| `network/tight-figure-proved` | Onsite exfiltration capacity proved in Lean | 11/24 | off-track |  | parked | `top/parked` |
| `network/certifier-signs-session-start` | Network certifier signs each session's start | 11/13 | on-track |  | ? |  |
| `network/cheats-tried-on-served-run` | Six cheating servers tried against a served run | 11/13 | on-track |  | ? |  |
| `network/wipe-span-counted` | Certifier counts what left during a wipe | 11/17 | on-track |  | ? |  |
| `network/onsite-audits-timed` | Onsite network audits timed at the demo's scale | 11/18 | on-track |  | ? |  |
| `network/row-from-registered-site` | Network's accounting row read from the registered site | 11/20 | on-track |  | ? |  |
| `network/check-reads-verified-answers` | Network check accepts only verified audit answers | 12/01 | on-track |  | ? |  |

## proofs: 33 open goals (K1/K2 2, P? 29, parked 2)

| goal | title | due | state | at the top today | first sort | because it is under |
| --- | --- | --- | --- | --- | --- | --- |
| `proofs/slice-clean` | Proof code cleaned up for review | 11/17 | off-track |  | K1/K2 | `top/repo-cleaned-up` |
| `proofs/terminology-clean` | Retired terms removed, new uses refused | 11/20 | off-track |  | K1/K2 | `top/repo-cleaned-up` |
| `proofs/cached-serving-audit-design` | Cached and speculative serving: audit design chosen | 11/12 | on-track |  | P? | `vllm/production-vllm-reproducible` |
| `proofs/duplicate-parameter-names-refused` | Duplicate parameter names refused by every decoder | 11/12 | off-track |  | P? | `proofs/exfiltration-under-300gb` |
| `proofs/network-number-printed` | One command prints exfiltration capacity | 11/12 | on-track |  | P? | `accounting/network-glm53` |
| `proofs/spot-verifier-cost` | Verifier cost measured and minimized | 11/12 | off-track |  | P? | `benchmarks/proof-system-performance` |
| `proofs/spot-overhead-measured` | Proof overhead measured per block type | 11/13 | off-track |  | P? | `accounting/network-glm53` |
| `proofs/spot-128k-attention-proved` | 128k-context request served and SPOT-proved | 11/13 | off-track |  | P? | `proofs/exfiltration-under-300gb` |
| `proofs/fully-proved-items-listed` | Everything left to prove listed with owners and dates | 11/13 | on-track |  | P? | `proofs/glm53-audit-fully-proved` |
| `proofs/serving-cost-three-parts` | Serving cost target prices its three unpriced parts | 11/13 | on-track | nothing waits on it | P? |  |
| `proofs/spot-draft-matches-lean` | SPOT draft and Lean proofs agree | 11/13 | off-track |  | P? | `proofs/spot-post-ready` |
| `proofs/spot-figures` | Figures generated from the repository | 11/13 | off-track |  | P? | `proofs/spot-post-ready` |
| `proofs/spot-numbers-lean-checked` | Every plotted number checked in Lean | 11/13 | on-track |  | P? | `proofs/spot-post-ready` |
| `accounting/network-glm53` | Network accounting on GLM-5.3 | 11/13 | off-track | set by hand | P? |  |
| `proofs/spot-advice-certified` | Scheduling leaks nothing under certification, draft off | 11/13 | off-track |  | P? | `proofs/spot-post-ready` |
| `proofs/spot-checks-closed` | Guarantee's untested conditions tested or disclosed | 11/13 | off-track |  | P? | `proofs/spot-post-ready` |
| `proofs/spot-claims-checked` | Every claim traced to evidence | 11/13 | off-track |  | P? | `proofs/spot-post-ready` |
| `proofs/spot-output-accounting` | Every output bit explained, certified or charged | 11/13 | off-track |  | P? | `proofs/exfiltration-under-300gb` |
| `proofs/spot-post-ready` | SPOT post results ready to publish | 11/13 | off-track | nothing waits on it | P? |  |
| `proofs/spot-two-stage-e2e` | Two-stage audit runs end to end | 11/13 | off-track |  | P? | `accounting/network-glm53` |
| `proofs/weight-security-bridge` | Weight-theft bound proved for deployed audit | 11/13 | off-track |  | P? | `proofs/spot-post-ready` |
| `proofs/spot-fused-cut-served` | MoE layers served with small commitments | 11/14 | off-track |  | P? | `proofs/exfiltration-under-300gb` |
| `proofs/bind-cost-theorem` | Binding finder's cost proved in Lean | 11/16 | off-track |  | P? | `proofs/exfiltration-under-300gb` |
| `proofs/spot-bind-term` | SPOT's binding error reaches 100 bits | 11/18 | off-track |  | P? | `proofs/exfiltration-under-300gb` |
| `proofs/security-numbers-audited` | Every security number evaluated at served sizes | 11/18 | off-track |  | P? | `proofs/spot-post-ready` |
| `proofs/spot-ten-percent` | Weights safe at 10% compute, measured | 11/18 | off-track |  | P? | `proofs/exfiltration-under-300gb` |
| `proofs/spot-moe-binding` | MoE expert reads proved in Lean | 11/30 | off-track |  | P? | `proofs/exfiltration-under-300gb` |
| `proofs/exfiltration-under-300gb` | GLM-5.3 exfiltration capacity comes under 300 GB | 11/30 | off-track |  | P? |  |
| `proofs/glm53-audit-fully-proved` | GLM-5.3 audit fully proved | 11/30 | off-track | nothing waits on it | P? |  |
| `proofs/sampling-generic` | SPOT proved for any proof system | 11/30 | on-track |  | P? | `proofs/exfiltration-under-300gb` |
| `proofs/spot-two-stage-sound` | Two-stage soundness proved for running code | 11/30 | on-track |  | P? | `proofs/exfiltration-under-300gb` |
| `proofs/zero-knowledge` | Zero-knowledge proof system complete | 11/20 | off-track |  | parked | `top/parked` |
| `proofs/zk-leaks-closed` | ZK server leaks closed in code | 11/20 | off-track |  | parked | `top/parked` |

## soundness: 18 open goals (? 9, K2 8, P? 1)

| goal | title | due | state | at the top today | first sort | because it is under |
| --- | --- | --- | --- | --- | --- | --- |
| `proofs/sampled-audits-sound` | Sampled proofs sound as deployed | 11/13 | on-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `soundness/wipe-count-tie-stated` | Wipe's egress count stated as the certifier's | 11/13 | on-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `soundness/inference-clause-on-program` | Inference-only clause proved on the checked Program | 11/16 | on-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `soundness/remote-verdict-by-lean` | Remote demo's verdict decided in Lean | 11/16 | on-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `soundness/several-member-leg` | Several-member windows proved sound | 11/16 | on-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `soundness/onsite-deployed-instance` | Onsite theorem's deployed instance proved | 11/19 | on-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `proofs/soundness-nothing-open` | Remote demo's soundness fully proved | 11/20 | on-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `soundness/rows-link` | Served weights proved to be the registered ones | 11/20 | on-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `soundness/value-tie-scoped` | Linear layers tied into inference-only check | 11/18 | on-track |  | P? | `training/small-model-slowdown-1000` |
| `soundness/composite-list` | A demo's protocols as one list in the theorem | 11/14 | on-track | nothing waits on it | ? |  |
| `soundness/challenger-as-run` | The challenger modelled as it actually runs | 11/16 | on-track | nothing waits on it | ? |  |
| `soundness/devices-pinned` | Auditor devices pinned in the play | 11/17 | on-track |  | ? |  |
| `soundness/security-areas-owned` | Every area of security/ has a lead | 11/17 | on-track | nothing waits on it | ? |  |
| `soundness/spot-theorem-listed` | SPOT's own soundness theorem listed | 11/17 | on-track | nothing waits on it | ? |  |
| `soundness/claims-from-locks` | Claim ids generated from the locks | 11/18 | on-track | nothing waits on it | ? |  |
| `soundness/onsite-composite-v1` | Onsite composite of record under Pearl+ | 11/19 | on-track | nothing waits on it | ? |  |
| `soundness/remote-witness-session` | Deployed remote theorem's honest witness runs a session | 11/20 | on-track | nothing waits on it | ? |  |
| `soundness/challenger-logs-via-devices` | The challenger's logs read through the auditor devices | 11/23 | on-track | nothing waits on it | ? |  |

## top: 13 open goals (K1/K2 8, K4 4, parked 1)

| goal | title | due | state | at the top today | first sort | because it is under |
| --- | --- | --- | --- | --- | --- | --- |
| `top/hotspots-ranked-and-closing` | Hotspots ranked nightly, top rows closing | 11/13 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `top/public-export-rehearsed` | Public export rehearsed on the moved tree | 11/14 | off-track |  | K1/K2 | `top/repo-cleaned-up` |
| `top/branches-cross-move` | Open branches carried across the move | 11/16 | off-track |  | K1/K2 | `top/repo-cleaned-up` |
| `top/architecture-signed` | Open architecture choices decided | 11/18 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `top/public-split-sunday` | Public export published | 11/20 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `top/unrun-code-archived` | Unused code deleted | 11/20 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `top/repo-cleaned-up` | Repository reorganized and published | 11/27 | off-track | set by hand | K1/K2 |  |
| `top/slices-clean` | Every area's code cleaned up for review | 11/27 | off-track |  | K1/K2 | `top/repo-cleaned-up` |
| `pm/work-under-goals` | All work tracked under goals | 11/13 | off-track |  | K4 | `infra/platform` |
| `top/merge-conflicts-cut` | Fewer changes refused for merge conflicts | 11/13 | on-track | nothing waits on it | K4 |  |
| `top/velocity-loop` | Research velocity scored and steered every six hours | 11/13 | on-track |  | K4 |  |
| `top/no-stranded-waits` | No job stranded beside a free machine | 11/17 | on-track | nothing waits on it | K4 |  |
| `top/parked` | Paused work resumed or dropped | 11/30 | on-track |  | parked |  |

## training: 21 open goals (? 1, K1/K2 14, K2 2, P? 3, parked 1)

| goal | title | due | state | at the top today | first sort | because it is under |
| --- | --- | --- | --- | --- | --- | --- |
| `training/gate-count-change-in-queue` | Gate-count change read and queued | 11/13 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `training/onsite-slowdown-options` | Onsite slowdown options priced | 11/13 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `training/open-premises-sorted` | Slowdown's open premises sorted | 11/16 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `training/algorithm-library` | Library holds standard algorithms and most frontier points | 11/16 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `training/gate-count-is-traced-circuits` | Gate count is the traced circuit's | 11/16 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `training/names-say-slowdown` | Slowdown replaces rho in every name | 11/16 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `training/one-slowdown-calculator` | One slowdown calculator for every protocol and demo | 11/16 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `training/slowdown-from-submitted-circuit` | Slowdown computed from a submitted circuit | 11/16 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `training/search-frontier-in-library` | Every search frontier point runs from the library | 11/20 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `training/attack-search-one-command` | Attack search runs from one command | 11/20 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `training/grader-gpu-routes` | Training circuits match GPU kernels | 11/20 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `training/second-swarm-results` | Training slowdown results final and proved | 11/20 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `training/one-pytorch-translator` | One translator from PyTorch to circuits | 11/23 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `training/code-in-final-places` | Training code consolidated | 11/23 | on-track |  | K1/K2 | `top/repo-cleaned-up` |
| `training/check-reads-served-record` | Inference-only demo done | 11/16 | off-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `accounting/inference-only` | Inference-only check accepts a served day | 11/20 | off-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `accounting/inference-only-small-model` | Inference-only accounting on a small model | 11/16 | off-track | set by hand | P? |  |
| `training/record-check-copies-removed` | Record check's copied definitions removed | 11/16 | on-track |  | P? | `training/small-model-slowdown-1000` |
| `training/small-model-slowdown-1000` | Small model's proved slowdown reaches 1,000 | 12/02 | off-track |  | P? |  |
| `training/slowdown-across-scale` | Slowdown plotted against model size | 11/20 | off-track |  | parked | `top/parked` |
| `training/count-skips-repeated-positions` | Slowdown count skips repeated positions | 11/17 | on-track | nothing waits on it | ? |  |

## vllm: 17 open goals (? 1, K1/K2 6, K2 5, P? 5)

| goal | title | due | state | at the top today | first sort | because it is under |
| --- | --- | --- | --- | --- | --- | --- |
| `vllm/after-move-path-fixes` | Leftover path fixes merged after the move | 11/13 | off-track |  | K1/K2 | `top/repo-cleaned-up` |
| `vllm/open-changes-over-move` | vLLM's open changes carried over the move | 11/13 | off-track |  | K1/K2 | `top/repo-cleaned-up` |
| `vllm/root-readme` | Root README describes the approved architecture | 11/13 | off-track |  | K1/K2 | `top/repo-cleaned-up` |
| `vllm/import-rules-on-new-layout` | Import rules checked on the new layout | 11/13 | off-track |  | K1/K2 | `top/repo-cleaned-up` |
| `vllm/slice-clean` | vLLM integration cleaned up for review | 11/14 | off-track |  | K1/K2 | `top/repo-cleaned-up` |
| `demos/remote-demo-on-main` | Remote demo code merged into main | 11/17 | off-track |  | K1/K2 | `top/repo-cleaned-up` |
| `vllm/serve-registers-weights` | Weights registered before serving starts | 11/13 | off-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `vllm/remote-rehearsal` | Remote demo rehearsed | 11/14 | on-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `demos/remote-demo` | Remote demo done | 11/20 | off-track | set by hand | K2 | `compute-research/accounting-code-vllm-quality` |
| `soundness/one-circuit-per-day` | Daily program committed before serving | 11/20 | on-track |  | K2 | `compute-research/accounting-code-vllm-quality` |
| `demos/onsite-demo` | Onsite demo done | 11/24 | off-track | set by hand | K2 | `compute-research/accounting-code-vllm-quality` |
| `vllm/production-vllm-reproducible` | Production vLLM reproducible under 1% serving overhead | 11/13 | off-track | set by hand | P? |  |
| `vllm/second-round-on-served-data` | Audit's second round runs on served data | 11/13 | on-track |  | P? | `accounting/network-glm53` |
| `vllm/serving-cost-benchmark` | vLLM serving cost benchmarked | 11/13 | off-track | set by hand | P? |  |
| `vllm/serving-cost-glm53` | Serving cost measured on GLM-5.3 | 11/13 | off-track |  | P? | `vllm/serving-cost-benchmark` |
| `vllm/served-128k-window-committed` | Long-context request served and committed | 11/16 | off-track |  | P? | `proofs/exfiltration-under-300gb` |
| `vllm/serving-cost-tick-arm` | Serving cost's tick arm measured behind certifier | 11/13 | on-track |  | ? |  |
