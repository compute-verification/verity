Lane circuits
Refs CIR-270
Goal proofs/security-numbers-audited

**`verity lean bits` prints one line per guarantee the security locks list: 1,679 guarantees on 1,741 lines. 14 of them have a number, all from Python mirrors pinned to the locks. A guarantee its lock lists under `unwitnessed` gets no number: its line reads `not shown to accept any play`.** [run: `r20261113-224235-ea48` at `d2a084f91f38`] This is step 1 of the security table's design.

The tags: [run] is a run named with its commit, [read] is what I read in the tree or a log, [inferred] is my conclusion from those.

This PR holds #9346 (proofs' `cursor/proofs-expcalls-4d1e` at `949c03c3a7a1`), which adds `Flock.SecurityProofs.SpotOneWaitTwo`. [read] The coordinator marks this PR. Until #9346 is on `main`, `repository`'s unwitnessed cap and the ratchet gate fail on #9346's files (see The gate and Runs). [run]

## Unwitnessed guarantees print no number

**The rule.** A guarantee its lock lists under `unwitnessed` is not shown to accept any play, so `verity lean bits` prints no number for it from any source: no bits, no week, month or year figure. Ruled by soundness on LEAN-306, Fri 13 Nov 2:08 PM PST. [read]

**What the command reads.** Each lock's `unwitnessed` map, and each entry through `check.unwitnessed_entry`, the same function `check.py` and `merge.py` read it with. [read] The guarantee gets one line, with:
- the lock's facts and the error term read from its text;
- the bits cell `not shown to accept any play`;
- empty point, budget, arguments, source, row and period cells;
- a last column, `witness missing`, naming the witness the entry exempts.

Pins are still checked, so a moved pin still shows in the line's note. `--json` gives the line `"unwitnessed": {"missing": ..., "mirrored": ...}`, with `bits` and `periods` null. [read]

The row text for each form of entry (bits cell / witness missing cell), each held by a test on a small lock [run: `test_bits.py` locally at `d2a084f91f38`, 18 passed]:

| Entry | Bits cell | Witness missing |
|---|---|---|
| a type hash (a string) | `not shown to accept any play` | `honest and nontrivial` |
| `{type_hash, unfolded_hash}` | `not shown to accept any play` | `honest and nontrivial` |
| `{type_hash, ..., missing: "honest", why}` | `not shown to accept any play` | `honest` |
| `{type_hash, ..., missing: "nontrivial", why}` | `not shown to accept any play` | `nontrivial` |

**Counts by lock** [run: `r20261113-224235-ea48` at `d2a084f91f38`]:
- `security`: 222 guarantees under `unwitnessed`, 222 lines reading `not shown to accept any play`. A mirror would number 2 of them.
- `security/Properties`: 184 under `unwitnessed`, 184 lines with those words. A mirror would number none.
- `security/Proofs` lists no guarantee.

**The two mirrored guarantees whose numbers it removes** [run, same run; compared with the command at `7effbf40c705`, run locally]:
- `Flock.SecurityProofs.SpotOneWaitTwo`, mirrored by `hashBound`. Its entry is `missing: honest`.
- `Flock.SecurityProofs.MaskedKS`, mirrored by `KS`. Its entry has both hashes, so both witnesses are missing.

Guarantees with a number went from 16 to 14 and mirrored lines from 84 to 76. No other guarantee's number changed. [run]

As printed:

```
| Flock.SecurityProofs.MaskedKS | @proofs | ~yes | ~κ x + Real.sqrt (γ x * PrB H V P x) | - | - | not shown to accept any play | - | - | - | - | honest and nontrivial |
| Flock.SecurityProofs.SpotOneWaitTwo | @proofs | ~yes | ~pb + ε | - | - | not shown to accept any play | - | - | - | - | honest |
```

Both mirrors and their pins stay. When a guarantee's entry leaves its lock's `unwitnessed`, its numbers print again with no change to the command. A test removes the entry from a small lock and holds the printed figures to the mirror's. [run: `test_bits.py` locally at `d2a084f91f38`]

## What the command does

`verity lean bits [PKG...] [--facts F... | --run RUN] [--census DEPLOYMENT] [--json] [--locks-only]` is `tools/verity/lean/bits.py`. It reads each package's `lean-audit.json`. With no PKG it reads `security`, `security/Proofs` and `security/Properties`. It builds nothing and exits 0 whenever it prints, so it gates nothing. [read]

It prints one line for each guarantee under `guarantees`. A mirrored guarantee gets one line per budget and point instead. The columns:
- the guarantee and its owner;
- whether its statement bounds a probability;
- the error term;
- the point and the budget;
- the bits (−log₂ of the error term there);
- the source of the number;
- the row (the Lean instance at that point, where one exists);
- the bits left over a week, a month and a year of audits, at one datacenter and at the world;
- the witness missing, for a guarantee under `unwitnessed`.

The lines are ordered: no number first, then `mirrored`, then ascending bits. [read]

**What comes from where.** [read]
- **Exact, from the lock:** the guarantee, its owner, its hashes, whether it is under `unwitnessed`, and whether the lock lists it as another guarantee's witness. A witness bounds nothing.
- **From the statement's text, marked `~`** (as `trusted_size.py` marks its readings): whether the statement bounds a probability, and its error term. A lock's `reads` lists what a package reads, not what each guarantee reads, so the lock can't give these.
  - The text pass resolves names through the namespaces and `open`s around each declaration.
  - It follows definitions three deep, looking for a probability definition: the locks' `shape.events` plus a short list.
  - The error term is the right side of the `≤` after it.
- **`--facts F` or `--run RUN` (a records run in the evidence store):** the probability column becomes exact. A guarantee bounds a probability when its pin's `reads` holds a probability definition. A pin with no `reads` (`missing`: the package lists the guarantee and another package proves it) leaves the text's reading.
- **`--census DEPLOYMENT`:** picks the census deployment for the period columns; the default is the reference.
- **`--locks-only`:** skips the text pass.

**The period columns** apply a union bound over a period's audits: the bits minus log₂ n, where n is the number of audits in the period. They appear only where the mirrored term bounds one audit. [read]
- At one datacenter, n counts the census deployment's machines, each audited once a window.
- At the world, n counts the census adversary resource model's GPUs (its FLOP a year divided by its rate per H100-equivalent), in machines of the deployment's size, audited the same way.
- A week is 7 days, a month is a twelfth of the adversary's year, and a year is its `monitoring_period_s`.

**Code packages' locks** (every `lean-audit.json` outside `security/`) are left out of the default. Naming one as PKG prints it. [read]

### The three sources of a number

- **`proved`:** a Lean row, a theorem that the bound at one point and budget is at most a stated figure, which the lock records with the bound it reads. No lock holds one yet. That is step 2, the facts' `bound` field. [read]
- **`computed`:** a Lean evaluator with a soundness property. None is built yet. [read]
- **`mirrored`:** a Python restatement of one error term. This is the only source built today. It is printed only while every hash it pins still matches the locks, and never for a guarantee under `unwitnessed`. [read]

A mirror evaluates one term of a bound, so a guarantee's security in bits is at most the printed figure. [inferred]

### The mirrors

`bits_mirror.py` holds the formulas: one function per error term, named for the Lean definition it restates, in exact rationals wherever there is no root. `bits_pins.json` holds what each mirror pins. It is data, because it holds Lean's names and the locks' hashes, and as `.py` prose those names tripped the retired-terms lint. [read]

| Mirror (Lean definition) | Guarantees pinned | Definitions pinned by `reads` hash | Per audit | Printed |
|---|---|---|---|---|
| `Claims.Flock.Numbers.hashBound` | `Flock.SecurityProofs.SpotOneWaitTwo` (`type_hash`) | none (a proofs-tree definition has no `reads` record) | yes | no: `not shown to accept any play` |
| `Claims.SamplingGeneric.Compiled.εregAt`, its binding terms | `compilerKSReg`, `compilerKSRegAt` | `Compiled.εreg`, `Compiled.εregAt`, `Compiled.bconfCost` | yes | yes |
| `Claims.SamplingGeneric.Compiled.KSAt`, its extractor term | `compilerKSAt`, `compilerKSAtTwo` | `Compiled.KSAt`, `Compiled.work`, `Claims.Sampling.KSAt`, `Claims.ecrBound` | yes | yes |
| `Claims.KS` at the deployed zero-knowledge record (C-Flock) | `Flock.SecurityProofs.MaskedKS` | `Claims.KS`, `Claims.Flock.KS` | yes | no: `not shown to accept any play` |
| `Pouw.PearlC.δPearlC` | the 9 Pearl-C guarantees whose statements apply it | `Pouw.PearlC.δPearlC` | no (per challenge) | yes |
| PoUW's binding per prover, `\|L\|·√(q/2^256 + κ)` at κ = 0 | `Pouw.SecurityProofs.MonitoringPeriodBindingPerProver` | its `Guarantees` definition | no (per verification phase) | yes |

[read: `bits_pins.json` at `d2a084f91f38`; the Printed column: run `r20261113-224235-ea48`]

Every guarantee except `SpotOneWaitTwo` is pinned by its `unfolded_hash`. Definitions are pinned as well as guarantees because a statement's hash names the definitions it reads only by name: restating `εregAt` moves its `reads` hash and leaves `compilerKSReg`'s `unfolded_hash` unchanged. [read]

Every point whose sizes no card declares says so in its `about` field: SPOT's `N`, `n_PU`, `k` and `openCalls`, Pearl-C's `2^30`, and PoUW's `|L|`. [read]

Where the mirrors differ from the design's `table_v1.py`, for proofs to read:
- **`εregAt`:** it has one `N·√(c_T/2^256)` term and includes `2·n_PU·openCalls` in `c_T`. The script uses two `N·√` terms and leaves out `openCalls`, so at each SPOT point the mirror prints about one bit more than the script. [read; inferred from the formulas, not rerun at this head]
- **The world's audit count:** the census fields give 2^39.83 a year. The script took 2^41.16, relayed from another auditor. The datacenter's, 2^29.29, agrees. [run: `r20261113-224235-ea48`]

### The test

`tools/verity/lean/tests/test_bits.py` has 18 tests. It builds no Lean and runs in under a second. [run: locally at `d2a084f91f38`, 18 passed in 0.74 s]

**Pins.** [read]
- Every mirror's pinned hashes must equal the locks'.
- A guarantee whose record has an `unfolded_hash` must be pinned by it.
- A moved pin fails with: re-read the statement, update the mirror to what it now says, then its pin.
- Until then the command prints the guarantee with no number. The test checks this on a lock with one hash changed.

**Unwitnessed** (general, not by name): [read]
- On a small lock, for each of the four entry forms: one line, null bits, periods, point, budget, arguments, source and row, the words in the bits cell, no digit in the bits or period cells, the witness missing, and the summary's counts. With the entry removed, the same guarantee prints the mirror's figures. With it restored, the command's text and `--json` carry no figure for it.
- On the real locks: the set of lines marked unwitnessed equals the locks' `unwitnessed` entries, and none has bits or periods.

**The two lines held to an instance.** `INSTANCES` lists `Claims.Flock.Numbers.hashBound_oneWait_513` and `Claims.Flock.Numbers.hashBound_oneWait_513_k40`, both in `security/Proofs/Flock/SamplingGeneric/TwoOneWaitNumbers.lean`, proved by `norm_num [hashBound, firstHitCalls]`. [read] For each one, the test now:
1. reads the theorem's arguments and right side from the source;
2. evaluates the `hashBound` mirror there in exact rationals;
3. requires that value to be at most the right side.

Both instances are of `SpotOneWaitTwo`, which is under `unwitnessed`. So the test then requires that neither line is printed: no line has the instance as its row, no line of `SpotOneWaitTwo` has bits, and the command's text carries neither the row nor the figure. [read] Once the entry leaves the lock, the same test requires each line printed with the instance's arguments and the mirror's figure. That branch is not exercised at this head. The small-lock test holds the printed figure equal to the mirror's on another mirror. [inferred]

**What the test does not hold:** that `hashBound` at those numerals is `SpotOneWaitTwo`'s bound. That is the mirror's own reading (the guarantee's error reads `hashBound` twice). The facts' `bound` field will check it in step 2. [read]

**Other tests:** the text pass on a small package; `--facts`, including a `missing` pin; `--run`, with the store stubbed; the period counts on a small census; ordering and exit codes. [read]

**A mutation check**, run locally and not committed [run, at `d2a084f91f38` with the edit]: making an unwitnessed guarantee's mirror run again (removing the `continue`) failed 7 tests. An earlier mutation check, before the move: changing `hashBound`'s `2^256` to `2^255` failed both instance tests, and changing one digit of a pinned `reads` hash failed that mirror's pin test. [run, at an earlier head; not rerun]

**Declared inputs.** The suite `verity-lean-audit` declares the test's reads under `[tool.verity.tests] inputs`: the three security locks, that one Lean source, and the census's `deployments.json` and `adversary.json`. [read]

## Output on the security locks (counts)

`verity lean bits` at `d2a084f91f38`, run `r20261113-224235-ea48`, its text and `--json` kept under its `out/`: [run]
- **Lines:** 1,741, for 1,679 guarantees: `security` 1,201 lines for 1,139 guarantees, `security/Properties` 540 for 540, `security/Proofs` none.
- **Bounding a probability:** 539, read from the text (`~`). The other 1,140 do not, 513 of them witnesses (from the lock).
- **With a number:** 14 guarantees on 76 lines, all `mirrored`. Lines by source: none 1,665; `mirrored` 76.
- **Under `unwitnessed`, so no number:** 406 lines (`security` 222, `security/Properties` 184), each reading `not shown to accept any play`.
- **Pins moved:** none.
- **Mirrored per-audit lines that are vacuous over a year:** 16 of 36 at one datacenter, and 26 of 36 at the world.

**`--facts`:** I ran it on `r20261110-234559-f99c`'s facts for `Security`, fetched by their path from before the move. Every pin in that file is `missing`, so the text decided every guarantee. `--run` on that run finds no `security` files, because its tag predates the move; the command says so and reads the tree. [run, at an earlier head; not rerun]

## The gate

`python adapters/vllm/tools/ratchet_gate.py origin/main HEAD --json` at `d2a084f91f38`, against `origin/main` at `eba3ea48d4db`, locally: `"failures"` is not empty. [run] It names one cause: `tests/lint/allowlists/repo_retired_terms.json` grows from 105,743 to 105,810, through five Lean files of #9346 that this branch holds and `main` does not (`TwoFirstHitAvg.lean` 42, `ChargeOnce.lean` 5, `TwoOffClassAvg.lean` 4, `TwoOneWaitAvg.lean` 3, `RerunCalls.lean` 2). [read] The gate fails the same way at `7effbf40c705`, before this change. [run] The cause is outside this PR's files and is #9346's; it passes once #9346 is on `main`. [inferred]

## Runs

Each suites run went through `research run --on auto --cwd source --wait -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py SUITE...'`, over the six suites that read this change's files: `verity-lean-audit`, `verity-cli`, `verity-check`, `research`, the certifier's network verifier and `repository`. What I read in each, from its `stdout.log`:
- `r20261113-181406-a0b2` (source `7effbf40c705`): 5 passed. `repository` had 361 passed and 1 failed, `test_the_unwitnessed_list_only_shrinks`: `security` has 222 > 221, because #9346's `SpotOneWaitTwo` is unwitnessed and `main` has since lowered the cap. [run]
- SUITES_RUN_LINE
- `r20261113-224235-ea48` (source `d2a084f91f38`): the command, text and `--json`, exit 0. [run]

## Not done

- **More mirrors:** [read]
  - the recursion (`recursiveBitsK_deployed`'s sum);
  - the ZK loop (`Loop.errAt`);
  - `deltaPub`;
  - PoUB's receipt sketch;
  - C-Flock's `Claims.Bits` (`t·2^-λ`). No listed guarantee other than a witness states it, so nothing pins a mirror of it.
- **Period columns for terms that are not per audit:** Pearl-C, per challenge, and PoUW's binding, per verification phase. Each needs its family's own count from the census.
- **`--run` on a records run of this layout:** none was at hand.

## Not run at `d2a084f91f38`

- The whole `check`.
- `lean-audit`: this PR changes no Lean file and no lock.
- `circuit-check`: no circuit changes.
- The suites other than the six above.
- `--facts` and `--run` on a records run.
