Lane circuits
Refs CIR-270
Goal proofs/security-numbers-audited

**`verity lean bits` prints one line per guarantee the security locks list: 1,569 guarantees on 1,637 lines. 16 of them have a number, all from Python mirrors pinned to the locks. For each point where a `norm_num` Lean instance exists, the new test holds the printed figure to it.** This is step 1 of the security table's design.

This PR holds #9346 (proofs' `cursor/proofs-expcalls-4d1e` at `949c03c3a7a1`), which adds `Flock.SecurityProofs.SpotOneWaitTwo`, the guarantee that the instance's mirror pins. I mark this PR ready once #9346 is on `main`; if its marked head moves first, I merge that head in.

## What the command does

`verity lean bits [PKG...] [--facts F... | --run RUN] [--census DEPLOYMENT] [--json] [--locks-only]` is `tools/verity/lean/bits.py`. It reads each package's `lean-audit.json`. With no PKG it reads `security`, `security/Proofs` and `security/Properties`. It builds nothing and exits 0 whenever it prints, so it gates nothing.

It prints one line for each guarantee under `guarantees`. A mirrored guarantee gets one line per budget and point instead. The columns:
- the guarantee and its owner;
- whether its statement bounds a probability;
- the error term;
- the point and the budget;
- the bits (−log₂ of the error term there);
- the source of the number;
- the row (the Lean instance at that point, where one exists);
- the bits left over a week, a month and a year of audits, at one datacenter and at the world.

The lines are ordered: no number first, then `mirrored`, then ascending bits.

**What comes from where.**
- **Exact, from the lock:** the guarantee, its owner, its hashes, and whether the lock lists it as another guarantee's witness. A witness bounds nothing.
- **From the statement's text, marked `~`** (as `trusted_size.py` marks its readings): whether the statement bounds a probability, and its error term. A lock's `reads` lists what a package reads, not what each guarantee reads, so the lock can't give these.
  - The text pass resolves names through the namespaces and `open`s around each declaration.
  - It follows definitions three deep, looking for a probability definition: the locks' `shape.events` plus a short list.
  - The error term is the right side of the `≤` after it.
- **`--facts F` or `--run RUN` (a records run in the evidence store):** the probability column becomes exact. A guarantee bounds a probability when its pin's `reads` holds a probability definition. A pin with no `reads` (`missing`: the package lists the guarantee and another package proves it) leaves the text's reading.
- **`--census DEPLOYMENT`:** picks the census deployment for the period columns; the default is the reference.
- **`--locks-only`:** skips the text pass.

**The period columns** apply a union bound over a period's audits: the bits minus log₂ n, where n is the number of audits in the period. They appear only where the mirrored term bounds one audit.
- At one datacenter, n counts the census deployment's machines, each audited once a window.
- At the world, n counts the census adversary resource model's GPUs (its FLOP a year divided by its rate per H100-equivalent), in machines of the deployment's size, audited the same way.
- A week is 7 days, a month is a twelfth of the adversary's year, and a year is its `monitoring_period_s`.

**Code packages' locks** (`verity/protocols/certifier/verifier/lean`, `verity/protocols/flock/verifier`) are left out of the default. Their 15 guarantees are all equalities about executables, with no error term. Naming either as PKG prints it.

### The three sources of a number

- **`proved`:** a Lean row, a theorem that the bound at one point and budget is at most a stated figure, which the lock records with the bound it reads. No lock holds one yet. That is step 2, the facts' `bound` field.
- **`computed`:** a Lean evaluator with a soundness property. None is built yet.
- **`mirrored`:** a Python restatement of one error term. This is the only source built today. It is printed only while every hash it pins still matches the locks.

A mirror evaluates one term of a bound, so a guarantee's security in bits is at most the printed figure.

### The mirrors

`bits_mirror.py` holds the formulas: one function per error term, named for the Lean definition it restates, in exact rationals wherever there is no root. `bits_pins.json` holds what each mirror pins. It is data, because it holds Lean's names and the locks' hashes, and as `.py` prose those names tripped the retired-terms lint (`Cap`, `PeriodBinding…`).

| Mirror (Lean definition) | Guarantees pinned | Definitions pinned by `reads` hash | Per audit |
|---|---|---|---|
| `Claims.Flock.Numbers.hashBound` | `Flock.SecurityProofs.SpotOneWaitTwo` (`type_hash`) | none (a proofs-tree definition has no `reads` record) | yes |
| `Claims.SamplingGeneric.Compiled.εregAt`, its binding terms | `compilerKSReg`, `compilerKSRegAt` | `Compiled.εreg`, `Compiled.εregAt`, `Compiled.bconfCost` | yes |
| `Claims.SamplingGeneric.Compiled.KSAt`, its extractor term | `compilerKSAt`, `compilerKSAtTwo` | `Compiled.KSAt`, `Compiled.work`, `Claims.Sampling.KSAt`, `Claims.ecrBound` | yes |
| `Claims.KS` at the deployed zero-knowledge record (C-Flock) | `Flock.SecurityProofs.MaskedKS` | `Claims.KS`, `Claims.Flock.KS` | yes |
| `Pouw.PearlC.δPearlC` | the 9 Pearl-C guarantees whose statements apply it | `Pouw.PearlC.δPearlC` | no (per challenge) |
| PoUW's binding per prover, `\|L\|·√(q/2^256 + κ)` at κ = 0 | `Pouw.SecurityProofs.PeriodBindingPerProver` | its `Guarantees` definition | no (per verification phase) |

Every guarantee except `SpotOneWaitTwo` is pinned by its `unfolded_hash`. Definitions are pinned as well as guarantees because a statement's hash names the definitions it reads only by name: restating `εregAt` moves its `reads` hash and leaves `compilerKSReg`'s `unfolded_hash` unchanged.

Every point whose sizes no card declares says so in its `about` field: SPOT's `N`, `n_PU`, `k` and `openCalls`, Pearl-C's `2^30`, and PoUW's `|L|`.

Where the mirrors differ from the design's `table_v1.py`, for proofs to read:
- **`εregAt`:** it has one `N·√(c_T/2^256)` term and includes `2·n_PU·openCalls` in `c_T`. The script uses two `N·√` terms and leaves out `openCalls`, so at each SPOT point the mirror prints about one bit more than the script.
- **The world's audit count:** the census fields give 2^39.83 a year. The script took 2^41.16, relayed from another auditor. The datacenter's, 2^29.29, agrees.

### The test

`tools/verity/lean/tests/test_bits.py` has 13 tests. It builds no Lean and runs in under a second.

**Pins.**
- Every mirror's pinned hashes must equal the locks'.
- A guarantee whose record has an `unfolded_hash` must be pinned by it.
- A moved pin fails with: re-read the statement, update the mirror to what it now says, then its pin.
- Until then the command prints the guarantee with no number. The test checks this on a lock with one hash changed.

**Instances** (`INSTANCES`, one line each):
- `Claims.Flock.Numbers.hashBound_oneWait_513`
- `Claims.Flock.Numbers.hashBound_oneWait_513_k40`

Both are in `security/Proofs/Flock/SamplingGeneric/TwoOneWaitNumbers.lean`, proved by `norm_num [hashBound, firstHitCalls]`. For each one, the test:
1. reads the theorem's arguments and right side from the source;
2. evaluates the `hashBound` mirror there in exact rationals;
3. requires that value to be at most the right side;
4. requires the command's line for that row to carry the same arguments and print the same figure.

**What the test does not hold:** that `hashBound` at those numerals is `SpotOneWaitTwo`'s bound. That is the mirror's own reading (the guarantee's error reads `hashBound` twice). The facts' `bound` field will check it in step 2.

**Other tests:**
- the text pass on a small package;
- `--facts`, including a `missing` pin;
- `--run`, with the store stubbed;
- the period counts on a small census;
- ordering and exit codes.

**A mutation check**, run once and not committed: changing `hashBound`'s `2^256` to `2^255` failed both instance tests, and changing one digit of a pinned `reads` hash failed that mirror's pin test.

**Declared inputs.** The suite `verity-lean-audit` declares the test's reads under `[tool.verity.tests] inputs` and a `reads` entry for `test_bits.py`:
- the three security locks;
- that one Lean source;
- the census's `deployments.json` and `adversary.json`.

## Output on the security locks (counts)

`verity lean bits` at this branch's head:
- **Lines:** 1,637, for 1,569 guarantees: `security` 1,104, `security/Properties` 465, `security/Proofs` none.
- **Bounding a probability:** 528, read from the text (`~`). The other 1,041 do not, 437 of them witnesses (from the lock).
- **With a number:** 16, all `mirrored`.
- **Lines by source:** none 1,553; `mirrored` 84.
- **Pins moved:** none.
- **Mirrored per-audit lines that are vacuous over a year:** 16 of 44 at one datacenter, and 26 of 44 at the world.

**The text pass against the design's inventory.** The inventory lists 547 guarantees from an older `main`, 546 of which are still in the locks.
- The text pass flags 510 of those 546, and 18 guarantees that the inventory doesn't list. Several of the 18 were added since.
- Following four to six definitions deep instead of three adds 63 to 336 guarantees the inventory doesn't list, so the pass stays at three.

**`--facts`:** I ran it on `r20261110-234559-f99c`'s facts for `Security`, fetched by their path from before the move. Every pin in that file is `missing`, so the text decided every guarantee. `--run` on that run finds no `security` files, because its tag predates the move; the command says so and reads the tree.

The two lines held to an instance, as the command prints them. **Proofs has not read these figures against the script's; they are quoted nowhere else:**

```
| Flock.SecurityProofs.SpotOneWaitTwo | @proofs | - | hashBound(T, e, N_s, K_s, K_m, c) = (2·(T·(1 + e·N_s) + c) + 4·firstHitCalls)/2^256, read… | one table, one wait a session, K up to 2^40 | T = 513·2^80 calls a run, a call charged as 513 compressions | 101.4 | mirrored | Claims.Flock.Numbers.hashBound_oneWait_513_k40 | 77.9 · 75.7 · 72.1 | 67.3 · 65.2 · 61.6 |
| Flock.SecurityProofs.SpotOneWaitTwo | @proofs | - | hashBound(T, e, N_s, K_s, K_m, c) = (2·(T·(1 + e·N_s) + c) + 4·firstHitCalls)/2^256, read… | one table, one wait a session | T = 513·2^80 calls a run, a call charged as 513 compressions | 103.4 | mirrored | Claims.Flock.Numbers.hashBound_oneWait_513 | 79.9 · 77.7 · 74.1 | 69.3 · 67.2 · 63.6 |
```

## The gate

`python3 adapters/vllm/tools/ratchet_gate.py origin/cursor/proofs-expcalls-4d1e HEAD --json` at `b694e1efa637d9a11f6fda3337d893a63b203be5`:

```
"failures": []
```

## Runs

Each run's suites went through `research run --on auto --cwd source --wait -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py [SUITE...]'`. What I read in each, from its `stdout.log`:
- `r20261112-185255-879d` (source `ae5f7461cc`): cancelled while it waited for a slot, because its source was stale. It never started.
- `r20261112-190622-216d` (source `3bf908ae14ee643d640575dd9c3a226269530a66`), every suite: 123 suites, 122 passed (0 cached), 1 failed. `verity-lean-audit` had 753 passed and 2 failed, both in `test_not_read.py`: the three new files in the Lean tool's directory were not in `lean_audit.NOT_READ`. `b694e1efa6` adds them, and changes nothing else.
- `r20261112-230540-4042` (source `b694e1efa637d9a11f6fda3337d893a63b203be5`), `verity-lean-audit` alone: 755 passed, 0 failed.
- `r20261112-232904-fee1` (source `b694e1efa637d9a11f6fda3337d893a63b203be5`), the five other suites that declare `lean_audit.py` as an input: 5 passed (0 cached), 0 failed. `verity-check` 970 passed; `verity-network-e2e` 70 passed; `verity/protocols/certifier/verifier/network` 85 passed; `research` 4,697 passed, 3 skipped; `repository` 356 passed.

The other 117 suites passed at `3bf908ae14` and read none of what `b694e1efa6` changed.

## Not done

- **More mirrors:**
  - the recursion (`recursiveBitsK_deployed`'s sum);
  - the ZK loop (`Loop.errAt`);
  - `deltaPub`;
  - PoUB's receipt sketch;
  - C-Flock's `Claims.Bits` (`t·2^-λ`). No listed guarantee other than a witness states it, so nothing pins a mirror of it.
- **Period columns for terms that are not per audit:** Pearl-C, per challenge, and PoUW's binding, per verification phase. Each needs its family's own count from the census.
- **`--run` on a records run of this layout:** none was at hand.

## Not run

- The whole `check`.
- `lean-audit`: this PR changes no Lean file and no lock.
- `circuit-check`: no circuit changes.
- The 117 suites that don't read `lean_audit.py`, at the head: they passed at `3bf908ae14`, and the head differs from it only in `lean_audit.NOT_READ`.
- Suites other than the six that read this change's files, at `7effbf40c705`. The branch is carried across the layout moves and holds `main` at `fa3e619ee146`. At that head, run `r20261113-181406-a0b2`: `verity-lean-audit` 796, `verity-cli` 133, `verity-check` 1,006, `research` 5,086 and the certifier's network verifier 104 passed; `repository` had 361 passed and 1 failed. The failure is `test_the_unwitnessed_list_only_shrinks`: `security` has 222 > 221, because #9346's `SpotOneWaitTwo` is unwitnessed and `main` has since lowered the cap.

