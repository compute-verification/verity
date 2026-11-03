# facet_break: try to beat every facet of the machine model

The machine model (`../machine_model/`) bounds what a program can issue on sm_120 by a polytope of facets: row r says
Σ_k c_kr·x_k ≤ 1 for the program's warp instructions x_k of census class k a SMSP-cycle. ε is proved conditional on every
facet being a true ceiling. This directory is the standing search for programs that beat one: it generates MMA + lane
programs around each facet's vertices, times them with opcode_probe's gates, and loads every measured program (its own and
any other probe's or tensor_kinds' records) on every facet.

| File | What |
|---|---|
| `loads.py` | A program's load on every row at an epoch (`loads`), every violation (`summary`, load > 1: rate rows carry the 2% margin), the results table (`table`), the refit a beaten row needs (`refit`: the upper hull of the measured points at the margin, `eps1.facets`), and a beaten family refitted as one at a new epoch (`family`: the upper hull of the old rows' vertices and every measured program, VF3m-16's rule; `--forms` for the joint issue rows, each MMA form's own hull beside the rows' vector classes; `--emit EPOCH` prints the rows.csv and census lines and the old lines to close). IMMA.16832 loads every row its own kind has no line on through the generic `MMA` lines (they were measured with it); every other form only through its own (`--generic`: through `MMA` too, advisory) |
| `mix.cuh` | The kernels: a MMA form and up to two lane ops in one warp's loop (`FB_MIX`), in separate warps (`FB_SPEC`), the lanes alone (`FB_ALONE`), the MMA alone (`FB_MMA`), over `../opcode_probe/probe.cuh` |
| `mem.cuh` | The table routes' loads (LDS.32/64/128/U8, LDG.E/.CONSTANT/.64/.128 from an L1-resident 32 KB region, TLD of 1, 2 and 4 words from a 4- or 16-byte texel), alone or two kinds in one loop (`FB_LD`) and beside a MMA form (`FB_LDMMA`), in three address patterns (lane-per-word, broadcast, four lanes a word); `soff` gives the gate's bank check the kernel's own addresses. `FB_L2`: 128- and 256-bit loads over a block's 256 KB region (in L2, past L1), plain, `L1::no_allocate` and `.cg`, coalesced or one 32-byte sector a lane |
| `generate.py` | The seeded grid. `--family mix` (default): the targeted part (each lane class beside IMMA.16832 at the vertices of Π's (x_MMA, x_X) section and around them, two-lane pairs, warps a SMSP, each form alone, the JI3n family) and `--n` sampled programs. `--family mem`: each load alone, load pairs at 1:1, 3:1 and 1:3, each load beside IMMA.16832 (and HMMA.16816.F16) at 2-32 loads a 8 MMAs, and `--n` sampled. `--family fine`: a fine q grid of each lane beside IMMA.16832 where the joint rows were beaten or tight, and every form beside FFMA at 64-192 a 8 MMAs. `--family l2`: the L2 loads at 2-8 warps a SMSP and the 256-bit load from L1. `--emit` writes the probe source with each program's exact SASS counts, `--only` keeps the gate's passes |
| `run.sh` | One batch under `research run`: generate, build, gate, drop the refusals by name, rebuild in chunks of 64, time under one `gpu-lease`, then `loads.py table` and `summary` into `$OUT` |

~~~bash
research run --on MACHINE --project verity --source . --cwd source --custody-ttl 12h \
  --env GPU_LEASE_WHO=facet-break -- bash benchmarks/pouw/facet_break/run.sh --seed 2 --n 300
python benchmarks/pouw/facet_break/loads.py summary ~/.research/runs/*/out/rows.jsonl ~/.research/runs/*/out/tensor_kinds.jsonl
python benchmarks/pouw/facet_break/loads.py refit RUN/out/rows.jsonl --row R1
python benchmarks/pouw/facet_break/loads.py family RUN/out/rows.jsonl --family R1 KFFMA0 --name EFFMA --kind FFMA --emit VF3q-16
python benchmarks/pouw/facet_break/loads.py summary RUN/out/rows.jsonl --full --epoch VF3s-16
~~~

A load counts only the gate's pinned opcodes, each in its class when the class is certain (`loads.class_of`; NOP, BRA
and ISETP have none, MOV is NOROW and a uniform instruction is UNI), so it is a lower bound and a reported violation is
real. `--full` counts the rest of a single-loop program's timed loop too, from its run's gate verdicts and `sass/`: each
other loop instruction runs once an iteration, at a pinned instruction's rate, in its class, or on `issue` alone when it
has none. The machine model's `Run.n` counts every executed instruction, so the full load is the one a row must hold, and
it is still a lower bound. A TLD's class is its width, which its
opcode doesn't show, so a generated program's name carries it (`_tld4_`). A load whose class has no line at the epoch
(LDS.64, LDG.E.64, the 256-bit load and the L1-skipping loads before VF3q-16: Π prices it at 0) is listed by `summary`
as unpriced. A violation goes in the lane's note with its row, load,
program, run id and SASS, and its refit line. `refit --form F` puts form F on the tensor axis of a row over several
vector classes (JI3n), the vector axis each class at its coefficient over the row's commonest one.

The table route's L2 facet (`../machine_model/data/table.toml`'s `l2`, read bytes a second, what ε's `--lookups` prices a
lookup against) is the row `L2tbl`: a program's L2 bytes a second (each load that skips L1, `.NA` or `.STRONG.GPU`, at its
own width, x 32 lanes x the SMs timed x the measured clock) over `l2`. `--l2 B` sets it where `table.toml` is absent.
