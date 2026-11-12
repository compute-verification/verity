# sass_census: every SASS opcode a device runs, against what the model prices

The opcode census of a device family (sm_90: the H200, sm_100: the B200, sm_120: the RTX PRO 6000): every binary with
device code on a node, disassembled and its dotted opcodes counted, then diffed against the menu the ACCEL model
prices. An opcode the menu doesn't price is a row a person reads: can it head a committed word, and by which route.
The probes beside it answer what the diff raises: the rate of a class the census leaves unpriced, what ptxas lowers a
PTX form to, and what `multimem` does on a card with no NVLink. The opcode table's rows
(`../../machine_model/data/cover/rows.toml`) cite these probes.

## Jobs

Each is the command of a `research run`, from a pushed commit; its header has the full command and its knobs.
`run.sh JOB [ARGS...]` is the directory's runner: it starts the job's script, named without `.sh`, with the rest of
its arguments and the environment as they are.

| Job | Runs on | What it does | Writes under `$OUT` |
| --- | --- | --- | --- |
| `dump.sh [ROOT...]` | a CPU node with the CUDA toolkit | finds every binary with device code under the roots (the toolkit's libraries, NCCL, the node's wheel and kernel caches, the repository's kernels), dumps each once with `cuobjdump -sass` and counts its opcodes (`sass_count.py`) | `toolkit.txt`, `corpus.tsv`, `counts/*.json`, `census.json`, `summary.txt` |
| `rates.sh` | one B200 (`ARCH=sm_100a`, the default) or one H200 (`ARCH=sm_90a`) | builds and SASS-gates `unpriced.cu`, `unpriced90.cu` or `SRC=unpriced_fp4.cu` with `../opcode_probe/probe.py`, times them under one lease, and gives the ratios the census needs (FFMA2 and FMUL2 against FFMA and FMUL, DMMA against the s8 tensor forms and beside a saturated tensor pipe); `BUILD_ONLY=1` stops after the gate | `build.log`, the binary's `.sass`, `gate-build.json`, `gate-build.txt`, `env/`, `probe.py run`'s `rows.jsonl`, `gate.json` and `times.jsonl`, `rates.json` |
| `lowering.sh` | a CPU node with the CUDA toolkit | compiles each case of `lowering.cu` for sm_90a and sm_120a and reads its SASS: the opcodes ptxas emits for a PTX form an opcode-table row cites, or its refusal, against the expectation in the script; exits 1 if a case lowers otherwise | `lowering.txt`, `sass/CASE_ARCH.sass` |
| `multimem.sh` | one RTX PRO 6000 | `multimem.cu`: the device's multicast attribute, what `cuMulticastCreate` answers, and what `multimem.ld_reduce` and `multimem.red` do on ordinary memory, each in its own child process | `multimem.txt` (its last line the verdict), `multimem.sass` |
| `eps_counts.sh` | a CPU node with the CUDA toolkit | builds the kernels ε_skip's per-instruction counts came from, out of the source trees sent with the run (`src-TAG.tar`), for sm_120a, sm_90a and sm_100a with line info, and dumps each cubin's SASS for `eps_counts.py` | `toolkit.txt`, `TAG/ARCH/`: `build.log`, `apv.gi.sass.gz`, `apv.sass.gz`, `lineinfo.txt` |

~~~text
research run --on MACHINE --cwd source -- bash benchmarks/pouw/sass_census/run.sh dump [ROOT ...]
research run --on MACHINE --cwd source -- bash benchmarks/pouw/sass_census/dump.sh [ROOT ...]     # the same job
research run --on MACHINE --project verity --source <tree> --cwd source --timeout 30m \
  --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/sass_census/rates.sh
~~~

## Reading a census

~~~text
cuobjdump -sass lib.so | python3 sass_count.py count --source NAME > NAME.json   # one binary's opcodes, by architecture
python3 sass_count.py merge counts/*.json > census.json                          # one census a device family
python3 sass_count.py summary census.json
python3 diff_menu.py FAMILY census.json [--out diff.json] [--unused]              # FAMILY: sm_90, sm_100 or sm_120
python3 eps_counts.py OUT --src TAG=APV_CU [--src ...] [--json FILE]              # an eps_counts.sh run's counts
~~~

- `sass_count.py` is stdlib only, so it runs on a node's own Python beside the toolkit.
- `diff_menu.py` buckets every opcode of the census and of NVIDIA's instruction-set table for the architecture
  (`../../machine_model/data/cover/sass_isa_tables.json`): priced on a lane, tensor or side route, a mover, harmless, or a
  gap (a tensor form with no rate, a product on no route, a paired multiply counted as one, a word the model prices at
  nothing), and flags where its own table and the model's classifier (`../facet_break/loads.py`'s `class_of`) disagree.
  Its module docstring defines each bucket.
- `sass_semantics.py` is the table `diff_menu.py` ranks by: for each SASS base opcode, whether it can form a product,
  whether it writes a word, and which route prices it.
- `eps_counts.py` counts, from each architecture's SASS, the instruction differences ε_skip is priced with, placed by
  the source lines each instruction's inline chain passes through.

Nothing outside the directory reads its Python. `sass_semantics.py` offers `classify` to `diff_menu.py`. The
`unpriced*.cu` probes include `../opcode_probe/probe.cuh` and are started by its `probe.py`; `unpriced.cuh`, which
all three include, holds the lane and mma.sync chains `unpriced.cu` and `unpriced90.cu` time (each op on 8
independent chains a thread, alone or alternating with another op). `dump.sh`, `lowering.sh` and `eps_counts.sh`
never take a GPU.

The tests are `../tests/test_sass_census.py` and `../tests/test_eps_counts.py`.
