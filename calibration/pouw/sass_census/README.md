# sass_census: every SASS opcode a device runs, against what the model prices

Does the ACCEL model price every instruction a GPU can run? The model, which PoUW's ε is computed from, prices a menu of
instructions. This directory takes the census of a device family (sm_90: the H200, sm_100: the B200, sm_120: the RTX PRO
6000): every binary with device code on a node, disassembled and its opcodes counted, then diffed against the menu. An
opcode the menu does not price is a row a person reads: can it head a committed word, and by which route.

The probes beside the census answer what the diff raises: the rate of a class the census leaves unpriced, what ptxas
lowers a PTX form to, and what `multimem` does on a card with no NVLink. The opcode table's rows
(`../../machine_model/data/cover/rows.toml`) cite them.

## Run it

Each job is the command of a `research run` from a pushed commit. `run.sh JOB [ARGS...]` starts the job's script, named
without `.sh`, with the rest of its arguments:

| Job | Runs on | What it does | Writes under `$OUT` |
| --- | --- | --- | --- |
| `dump.sh [ROOT...]` | a CPU node with the CUDA toolkit | finds every binary with device code under the roots (the toolkit's libraries, NCCL, the node's wheel and kernel caches, the repository's kernels), dumps each once with `cuobjdump -sass` and counts its opcodes (`sass_count.py`) | `toolkit.txt`, `corpus.tsv`, `counts/*.json`, `census.json`, `summary.txt` |
| `rates.sh` | one B200 (`ARCH=sm_100a`, the default) or one H200 (`ARCH=sm_90a`) | builds and SASS-gates `unpriced.cu`, `unpriced90.cu` or `SRC=unpriced_fp4.cu` with `../opcode_probe/probe.py`, times them under one lease, and gives the ratios the census needs (FFMA2 and FMUL2 against FFMA and FMUL, DMMA against the s8 tensor forms and beside a saturated tensor pipe); `BUILD_ONLY=1` stops after the gate | `build.log`, the binary's `.sass`, `gate-build.json`, `gate-build.txt`, `env/`, `probe.py run`'s `rows.jsonl`, `gate.json` and `times.jsonl`, `rates.json` |
| `lowering.sh` | a CPU node with the CUDA toolkit | compiles each case of `lowering.cu` for sm_90a and sm_120a and reads its SASS: the opcodes ptxas emits for a PTX form an opcode-table row cites, or its refusal, against the expectation in the script; exits 1 if a case lowers otherwise | `lowering.txt`, `sass/CASE_ARCH.sass` |
| `multimem.sh` | one RTX PRO 6000 | `multimem.cu`: the device's multicast attribute, what `cuMulticastCreate` answers, and what `multimem.ld_reduce` and `multimem.red` do on ordinary memory, each in its own child process | `multimem.txt` (its last line the verdict), `multimem.sass` |
| `eps_counts.sh` | a CPU node with the CUDA toolkit | builds the kernels ε_skip's per-instruction counts came from, out of the source trees sent with the run (`src-TAG.tar`), for sm_120a, sm_90a and sm_100a with line info, and dumps each cubin's SASS for `eps_counts.py` | `toolkit.txt`, `TAG/ARCH/`: `build.log`, `apv.gi.sass.gz`, `apv.sass.gz`, `lineinfo.txt` |

The census itself needs a CPU node with the CUDA toolkit, and the rates a B200 or an H200:

~~~text
research run --on MACHINE --cwd source -- bash calibration/pouw/sass_census/run.sh dump [ROOT ...]
research run --on MACHINE --cwd source -- bash calibration/pouw/sass_census/dump.sh [ROOT ...]     # the same job
research run --on MACHINE --project verity --source <tree> --cwd source --timeout 30m \
  --env GPU_LEASE_WHO=LANE -- bash calibration/pouw/sass_census/rates.sh
~~~

## Read a census

The readers are plain Python and need no GPU:

~~~text
cuobjdump -sass lib.so | python3 sass_count.py count --source NAME > NAME.json   # one binary's opcodes, by architecture
python3 sass_count.py merge counts/*.json > census.json                          # one census a device family
python3 sass_count.py summary census.json
python3 diff_menu.py FAMILY census.json [--out diff.json] [--unused]              # FAMILY: sm_90, sm_100 or sm_120
python3 eps_counts.py OUT --src TAG=APV_CU [--src ...] [--json FILE]              # an eps_counts.sh run's counts
~~~
