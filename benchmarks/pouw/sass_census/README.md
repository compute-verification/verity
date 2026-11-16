# sass_census: every SASS opcode a device runs, against what the model prices

Does the ACCEL model price every instruction a GPU can run? The model, which PoUW's ε is computed from, prices a menu of
instructions. This directory takes the census of a device family (sm_90: the H200, sm_100: the B200, sm_120: the RTX PRO
6000): every binary with device code on a node, disassembled and its opcodes counted, then diffed against the menu. An
opcode the menu does not price is a row a person reads: can it head a committed word, and by which route.

The probes beside the census answer what the diff raises: the rate of a class the census leaves unpriced, what ptxas
lowers a PTX form to, and what `multimem` does on a card with no NVLink. The opcode table's rows
(`../../machine_model/data/cover/rows.toml`) cite them.

## Run it

Each job is the command of a `research run` from a pushed commit. The census itself needs a CPU node with the CUDA
toolkit, and the rates a B200 or an H200:

~~~text
research run --on MACHINE --cwd source -- bash benchmarks/pouw/sass_census/run.sh dump [ROOT ...]
research run --on MACHINE --cwd source -- bash benchmarks/pouw/sass_census/dump.sh [ROOT ...]     # the same job
research run --on MACHINE --project verity --source <tree> --cwd source --timeout 30m \
  --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/sass_census/rates.sh
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
