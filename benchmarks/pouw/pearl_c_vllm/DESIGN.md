# Pearl-C on every linear of a served vLLM model: design

The kernels, the scripts with no job of their own and who reads them, beside [the README](README.md).

## The kernels

The cards are the RTX PRO 6000 (`../../../verity/protocols/resource_accounting/compute/pouw/prover/kernels/pearl_c_sm120/`'s kernels) and the H100 and H200
(`../../../verity/protocols/resource_accounting/compute/pouw/prover/kernels/pearl_c_sm90/`'s).

## The scripts with no job of their own

| Script | What it does |
| --- | --- |
| `verify_run.py` | the reference replay of one retained pass: the execution phase restarted from its salt coin, the samples, every opened tile; prints ACCEPT or REJECT |
| `fast_tile.py` | the batch forms of the tile check that `verify_run.py --fast` replays on |
| `served_table.py` | the served table, per model, from `served_inputs`' measurements and the verify verdicts |
| `forming_headroom.py` | whether the forming term the replayed inert work leaves out could change a measured tile's verdict |
| `h3_screen.py` | the device path's calls against the reference: `capture` on one GPU, `check` on CPUs after |
| `profile_decode.py` | an untimed profile of the arm, which `window` runs under `MODE=profile` |
| `panel_rows.py` | a window's result and its verify verdicts as the kernel panel's benchmark results |
| `volunteer_pass.py` | the honest remedy on a retained pass before its sampling coin: every tile's inert work replayed, the rows that leave a tile or a credited row over its inert-work bound given up, and the manifest's records rewritten |
| `bench_arm.py` | Pearl-C as an arm of the `verity-bench` harness (`--arm-module pearl_c_vllm.bench_arm:PEARL_C`) |

## Who reads it

`../../../verity/protocols/resource_accounting/compute/pouw/prover/kernels/pearl_c_sm90/` and `../../../verity/protocols/resource_accounting/compute/pouw/prover/kernels/served_zk/` load this directory's Python by path, and `../../../verity/protocols/resource_accounting/compute/pouw/prover/kernels/served_zk/`'s jobs start
`window.sh`; `benchmarks/vllm_encoded/stack.sh` sources `flashinfer_env.sh`; the harness loads `bench_arm.py` by its
module name, `pearl_c_vllm.bench_arm` with `benchmarks/pouw` on `PYTHONPATH` and
`benchmarks.pouw.pearl_c_vllm.bench_arm` from the repository's root (its two Pearl-C specs under
`benchmarks/harness/specs/`, its vLLM runner and its tests); the tests in `../../../tests/benchmarks/pouw/` run the arm on a stand-in driver.
