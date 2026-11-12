# pearl_c_vllm: Pearl-C on every linear of a served vLLM model

What Pearl-C costs when it serves: a vLLM model on one GPU with every linear call computed and committed on the device
by the adapter's Pearl-C run, timed against stock vLLM's FP8 on the same card, rep by rep, and the reference replay of
the passes that run retained. The cards are the RTX PRO 6000 (`../pearl_c_sm120/`'s kernels) and the H100 and H200
(`../pearl_c_sm90/`'s).

`e2e.py` is the measurement. Before any timing it gates every distinct linear shape in the arm's own process: the
served output against the float product, the commitment twice bit for bit, and a control with its tile leaves omitted
that must differ. It then times prefill and decode with both engines resident, in the modes `fp8` (stock), `bf16`,
`pearlc-nohash` and `pearlc`, and with `--retain` replays the timed prompts as retained passes over poisoned buffers.
Its docstring has every mode and switch.

## The runner, `run.sh`

`run.sh JOB [ARGS...]` starts one of the directory's jobs, the script of its name (`window` is `window.sh`), with its
arguments and environment unchanged. Each job is the command of a `research run` on a node with the card; the job's
script has the full command, its variables and its exit codes in its header, and may be started by its own path as
before.

| Job | Runs on | What it does | What it writes |
| --- | --- | --- | --- |
| `setup` | the node's CPUs | the environment a window needs: the pinned vLLM in a venv, the model at its pinned revision and FlashInfer's sampling kernels | under `$WORK` |
| `window` | 8 leased GPUs, timing on the first | the timed window: the card's identity, the shipped kernel tree against its manifest, the device check, then `e2e.py`; `MODE=smoke`, `profile` and `perplexity` are its untimed forms | `e2e.json` (or `smoke.json`, `profile.json`, `quality.json`); the retained passes under `$WORK/passes/<run id>` |
| `verify` | CPUs, after the window | `verify_run.py` on each retained pass; exits non-zero unless prefill and decode are accepted and every control is rejected | `<pass>-verify.json` and `<pass>-manifest.json` |
| `served_passes serve`, `verify` | one leased GPU, then CPUs | one served run through `verity_vllm.LLM` with every pass retained, then each replayed (`served_passes.py`) | `served.json`, `verify.json` |
| `served_inputs` | CPUs, after a pass | a retained pass's inert work per input against the scheme's bound, by census or sample, or by fresh trials | a directory of `inputs.json` and the tiles' lines, which `served_table.py` reads |
| `served_step` | one leased GPU, then CPUs | where a served decode step's time goes, one engine process an arm, under Nsight Systems or not (`served_step.py`) | a directory an arm and `breakdown.md` |
| `served_profile` | one leased GPU, then CPUs | `e2e.py`'s own run with a range on every generate and forward, profiled (`served_profile.py`) | `served-profile.json` |
| `noise_sweep` | leased GPUs, a point a lease | the served decode overhead over stock FP8 across a list of points, untimed screens (`noise_sweep.py` summarizes) | `points/<point>/`, `progress.jsonl`, `sweep.json` and `sweep.md` |
| `forming_domain` | CPUs, after a pass | whether a retained pass's credited rows meet the forming lemma's domain conditions (`forming_domain.py`) | one JSON file |

`flashinfer_env.sh` is sourced by `setup`, `window` and `served_profile`, and by `benchmarks/vllm_encoded/stack.sh`;
it is not a job.

## The scripts with no job of their own

| Script | What it does |
| --- | --- |
| `verify_run.py` | the reference replay of one retained pass: the execution phase restarted from its salt coin, the samples, every opened tile; prints ACCEPT or REJECT |
| `fast_tile.py` | the batch forms of the tile check that `verify_run.py --fast` replays on |
| `served_table.py` | the served table, per model, from `served_inputs`' measurements and the verify verdicts |
| `forming_headroom.py` | whether the forming term the replayed inert work leaves out could change a measured tile's verdict |
| `h3_screen.py` | the device path's calls against the reference: `capture` on one GPU, `check` on CPUs after |
| `profile_decode.py` | an untimed profile of the arm, which `window` runs under `MODE=profile` |
| `panel_rows.py` | a window's result and its verify verdicts as the kernel panel's rows |
| `bench_arm.py` | Pearl-C as an arm of the `verity-bench` harness (`--arm-module pearl_c_vllm.bench_arm:PEARL_C`) |

## Who reads it

`../pearl_c_sm90/` and `../served_zk/` load this directory's Python by path, and `../served_zk/`'s jobs start
`window.sh`; `benchmarks/vllm_encoded/stack.sh` sources `flashinfer_env.sh`; the harness loads `bench_arm.py` by its
module name, `pearl_c_vllm.bench_arm` with `benchmarks/pouw` on `PYTHONPATH` and
`benchmarks.pouw.pearl_c_vllm.bench_arm` from the repository's root (its two Pearl-C specs under `benchmarks/specs/`,
its vLLM runner and its tests); the tests in `../tests/` run the arm on a stand-in driver. A replay here opens rows and
words in the clear, a development shortcut: its ACCEPT is never the protocol's verdict, which is a zero-knowledge proof.
Nothing here is read by a verifier.
