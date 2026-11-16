# Pearl-C on every linear of a served vLLM model

This directory measures what Pearl-C costs when it serves: a vLLM model on one GPU with every linear call computed and
committed on the device by the adapter's Pearl-C run, timed against stock vLLM's FP8 on the same card, rep by rep, and
the reference replay of the passes that run retained. The cards are the RTX PRO 6000 and the H100 and H200.

`e2e.py` is the measurement. Before any timing it gates every distinct linear shape in the arm's own process: the
served output against the float product, the commitment twice bit for bit, and a control with its tile leaves omitted
that must differ. It then times prefill and decode with both engines resident, in the modes `fp8` (stock), `bf16`,
`pearlc-nohash` and `pearlc`, and with `--retain` replays the timed prompts as retained passes over poisoned buffers.
Its docstring has every mode and switch.

A replay here opens rows and words in the clear, a development shortcut: its ACCEPT is never the protocol's verdict,
which is a zero-knowledge proof. Nothing here is read by a verifier.

## Try it

List the jobs:

~~~sh
bash benchmarks/pouw/pearl_c_vllm/run.sh
~~~

It prints the usage with every job's name and exits 2. Each job's script has its full `research run` command in its
header.

## The runner, `run.sh`

`run.sh JOB [ARGS...]` starts one of the directory's jobs, the script of its name (`window` is `window.sh`), with its
arguments and environment unchanged. Each job is the command of a `research run` on a node with the card; the job's
script has the full command, its variables and its exit codes in its header, and may be started by its own path as
before.

| Job | Runs on | What it does | What it writes |
| --- | --- | --- | --- |
| `setup` | the node's CPUs | the environment a window needs: the pinned vLLM in a venv, the model at its pinned revision and FlashInfer's sampling kernels | under `$WORK` |
| `window` | 8 leased GPUs, timing on the first | the timed window: the card's identity, the shipped kernel tree against its manifest, the device check, then `e2e.py`; `MODE=smoke`, `profile` and `perplexity` are its untimed forms | `e2e.json` (or `smoke.json`, `profile.json`, `quality.json`); the retained passes under `$WORK/passes/<run id>` |
| `verify` | CPUs, after the window | Pearl+'s verdict on each retained pass in Lean (`served_passes.py lean`), beside the Python twin it is held to (`served_passes.py verify`); exits non-zero unless the Lean accepts prefill and decode and rejects every control | `<pass>-lean-check.json`, `<pass>-lean.json`, `<pass>-verify.json` and `<pass>-manifest.json` |
| `served_passes serve`, `verify` | one leased GPU, then CPUs | one served run through `verity_vllm.LLM` with every pass retained, then each replayed (`served_passes.py`) | `served.json`, `verify.json` |
| `served_inputs` | CPUs, after a pass | a retained pass's inert work per input against the scheme's bound, by census or sample, or by fresh trials | a directory of `inputs.json` and the tiles' lines, which `served_table.py` reads |
| `served_step` | one leased GPU, then CPUs | where a served decode step's time goes, one engine process an arm, under Nsight Systems or not (`served_step.py`) | a directory an arm and `breakdown.md` |
| `served_profile` | one leased GPU, then CPUs | `e2e.py`'s own run with a range on every generate and forward, profiled (`served_profile.py`) | `served-profile.json` |
| `retain_profile` | the node's CPUs for the kernel's build, then one leased GPU (an H100 or H200) for the probe and for each serve | what retaining every pass costs a served run and where its time goes: the node's copy and write rates, then each load served with no pass retained and with every pass retained under timers (`retain_profile.py`) | each load's served JSON, its profile and the logs |
| `noise_sweep` | leased GPUs, a point a lease | the served decode overhead over stock FP8 across a list of points, untimed screens (`noise_sweep.py` summarizes) | `points/<point>/`, `progress.jsonl`, `sweep.json` and `sweep.md` |
| `forming_domain` | CPUs, after a pass | whether a retained pass's credited rows meet the forming lemma's domain conditions (`forming_domain.py`) | one JSON file |

`flashinfer_env.sh` is sourced by `setup`, `window` and `served_profile`, and by `benchmarks/vllm_encoded/stack.sh`;
it is not a job.

[`DESIGN.md`](DESIGN.md) has the kernels each card runs, the scripts with no job of their own, and who reads the
directory.
