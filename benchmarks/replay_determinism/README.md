# Replay determinism

Can a batch that vLLM served be replayed to give the same bytes, when serving is not batch invariant? The replay
pins the order: each request joins at the step it joined while serving. This benchmark serves a set of requests once,
replays them twice under the schedule it recorded, serves each request alone, and compares every module output of the
model, bit for bit, between those runs.

It runs on one GPU with one MoE model, Moonlight-16B-A3B-Instruct, which vLLM runs through the same model file as
GLM-5.3. vLLM is at the adapter's pin, with the adapter's engine profile and `VLLM_BATCH_INVARIANT` unset.

## Run it

It needs one GPU, and runs under `research run`:

~~~text
research run --on NODE --project verity --source . --cwd source -- bash benchmarks/replay_determinism/run.sh fetch
research run --on NODE --project verity --source . --cwd source --env GPU_LEASE_WHO=ID \
    -- bash benchmarks/replay_determinism/run.sh
~~~

The first downloads the model (32 GB) and keeps it, and refuses unless 50 GB stay free on `/workspace`. The second
runs the benchmark and writes `compare.md` to the run's output. It answers whether replay matches replay and served
matches replay, how a request served alone differs from the same request in its batch, and which operations vary with
the batch.

Its tests check the fingerprint and the comparison on a synthetic batch, on a CPU:

~~~sh
uv run tools/check/suites.py benchmarks/replay_determinism
~~~
