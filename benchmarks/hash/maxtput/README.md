# maxtput: a Build of record from the served match alone

`build_job.sh` writes a Build of record for a model that has none, Llama-3.1-8B on the RTX PRO 6000. It runs the vLLM
integration's served match (`verity-vllm served-match`, `adapters/vllm/verity_vllm/pipeline/served_match.py`) with no
Commit stage after it. vLLM serves the match workload with every value of every engine step committed. The Build is derived
from the serving process's own compile artifact, and its Program is checked word for word against the committed values.

## Running

From the source tree's root, on one RTX PRO 6000, with the match workload in `$RESEARCH_RUN_DIR/inputs/match.json`:

```text
RESEARCH_RUN_DIR=DIR VERITY_BENCH_VENV=VENV CASE=LLAMA31_8B CKPT_REPO=unsloth--Llama-3.1-8B CKPT_REV=REV \
  bash benchmarks/hash/maxtput/build_job.sh
```

| Variable | What it gives |
|---|---|
| `CASE` | the checkpoint role (`verity_vllm.config.CHECKPOINTS`), `LLAMA31_8B` |
| `CKPT_REPO`, `CKPT_REV` | the checkpoint's directory and revision under `$HF_HOME/hub`, read offline |
| `VERITY_BENCH_VENV` | the Python environment with CUDA torch and vLLM |
| `HF_HOME` | the Hugging Face cache, default `/workspace/hf` |
| `NATIVE_COLLECT_BUILD` | the parent directory of the native collector's builds (the vLLM integration's settings) |
| `MATCH_EXTRA` | further `served-match` arguments, such as `--build-of-record BUILD.json` or `--max-num-seqs N` |

## Output

`$RESEARCH_RUN_DIR/out` gets:

- `build.json`: the Build, with its compile artifact pinned;
- `match.json`: the Match's verdict;
- `facts.json`, `launch_configs.json`, `routes.json` and, with fused-MoE layers, `moe.json`, as `served-match` writes them;
- `triton_ptx.tgz`: the Inductor directory's autotune configs and PTX.

The exit code is `served-match`'s: 0 when the Match passes, 1 when it fails, and 3 when the Build refuses the artifact, the
device or the served switches. It is 8 when the checkpoint's snapshot is absent.

The script has no tests of its own. What `served-match` runs, the served Build's derivation and its check, is tested in
the vLLM integration (`adapters/vllm/tests/acquire/`, `adapters/vllm/tests/check/`).
