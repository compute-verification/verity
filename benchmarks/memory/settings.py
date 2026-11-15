"""`verity bench memory`'s environment variables, each declared once (`verity_config.settings`). It reads two of its
own: the node run's session in either of its forms;
the ones a research run sets are the shared settings every registry reads, and what `gate_run.sh` and `run.sh` read
passes through to them whole."""
from __future__ import annotations

from verity_config.settings import VALUE, Registry
from verity_config.settings import Setting as S

REGISTRY = Registry(
    __file__, (),
    S("PACED", str, "1:1000:2773:7", VALUE,
      "the node run's one session as the verifier host's `--paced` takes it, one word of four numbers: the passes a "
      "coin, a stretch's milliseconds, the samples a stretch and the credited stretches"),
    S("UNPACED", str, None, VALUE,
      "the node run's one session with one coin a window, in place of PACED's paced one, as one word of three "
      "numbers: the windows, a window's milliseconds and the openings a window (the verifier host's `--windows`, "
      "`--window-ms` and `--samples`)"),
    S("PLAN", str, None, VALUE,
      "the session's plan under --loopback or --local: unset, the honest one (`honest`, `serve`); another plan, "
      "such as --loopback's `skip:0.0625`, is a cheat the verifier refuses, so the command prints no number"),
    S("LOCAL_WORK", str, "~/.cache/verity/bench-memory", VALUE,
      "--local's working directory, the only place it writes: the vLLM environment, the checkpoint, the package "
      "caches and the tap builds the tree's bootstrap makes there, the lease's lock files where no `gpu-lease` is "
      "installed, and each run's records under runs/"),
    S("LOCAL_NODE", str, None, VALUE,
      "--local's node sheet (`hardware/nodes/`), in place of the one for this machine's card (`LOCAL_SHEETS`); the "
      "verifier's node is one card of it"),
    S("LOCAL_ROW", str, None, VALUE,
      "--local's served row, in place of the Llama 3.2 1B row for this machine's card (`LOCAL_ROWS`)"),
    S("VLLM_PY", str, None, VALUE,
      "--local's vLLM interpreter, a Python with vLLM and the `verity-vllm` launcher beside it; set, the bootstrap "
      "does not run, the checkpoint is the one under HF_HOME and the taps are the row's defaults or ROW_ARGS's"),
    S("GPU", str, None, VALUE, "--local's card, an index nvidia-smi gives; unset, the first"),
)
