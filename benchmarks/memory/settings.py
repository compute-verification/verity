"""`verity bench memory`'s environment variables, each declared once (`verity_config.settings`). It reads one of its
own, the node run's session; the ones a research run sets are the shared settings every registry reads, and what
`gate_run.sh` and `run.sh` read passes through to them whole."""
from __future__ import annotations

from verity_config.settings import VALUE, Registry
from verity_config.settings import Setting as S

REGISTRY = Registry(
    __file__, (),
    S("PACED", str, "1:1000:2773:7", VALUE,
      "the node run's one session as the verifier host's `--paced` takes it, one word of four numbers: the passes a "
      "coin, a stretch's milliseconds, the samples a stretch and the credited stretches"),
)
