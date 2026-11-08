"""The route capture's environment variables, each declared once (`verity_config.settings`): run.sh sets them for the
served engine, and route_capture_plugin reads them in every process vLLM starts (``of(__file__)``)."""
from __future__ import annotations

from verity_config.settings import VALUE, WHERE, Registry, Setting as S, ints, not_zero

REGISTRY = Registry(
    __file__, ("ROUTECAP_",),
    S("ROUTECAP_OUT", str, None, WHERE, "the directory the plugin writes each recording rank's steps, kernel traces and "
                                        "capture-time operator table to; unset, the plugin does nothing"),
    S("ROUTECAP_RANKS", ints, (0,), VALUE, "the global ranks that record their steps and profile sampled ones"),
    S("ROUTECAP_PER_CLASS", int, 2, VALUE, "how many steps of each launch-shape class a recording rank profiles"),
    S("ROUTECAP_BUDGET", int, 300, VALUE, "the most steps a recording rank profiles over the run"),
    S("ROUTECAP_DUTY", float, 0.15, VALUE, "the largest share of the time since the first step a rank may spend profiling, "
                                           "once its first WARM profiled steps are done"),
    S("ROUTECAP_WARM", int, 24, VALUE, "how many profiled steps the duty cap doesn't hold back"),
    S("ROUTECAP_M_EDGES", ints, (1, 2, 4, 8, 16, 24, 31, 32, 42, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096,
                                 8192), VALUE,
      "the token-count edges a step's class rounds up to: the CUDA-graph sizes' range and every M where a GLM-5.3 route "
      "switches (FlashInfer below 32 and DeepGEMM from it, the fused all-reduce and RMSNorm up to 42 at TP8)"),
    S("ROUTECAP_CAPTURE_OPS", not_zero, True, VALUE, "record the operators run while CUDA graphs are captured, by input "
                                                    "shape (a CPU-only profiler around capture_model); 0 turns it off"),
)
