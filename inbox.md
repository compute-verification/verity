[2026-11-17T05:05Z] 61e9500ded70
05:05Z from the circuits lead, one more for your pull request, as its own small commit: `verity bench compile_cost` with no argument exits 2 (argparse: neither --window nor --requests), so the bare command the list names does not start. Make `--window` default to `long-w0`, the window its metric's command names, when neither is given; keep --requests working. Test: the bare command's argument parsing picks long-w0. That also settles which window the series holds: long-w0 only from now.

