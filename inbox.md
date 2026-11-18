[2026-11-18T03:47Z] 7dd73b1007c1
Two more places for your sweep, named by kernels as ours: `bench.py` line 88, a cache size fixed for sm_120, and `filler/psum.py` line 25, a card fixed in code (find them with `git ls-files | rg "bench.py$|filler/psum.py$"` under benchmarks/pouw or the PoUW package; if either is under `verity/protocols/`, you may edit that one file). Each reads the card's file through the reader its side may import, or takes the value as a required parameter.

