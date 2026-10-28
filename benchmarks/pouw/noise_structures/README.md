# PoUW noise structures

Which noise keeps PoUW's work floor at G (the call's plain s8 GEMM) at A = B = 0, and what it costs the honest prover on
the H200: de-noising (tensor and CUDA-core parts), noise generation per call, the words committed a cell, and the proof
blocks a draw. CPU only; every cost is an estimate from rates, not a timing.

~~~bash
uv run benchmarks/pouw/noise_structures/check.py --out DIR   # exact integer checks of every cheater route; nonzero on a mismatch
uv run benchmarks/pouw/noise_structures/model.py --out DIR   # DIR/structures.json and DIR/structures.md
uv run benchmarks/pouw/noise_structures/model.py --out DIR --wrap-cost alu   # the wrap correction at the IADD3 bound instead
~~~

- `check.py`: Dither v3.2's segment split with U permuted per noise tile, per segment or not at all (and one word over the
  whole K); the merged factor's limbs and its high limb on 2:4 sparse s8; bit-plane routes for a word over fresh noise;
  synthetic wrap rates; the ANDs of an int8 MAC into a b-bit accumulator.
- `model.py`: one row a structure × call × m at its best parameters (`rows[*].gd_model` carries GD-MODEL's `pouw` keys);
  a 16-layer Llama-3.2-1B step's overhead over its linears' stock time; and `tiles`, each structure's proof block by tile
  r×c at K 2048 and K 8192, its k_log against the 2^27 cap, proving GPU-s a window at 2,773 draws, and the optimum tile
  for a window of S steps.

The rates and their sources are the constants at the top of `model.py`. A floor is the cheapest cheater route found, so
an upper bound on the true floor.
