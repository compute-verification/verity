# Proof overhead against proof-block width (`proof_overhead/`)

How proof overhead moves with a proof block's output width, per proof system and commitment scheme, and what the
sampled audit's exfiltration bound makes of each curve. A block of R rows of an int8 matmul at depth K computes R·K MACs
and outputs 32·R bits. Small blocks pay for opening their committed inputs and outputs, which doesn't shrink with their
work; this benchmark shows where that cost takes over.

Three parts, run in order:

1. **Flock with hm96 rows in the circuit (measured).** `benchmarks/flock_s8_block`, at one K, with two instance counts at
   each rows-per-block R, so a fit separates each block's seconds from the session's:

   ~~~sh
   research run --on vy-dc-h200 --project verity --source . --cwd source -- bash benchmarks/flock_s8_block/run.sh \
     "STEPS=build stage gpu" SANITY=0 NSYS=0 LEASE_MIN=30 \
     "STATEMENTS=4096:1024:1 4096:2048:1 4096:512:2 4096:1024:2 4096:256:4 4096:512:4 4096:128:8 4096:256:8 4096:64:16 4096:128:16 4096:32:32 4096:64:32 4096:16:64 4096:32:64"
   ~~~

2. **A served window under Flock's own PCS (measured on the CPU).** `served_pcs/run.sh` builds `served_pcs/crate` inside a
   Flock b684b12 checkout with this tree's patches, as `verity/ml/flock/check_build.sh` does. It commits 2^M seeded bits with
   Ligerito under `hm96-sha512` column leaves at each code rate, opens 1 and 16 claims at random points, verifies them, and
   hashes the same bytes as hm96 rows beside it:

   ~~~sh
   research run --on auto --cwd source -- bash benchmarks/proof_overhead/served_pcs/run.sh M="26 28 30"
   ~~~

3. **The curves and the bound.** `overhead.py` reads part 1's `out/` (and part 2's `served_pcs.jsonl`), and writes
   `overhead.json`, `overhead_vs_width.png` and `exfil_vs_width.png` (its docstring has every formula):

   ~~~sh
   python benchmarks/proof_overhead/overhead.py --flock RUN/out --served-pcs RUN2/out/served_pcs.jsonl --out DIR
   ~~~

What each curve is:

- **Flock, hm96 in the circuit:** measured GPU seconds per block, over the card's peak s8 MAC rate (`verity.hardware`).
- **The split:** the block's own ANDs, from its staged record, and its hm96 openings' ANDs, modeled from the row format; a
  two-rate fit gives seconds per AND of each kind.
- **Flock, a commitment it opens by linear claims:** modeled. It is the block's own ANDs at the fitted rate, plus part 2's
  per-claim opening cost scaled from the CPU to the GPU. It doesn't include the session-level change that would let a
  Flock session open a commitment made at serving.
- **The exfiltration bound:** the 3-pager's model (one year of world inference, 1% budget), at each width's unit size,
  output bits and overhead.

Status: maintained.
