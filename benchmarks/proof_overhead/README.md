# Proof overhead against proof-block width (`proof_overhead/`)

How proof overhead moves with a proof block's output width, and what the sampled audit's exfiltration bound makes of each
curve. A block opens its committed inputs and outputs, a cost set by the bytes it reads rather than the work it does, and
this benchmark shows where that cost takes over and what amortizes it. Two block shapes of int8 matmul:

- **Row blocks:** one row of x against R rows of W at depth K (R·K MACs, 32·R output bits), packed, every instance's own
  rows. Each weight byte is read once per block, so widening the block amortizes only x's row.
- **Tiles:** C rows of x against P rows of W (P·C·K MACs, 32·P·C output bits), each row hashed once per tile, one u16 leaf an
  s8 (the tile's layout). A square tile amortizes both operands.

Three parts, run in order:

1. **Flock, its rows' hiding commitments (`hm96-sha512`) opened in the circuit (measured).** `benchmarks/flock_s8_block`
   with two instance counts for each shape, so the analysis separates each block's seconds from the session's:

   ~~~sh
   research run --on vy-dc-h200 --project verity --source . --cwd source -- bash benchmarks/flock_s8_block/run.sh \
     "STEPS=build stage gpu" SANITY=0 NSYS=0 LEASE_MIN=30 \
     "STATEMENTS=4096:1024:1 4096:2048:1 4096:512:2 4096:1024:2 4096:256:4 4096:512:4 4096:128:8 4096:256:8 4096:64:16 4096:128:16
                 512:2048:1x1 512:4096:1x1 512:1024:2x2 512:2048:2x2 512:512:4x4 512:1024:4x4 512:256:8x8 512:512:8x8 512:64:16x16 512:128:16x16 512:256:16x1 512:512:16x1"
   ~~~

2. **A commitment Flock opens by linear claims (measured on the CPU).** `served_pcs/run.sh` builds `served_pcs/crate` inside
   a Flock b684b12 checkout with this tree's patches, as `verity/ml/flock/check_build.sh` does. It commits 2^M seeded bits
   with Flock's PCS (Ligerito, under `hm96-sha512` and under plain `sha512` column leaves) at code rates 1/2 and 1/4, opens 1
   and 16 claims at random points, and verifies them. Beside that it commits the same bytes as hiding commitments
   (`hm96-sha512` rows), serving's cost today:

   ~~~sh
   research run --on auto --cwd source -- bash benchmarks/proof_overhead/served_pcs/run.sh "M=26 28 30"
   ~~~

3. **The curves and the bound.** `overhead.py` reads part 1's `out/` and writes `overhead.json`, `overhead_vs_width.png`,
   `share_vs_width.png` and `exfil_vs_width.png` (its docstring has every formula):

   ~~~sh
   python benchmarks/proof_overhead/overhead.py --flock RUN/out --out DIR
   ~~~

What each curve is:

- **Measured:** GPU seconds per block, over the card's peak s8 MAC rate (`verity.hardware`). This is C-Flock's prover in
  warm loopback sessions with the ZK server's send check off (`FC_FIREWALL` unset): SPOT's proving cost. The ZK server is
  a separate layer, priced apart from it, by `benchmarks/one_stage/zk_cost.py`'s `fw` arm against its `nofw` arm.
- **The split:** Flock's prover pays for cells. Each instance's block is 2^k_log of them, holding the rows' SHA-512
  compressions, one hm96 slot a row and the block's own units, as `verity_flock.circuit` lays them out (`overhead.layout`,
  checked against every record's k_log). The commitment's share is its slots' share of the used cells.
- **The floor:** the units' own cells at the rate fitted over every shape (seconds per block cell), with the commitment
  opened for free, which no commitment scheme beats with this prover. A linear-claim commitment's real costs are part 2's.
  Its opening work grows with the committed polynomial's size, so where it lands between the curve and the floor depends on
  how much data one commitment covers.
- **The exfiltration bound:** the 3-pager's model (one year of world inference, 1% budget), at each shape's unit size,
  output bits and overhead.

Status: maintained.
