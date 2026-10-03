# PoUS HBM-residency bulk-bandwidth audit (benchmark)

A performance measurement for PoUS: how cleanly a timed bulk read of the store `C` separates **device-resident** (HBM)
from **host-resident** (system DRAM, reached over PCIe) placement on an RTX PRO 6000 Blackwell (96 GB GDDR7, peak
~1.79 TB/s device bandwidth; PCIe Gen5 x16 host↔device, ~50–64 GB/s).

The verifier names a random set of `R` blocks of `C` (64 KiB each) and a fresh key; the server must return a digest that
touches every byte of those blocks within a deadline `T`. If every block is in HBM an honest GPU reads `R·64 KiB` at
device-memory speed; if a fraction `f` of them live in host DRAM, those bytes must first cross PCIe (~30× slower), so the
digest takes longer. This directory measures the timing of both placements and the separation.

## The digest (`digest.py`)

A keyed two-level polynomial (Carter–Wegman / Toeplitz) universal hash over `GF(p)`, `p = 2^31 - 1`:

- each 64 KiB block is 16,384 LE `uint32` words `w`; its block hash is `P_g(block) = Σ_j w[j] g^{W-1-j} mod p` at a
  verifier point `g` (Horner), so changing any word changes it except with probability `W/p ≈ 2^-17`;
- the block hashes are folded by a second point `z`: `D = Σ_r h[r] z^r mod p`, so dropping, altering or reordering any
  block changes `D` except with probability `R/p`.

One lane gives per-block almost-universality `≈ 2^-17`; `HL` independent lanes with fresh `(g, z)` give `≈ 2^-17·HL` (the
driver measures `HL = 1` and `2`). The Triton kernel reduces mod `p` by **Mersenne folding** (`(x&p)+(x>>31)` twice, one
conditional subtract) rather than a 64-bit `% p`, so it stays memory-bound; the value is identical to the plain-`%`
numpy reference, and `driver.py validate` checks the device kernel against that reference bit-exactly before any timing.

## Driver (`driver.py`) and analysis (`analyze.py`)

`driver.py {validate,honest,cheater,all}` writes one JSON per command under `--out`:

- `validate`: the device kernel vs the numpy reference, bit-exact; a mismatch aborts the job (no wrong number published).
- `honest`: the honest digest time over `R` random 64 KiB blocks of an HBM-resident `C`, its distribution over many
  trials and the device read rate, for a sweep of `C` sizes and `R`; plus raw HBM-read, PCIe H2D (contiguous and
  gathered) and UVA zero-copy reference rates. `C` is kept ≫ L2 (128 MB) and blocks are random so reads hit HBM.
- `cheater`: for a fixed `C`, the best honest time when a fraction `f` of the challenged blocks live in **pinned host
  DRAM**, by the faster of two methods — (a) async H2D copy of the host blocks overlapped with the device digest of the
  rest, (b) UVA zero-copy where the kernel reads host memory directly over PCIe — over a sweep of `R` and `f`.

`analyze.py --dir DIR` reads `honest.json` and `cheater.json` and writes `separation.md` and `summary.json`: the
separation table (cheater/honest time ratio per `R`, `f`), and the recommended `(R, T)` for which `f ≥ 5%` host-resident
is distinguishable from fully device-resident with margin.

## Running (node 1, through the research queue)

```bash
cd /workspace && uv run research run --queue --on vy-nebius-1 --source . --gpus 1 --cpus 8 --max-min 150 \
  --declared-output 'out/*' -- \
  bash -lc '/workspace/jobs/venv312/bin/python benchmarks/pous/hbm_audit/driver.py all --out "$RESEARCH_RUN_DIR/out"'
```

The GPU Python with torch + Triton is the node's `/workspace/jobs/venv312`; `uv run --extra torch-cpu` is CPU-only (the
reference test `tests/test_digest_reference.py` runs there). Results are preserved to the evidence store as
`evidence/v1` artifacts and cited by art id in the memory-accounting report.
