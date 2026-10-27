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

## Running (a GPU node)

```bash
bash -lc 'export TRITON_CACHE_DIR=/workspace/jobs/hbm-triton-cache; O=/workspace/jobs/hbm-out; \
    cd benchmarks/pous/hbm_audit; PY=/workspace/jobs/venv312/bin/python; \
    "$PY" driver.py validate --out "$O"; "$PY" driver.py honest --out "$O" --c-gib 1,4,8,16 --r-lo 10 --r-hi 16 --trials 150; \
    "$PY" driver.py cheater --out "$O" --cheater-c-gib 8 --r-lo 10 --r-hi 16 --fracs 0.0,0.01,0.05,0.10,0.50,1.0'
```

The GPU Python with torch + Triton is the node's `/workspace/jobs/venv312`; `uv run --extra torch-cpu` is CPU-only (the
reference test `tests/test_digest_reference.py` runs there). `honest`/`cheater` write **per point** to `--out` and skip
points already there, so pointing `--out` at a path that survives across runs (not the per-run `$RESEARCH_RUN_DIR/out`)
lets a sweep **resume** after a lease preemption: on a preemptible lease, short jobs with a warm Triton cache each
advance the sweep until it is complete. `analyze.py` excludes from the recommendation every `R` whose challenged bytes
are under `--cache-factor` (default 4) times the L2, since those reads are served partly from the cache, not HBM.
