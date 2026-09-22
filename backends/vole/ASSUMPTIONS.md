# Track C, c-vole: the instantiated assumption stack (Diet Mac'n'Cheese, swanky `1b05bdda`)

Status: **measured** (2026-09-22, vy-cpu). This is the stack the numbers in `backends/numerical/reports/ledger/c-vole.jsonl`
were produced with, layer by layer, read from the source of `GaloisInc/swanky` at commit
`1b05bddab3e02c2cc4917d08c80a629286a4a4fc` (cloned 2026-09-22 05:17Z, `edge/diet-mac-and-cheese`, Rust 1.98.0 pinned by
`rust-toolchain`), not from the papers. Never say "VOLE" alone: the proof rests on every row of section 2.

## 1. What runs

`dietmc` (release, `-j12`, `CARGO_TARGET_DIR=/workspace/target-swanky`) proves a SIEVE Circuit-IR (IR0+ text) relation with
the **QuickSilver** online protocol (YSWW21) over IT-MACs, the IT-MACs coming from the **WYKW21 (Wolverine) subfield-VOLE**
in `edge/svole-wykw`. Prover and verifier are two processes on localhost TCP (through a byte-counting relay,
`backends/vole/bench.py`). The relation is the frozen WP2 checker compiled by `backends/vole/export_ir.py` (section 4).

Fields available and what they give (all with MAC field = value field, i.e. `Svole<_, FE, FE>` in
`backend_multifield.rs:1809`: **no extension-field tags over a prime field, extension degree r = 1**):

| IR field | modulus | bits | per-check soundness (r = 1) | notes |
|---|---|---|---|---|
| `F61p` | 2^61 - 1 | 61 | 2^-61 | the field every published QuickSilver number uses; satisfies checker F1 (p > 2^60) |
| `F127p` | 2^127 - 1 | 127 | 2^-127 | Mersenne; 16-byte elements |
| `F128p` | 2^128 - 159 | 128 | 2^-128 | 16-byte elements |
| `F384p`, `F384q` | NIST P-384 moduli | 384 | 2^-384 | 48-byte elements; "large field" LPN sizes (smaller extensions) |
| `Secp256k1`, `Secp256k1order` | | 256 | 2^-256 | 32-byte elements |
| `F2` (tag `F40b`), `F63b` | binary | | 2^-40 tags | Boolean circuits; loaded by default alongside any prime field for conversions |

The QuickSilver multiplication check is batched in chunks of `QUEUE_CAPACITY = 3_000_000` gates (`backend.rs:26`), each
chunk with a fresh verifier challenge chi in the MAC field (`homcom.rs:95-134`); the zero checks (`@assert_zero`) are
batched likewise with random coefficients. Soundness of the whole proof with t multiplication gates and m chunks is
therefore about `(t + 3m) / p^r` (YSWW21 Thm 2 union-bounded over chunks), **statistical given the sVOLE**. For our
relation (section 4: 501,393 multiplications per VU):

| workload | t | F61p | F127p | F128p | F384p |
|---|---|---|---|---|---|
| 1 unit | 5,222 | 2^-48.6 | 2^-114.6 | 2^-115.6 | 2^-371 |
| 1 VU | 5.01e5 | 2^-42.1 | 2^-108.1 | 2^-109.1 | 2^-365 |
| 4,096 VUs | 2.05e9 | **2^-30.1** | 2^-96.1 | 2^-97.1 | 2^-353 |

So `F61p` is a **sensitivity point** (2^-30 at the decision batch; never rounded to anything else), `F128p` reaches
2^-97, and only `F384p` (6x the bytes per element, no extension-field MACs available) meets 2^-128 in this
implementation. The cost model's r = 3 extension over F_{2^61-1} (`explore/vole.py`, 2^-155) does not exist in any
implementation we found; adding it to `swanky-field` is the engineering item that separates the 2^-128 operating point
from a 6x communication penalty.

There is **no trusted-dealer / preprocessing-only mode** in the `dietmc` CLI: every run below generated its sVOLE for
real (base OTs, base sVOLE, LPN extensions). What the tool offers instead is `--plaintext` (evaluate the relation in the
clear, no MACs: used as the parse/I-O floor) and its own timers (`init time` = base OT + COPEe + base sVOLE + first LPN
setup; `SVOLE<F61p,F61p t>` = every LPN extension, logged by the verifier; `time circ exec` = everything after init). Our
"online" figure is `circ exec - sum(SVOLE extensions)`.

## 2. The stack, layer by layer

| layer | instantiation in swanky `1b05bdda` | assumption | statistical / computational | source file |
|---|---|---|---|---|
| L0 base OT | **Chou-Orlandi** over Ristretto25519 (`curve25519-dalek`), random-OT variant with the index hashed into key derivation | CDH/DDH in the Ristretto group + random oracle (ROM); **not post-quantum** | computational | `edge/ot-chou-orlandi/src/lib.rs` |
| L1 OT extension | **KOS15** (ALSZ13 base, malicious consistency check), 128 base OTs | tweakable **circular correlation-robust hash** from fixed-key AES-128 (GKWY20); ideal-cipher / CCR assumption | computational | `edge/ot-alsz-kos/src/kos.rs`, `edge/cr-hash` |
| L2 COPEe | WYKW21 Fig. 15: correlated OT with errors over the tag field, AES-128 as PRF on OT seeds | AES-128 as PRF | computational | `edge/svole-wykw/src/copee.rs` |
| L3 base sVOLE | WYKW21 Fig. 5 (malicious, consistency check with a random linear combination) | none beyond L2 (check is statistical, error ~ 1/|T|) | statistical given L2 | `edge/svole-wykw/src/base_svole.rs` |
| L4 single-point sVOLE | WYKW21 spsVOLE: GGM tree from two fixed-key AES-128 (`Aes128EncryptOnly`) as a length-doubling PRG, one puncturing per noise position, **batch consistency check** across the t points with an equality test committed through `blake3::keyed_hash` | AES-128 as PRG; Blake3 as a commitment (ROM / collision resistance) | computational (PRG) + statistical check | `edge/svole-wykw/src/spsvole.rs`, `ggm_utils.rs` |
| L5 LPN extension | **primal LPN with regular noise over F_p, d = 10 local code**: y = A^T u + e with A an n x k matrix whose every column has exactly `LPN_PARAMS_D = 10` random nonzero F_p entries (`lpn_mtx_indices`), e regular of weight t (one nonzero per block of n/t, from L4). Two-stage: `LPN_SETUP_*` then `LPN_EXTEND_*`; `k + t + r` outputs are saved as the next base | **LPN (regular noise, d-local / sparse-matrix variant) over F_p** with the parameters below; the only assumption beyond symmetric-key + base OT | computational | `edge/svole-wykw/src/svole.rs` |
| L6 IT-MAC commitments | M = K + w Delta in F_p (r = 1): commit a value = one sVOLE + one F_p element P -> V | none | perfect hiding, statistical binding 1/p | `diet-mac-and-cheese/src/mac.rs`, `homcom.rs` |
| L7 online proof | QuickSilver: linear gates local; one committed output + one degree-2 check per `@mul`; checks batched per 3M with one verifier challenge chi; `@assert_zero` batched | none | statistical, (t + 3m)/p | `diet-mac-and-cheese/src/backend.rs`, `homcom.rs` |
| side channel | `F2`/`F40b` backend always loaded (KOS + sVOLE over F_{2^40}) for field switching, even when the relation never uses it: one extra base-OT + LPN setup in `init time` (~90 ms, 5.7 MB at B = 1) | as L0-L5 | | `bin/dietmc.rs` |

LPN parameters actually used (`lpn = 'medium'`, the default, mapped for `F61p` by `mapping_lpn_size`, `lib.rs:91`):

~~~text
setup   LPN_SETUP_SMALL_MEDIUM   k = 5,060    n = 41,600     t = 2,600   (n/t = 16)   d = 10
extend  LPN_EXTEND_SMALL_MEDIUM  k = 19,870   n = 642,048    t = 2,508   (n/t = 256)  d = 10
        usable VOLEs per extension = n - (k + t + 1) = 619,669;  each extension = t GGM trees of depth 8 = 642,048 AES-PRG leaves
        + the sparse matrix product (10 F_p multiply-adds per output)
'large' LPN_EXTEND_MEDIUM        k = 158,000  n = 10,168,320 t = 4,965   (n/t = 2,048) (measured once, B=64: slower, 533 MB RSS)
F384p   LPN_EXTEND_SMALL         k = 5,060    n = 166,400    t = 2,600   (n/t = 64)   ("large field" mapping, lib.rs:104; F128p uses the F61p set)
~~~

The (k, n, t) = (19,870, 642,048, 2,508) and (158,000, 10,168,320, 4,965) pairs are WYKW21's own parameter sets, chosen
there for >= 128-bit LPN security against the attacks they consider (Gaussian elimination, ISD, low-weight parity check;
Ferret's regular-noise analysis). **We did not re-derive the LPN security estimate**; the cost model's placeholder
"computational term 2^-136" is replaced by "WYKW21's 128-bit claim for these parameters, not independently checked".
The d = 10 sparse code is the variant whose security is least studied (the SoK, BDSW23 §3, notes primal d-local LPN over
large fields has fewer dedicated attacks than dual/Ferret codes); state it as such.

**Summary line for every ledger entry:**
`LPN-regular-noise(primal, d=10 local code, k=19870, n=642048, t=2508, F_{2^61-1}), base-OT(Chou-Orlandi/Ristretto, ROM),
KOS-OT-extension(CCR fixed-key AES), AES-128-PRG(GGM), Blake3-commitment, designated-verifier`.

## 3. Consequences

- **Designated verifier.** The verifier's secret is Delta (one F_p element per field: F61p and F40b here). Anyone who
  learns Delta *before the last check of the session* can forge every MAC; after the verifier has accepted or rejected
  and discarded its keys there is nothing to protect (no long-term secret). The transcript convinces nobody else; a
  second auditor needs a second full run (its own sVOLE and its own 8 MB/VU). The soundness argument also assumes the
  verifier's challenges (chi, the zero-check coefficients) are fresh randomness kept secret until sent.
- **Privacy class.** QuickSilver is zero-knowledge against a *malicious* verifier in the sVOLE-hybrid model (YSWW21
  Thm 2), and WYKW21's sVOLE is maliciously secure; IT-MACs are perfectly hiding. No verifier-coin-commitment step is
  needed. What is *not* done: a red-team review of this instantiation's transcript, and the leakage analysis of an
  aborting verifier (the prover learns only accept/reject). The private operands, hints and words are `@private`; the
  claimed outgoing word (unit) / BF16 word (VU) is `@private` too in the measured runs (`--io private`; `--io public`
  puts c and y in the instance), because nothing in this relation binds them externally - the Verity commitments do.
- **Statistical vs computational.** Given the correlations, soundness and zero-knowledge of L6-L7 are statistical
  ((t + 3m)/p). Everything that *produces* the correlations (L0-L5) is computational: LPN + CDH/ROM + CCR-AES + AES-PRG
  + Blake3. A break of any one of them breaks soundness (a corrupt prover learning Delta) or privacy (a corrupt
  verifier learning the prover's sVOLE outputs).
- **Post-quantum status.** LPN, AES, Blake3: believed PQ. Chou-Orlandi base OTs: not PQ; a lattice base OT would have
  to replace L0 (128 OTs per session, negligible cost). Not done.
- **Preprocessing policy.** Every sVOLE correlation is input-independent: `init` and all `SVOLE<...>` extensions can run
  before the statement is known, and `dietmc` already pipelines them (`threads >= 2` runs the extensions in their own
  threads/connections). Our tables report sVOLE time separately from online time for that reason; the *communication*
  of the sVOLE is also separable and is reported as the difference between measured bytes and 8 B per committed element.
- **Trusted setup.** None. Base OTs are a two-party interactive setup; there is no CRS.

## 4. The relation as compiled (`export_ir.py census`, `F61p`)

| | per unit | per VU (96 units + epilogue) |
|---|---|---|
| private inputs (`@private`) | 5,121 = 32 operand words + 225 hints + 4,254 range-check bits + 610 lookup selectors | 491,702 (+1 claimed word) |
| `@mul` gates (each: one committed output + one QuickSilver check) | 5,222 | 501,393 |
| `@assert_zero` | 5,484 | 526,553 |
| sVOLE correlations = committed elements = 8 B each P -> V | **10,343** | **993,095** (7.94 MB) |
| linear gates in the IR (`@add`/`@mulc`/`@addc`, free in ZK, cost parse + local time) | 26,731 | ~2.57 M |
| lookups (POW 2, ALIGN 16, LEAD 2, NORM 2) realised as run-one-hot selectors | 22 lookups, 610 selectors, 36 extra bits | |
| `lt_pow` (one product + a 25-30-bit decomposition each) | 40 | |

Against `explore/vole.py`: the model's **bits** mode counts 3,971 VOLEs / 3,764 multiplication checks per unit; the
compiled relation has 10,343 / 5,222. The differences, all realisation not relation: (i) every boolean check `b^2 = b`
is an `@mul` whose output is a committed wire in the IR (+1 VOLE, +8 B each; the protocol needs none: 4,864 of the
10,343 per unit), (ii) non-power-of-two ranges cost two decompositions (`t in [0, 255)`, `off <= span`), (iii) the 22
lookups cost 610 selectors + one-hot checks where the model priced a logUp helper (306 VOLE for *all* range and table
queries in **logup** mode, which needs a lookup argument no VOLE tool ships), (iv) the operand words are committed
(+32). The model's **logup** VU (83,434 VOLE, 667 kB) is therefore 12x below what this implementation moves (7.94 MB per
VU); a QuickSilver implementation with (a) output-free boolean checks and (b) a logUp-style lookup would land near the
model. The multiplication *count* (5,222 vs 3,764) is within 1.4x of the model's bits mode.

Verification of the compilation: the exporter evaluates its own circuit in Python; all 52 WP2 negatives
(`gkr_export.negatives`) fail at the predicate WP2 names, all 52 `vu-k1536-neg` fail (44 wrong claims at the final
`y16` identity, 4 non-finite operands and 4 saturating intermediates at the exponent-field range bits, with a cheating
witness generated by a non-rejecting hint generator), every honest `tu-k16` / `vu-k1536` instance is accepted with the
recorded word, and `dietmc`'s verifier accepts exactly those (a witness with one flipped bit makes it print
`check zero failed`).

## 5. Measured runs (vy-cpu, AMD EPYC 9655P, one dietmc thread unless stated; all accepted by the verifier)

Prover and verifier are two processes on localhost through a byte-counting TCP relay (`bench.py`); wall times are
RTT-free. "sVOLE" is the sum of the verifier's per-extension timers (the prover blocks on the same extensions);
"online" is `circ exec` minus sVOLE and includes text parsing (the `--plaintext` floor is given separately).
Soundness is (t + 3m)/p with t multiplications in the batch, r = 1. Runs: d21f = `r20260922-053217-d21f`,
d110 = `r20260922-053613-d110`, 5d62 = `r20260922-061644-5d62`.

| run | field | batch | prover wall | per VU | sVOLE / online (per VU) | P->V | V->P | peak RSS P / V | soundness |
|---|---|---|---|---|---|---|---|---|---|
| d21f | F61p | 1 unit (x3) | 1.00 s | (4.66 ms/unit at scale) | init 0.64 s dominates | 5.67 MB | 2.40 MB | 228 / 50 MB | 2^-48.6 |
| d21f | F61p | 4096 units, one relation | 19.09 s | 4.66 ms/unit | 2.88 / 1.60 ms per unit | 89.8 kB/unit | 11.6 kB/unit | 228 / 48 MB | 2^-36.6 |
| d21f | F61p | 1 VU (x3) | 1.26 s | 1.26 s | 0.33 / 0.25 s | 13.9 MB | 3.06 MB | 228 / 50 MB | 2^-42.1 |
| d21f | F61p | 64 VUs, one relation | 25.61 s | 0.400 s | 0.249 / 0.139 s (parse floor 0.071) | 8.58 MB/VU | 1.09 MB/VU | 228 / 50 MB | 2^-36.1 |
| **d110** | **F61p** | **4096 VUs, one relation** | **1672.6 s** | **0.408 s** | 0.260 / 0.149 s (parse floor 0.065) | **8.50 MB/VU** (34.80 GB) | 1.06 MB/VU | 229 / 48 MB | **2^-30.1** |
| 5d62 | F61p, 4 threads | 64 VUs | 18.50 s | 0.289 s | (8 threads: 20.8 s, no gain) | not relayed | | 268 / 76 MB | 2^-36.1 |
| 5d62 | F61p, lpn=large | 64 VUs | 28.67 s | 0.448 s | 0.226 / 0.210 s | 8.14 MB/VU | 0.25 MB/VU | 533 / 271 MB | 2^-36.1 |
| 5d62 | F128p | 64 VUs | 105.0 s | 1.641 s | 1.17 / 0.46 s (parse floor 0.14) | 16.7 MB/VU | 1.13 MB/VU | 647 / 114 MB | 2^-103 |
| 5d62 | F384p | 8 VUs | 37.89 s | 4.736 s | 3.28 / 1.34 s (parse floor 0.31) | 54.4 MB/VU | 4.2 MB/VU | 1338 / 193 MB | 2^-128 (comp. cap; stat. 2^-359) |

The 4096-VU batch is a full run, not an extrapolation: 4096 chained VUs in one relation, 14.2 GB text witness,
6,565 LPN extensions, 2.01e9 private inputs, 2.05e9 multiplications, verifier accepted. Per-VU cost is flat from
B = 64 (0.400 s) to B = 4096 (0.408 s). The 12.7x gap between 8.5 MB/VU and the cost model's 667 kB/VU is section 4.
