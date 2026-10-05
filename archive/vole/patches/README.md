# swanky patch for the c-compress lane (on top of `1b05bddab3e02c2cc4917d08c80a629286a4a4fc`)

Apply on a clean checkout:

~~~text
python3 archive/vole/patches/apply.py /workspace/swanky --fixed-lpn
CARGO_TARGET_DIR=/workspace/target-swanky-v2 cargo build --release -p diet-mac-and-cheese --bin dietmc
~~~

`apply.py` is anchored string surgery, idempotent, and refuses to run if an anchor is missing or ambiguous. Its
result on the pod is recorded verbatim in `swanky-verity_v0.patch` (`git diff` of the tracked files; the new file
`edge/diet-mac-and-cheese/src/plugins/verity_v0.rs` is this directory's `verity_v0.rs`, copied in by `apply.py`).

What it changes, and why each change preserves the protocol:

| file | change | protocol effect |
|---|---|---|
| `plugins/verity_v0.rs` (new) | the `verity_v0` plugin: `assert_mul`, `range, W`, `lookup, C, R, table...` | new *gadgets* over the existing `BackendT` (`input_private`, `random`, `assert_mul`, linear ops, `assert_zero`). `range`/`lookup` are Haböck's logUp identity `sum_i 1/(x_i - r) = sum_j m_j/(T_j - r)` proved with committed inverses; soundness terms `(N + R)/p` per call plus `N R (C-1)/p` for multi-column tables (random column combination). The verifier's challenges come from `random()`, which the verifier sends only after it has read the prover's commitments of the multiplicities (sequential channel). Prover-side inverses use one field inversion per call (batch trick). |
| `backend_trait.rs` | `fn assert_mul(a, b, c)` with a default (`mul`, `sub`, `assert_zero`) | plaintext backend unchanged in behaviour |
| `backend.rs` | `assert_mul` = one QuickSilver degree-2 triple `(a, b, c)` pushed into the existing `MultCheckState` batch, **no committed output**; a monitor counter | exactly the check `@mul` performs on its output wire, minus the output commitment the protocol never needed (YSWW21 §4: the multiplication check is over the triple) |
| `backend_multifield.rs`, `plugins/mod.rs`, `circuit_ir.rs` | plugin registration and dispatch (inputs are read from the caller's frame, no outputs) | none |
| `bin/dietmc.rs` | `TCP_NODELAY` on every stream | transport only. Without it every verifier challenge (a small segment after a long prover stream) stalled ~30 ms under Nagle + delayed ACK: 21.8 s -> 2.4 s for 64 VUs at F61p |
| `svole-wykw/src/svole.rs` (`--fixed-lpn`) | the sparse LPN matrix rows `(index, coefficient) x 10` are sampled **once per (rows, cols) per sVOLE instance** from the same coin-tossed `lpn_rng`, in the same order the first extension used, and reused by every later extension | the matrix was always public (derived from a coin-tossed seed); WYKW21 §6 / Fig. 7 fix one public matrix A for all iterations (fresh secret u and noise e per iteration), which is the LPN assumption as stated in `ASSUMPTIONS.md` L5. Re-sampling per extension was an implementation choice that avoided storing 10 (usize, F_p) per column (103 MB for the F61p extend matrix, +50 MB RSS measured). No assumption changes; the LPN samples now share one matrix exactly as in the paper's analysis. |

Not done: a flatbuffer emitter (the exporter would need a Python flatbuffers writer for `sieve_ir.fbs`, or a
text-to-fbs converter binary in swanky). With the compressed relation the text-IR parse + plaintext-evaluation floor
(`--plaintext`) is 0.016 s/VU: up to half of the 0.030 s/VU F61p prover, but 5% of the 0.32 s/VU F384p prover that is
the 2^-128 operating point. It is the next item at F61p, not at the decision field.
