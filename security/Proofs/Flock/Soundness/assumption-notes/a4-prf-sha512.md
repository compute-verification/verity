# A4: η for a 32-byte key, and what truncation to 32 bytes costs

The statement is in `../ASSUMPTIONS.md`.

- **The derivation of η ≤ 2⁻¹²⁸** (`S = 32`, the chain's tests). It is generic key search over `min(256, 8S)` bits, plus a
  hybrid over the `q` compression calls the test's streams take. With `q ≤ 2⁴⁰` and a test of `T ≤ 2⁸⁰` hash
  evaluations, the key's term is `q·T·2⁻²⁵⁶ = 2⁻¹³⁶`. The hybrid's terms are keyed by 512-bit chaining values, so they
  are smaller still.
- **Where the secret sits.** SHA-512's message block is 128 bytes, so a 32-byte secret sits whole in the first block after
  either tag (28 bytes for `stream`, 37 for `derive`). A source of up to 91 bytes does.
- **What truncation to 32 bytes costs.** `derive` and `shard` keep SHA-512's first 32 bytes, so a key is 256 bits (SHA-512
  truncated to 256 bits counts as SHA-512, SP 800-107 §5.1). A stream block is SHA-512's full 64 bytes.
  - Truncation gives a test nothing: a test of the truncated output is a test of the full output, so its advantage is at
    most the full output's.
  - What the 256-bit key bounds is key search, which the source already caps at `8S = 256` bits for `S = 32`. So at
    `S = 32` truncation costs nothing, and 256 bits keeps 128-bit security against a quantum (Grover) search too.
  - Only a source longer than 32 bytes loses anything: its key search stops at 256 bits, still past 128-bit security.
