# A4: η is a parameter, and what truncation to 32 bytes costs

The statement is in `../ASSUMPTIONS.md`.

- **η has no value.** It is an argument of the assumption, and no model fixes it yet: nothing here or in Lean bounds
  it. What a model would have to bound is described here, and the description bounds nothing.
  - A search for the key, over `min(256, 8S)` bits.
  - A hybrid over the compression calls the test's streams take, each step keyed by a 512-bit chaining value.
- **The key guess is an attack, not a bound.** A test that tries `T` sources against the streams accepts the keyed
  streams with probability at least `T·2⁻²⁵⁶` at `S = 32`, and uniform streams of `ℓ` bytes in all with probability
  at most `T·2^(−8ℓ)`. That is the attack's success: for such a test no η below the difference holds, and it says
  nothing about how small η is.
- **Where the secret sits.** SHA-512's message block is 128 bytes, so a 32-byte secret sits whole in the first block after
  either tag (28 bytes for `stream`, 37 for `derive`). A source of up to 91 bytes does.
- **What truncation to 32 bytes costs.** `derive` and `shard` keep SHA-512's first 32 bytes, so a key is 256 bits (SHA-512
  truncated to 256 bits counts as SHA-512, SP 800-107 §5.1). A stream block is SHA-512's full 64 bytes.
  - Truncation gives a test nothing: a test of the truncated output is a test of the full output, so its advantage is at
    most the full output's.
  - What the 256-bit key bounds is key search, which the source already caps at `8S = 256` bits for `S = 32`. So at
    `S = 32` truncation costs nothing. A quantum (Grover) search over 256 bits takes about `2¹²⁸` evaluations of the
    keyed function: that is the attack's cost, not a bound on η.
  - Only a source longer than 32 bytes loses anything: the search for its key stops at 256 bits.
