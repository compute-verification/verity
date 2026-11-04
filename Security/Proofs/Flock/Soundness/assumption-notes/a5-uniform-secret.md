# A5: the platform

The statement is in `../ASSUMPTIONS.md`.

- **The platform: Linux, with CPython 3.12 or later.** Re-read this on any other platform.
  - `secrets.token_bytes` reads `os.urandom`, which is `getrandom(2)`, the kernel's CSPRNG. It blocks until that is
    initialized.
  - It keeps no user-space state, so a forked process doesn't repeat its output.
  - A VM snapshot restored more than once must not repeat the kernel's CSPRNG state. That holds when the kernel reseeds
    on a VM generation change (vmgenid, Linux 5.18 and later, with hypervisor support); RDSEED, mixed in at reseeds, doesn't.
  - The CSPRNG is idealized as exactly uniform. The true bound carries its distinguishing advantage.
