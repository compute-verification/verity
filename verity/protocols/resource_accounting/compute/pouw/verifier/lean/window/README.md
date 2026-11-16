# `pouw_window`

This Lean package is PoUW's execution phase as the verifier sees it: the grid a phase's tile count is padded to, how
many tiles it must sample, what it makes public, its fixed length, the entries and side trees it commits, and a live
phase's round coins. It is the Lean port of `window.py`, beside it in [the verifier](../../README.md), and the
[check](../check/README.md) builds on it. Once its properties are proven it takes over from the Python, which keeps
running until then.

Tokens served past the honest run's end-of-sequence are not useful work, so a workload of record serves at a public
fixed length, ignore_eos at each slot's public max_tokens, which the execution phase registers and the verifier checks
(`PouwWindow.Length`).

## Try it

In this directory:

~~~sh
lake build
~~~

This builds the library and `pouw-window-difftest`, the server the difftest asks. The difftest itself, which holds each
ported function to `window.py` on generated inputs, is a suite of the check:

~~~sh
uv run tools/check/suites.py verity-pouw-window-difftest
~~~

Each Lean module's docstring names the part of `window.py` it ports. [`DESIGN.md`](DESIGN.md) says what the difftest
covers.

Its properties are stated in `security/Properties/Pouw.lean` (the package `security_properties`), and the proved ones
are in `security/Proofs/Properties/Pouw.lean`.
