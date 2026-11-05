"""vLLM's scheduler as the network audit reads it: the format a vLLM server's decisions are committed in, and the circuit
that checks a committed string is one.

- `advice`: advice v0, what a vLLM server decided in a certifier session as one canonical bit string (`encode`,
  `decode`, which refuses every string that isn't an advice's one encoding), its length split by egress window
  (`bits_by_window`);
- `advice_check`: `AdviceCheck_v1{L}`, advice v0's validity as a Boolean Definition, 1 exactly when the committed bits (at
  most L) decode to a v0 advice, up to the three refusals of `decode` it defers; its AND count at each L is `size(L)`.

Verity reads the advice only as a bit string whose length is charged; the network certifier, its rows and the charge are
`verity.core.protocols.certifier`'s. Each Definition's circuit-check binding is its module's sidecar
(`<module>.circuit-check.py`, pins beside it).
"""
