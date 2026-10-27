# The bandwidth audit v0 (retired)

`verity/core/protocols/bandwidth` was the bandwidth audit v0: a window's tags over a PoUS store before a deadline, the
verifier's sampled check, and the ingress it certifies. Its own `PROTOCOL.md` had marked it superseded: the bandwidth
audit on the served path is the remote bandwidth audit (`verity/core/protocols/remote_bandwidth/`), and v0's certified
tag is BLAKE3, which the one-hash rule takes off the served path. Under Daniel's ruling of 26 Oct (the tree gets smaller)
it left core on 27 Oct 2026. What is here is kept, not maintained: nothing imports it, no suite runs it, and nothing here
is expected to run against the current tree.

| path | what it was |
|---|---|
| `PROTOCOL.md` | v0 as the red team granted it: where each part runs, the tags, the window, the challenger state, the guarantee and the granted rate |
| `openings.py` | the openings of a remote window of chained passes: the pooled draw, the slack `λ` three ways, and the bytes and work per opening; no code imported it. Its tests, `tests/test_openings.py`, are at `1ac082f4a1` |

The rest of the package moved, not here:

- `audit.py` and `tags.py`, the Python reference and verifier, with their tests (`test_bandwidth.py`,
  `test_bandwidth_vectors.py`, `bandwidth_vectors.py`) and pinned vectors (`vectors.json`), are in
  `benchmarks/bandwidth/`, where the GPU tagger and the benchmarks that measure its tags (`codec_tensor`, `word_codec`,
  `vllm_encoded`, `resource_accounting`, `kv_filler` and PoUS's joint runs) import them as `benchmarks.bandwidth.audit`
  and `benchmarks.bandwidth.tags`.
- The guarantee, `Bandwidth.Guarantees.AuditRO`, and its proof stay in `Security/` (the `Accounting` area): the rate audit
  (`Accounting.Guarantees.RateAudit`) is proved from it, and the claims `bandwidth-device`, `bandwidth-s-chip`,
  `bandwidth-side-cap`, `bandwidth-regen-time` and `bandwidth-instantiation` stay in `verity.core.claims` with it.
