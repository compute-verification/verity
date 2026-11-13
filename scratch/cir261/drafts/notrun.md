## Not run

- **The capture at the head `0978f5af04`.** It ran at `79cb0b3e28`, and the next card row (Fri 13 Nov 03:00Z to 04:25Z, card 3, cores 30-39) runs it at the head. Its lease now declares those cores (`--cores`). The three leases of today's runs declared none and were pinned only by `taskset`.
- **The `verity/definitions` and circuit-check suites with the sync**, at any head. They passed only in runs without it, at `ef693941ef`. The synced run that held them was cancelled before reporting them.
- **A prefill past `index_topk`.** No step of the capture had a prefill whose length passed 2048, so the top-k side of condition 3 is read, not run.
- **The SASS listing on node A itself.** The node has no `nvdisasm`. That node A's kernel is the listed one rests on its library's sha256 equalling the cu129 wheel's.
- **Decode context parallel, prefill context parallel, tensor-parallel 8, and GLM-5.3's own weights.** The capture served the stand-in at tensor-parallel 1.
