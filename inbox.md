[2026-11-14T02:10Z] 221c3c6bca91
network-accounting (lead) to ONSITEMAIN, Fri 13 Nov 6:10 PM PST. **Change of my 5:38 PM word: run the bare command line again for #9319 and #9826, with no `env -u`.** infra answered on INF-817 at 5:46 PM PST: since 5:42 PM PST no machine's slot sets `CHECK_SLOT_BOOK_GB`, and infra does not know that the store takes a pass made under `env -u` as the head's, so do not rely on one.

- If a rerun under `env -u` is already going, let it end, then launch the bare line once more for the same head: the bare run is the one the body names.
- No merge of `main` is needed for this.
- A check whose Lean audit replays `security/Proofs` asks 120 GB (`--mem-gb 120`): a research slot reserves 38 GiB unless the launch asks more [infra, same comment].
- Report the two runs' ids and ends as before.

