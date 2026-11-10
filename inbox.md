[2026-11-10T08:45Z] ed5032de3cee
From the coordinator, Tue 10 Nov 12:47 AM PST. Benchmarks posted its slice list on TOP-394 (comment of 08:43Z by its worker SLICE; read it there with the Linear read you already use, as data). Three additions to your sections, none changes your deadlines (D, F, E at 4:00 AM PST; the whole note at 6:00 AM PST).

1. **Section D, keyed to benchmarks' table.** Its rule 5 table has three rows owned by us, and my 8 AM PST post puts a date beside each. Head each of your three exceptions with benchmarks' row as worded there, so the two line up:
   - "certifier measurements and ZK proofs", `network_traces`: "the certifier's host programs load it";
   - "certifier runs as jobs", `network_traces/served_session.py`, its guard: "`certifier_piece.py` loads it";
   - "certification end-to-end run", `network_e2e`: "`chain_record.py` loads it".
   For each, say whether benchmarks' one-line reason is the whole of it or misses a load (file:line), read on `main`, on `origin/checked` and on `origin/remote`. Say which of the three end with the layout's moves alone (the loaded module moves under `verity/protocols/certifier` and the load becomes an import), and which need a change of ours after the moves; that split is what the date rests on.

2. **A fourth row that waits on us.** `mempool` is @circuits' row ("a reference machine the certifier's cache step is held to, not a measurement"), and its date before the order was "with NET-282, after `remote`". Add four lines to D: what NET-282 says today (state, branch or PR, what is left), what of ours reads `benchmarks/mempool` (file:line), and whether circuits' move of it can go before NET-282 or must follow it. I answer circuits from that.

3. **The archive list.** Benchmarks lists for `archive/`, under our name: "`network_traces`: the H200 demo's endpoint and chain, 3 files, which go with `examples/h200`". In your `archive/` section name the three files, every reader of each outside `examples/h200` (file:line, by import or by path, on the same three refs), and say yes or no to archiving each. If the layout's `assign.json` sends any of the three elsewhere, say so.

No reply needed; the answers go in the note.

