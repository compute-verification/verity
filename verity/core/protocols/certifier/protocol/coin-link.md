Part of the certifier's [PROTOCOL.md](../PROTOCOL.md).

## The coin link (`coin_link`)

D7 (`Security/Definitions/NetworkCertifier/Guarantees.lean`, "The coins (D7)") puts the challenger's coins on one more
ingress link of the server's network certifier: its timing is charged on the ingress grid, its content is exempt for
coin openings that open against the challenger's committed coin root, and everything else on it is counted. The
prover's requests for coins are egress, charged on their own grid. `coin_link.CoinLink` is that link for one window's
session, and `benchmarks/live_challenger/coin_link.py` runs it on the challenger's host.

### The path

The challenger (`sampled_proofs.remote`, served by `serve.py`) and the link run on one auditor host, the device that
runs the server's network certifier (`trusted-host`). A window's session takes this path:

1. The gate's `open` signs the open entry, which binds the session to the prover's root, then starts `auditor.py run`.
2. `serve.py` signs the challenger's record, which states the session's coin commitment, then listens on the host's
   127.0.0.1:P, P being one of the peer's coin ports.
3. `run` starts the link only after that. The link reads the commitment from the signed record and listens on another
   port of the host's loopback, L. Only then does it open `ssh -N -R 127.0.0.1:P:127.0.0.1:L` to the server's own sshd.
4. The prover's client connects to the server's 127.0.0.1:P. Each connection reaches the link through the tunnel, and
   the link opens one to `serve.py` for it, one connection at a time.
5. Once `serve.py` exits, `run` stops the link (SIGTERM). The link runs to the end of its link-window, cuts whatever is
   still open, closes the tunnel and writes `coin-link.json` to the session's `out/`. The `end` entry's `files` seals
   that file and `coin-link.log`.

A link that refuses to start ends the session at once, and the `end` entry says why. The link refuses when the record
is missing or unreadable, when its parameters are invalid, or when the peer's line routes through another host. A link
that doesn't listen within the time `serve.py` has to start (`START_SECONDS`, 60 s) is killed and ends the session too,
and so does a link that ends while `serve.py` still serves, because its tunnel ended.

### What it guarantees

- **Content (RP5).** `remote.serve` answers frames in order, so each challenger frame answers the prover's oldest
  unanswered one. The answer is exempt, a `COIN`, only when four things hold:
  - the frame it answers asked for coin (label, index) after `after`;
  - the answer is an opening of that coin that opens against the committed root (`live_reason`), where the commitment
    is the one the challenger's signed record states, not one the stream carries;
  - it is that family's next coin to be exempt, so a family's exempt coins are a prefix of its coin stream, in order,
    each once;
  - its head is the opening's canonical encoding (`canonical`), byte for byte.

  Every other challenger frame is counted at 8 bits a byte, under the name of what it is (`COUNTED`):
  - `commitment` (the hello's answer), `closed` and `refused`;
  - `other` (a frame of another type) and `malformed` (not a JSON object with a type);
  - `unrequested` (an opening no open asked for);
  - `unverified` (one that doesn't open against the root for what was asked);
  - `non-canonical`;
  - `replayed` (an earlier coin again) and `out-of-order` (a later coin first).

  Every prover frame is egress and counted. A frame `remote` would refuse whole fails the link closed: one with a body,
  a head past `MAX_HEAD`, or a stream that ends inside a frame.
- **Timing (RP6).** Every frame is re-cut into whole cells of B bytes (`recut`). Its head is padded with spaces, which
  JSON reads as nothing, so a frame's cells are a function of its head alone. Cells to the prover leave on the `coins`
  grid, at r_coin cells a bucket of τ. Cells to the challenger leave on `coin-requests`, at its own r. Each grid is a
  `grid.Certifier`: constant-rate, left-packed and FIFO.

  A cell leaves only at its bucket's tick, and so do a direction's end and a connection's close. What either side sees
  of the other's timing is therefore the grid's occupancy, T·log2(r + 1) bits a link-window. Arrival jitter within a
  bucket reaches neither side: the tests run the same session under different arrival times and get the same writes
  and the same record. Flow control carries nothing from inside either. A side that leaves more than `BACKLOG` bytes
  untaken fails the link closed (`delivery-refused`), as does a queue past Q (`overflow`) or cells still queued when
  their connection ends (`dropped`). Each fault cuts the connection at that tick and is its link-window's status.
- **One clock (RP8).** The ticks read the host's monotonic clock from one origin, and `serve.py`'s release log reads
  the same clock. A tick later than `slack` (25 ms) faults both grids for the rest of the link-window
  (`monitor-unavailable`). It never shifts the grid.
- **Release order.** A challenger device releases its coin-commitment root only
  after the prover's commitment is in. The open entry, which binds the prover's root, comes before `serve.py` starts.
  The record comes before `serve.py` listens, and the link starts after it listens, so no challenger frame reaches the
  prover before the prover's root is in the ledger. The gate's `open` answer carries the commitment root too. It is
  given after the open entry, and on the gate's own ssh channel, which carries no coin.
- **WitnessPathPrivate.** The coins leave the trusted host only inside the tunnel's ssh channel. That channel goes
  straight to the server's own sshd: the link sets `ProxyJump` and `ProxyCommand` to `none`, and refuses a peer whose
  line names a jump host or a proxy (`jump`). From the server's sshd to the prover's client is the server's loopback.
  So the hop from the certifier to the server crosses no prover-controlled host beyond the server itself, and what the
  server's sshd reads is exactly what the link released.

### What it rests on

The challenger's premises `CoinsConfidential` and `OneClock`, stated in `verity/core/protocols/sampled_proofs/README.md`.
Here the challenger and the link are one device, the trusted host: the coins cross only its loopback until the link
releases them, and the link's ticks read the host's monotonic clock, which the challenger's release log reads too (RP8).

### Parameters and record

`CoinLinkParams` are public and fixed before the window, and `digest` binds them together with `COUNTED`. The defaults
are D7's, under which each grid's timing charge is 600 bits a link-window:

| B | r_coin | requests' r | τ | T | Q |
|---|---|---|---|---|---|
| 4,096 bytes | 1 | 1 | 100 ms | 600 buckets | 1,024 cells |

`coin-link.json` (`CoinLink.summary`) holds counts only:
- the parameters and their digest;
- each link-window's status on both grids;
- the coins' frames and bytes, and each counted name's frames and bytes;
- the requests' frames and bytes, and the cells each grid released;
- each connection's end, with the error type, if any, that ended either side's stream;
- the late ticks;
- `charge_bits`: each grid's timing, (ticks)·log2(r + 1), and its content, 8 bits a counted byte.

Each bucket's occupancy and every frame's content stay witness. No audit reads the record yet. The charge reaches K
once the certifier's records go through the proof service (`publish`).
