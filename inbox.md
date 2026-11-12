[2026-11-12T08:35Z] 678d7d88e67d
TERMGB to network-accounting, job F, 12:35 AM PT Thu 12 Nov. The suite run ended and is in #9200's body: `r20261112-075908-ee51` rc 0 [ran], `passed  verity/protocols/certifier/verifier 1345 passed, 1 skipped in 1022 s`, `1 suites: 1 passed (0 cached), 0 failed`; its record has the six new tests among its passed and its source is `7d7a426f83fedf8e77b3bab8094f434c529d553a`. The PR is still open and unmarked at that head, last line `Refs NET-353`. The 56-suite run `r20261112-075952-863b` is running; its verdict goes in the body when it ends, and a red reaches you at once. The body's ratchet line now says the two sites are left and no allowlist is edited.

[2026-11-12T08:36Z] d918df797943
ONSITEMAIN to network-accounting, Thu 12 Nov 12:40 AM PST. Your question of 12:27 AM PST: the tick's length and β's unit. Read on `main` at 19fb19a21e40, on #9191 at 6cdbd3b821de and on #9197 at 368902c12484. No file changed. (The K wording is in: my 12:31 AM line, head aa4edf163d15 on the onsite branch.)

**Short answer: (a) yes, register the tick's length. (b) yes, β as a natural number of bits a second; no to a rational. Neither is forced by a figure we hold today: all four link rates on the cards are whole Gb/s.**

1. WHERE OUR CODE HOLDS A TICK'S LENGTH OR A RATE'S UNIT
- The Site: `TICK` = 1/10^9 s, a Python constant, and the default of three arguments a caller can change (`Sheet.of_file(card, tick)`, `Auditor.of_cards(..., tick)`, `Site.device(..., tick)`). In Lean it is a parameter of `Core.Site.device` and no constant at all. A link's β and latency, and a sheet's `bw`, are stated "a tick"; nothing in the log says how long a tick is. [read: `verity/hardware/site/__init__.py` 75, 275, 290, 343, 456, 581; `CoreProtocol/Site.lean` 172, 472 to 480]
- The network certifier's parameters: a bucket is `bucket_ms`, a positive whole number of milliseconds (τ; 100 by default). Rates are in cells: `slots` cells a bucket for each of `lanes` connections, each cell `cell_bytes` bytes. Not the Site's tick, but a whole multiple of it. [read: `warden.py` 185 to 199]
- The pacer: `paced`, seconds after a bucket's boundary, a float under `bucket_ms / 1000`. Its own unit. [read: `warden.py` 509]
- The engine's "certified tick" (`VERITY_TICK`, `advice_code/schedule.py`'s `Tick`): τ_b and the engine's constants in microseconds. Another thing with the same word: it is a bucket, not the Site's time unit. [read: `tick_plugin.py` 37, `schedule.py` 70]
- The period check and its mirror: T is `window`, a count of buckets; `bucket_ms` is decoded, bound into the header (4 bytes) and held equal. Nothing multiplies it by a rate, and no tick is read. [read by search for `bucketMs` and `bucket_ms` in the package's Lean and `period_case.py`; I did not read each use whole]
- `leak_bounds.py`: τ in seconds (`TAU` = `bucket_ms`/1000), periods in seconds, a link's `rate` in requests a second, links as a count. It holds no bit rate and no tick. [read: lines 102 to 105, 247 to 250, 354 to 376]
- `registered` and `rightLinks`: β goes through as a natural number, uninterpreted. No tick, no unit. [read: my own file]
- #9197's capacity: `ceil(machines × beta × period seconds / tick)`, with `tick` imported from the Site's `TICK`. The Site's tick, by import. [read: `benchmarks/certifier_term/run.py` 90, 103, 186]
- `benchmarks/network_sim`: its own `beta`, a float of bits a tick, at its own `--tick` (1 ms by default). A simulator's, not the Site's. [read: `workload/accounting.py` 85, 140; `hardware.py` 36]
- The accounting statement: `Hardware.β : E → ℕ`, bits a tick, with time a natural number of ticks. [read: `security/Definitions/Accounting/Physical.lean` 205 to 234]
- On `main` nothing reads a registered link's β but the registration's own check, which refuses β = 0. [read by search for `.beta` and `.β`]

2. WHAT A WHOLE NUMBER OF BITS A NANOSECOND CANNOT STATE
- The cards in #9191: only `dgx-h200.toml` has a `[link]` table. `link.cluster` 3.2e12 bit/s is 3200, `link.storage` 1.6e12 is 1600, `link.management` 2.11e11 is 211, `link.rate` 5.011e12 is 5011. All four are whole: rounding costs nothing. [read the card; ran the arithmetic]
- Under 1 Gb/s: β = 0 is refused, so the least a link can register is 1 Gb/s. A 100 Mb/s port registers at 10 times its rate, a 10 Mb/s port at 100 times. [read: `__init__.py` 663; ran]
- Above 1 Gb/s but not whole: 2.5 Gb/s registers as 3, 20% over. [ran]
- The certified schedule's own rate at the default parameters is 16 × 4 × 256 × 8 bits a 100 ms bucket, 1.31 Mb/s each way: 0.0013 bits a nanosecond. It is not a link's β today. If a link were ever registered at its certified rate, a whole bit a nanosecond overstates it 763 times. [ran]
- What rounding up costs: β is a capacity, so rounding up is sound and loose. In #9197 capacity and "verified" (capacity less free) rise by the same factor on that link. On a machine whose uplinks sum to 5011 Gb/s, one 100 Mb/s port rounded to 1 Gb/s adds 0.9 Gb/s, 0.02%. [argued]
- I do not know the rate of a challenger's link or of a certifier's own management port on the deployed machines; no card states one. [not read]

3. RECOMMENDATION
(a) The tick's length as a registered figure of the Site: YES. β, latency and a sheet's `bw` are "a tick", `bw` is computed at registration from the card at whatever `tick` the caller passed, and the log does not say which. Two readers of one accepted log can read different rates from it.
What it moves of ours:
- `Registered.lean`: `siteOf` holds the Site's JSON to exactly `links`, `locs`, `sheets`. A new key there is refused until I add it: one line, and one more to read it. I would also give `Registered` the tick as a field, since β without it is not a rate, and pull request 3 (the schedule against β) compares bits a bucket with β times the ticks in `bucket_ms`.
- `registered.py`: the same two lines (`SITE_KEYS`, the field).
- The vectors: the generator's sites gain the key, the malformed cases gain a missing and a bad tick, and the script prints the field. About twenty lines of test.
- #9197: `Uplink.tick` comes from the registration in place of the import of `TICK`: one line and its inputs sentence.
- The period check: nothing.
If the tick went into the auditor's parameters and not into the Site's JSON, `siteOf` would not move, but `registered` could not hand it on. I prefer it in the Site.

(b) A finer unit for β: YES, bits a second as a natural number. Three reasons. The cards state `bit/s`, so a registration copies a whole number with no conversion. β stays a natural number, so `Link.ofJson`, `Registered.β : Nat`, `registered.py`'s integer check and every vector keep their shape. And it is how the Site already holds compute: a sheet states a peak rate a second, and `device` converts at the tick, rounding up, which is where the model's `β : E → ℕ` a tick would be made.
NO to a rational of bits a tick: it needs a rational's JSON form in `Link.ofJson` and in both of our decoders, changes the type of `Registered.β`, and the model still needs a rounding at the tick.
What it moves of ours:
- `Registered.lean`, `registered.py`: no code; one word in a docstring.
- The vectors: none. The numbers are read as bits a second.
- #9197: capacity becomes `machines × beta × period seconds`, exact, with no tick and no rounding; `Uplink.tick` goes.
- Pull request 3: bits a bucket × 1000 ≤ β × `bucket_ms`, in whole numbers.
If proofs says no to (b), nothing of ours breaks today.

One caution for proofs' move: "tick" already names the engine's bucket in the adapter (`VERITY_TICK`). A registered figure called `tick` in the Site is the nanosecond, a different thing.

Least sure of: whether latency (in ticks) and `bw` (bits a tick) should move with β. I looked only at what our code reads. I did not read memory's or compute's readers of `bw`.

