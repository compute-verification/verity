[2026-11-12T17:55Z] 584b4c44b396
WATERLINE to the network-accounting coordinator, Thu 12 Nov 9:55 AM PST. **A correction to line 1 of my 9:24 AM lines, before your noon ask to infra, and a snapshot length that keeps no content.** Both come from the id-form loopback run, `r20261112-174228-bacb` (done, exit 0, both directions "agrees"; the full report follows with the pull request).

1. **Wrong in line 1:** "an id event, whose id sits at byte 102 and so is not kept". A segment on a loopback does not start where an id event starts: the server's writes reach the wire in segments of up to 64 KB. In the run's capture at `-s 130`, 101 of the 750 data segments from the server hold a token id inside their 64 kept bytes, and each request's first 64 bytes are kept too, its request line and release headers [ran: counted in the run's `front.pcap`].
2. **`-s 54` on the loopback keeps no byte of content.** An IPv4 packet there is 14 bytes of link header, 20 of IP and at least 20 of TCP, and the reader reads only the first 48 [read: `capture_flows.py` 162 to 210]. I cut the run's filtered file to 54 bytes a packet: 0 content bytes left, and the reader's result on it equals the 130-byte file's field for field but the snapshot length, both directions "agrees", 201 pairs [ran, on my machine; the file was cut after the capture, it is not a capture taken at 54].
3. **The command for the ask, if the file should hold no content:** `tcpdump -i lo -U -s 54 -w run.pcap 'ip host 127.0.0.1 and tcp port PORT'`. It fits IPv4 on a loopback only. A packet with IP options, or an IPv6 one, is then refused by name ("kept too short for its headers"), never misread [read].
4. **Not done, for your yes:** the README's loopback command in my pull request says `-s 130`, as line 1 did. Changing it to 54, with a test that a capture kept at 54 bytes is read to the same sums, is one more commit and the suites again at the new head. I do it only if you say so.

[2026-11-12T18:12Z] 5ce2edcf495d
SCANCOUNT (bc-a6f565e0-c510-570e-873d-53e7770b1924) to the network-accounting coordinator, Thu 12 Nov 10:25 AM PST.

**Stop-and-tell, your second condition: `#Θ ≤ 2^(schedules + cells)` is false as written and needs a hypothesis on the schedules. Here is the instance, and the two forms that are true. The other four statements are not blocked and I go on with them.**

Read on lean's branch at `238ad6621c27`; nothing built yet.

## What the bound is about [read]

- The one `ReadsBy` instance is `RecordFormat.readsBy_read : f.ReadsBy f.read fun θ s => θ = s` (`security/Proofs/NetworkCertifier/PeriodCheckedChecks.lean` 99). At it `Par` is `CertifierSchedule Conn`, and `PeriodChannelReg` makes Θ hold every secret's schedule: `schedOf z ∈ Θ` for each `z ∈ Zs`.
- A `CertifierSchedule` is `r`, `D`, `adm : κ → ℕ` and `cells : κ → ℕ` (`security/Definitions/NetworkCertifier/Warden.lean`). Nothing ties `adm` to anything: `PeriodLeakChecked` takes `schedOf` as the sender's, and the rows format's `Fits` asks only that each cell's bucket is before `n·T` and that a bucket holds at most `S` frames.
- The check publishes `schedules = cells = capsTerm p.cells = Σ_c ⌈log2 (cap_c + 1)⌉` (`rowsTerms`, `NetworkCheck/Period.lean` 1322).
- The reason `#Ss ≤ ∏_c (cap_c + 1)` is in that file's docstring only, in words: "the admissions are a function of the lengths". No Lean definition is the network certifier's admission rule [ran: `rg` for `adm` and "admission" over every `.lean` of the branch; `adm` is only ever a field read].

## The instance [argued, not built]

One connection, cap 1, `r = D = 1`, `S = 1`, one window of `T = 32` buckets.

- `Θ = { ⟨1, 1, fun _ => a, fun _ => 1⟩ | a ≤ 30 }`: 31 schedules.
- Each has `cells ≤ cap`, and each fits the rows format: its one cell is in bucket `a + 1 < 32`, and no bucket holds more than one frame.
- Each gives another client view: the cell is in bucket `a + 1`. So a sender that picks the schedule sends `log2 31` bits.
- `2^(schedules + cells) = 2^(1 + 1) = 4 < 31`.
- At cap 0 every `a : ℕ` fits (a lane of no cells), so `Θ.encard = ⊤` against `2^0`.

## What is true

**(a) With a hypothesis on Θ.** If on Θ the schedules share `r` and `D`, each lane has at most `cap c` cells, and the admissions are a function of the lanes' lengths (`∀ s ∈ Θ, ∀ s' ∈ Θ, s.cells = s'.cells → s.adm = s'.adm`), then `Θ.encard ≤ ∏_c (cap c + 1) ≤ 2^schedules ≤ 2^(schedules + cells)`.
- The proof is short: `s ↦ s.cells` is injective on Θ into the functions below the caps, and `caps_le` (`Proofs.Properties.NetworkPeriod`) gives the second step.
- The `cells` term is then slack: in `PeriodLeakChecked` the fail cell is read off the record, not counted.
- The hypothesis is new. It is of the kind of `PeriodChannelReg`: about the sender's schedules, not about the verifier's or the prover's code. `NetTiming.PeriodChannel` is on `main`'s lock under `domain`.
- **The definition `main` lacks, which would discharge it: the network certifier's admission rule** (a queued connection admitted at the first lane that frees, in the clients' arrival order), as a function from the clients' arrivals and the lanes' lengths to `adm`, with the device's run computing its schedule by it. Today the schedule is the run's input (`run : Dev → CertifierSchedule Conn × Srv → …`).

**(b) With no hypothesis, by another read-back.** The rows record names each frame's connection and holds `r` padding frames a bucket from `adm c` to `adm c + D`, then the lane's cells. So `adm c` and `cells c` read off the record given `r` and `D` alone.
- A `ReadsBy` at `Par = ℕ × ℕ`, with `allows (r, D) s := s.r = r ∧ s.D = D`, makes Θ the one registered pair. Then `#Θ = 1`, and no factor counts the schedules or a connection's cells, which is what `PeriodLeakChecked`'s docstring says of itself.
- It is a new lemma about `RowsFormat` beside `lanesOfRows_bucketFrames` (`security/Proofs/NetworkCertifier/RecordRows.lean`), and no premise. [argued from `laneFrames`, `lanesOfRows` and `RowsFormat.Fits`; not built. The one edge I checked by hand: a lane of no cells reads back as the empty lane whatever admission is recovered.]

## What I do unless you say otherwise

- I state (a) in the draft with its hypothesis as one named `Prop`, called a draft, in no lock, and restate `ScanCountBounded` against it. It is the smallest step that makes soundness's point 2 a statement.
- I do not start (b): it adds a definition and a proof under `NetworkCertifier/RecordRows`, outside the scan's files. Say if you want it instead of (a) or after it, and whether that file is PERIODPROOF's.
- Which lock list (a)'s hypothesis would go in is yours and Daniel's; my statements file will carry it as new, with this instance.

