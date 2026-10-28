Part of the certifier's [PROTOCOL.md](../PROTOCOL.md), "Advice for arbitrary serving: what it can carry".

### Attacks

The prices are estimates for one 60 s window at the llama8b-node1 calibration (T = 600, B = 4,096, r = 1,686, and
16 × 8 sync values), under an assumed load of 3,000 output tokens a second. That is 180,000 decode rows, 3,000 engine
steps and 900 requests a window, with Llama-3's V = 128,256 tokens (log2 V ≈ 17).

| # | Attack | Under the design | Cost to the prover | What closes it |
|---|---|---|---|---|
| 1 | A covert channel in the advice: batch compositions or chunk sizes chosen to encode weights | Charged: it is the channel `A_c` counts. v0 sends every row's fields at 48 bits a row: 10.6 Mbit a window, 3.5 times the 3.0 Mbit of token ids the window's egress itself carries, so v0's length bounds nothing the trivial bound doesn't | Nothing but throughput | An encoding that charges only departures from a public predictor: `compile` replaying a deterministic scheduler from the arrivals, or the trusted text's κ step clock and registered profiles (log2 N bits once). It is priced, never refused |
| 2 | Scheduling choices the advice describes but that don't change the outputs | Charged. The registered Program digests are a function of the advice, so the choice is in the record whether or not the advice is published | Nothing | A canonical schedule: the advice must be the predictor's, and a departure is charged as in 1 |
| 3 | Timing | Jitter is n·log2 D, which is 0 under J. The advice's `release` is timing advice, charged as advice. It replaces the sync's 7 bits with up to log2 C(S + T − 1, S) ≈ 2,330 bits a window at S = 3,000 steps (30,000 in a fixed-width field). Proposition 2 caps it at the envelope T·log2(r + 1) ≈ 6,430 bits a verification phase, since once the frames and their order are fixed, the counts are all the timing the wire shows | Nothing | Release derived from the anchors, the declared sync and the schedule (zero-bit fields), which brings back the sync's 7 bits. The times at which registrations and proofs reach the challenger are a separate channel (A6) |
| 4 | Token choice at temperature 0 | None, if A1, A2 and the tie hold: the argmax with its tie-break fixed, the decode input wired to the previous sample, and the streamed token tied | — | A2 refuses each of three breaks. (a) A decode input id not wired to the previous sampled token frees 17 bits a decode row, about 3 Mbit a window; the Commit holds `c00.input/ids` and `runner.sampler/sampled_token_ids` as separate values, so the wiring must be checked. (b) A prompt in another tokenization of the same text frees the choice among the tokenizations. (c) A request parameter read from the advice rather than the body is uncharged |
| 5 | Sampling at temperature > 0: the seed and the RNG | Open. A seed the prover chooses is 64 bits a request, about 58 kbit a window. An RNG state shared across requests is advice. A `splits` input not derived from the schedule frees each event's sample | k usable bits a request cost about 2^k runs of the request | The seed from the request body, or from coins nobody inside chooses (the challenger's, or the certifier's salt, as D7 exempts them), with `splits` derived in the circuit. Otherwise, charge 64 bits a request |
| 6 | Preemption or cache-hit claims that don't match the real cache | The real cache is never seen, so a claim is advice: charged (v0's `cache_hit` and `preempted`), and under batch-variant kernels it steers near-ties. The dangerous form is a hit whose KV has no checked writer. Fabricated KV makes the request's whole output free, and no sample catches it, since every step unit computes correctly from what it reads | Nothing | A2: a hit names its donor, the donor's token prefix equals the request's certified one, and the KV is the donor writer's output. `compile` refuses a read with no writer. A hit that an encoding derives from the prompt history is recomputed, never read |
| 7 | Advice for a different but plausible schedule that computes the same outputs | Not refused. It costs its own encoding and no more, since acceptance is existential. Nothing the prover runs is trusted, so the Commit needn't be serving's, only registered by the deadline. The leak is log2 of the number of schedules that give the same tokens and release, at most the schedule fields' charge (v0: 8.7 Mbit a window); under a replay encoding, each departure is charged at its own encoding | Under batch-invariant kernels, nothing. Otherwise, a re-execution under the claimed schedule before the registration deadline. Under a replay encoding, sending the replay's own prediction costs 0 bits whatever serving did, but the streamed tokens and the release must then be the prediction's | Nothing needs to for the network bound, since the charge counts every explanation. It breaks any claim that the advice is serving's schedule: an account that credits work or live slots by the advice's rows credits work when it wasn't done. Closing that needs the schedule attested on a trusted clock, which the design doesn't have |
| 8 | A frame let through by a wrong tie unit | Open under a sampled tie. The frame is witness that only the tie holds, so a missed frame frees all 8·B = 32,768 bits of it. At ε = 10^−3 that is 180 frames, about 5.9 Mbit a window, and the advice doesn't bound it. `io_tie`'s design samples K frames a verification phase in a unit on `All`. With a content check on `All` holding every frame canonical, a missed frame frees only its id (log2 V ≈ 17 bits), and an accept at δ leaves fewer than N·ln(1/δ)/K wrong frames: about 49,000 a link-day, 844 kbit, at K = 20, δ = 2^−40 and the busiest served window's N = 35,386 | Nothing | The tie with Selection `All` over every frame. Or the sampled positions with a content check on `All`, and the term ε·N·log2 V added beside `A_c` |
| 9 | The lengths of the windows' parts | If the parts aren't self-delimiting, how many bits each window gets is free: log2 C(c + W, W) over W windows. That is about 11 bits a window at 1 kbit a window, and 24 at 8 Mbit, more than the sync's 7 | Nothing | A5: each part self-delimiting, or a registered cap per window |
| 10 | Encodings that decode alike: padding bits, trailing bytes, non-minimal integers, unused enumeration values, an arithmetic coder's several codewords for one message | The count is unaffected while the charge is a function of the decoded advice. Free wherever the prover's bytes reach the record: a digest the registration binds can be ground to any of the message's encodings, and a seed or coin derived from it chosen. v0's `decode` refuses them all; an arithmetic-coded advice has 2 to 4 codewords a segment and reads 32 bits past it | Nothing | A5: `decode` refuses them, or the record and every digest hold the re-encoding |
| 11 | Two advices: N2, the tie and the compiler reading different strings, or a consumer reading a field from the Commit's metadata | Free: the counts, the frames' mapping and the circuits each follow a different explanation | Nothing | A4 and A5: one digest in the registration, and every consumer decodes those bytes |
| 12 | Advice chosen after the sample | The error bound grows to #𝒜_c · δ | Nothing | A4 |
| 13 | A field an encoding stops charging but a consumer still reads as sent (`finish`, `output_len`, `prompt_len`, `cache_hit`) | Free. `output_len` alone is log2 of the token cap per request, 10 bits at 1,024 | Nothing | The consumer recomputes a derived field (EOS from the sampled tokens, and `length` only at the body's cap) and never reads it |
| 14 | A request with no certified ingress, or two requests with one `user` | A phantom request's prompt is free and steers the rows batched with it. A shared id lets the prover swap outputs. v0 refuses a shared `user`, so honest traffic that shares or omits one has no advice | Nothing | Refuse both: every request the advice names is a certified request, named once, by its certified ingress position rather than a body field |
| 15 | Making `compile` or any check refuse | The window is rejected, which is visible, and a rejected window may carry up to its unbound capacity, as now | The run | The policy that already applies to rejected windows |

### Adversarial cases

Each part's tests carry these cases, each refused or charged as stated. They are small and CPU-only, and each goes into
its part's tests as the part lands. Those marked (written) run against the parts' drafts in
`test_network_certifier_advice_adversarial`, `test_network_certifier_advised_adversarial` and
`test_network_certifier_io_tie_adversarial`.
- **The advice codec:**
  - `decode(encode(a)) == a`, and `encode(decode(x)) == x` for every x that `decode` accepts: every one-bit flip, cut
    and appended byte of 12 random schedules' encodings. A wider field, a longer length prefix, a set padding bit and
    a trailing byte are refused (10, written);
  - for every window prefix, the windows' charges cover the header, every step up to the last that samples a frame
    released in it, every request first scheduled by then and those frames' release fields. The windows' charges sum
    to the encoding's length (9, 13, written). The windows' parts, concatenated, parse one way (9);
  - a skipped position, a recomputation without a resume, two rows of one request in a step, a hit on a running row,
    a hit on KV no request computed by that step, a resume of a finished request, a second row past a stop and a
    `kind` other than the derived one are refused (6, A2, written);
  - a duplicate `id`, and requests out of their first rows' order, are refused (14, written);
  - schedule choices that leave the outputs unchanged (a chunked prefill, a hit instead of a recomputation, a resume in
    place) are distinct advices, each charged (2, 7, written).
- **N2's advice mode:**
  - N2's frames are advice v0's outputs, step, request and index, on every advice v0 accepts (11, written);
  - a response handed before its request's anchor plus the window's δ and ⌊λ/ρ⌋ is refused, and with δ = 0 the
    anchor's own bucket is admitted (written);
  - egress counts one frame off in any bucket, a queued frame released early, and ingress over RIN, past or short of
    the window's `received` or after the response, are refused (11, written);
  - a response with one frame more than its `output_len` plus the framer's constant overhead is refused.
- **The I/O tie:**
  - a frame with the right token id and any other byte changed is refused: another stream, frame kind or payload
    length, a finish reason, an id in the fill (8, written);
  - a payload in m frames fails exactly those m positions, so it passes K samples with probability (1 − m/N)^K (8,
    written);
  - two frames mapped to one (request, position), and a position with no frame, are refused;
  - a prompt whose committed ids are another tokenization of the certified text is refused (4);
  - the tie and N2 read one advice digest (11).
- **The compiler and the Commit:**
  - a decode row whose input id is not the previous position's sampled token is refused (4);
  - a cache hit with no donor of an equal prefix, and a KV read with no writer, are refused (6);
  - a stochastic request's seed is the body's or the coins', never the advice's (5);
  - `compile` under an unregistered pin either gives the same Programs or refuses (A1).
