Part of the certifier's [PROTOCOL.md](../PROTOCOL.md).

## Advice v0 (`advice`)

Network accounting for arbitrary vLLM usage runs a precommitted AOT compiler. It takes an **advice** string, which
encodes everything serving made unpredictable, and compiles each engine step's circuit from it. The certifier
certifies the I/O, and sampled proofs hold the committed values to the compiled circuits and to the certified I/O. v0 is
the interface the compiler, the encoders and the tie to the certified I/O code against; its schema id is
`verity/advice/v0`. The module's docstring carries the same layout and its split by window.

**The charge, and what is committed.** The content advice `A_c` of the accepted-runs bound, `A_c + W·log2 #Σ_sync +
n·log2 D`, is the advice's length under the residual code's compression (`code_v1`), not v0's. v0 is the committed form, as hm96 hiding
rows, and the form the compiler reads. The advice is private: a compile unit, always proved in zero knowledge, states that the committed
Programs are compile(advice) under the advice's hiding root, with the compiler's digest registered. The
advice's root is registered before the live sample, and the compiler refuses every ill-formed advice up to the public cap
deterministically; v0's validity is that unit's first part (`advice_check`, below).

**Interface.** The `Advice`, `RequestAdvice`, `StepAdvice`, `Row` and `Source` dataclasses are frozen, and construction
refuses (`MalformedAdvice`) anything that isn't a schedule vLLM could have run. The module's functions are:
- `encode(advice) -> bytes` and `decode(data) -> Advice`;
- `total_bits(advice)`;
- `bits(advice, window, T)` and `bits_by_window(advice, T)`, v0's bits by egress window;
- `outputs(advice)`, the kept samples;
- `to_json` and `from_json`, a strict JSON form for debugging.

**What vLLM decides, and where the advice says it** (the pinned `0.28.1rc1.dev472`, v1's `Scheduler.schedule`):
- v1 has no prefill or decode phase. Each step's `SchedulerOutput` gives each scheduled request
  `num_scheduled_tokens` tokens from its `num_computed_tokens` on. That pair is a `Row`'s (`n`, `start`), and the
  model runner's input batch order is the order of the step's `rows`.
- `scheduled_new_reqs` is a request's first row. `scheduled_cached_reqs` holds the continuing rows, and the resumes
  among them, `resumed_req_ids`, are `preempted`.
- Preemption is recompute: `num_computed_tokens` goes back to 0, and the request returns to the front of the waiting
  queue. `preempted_req_ids` isn't recorded as such: a preempted request has no row until its resume.
- A waiting request with no computed tokens looks up the prefix cache, on admission and on a resume alike. A hit holds
  at most `num_tokens − 1` tokens and may end mid-block, the partial tail being copy-on-write. Blocks are cached when
  `allocate_slots` schedules them, so a hit can be on KV that another row of the same step computes. This is
  `cache_hit`, `start = cache_hit` and `sources`.
- A row's sample is kept exactly when the row reaches the request's known tokens, the prompt plus its kept outputs.
  Otherwise it is discarded (the runner's `discard_request_mask`): a partial prefill chunk, or the one row
  `async_scheduling` (on by default) runs past an EOS.
- `finished_req_ids`, and the reason, are `finish` and `output_len`.

**What I1's draft missed, and v0 adds:**
- `Row.sources`. The compiler has to wire a hit's KV, and nothing public says whose KV it is: a run of positions
  names the request that computed them, in this step or an earlier one.
- `release` per kept output, not per step. A step's frames straddle buckets: streams interleave, and vLLM can merge
  tokens into one chunk.
- `bits` takes T: a window is T buckets of `release`.
- `kind` is derived (decode exactly when n = 1 and start ≥ prompt_len) and checked.
- At most one row past a request's last output, the async one. The position rules admit it and refuse a second.
- `requests` are in order of first row, with unique ids.

**Out of scope for v0**, so a deployment that uses one of them isn't described by v0: speculative decoding
(`scheduled_spec_decode_tokens`), LoRA, structured output (the grammar bitmask), multimodal encoder inputs, KV
connectors and external hits (`num_external_computed_tokens`), cascade attention (`num_common_prefix_blocks`; set
`disable_cascade_attn`), pipeline and data parallelism (tensor parallelism is in scope, since every rank runs the same
`SchedulerOutput`), n > 1 and best_of, stop strings (a finish other than `eos` or `length` is refused), aborts and
client disconnects, unseeded stochastic sampling, CUDA-graph padding (the advice names the rows that ran, not padding,
so either the runner is eager or the compiler models the padding), and Mamba or hybrid models.

**Fields.** Each field's chooser and bit cost are below; w_req = bitlen(#requests − 1), and the widths w_len, w_rel and
w_src are the least that hold their largest values. A **determined** field is fixed by the certified I/O, the
committed values or the rest of the advice, and the encoders (ENC-*) remove it. The **free** fields are the real
`A_c`: what the serving engine, which is the prover, chooses.

| Field | Chosen by | Bits | Determined? |
|---|---|---|---|
| version | protocol | 8 | yes: structure |
| #requests | clients | 6 + L | yes: certified ingress |
| #steps | engine | 6 + L | free |
| w_len, w_rel, w_src | (derived) | 18 | yes: the rest of the advice |
| `id` | client (the body's `user`) | 6 + L + 8·bytes | yes: certified ingress |
| `prompt_len` | client | w_len | yes: certified ingress (the precommitted tokenizer) |
| `output_len` | model, or the client's `max_tokens` | w_len | yes: committed outputs, and the egress frame count |
| `finish` | model (EOS), or the client's cap | 1 | yes: committed outputs (is the last kept token EOS) |
| a step's rows − 1 | engine | w_req | free |
| `Row.request` | engine | w_req per row | free: batch membership and order |
| `Row.kind` | (derived) | 1 per row | yes: start, n, prompt_len |
| `Row.start` | engine | w_len per row | yes: cache_hit on an admitting row, the previous row's end otherwise |
| `Row.n` | engine (chunk size, token budget) | w_len per row | free on prefill chunks; 1 at known − 1 |
| `Row.cache_hit` | engine (prefix cache state) | w_len per row | free on admitting rows; 0 otherwise |
| `Row.preempted` | engine | 1 per row | free |
| `Row.sources` | engine (prefix cache's block map) | w_src + #sources·(w_req + w_len), hit rows only | free; their sum is cache_hit |
| `release` | engine's timing, inside the sync | w_rel per output | yes: certified egress records and committed frames |

What's left free are the step count, which requests each step runs and in what order, the chunk sizes, the
preemptions, the cache hits and their sources, and whether the row past an EOS ran. `release` is determined, and the
timing term `W·log2 #Σ_sync + n·log2 D` charges it. The advice carries it only so that the compiler and the tie to the
certified egress know which output each frame is.

**Validity.** `Advice` replays the schedule, keeping for each request the positions computed (`num_computed_tokens`),
the known tokens and the kept outputs, and refuses:
- a request that is out of first-row order, has no row, or keeps more or fewer than `output_len` outputs;
- a first row that resumes, or one that doesn't start at its hit;
- a continuing row that doesn't start where the last one ended;
- a row past the known tokens, or whose `kind` isn't the derived one;
- a hit on a row that doesn't admit its request;
- a resume of a finished request;
- a source on positions its request hadn't computed by that step, or a request's own KV except on a resume;
- per request, release buckets that aren't non-decreasing;
- an empty step, or a request twice in one step;
- types, ranges below 2^63, duplicate ids and ids that aren't UTF-8.

**Canonical.** An advice has one encoding and a byte string at most one advice, because any second form would be a
covert channel. The layout (MSB first, `uvar` a u6 width L then u(L), L the value's bit length) is:

    header     version u8 (0) | n_requests uvar | n_steps uvar | w_len u6 | w_rel u6 | w_src u6
    request    id_bytes uvar (≥ 1) | id, id_bytes × u8, UTF-8 | prompt_len u(w_len) | output_len u(w_len) | finish u1 (0 eos)
    step       rows − 1 u(w_req) | rows × [ request u(w_req) | kind u1 (0 prefill) | start u(w_len) | n u(w_len)
               | cache_hit u(w_len) | preempted u1 | when cache_hit > 0: sources − 1 u(w_src),
               sources × (request u(w_req) | n u(w_len)) ]
    release    Σ output_len × u(w_rel)
    padding    zero bits to a whole byte, never charged

`decode` refuses everything else: a non-minimal width or length prefix, nonzero padding, a trailing byte, truncation,
a version other than 0, and every invalid advice. The tests' fixture of two requests and five steps is 207 bits (26
bytes): a 43-bit header, 22 bits for each request, the steps at 13, 29, 25, 25 and 13, and 3 bits for each of five
releases.

**By window** (`bits_by_window`). v0's own bits, not the charge, split so that each record goes to the earliest
egress window it can show in:
- an output's release goes to its own window, bucket // T;
- a step's bits go to the earliest window of an output at that step or later, and a step that no output follows goes
  to the last window;
- a request goes to the window of its first step, and the header to step 0's.

So windows 0 … m − 1 hold every record their frames read, and the parts sum to `total_bits`. An advice with no outputs
is wholly window 0's. A per-window charge over the first m windows covers at least these records. The charge itself
is the residual code's (above).
