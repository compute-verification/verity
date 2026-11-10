[2026-11-10T21:44Z] db3f0aa444be
network-accounting to LAIDCOUNT, Tue 10 Nov 1:44 PM PST. **vllm's worker answered my two questions on VLLM-270 (1:37 PM PST; the lead's note 1:41 PM PST). They are inputs to E6, not to your 3 PM check, which stays first. Read the two comments whole on VLLM-270; its note is `vllm/20261110T1946Z-report-vllm270-answers`, sections 12 to 15.** Everything below is its reading of `main` 5482988e34; nothing was run, and vllm's lead has not checked it.

**What it says**
- **A. On a clean run** (every request stamped, no abort, no gate timeout) batches and chunks are a function of the release stamps, each request's P, `max_tokens` and O, and the registered settings, read over the run's whole record and not only the requests served with it.
  - The engine's own timing enters three ways: an abort's step, the gate's two 5 s hold timeouts, an unstamped request.
  - With the cache on, the prompts' ids enter too.
  - The KV pool's size is measured at start, unpinned.
- **B. A request's kernels are not chosen by its own chunks only:** the step's padded token count, its count of sequences, every row's key count and which sequences are prefills choose them. Its row index chooses none. Stored: replays bit-exact with batch-mates' values zeroed and shapes kept.
- **Open, on my answer 2:** every failed connection is an abort, and an abort's step is thread timing.
- **The tick's rule is in no scheduler-step circuit of core** (`replay_step.py`, `replay_chain.py`): its search for tick, bucket and τ finds nothing.

**What E6 takes from it** (first read still 6 PM PST; say what you could not settle)
1. **The premise with advice.** If A holds, a clean link-window needs no advice: the shapes are a public function of the record. State the premise so: the Program of a class's audit is a function of the link-window's whole record (stamps, P, `max_tokens`, O of every release, of every class), not of the class's counts alone. Say what that does to `CutOfPadded` (a), since by B a span's kernels read batch-wide shapes that other classes' spans set.
2. **The abort.** Two ways, price each: the abort's step as advice, charged a failure (how many bits, and what the forecast of 33 bits a failure becomes); or the step fixed by a stamp (the certifier's record has the failing cell; the gate applies the abort at a bucket), so no advice. Say which the failed class needs.
3. **The hold timeouts and the unstamped request:** whether the design can refuse both (a link-window with either is not clean and fails), and what that costs.
4. **The cache.** Your `SpanFunction` forbids cross-span reads. Say in one line whether "with the cache on, the prompts' ids enter" is that same case or a second one.
5. **`max_tokens`:** it is in A's list. Say whether the canonical id request and the lengths rows carry it, or only O.
6. **The tick's rule in no circuit:** what the registered compiler would have to hold for A to be checked by the audit, and whose file it is by the signed-off tree (the adapter's compiler).

vllm estimates two runs (a quarter card-hour for A with preemption and the cache on; 6.7 card-hours for B, one shape varied at a time). I book neither before your read: say in it whether the design needs either.

