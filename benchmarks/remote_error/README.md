# remote_error

SPOT's cryptographic error in the remote application: SPOT's part of the cryptographic soundness error of the
composite `verity/remote/v1` (SPOT, PoUW's Pearl+ run verdict and the inference-only policy), for a prover making at
most T SHA-512 calls. Bound's formula proved in Lean under four open premises; value evaluated in Python; SPOT's part
only. A security figure, computed from parameters, not measured.

```text
uv run python benchmarks/remote_error/run.py              # the tables
uv run python benchmarks/remote_error/run.py --json       # the same as JSON
uv run python benchmarks/remote_error/run.py --blocks N   # at N proof blocks registered, which bound the commit strings read
research run --on auto --cwd source -- uv run python benchmarks/remote_error/run.py   # records metrics.toml's numbers
```

## Which error this is

A protocol's verifier gives a verdict; its soundness error is the chance it accepts a play whose claim is false. The
remote composite's error is the union bound over its subprotocols' errors (`Verity.Vocab.error`), and each splits in two:

- **Cryptography's part of SPOT's error, this figure:** SHA-512 collisions under `ecr/sha-512` and `cr/sha-512`, LMS
  (n = 64) forgery under `euf-cma/lms-sha512-n64`, the coin trees' hiding term, and the session's knowledge error (the
  `--zk` table's bound, its rewinding finders, the link finders and extraction). These grow with T, so the figure pools
  them over the prover's compute.
- **Sampling's part of SPOT's error, not this figure:** ε at δ, the chance a sample misses more than an ε share of wrong
  proof blocks (`Verity.Sampled.Live.sampleMiss`). It doesn't grow with T, and `benchmarks/sampling_plan/` evaluates it.
- **The other subprotocols' errors, not in the total.** The output and the figure's record say each:
  - `pouw (Pearl+): open, not in the total`: `Verity.Vocab.PouwSound`, an open premise of the guarantee.
  - `inference-only policy: open, not in the total`: `Verity.Vocab.InferenceOnlySound`, which
    `Verity.Vocab.inferenceOnlySound` proves with an empty event; its error, the base vocabulary's δ and ε_forge at its
    forger's budget, has no value in Lean.

## Contract

- **Guarantee.** `Verity.Sampled.remote_soundness_deployedReady`, the remote composite's soundness at SPOT's live session
  (`Verity.Sampled.Live.delta`), and `Verity.remote_soundness_of_subprotocols`, the union bound it instantiates. The
  script reads their open premises (four: `Verity.GateARow`, `Verity.Sampled.Instance.FamiliesPinned`,
  `Verity.Vocab.PouwSound`, `Verity.Vocab.WiresSound`) and named assumptions from the locks and prints them.
- **Bases.** Lean doesn't instantiate the remote guarantee at numeric parameters: its one instance, the honest witness,
  has every error ⊤, and no lock guarantee states a term's value. So no term is `proved at these parameters`. A term is
  `formula proved, value evaluated here` where each parameter it reads is of kind (a) or (b) below, and `assumed` where
  one is an assumption, the adversary's, or a value the tree doesn't determine. PoUW's and the policy's terms are
  `open`, sampling's `excluded`. Each term names the Lean declarations it reads.
- **Parameters, by kind.** Each says its kind and its source in the output; `run.py` reads each value of kind (a) from
  its file and fails, naming the file, when the file no longer says it.
  - (a) determined by the remote application's parameters, computed from the file:
    - `positions`, the commit strings a window's sample reads (Σ_p ratio_p): an **upper bound**, 2^(m_max − SLOT_LOG).
      The session proves one `--zk` table (`Verity.Sampled.LiveSpec.count`) of at most 2^m_max bits, m_max the largest m
      `Flock.Zk.queries0` accepts (`verity/protocols/proofs/flock/verifier/Flock/Zk.lean`), and each row a sampled proof
      block reads or writes takes an hm96 slot of 2^SLOT_LOG bits (`verity/commitments/periods/interface.py`). The exact
      count is K times the rows a sampled block reads, with K = ceil(NUM·N/DEN) by the remote application's sampling design
      (`applications/remote.toml`) and at most MAX_ROWS rows a block. N, the class 4,27,2 circuit's proof blocks, is the output
      of lowering it, which the tree doesn't hold; `--blocks N` takes it and gives min(the bound, K·MAX_ROWS).
    - `N` (the inner proof's), `innerN` at `Flock.Zk.PadLayout`'s largest length: `Flock/Zk.lean`, with the statement
      in `security/Proofs/Flock/Soundness/Discharge/ZkExec/Statements.lean`.
    - `levels` and k₀: `verity/protocols/proofs/flock/verifier/Flock/Schedule.lean`; `queries`, the coin tree's rounds:
      `Flock/Zk.lean`; `log_len`: `security/Proofs/Flock/Soundness/Discharge/ZkBind/Rate.lean`; `max_leaves`:
      `security/Properties/Verity/Game.lean`; `calls_mul`: `security/Proofs/Properties/Verity/Sampled/Instance.lean`.
    - `v`, `w`, `h` and the roots' calls (one opening's verification, one acceptance check, hashing one message,
      recomputing the registration's roots), in SHA-512 calls: the verifier's code fixes them, since the finders the
      guarantee's constructions build must run within them, but the tree holds no count of them. Each is a stated
      value, its terms are apart from the rest and `assumed`, and the output shows the total at two larger values.
  - (b) free parameters of the bound, which `Verity.Sampled.Live` quantifies over:
    - `M` and `Rw`, the link finder's trial limits (any value ≥ 1): the bound falls as either grows, with no minimum,
      and the finders' expected cost doesn't read them. Chosen at 2^256; the output shows 2^64 and 2^128.
    - `k`, the extraction budget, and `ρ`, the rate margin, the least the rate condition 2^logLen₀/(e·k) ≤ ρ admits:
      chosen where the headline's total is least; the output shows k = 2^23 and 2^25.
  - (c) an assumption about the adversary: the SHA-512 compressions a second a GPU, the figure's one adversary
    parameter, which the census doesn't state: at least the measured rate of
    `benchmarks/spot_figures/cards/commitment.all.json`, its bytes a second over a 128-byte block. The output prints that
    rate beside it and shows the total at half and twice it.
  - Named assumptions at a stated level: ε_forge under `euf-cma/lms-sha512-n64` (`Hash.Assumptions.LmsUnforgeable`
    takes it as a parameter) at ε_forge(q, Q) = q/2^256, and the `--zk` table's bound 2^-205 by
    `Claims.Flock.Numbers.εtable_zk`, a lemma where the table meets `TabZK.M1`. The output gives the total with and
    without the forgery term.
- **Arithmetic.** Exact rationals: each irrational (√, e) is bounded on the side that keeps the figure an upper bound.
- **Monitoring periods and fleets.** One play a machine a window. Over one window and each monitoring period
  (`verity.protocol.MONITORING_PERIODS`), over one datacenter (`benchmarks/census/deployments.json`'s reference, its
  machines and window length) and over the world's compute (`benchmarks/census/adversary.json`'s reference, as machines
  of the datacenter's size), at the adversary's SHA-512 rate a GPU. A term a + b·t + c·t² a play pools to at most
  plays·a + b·T + c·T². Each total is given with its share a machine-window audited. The census gives no AND gates a
  machine-window, so no share is given in AND gates.
- **Metrics.** `metrics.toml`: the total and the share, each for each fleet's week, month and year, the world's month
  last.

Its tests are `tests/benchmarks/remote_error/`.
