# SPEC: the core spec for forcing, and its margin

Daniel, 31 Oct 4:01 PM PDT. The core, core, core spec:

1. **Efficient to hash everything if honest.**
2. **Inefficient to hash everything if dishonest.**

"The goal should be to make it so that even if you can create the honest transcript 10% faster or whatever than
expected, (2) outweighs that." The trace needn't be cheap to *check*. Unpredictability can come from the protocol
(live coins); it needn't be hardware-enforced.

## Statement (coordinator's formalization, for MARGIN to tighten)

- **Honest cost:** C_H(W) is the cost of doing declared work W and producing its transcript T(W).
- **Dishonest cost:** C_D(W, U) is the least cost of any strategy that produces a T(W) the verifier accepts while doing
  undeclared work U (U ≥ 0, and U = 0 covers forging credit).
- **The spec, with margin m:** C_D(W, U) ≥ C_H*(W) + cost(U) + m·C_H*(W) for all U. C_H*(W) is a *floor*, the cheapest
  any implementation could produce T(W) honestly, and m must exceed our uncertainty in that floor (e.g. 10%).
- **Equivalently:** the cheapest dishonest route is at least (1 + m)× the fastest honest route, so a faster honest
  implementation can't buy room for U.

## Where a margin can come from

- **(a) Placement (silicon).** Honest hashing happens in line, on data already at the write port, at about 0.3 a MAC
  (FT). Any other route to the same transcript must move or recompute the data and hash it off the path, at about
  127.5 a MAC in software: m ≈ 400.
  - **With a keyed absorb under a hidden live coin** (GOV-HW), the off-path route is *impossible*, not just expensive:
    only the chip's unit knows the coin.
  - **With always-on logging,** U is in the honest transcript anyway.
  - What remains is physical: a bypass, or a leak of the coin. That's inspection.
- **(b) Resource competition (software, stock GPUs).** Honest and dishonest provers run the same instructions, so
  there is no structural asymmetry. The only margin is that undeclared work must displace transcript production,
  which is exactly the slack between calibration and the fastest implementation: the 10% Daniel worries about. No
  multiplicative m.
- **(c) Floors instead of estimates (the lead).** Credit only what a premised lower bound (H32, ShaWrites) says any
  implementation must spend. Then "10% faster than expected" is ruled out by premise, not outweighed, at the price of
  a premise that must hold for the fastest possible code. That gives m ≈ 0, with a gap of about 0.13%.

## Lane MARGIN (note `lanes/compute-research/<ts>-report-spec-margin.md`, due Sun 1 Nov 03:00Z)

For each design (the lead, design U, FT silicon, always-on logging with the keyed absorb), state C_H*, C_D and m
precisely:
- what sets m;
- whether m survives a 10% (or 2×) faster honest implementation;
- the cheapest dishonest route, priced.

Then two more:
- Is there **any software-only asymmetry on stock GPUs** that gives a multiplicative m? Candidates: data movement
  (honest hashes where the data already is), the register-file capacity, latency deadlines with live coins, and the
  noised multiply's coin-dependence. Show it, or show why not.
- **What is the smallest silicon change that gives m ≥ 1?**
