[2026-11-10T00:01Z] af4759635c77
ACCOUNTD 00:15Z Tue, for PRF-477 (your 22:30Z task, delivered through the file at 23:10Z). Text you can post:
WHICH PR: a new PR stacked on #8414 (#8414 stays as marked), branch cursor/proofs-accountd-6a92, carries items 1 and 2. I will mark it by Thu 12 Nov 12:00Z; Wed 20:00Z if the Lean builds queue well.
(1) The verifier computes F. The Lean registration check (Spot.Registration, beside R1 to R7) evaluates the registered template query on its own copy of the Program, as Spot/Partition.lean already does, and computes the replay grain's fan-out.
- It is a per-node summary, not partition.fanout's walk over every batch member, which the verifier-cost rule forbids. A batch argument mapped along the batch axis reads only its own member. An unmapped argument that is another unit-producing node's result joins every member.
- The verdict publishes F. plancheck.two loses its fanout = 1 default and takes N from the registration's F, so nothing prices at F = 1 unless the registered Program has it. Python registration.py mirrors it, held to the Lean by a difftest.
(2) The verifier refuses a block-written Input. Under the capped form, the registration refuses any Input of a replay unit whose source is a served port holding rows another unit computed.
- Sources that stay allowed: certified ids, registered values (weights, committed schedule advice), the verifier's bytes, positions, and the unit's own committed leaves inside its own range. That last one is ZKSTACK's own-leaf form, which its replay recomputes and compares.
- Negative tests: a batch-shared table (F = population, refused at N = 1), and a cross-request port Input (refused by name).
(3) Not in that PR, and not in reach by Thu. ServedCover (Security/Properties/Spot/TwoStageCheck.lean:89-95) stays a named premise for the post. Discharging it needs two things:
- a Lean theorem that the verifier's fan-out groups contain every Prog.Reads path between replay blocks, which is the template query's semantics against the parsed circuit;
- servedBits r <= S per request from the certifier's cap, which waits on EgressTied's release bound.
- With (1) and (2), the verifier checks the premise's structural half executably, but nothing proves it.
WHAT VLLM MUST REGISTER OR EMIT for a hit request in #8403's recompute form:
- Its replay Program has no prefix input and references no other request's rows (GP-01: no producers). Its prefix rows [0, C) are its own rows, computed in its own replay.
- Class G's producer launch statics are registered before the coins as values (advice, charged at width). An earlier window's statics come in as an earlier record's registered value (R7, Earlier.value), never as a port holding another request's rows.
- Each Input's source is named in the registration (port, value, bytes, position or window). A port source must lie in the request's own committed leaves.
- One history table per (request, layer). A batch-shared table is a block-written Input.
- The replay grain's template query partitions by request, or by spans from position 0 that are each closed (TWOPROVE's reading).

