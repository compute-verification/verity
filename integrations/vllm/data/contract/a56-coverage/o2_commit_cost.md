# O2 interior commit cost (MODELED) beside O1 (zero extra commit)

MODELED (hash costs MEASURED by L1/D5 on one Xeon vCPU, sha_ni; the layout is PROPOSED, nothing writes it)

Inputs: interior 389 GB per RU (census, 6500 x 1599); 2,186,730 head units x 7 ports; SHA-256 0.5225 ns/byte marginal; leaf frame 1333 ns + node 1302 ns (= 2635 ns per committed value at one value per leaf); machine Intel(R) Xeon(R) Processor x4, sha_ni.

| option | extra committed bytes per RU | leaves per RU | hash CPU-seconds per RU (single core) | 4 RUs |
|---|---|---|---|---|
| O1 head VU (PROPOSED partition change) | 0 | 0 | 0 | 0 |
| O2 `attn-interior/v0`, packed rows | 389 GB | 15,307,110 | 244 s (0.07 h) | 1.56 TB, 0.27 h |
| O2 `attn-interior/v0`, one value per leaf | 389 GB payload (+ frames) | 97,250,000,000 | 256,275 s (71.2 h) | 285 h |

The bytes, not the hashing, are the cost of O2 at packed framing: 389 GB per RU to write, keep and serve openings from, against the record's boundary trees today. One value per leaf (what a per-primitive VU opening single words would want) multiplies the hash cost by ~500x. Neither is measured on the interior; both use L1/D5's measured per-byte and per-frame costs.
