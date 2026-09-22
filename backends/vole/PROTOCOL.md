# Candidate V (exploratory) — VOLE-based designated-verifier ZK (QuickSilver family) on the WP2 checker

Status: **cost model / exploratory**. Nothing here is implemented or measured. Every number comes from
`verity_numerical.explore.vole` (census from `security.census`, throughput from the cited tables) and is
reproduced in `notes-asset:campaigns/r20-proof/assets/explore/reports/vole.json`. Same tags as the two candidate documents: **[source]**,
**[derivation]**, **[proposal]**, **[assumption]**.

**Proof class, stated plainly.** This is *designated-verifier* zero knowledge: the auditor holds a secret MAC key
\(\Delta\) and the transcript convinces nobody else. It is **not** one of the contract's public-coin classes
(`backends/AGENTS.md`). If built, it would be recorded as `COMPLETE_ZK_BACKEND` with
`zk_mode = "vole-itmac-designated-verifier"` and never compared against Candidates A/B without that caveat in the
same row. It is also **not hash-based**: soundness of the VOLE rests on LPN + oblivious transfer (section 6).

## 0. Sources (version read 2026-09-22)

| tag | document | used for |
|---|---|---|
| YSWW21 | Yang–Sarkar–Weng–Wang, *QuickSilver*, CCS 2021, ePrint 2021/076 (https://eprint.iacr.org/2021/076) | Fig. 5 (1 field element per multiplication gate, 3-message online phase), Thm 1 (ext-sVOLE error \((d-1)/p^r\)), Thm 2 (soundness \((t+3)/p^r\), ZK in the \(F_{\text{sVOLE}}\)-hybrid), Thm 3 (polynomial sets \((d+t)/p^r\)), Table 2 (throughput: 4.8M mult/s 1 thread, 8.9M 4 threads, 7.6M/15.8M AND/s; m5.2xlarge, \(F_{2^{61}-1}\), \(\rho\ge40\)) |
| BMRS21 | Baum–Malozemoff–Rosen–Scholl, *Mac'n'Cheese*, CRYPTO 2021, ePrint 2020/1410 | abstract: 1.5 µs/mult over \(F_{2^{61}-1}\) at 95 ms latency / 31.5 Mbps; disjunctions; Fiat–Shamir variant (not used) |
| WYKW21 | Weng–Yang–Katz–Wang, *Wolverine*, S&P 2021, ePrint 2020/925 | abstract: 2–4 elements/gate, 1.6 µs/gate (5 threads, 200 Mbps, \(\rho=40\)); the maliciously secure sVOLE both later works use |
| DIO21 | Dittmer–Ishai–Ostrovsky, *Line-Point ZK*, ITC 2021, ePrint 2020/1446 | abstract: 2 elements/mult (1 in the ROM), prover 2–5× the cleartext evaluation; designated-verifier NIZK from VOLE |
| WYYXW22 | Weng–Yang–Yang–Xie–Wang, *AntMan*, CCS 2022, ePrint 2022/566 | SIMD packing: \(O(B+|C|)\) communication for \(B\) copies; Table 1 (elements/gate 0.82 at \(B=16\) … 0.0064 at \(B=2048\); 5.1 MB setup), Table 2 (14.07 mgps at 16 threads / 50 Mbps), §6 (m5.8xlarge, \(p=2^{59}-2^{28}+1\), single-level BGV), Thm 2 (\((2B+3)/|F|\)), 6 rounds |
| YH24 | Yang–Heath, *Two Shuffles Make a RAM*, USENIX Sec 2024, ePrint 2023/1115 | §4.1 ZK ROM: \(\ell+1\) inputs per lookup, two fan-in-\((n+T)\) products, teardown \(n(\ell+1)\); Fig. 10 (0.83 µs, 24.9 B per access at \(n=2^{20},T=2^{23}\), 1 thread, 1 Gbps, m5.2xlarge) |
| YHHKV23 | Yang–Heath–Hazay–Kolesnikov–Venkitasubramaniam, *Batchman and Robin*, CCS 2023, ePrint 2023/1257 | batched disjunctions (not needed: the checker has no branches, both arms are evaluated); cited to say so |
| BDSW23 | Baum–Dittmer–Scholl–Wang, *SoK: VOLE-based ZK*, ePrint 2023/857 | designated-verifier drawback stated (§1), LPN post-quantum status (§1), conversions §5.4 (Mystique / edabits) |
| Hab22 | Haböck, ePrint 2022/1530 | Lemma 5 / Theorem 4: the logUp identity used for the "logup" range mode (**the IT-MAC realisation is this lane's derivation**) |

No VOLE-based *lookup argument* with a published cost for a public table was found beyond YH24's ROM/set
structures and the permutation-check plugin of Diet Mac'n'Cheese; the search (ePrint 2024–2025 "VOLE lookup")
returned only an IEICE range argument built from QuickSilver. Nothing else is claimed.

## 1. Relation and embedding

Same public data as Candidate A section 1 (profile, \(K\), \(B\), \(U=BK/16\), tables, WP2 census). Values live in
\(F_p\), \(p=2^{61}-1\) — the field of every cited benchmark, and \(p>2^{60}\ge2^{2B}\) so the checker's `lt_pow`
products embed without wrap (checker F1) **[source, checker docstring]**. MACs live in \(F_{p^r}\) (subfield
VOLE); the prover sends **one \(F_p\) element per committed value whatever \(r\)** [source YSWW21 §4.2].

| census predicate | IT-MAC realisation | per unit | per VU (\(K=1536\)) |
|---|---|---|---|
| committed value (34 words, 225 hints, 62 row columns) | 1 VOLE correlation + 8 B P→V | 321 | 30 631 |
| `mul` output used downstream (358 − 114 booleans) | 1 VOLE + 8 B; 1 multiplication check | 244 | 23 424 |
| boolean \(b(1-b)=0\) | 1 multiplication check, no output, no bytes | 114 | 10 946 |
| `lin` (137 incl. chunk splits) | free (local on MACs); one batched CheckZero | 137 | 13 059 |
| range / table query, mode **logup** [derivation] | helper \(h_j=1/(z-w_j)\): 1 VOLE + 8 B + 1 mult check | 306 | 29 379 |
| mode **rom** [source YH24 §4.1] | +1 version input, 2 mult checks per element | 306 / 612 | 29 379 / 58 758 |
| mode **bits** [derivation] | one committed bit + one boolean check per bit of every range check | 3 384 bits (+22 fixed lookups by logup) | 324 911 |
| per batch, logup | multiplicities \(\sum_T|T|=66\,623\) committed; table side linear in \(m_T\) | — | — |

Totals: **unit** 871 VOLE / 664 multiplication checks / 6 968 B (logup); 871 / 970 (rom); 3 971 / 3 764 / 31 768 B
(bits). **VU** 83 434 VOLE / 63 749 checks / 667 KB (logup); 83 434 / 93 128 (rom); 381 078 / 361 393 / 3.05 MB (bits).
Range checks as bits cost \(4.6\times\) the VOLEs of the whole rest of the checker: the lookup realisation is the
whole game, exactly as for A/B (284 of the 306 queries are 16-bit range chunks).

## 2. Fields and the operating points

The MAC extension degree \(r\) is the soundness knob (`first_extension`):

| operating point | first \(r\) | statistical total, \(B=4096\) | why the smaller \(r\) fails |
|---|---|---|---|
| \(2^{-80}\) | **2** (\(F_{p^2}\approx2^{122}\)) | \(2^{-93.4}\) (\(B=1\): \(2^{-104.6}\)) | \(r=1\): \((t+3)/p\approx2^{-33}\) |
| \(2^{-96}\) | **3** at \(B=4096\) (\(r=2\) gives \(2^{-93.4}\)) | | the batched check \((t+3)/p^2\), \(t=2.6\cdot10^8\) |
| \(2^{-128}\) | **3** (\(F_{p^3}\approx2^{183}\)) | \(2^{-155}\) statistical; total \(2^{-136}\) set by the computational LPN term | \(r=2\): \(2^{-93}\)…\(2^{-105}\) |

The same quadratic/cubic split as the Goldilocks candidates, for the same reason: a \(\approx2^{122}\)-size
challenge field cannot carry a \(2^{28}\)-term union bound to \(2^{-96}\). **The benchmarks are all at \(r=1\),
\(\rho\approx40\)**: every wall-time figure below multiplies them by `ext_factor(r) = (1 + EXT_MULT_COST[r])/2`
(2 for \(r=2\), 3.5 for \(r=3\)) **[assumption]**, uncertainty factor 3.

## 3. Commitment interface

An IT-MAC: prover holds \((w, M_w)\), verifier holds \((K_w, \Delta)\) with \(M_w = K_w + w\Delta\) in \(F_{p^r}\);
committing \(w\) consumes one random sVOLE correlation and one \(F_p\) element P→V [source YSWW21 Fig. 5]. No
Merkle tree, no encoding, no hashing of the witness (`census.hash_bytes = 0`); memory is streaming
(\(O(M)\) of the cleartext check), peak taken as 4 VUs of value+MAC bytes (10.7 MB at \(r=3\)) **[assumption]**.
The preprocessing (base OTs + LPN-based sVOLE extension, WYKW21) is statement-independent, constant-round and
sublinear in communication; its rounds are reported as `vole.preprocessing_rounds = 3` **[assumption]**, never
inside the sequential depth, and its communication is **not counted** (open item).

## 4. Interactive reductions

* **Multiplication checks** [source YSWW21 §4.2]: for every \(x y = z\) (and \(b^2 = b\)) the prover forms the
  degree-2 relation on MACs; all \(t\) checks are batched with one verifier challenge \(\chi\in F_{p^r}\) into one
  pair \((U,V)\); soundness \((t+3)/p^r\) (Thm 2). Linear identities are local; all CheckZero's batch into one.
* **Lookups, mode logup [derivation]**: after the witness and multiplicities are committed the verifier sends
  \(z_T\) (and \(\beta_T\) for vector rows); the prover commits \(h_j = 1/(z_T - w_j(\beta_T))\) per query and the
  parties check \(h_j (z_T - w_j) = 1\) (1 multiplication check each) and
  \(\sum_j h_j - \sum_v m_T(v)/(z_T - t_v) = 0\) (linear: \(z_T, t_v\) public). Error per table
  \((N_T+|T|-1)/(|F_{p^r}|-|T|)\) on the conditioned challenge space (Hab22 Thm 4 as charged by
  `accounting.logup_identity_error`); needs \(\mathrm{char}\,F > N_T\): \(p=2^{61}-1 > 1.1\cdot10^8\) at \(B=4096\).
  The public pole-redraw rule of the candidates applies verbatim (the same \(z_T\notin\mathrm{Img}_T\) condition).
* **Lookups, mode rom [source YH24]**: reads/writes tuples with versions and one permutation proof per table,
  realised as two fan-in-\((n+T)\) products (Schwartz–Zippel error \((n+T)/|F_{p^r}|\)); Diet Mac'n'Cheese ships
  the `permutation-check` plugin this needs.
* **SIMD packing [source WYYXW22]**: the \(B\) VU circuits are identical, which is exactly AntMan's \((B,|C|)\)-SIMD
  setting: communication \(O(B+|C|)\), i.e. \(\approx13.1/B\) elements per gate (Table 1). Costs a single-level BGV
  AHE (an RLWE assumption on top of LPN), 6 rounds, and its benchmark field \(2^{59}-2^{28}+1\) **fails checker
  F1** (\(<2^{60}\)); a Goldilocks or \(2^{61}-1\) instantiation is untested **[assumption]**.

## 5. Transcript, rounds, depth, RTT

~~~text
 P0  preprocessing: base OTs + sVOLE extension (statement-independent, constant rounds, reported apart)
 1 M  every committed value of all B VUs (words, hints, row columns, product outputs) + m_T: one F_p element each
 2 C  z_T (+ beta_T) for the 5 tables, one message           [logup / rom only]
 3 M  helpers h_j (logup) / version inputs (rom): one F_p element each
 4 C  chi in F_{p^r}
 5 M  (U, V) in F_{p^r}^2 for the batched multiplication check; one MAC for the batched CheckZero
 6 V  checks W = U + V*Delta and the zero test; accept / reject
~~~

`rounds.protocol` = 2 (\(z\), \(\chi\)); `rounds.sequential_depth` = **2** (1 in bits mode; AntMan adds its 6).
Compare A 381 (baseline) and B 3. Network floor `depth × RTT`: +0.2 ms, 2 ms, 20 ms, 100 ms at RTT 0.1 / 1 / 10 / 50 ms.
**The RTT axis is not where VOLE-ZK loses; bandwidth is**: 2.73 GB at \(B=4096\) is 21.9 s at 1 Gbps, 219 s at 100 Mbps
(`estimate.transfer_seconds_1gbps`); QuickSilver's own Table 2 shows the arithmetic prover bandwidth-bound below
\(\approx500\) Mbps (1.2M vs 4.8M mult/s at 10 Mbps vs local-host).

## 6. Requirements gap (`backends/AGENTS.md`, `target.py`, README)

| requirement | VOLE-ZK | verdict |
|---|---|---|
| verifier model | designated verifier: the auditor's \(\Delta\) is secret; anyone learning \(\Delta\) before the check can forge; the transcript convinces no third party; a second auditor needs a second VOLE and a second proof (cost \(\times\) auditors) | The campaign's protocol is already *interactive prover–one auditor without Fiat–Shamir*, so A/B transcripts are not transferable either (a third party cannot exclude verifier collusion). The genuine losses are: no public-coin/beacon path to public verifiability later; secret verifier state to protect; per-auditor cost. README/`target.py` require acceptance "against the intended statement and approved backend identity" and say nothing about public verifiability — **acceptable only if the coordinator rules that Verity never needs a transferable or multi-auditor proof of the same batch**. |
| no trusted setup | none: base OTs are a two-party interactive setup, no common reference string | satisfied |
| assumptions | LPN (decoding random linear codes) + OT (Wolverine sVOLE) for soundness; the online phase is information-theoretic; AntMan adds RLWE (BGV) + PRG | **not hash-based**; a literal "hash-based" requirement is not met. If the requirement is "transparent, post-quantum, no trusted setup", LPN-based VOLE is transparent and believed post-quantum (BDSW23 §1) — the base OTs must then be post-quantum too (lattice OT), which none of the cited benchmarks use |
| privacy | ZK against a *malicious* verifier in the \(F_{\text{sVOLE}}\)-hybrid model (YSWW21 Thm 2, malicious-verifier case), perfect hiding of IT-MACs; requires the sVOLE protocol itself to be maliciously secure (WYKW21) | satisfied without the verifier-coin-commitment step 0 the candidates need |
| soundness \(2^{-128}\) | \(r=3\) MACs (\(F_{p^3}\)), statistical \(2^{-155}\), computational LPN term recorded at \(2^{-136}\) like the candidates' hash term | satisfied on paper; no implementation supports \(F_{(2^{61}-1)^3}\) MACs (section 8) |
| census columns | all `census.*`, `rounds.*`, `transcript.bytes`, `t.e2e.*` produced; `census.hash_bytes = 0`; `census.bytes_encoded` = value + MAC bytes | comparable only with the proof-class caveat |

## 7. Soundness decomposition (calculator `account_vole`, \(K=1536\), \(B=4096\), \(r=3\))

| component | term | log2 |
|---|---|---|
| batching | QuickSilver Thm 2, \((t+3)/p^3\), \(t=261.1\)M | \(-155.0\) |
| batching | ext-sVOLE Thm 1 \((d-1)/p^3\); batched CheckZero \(1/p^3\) | \(-183\), \(-183\) |
| lookup | logUp identities, `T_range16` \(N=111.7\)M dominates | \(-155.9\) (sum) |
| commitment/hash → LPN | computational, \(2^{-136}\) (modelling choice: matched to the candidates' \(q^2/2^{256}\), \(q=2^{60}\)) | \(-136\) |
| **total** | | **\(-136\)** (statistical alone \(-154.4\)) |
| zk | perfect (IT-MAC), in the hybrid model | — |

## 8. Census and wall time (`census_estimate_vole`, all ESTIMATES; anchor QuickSilver Table 2, 1 thread)

| column | unit (logup) | \(B=1\) | \(B=64\) | \(B=4096\) | A \(B=4096\) | B \(B=4096\) |
|---|---|---|---|---|---|---|
| VOLE correlations = committed elements | 871 | 150 k (66.6 k are multiplicities) | 5.41 M | **341.8 M** | 181 M | 780 M |
| multiplication checks | 664 | 63.7 k | 4.08 M | 261.1 M | — | 261 M quadratic |
| communication P→V (+V→P) | 7.0 KB | 1.20 MB | 43.3 MB | **2.73 GB** (+0.2 KB) | 74.7 MB | 318 MB |
| sequential depth; protocol rounds | 2; 2 | 2; 2 | 2; 2 | 2; 2 | 381; 1 589 | 3; 7 |
| prover s, \(r=3\) (\(\times/\div3\)) | 0.56 ms | 0.078 | 3.46 | **220** (4 threads ≈ 119) | not published | not published |
| prover s, \(r=2\) (\(2^{-80}\)) | 0.32 ms | 0.045 | 1.98 | 126 | | |
| verifier s (= prover, assumption) | 0.56 ms | 0.078 | 3.46 | 220 | opened 9.1 M elts | 39 M elts |
| field ops (\(F_p\) equiv.) | — | 2.35 M | 117 M | 7.43 G | 121 G | 64 G |
| peak memory (streaming) | — | 10.7 MB | 10.7 MB | 10.7 MB | 14.0 GB | 32.8 GB |
| hash bytes | 0 | 0 | 0 | 0 | 6.19 GB | 26.6 GB |
| transfer at 1 Gbps | — | 0.01 s | 0.35 s | 21.9 s | 0.6 s | 2.5 s |
| `t.e2e.rtt_10ms` (compute + 2 × RTT, no transfer) | — | 0.176 | 6.94 | 440 | \(t_c+3.81\) | \(t_c+0.03\) |
| mode rom, \(B=4096\) | 871 / 970 | | | 341.8 M / 381.6 M / 2.73 GB / 264 s | | |
| mode bits, \(B=4096\) | 3 971 / 3 764 | 3.06 MB, 0.27 s | 195 MB, 17.3 s | 1.56 G / 1.48 G / **12.5 GB** / 1 110 s | | |
| AntMan SIMD (comm only; 16-thread anchor) | — | 6.8 MB | 21.1 MB | **21.1 MB**, 75 s | | |

Per VU at \(B=4096\): 667 KB and 54 ms (\(r=3\), 1 thread). SP1 core is 3.58 s for one VU (system-level, 82 %
statement parsing); the VOLE \(B=1\) figure is 78 ms ± 3×.

## 9. Killing questions, with numbers

1. **Per-batch communication.** 2.73 GB at \(B=4096\) (logup or rom), 12.5 GB with bit decomposition: \(36.6\times\)
   Candidate A, \(8.6\times\) Candidate B. It streams (no GB of memory) but costs 22 s at 1 Gbps — longer than the
   compute of either candidate's \(B=1\) case by two orders. Only AntMan-style SIMD packing brings it to 21 MB
   (\(<\) A's 74.7 MB), at the price of a BGV assumption and an unbenchmarked field.
2. **Prover time.** 220 s single-thread at \(r=3\) (74–660 s), ≈119 s at 4 threads; per VU 54 ms. A/B publish no wall
   time; their field-operation columns (121 G / 64 G \(F_t\) mult-equivalents) are \(9\)–\(16\times\) the VOLE
   column (7.4 G), so VOLE-ZK is plausibly faster per VU at \(B=4096\) and certainly at \(B=1\) (78 ms vs SP1 3.58 s),
   but the factor-3 model uncertainty and the \(r=3\) extrapolation from \(r=1\) benchmarks mean **not proven better**.
3. **Designated verifier.** Acceptable only under a coordinator ruling that no transferable / multi-auditor proof
   of a batch is ever needed and that the auditor can protect \(\Delta\) for the duration of a session; the
   interactive candidates lose transferability too, but keep a public-coin path (beacon / Fiat–Shamir later) that
   VOLE-ZK forecloses. Cost per additional auditor: the full 2.73 GB and 220 s again.

## 10. Tooling

| item | finding (read 2026-09-22) |
|---|---|
| `GaloisInc/swanky` | Rust monorepo, MIT, last commit **2026-09-21** (active); Rust 1.98.0 pinned; "research software, do not deploy" |
| `diet-mac-and-cheese` (edge crate) | QuickSilver / Mac'n'Cheese over SIEVE Circuit IR; fields `F2, F40b, F63b` (binary), `F61p`, `F127p`, `F128p`, `F384p/q`, secp256k1; plugins: `permutation-check` ✔, `mux` ✔, `ram` ✘; LPN sizes small/medium/large; `svole-wykw` = WYKW21 sVOLE; edabits conversions |
| gap | **no \(F_{p^r}\) MACs over a prime \(p\)**: \(2^{-128}\) means `F384p` values+MACs at \(r=1\) (48 B/element, \(6\times\) the bytes, unbenchmarked rate) or adding an extension backend |
| `emp-toolkit/emp-zk` | C++ reference of Wolverine/QuickSilver/Mystique/AntMan (\(F_{2^{61}-1}\), Boolean); not Rust |
| overnight prototype | yes for a **diagnostic**: the transition unit (871 inputs / 664 mults) as SIEVE IR over `F61p`, `dietmc` prover/verifier on one pod, \(r=1\) (\(\approx2^{-33}\) soundness, not an operating point), measuring VOLE + check time and bytes; a \(2^{-128}\) prototype needs the extension backend first |

**Recommendation (3 lines).** Do not open a third candidate lane: communication is \(8.6\)–\(37\times\) the candidates'
and the proof is designated-verifier and LPN-based, two requirement misses. If the coordinator accepts a
designated verifier, one overnight `dietmc` diagnostic of the unit over `F61p` is cheap and would calibrate the
factor-3 uncertainty; AntMan-style packing is the only route to competitive bytes and adds an RLWE assumption.

## 11. Unproved obligations and open items (verbatim for the coordinator)

1. logUp with IT-MAC helpers (section 4) is this lane's derivation; no VOLE paper states it. The ROM alternative
   is published (YH24) but costs 2 checks per element.
2. All throughput is extrapolated from \(r=1\) benchmarks by `ext_factor(r)`; no \(F_{p^3}\)-MAC implementation exists
   to check it.
3. VOLE preprocessing communication (LPN extension, base OTs) is not counted; the SoK calls it sublinear.
4. The LPN computational term is a modelling choice (\(2^{-136}\)) matched to the candidates' hash term, not a
   parameter set; post-quantum base OTs are not in any cited benchmark.
5. AntMan's field fails checker F1; its soundness \((2B+3)/|F|\) needs an extension it was not benchmarked with.
6. Whether Verity's auditor model tolerates a designated verifier (no transferability, secret verifier state,
   per-auditor cost) is a coordinator decision, not derivable from README / `target.py`.
7. The verifier-time = prover-time assumption and the streaming peak-memory figure are unmeasured.
