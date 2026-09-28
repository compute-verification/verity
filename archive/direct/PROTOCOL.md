# B-Ligero — direct local constraints in an interactive hash-based ZK Ligero argument

Status: **specification**. B-Ligero is frozen; `README.md` (the Plonky3 diagnostic) and `ligero/README.md` (the Ligero
prover) say which parts are built and measured. The argument is
AHIV22's interactive ZKIPCP-plus-CRH construction applied to the census constraints of one 16-product update,
with logUp lookups (Hab22) as a second commitment round. Same tags as `archive/gkr/PROTOCOL.md`: **[source]**,
**[derivation]**, **[proposal]**, **[assumption]**.

**Label rule** (`archive/AGENTS.md`). A plain Plonky3 AIR/STARK over the same columns is
`NON_ZK_PROOF_DIAGNOSTIC`: its FRI commitment is non-interactive by Fiat–Shamir (random-oracle assumption the
requirements avoid) and its trace openings are not hidden (no masking, no ZK). It measures arithmetic and
lookup throughput only. An implementation of sections 3–8 **without step 0** (section 5.3) is
`COMPLETE_HVZK_BACKEND` (honest-verifier privacy, AHIV22 Lemma 4.15) — a milestone, never
`COMPLETE_ZK_BACKEND`. `COMPLETE_ZK_BACKEND` requires step 0 implemented, written down as in 5.3 and red-team
reviewed, and section 9 reproduced by `verity_numerical.security.accounting.account_candidate_b`.

## 0. Sources (version read 2026-09-21)

| tag | document | used for |
|---|---|---|
| AHIV22 | Ames–Hazay–Ishai–Venkitasubramaniam, *Ligero*, ePrint 2022/1608 (2022-11-18; extended CCS 2017) | ZKIPCP for arithmetic circuits (4.7: linear and quadratic constraints over interleaved RS codes), ZK variants of the three tests (4.6, App. B.4–B.5), interactive variant via CRH (5.1), Lemma 4.15 (ZK needs \(k>l+t\)), Theorem 4.7 and Appendix C (soundness), amortisation remarks |
| libzk | IETF `draft-google-cfrg-libzk-02` (22 July 2026), section 4 | concrete tableau layout: blinding rows, witness rows with `NREQ` random symbols, aligned quadratic row triples (4.3), prove/verify (4.4–4.5) |
| Hab22 | Haböck, ePrint 2022/1530, revision 2025-02-20 | logUp with helper (inverse) columns: Lemma 5, Theorem 4 (challenge from a set of size \(\ge|F|-|H|\)), Remark 3, section 3.4 (vector-valued lookups), batch inversion cost (3 mult. per helper) |
| PH23 | Papini–Haböck, ePrint 2023/1284, revision 2025-02-20 | optional `gkr` lookup mode (section 4.3) |
| FS24 | Frigo–shelat, ePrint 2024/2010, revision 2026-04-27, §2.2 | the Ligero bound used for parameters (Appendix C form) |
| HM96 | Halevi–Micali, CRYPTO 1996 (sections 1.2, 3, 4 read; see `archive/gkr/PROTOCOL.md` section 0) | statistically hiding, computationally binding commitment from CRH: HM96 leaves (section 3) and step-0 coin commitments (5.3) |

Not read (cited by name only): Goldreich–Kahan 1996 (verifier-coin commitment, simulator); the simulator sketch
in 5.3 is written out.

## 1. Relation, shapes, public data

Same public data as `archive/gkr/PROTOCOL.md` section 1 (profile, \(K\), \(B\), \(U=BK/16\), tables, census). Differences:

| item | B-Ligero |
|---|---|
| committed per unit | **every** value of the checker: the 34 words, the 225 hints, the 62 looked-up row columns, the 244 products and the 100 materialised range expressions (665 columns), plus Ligero alignment copies for quadratic operands |
| constraints per unit **[derivation from the WP2 contracts, `archive/gkr/PROTOCOL.md` section 1]** | 137 linear (115 identities + 22 range-chunk splits), 358 quadratic (`mul`, 114 of them booleans), 306 lookups into 5 public tables (284 into the one 16-bit range table, 22 into `ALIGN`/`POW`/`LEAD`/`NORM`) |
| alignment copies **[source, libzk 4.3]** | Ligero's quadratic test needs \(x,y,z\) of each product in the same column of three rows: \(3\cdot358=1\,074\) quadratic slots per unit; values reused across products are copied and bound by linear equalities |
| committed elements per unit **[estimate]** | 1 064 = \(3\cdot358\) slots + 104 columns that are neither a direct operand nor a product (`census.columns_committed_b`, `CandidateB.witness_per_unit`); the layout itself is WP3's |
| linear constraints per unit **[estimate]** | \(137+2\cdot358=853\) (census `lin` incl. splits, two copy equalities per product) — `CandidateB.linear_per_unit`; total at \(B=4096\): 335.5M incl. table-side helper equations |
| quadratic constraints per unit | 358 `mul` + 306 helper products = 664; total at \(B=4096\): 261.1M |
| lookup helpers (mode `helper_columns`) | one \(F_c\) element per query and per table row, committed **after** the lookup challenge: \(306U+\sum|T|\) elements \(\times\deg(F_c/F_t)\) coordinates |
| multiplicities \(m_T\) | 66 623 (one 16-bit range table + four fixed tables), committed in round 1 (challenge-independent) |

**Census source.** The counts come from the WP2 checker contracts (`verity_numerical.security.census`; Candidate
A section 1 gives the conventions, the per-width range table, the layered shape, the diff against
`verity.ml.circuits.silicon.relation.vu_census` — +212 `mul`, +33 `lin`, +134 lookups per unit — and the two unit variants).
For B-Ligero the relevant difference between the transition unit and the VU is nil (\(<0.01\%\): `step.py`
decodes and packs the word in every unit).

Committed trace-field elements, \(K=1536\), \(F_c=F_{p^3}\) (calculator): \(B=1\): 0.46M; \(B=64\): 12.4M;
\(B=4096\): **779.6M** = 418.4M witness + 0.07M multiplicities + 361.2M helper coordinates → 203 030 rows of
\(l=3840\), \(3.33\)G encoded symbols at rate \(1/4\). A-GKR commits 181.4M for the same batch (126.3M
without its committed query wires): the \(4.3\times\) (\(6.2\times\)) difference is handoff 6.2's "encoding
traffic and auxiliary lookup witness", quantified. Against the relation-based figures (404.9M vs 100.9M, 4.0×)
both candidates roughly doubled (B: \(1.9\times\), A: \(1.8\times\)) and the ratio moved from 1:4.0 to 1:4.3.
It is an explanatory variable of the census columns (section 9.4), not a verdict.

## 2. Fields

Identical to `archive/gkr/PROTOCOL.md` section 2: Goldilocks trace/code field (64-bit reference first); the quadratic
extension \(F_{p^2}\) first meets \(2^{-80}\) and \(2^{-96}\), the cubic \(F_{p^3}\) first meets \(2^{-112}\)
and \(2^{-128}\) (the logUp identity term floors at \(2^{-111.3}\)…\(2^{-100.9}\) over \(F_{p^2}\), section 9.2);
small-field study with a degree-6 extension of a 31-bit field, capped by \(\mathrm{char}\,F>N_T\) (Hab22 Lemma
5). Helper columns live in \(F_c\) and are stored as coordinates; the constraint \(h_j\cdot(z-w_j)=1\) with
\(w_j\in F_t\) expands to \(\deg\) coordinate products **[derivation]**.

## 3. Commitment interface (AHIV22 4.7 / libzk 4.2–4.3)

Same code and parameters as `archive/gkr/PROTOCOL.md` section 3: \([n=2^{14},k=2^{12}]\) RS over \(F_t\), \(l=3840\),
\(t=192\) (\(2^{-128}\) reference; minimum 191) or \(121\) (\(2^{-80}\); minimum 120), \(e=(n-k)/2\),
\(k\ge l+t\), three blinding rows per tableau, leaf \(j\) = HM96 commitment to column \(j\) with \(k=160\)
(statistical hiding \(2\cdot n\cdot2^{-160}=2^{-145}\) over both tableaux; binding by CRH) — the unsalted
\(H(\text{column})\) alternative needs a hiding assumption beyond CRH. Two tableaux (two Merkle roots) in
`helper_columns` mode; the Ligero tests run once over the vertical juxtaposition of both (the tests are linear
in rows, AHIV22 4.7 uses \(U\in L^{4m}\) the same way). Row classes:

| row class | contents | randomizer |
|---|---|---|
| witness rows (round 1) | per-unit values, linear-constraint operands | \(k-l\) random symbols per row |
| quadratic row triples (round 1) | aligned \(x,y,z\) slots (libzk `layout_quadratic_rows`) | \(k-l\) random symbols per row |
| multiplicity rows (round 1) | \(m_T\) for all tables | idem |
| helper rows (round 2) | \(h_j=1/(z_T-w_j(\beta_T))\), \(h_v=m_T(v)/(z_T-t_v(\beta_T))\), coordinates | idem |
| blinding rows (each round) | LDT row; linear-test row (payload sums to 0); quadratic-test row (zero payload) | random codewords |

## 4. Constraint system

### 4.1 Local constraints (round 1 witness)

| predicate | Ligero form | per unit |
|---|---|---|
| `lin` (decompositions, `div` identities, one-hot sums, range-chunk splits) | linear rows of \(A\) over \(F_t\) | 137 |
| `mul` (booleans, flag proofs, `q*pow`, `r*ipow`, `div*M`, selector products, `lt_pow` products) | quadratic triples \(x\circ y=z\) | 358 |
| alignment equalities | linear \(w_a-w_b=0\) | ≤ 716 (two per product) |
| range \(v<2^w\) | \(\lceil w/16\rceil\) lookups into `T_range16` (\(w<16\): \(v\cdot2^{16-w}\)) | 284 of the 306 |
| table rows (`POW`, `ALIGN`, `LEAD`, `NORM`) | lookup with 3–4 columns | 22 of the 306 |

Every field congruence used as an integer equality carries a range predicate; the widths are the WP2 contracts'
(`contract().hints` / `checked`), including the remainder ranges, the tight quotient bounds and RANGE(`M`) that
`relation.py` lacks (checker F1/F2, red-team `magW`). Over Goldilocks the `lt_pow` products do not wrap
(\(p>2^{60}\)); a 31/40-bit trace field would need the direct realisation (F1), which changes these counts.

### 4.2 Lookups, mode `helper_columns` (Hab22 §3–4) **[source]**

For table \(T\), challenge \(z_T\) (and \(\beta_T\) for vector-valued rows), queries \(w_j\) (committed round-1
values), multiplicities \(m_T\) (round 1), helpers (round 2):

| constraint | type | count per table |
|---|---|---|
| \(h_j\,(z_T-w_j(\beta_T))=1\) | quadratic (\(h_j z_T\) is linear in \(h_j\); \(h_j w_j\) needs \(\deg\) coordinate products) | \(N_T\) |
| \(h_v\,(z_T-t_v(\beta_T))=m_T(v)\) | **linear** (\(z_T,t_v\) public) | \(|T|\) |
| \(\sum_j h_j-\sum_v h_v=0\) | linear | 1 |

**Challenge space and the conditioned bound [derivation, as A-GKR 4.2].** \(z_T\) is uniform over
\(F_c\setminus\mathrm{Img}_T\), \(\mathrm{Img}_T=\{t_v(\beta_T)\}\) (public, \(\le|T|\) points), by the pre-issued
redraw slots of step 2. The cleared identity is a nonzero polynomial of degree \(\le d_T=N_T+|T|-1\) when the
multisets differ (Hab22 Lemma 5), hence \(\Pr[\text{accept}\mid\text{differ}]\le d_T/(|F_c|-|T|)\) over the
conditioned space — Hab22 Theorem 4's own sampling set (Remark 3), not a whole-field theorem reused. Whole-field
sampling would give \(d_T/|F_c|\); the difference is \(d_T|T|/(|F_c|(|F_c|-|T|))=2^{-341.3}\) for `T_range16` at
\(B=4096\), \(F_{p^3}\) (\(N_T=111.7\)M; relative \(2^{-176}\)). Vector-valued rows add \(d_T(\mathrm{cols}-1)/|F_c|\). The
constraint-satisfaction part is Ligero's. No sumcheck, no batching term outside Ligero. Prover cost: one batch
inversion per table (\(3(N_T-1)\) multiplications + 1 inversion, Hab22 §5) plus encoding of the helper rows.

### 4.3 Lookups, mode `gkr` (PH23) **[proposal]**

The fractional sumcheck of A-GKR 4.2 with **committed** query columns: leaf claims become Ligero linear
constraints directly (no claim merging), pads for the fractional-GKR transcript are committed in round 1 (their
number is public), and there is **no second commitment round**. Trades 361M helper coordinates for
\(\approx3\sum_T n_T(n_T-1)/2+4n_T\) padded messages (\(\approx4\)k \(F_c\) elements) and 1 279 sumcheck rounds
at \(B=4096\); committed elements drop to 418.5M; sequential depth rises from 3 to 381 (A-GKR 5.4 applies
verbatim, without the checker chain). Not the default: it re-introduces the padded-transcript machinery of
A-GKR.

### 4.4 Every denominator, and why none can abort on witness data

| where | denominator | status |
|---|---|---|
| helper \(h_j=1/(z_T-w_j(\beta_T))\) (prover-side division, 4.2) | \(z_T-w_j(\beta_T)\) | honest prover: \(w_j\in T\) ⇒ \(w_j(\beta_T)\in\mathrm{Img}_T\), and \(z_T\notin\mathrm{Img}_T\) by the public rule ⇒ nonzero, always. Dishonest prover with \(w_j(\beta_T)=z_T\): no \(h_j\) satisfies \(h_j\cdot0=1\); the quadratic constraint fails and Ligero's quadratic test catches it (soundness term `encoding_opening`) |
| helper \(h_v=m_T(v)/(z_T-t_v(\beta_T))\) | \(z_T-t_v(\beta_T)\) | public data only; nonzero by the same rule |
| batch inversion (Montgomery trick) | product of all \(z_T-w_j(\beta_T)\) of a table | nonzero iff every factor is (above); the prover's division-by-zero check would be witness-dependent, so the implementation **must not** branch on it: it relies on the public rule, and a zero factor (dishonest witness) yields garbage helpers that fail the constraints |
| verifier's Ligero tests: RS interpolation at \(\zeta_c\), \(\eta_j\); IRS test | Vandermonde / NTT constants over the public evaluation domain | public constants |
| \(r_{add},r_q\) as powers of \(\rho\) (section 5) | none | — |
| HM96 commitments, Merkle tree | none | — |

The only prover-side division is the helper inversion; its denominator is nonzero for every honest witness given
the public rule, and a dishonest zero is caught by the constraint system, never by a prover branch.

## 5. Numbered transcript (mode `helper_columns`)

~~~text
 0  C  [proposal, malicious-verifier ZK; section 5.3]  HM96 commitments to all verifier coins of steps 2, 4, 6
       (R = 7 slots: 5 (z_T^(1), z_T^(2), beta_T) tuples, the step-4 tuple, Q).  Without step 0:
       COMPLETE_HVZK_BACKEND at most.
 1  M  com_1 = MerkleRoot_H( HM96-leaves( Encode(W_1) ) ),  W_1 = witness rows || quadratic triples || m_T rows
       || row randomizers || blinding rows.  Fixed here: every checker value and every multiplicity;
       W_1 is challenge-independent.
 2  C  for all 5 tables in one message: z_T^(1), z_T^(2) (redraw slots, S = 2), beta_T.  Public rule:
       z_T := first slot not in Img_T = { t_v(beta_T) }; all S in Img_T -> public abort ((|T|/|F_c|)^S).
 3  M  com_2 = MerkleRoot_H( HM96-leaves( Encode(W_2) ) ),  W_2 = helper rows (h_j, h_v as F_t coordinates)
       || randomizers || blinding rows.  The only challenge-dependent auxiliary; committed, never in the clear.
 4  C  r in F_c^{rows_1+rows_2} (explicit uniform vector, 4.9 MB at B=4096);  rho_add, rho_q in F_c
       (r_add := (1, rho_add, rho_add^2, ...) over the 335.5M linear constraints, r_q likewise over the
       quadratic rows: 'powers' combination, error (#-1)/|F_c| each, section 9)
 5  M  v = r^T [W_1;W_2] + u_0 in F_c^n ;  q_add(.) deg < k+l-1 ;  p_0(.) deg < 2k-1     (AHIV22 4.7 step 2)
 6  C  Q subset [n], |Q| = t
 7  M  columns j in Q of both tableaux, HM96 openings x_j, Merkle paths to com_1 and com_2
 8  V  HM96 openings, Merkle paths; IRS test on v; linear test (sum_c q_add(zeta_c) = 0, column consistency);
       quadratic test (p_0(zeta_c) = 0, column consistency); accept/reject.
~~~

Linear constraints in step 4 include: the 137 `lin` per unit, alignment equalities, table-side helper
equations, the 5 sums, and the integer-width predicates realised as lookups; quadratic constraints: the 358
`mul` per unit and the \(N_T\) helper products.

### 5.1 Protocol rounds vs. sequential challenge depth

| quantity | value (any \(B\)) | derivation |
|---|---|---|
| `rounds.protocol` (challenge slots) | 7 (8 with step 0) | 5 per-table tuples \((z_T^{(1)},z_T^{(2)},\beta_T)\) + step 4 + step 6 |
| `rounds.sequential_depth` | **3** (4 with step 0) | step 2 (com_2 depends on it) → step 4 (responses depend on it) → step 6 (openings depend on it); the 5 table tuples are one message (independent coins); the redraw needs no trip |
| `gkr` mode | 381 at \(B=4096\) (156 at \(B=1\), 234 at \(B=64\)) | A-GKR 5.4 without the checker chain: \(1+378+2\) (`T_range16`, \(n_T=27\)) |

All three depth-1 messages are forced: com_2 must follow \(z_T\) (helpers depend on it), the Ligero responses
must follow \(r\), the openings must follow \(Q\). No known technique removes any of them without changing the
argument (a non-interactive Ligero would use Fiat–Shamir, excluded).

**RTT sweep [derivation]**, prover and verifier compute given (\(t_c\)), network floor `depth × RTT`:

| configuration | `rounds.protocol` | `rounds.sequential_depth` | `t.e2e.rtt_0ms` | `rtt_0.1ms` | `rtt_1ms` | `rtt_10ms` | `rtt_50ms` |
|---|---|---|---|---|---|---|---|
| B `helper_columns` | 7 | 3 | \(t_c\) | \(t_c+0.0003\) | \(t_c+0.003\) | \(t_c+0.03\) | \(t_c+0.15\) |
| B `helper_columns` + step 0 | 8 | 4 | \(t_c\) | \(+0.0004\) | \(+0.004\) | \(+0.04\) | \(+0.20\) |
| B `gkr` mode, \(B=4096\) | 1 291 | 381 | \(t_c\) | \(+0.038\) | \(+0.38\) | \(+3.81\) | \(+19.1\) |
| A baseline, \(B=4096\) (for comparison) | 1 589 | 381 | \(t_c\) | \(+0.038\) | \(+0.38\) | \(+3.81\) | \(+19.1\) |

Transport time for the 318 MB transcript at \(B=4096\) is not in the floor (`e2e_sweep(bandwidth_bytes_per_s=…)`
adds it when a bandwidth is given).

### 5.2 Masked values and how masks are bound

| value | mask | binding |
|---|---|---|
| every committed row (witness, triples, multiplicities, helpers) | \(k-l\) uniformly random symbols per row | any \(t\) opened columns of a row are uniform (AHIV22 Lemma 4.15; libzk `random_row(NREQ)`) |
| IRS response \(v\) | blinding row \(u_0\) (random codeword) | AHIV22 4.6.1 |
| linear response \(q_{add}\) | blinding row with payload summing to 0 | AHIV22 4.6.2 / App. B.4 |
| quadratic response \(p_0\) | blinding row with zero payload | AHIV22 4.6.3 / App. B.5 |
| helpers \(h_j\) | committed (round 2), never sent | Ligero constraints of 4.2 |
| unopened columns | HM96 leaf commitments, \(k=160\) | statistical \(2^{-145}\) over both tableaux |

No prover message other than the two roots, the three blinded test responses and the \(t\) opened columns is
ever sent; there is no transcript to pad in `helper_columns` mode.

### 5.3 Step 0: verifier-coin commitment for malicious-verifier privacy **[proposal]**

An implementation **without** this step is `COMPLETE_HVZK_BACKEND`, never `COMPLETE_ZK_BACKEND`. The
construction, binding argument, abort rules and simulator are those of A-GKR 5.5 with \(R=7\) slots;
the differences:

| item | B-Ligero |
|---|---|
| coins committed (transcript order) | slot 1–5: \((z_T^{(1)},z_T^{(2)},\beta_T)\) per table (HM96 first scheme, \(\ell=3\cdot192\) bits, \(L=4\cdot160+2\ell+4\)); slot 6: \((r,\rho_{add},\rho_q)\) — \(r\) is 203 030 \(F_c\) elements (4.9 MB), committed by the \(O(k)\) variant (commit to \(H(r\|\rho_{add}\|\rho_q)\), open with the vector and \(x_6\)); slot 7: \(Q\) (\(t\cdot14\) bits), \(O(k)\) variant |
| message order | 0: \(c_1..c_7\) → 1: com_1 → 2: \(x_1..x_5\) (P verifies all, derives \(z_T,\beta_T\)) → 3: com_2 → 4: \(x_6\) with \(r,\rho\) → 5: responses → 6: \(x_7\) with \(Q\) → 7: openings |
| prover message timing | com_2 only after all 5 openings verified; responses only after \(x_6\) verified; column openings only after \(x_7\) verified |
| abort rules | P aborts iff an HM96 opening fails, a coin is out of domain, or all \(S\) redraw slots of a table lie in \(\mathrm{Img}_T\) — public data only; no other prover abort exists (4.4: the helper inversion never branches). V aborts on a failed check. No restart within a session |
| binding suffices because | the verifier's coins for steps 2, 4, 6 are fixed before com_1 up to an \(H\)-collision; a malicious verifier keeps only the choice to stop, so its view is a prefix of an honest-verifier view on pre-fixed coins |
| simulator | A-GKR 5.5 steps (1)–(5) with the HVZK simulator of AHIV22 Lemma 4.15 for both tableaux (helper rows are simulated as random rows too; the constraints are never checked by the simulator's verifier since the opened columns are uniform); the round-2 commitment com_2 is simulated like com_1 |
| additional depth | +1 (message 0); depth 3 → 4 |
| additional communication | \(7\cdot(144+100)\) B \(=1.7\) KB (317.72 MB → 317.72 MB) |
| additional prover work | 7 HM96 opening checks: one hash over the 4.9 MB vector \(r\) and 6 hashes of \(<1.4\) kbit strings — \(<10^{-4}\) of the batch's hashing (26.6 GB) |
| soundness | \(+7\cdot2^{-160}=2^{-157.2}\); total unchanged at \(2^{-130.2}\) |

## 6. Constraint generation

The linear system \(A\) and the quadratic triples are public functions of \((K,B)\), the tables, \(z_T,\beta_T\)
(table-side helper equations) and the layout. Combination coefficients are powers of one uniform element per
test (\(\rho_{add},\rho_q\)): the verifier's step-4 message is one vector \(r\) plus two elements; the
linear/quadratic field terms of the Ligero bound become \((161.5\mathrm{M}-1)/|F_c|=2^{-164.7}\) and
\((\lceil125.0\mathrm{M}/3840\rceil-1)/|F_c|=2^{-177}\) at \(B=4096\), \(F_{p^3}\) **[derivation, Schwartz–Zippel]**;
the IRS test keeps its uniform row coefficients \(r\) (no geometric-sequence proximity statement is relied on).

## 7. Hash properties assumed, and where

Same table as `archive/gkr/PROTOCOL.md` section 7: collision resistance for both Merkle trees and the HM96 commitments
(soundness `commitment_hash` term; binding of step-0 coins); statistical hiding of HM96 leaves from CRH alone
(the unsalted alternative needs a hiding/PRG assumption on \(H\)); a PRG assumption for GPU-side generation of
row randomizers, blinding rows and HM96 randomness (ZK only; uniform at the ideal level); the statistically
hiding, computationally binding step-0 commitment is HM96 from the same \(H\). No random oracle, no correlation
intractability, no extractability of \(H\).

## 8. Leakage rules

| leak channel | rule |
|---|---|
| branch-dependent layout | every unit occupies the same slots; both arms of every branch are materialised; selectors are committed values |
| batch partitioning by exponent / group state | forbidden; one tableau pair per batch; padding instances public and prescribed; public sub-batches carry `N_subbatches` and are union-bounded |
| multiplicities, helpers | committed only; never sent |
| logUp pole | honest queries lie in \(T\), so a pole ⇔ \(z_T\in\mathrm{Img}_T\): public, resolved by the pre-issued redraw slots of step 2 and recorded deterministically; never a witness-dependent abort (calculator `completeness`: redraw \(2^{-175}\), abort \(2^{-350}\)) |
| helper inversion | never branches on a zero (4.4) |
| restart on a bad challenge | forbidden; abort the session; a new session recommits with fresh randomizers; sessions counted by the outer accounting (handoff 7.3) |
| number of opened columns / row counts | public functions of \((K,B)\) |
| helper commitment after the challenge | round 2 is a fresh randomized tableau with HM96 leaves; its root reveals nothing beyond section 7 |
| verifier-adaptive \(Q\) or \(r\) | with step 0 fixed before com_1 (5.3); without it, honest-verifier ZK (`COMPLETE_HVZK_BACKEND`) |
| timing | fixed-shape kernels; encoding and inversion work independent of values |

## 9. Soundness error decomposition (calculator output)

Soundness is a parameter: per-proof target from the application budget (`per_proof_target`), sub-batches
union-bounded (`union_over_proofs`). \(K=1536\), Goldilocks trace field, \(n=2^{14},k=2^{12},l=3840\), 256-bit
hash, \(q=2^{60}\) unless stated; `account_candidate_b`.

### 9.1 Reference decomposition, \(B=4096\)

| component | term | \(2^{-128}\) set: \(F_{p^3}\), \(t=192\), \(q=2^{60}\) | \(2^{-80}\) set: \(F_{p^2}\), \(t=121\), \(q=2^{64}\) |
|---|---|---|---|
| batching | none outside Ligero | 0 | 0 |
| sumcheck | none (`helper_columns`) | 0 | 0 |
| lookup | \(\sum_T (N_T+|T|-1)\big(\tfrac1{|F_c|-|T|}+\tfrac{\mathrm{cols}-1}{|F_c|}\big)\), conditioned challenge space; `T_range16` \(N=111.7\)M dominates | \(2^{-164.9}\) | \(2^{-100.9}\) |
| encoding/opening | \((1-e/n)^t+((k+l)/n)^t+(2k/n)^t+(n+1)/|F_c|+(\#\mathrm{lin}-1)/|F_c|+(\#\mathrm{quad\ rows}-1)/|F_c|\) (powers combination; \(\#\mathrm{lin}=335.5\)M gives \(2^{-163.7}\)) | \(2^{-130.2}\) | \(2^{-82.0}\) |
| commitment/hash | \(q^2/2^{256}\) (two roots, same bound); with step 0 also \(7\cdot2^{-160}\) | \(2^{-136}\) | \(2^{-128}\) |
| **total** | | **\(2^{-130.2}\)** (\(B\in\{1,64,4096\}\)) | **\(2^{-82.0}\)** |
| zk (not soundness) | HM96 leaves, both tableaux | \(2^{-146}\) (one tableau counted by the calculator; \(2^{-145}\) for two) | idem |
| completeness (not soundness) | redraw \(\sum_T|T|/|F_c|\); abort \(\sum_T(|T|/|F_c|)^2\) | \(2^{-176.0}\); \(2^{-352}\) | \(2^{-112.0}\); \(2^{-224}\) |

The WP2 census moved no term by more than \(2^{\pm0.2}\) (the 16-bit range table's \(N\) grew \(6\times\), the
13 other relation tables disappeared; the powers-combination linear term rose \(2^{1.1}\) to \(2^{-163.7}\) and is
still far below the Ligero query term); no operating point moved.

### 9.2 Operating points (`first_configuration`; identical to A-GKR because the dominant terms are shared)

| operating point | first configuration | total, \(B\in\{1,64,4096\}\) | why the smaller extension fails |
|---|---|---|---|
| \(2^{-80}\) (diagnostic) | **quadratic** \(F_{p^2}\), \(t=120\) | \(2^{-81.4}\) | — |
| \(2^{-96}\) (sweep) | **quadratic** \(F_{p^2}\), \(t=144\) | \(2^{-97.6}\) / \(2^{-97.6}\) / \(2^{-97.2}\) | — |
| \(2^{-112}\) (sweep) | **cubic** \(F_{p^3}\), \(t=167\) | \(2^{-113.2}\) | quadratic: lookup identity \(2^{-111.3}\)…\(2^{-100.9}\) alone exceeds the target |
| \(2^{-128}\) (primary) | **cubic** \(F_{p^3}\), \(t=191\) (reference \(t=192\)) | \(2^{-129.5}\) | quadratic: same floor |

The dominant term is Ligero's IRS query term \((1-e/n)^t=(5/8)^{192}\); the two candidates have the same
statistical budget at matched parameters and differ in committed elements (4.3×), sequential depth (3 vs 381),
and opened elements (39.0M vs 9.1M \(F_t\) at \(B=4096\)). Ligero's amortisation **[source, AHIV22 abstract,
4.7]**: the three tests and the \(t\) column openings are shared by all rows, so verifier work and proof size grow
as \(t\cdot\mathrm{rows}+O(n)\) while soundness is independent of the row count.

### 9.3 Sub-batches

As A-GKR 9.3: 16 sub-batches of 256 turn \(2^{-130.2}\) into \(2^{-126.2}\) and miss \(2^{-128}\)
(`test_sub_batches_are_union_bounded_mechanically`, which uses B-Ligero).

### 9.4 Census, \(K=1536\) (`census_estimate_b`; rows marked (est.) are estimates from the cost model of
A-GKR 9.4)

Per-unit inputs (WP2 census): 1 064 committed slots (est.), 306 helper columns in \(F_c\), 853 linear and 358
quadratic constraints. Primary \(F_c=F_{p^3}\), \(t=192\); the \(2^{-80}\) diagnostic (\(F_{p^2}\), \(t=121\)) is
the last row.

| column | B \(B=1\) | B \(B=64\) | B \(B=4096\) | A \(B=4096\) (for reference) |
|---|---|---|---|---|
| materialized witness elements (est.; committed + helpers as \(F_t\) coordinates) | 0.46M | 12.4M | 780M | 2.01G |
| committed elements (`census.committed_elements`) | 457k | 12.4M | **780M** | **181M** |
| bytes encoded (`census.bytes_encoded`) | 16 MB | 425 MB | 26.6 GB | 6.19 GB |
| hash/commit work (`census.hash_bytes`, est.) | 23 MB | 432 MB | 26.6 GB | 6.19 GB |
| field operations (`census.field_ops`, \(F_t\) mult-eq., est.: NTT + 3 mults/helper + Ligero tests) | 39M | 1.03G | 64G | 121G |
| memory passes / bytes moved (`census.memory_passes` / `census.bytes_moved`, est.) | 5 / 56 MB | 5 / 1.47 GB | 5 / 92 GB | 25 / 97 GB |
| sequential challenge depth (`rounds.sequential_depth`; `rounds.protocol`) | 3; 7 | 3; 7 | 3; 7 | 381; 1 589 |
| communication (`transcript.bytes`, est.; P→V / V→P) | 1.18 MB (1.18 / 0.003) | 6.05 MB (5.97 / 0.08) | 318 MB (313 / 4.9) | 74.7 MB (73.5 / 1.2) |
| peak memory (retained, est.) | 20 MB | 525 MB | 32.8 GB | 14.0 GB |
| \(2^{-80}\) diagnostic (\(F_{p^2}\), \(t=121\)): committed / bytes encoded / field ops / peak | 361k / 12.7 MB / 26M / 15.6 MB | 10.5M / 359 MB / 0.71G / 443 MB | 659M / 22.5 GB / 45G / 27.8 GB | 181M / 6.19 GB / 65G / 11.9 GB |

Against the relation-based table (405M at \(B=4096\)): committed elements rose \(1.9\times\) (1 064 + 306·3 per
unit instead of 512 + 172·3); the A:B ratio is 1:4.3 (1:4.0 before; 1:6.2 against A without committed query
wires). The field-operation column now favours B by \(\approx2\times\) under A's carry-everything width bound
(A-GKR 9.4). No verdict is drawn here.

## 10. Unproved obligations (verbatim list for the coordinator)

1. Knowledge soundness of the two-round variant (com_1 before, com_2 after the lookup challenge): AHIV22's
   extractor (Theorem 4.7 / 5.1) is stated for a single oracle; the extension to a challenge-dependent second
   oracle with the same query set is standard but not cited for the interactive CRH variant.
2. Simulation: honest-verifier ZK inherited from AHIV22 Lemma 4.15; malicious-verifier ZK via step 0 (5.3) is
   proposed with a simulator sketch, not proved; Goldreich–Kahan is cited, not read; the cost is estimated, not
   measured. An implementation without step 0 is `COMPLETE_HVZK_BACKEND`.
3. If unsalted Merkle leaves replace the HM96 leaves (section 3), hiding is an assumption on \(H\) beyond
   collision resistance.
4. The IRS/proximity statement for a subfield RS code over \(F_t\) with combination coefficients in \(F_c\): the
   calculator charges the Appendix C field terms over \(|F_c|\); the lemma for this setting is not cited. The
   powers-of-\(\rho\) replacement of the linear/quadratic combination (section 6) is this lane's derivation.
5. Extraction/composition with Verity's outer commitments (handoff 7.5): W_1 is a fresh commitment to the
   operands; equality with the values fixed by Verity's earlier commitments is a separate proof obligation, and
   the outer draft's extraction/composition requirements for the numerical backend remain unproved.
6. Integer semantics inside \(F_t\) (handoff 7.2): range widths for every congruence are a WP2 obligation.
7. Application-level accounting of sessions/batches (handoff 7.3) is not fixed here; `per_proof_target` applies
   once it is.
8. The layout estimates (1 064 committed elements, 853 linear constraints per unit, from the census's 3-slots-per-
   product rule) pending WP3; the census counts the checker's reference realisation (`lt_pow` as a product,
   operand decode by hints, duplicated `q*sr` products), which a layout may change.
9. The census figures of 9.4 marked (est.) come from the stated cost model, not from measurement.

## 11. Open items (questions, not decisions)

- Encoding traffic: \(3.33\)G encoded \(F_t\) symbols (26.6 GB) and 33 GB retained peak per batch of 4096 at
  rate \(1/4\); rate \(1/2\) is excluded by the quadratic test (\((2k/n)^t=1\)). Sub-batching is allowed with
  `N_subbatches` and the union bound (9.3); the split point is a measurement question.
- Helper columns are \(46\%\) of the committed elements; the `gkr` mode removes them at the price of depth 381.
- The 284 range-chunk lookups per unit (of 306) are the bulk of the helper columns; a checker realisation with
  fewer, wider range checks, or `Bf16Decode` by table (`archive/gkr/PROTOCOL.md` section 11), moves this row first.
  Which of depth and committed bytes binds for the GPU end state is decided by the RTT sweep and the
  measurements, not here.
- Whether the first complete backend ships as `COMPLETE_HVZK_BACKEND` (no step 0) with 5.3 as the path to
  `COMPLETE_ZK_BACKEND`.

## 12. The K=1536 VU as one AIR: the chain (`ChainAir`, implemented 2026-09-22) **[implemented, diagnostic PCS]**

The wave-1 diagnostic proved independent transition units (one row each, no relation between rows). The
proof unit is 96 word-chained units from \(c_0 = +0\) followed by the epilogue (`step.py::gen_vu`,
`accepts_vu`; `Epilogue` contract in `checker/gadgets.py`). This section fixes how the chain and the epilogue
enter the AIR. The relation is unchanged: every WP2 contract of the unit is emitted per row exactly as in 4.1,
the `Epilogue` contract is added, and the only new constraints are the linkage and boundary constraints below.

### 12.1 Design chosen: one row per transition unit, accumulator carried by a transition constraint

Trace: `96 * B` rows in VU order (VU \(i\) occupies rows \(96i \ldots 96i+95\)), padded to the next power of two
with the rows of the *zero VU* (\(a = b = 0\), \(c\) stays \(+0\), \(y_{16} = 0\); the padding may end inside a
VU because the last row of the trace has no transition constraint and every zero-VU row satisfies the row-local
constraints, the epilogue included). \(B = 4096\): 393,216 real rows in a \(2^{19} = 524,288\)-row trace
(25% padding; a non-power-of-two-friendly PCS would drop it).

Row-local columns beyond the unit's: five committed selectors/hints and the epilogue's columns.

| column | meaning | constraints |
|---|---|---|
| `vu.step` \(s\) | position in the VU, \(0 \ldots 95\) | `vu.first_step`: \(\mathrm{first}\cdot s = 0\); `vu.next_step` (transition): \((1-\mathrm{first}')(s' - s - 1) = 0\) |
| `vu.first` | 1 on the first row of a VU | boolean; `vu.row0_first` (first row of the trace): \(=1\); `vu.next_first` (transition): \(\mathrm{first}' = \mathrm{last}\) |
| `vu.last` | 1 on the last row of a VU | boolean; `vu.last_def`: \((s-95)\cdot\mathrm{inv} = 1 - \mathrm{last}\); `vu.last_key`: \(\mathrm{last}\cdot(s - 95) = 0\) |
| `vu.inv` | \((s-95)^{-1}\) or 0 | (the exclusion witness of `last`) |
| `vu.claim` \(\hat y\) | the claimed BF16 output word of this row's VU | `vu.out`: \(\mathrm{last}\cdot(y_{16} - \hat y) = 0\) |
| `cast.*` (epilogue) | `Epilogue` hints: `hi`, `lo`, `hi2`, `lsb`, `carry`, `rem` and their range limbs | the contract's identities: split \(y = 2^{16}\,\mathrm{hi} + \mathrm{lo}\), \(\mathrm{hi} = 2\,\mathrm{hi2} + \mathrm{lsb}\), round-to-nearest-even \(\mathrm{lo} + \mathrm{lsb} + (2^{15}-1) = 2^{16}\,\mathrm{carry} + \mathrm{rem}\), \(y_{16} = \mathrm{hi} + \mathrm{carry}\); ranges as 2-bit limbs |

Chain and boundary constraints (`ChainAir::eval`, after the row program):

- `vu.row0_step`, `vu.row0_first` (first row of the trace): \(s = 0\), \(\mathrm{first} = 1\).
- `vu.first_c[i]` (row-local): \(\mathrm{first}\cdot c_{\mathrm{in}} = 0\) -- the chain starts at \(+0\) (bit
  pattern 0; the checker's `gen_vu` starts from `c_0 = 0`).
- `vu.link[i]` (transition): \((1-\mathrm{first}')\,(c_{\mathrm{in}}' - y_{32}) = 0\) -- the next row's input
  accumulator word is this row's `pack.y` unless the next row starts a VU. Over BabyBear the 32-bit word is two
  16-bit limbs (`x.c_hi`, `x.c_lo`; `pack.y_hi`, `pack.y_lo`) and the link is two constraints.

Why the step counter: `first`/`last` alone admit a trace whose "VUs" have any length. With \(s\) pinned to 0 at a
`first` row, forced to increment inside a VU and forced to 95 exactly on a `last` row, and `first' = last`, every
VU in the trace has exactly 96 rows and the trace is a sequence of complete VUs (the padding's trailing partial
VU is never followed by a `last` row that could bind a claim: its `vu.out` is satisfied by the zero VU's own
\(y_{16}=0\) anyway).

The epilogue runs on every row (its identities hold for any finite or infinite FP32 word, and `Pack` guarantees
one), and only the `last` row's result is bound to the claim. Running it everywhere costs its columns 96 times
per VU (about 45 of 2 844 Goldilocks columns, 1.6%) and avoids a second AIR or a selector-gated degree increase;
this is the simplest sound option (12.3 discusses the alternatives).

### 12.2 Sizes

| | unit AIR (wave 1) | chain, Goldilocks | chain, BabyBear |
|---|---|---|---|
| width (columns) | 2 794 | 2 844 | 2 900 |
| constraints per row | 3 090 | 3 151 | 3 215 |
| of which range-check limbs (2-bit) / one-hot table selectors / exclusion inverses | 1 828 / 610 / 66 | 1 868 / 610 / 67 | 1 922 / 610 / 67 |
| max constraint degree (log quotient degree) | 5 (2) | 5 (2) | 5 (2) |
| rows per VU | -- | 96 | 96 |
| committed trace cells per VU (`rows x width`) | 2 794 per unit | 273 024 | 278 400 |
| committed bytes per VU (base field) | -- | **2.18 MB** | **1.11 MB** |
| trace, \(B=4096\) | \(2^{12}\) rows | \(2^{19} \times 2844\) = 1.49 G cells, 11.9 GB | \(2^{19} \times 2900\) = 1.52 G cells, 6.1 GB |

Against the census: `census_vu(1536)` counts 63.7 k slots per VU (3 073 inputs, 21 606 hints, 5 952 lookup
outputs, 23 424 products, 9 696 materialised) *with a lookup argument for the range checks and tables*; the AIR
commits 4.3x that because each range check is materialised as 2-bit limbs (1 868 of 2 844 columns) and each
table lookup as a one-hot selector row (610 columns). note:r20-proof/tensor-cost/20260922T0450Z-report-tensor's 1.15 MB/VU for B over a 31-bit field is met
by the BabyBear chain (1.11 MB/VU) only because note:r20-proof/tensor-cost/20260922T0450Z-report-tensor's figure already assumed the lookup-free 2-bit-limb
layout; a range lookup argument (4.2) would cut both fields' committed bytes by ~3x and is the next structural
step, not part of this measurement.

### 12.3 Alternatives considered

1. **Separate epilogue AIR joined by a lookup.** One row per VU in a second table (4 096 rows), the chain's
   last-row `pack.y` and the claim looked up in it. Saves the 96x-repeated epilogue columns (1.6%) at the price of
   a multi-table lookup argument uni-stark does not have; not worth it at this width.
2. **Per-VU sub-traces** (one STARK per VU, or a batch of 4 096 small traces). Prover work is the same, but each
   proof pays FRI/Merkle fixed costs and the verifier does 4 096 verifications; the single-trace chain amortises
   both. A batched variant (one commitment over 4 096 columns groups) is what the Ligero-style PCS of 3--5 does
   natively and is the proposal's path; irrelevant for the diagnostic PCS.
3. **Epilogue rows** (a 97th row per VU with a different row program). Needs a row-type selector multiplying every
   unit constraint (degree +1, so log quotient degree 3 -> blowup 8) or two AIRs; the always-on epilogue is
   cheaper.
4. **Public values** for the output words. uni-stark's `public_values` are per-proof, not per-row, and would need
   a 4 096-term boundary polynomial per column; the claim column is bound to the epilogue output by `vu.out`
   instead. This binds *committed cells*, not a statement: the verifier learns the words by reading the
   claim column's opened values, and authentication of the words (and of the operands) is excluded from this
   diagnostic exactly as in wave 1 (`authentication = excluded`).

### 12.4 Soundness notes specific to the chain

- The linkage constraint transports a *word* (`pack.y`, a 32-bit FP32 bit pattern, or its two 16-bit limbs);
  `Fp32Decode` on the next row re-derives sign/exponent/mantissa from that word under its own range checks, so
  the chain never relies on a field element being "the same number" in two encodings.
- The zero VU's rows satisfy every constraint (checked in `chain_trace`), so padding adds no constraint case.
- `vu.claim` (`vu.y16`) was a free column on non-`last` rows in the diagnostic (only `last` rows were read); since 12.5 it is a public column equal to the statement's word on every row of the VU.
- The BabyBear port splits every 32-bit word (`x.c`, `f`, `pack.low`, `pack.y`) into 16-bit limbs with their own
  range checks, realises `LtPow` as a direct range check on \(2^w - 1 - x\) (no product), and adds a bound check
  on the normalised mantissa at \(k = k_{lo}\) so no identity has a term reaching \(p = 2^{31} - 2^{27} + 1\)
  (all constants are kept below \(2^{24}\)); +56 columns over Goldilocks. The 52 `vu-k1536-neg` instances and the
  mutation tests pass identically over both fields (`tests/chain.rs`, `direct-unit vu-neg`). **Section 13 audits

### 12.5 Statement binding (SOUNDNESS FIX, lane `soundness-fixes`, 2026-09-22; red team F1–F3) **[implemented]**

The red team (`note:r20-proof/red-team-vu/20260922T1042Z-report-redteam-vu`) forged the diagnostic B verifier twice, over both fields:
**F1** a trace of two zero VUs verified "as" two real VUs (the verifier was handed no public values; the sentence
"`vu.claim` … not bound to an external statement" above was the accepted forgery), and **F2** a 256-row proof was
accepted by the verifier configured for \(B=4096\) (uni-stark 0.4.3 reads `degree_bits` from the proof). Both are
closed by `src/statement.rs` + `stark::prove_bound` / `stark::verify_bound_bytes`; the relation and the AIR are
unchanged (no new column, no new constraint, no change to `air.rs`).

**The statement.** \(\mathsf{stmt} = (B, K, \text{words}[0..B])\), the claimed 16-bit output word of every VU in
VU order; optionally the operands \((x, w)\) (`Statement::with_operands`, `vu-bench --bind-operands`). The trace
height is a *function of the statement*: \(n = 2^{\lceil \log_2 \max(96B, 2)\rceil}\); `degree_bits` \(= \log_2 n\).

**Mechanism 1 — header absorbed before any challenge (F1, F2, F3).** Prover and verifier absorb into the
Fiat–Shamir challenger, before the trace commitment is observed and before `alpha` is drawn, the header
(`statement::header`, `DOMAIN_TAG = "verity-direct/chain-air/statement/v1"`): domain tag; the AIR identity
`air_id` = SHA-256(field name, width, every column name, every constraint name in emission order); the trace
field's prime; the width; the challenge-extension degree; \(B\); \(K\); 96; \(n\); `degree_bits`; the FRI
parameters (`log_blowup`, `num_queries`, `proof_of_work_bits`); the \(B\) claimed words (length-prefixed); and
either `0` or `1 ‖ SHA-256(x) ‖ SHA-256(w)`. Every variable-length block is length-prefixed and every integer is
split into two 32-bit halves so the encoding is canonical in both fields. **The verifier builds the header from
its own configuration and statement; nothing in it is read from the proof.** Consequently a proof for another
AIR, field, extension, batch size, FRI configuration or word list yields different challenges and fails the
opening argument.

**Mechanism 2 — `degree_bits` fixed by the verifier (F2).** `verify_bound_bytes` computes `degree_bits` from
\(B\) and rejects a proof whose `degree_bits` differs (`BoundError::DegreeBits{expected, got}`) *before* anything
else is read; the PCS is configured for exactly that height (`B::config(degree_bits, ·)`), never for an upper
bound. Pinned by `tests/chain.rs::statement_binding_*` (a 2-VU proof offered to the \(B=4096\) verifier →
`DegreeBits{expected: 19, got: 8}`) and the battery case `b_degree_bits_unbound`.

**Mechanism 3 — public columns checked at \(\zeta\) (F1).** `vu.y16` is a committed column whose value on every
row the verifier knows from the statement: \(W(g^i) = \text{words}[\lfloor i/96\rfloor]\) for \(i < 96B\), \(0\)
on the zero-VU padding. \(W\) is a public polynomial of degree \(< n\) over the natural trace domain
\(\{g^i\}\). After \(\zeta\) is drawn the verifier evaluates \(W(\zeta)\) and \(W(\zeta g)\) itself
(`statement::eval_pair`, barycentric with one batch inversion, \(O(n)\) extension operations, the two points
share the inverse denominators) and requires the opened `trace_local[y16] = W(\zeta)`,
`trace_next[y16] = W(\zeta g)` (`BoundError::StatementColumn`). The FRI opening argument binds the opened values
to the committed column polynomial \(C\) of degree \(< n\); \(C \ne W\) agree at \(\zeta\) with probability
\(\le (n-1)/|F_c|\) (Schwartz–Zippel; \(\zeta\) is sampled after the trace commitment and after the header, so
the prover commits before knowing it). The existing last-row constraint `vu.out: last·(epilogue − y16) = 0` then
ties the bound word to the epilogue; the chain constraints tie the epilogue to the operands. With
`--bind-operands` the 32 operand columns `x.a[j]`, `x.b[j]` are checked the same way (\(W\) from \(x, w\)), which
removes the "any \(x, w\)" freedom entirely; without it the operands are witness and their binding is the
authentication layer's job (`note:r20-proof/auth-integration/20260922T0737Z-report-auth-integration`; `authentication = excluded`).

**Consequence for 12.4.** `vu.y16` is no longer free on non-`last` rows: it must equal the VU's word on every row
(the battery case `b_claim_free_nonlast` moved from `accept_by_design` to `reject`). Honest traces already had
this property (`chain_program` writes `io.y16` on every row).

**Soundness accounting.** Added term: (number of public columns) \(\times\, n/|F_c|\) — \(2^{19}/2^{128} =
2^{-109}\) (Goldilocks²), \(2^{19}/2^{124} = 2^{-105}\) (BabyBear⁴) for \(B = 4096\) with one column; \(\times 33\)
with operands (\(2^{-104}\) / \(2^{-100}\)). These are the same order as the DEEP/quotient-batching field term
the proof already carries (`~degree/|F_c|`, 12.2) and, like it, are field terms that the challenge extension —
not the query count — controls; they do not move the FRI query term that sets \(t\) at the \(2^{-128}\) operating
point (the accountant's field-term row for B gains this entry).

**Cost.** Prover: none (no column, no constraint; the header is ~100 + 2B field elements hashed once).
Proof: none. Verifier: one bincode decode (the verifier receives bytes), \(O(n)\) extension operations per
public column (\(\approx 2^{19}\) inversions batched + \(2^{20}\) multiplications for \(B=4096\)); measured in
`vu-bench` as `verifier.seconds` (it includes the statement checks).

**F3 (transcript absorption gaps).** In our control and done: public values (the words), `degree_bits`, the AIR
id and the FRI parameters are absorbed before the first challenge (above). Not done: uni-stark 0.4.3 / p3-fri
0.4.3 sample the batch-combination `alpha` (`two_adic_pcs.rs:330`) and the query indices without absorbing the
claimed openings at \(\zeta\) (they are transmitted in the proof, and `pcs.verify` uses them only after `alpha`);
fixing that requires patching `p3-fri`'s `open` (compute the barycentric `ys` for every matrix first, observe
them, then sample `alpha`) and `verify` (observe the claimed values before `alpha`) — the prover side cannot be
done from outside the crate without running `open` twice. Residual, as the red team argued: the claimed values
enter the reduced FRI openings through two scalars per opening point, so matching a committed fold at all
\(t=100\) query points is 100 equations in 2 unknowns; a checklist violation, not a known forgery. Tracked in
`TRANSCRIPT_CHECKLIST.md`; take upstream's fix (observe openings before `alpha`) when Plonky3 is bumped.
**Closed 2026-09-22 by lane `b-bind`** (14.7, last paragraph): `p3-fri` is vendored at `vendor/p3-fri`
(`[patch.crates-io]` in this crate and in `redteam/vu-battery-b`); `TwoAdicFriPcs::open` computes every opened
value first, absorbs them (`observe_ext_element`), then samples `alpha`; `verify` absorbs the claimed openings before
`alpha`. `alpha`, the FRI betas and the query indices now depend on the openings. **The transcript differs from
upstream 0.4.3-succinct**: proofs made before the patch do not verify after it (and vice versa); every number in
this file from 14.7 on is on the patched transcript.

**What this does not do.** The zero-VU padding rows are not part of the statement (they carry word `0`, and
\(B\) is bound, so a prover cannot smuggle a real VU into the padding without changing \(B\)). Binding the
operands to the leaf/v2 tableau (the digest in the statement, the chunk roots in the header) is the next step of
`note:r20-proof/auth-integration/20260922T0737Z-report-auth-integration`, not this section.

  this port against the red team's six embedding rules: one rule was half-implemented (fixed 2026-09-22).**

## 13. The BabyBear port against the red team's embedding rules (b-lookup, 2026-09-22)

`note:r20-proof/red-team-babybear/20260922T0605Z-report-babybear-rules` derives six rules from all 109 WP2 predicates (interval analysis,
then z3 on every embedded gadget: UNSAT under the six rules, a concrete forgery when any necessary one is dropped)
and ships a battery of 83 adversarial witnesses (`fixtures/redteam/babybear-battery/v1`). The port of section 12.4
was written before either existed. This section maps each rule to the port's constraints and records the battery
verdict of the AIR's own constraints (`direct-unit check-rows`, `tools/battery_adapter.py`).

### 13.1 Rule mapping

Column and constraint names are `program.rs` names (`tag.name`); "range" is a 2-bit-limb decomposition of the
named linear wire (`ctx.rs::rng`).

| rule | what it requires | in the port | verdict |
|---|---|---|---|
| **LIMB16** | 32-bit words as two public 16-bit limbs; word identities per limb; fraction limbs `f_lo` (16) / `f_hi` (7); `pack.low` as `low_lo`/`low_hi`; the epilogue reads the limbs | `x.c_hi`, `x.c_lo` (16-bit ranges); `acc.f_lo`/`acc.f_hi` hints with 16/7-bit ranges, identities `acc.word_lo` (\(f_{lo}=c_{lo}\)) and `acc.word_hi` (\(s\,2^{15}+t\,2^7+f_{hi}=c_{hi}\)); `pack.low_lo`/`pack.low_hi` (16/7-bit), `pack.M` (\(\mathrm{hidden}\,2^{23}+\mathrm{low}_{hi}2^{16}+\mathrm{low}_{lo}=M\)); the word selection per limb (`pack.y_hi`, `pack.y_lo` materialised); `cast.*` on the limbs directly with the contract's 16-bit ranges `cast.hi`, `cast.lo`; the link `vu.link[0..1]` is per limb | **satisfied** |
| **LT_DIRECT** (upper) | every `lt_pow` of width > 15 as `rng(bnd - 1 - x, 0, 2^B)` against the table's bound column | `Ctx::lt_pow` in `small_field`: `align_state.rem` (\(pow-1-r\)), `align_state.quot` (\(ipow-1-q\)), `align_prod.rem` (\(sr-1-r\)), `align_prod.quot` (\(isr-1-q\)), `lead.window` (\(span-1-(v-lo)\), LEAD's third column `span`), `norm.rem` (\(div-1-r\), width 7: direct too) | **satisfied** |
| **LT_DIRECT** (lower, the two-range form) | `rng(x, 0, 2^B)` in addition when `x` is **not** itself a range-checked hint -- the leading-bit window `v - lo` | `align_*`: `q`, `r` are hints with their own ranges (`align_state.q/.r`, `align_prod.q/.r`): nothing more needed. `lead.window`: **MISSING** as shipped. `v - lo` is a linear wire; for \(v<lo\) the direct difference \(span-1-(v-lo)=2^l-1-v\) is a small positive integer that passes. Forgery: \(v=16,\ l=29\) (\(2^{29}-17\in[0,2^{30})\)); at the unit level `g1.lead.l` raised by one on an honest golden instance re-derives to a different output word (\(M\) halved, \(e_1+1\): `0x3fd82158 -> 0x3fec10ac`). **Fixed**: `lead.window_lo` = `rng(v - lo, 0, 2^30)`; with \(v\le S_{MAX}\) (`sum.premise`) and \(lo\le2^{29}\) a negative \(v-lo\ge-2^{29}\) has representative \(\ge p-2^{29}>2^{30}\). +15 limb columns and +16 constraints per group (2 groups): **+30 columns, +32 constraints** (width 2 900 -> 2 930, 3 215 -> 3 247 constraints) | **was MISSING; fixed** |
| **QUOT_TIGHT** | the tight quotient bound \(q<2^B/pow\) (direct) so that \(q\,pow+r<2^{B+1}\) keeps the division identity inside \((-p,p)\) | `align_state.quot` (\(q<ipow\)) and `align_prod.quot` (\(q<isr\)) are emitted unconditionally (`align_state`, `align_product` in `program.rs`), in the direct form over BabyBear. With them: `align_state.div` has \(q\,pow+r<2^{25}+2^{25}\) vs \(2M<2^{25}\); `align_prod.div` has \(q\,sr+r<2^{17}\) vs \(u<2^{16}\): both differences inside \((-p,p)\) | **satisfied** |
| **NORM_MBND** | NORM column \(mbnd=2^{30}/div\) and `rng(mbnd - 1 - M, 0, 2^30)`: the quotient tight to the dividend, excluding \(M'=(v+p)/div\) | The port's `norm.k_lo_bound`: `rng(k_lo * M, 0, 2^23)` with `k_lo` = NORM's fourth column \([k=k_{lo}]\). Equivalent: for \(div=2^7\) (the \(k=k_{lo}=-7\) row) \(mbnd=2^{23}\) and `mbnd - 1 - M >= 0` is exactly \(M<2^{23}\); for every other \(k<0\) row \(mbnd=2^{30}/div\ge2^{24}\) is implied by `norm.M`'s 24-bit range; for \(k\ge0\) rows \(div=1\), \(r=0\) and \(M\equiv v\,2^k\) is exact because \(v\,2^k<2^{24}\) once `lead` pins \(l\) (the NORM_VBND redundancy below, which the LT_DIRECT fix restores). The shadow \(M'\ge p/128=15\,728\,640>2^{23}\) is rejected; the honest \(M=v\gg7\le S_{MAX}/128<2^{22.1}\) is accepted. Realised with one product instead of a table column (`k_lo` is linear in the one-hot selectors: degree 2, 12 limb columns) | **satisfied** (equivalent form) |
| **NORM_VBND** | \(v\,mul<2^{24}\) via a `vbnd` column -- **redundant** (z3: UNSAT without it, in isolation and composed with `leading_bit` with `l` free) | never carried by the port | nothing to drop |
| **V_PREMISE** | \(v\in[0,S_{MAX}]\), \(S_{MAX}=570\,417\,150\), as the premise of `leading_bit`/`normalise` (a guarantee of `signed_sum`'s ranges, not a check) | guaranteed: `sum.v` = \(S-2\,\mathrm{sgn}\,S\) materialised, `sum.v` 30-bit range; a wrong sign gives a negative \(v\ge-S_{MAX}\) with representative \(\ge p-S_{MAX}>2^{30}\). The port **also** checks it explicitly: `sum.premise` = `rng(S_MAX - v, 0, 2^30)` (15 limbs per group; harmless, and what the `window_lo` argument above uses) | **satisfied** (guarantee + explicit check) |

The mapping says the port implemented the rules' *first draft* faithfully -- the direct form of every wide `lt_pow`
-- and missed exactly the amendment z3 forced on the red team (`note:r20-proof/red-team-babybear/20260922T0605Z-report-babybear-rules` 6, "run 1 -> run 2": the
direct form bounds only from above). The battery does not contain that forgery (its `leading_bit -LT_DIRECT` case
\(v=16,l=0\) is the *product-form* hole, which the direct form catches); this lane's extra cases do
(`battery_adapter.py --extras`: 2 gadget-level, 4 unit-level).

### 13.2 The battery against the AIR's constraints

`direct-unit check-rows --field babybear --in jobs.json --out report.json` evaluates every constraint of
`ChainAir` (row-local, transition between consecutive supplied rows, boundary when the job starts a trace) on rows
built by imposing the case's hints by name on the witness generator (`IntCtx::overrides`; limbs, selectors,
inverses and materialised copies derived from the imposed values, the only assignment that can satisfy their own
constraints), and reports the first violated constraint. Nothing in the verdict depends on witness generation
rejecting. Case mapping (`tools/battery_adapter.py`): `step` = the forged unit at VU position 1 (so \(c\ne0\) is
allowed) followed by an honest zero unit consuming its output limbs through `vu.link`; `vu` = the two forged units
from \(c_0=0\) plus 94 honest zero units (the identity on a finite accumulator) carrying the word to row 95, where
the case's epilogue hints and the claimed \(y_{16}\) meet `vu.out`; `gadget` = `program::gadget_program`, the
gadget alone with inputs pinned to the contract's premises and outputs bound to the claimed values. Every case is
paired with an honest control (same operands, honest hints, honest word) that must be accepted.

Result (vy-cpu2, `notes-asset:campaigns/r20-proof/assets/b-air/results/battery_babybear.json`, 1.3 s for 176 jobs; the pre-fix constraints of db1f9d8 rebuilt
with the same tool: `notes-asset:campaigns/r20-proof/assets/b-air/results/battery_babybear_prefix_db1f9d8.json`):

| constraints | width / constraints | battery (83, must reject) | honest controls (87) | this lane's 6 extras (must reject) |
|---|---|---|---|---|
| port as shipped (db1f9d8, section 12.4) | 2 900 / 3 215 | **82/83**: `z3-d65d03b3ff` ACCEPTED (`lead_norm`, rule LT_DIRECT dropped: \(E=-100,\ v=0,\ l=1\) -> \(e_1=-124\), reference \(-125\); Python embedded checker: `lead.window.nonneg`) | 87/87 | **0/6**: all six accepted, incl. four unit-level forgeries with a different output word |
| with `lead.window_lo` (426700b) | 2 930 / 3 247 | **83/83** rejected | 87/87 accepted | 6/6 rejected (first failing: `lead.window_lo.limbs`) |

Where the 83 die (first failing constraint), post-fix: `wrap` 58 -- 32 on an operand decode's `s`
boolean (`a[i].s`, `b[i].s`), 18 on a quotient's own range (`g*.align_prod[i].q.limbs`, `g*.align_state.q.limbs`),
the rest on `acc.t.limbs`, `norm.clamp`, `out.z`, `sum.sgn`, `pack.hidden`; `fractional` 1 -- `g0.max.E.limbs`;
`zero_swap` 4 -- `acc.word_hi` (the sign bit of the word against the honest `s`); `hint_swap` 3 -- `a[0].word`;
`chain` 4 -- `acc.word_lo` (2: the +p shadow's fraction limb differs from the carried limb) and `g0.max.E.limbs`
(2); `epilogue` 1 -- `vu.out` (the claim binding on row 95: the mutated hints are the public limbs the port does not
carry, so the row is honest and the claimed word is wrong); `z3` 12 -- `acc.word_hi`, `cast.lsb_split`,
`align_state.rem/quot.limbs`, `align_prod.rem/quot.limbs`, `lead.window.limbs`, `lead.window_lo.limbs` (the case
above), `norm.k_lo_bound.limbs` (2: NORM_MBND's role, in the port's equivalent form), `in.v.hi.limbs` (2: the
V_PREMISE models' input is outside the premise; the premise is the composition's `sum.premise` check).

Conclusion of goal 1: five of six rules were satisfied by the port as shipped (LIMB16, LT_DIRECT-upper,
QUOT_TIGHT, NORM_MBND in an equivalent form, V_PREMISE as an explicit check); NORM_VBND was never carried (nothing
to drop); LT_DIRECT's lower range on the leading-bit window was missing and is a real forgery of the VU relation
over BabyBear (different output words on golden instances). It is fixed at +30 columns (+1.0%). The Goldilocks path
is unchanged (its `lt_pow` is the product form, which bounds both sides). Gate after the fix: section 13.3.

### 13.3 Gate after the fix

`run_vu.sh babybear 4096 166` on vy-cpu2 at 12 threads (run `r20260922-065546-a242`, `notes-asset:campaigns/r20-proof/assets/b-air/results/vu_babybear_q166_fix_r20260922-065546-a242.json`; the box was shared with a-babybear, checker-min and this lane's own lookup build, so the times carry that noise):

| | b-chain, as shipped (`r20260922-060503-6847`) | after `lead.window_lo` |
|---|---|---|
| width / constraints | 2 900 / 3 215 | 2 930 / 3 247 (+1.0%) |
| 23 unit negatives, 3 chain mutations | pass | pass |
| 52 `vu-k1536-neg` | 52 rejected by witness generation AND unsatisfiable in the field | 52 / 52 |
| battery 83 + 6 extras | 82/83, 0/6 | **83/83, 6/6**; 87/87 controls |
| 4 096 VUs, words + accumulators | all match | all match |
| witness / prove / total | 21.2 / 146.9 / 168.1 s | 22.1 / 151.1 / **173.2 s** (+3%, within the shared-box noise) |
| s/VU, overhead vs native peak | 0.0410, 4.17e9 | 0.0423, 4.30e9 |
| verify / proof / peak RSS | 0.425 s / 3.45 MB / 30.6 GB | 0.426 s / 3.47 MB / 30.9 GB |
| FRI bound (166 queries, PoW 16) | 2^-128.6 proven | 2^-128.6 proven |

Ledger: `backends/numerical/reports/ledger/b-lookup.jsonl`, label `SOUNDNESS FIX: LT_DIRECT; BabyBear AIR 2^-128, B=4096`.

## 14. Range lookups instead of 2-bit limbs: LogUp inside the AIR (b-lookup, 2026-09-22) **[implemented, diagnostic PCS]**

Section 12.2: 1 922 of the BabyBear chain's 2 930 columns are 2-bit range limbs and 610 are one-hot table selectors;
note:r20-proof/tensor-cost/20260922T0450Z-report-tensor makes committed bytes per VU the GPU lever.  This section replaces both realisations by one LogUp
argument [Haboeck 2022] inside the Plonky3 AIR.  The *relation* is unchanged: every `rng(x, lo, hi)` and every
`lookup(T, key)` of the row program (`program.rs`) is still enforced; only the realisation (`ctx.rs`, `Realisation::LogUp`)
differs.  Code: `lookup.rs` (tables, queries, helper packing, prover-side auxiliary columns, the constraints),
`lookup_stark.rs` (the three-round uni-stark), `air.rs::LookupChainAir`, `chain-layout --lookup`,
`vu-bench --lookup`, `check-rows --lookup`, `tests/lookup.rs`, `run_lookup.sh`.

### 14.1 The argument

One preprocessed matrix \(P\) (6 base columns, committed once, recomputed by the verifier) holds every table the
program needs as tagged rows \((key, o_1, .., o_4, tag)\):

| table | rows | contents |
|---|---|---|
| \(T_7, T_8, T_9\) | 128, 256, 512 | range tables \([0, 2^w)\), key only |
| \(T_{14}, T_{15}, T_{16}\) | 16 384, 32 768, 65 536 | range tables; \(T_{16}\) is the low half of every split (below), \(T_{14}/T_{15}\) the high halves of the 30/31-bit ranges |
| POW, ALIGN | 537, 538 | the saturating shift tables expanded to contiguous keys up to their slack bound (`keys[last] + slack_hi`): saturation needs no selector and no slack range any more |
| LEAD, NORM | 31, 32 | exact rows |
| padding | rest | \((0, .., 0; tag\,0)\); no query ever carries tag 0 |
| total | **116 722** | so the trace has at least \(2^{17}\) rows (the 4 096-VU trace has \(2^{19}\)) |

A **query** is a tuple \((v_0, .., v_4)\) of row expressions and a constant tag; it is combined with the second
challenge as \(q = v_0 + \beta v_1 + \dots + \beta^4 v_4 + \beta^5\, tag\), a table row likewise as \(t_r\).  Range
checks become queries as follows (`Ctx::rng`, lookup mode):

* \(n = hi - lo \le 2\): the boolean polynomial, as before.
* \(w = \lceil \log_2 n \rceil \le 16\): the query \((y, tag_w)\) with \(y = x - lo\); when \(n\) is not a power of two a
  second query \((n - 1 - y, tag_w)\) -- the intersection of the two is exactly \([0, n)\) because \(n \le 2^w\).  This
  replaces both the exclusion inverses (\(k \le 2\)) and the second limb decomposition of section 12.
* \(w > 16\): two hint columns `lo16`, `hi16` with the identity \(y = lo16 + 2^{16} hi16\) and the queries
  \((lo16, T_{16})\), \((hi16, T_{w-16})\).  Used by the 24/26/30/31-bit ranges (7 + 8 + 10 + 1 per row).

Table lookups (`Ctx::lookup`, lookup mode): the outputs are hint columns `<name>.out[c]` and the tuple
\((key, out_0, .., out_{c-1}, tag_T)\) is one query.  A saturating table's key must lie in the expanded key range
(previously `key = last + slack` with `slack in [0, slack_hi)`, now the same set as table rows).

**Queries per row: 391** (337 of degree 1, 54 of degree 2; `chain-layout --lookup --names` lists them).  The
LogUp identity over the whole trace (rows \(i\), queries \(j\), table rows \(r\)) is

\[\sum_{i,j} \frac{1}{z - q_{ij}} \;=\; \sum_r \frac{m_r}{z - t_r},\]

which holds iff every queried tuple is a table row (Haboeck Lemma 5; needs \(\mathrm{char}(F) > \) total lookups:
\(2^{27.6}\) at 4 096 VUs against \(p = 2^{30.9}\), so one proof may hold at most \(2^{22}\) rows = 43 690 VUs; larger
batches are several proofs).  Columns:

* **multiplicity** \(m\) (1 base column, last column of the main trace): the histogram of all query indices,
  computed after witness generation (`lookup::multiplicities`, a parallel reduction over the flat
  \(rows \times 391\) `u32` index array the witness generator emits alongside the rows).
* **helpers** \(h_g\) (extension elements, 4 base columns each): the queries of a row are packed greedily in program
  order into groups \(g\) of total degree \(\le 4\); \(h_g = \sum_{j \in g} 1/(z - q_j)\) with the constraint
  \(h_g \prod_{j \in g}(z - q_j) - \sum_{j \in g} \prod_{k \ne j}(z - q_k) = 0\) of degree \(1 + \sum_j \deg q_j \le 5\)
  (`MAX_DEG`, log-blowup 2 unchanged).  **112 helpers = 448 base columns**; the packing is optimal to within one
  group (\(337 + 2 \cdot 54 = 445\) degree units over 4).
* **table side** \(g = m / (z - t)\) (4 base columns): \(g (z - t_{row}) - m = 0\) on every row (\(t_{row}\) from \(P\);
  padding rows have \(m = 0\), \(g = 0\)).
* **running sum** \(S\) (4 base columns): \(S_0 = \sum_g h_g - g\) (first row), \(S_{i+1} = S_i + \sum_g h_g' - g'\)
  (transition), \(S_{last} = 0\) (last row); degrees 2.

Widths (`chain-layout --field babybear --lookup`):

| | chain, BabyBear, limbs (section 12/13) | chain, BabyBear, LogUp |
|---|---|---|
| main width | 2 930 | **401** (400 program columns + \(m\)) |
| auxiliary width (after \(z, \beta\)) | -- | **456** (112 helpers + \(g\) + \(S\), \(\times 4\)) |
| preprocessed width | -- | 6 |
| **committed width per row** | **2 930** | **863** (**3.40x fewer**) |
| constraints per row | 3 247 | 402 (286 program + 116 lookup) |
| max degree (log quotient degree) | 5 (2) | 5 (2) |
| committed cells per VU (96 rows) | 281 280 | 82 848 |
| committed bytes per VU | 1.13 MB | **0.33 MB** |
| hints / materialised | 1 066 / 14 | 386 / 14 |
| minimum trace height | 2 rows | \(2^{17}\) rows (the tables) |

The 400 program columns are the census' value slots (words, hints, named materialisations) plus 25 split hints and
66 table-output hints; nothing structural is left to remove except by changing the program itself.

### 14.2 Soundness: the challenge field (accountant, `security.accounting.logup_identity_error`)

The LogUp identity is a rational-function identity in \(z\) of degree \(N + |T| - 1\) with \(N = 2^{19} \cdot 391 =
2^{27.6}\) lookups, and the \(\beta\)-compression of 6 columns adds \(5 (N + |T| - 1)/|F|\) (Haboeck 3.4).  Over the
challenge field (`whole_field` sampling; the AIR does not redraw poles):

| challenge field | LogUp term (4 096 VUs) | in Plonky3 0.4.3 |
|---|---|---|
| BabyBear\(^4\) (\(2^{124}\)) | \(2^{-93.4}\) | yes (`BinomialExtensionField<BabyBear, 4>`) -- **the implemented configuration** |
| BabyBear\(^5\) (\(2^{155}\)) | \(2^{-124.3}\) | yes (`<BabyBear, 5>`): `BabyBear5Backend`, `vu-bench --lookup --ext 5` -- **measured** below (aux width 570; the prover's own count of the term, with its actual \(N\), is \(2^{-124.8}\)) |
| BabyBear\(^6\) (\(2^{186}\)) | \(2^{-155.2}\) | **no** (BabyBear has binomial extensions of degree 4 and 5 only in 0.4.3) |

So with the lookup argument the proof's soundness label is bounded by the challenge field: **\(2^{-93}\) at
BabyBear\(^4\)** even with 166 FRI queries (the "2^-128 proven" of section 12 referred to the FRI term alone; the
DEEP/quotient-batching field terms of uni-stark are \(\approx\) constraints \(\cdot\) degree \(/ |F_{ext}|\) and were
already \(2^{-110}\)-ish at BabyBear\(^4\), which section 12 omitted -- the lookup makes the field term the
binding one).  \(2^{-128}\) needs BabyBear\(^6\): B-Ligero's own PCS (sections 2--4) is not tied to Plonky3 and
takes any extension; in the diagnostic prover the honest option is BabyBear\(^5\) at \(2^{-124}\) or, for a full
\(2^{-128}\), a smaller batch per proof (\(2^{17}\) rows: \(2^{-126.3}\) at degree 5).  The measurement below is
labelled with the lookup term it actually has.

### 14.3 Obtaining the challenges in uni-stark

Plonky3 0.4.3's `p3_uni_stark::prove` commits one matrix and samples only \(\alpha\) (constraint batching) and
\(\zeta\) (DEEP point); it has no second trace stage and `AirBuilderWithPublicValues` only exposes base-field public
values.  `lookup_stark.rs` is therefore a fork of `prove`/`verify` (unchanged PCS, challenger, FRI parameters) with
the round structure

1. commit \(P\) (preprocessed tables; the verifier recomputes \(P\) from the public table set and compares the
   commitment: 6 columns, an LDE + Merkle tree of \(2^{19} \times 6\) at verification, see the verify time below);
2. commit the main trace (401 columns, \(m\) included -- \(m\) depends only on the witness, so it can be committed
   *before* the challenges; committing it after would let the prover pick \(m\) as a function of \(z\), which is unsound);
3. observe both, sample \(z, \beta \in F_{ext}\);
4. build and commit the auxiliary matrix (456 columns: helpers, \(g\), \(S\); `lookup::aux_matrix` -- one batch
   inversion of the 116 722 table denominators, then per row a sum over its 391 query indices, then a sequential
   prefix sum);
5. sample \(\alpha\), evaluate the quotient over (main, \(P\), aux) on the coset with a `LookupAirBuilder`
   (`ProverLookupFolder`: `PackedVal` rows for the three matrices, `PackedChallenge` for \(z, \beta\), extension
   constraints folded into the same \(\alpha\)-accumulator), commit the quotient chunks;
6. sample \(\zeta\), open all four commitments at \(\zeta, \zeta \omega\) (quotient at \(\zeta\)) in one `pcs.open`.

Cost in depth: **+2 rounds** (the preprocessed commitment could be cached across proofs; the aux round is
inherent: \(\log_2 n + 7\) sequential rounds instead of \(\log_2 n + 5\)).  Proof size: two more commitments and
two more opened-row pairs (\(6 + 456\) extension values) -- but 2 930 - 401 fewer main openings, so the proof is smaller.

### 14.4 Correctness gate

Same relation, so the section 13 battery applies unchanged: `battery_adapter.py --lookup` runs `check-rows
--lookup`, which evaluates every program constraint and the lookup **helper** constraints on the adversarial rows
(an out-of-table value leaves its group's helper without a term, which is the verifier's constraint that fails;
the global table-side and running-sum constraints need the whole trace and are covered by the round-trip
mutations of `tests/lookup.rs`).  Result: **83/83 rejected, 87/87 honest controls accepted, 6/6 lane extras
rejected** (`notes-asset:campaigns/r20-proof/assets/b-air/results/battery_babybear_lookup.json`).  `tests/lookup.rs`: layout consistency; the evaluator on an
honest VU (no failing constraint) and on mutated rows; the pod-sized (\(2^{17}\)-row) round trip -- exhaustive
evaluation of every constraint on every row of the honest committed trace (`lookup_failing_trace`), honest proof
verifies, and six mutations reject (claim word, accumulator link, \(c_0 \ne 0\), step skip, a range-checked column
pushed out of \(T_{16}\), a tampered multiplicity).  The 4 096-VU positive gate (every accumulator and output word
against bench-instances/v1) is in `vu-bench --lookup`.

### 14.5 Measurement (vy-cpu2, 12 threads, shared with a-babybear and checker-min)

All three columns: BabyBear, \(B = 4096\) VUs (\(K = 1536\), 96 units per VU, \(2^{19}\) rows of which \(393\,216\)
real), FRI `log_blowup=2`, 166 queries, PoW 16 (FRI term \(2^{-128.6}\) proven), Keccak Merkle, 12 Rayon threads on
vy-cpu2 while a-babybear and checker-min ran on the other 20 vCPUs (so every time is an upper bound), authentication
excluded, `NON_ZK_PROOF_DIAGNOSTIC`.  Every accumulator and output word checked against bench-instances/v1; all
three proofs verify.  Runs `r20260922-065546-a242` (limbs, after the section 13 fix), `r20260922-070654-177a`
(LogUp, BabyBear\(^4\)), `r20260922-071206-c4b0` (LogUp, BabyBear\(^5\)); `results/vu_*_q166_*.json`.

| | 2-bit limbs (section 12 + 13 fix) | LogUp, BabyBear\(^4\) | LogUp, BabyBear\(^5\) |
|---|---|---|---|
| main trace width (base columns) | 2 930 | 401 (+ 6 preprocessed) | 401 (+ 6) |
| committed width (base-field equivalent) | 2 930 | 863 (401 + 6 + 114 \(\cdot\) 4) | 977 (401 + 6 + 114 \(\cdot\) 5) |
| committed cells / bytes per VU | 281 280 / **1.125 MB** | 82 848 / **331 kB** (3.40x less) | 93 792 / **375 kB** (3.00x less) |
| witness s | 22.1 | 5.4 | 5.9 |
| prove s | 151.1 | 54.7 | 64.5 |
| total s | 173.2 | **60.1** | 70.4 |
| s / VU | 0.0423 | **0.0147** (2.88x) | 0.0172 (2.46x) |
| overhead vs native peak | 4.30e9 | 1.49e9 | 1.75e9 |
| verify s | 0.43 | 0.44 | 0.51 |
| proof MB | 3.47 | 2.27 | 2.37 |
| sequential depth (rounds) | 24 | 26 | 26 |
| peak RSS GB | 30.9 | 9.5 | 10.7 |
| soundness: FRI / field (lookup) term | \(2^{-128.6}\) / (\(\approx 2^{-110}\), unaccounted) | \(2^{-128.6}\) / **\(2^{-93.8}\)** | \(2^{-128.6}\) / **\(2^{-124.8}\)** |
| ledger label | `SOUNDNESS FIX: LT_DIRECT; ...` | `LogUp range lookups ... 2^-93.8` (`--breakthrough`) | -- |

Both breakthrough criteria are met: committed bytes per VU fall 3.4x (> 2x) and s/VU falls 2.9x (> 1.5x).  The
verifier is not faster: it recomputes the 6 preprocessed table columns (an LDE + Merkle tree of \(2^{19} \times 6\),
\(\approx 0.1\) s) and opens two more commitments; caching the preprocessed commitment across proofs removes that.
The \(2^{-128}\) column of section 12's headline does not survive the lookup at BabyBear\(^4\): the honest label for
the fast configuration is **\(2^{-93.8}\)**, and \(2^{-124.8}\) costs 17% more time at BabyBear\(^5\).  A full
\(2^{-128}\) with this Plonky3 version needs \(2^{17}\)-row batches (1 024 VUs per proof) at BabyBear\(^5\).

### 14.6 Where the time goes (profile)

`run_profile.sh` runs both realisations with `RUST_LOG=debug` (a `tracing_subscriber` behind the env var prints
Plonky3's `info_span!`s on close with their busy time; for the lookup prover the stage timers of `lookup_stark.rs`
are printed as `lookup stages:`).  Run `r20260922-072959-ef63`, same configuration as 14.5, summary in
`notes-asset:campaigns/r20-proof/assets/b-air/results/profile_spans_babybear_4096_r20260922-072959-ef63.txt`.  Busy seconds inside `prove` (12 threads):

| stage | 2-bit limbs (width 2 930) | LogUp (401 + 6 + 456) | what it is |
|---|---|---|---|
| quotient evaluation | **102.0** (69%) | **36.7** (68%) | every constraint on every row of the 4x LDE coset, packed: the CPU bottleneck in both |
| commit: coset LDE | 20.9 | 7.3 (3.3 main + 3.5 aux + 0.5 P, quotient) | DFT of the committed columns |
| commit: Merkle (Keccak) | 13.8 | 5.0 | hashing the LDE rows |
| open (`reduce matrix quotient`) | 7.3 | 3.0 | DEEP quotients over all committed matrices |
| FRI (commit phase, 19 folds) | 0.5 | 0.5 | |
| witness (outside `prove`) | 21.6 | 5.5 | integer forward run + hint materialisation (single pass, Rayon over VUs) |
| total prove | 148 | 54 | |

Reading:

- **Constraint evaluation, not committed bytes, is the CPU bottleneck** in both realisations: 69% / 68% of `prove`.
  With 2-bit limbs the quotient pass evaluates 3 247 constraints of degree \(\le 3\) per coset row, most of them
  the trivial `l(l-1)(l-2)(l-3)` limb constraints and the wide limb recompositions -- cheap each, but 2 930 columns
  of coset data stream through it.  With LogUp the program's constraints (402 base constraints, incl. the chain
  tail) remain and the lookup adds 112 helper constraints + the \(g\) and \(S\) constraints over \(F_{ext}\), each
  helper a degree-4 product over its 4 query terms; after the `prod_and_sum` rewrite (one extension multiplication
  per query term instead of three) the pass is 2.8x cheaper -- the same factor as the total, so on CPU the
  quotient pass scales with committed width \(\times\) constraint degree, and there is no separate encoding wall.
- Encoding (LDE + Merkle) is 23% of `prove` in both; per committed base column it costs the same (\(\approx
  12\) ms per column at \(2^{19}\) rows on 12 threads).  On the GPU (note:r20-proof/tensor-cost/20260922T0450Z-report-tensor) this is the part that becomes the whole
  story, and it fell 3.4x in bytes.  The \(2^{19}\)-row LDE of a 2 930-column trace also set the 31 GB peak RSS;
  the lookup prover peaks at 9.5 GB.
- The lookup-specific work is small: `aux_build` (batch inversion of the 116 722 table denominators, the per-row
  sums over 391 queries, the prefix sum) is 0.5 s, the preprocessed round 0.3 s.  The aux round's LDE + Merkle
  (5.8 s) is the price of 456 base columns of extension helpers; BabyBear\(^5\) adds 114 more (7.3 s) and a wider
  quotient pass (44 s), hence 70.4 s total.
- Witness generation fell 4x (21.6 s to 5.5 s): the 1 868 2-bit limb decompositions per row are no longer
  computed and written; the multiplicity histogram is one pass over the query values.

Where the next factor is: (i) the committed aux width -- the 112 helpers could be halved by sharing one helper
across two tables (a second \(\beta\)-power keeps the degree at 4) and \(g\) folded into \(S\), saving \(\approx
60\) extension columns of encoding and of quotient input; (ii) the 401 main columns are now mostly the program's own
hints (words, carries, quotients, selectors), which lookups do not touch -- that is a relation-side question
(section 12.2's per-gadget census); (iii) on the GPU only the encoding term matters and the 3.4x is the number to
carry to note:r20-proof/tensor-cost/20260922T0450Z-report-tensor.

### 14.7 Statement binding of the LogUp realisation (SOUNDNESS FIX, lane `b-bind`, 2026-09-22) **[implemented]**

Section 12.5 bound the limb realisation; the three-round prover/verifier of 14.3 (`lookup_stark.rs`) was written
in parallel and shipped with the pre-fix behaviour: `verify_lookup` took no statement, read `degree_bits` from the
proof and built the public table matrix at `1 << proof.degree_bits` -- the red team's F1 and F2 verbatim
(`redteam-logup-b` cases `zero_batch_as_any_statement`, `degree_bits_from_proof`), and the third lane's own note
("the lookup path reads degree_bits the same way and will need verify_bound"). `prove_lookup_bound` /
`verify_lookup_bound_bytes` close it; `vu-bench --lookup` uses them (`--unbound` is the labelled timing baseline,
as in 12.5). The relation, the AIR and the round structure are unchanged.

**Statement.** The same \(\mathsf{stmt}\) as 12.5 with one difference in the height: the tables live in the trace
domain (14.3 round 1), so \(n = 2^{\lceil \log_2 \max(96B,\ |P|)\rceil}\) with \(|P| = 116\,722\) table rows --
\(n = 2^{17}\) for any \(B \le 1365\), \(2^{19}\) at \(B = 4096\) (`Statement::with_min_rows`,
`lookup_stark::lookup_statement`). `degree_bits` is fixed by the verifier from it (F2); the proof's must equal it
and the verifier builds \(P\) at its own height.

**Header.** Before \(z, \beta\) (the first challenges) both parties absorb the 12.5 header (domain tag, AIR id of
the *lookup* layout -- 401 columns, `lookup.m` included, the lookup constraint names -- field, width, extension
degree, \(B, K\), `STEPS`, rows, `degree_bits`, FRI parameters, every claimed word, operand hashes) followed by a
LogUp header (`statement::lookup_header`): tag `verity-direct/chain-air/logup/v1`, preprocessed width, aux width,
the extension degree the aux columns are laid out for, and **the SHA-256 of the verifier's table set**
(`statement::table_digest`: every table's name, tag, offset, rows, lowest key and every table row's tuple). The
verifier's challenges thus depend on the table set it holds, independently of the proof; the recomputed
preprocessed commitment (14.3 round 1) is then compared as before. Both are needed for the stated property "a
prover cannot substitute a table": the commitment check catches a prover who committed to another \(P\); the
digest makes a verifier who holds another table set draw other challenges (the two parties never agree on a
transcript unless they agree on the tables).

**Public column.** `vu.y16` is a column of the main matrix (round 1, committed before any challenge); the verifier
evaluates the statement's word polynomial at \(\zeta, \zeta\omega\) (`statement::eval_pair`) and requires the
opened `main_local[col]`, `main_next[col]` to equal it, before `pcs.verify` (same soundness term as 12.5:
\(+ n / |F_{ext}|\) per public column, \(2^{19-124} = 2^{-105}\) BabyBear^4, \(2^{-136}\) BabyBear^5). With
`--bind-operands` the 32 operand columns are public too.

**Latency hooks.** `prove_lookup_bound` builds its challenger through `Backend::challenger()` (the
`RttChallenger`-wrapped one) *after* `rtt::begin`, so the header observes count as part of the first prover message
and `rounds.sequential_depth` is unchanged by the fix.

**Checks and pins.** `tests/lookup_statement.rs` (`--include-ignored`, 2^17 rows): zero batch claiming the real
words → `StatementColumn`; one word changed / swapped → `StatementColumn`; other FRI parameters → opening argument
fails; the 2-VU (17-bit) proof for the 4096-VU (19-bit) verifier → `DegreeBits{19, 17}`; the same words with the
limb realisation's 8-bit height → `DegreeBits{8, 17}` (the height is the tables', not the words'); a verifier
holding a table set with one entry changed rejects the honest proof, and a proof made against the changed table set
fails the honest verifier; operand flip with `--bind-operands` → `StatementColumn`; header sensitivity (digest,
widths, extension degree). `vu-bench --lookup` asserts the wrong-word, wrong-\(B\) and substituted-table rejections
on every run. `vu-neg --lookup` (52 negatives, `lookup_failing_rows`), the babybear-rules battery in `--lookup`
mode and the limb battery (`run_battery.sh`) are in `run_bind_gate.sh`; results in 14.8.

**The F19 residual (openings before \(\alpha\)), closed for both realisations.** 12.5 left the ζ-openings
unabsorbed before the PCS batching challenge \(\alpha\) and the FRI query indices, because the sampling sits in
`p3-fri 0.4.3`'s `TwoAdicFriPcs::{open, verify}`. This lane vendors the crate (`archive/direct/vendor/p3-fri`, a
copy of `0.4.3-succinct` with the two functions changed, each change marked `verity patch`) and points both root
manifests at it with `[patch.crates-io]`. `open` now runs two passes: the barycentric openings of every matrix at
every point are computed and absorbed (`observe_ext_element`, one element per opened value: \(2 \times (401 + 6 +
456) + 4 \times 5\) extension elements for the LogUp proof), *then* \(\alpha\) is sampled and the reduced openings
are formed; `verify` absorbs the claimed openings from the proof before sampling \(\alpha\). Since \(\alpha\) is
sampled before `prover::prove` (the FRI commit phase), the betas and the query indices depend on the openings too.
Cost: \(\approx 1\,750\) extra challenger observes per proof (µs). **The transcript changes**: proofs produced
before this patch (every `result.json` of 12.5, 14.5, the b-lookup lane) verify only against the unpatched verifier;
nothing else in the proof (bytes, shape, opened values) changes. The `RttChallenger` counts the observes as part of
the message before \(\alpha\); since the openings are a prover message the verifier's \(\alpha\) now answers,
`rounds.sequential_depth` grows by one (measured in `r20260922-094310-af1a` vs `r20260922-091130-525e`: limb 2^13 rows 16 -> 17, LogUp 2^17 rows 21 -> 22, LogUp 2^19 rows 23 -> 24; the legacy count is unchanged) -- the honest interactive count of a transcript in
which \(\alpha\) depends on the openings.

The `verify_lookup` (unbound) entry point remains for the diagnostic tests and the `--unbound` baseline; it is not a
proof verifier and its result is labelled `statement_binding.bound = false`.

### 14.8 Gate and timing of the bound LogUp realisation (vy-cpu3, 12 threads, shared with red-team-3)

Gate run `r20260922-091130-525e` (before the p3-fri patch; the same gate is repeated in the timing run below):
`cargo test --release --tests -- --include-ignored` 42 tests green (unit negatives 24, `chain.rs` 6 incl.
`statement_suite`, `statement.rs` 4, `lookup.rs` 3 incl. the 2^17 round trip, `lookup_statement.rs` 5); `vu-neg`
52/52 rejected in all four (field × realisation) modes; `vu-bench` 64 VUs bound in all four modes plus BabyBear^5
LogUp and `--bind-operands` in both realisations, the wrong-word / wrong-\(B\) / substituted-table rejections
asserted inline; `run_battery.sh` exit 0 (36 B cases 0 FORGED, A 14/14); the babybear-rules battery in `--lookup`
mode 83/83 rejected, 87/87 controls, 6/6 extras.

Run `r20260922-094310-af1a` (the p3-fri patch of 12.6 applied) repeated the whole gate -- identical counts, all green
-- and then timed the LogUp realisation at 4096 VUs, bound and unbound back to back, twice, on the same box
(`vu-bench --lookup --vus 4096 [--unbound]`, `--ext 5` for BabyBear^5; 2^19 rows, 401 main + 570 aux columns,
`RAYON_NUM_THREADS=12`):

| 4096 VUs, prove s | bound #1 | unbound #1 | bound #2 | unbound #2 |
|---|---|---|---|---|
| BabyBear^4 | 199.4 | 197.6 | 182.8 | 201.8 |
| BabyBear^5 | 213.3 | 225.4 | 217.5 | 218.1 |

Binding costs nothing measurable: the extra work is a few field elements absorbed into the challenger and one
barycentric evaluation of a public column at two points (verifier, ~ms). The pairs differ by less than the box's
own swing (up to 10% between repeats here; the brief warned of 40%). Verify 1.1 s, proof 2.37 MB, peak RSS 10.7 GB
in every configuration. vy-cpu3 is ~3.2x slower than vy-cpu2, where the unbound LogUp prover measured 60.1 s / 70.4 s
(14.5): 197-202 s / 218-225 s here for the same unbound configurations -- the ratio, not the absolute, is the
comparison. Records: `results/b-bind/lookup4096_*`.

## 15. Checker v2 as the AIR: state-chained units, tuple lookups, BabyBear limb repairs (b-air-v2, 2026-09-22) **[implemented, diagnostic PCS]**

The relation is unchanged (section 1; `checker/v2/` accepts exactly the reference's set, note:r20-proof/checker-min/20260922T0727Z-report-checker-min).  What
changes is the *program* the row evaluates: checker v2's 15-hint unit instead of v1's, on the LogUp prover of
section 14.  Design and the per-column derivation: `V2_AIR.md`; code `program_v2.rs` (gadgets and the row program),
`air_v2.rs` (layout, trace, statement-bound prove/verify, adversarial-row evaluation), `statement.rs` (F1-F3);
CLI `direct-unit <chain-layout|vu-neg|vu-honest|vu-bench|check-rows> --v2 [--ext 4|5]`; pod script `run_v2.sh`.

### 15.1 What the row is

One row per transition unit, the accumulator carried as the **state** \((s, \hat e, M)\) (three base columns) by a
transition constraint, not as an FP32 word: `next.st = (sgn_1 (1 - z_1), \hat e_1, (v2_1 2^{10} + v1_1) A1_1 + t_1)`
on every row with `next.first = 0`, and `first * st = (0, floor, 0)` on the first row of a VU.  The word exists only
on the Pack columns (`pack.y_hi`, `pack.y_lo`: two 16-bit limbs, the `LIMB16` rule of section 13) and is read once, by
the epilogue on the last row (`vu.out`).  This is where the 31-bit repairs of note:r20-proof/checker-min/20260922T0727Z-report-checker-min 3.2 (`state_decode` and
`epilogue` SAT on 31-bit fields as stated) are realised: the 33-bit word \(s 2^{31} + fld\, 2^{23} + M\) is never one
field element -- `M = M_hi8 2^16 + M_lo16` (T8, T16 lookups), `y_hi = s 2^15 + fld 2^7 + M_hi8` is an *expression*
below \(2^{16}\) on every finite state (no hint), `y_lo = M_lo16`; the infinity blend and the BF16 epilogue run per
limb.  `state_decode` does not occur in the chained VU (unit 0 starts from the constant zero state); its limb form
is in the z3 report only.

Every checker-v2 lookup is a **tuple lookup**: LogUp binds each tuple component to its table column (powers of
\(\beta\)), so the packed linear keys of the Python checker -- and the separate range checks their halves needed on
31-bit fields -- disappear (`lookup::TupleTable`, `Ctx::lookup_tuple`).  The two big tables are chunked
(the brief's option (b), exact by `floor((A+B)/2^d) = floor(A/2^d) + floor(B/2^d)` for `A` a multiple of \(2^c\), `B <
2^c`; z3-checked): SHIFT (3.5 M) becomes `SHIFT_HI`/`SHIFT_LO` on 8-bit chunks (2 x 13 824), SSHIFT becomes three
8-bit chunk tables, TNORM (1 M) becomes `TNORM10` on the low 10-bit chunk (32 768) with the high chunks multiplied
by `A1 = 2^{10+k}` from LEADNORM, LEAD becomes `LEAD10` (1 024) with the bit length as a priority combination of
three chunk lengths.  `attained` is a product identity (\(\prod_i (E - cand_i)(E - floor) = 0\), sound over \(F_p\)
because every factor is a residue of an integer in \([0, 780)\) bound by ALIGN4/LEADNORM).  Table set: 13 tables,
**282 280 rows** (v2's 5.2 M), so the trace is \(2^{19}\) rows -- which the 4 096-VU trace is anyway.

### 15.2 Sizes (`direct-unit chain-layout --v2`)

| | v1 + LogUp (section 14) | **v2 + LogUp** |
|---|---|---|
| main columns (incl. multiplicity) | 401 | **281** (176 table outputs, 101 words/state/hints/position, 4 materialised) |
| preprocessed columns | 6 | 8 (7 tuple entries + tag) |
| lookup queries per row / helpers | 391 / 112 | **105** (104 degree 1, 1 degree 2) / **27** |
| aux columns (BabyBear\(^4\) / \(^5\)) | 456 / 570 | **116 / 145** |
| committed width (\(^4\) / \(^5\)) | 863 / 977 | **405 / 453** (2.13x / 2.16x fewer) |
| program constraints | 402 | **88** (+ 27 helper, table, running sum) |
| table rows | 116 722 | 282 280 |
| max constraint degree | 5 | 5 (log quotient degree 2, `log_blowup` 2 unchanged) |

### 15.3 Correctness gate (run r20260922-084854-d988, and the local `tests/air_v2.rs`)

* `tests/air_v2.rs`: layout consistency (\(^4\) and \(^5\) share the columns and constraints); the **exhaustive
  evaluator** (every program, tail, helper, table-side and running-sum constraint on every row of the committed
  \(2^{19}\)-row trace, `air_v2::failing_trace`) on two honest VUs: 0 failures; five mutations (claim word, state
  link, a NaN operand, a multiplicity, the state exponent out of range) each fail a constraint; the pod-sized
  **statement-bound round trip**: the honest proof verifies, the same proof against a statement with one word
  flipped or with three VUs is rejected.
* `vu-neg --v2`: **52/52 `vu-k1536-neg` rejected** by witness generation and by a failing field constraint (44
  `wrong`: `vu.out` on row 95; 4 `nonfinite-operand`: the T_OP lookup helper; 4 `sat-intermediate`: `g*.finite`);
  the **44 controls** (each `wrong` negative with its correct word) accepted with no failing constraint or lookup.
* `vu-honest --v2`: all **4 096 VUs, 393 216 accumulator words** (the row's Pack word per unit, including the
  \(\pm\infty\) word at unit 95 of 20 VUs) and **4 096 BF16 words** equal bench-instances/v1; the first 64 VUs also
  evaluated row-locally with every lookup: 0 failures.  Witness 0.9 ms/VU single-threaded-equivalent.
* Battery (`battery_adapter.py --v2 --extras`, `notes-asset:campaigns/r20-proof/assets/b-air/results/battery_babybear_v2.json`): v2 has its own hint layout, so
  a case is translated as its operands + the hints with the same meaning imposed by name (`max.E`, `sum.sgn`, the
  leading bit, the epilogue's `hi2/lsb/carry/rem`, an `acc` decode as the incoming state) + the claimed word bound
  (step level: the following unit's incoming state is the claimed word's state, `vu.link[*]` checks it; VU level:
  `vu.out`).  **81/83 rejected, 87/87 honest controls accepted (with v2's honest gadget outputs), 6/6 extras
  rejected.**  The 2 accepted are `zero_swap` cases that are *vacuous under v2*: they swap the incoming word's zero
  sign under the hints of \(+0\) and claim the honest word; v2 has no word input, \(-0\) and \(+0\) denote the same
  zero state (checked: the signed-zero state \((1, floor, 0)\) also yields the same word), and the claimed word is
  the correct output.  The adapter reports them and does not count them.

### 15.4 Measurement (vy-cpu2, 12 threads, shared; \(B = 4096\), \(K = 1536\), 100 FRI queries, PoW 16)

Runs `r20260922-085312-82a9` (BabyBear\(^4\), 166 queries), `r20260922-085547-05cc` (BabyBear\(^5\), 166 queries),
`r20260922-090904-cd75` (BabyBear\(^6\), 166 queries),
`r20260922-084854-d988` / `r20260922-085143-6e87` (the same at 100 queries: 22.88 s / 25.44 s, proofs 1.18 / 1.20 MB,
FRI \(2^{-83.8}\) proven); `results/vu_v2_*.json`.  \(2^{19}\) rows (393 216 real), `log_blowup` 2, Keccak Merkle,
authentication excluded, `NON_ZK_PROOF_DIAGNOSTIC`; the pod was shared (load average \(\approx 30\) on 32 vCPUs), so
every time is an upper bound.  The v1 + LogUp columns are section 14.5's (same queries, same pod, same threads).

| | v1 + LogUp, BabyBear\(^4\) | v1 + LogUp, BabyBear\(^5\) | **v2, BabyBear\(^4\)** | **v2, BabyBear\(^5\)** | **v2, BabyBear\(^6\)** (15.5) |
|---|---|---|---|---|---|
| committed width | 863 | 977 | **405** | **453** | 463 |
| committed bytes / VU | 331 kB | 375 kB | **155.5 kB** (2.13x less) | **174 kB** (2.16x less) | 177.8 kB (1.86x less) |
| witness s | 5.4 | 5.9 | **2.37** | 2.38 | 2.39 |
| prove s | 54.7 | 64.5 | **20.55** | 23.05 | 30.41 |
| total s | 60.1 | 70.4 | **22.92** | 25.43 | 32.80 |
| s / VU | 0.0147 | 0.0172 | **0.00560** (2.62x) | 0.00621 (2.77x) | 0.00801 (1.83x) |
| overhead vs native peak | 1.49e9 | 1.75e9 | **5.68e8** | 6.31e8 | 8.13e8 |
| verify s | 0.44 | 0.51 | 0.455 | 0.557 | 0.731 |
| proof MB | 2.27 | 2.37 | 1.95 | 1.99 | 2.03 |
| sequential depth (rounds) | 26 | 26 | 26 | 26 | 26 |
| peak RSS GB | 9.5 | 10.7 | **4.85** | 5.16 | 5.47 |
| stages: commit_pre / main / aux_build / aux / quotient / commit_q / open s | 0.31 / 5.46 / 0.56 / 5.87 / 37.07 / 0.58 / 4.08 | 0.33 / 5.35 / 0.66 / 7.35 / 43.98 / 0.70 / 5.27 | 0.36 / 3.87 / 0.18 / 1.76 / **10.76** / 0.57 / 2.60 | 0.38 / 4.03 / 0.23 / 2.20 / 11.46 / 0.67 / 3.72 | 0.37 / 4.07 / 0.33 / 2.47 / 15.19 / 0.76 / 6.68 |
| soundness: FRI / lookup term | \(2^{-128.6}\) / \(2^{-93.8}\) | \(2^{-128.6}\) / \(2^{-124.8}\) | \(2^{-128.6}\) / **\(2^{-94.9}\)** | \(2^{-128.6}\) / **\(2^{-125.8}\)** | **\(2^{-128.6}\)** / \(2^{-156.7}\) |

Both breakthrough criteria are met (s/VU 2.62x better than 0.0147, > 1.5x; committed bytes/VU 2.13x under 331 kB,
> 2x).  Where the time went: the quotient pass fell 3.4x (37.1 to 10.8 s: 88 program constraints instead of 402,
27 helpers instead of 112, 105 query terms per row instead of 391 -- the \(\beta\)-compressed query term is a
degree-1 expression per tuple entry, so the quotient cost tracks queries x tuple width), the aux round 3.3x
(116 columns instead of 456), the main commit 1.4x (281 vs 401 columns), the witness 2.3x (no division hints).
The quotient pass is now 52% of `prove` (was 68%): commits (5.6 s) and the opening (2.6 s) are the next third.  The
verifier is unchanged (it recomputes 8 preprocessed columns over \(2^{19}\) rows and opens four commitments); the
proof shrinks less than the width (1.95 vs 2.27 MB) because the 166 FRI queries open \(405 \times 2\) cells per query
either way and the FRI folds dominate.

### 15.5 Soundness label

The lookup identity term (section 14.2, `security.accounting.logup_identity_error`) with this AIR's actual counts,
\(N = 2^{19} \cdot 105 = 2^{25.7}\) lookups, 282 280 table rows, 8 \(\beta\)-compressed columns (7 tuple entries +
tag), `whole_field` sampling: **\(2^{-94.9}\) at BabyBear\(^4\), \(2^{-125.8}\) at BabyBear\(^5\), \(2^{-156.7}\) at
BabyBear\(^6\)** (the prover's own count with 7 columns: 95.3 / 126.3 bits; v1 + LogUp: 93.4 / 124.3 by the same
accountant).  The FRI term at 100
queries / `log_blowup` 2 / PoW 16 is \(2^{-83.8}\) proven (\(2^{-216}\) conjectured); at 166 queries \(2^{-128.6}\).
The honest label of a v2 proof is the minimum of the two: **\(2^{-94.9}\) at BabyBear\(^4\)** and
**\(2^{-125.8}\) at BabyBear\(^5\)**, both with 166 queries (at 100 queries the FRI term \(2^{-83.8}\) binds
instead).  \(2^{-128}\) needs BabyBear\(^6\)
(not in Plonky3 0.4.3) *and* 166 queries; BabyBear\(^5\) at 166 queries gives \(2^{-125.8}\) -- the same conclusion as
section 14.2; with 3.7x fewer lookups the v2 term is 1.5 bits better at each degree, not enough to change it.
**BabyBear\(^6\) is implemented** (the brief's item 6): Plonky3 0.4.3 has binomial extensions of BabyBear of degree
4 and 5 only, so `vendor/p3-baby-bear` is the registry crate (`[patch.crates-io]`) plus `BinomiallyExtendable<6>` /
`HasTwoAdicBionmialExtension<6>` over \(X^6 - 22\): 22 is neither a square nor a cube mod \(p\) (the irreducibility
criterion for \(X^6 - w\) when \(4 \nmid 6\)), \(X^{p^k} \ne X\) for \(k = 1, 2, 3\); the multiplicative generator
\(6 + X\) has order \(p^6 - 1 = 2^{28} \cdot 3^2 \cdot 5 \cdot 7^3 \cdot 13 \cdot 31 \cdot 1171 \cdot 7459 \cdot 103237
\cdot 345769 \cdot 32472031 \cdot 3939008426310247\) (checked against every prime factor); 2-adicity 28 with the
sparse generator \(22^7 X^3\).  `tests/air_v2.rs::babybear6_field` checks the axioms, inverses, \(x^{p^6} = x\) (and
\(\ne\) for smaller powers), the orders.  `BabyBear6Backend`, `vu-bench --v2 --ext 6`: **32.80 s, 463 committed
columns, 177.8 kB/VU, label \(2^{-128.6}\)** (FRI binds; lookup \(2^{-156.7}\)) -- the first honest \(2^{-128}\) of
the B AIR at \(2^{19}\) rows, 1.83x faster and 1.86x smaller than v1 + LogUp at its \(2^{-93.8}\).  The 43% over
v2\(^4\) is the generic degree-6 extension multiplication (`BinomialExtensionField` specialises D = 2, 3 only):
quotient 15.2 vs 10.8 s, `open` 6.7 vs 2.6 s -- a Karatsuba/Toom sextic multiplication would recover most of it.

Statement binding (red-team F1-F3): `absorb_header` puts the domain tag, the AIR id (hash of field, layout and
constraint names), \(B\), \(K\), 96, `degree_bits`, the FRI parameters and every claimed word (and the operands when
`--bind-operands`) into the challenger before the first challenge; `verify_bound` fixes `degree_bits` from \(B\) and
the table set and rejects a proof of another height; the claim column is opened at \(\zeta, \zeta g\) and compared to
the verifier's own evaluation of the statement polynomial.  `vu-bench --v2` verifies the proof, then checks that
the same proof is rejected against a statement with one word flipped.

### 15.6 Where the time goes (profile, run r20260922-090107-d0d7, `run_profile.sh` with `MODES=v2`)

Busy seconds inside `prove` (12 threads, 166 queries; `results/profile_spans_v2_*.txt`), against section 14.6:

| stage | v1 + LogUp (401 + 6 + 456) | **v2 (281 + 8 + 116)** | |
|---|---|---|---|
| quotient evaluation | 36.7 (68%) | **11.2 (53%)** | 88 program + 27 helper + \(g, S\) constraints; 105 query terms per row |
| commit: coset LDE | 7.3 | **4.2** | 9 LDEs (P, main, aux, quotient chunks) |
| commit: Merkle (Keccak) | 5.0 | **3.1** | |
| open (`reduce matrix quotient`) | 3.0 | 1.7 | |
| FRI (commit phase, 19 folds) | 0.5 | 0.4 | |
| witness (outside `prove`) | 5.5 | **2.5** | |
| total prove | 54 | 21 | |

The quotient pass is still the largest item but no longer two thirds: per committed column it costs about the
same as before (\(\approx\) 27 ms per column), so the 3.3x fall is the width and the constraint count.  Encoding
(LDE + Merkle, 7.3 s) is now 35% of `prove` -- on the GPU (note:r20-proof/tensor-cost/20260922T0450Z-report-tensor) this is the term that remains, and it fell
1.7x in time and 2.1x in bytes.  The remaining fixed costs -- the preprocessed round, the 8-column table
recomputation in the verifier, the \(2^{19}\)-row floor -- are the items below.

### 15.7 Open

* **BabyBear\(^6\) multiplication**: the vendored extension uses Plonky3's generic \(D\)-loop product; the 43%
  over \(^4\) (section 15.5) is mostly that.  A specialised sextic product (Karatsuba over the cubic split, or
  Toom-3) in the vendored crate is the obvious next 20%.  The vendored crate is a patch of 0.4.3-succinct only;
  upstream Plonky3 gained more extension degrees later and the patch should be dropped when the crate is bumped.
* **Operand authentication**: `--bind-operands` absorbs the operand arrays into the header and checks the 32 operand
  columns at \(\zeta\); it was implemented and is exercised by `vu-bench --v2 --bind-operands` but not measured
  in the table (the verifier's statement polynomial then has \(32 \times 2^{19}\) coefficients: a hash-chain
  commitment to the operands is the practical form).
* **The 2 vacuous battery cases** (`zero_swap`): a v2-native red-team pass -- mutations of v2's own 15 hints (the
  chunk hints `u_hi/u_lo`, `M2/M1/M0`, `v2/v1/v0`, the Pack limbs) rather than translations of v1's -- would
  replace the translated battery.  z3 covers each gadget (7/7 UNSAT); the composed unit is covered by the 52
  negatives, the 5 mutations and the translated battery only.
* The preprocessed commitment (0.36 s) and the 8-column table recomputation in the verifier could be cached.
* Trace height is pinned at \(2^{19}\) by the table set (282 280 rows): batches below 2 731 VUs pay for padding.
  `T16` (65 536) and `T_OP` (65 536) could share a key column; `TNORM10` and `T15` too -- a \(2^{18}\) table set
  (\(\le 262\,144\) rows) would allow 2 048-VU batches at half the rows.

### 15.8 After the `lane/b-bind` merge (2026-09-22)

The v2 path binds through the shared `lookup_stark::verify_lookup(.., expected_degree_bits, public)`, which is now a
wrapper over the LogUp verifier of 14.7 (`verify_lookup_inner`: the statement's `degree_bits`, the public columns
checked at \(\zeta, \zeta g\), the preprocessed commitment recomputed and compared -- the table set is bound by that
commitment; the v1 LogUp path additionally absorbs the table digest of 14.7, the v2 header does not and the v2
transcript is unchanged by the merge in that respect).  Two things *do* change the v2 transcript relative to the
runs of 15.3-15.6: the vendored `p3-fri` (12.6) absorbs the \(\zeta\)-openings before \(\alpha\) for every
realisation, so v2 proofs made with upstream 0.4.3 no longer verify and `rounds.sequential_depth` is one larger; and
`BabyBear6Backend`'s challenger is the `RttChallenger` wrapper as before (merge kept it).  `vu-bench --v2` reports
`statement_binding.bound = true` like the other bench paths (`BenchNumbers.binding`).  Gate after the merge:
`run_bind_gate.sh all` (adds `vu-neg --v2`, `vu-bench --v2 --ext 4|6` at 64 VUs with the wrong-statement rejection
asserted inline, and the battery in `--v2` mode), run `r20260922-111343-3ed9`.

## 16. B-Ligero at the checker-v2 shape, end to end (b-e2e-v2, 2026-09-22) **[implemented, NON_ZK_PROOF_DIAGNOSTIC]**

Section 15 (lane b-air-v2) is checker v2 as a Plonky3 AIR.  This section is the *Ligero* realisation of the same
relation (`checker/v2/`: 12 hints + 8 epilogue/pack limb hints per unit, 93 lookups into 13 tables, the
`(s, e, M, z)` state chain), on the GPU, with a torch-vectorised verifier: the first end-to-end B proof at the v2
shape.  Code: `archive/direct/ligero/v2/` (`system.py` compiles the `gkr_export.v2` circuit into columns /
products / asserts / lookups / tables; `protocol.py` prover + verifier; `fc.py` \(F_c = F_p[X]/(X^6 - 31)\);
`run.py` gate + bench; `tests/test_toy.py` a CPU toy of the whole protocol).  The relation is unchanged; every
constraint is the a-v2 export's (`derive`d columns included), no predicate was added.

### 16.1 Layout (transposed against V2_SHAPE.md; same content)

The Ligero message is `m x l`: one **column per unit** (`l` = units per proof, `n_vus = l / 96` VUs, the rest pad
units = the honest zero-state unit with zero operands), one **row per column of the constraint system** across
units.  Two tableaux, both committed with the same code (rate 1/4, non-systematic domain \(g\langle\omega_n\rangle\),
so the b-ligero2 HVZK drops in unchanged):

| tableau | rows at l = 16 384 | contents |
|---|---|---|
| 1 (before \(\beta, z\)) | 631 | 273 unit columns (state in/out, operands, hints, derived wires), 33 product wires, then per table \(\lceil size/l \rceil\) **multiplicity rows** (entry \(v\) at row \(\lfloor v / l \rfloor\), position \(v \bmod l\): 4 rows T_OP, 216 SHIFT, 64 TNORM, ...) |
| 2 (after \(\beta, z\)) | 2 508 | per lookup 6 **helper rows** \(h_q = [\text{active}] / (z - w_q(\beta))\) (93 x 6 = 558), then per table 6 x rows **table-side rows** \(h_v = m_v / (z - t_v(\beta))\) (1 950) |
| public (never committed) | 1 + 13 x rows + 6 x 13 x rows | ACTIVE mask, per table the ACT mask and the coefficient rows \(z - t_v(\beta)\) as position-varying public rows (the "structured" part of the linear test: the verifier encodes them itself) |

Tables are the checker's (`_materialise` from `checker/v2/tables.py` + the range tables R4/R8/R12/R15/R16 for the
limb hints): T_OP 65 280, ALIGN4 3 120, SSHIFT_HI/LO 2 x 221 184, SHIFT 3 538 944, LEAD 32 768, LEADNORM 12 152,
TNORM 1 048 576 -- 5.2 M entries, so **the table side dominates the tableau at every l below \(2^{18}\)**
(1 950 of 3 139 rows at l = 16 384; 2 019 rows total at l = 32 768).  That is the price of realising checker-min's
packed-key tables as-is; b-air-v2's chunked SHIFT_HI/LO + SSHIFT_2/1/0 (282 k entries, section 15) would cut
tableau 2 to ~700 rows and is the first thing to port.

### 16.2 Tests (all coefficient vectors uniform in \(F_p\) per \(F_c\) coordinate; \(D = 6\) coordinates)

1. **Proximity** (IRS): \(w = r^\top W\) for \(r \in F_p^{D \times m}\), checked on the \(t\) opened columns.
2. **Linear** (37 asserts, position-independent): \(\alpha^\top W = \beta\) on the opened columns, \(\alpha\) the
   \(\rho\)-combination of the assert rows, \(\beta\) the combined constant terms (pads are honest units, so the
   asserts hold on every column and no mask is needed).
3. **Quadratic** (591 constraints = 33 products + 93 x 6 helper coordinates + 13 tables x 6 x rows table-side
   coordinates): \(\sum_q \rho_q\,(A_q(x) B_q(x) - C_q(x)) = h(x) Z_H(x)\) with the helper constraint
   \(h_q \cdot (z - w_q(\beta)) = \text{ACTIVE}\) written per \(F_c\) coordinate as 6 quadratic constraints over
   \(F_p\) (the \(F_c\) product is a bilinear form in the 6 x 6 coordinates: `Constraints.quadratic_terms` streams
   the coordinate products in 96-row chunks), and the table-side constraint \(h_v \cdot (z - t_v(\beta)) = m_v\)
   with the public coefficient row.  \(h\) of degree \(< l - 1\) is sent (\(D \times (l - 1)\) elements).
4. **Sums** (LogUp identity, 6 per table): \(\sum_{\text{helpers of } T} h - \sum_{\text{table side of } T} h_v = 0\)
   per coordinate: the prover sends \(v = s^\top W\) (\(s\) the \(\pm\rho\) pattern), the verifier checks
   \(\sum_j v_j = 0\) and \(v\) against the opened columns.
5. **Chain** (position-dependent, section 12's general linear test): \(x_*[j+1] = y_*[j]\) inside a VU, \(x_* =\)
   zero state at VU starts and pads, `ovf = 0` on non-last units, \(y16[\text{last}] = \) the claimed word; one
   \(q\) of degree \(< 2l - 1\).
6. **Openings**: \(t\) columns of both tableaux with Blake3 Merkle paths (GPU Blake3 from `verity/ml/kernels/hash_gpu`,
   bit-exact with the host verifier's).

Statement binding (F1-F3): the digest absorbs the domain tag, the checker params, \((l, n, t, \text{steps}, n_{vus})\),
the target, the system id (SHA-256 of the compiled constraint system's description) and the claimed words; every
size the verifier uses comes from the config, never from the proof.  Public pole rule: \(z\) is drawn from two
slots; a slot in the table image (or the zero tuple) is skipped, both in the image aborts publicly.

### 16.3 Soundness (from `security.accounting`, code constants)

Per proof at l = 16 384, n = 65 536: t = 189 gives \(2^{-128.15}\) (IRS query term \(2^{-128.1}\) binding; linear /
quadratic query terms \(2^{-189}\); field terms \(2^{-169}\) to \(2^{-185}\); logUp identity terms per table
\(2^{-162.6}\) (SHIFT) to \(2^{-171.4}\), Haboeck Thm 4 with the conditioned challenge space).  A batch of 4 096 VUs
is \(N\) sub-batch proofs; the bench targets \(2^{-(128 + \log_2 N)}\) per proof (t = 194 at N = 25) so the union
bound over the batch meets \(2^{-128}\).  Field: BabyBear\(^6\); hash model as in `accounting` (2^64 queries: the
collision term alone is \(2^{-128}\), see note:r20-proof/red-team-babybear/20260922T0605Z-report-babybear-rules section 8).

### 16.4 What was verified (gate, vy-g2, run r20260922-091001-68f7) and what the label means

Honest `vu-k1536` VUs accepted with the recorded words, every intermediate packed word equal to the recorded
accumulator (the prover refuses to repair); the prover's own self-check finds every constraint satisfied on the
honest witness (so the a-v2 circuit + the limb pack/epilogue rows are *complete* on the bench instances).
`vu-k1536-neg` 52/52 rejected (44 by the linear test on `y16 = claimed`, 8 `reject`-class operands: no witness
exists, the verifier rejects a foreign proof on the statement digest); 39/39 unit + chain mutations rejected (every
hint +1, every epilogue limb hint +1, every state link, one lookup output per table, operands set to Inf/NaN,
hint swap, false overflow flag, chain relink); the BabyBear battery's `epilogue` class (RN-even tie hints
toggled) 2/2 rejected; the battery's `chain` class (word handed on as c + p) is **vacuous** under v2 -- the state
is chained as `(s, e, M, z)` elements, never as a 33-bit word, so `M + p` is the same committed element
(b-air-v2's battery adapter reaches the same verdict).  The `wrap`/`z3` classes of the battery are stated in the
WP2 (v1) hint layout and were not mapped onto the v2 columns here (open).

Label **NON_ZK_PROOF_DIAGNOSTIC**, with the **INCOMPLETE_CHECKER_DIAGNOSTIC** caveat: the BabyBear word-class
repair rows (`pack.M_hi8/M_lo16`, the `y_hi` expression, the epilogue on limbs) are a Python port of b-air-v2's
design, whose *Rust* realisation is z3-UNSAT (`notes-asset:campaigns/r20-proof/assets/b-air-v2/reports/b_air_v2_z3.json`: `pack_epilogue_limbs`, `state_decode_limbs`,
Z and \(F_{\text{babybear}}\)); the Python column set here was not itself put through z3.  No ZK (systematic-free
domain but no blinding rows yet).
