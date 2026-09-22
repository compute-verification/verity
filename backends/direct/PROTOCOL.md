# Candidate B — direct local constraints in an interactive hash-based ZK Ligero argument

Status: **specification / proposed parameterisation**. Nothing here is implemented or measured. The argument is
AHIV22's interactive ZKIPCP-plus-CRH construction applied to the census constraints of one 16-product update,
with logUp lookups (Hab22) as a second commitment round. Same tags as `backends/gkr/PROTOCOL.md`: **[source]**,
**[derivation]**, **[proposal]**, **[assumption]**.

**Label rule** (`backends/AGENTS.md`). A plain Plonky3 AIR/STARK over the same columns is
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
| HM96 | Halevi–Micali, CRYPTO 1996 (sections 1.2, 3, 4 read; see Candidate A section 0) | statistically hiding, computationally binding commitment from CRH: HM96 leaves (section 3) and step-0 coin commitments (5.3) |

Not read (cited by name only): Goldreich–Kahan 1996 (verifier-coin commitment, simulator); the simulator sketch
in 5.3 is written out.

## 1. Relation, shapes, public data

Same public data as Candidate A section 1 (profile, \(K\), \(B\), \(U=BK/16\), tables, census). Differences:

| item | Candidate B |
|---|---|
| committed per unit | **every** value of the checker: the 34 words, the 225 hints, the 62 looked-up row columns, the 244 products and the 100 materialised range expressions (665 columns), plus Ligero alignment copies for quadratic operands |
| constraints per unit **[derivation from the WP2 contracts, Candidate A section 1]** | 137 linear (115 identities + 22 range-chunk splits), 358 quadratic (`mul`, 114 of them booleans), 306 lookups into 5 public tables (284 into the one 16-bit range table, 22 into `ALIGN`/`POW`/`LEAD`/`NORM`) |
| alignment copies **[source, libzk 4.3]** | Ligero's quadratic test needs \(x,y,z\) of each product in the same column of three rows: \(3\cdot358=1\,074\) quadratic slots per unit; values reused across products are copied and bound by linear equalities |
| committed elements per unit **[estimate]** | 1 064 = \(3\cdot358\) slots + 104 columns that are neither a direct operand nor a product (`census.columns_committed_b`, `CandidateB.witness_per_unit`); the layout itself is WP3's |
| linear constraints per unit **[estimate]** | \(137+2\cdot358=853\) (census `lin` incl. splits, two copy equalities per product) — `CandidateB.linear_per_unit`; total at \(B=4096\): 335.5M incl. table-side helper equations |
| quadratic constraints per unit | 358 `mul` + 306 helper products = 664; total at \(B=4096\): 261.1M |
| lookup helpers (mode `helper_columns`) | one \(F_c\) element per query and per table row, committed **after** the lookup challenge: \(306U+\sum|T|\) elements \(\times\deg(F_c/F_t)\) coordinates |
| multiplicities \(m_T\) | 66 623 (one 16-bit range table + four fixed tables), committed in round 1 (challenge-independent) |

**Census source.** The counts come from the WP2 checker contracts (`verity_numerical.security.census`; Candidate
A section 1 gives the conventions, the per-width range table, the layered shape, the diff against
`verity.ml.tc.relation.vu_census` — +212 `mul`, +33 `lin`, +134 lookups per unit — and the two unit variants).
For Candidate B the relevant difference between the transition unit and the VU is nil (\(<0.01\%\): `step.py`
decodes and packs the word in every unit).

Committed trace-field elements, \(K=1536\), \(F_c=F_{p^3}\) (calculator): \(B=1\): 0.46M; \(B=64\): 12.4M;
\(B=4096\): **779.6M** = 418.4M witness + 0.07M multiplicities + 361.2M helper coordinates → 203 030 rows of
\(l=3840\), \(3.33\)G encoded symbols at rate \(1/4\). Candidate A commits 181.4M for the same batch (126.3M
without its committed query wires): the \(4.3\times\) (\(6.2\times\)) difference is handoff 6.2's "encoding
traffic and auxiliary lookup witness", quantified. Against the relation-based figures (404.9M vs 100.9M, 4.0×)
both candidates roughly doubled (B: \(1.9\times\), A: \(1.8\times\)) and the ratio moved from 1:4.0 to 1:4.3.
It is an explanatory variable of the census columns (section 9.4), not a verdict.

## 2. Fields

Identical to Candidate A section 2: Goldilocks trace/code field (64-bit reference first); the quadratic
extension \(F_{p^2}\) first meets \(2^{-80}\) and \(2^{-96}\), the cubic \(F_{p^3}\) first meets \(2^{-112}\)
and \(2^{-128}\) (the logUp identity term floors at \(2^{-111.3}\)…\(2^{-100.9}\) over \(F_{p^2}\), section 9.2);
small-field study with a degree-6 extension of a 31-bit field, capped by \(\mathrm{char}\,F>N_T\) (Hab22 Lemma
5). Helper columns live in \(F_c\) and are stored as coordinates; the constraint \(h_j\cdot(z-w_j)=1\) with
\(w_j\in F_t\) expands to \(\deg\) coordinate products **[derivation]**.

## 3. Commitment interface (AHIV22 4.7 / libzk 4.2–4.3)

Same code and parameters as Candidate A section 3: \([n=2^{14},k=2^{12}]\) RS over \(F_t\), \(l=3840\),
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

**Challenge space and the conditioned bound [derivation, as Candidate A 4.2].** \(z_T\) is uniform over
\(F_c\setminus\mathrm{Img}_T\), \(\mathrm{Img}_T=\{t_v(\beta_T)\}\) (public, \(\le|T|\) points), by the pre-issued
redraw slots of step 2. The cleared identity is a nonzero polynomial of degree \(\le d_T=N_T+|T|-1\) when the
multisets differ (Hab22 Lemma 5), hence \(\Pr[\text{accept}\mid\text{differ}]\le d_T/(|F_c|-|T|)\) over the
conditioned space — Hab22 Theorem 4's own sampling set (Remark 3), not a whole-field theorem reused. Whole-field
sampling would give \(d_T/|F_c|\); the difference is \(d_T|T|/(|F_c|(|F_c|-|T|))=2^{-341.3}\) for `T_range16` at
\(B=4096\), \(F_{p^3}\) (\(N_T=111.7\)M; relative \(2^{-176}\)). Vector-valued rows add \(d_T(\mathrm{cols}-1)/|F_c|\). The
constraint-satisfaction part is Ligero's. No sumcheck, no batching term outside Ligero. Prover cost: one batch
inversion per table (\(3(N_T-1)\) multiplications + 1 inversion, Hab22 §5) plus encoding of the helper rows.

### 4.3 Lookups, mode `gkr` (PH23) **[proposal]**

The fractional sumcheck of Candidate A 4.2 with **committed** query columns: leaf claims become Ligero linear
constraints directly (no claim merging), pads for the fractional-GKR transcript are committed in round 1 (their
number is public), and there is **no second commitment round**. Trades 361M helper coordinates for
\(\approx3\sum_T n_T(n_T-1)/2+4n_T\) padded messages (\(\approx4\)k \(F_c\) elements) and 1 279 sumcheck rounds
at \(B=4096\); committed elements drop to 418.5M; sequential depth rises from 3 to 381 (Candidate A 5.4 applies
verbatim, without the checker chain). Not the default: it re-introduces the padded-transcript machinery of
Candidate A.

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
| `gkr` mode | 381 at \(B=4096\) (156 at \(B=1\), 234 at \(B=64\)) | Candidate A 5.4 without the checker chain: \(1+378+2\) (`T_range16`, \(n_T=27\)) |

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
construction, binding argument, abort rules and simulator are those of Candidate A 5.5 with \(R=7\) slots;
the differences:

| item | Candidate B |
|---|---|
| coins committed (transcript order) | slot 1–5: \((z_T^{(1)},z_T^{(2)},\beta_T)\) per table (HM96 first scheme, \(\ell=3\cdot192\) bits, \(L=4\cdot160+2\ell+4\)); slot 6: \((r,\rho_{add},\rho_q)\) — \(r\) is 203 030 \(F_c\) elements (4.9 MB), committed by the \(O(k)\) variant (commit to \(H(r\|\rho_{add}\|\rho_q)\), open with the vector and \(x_6\)); slot 7: \(Q\) (\(t\cdot14\) bits), \(O(k)\) variant |
| message order | 0: \(c_1..c_7\) → 1: com_1 → 2: \(x_1..x_5\) (P verifies all, derives \(z_T,\beta_T\)) → 3: com_2 → 4: \(x_6\) with \(r,\rho\) → 5: responses → 6: \(x_7\) with \(Q\) → 7: openings |
| prover message timing | com_2 only after all 5 openings verified; responses only after \(x_6\) verified; column openings only after \(x_7\) verified |
| abort rules | P aborts iff an HM96 opening fails, a coin is out of domain, or all \(S\) redraw slots of a table lie in \(\mathrm{Img}_T\) — public data only; no other prover abort exists (4.4: the helper inversion never branches). V aborts on a failed check. No restart within a session |
| binding suffices because | the verifier's coins for steps 2, 4, 6 are fixed before com_1 up to an \(H\)-collision; a malicious verifier keeps only the choice to stop, so its view is a prefix of an honest-verifier view on pre-fixed coins |
| simulator | Candidate A 5.5 steps (1)–(5) with the HVZK simulator of AHIV22 Lemma 4.15 for both tableaux (helper rows are simulated as random rows too; the constraints are never checked by the simulator's verifier since the opened columns are uniform); the round-2 commitment com_2 is simulated like com_1 |
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

Same table as Candidate A section 7: collision resistance for both Merkle trees and the HM96 commitments
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

### 9.2 Operating points (`first_configuration`; identical to Candidate A because the dominant terms are shared)

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

As Candidate A 9.3: 16 sub-batches of 256 turn \(2^{-130.2}\) into \(2^{-126.2}\) and miss \(2^{-128}\)
(`test_sub_batches_are_union_bounded_mechanically`, which uses Candidate B).

### 9.4 Census, \(K=1536\) (`census_estimate_b`; rows marked (est.) are estimates from the cost model of
Candidate A 9.4)

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
(Candidate A 9.4). No verdict is drawn here.

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
  fewer, wider range checks, or `Bf16Decode` by table (Candidate A section 11), moves this row first.
  Which of depth and committed bytes binds for the GPU end state is decided by the RTT sweep and the
  measurements, not here.
- Whether the first complete backend ships as `COMPLETE_HVZK_BACKEND` (no step 0) with 5.3 as the path to
  `COMPLETE_ZK_BACKEND`.

## 12. The K=1536 VU as one AIR: the chain (`ChainAir`, implemented 2026-09-22) **[implemented, diagnostic PCS]**

The wave-1 diagnostic proved independent transition units (one row each, no relation between rows). The
verification unit is 96 word-chained units from \(c_0 = +0\) followed by the epilogue (`step.py::gen_vu`,
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
table lookup as a one-hot selector row (610 columns). tensor.md's 1.15 MB/VU for B over a 31-bit field is met
by the BabyBear chain (1.11 MB/VU) only because tensor.md's figure already assumed the lookup-free 2-bit-limb
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
  mutation tests pass identically over both fields (`tests/chain.rs`, `direct-unit vu-neg`).

### 12.5 Statement binding (SOUNDNESS FIX, lane `soundness-fixes`, 2026-09-22; red team F1–F3) **[implemented]**

The red team (`backends/numerical/reports/redteam_vu.md`) forged the diagnostic B verifier twice, over both fields:
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
authentication layer's job (`auth_integration.md`; `authentication = excluded`).

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

**What this does not do.** The zero-VU padding rows are not part of the statement (they carry word `0`, and
\(B\) is bound, so a prover cannot smuggle a real VU into the padding without changing \(B\)). Binding the
operands to the leaf/v2 tableau (the digest in the statement, the chunk roots in the header) is the next step of
`auth_integration.md`, not this section.
