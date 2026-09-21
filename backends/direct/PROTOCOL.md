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
| committed per unit | **every** value of the checker: operands (decoded), state in/out, products, quotients/remainders, leading-bit fields, flags, plus Ligero alignment copies for quadratic operands |
| constraints per unit **[source, census]** | 104 linear, 146 quadratic (`mul`), 172 lookups into 14 public tables (range checks are lookups into `T_range*`) |
| alignment copies **[source, libzk 4.3]** | Ligero's quadratic test needs \(x,y,z\) of each product in the same column of three rows: \(3\cdot146=438\) quadratic slots per unit, values reused across products are copied and bound by linear equalities |
| committed elements per unit **[assumption]** | 512 (438 slots + ≤74 values occurring only in linear/lookup predicates) until the WP3 layout lands (`CandidateB.witness_per_unit`) |
| linear constraints per unit **[assumption]** | \(104+292+14=410\) (census `lin`, alignment equalities, per-table sums) — `CandidateB.linear_per_unit`; total at \(B=4096\): 161.5M incl. table-side helper equations |
| quadratic constraints per unit | 146 `mul` + 172 helper products = 318; total at \(B=4096\): 125.0M |
| lookup helpers (mode `helper_columns`) | one \(F_c\) element per query and per table row, committed **after** the lookup challenge: \(172U+\sum|T|\) elements \(\times\deg(F_c/F_t)\) coordinates |
| multiplicities \(m_T\) | committed in round 1 (challenge-independent) |

Committed trace-field elements, \(K=1536\), \(F_c=F_{p^3}\) (calculator): \(B=1\): 0.75M; \(B=64\): 7.0M;
\(B=4096\): **404.9M** = 201.3M witness + 0.16M multiplicities + 203.4M helper coordinates → 105 440 rows of
\(l=3840\), \(1.73\)G encoded symbols at rate \(1/4\). Candidate A commits 100.9M for the same batch: the
\(4.0\times\) difference is handoff 6.2's "encoding traffic and auxiliary lookup witness", quantified. It is an
explanatory variable of the census columns (section 9.4), not a verdict.

## 2. Fields

Identical to Candidate A section 2: Goldilocks trace/code field (64-bit reference first); the quadratic
extension \(F_{p^2}\) first meets \(2^{-80}\) and \(2^{-96}\), the cubic \(F_{p^3}\) first meets \(2^{-112}\)
and \(2^{-128}\) (the logUp identity term floors at \(2^{-109.4}\)…\(2^{-100.8}\) over \(F_{p^2}\), section 9.2);
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
| `lin` (integer linear relations, reconstruction widths, carries) | linear rows of \(A\) over \(F_t\) | 104 |
| `mul` (products, quotient·divisor, selector products) | quadratic triples \(x\circ y=z\) | 146 |
| alignment equalities | linear \(w_a-w_b=0\) | ≤ 292 |
| range \(v<2^r\) | lookup into `T_range r` (below) | in the 172 |

Every field congruence used as an integer equality carries a range predicate; the width table is a WP2
obligation (handoff 7.2), not established here.

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
sampling would give \(d_T/|F_c|\); the difference is \(d_T|T|/(|F_c|(|F_c|-|T|))=2^{-343.9}\) for `T_range16` at
\(B=4096\), \(F_{p^3}\) (relative \(2^{-176}\)). Vector-valued rows add \(d_T(\mathrm{cols}-1)/|F_c|\). The
constraint-satisfaction part is Ligero's. No sumcheck, no batching term outside Ligero. Prover cost: one batch
inversion per table (\(3(N_T-1)\) multiplications + 1 inversion, Hab22 §5) plus encoding of the helper rows.

### 4.3 Lookups, mode `gkr` (PH23) **[proposal]**

The fractional sumcheck of Candidate A 4.2 with **committed** query columns: leaf claims become Ligero linear
constraints directly (no claim merging), pads for the fractional-GKR transcript are committed in round 1 (their
number is public), and there is **no second commitment round**. Trades 203M helper coordinates for
\(\approx3\sum_T n_T(n_T-1)/2+4n_T\) padded messages (\(\approx11\)k \(F_c\) elements) and 3 417 sumcheck rounds
at \(B=4096\); committed elements drop to 201.5M; sequential depth rises from 3 to 328 (Candidate A 5.4 applies
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
       (R = 16 slots: 14 (z_T^(1), z_T^(2), beta_T) tuples, the step-4 tuple, Q).  Without step 0:
       COMPLETE_HVZK_BACKEND at most.
 1  M  com_1 = MerkleRoot_H( HM96-leaves( Encode(W_1) ) ),  W_1 = witness rows || quadratic triples || m_T rows
       || row randomizers || blinding rows.  Fixed here: every checker value and every multiplicity;
       W_1 is challenge-independent.
 2  C  for all 14 tables in one message: z_T^(1), z_T^(2) (redraw slots, S = 2), beta_T.  Public rule:
       z_T := first slot not in Img_T = { t_v(beta_T) }; all S in Img_T -> public abort ((|T|/|F_c|)^S).
 3  M  com_2 = MerkleRoot_H( HM96-leaves( Encode(W_2) ) ),  W_2 = helper rows (h_j, h_v as F_t coordinates)
       || randomizers || blinding rows.  The only challenge-dependent auxiliary; committed, never in the clear.
 4  C  r in F_c^{rows_1+rows_2} (explicit uniform vector, 2.5 MB at B=4096);  rho_add, rho_q in F_c
       (r_add := (1, rho_add, rho_add^2, ...) over the 161.5M linear constraints, r_q likewise over the
       quadratic rows: 'powers' combination, error (#-1)/|F_c| each, section 9)
 5  M  v = r^T [W_1;W_2] + u_0 in F_c^n ;  q_add(.) deg < k+l-1 ;  p_0(.) deg < 2k-1     (AHIV22 4.7 step 2)
 6  C  Q subset [n], |Q| = t
 7  M  columns j in Q of both tableaux, HM96 openings x_j, Merkle paths to com_1 and com_2
 8  V  HM96 openings, Merkle paths; IRS test on v; linear test (sum_c q_add(zeta_c) = 0, column consistency);
       quadratic test (p_0(zeta_c) = 0, column consistency); accept/reject.
~~~

Linear constraints in step 4 include: the 104 `lin` per unit, alignment equalities, table-side helper
equations, the 14 sums, and the integer-width predicates realised as lookups; quadratic constraints: the 146
`mul` per unit and the \(N_T\) helper products.

### 5.1 Protocol rounds vs. sequential challenge depth

| quantity | value (any \(B\)) | derivation |
|---|---|---|
| `rounds.protocol` (challenge slots) | 16 (17 with step 0) | 14 per-table tuples \((z_T^{(1)},z_T^{(2)},\beta_T)\) + step 4 + step 6 |
| `rounds.sequential_depth` | **3** (4 with step 0) | step 2 (com_2 depends on it) → step 4 (responses depend on it) → step 6 (openings depend on it); the 14 table tuples are one message (independent coins); the redraw needs no trip |
| `gkr` mode | 328 at \(B=4096\) (156 at \(B=1\), 193 at \(B=64\)) | Candidate A 5.4 without the checker chain: \(1+325+2\) |

All three depth-1 messages are forced: com_2 must follow \(z_T\) (helpers depend on it), the Ligero responses
must follow \(r\), the openings must follow \(Q\). No known technique removes any of them without changing the
argument (a non-interactive Ligero would use Fiat–Shamir, excluded).

**RTT sweep [derivation]**, prover and verifier compute given (\(t_c\)), network floor `depth × RTT`:

| configuration | `rounds.protocol` | `rounds.sequential_depth` | `t.e2e.rtt_0ms` | `rtt_0.1ms` | `rtt_1ms` | `rtt_10ms` | `rtt_50ms` |
|---|---|---|---|---|---|---|---|
| B `helper_columns` | 16 | 3 | \(t_c\) | \(t_c+0.0003\) | \(t_c+0.003\) | \(t_c+0.03\) | \(t_c+0.15\) |
| B `helper_columns` + step 0 | 17 | 4 | \(t_c\) | \(+0.0004\) | \(+0.004\) | \(+0.04\) | \(+0.20\) |
| B `gkr` mode, \(B=4096\) | 3 447 | 328 | \(t_c\) | \(+0.033\) | \(+0.33\) | \(+3.28\) | \(+16.4\) |
| A baseline, \(B=4096\) (for comparison) | 3 607 | 328 | \(t_c\) | \(+0.033\) | \(+0.33\) | \(+3.28\) | \(+16.4\) |

Transport time for the 165 MB transcript at \(B=4096\) is not in the floor (`e2e_sweep(bandwidth_bytes_per_s=…)`
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
construction, binding argument, abort rules and simulator are those of Candidate A 5.5 with \(R=16\) slots;
the differences:

| item | Candidate B |
|---|---|
| coins committed (transcript order) | slot 1–14: \((z_T^{(1)},z_T^{(2)},\beta_T)\) per table (HM96 first scheme, \(\ell=3\cdot192\) bits, \(L=4\cdot160+2\ell+4\)); slot 15: \((r,\rho_{add},\rho_q)\) — \(r\) is 105 440 \(F_c\) elements (2.5 MB), committed by the \(O(k)\) variant (commit to \(H(r\|\rho_{add}\|\rho_q)\), open with the vector and \(x_{15}\)); slot 16: \(Q\) (\(t\cdot14\) bits), \(O(k)\) variant |
| message order | 0: \(c_1..c_{16}\) → 1: com_1 → 2: \(x_1..x_{14}\) (P verifies all, derives \(z_T,\beta_T\)) → 3: com_2 → 4: \(x_{15}\) with \(r,\rho\) → 5: responses → 6: \(x_{16}\) with \(Q\) → 7: openings |
| prover message timing | com_2 only after all 14 openings verified; responses only after \(x_{15}\) verified; column openings only after \(x_{16}\) verified |
| abort rules | P aborts iff an HM96 opening fails, a coin is out of domain, or all \(S\) redraw slots of a table lie in \(\mathrm{Img}_T\) — public data only; no other prover abort exists (4.4: the helper inversion never branches). V aborts on a failed check. No restart within a session |
| binding suffices because | the verifier's coins for steps 2, 4, 6 are fixed before com_1 up to an \(H\)-collision; a malicious verifier keeps only the choice to stop, so its view is a prefix of an honest-verifier view on pre-fixed coins |
| simulator | Candidate A 5.5 steps (1)–(5) with the HVZK simulator of AHIV22 Lemma 4.15 for both tableaux (helper rows are simulated as random rows too; the constraints are never checked by the simulator's verifier since the opened columns are uniform); the round-2 commitment com_2 is simulated like com_1 |
| additional depth | +1 (message 0); depth 3 → 4 |
| additional communication | \(16\cdot(144+100)\) B \(=3.9\) KB (165.48 MB → 165.48 MB) |
| additional prover work | 16 HM96 opening checks: one hash over the 2.5 MB vector \(r\) and 15 hashes of \(<1.4\) kbit strings — \(<10^{-4}\) of the batch's hashing (13.8 GB) |
| soundness | \(+16\cdot2^{-160}=2^{-156}\); total unchanged at \(2^{-130.2}\) |

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
| lookup | \(\sum_T (N_T+|T|-1)\big(\tfrac1{|F_c|-|T|}+\tfrac{\mathrm{cols}-1}{|F_c|}\big)\), conditioned challenge space | \(2^{-164.8}\) | \(2^{-100.8}\) |
| encoding/opening | \((1-e/n)^t+((k+l)/n)^t+(2k/n)^t+(n+1)/|F_c|+(\#\mathrm{lin}-1)/|F_c|+(\#\mathrm{quad\ rows}-1)/|F_c|\) (powers combination) | \(2^{-130.2}\) | \(2^{-82.0}\) |
| commitment/hash | \(q^2/2^{256}\) (two roots, same bound); with step 0 also \(16\cdot2^{-160}\) | \(2^{-136}\) | \(2^{-128}\) |
| **total** | | **\(2^{-130.2}\)** (\(B\in\{1,64,4096\}\)) | **\(2^{-82.0}\)** |
| zk (not soundness) | HM96 leaves, both tableaux | \(2^{-146}\) (one tableau counted by the calculator; \(2^{-145}\) for two) | idem |
| completeness (not soundness) | redraw \(\sum_T|T|/|F_c|\); abort \(\sum_T(|T|/|F_c|)^2\) | \(2^{-174.7}\); \(2^{-350}\) | \(2^{-110.7}\); \(2^{-222}\) |

### 9.2 Operating points (`first_configuration`; identical to Candidate A because the dominant terms are shared)

| operating point | first configuration | total, \(B\in\{1,64,4096\}\) | why the smaller extension fails |
|---|---|---|---|
| \(2^{-80}\) (diagnostic) | **quadratic** \(F_{p^2}\), \(t=120\) | \(2^{-81.4}\) | — |
| \(2^{-96}\) (sweep) | **quadratic** \(F_{p^2}\), \(t=144\) | \(2^{-97.6}\) / \(2^{-97.6}\) / \(2^{-97.3}\) | — |
| \(2^{-112}\) (sweep) | **cubic** \(F_{p^3}\), \(t=167\) | \(2^{-113.2}\) | quadratic: lookup identity \(2^{-109.4}\)…\(2^{-100.8}\) alone exceeds the target |
| \(2^{-128}\) (primary) | **cubic** \(F_{p^3}\), \(t=191\) (reference \(t=192\)) | \(2^{-129.5}\) | quadratic: same floor |

The dominant term is Ligero's IRS query term \((1-e/n)^t=(5/8)^{192}\); the two candidates have the same
statistical budget at matched parameters and differ in committed elements (4.0×), sequential depth (3 vs 328),
and opened elements (20.2M vs 5.0M \(F_t\) at \(B=4096\)). Ligero's amortisation **[source, AHIV22 abstract,
4.7]**: the three tests and the \(t\) column openings are shared by all rows, so verifier work and proof size grow
as \(t\cdot\mathrm{rows}+O(n)\) while soundness is independent of the row count.

### 9.3 Sub-batches

As Candidate A 9.3: 16 sub-batches of 256 turn \(2^{-130.2}\) into \(2^{-126.2}\) and miss \(2^{-128}\)
(`test_sub_batches_are_union_bounded_mechanically`, which uses Candidate B).

### 9.4 Census, \(K=1536\) (`census_estimate_b`; rows marked (est.) are estimates from the cost model of
Candidate A 9.4)

| column | B \(B=1\) | B \(B=64\) | B \(B=4096\) | A \(B=4096\) (for reference) |
|---|---|---|---|---|
| materialized witness elements (est.; committed + helpers as \(F_t\) coordinates) | 0.75M | 6.96M | 405M | 1.46G |
| committed elements (`census.committed_elements`) | 747k | 6.96M | **405M** | **101M** |
| bytes encoded (`census.bytes_encoded`) | 26 MB | 238 MB | 13.8 GB | 3.44 GB |
| hash/commit work (`census.hash_bytes`, est.) | 33 MB | 245 MB | 13.8 GB | 3.45 GB |
| field operations (`census.field_ops`, \(F_t\) mult-eq., est.: NTT + 3 mults/helper + Ligero tests) | 64M | 577M | 33.4G | 29.9G |
| memory passes / bytes moved (`census.memory_passes` / `census.bytes_moved`, est.) | 5 / 92 MB | 5 / 829 MB | 5 / 48 GB | 39 / 29.6 GB |
| sequential challenge depth (`rounds.sequential_depth`; `rounds.protocol`) | 3; 16 | 3; 16 | 3; 16 | 328; 3 607 |
| communication (`transcript.bytes`, est.; P→V / V→P) | 1.30 MB (1.29 / 0.005) | 3.83 MB (3.78 / 0.04) | 165 MB (163 / 2.5) | 42.2 MB (41.5 / 0.72) |
| peak memory (retained, est.) | 32 MB | 294 MB | 17.1 GB | 6.7 GB |

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
8. The layout assumptions (512 committed elements, 410 linear constraints per unit) pending WP3.
9. The census figures of 9.4 marked (est.) come from the stated cost model, not from measurement.

## 11. Open items (questions, not decisions)

- Encoding traffic: \(1.73\)G encoded \(F_t\) symbols (13.8 GB) and 17 GB retained peak per batch of 4096 at
  rate \(1/4\); rate \(1/2\) is excluded by the quadratic test (\((2k/n)^t=1\)). Sub-batching is allowed with
  `N_subbatches` and the union bound (9.3); the split point is a measurement question.
- Helper columns are \(50\%\) of the committed elements; the `gkr` mode removes them at the price of depth 328.
  Which of depth and committed bytes binds for the GPU end state is decided by the RTT sweep and the
  measurements, not here.
- Whether the first complete backend ships as `COMPLETE_HVZK_BACKEND` (no step 0) with 5.3 as the path to
  `COMPLETE_ZK_BACKEND`.
