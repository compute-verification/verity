# Candidate B — direct local constraints in an interactive hash-based ZK Ligero argument

Status: **specification / proposed parameterisation**. Nothing here is implemented or measured. The argument is
AHIV22's interactive ZKIPCP-plus-CRH construction applied to the census constraints of one 16-product update,
with logUp lookups (Hab22) as a second commitment round. Same tags as `backends/gkr/PROTOCOL.md`: **[source]**,
**[derivation]**, **[proposal]**, **[assumption]**.

**Label rule.** A plain Plonky3 AIR/STARK over the same columns is `NON_ZK_PROOF_DIAGNOSTIC`: its FRI
commitment is non-interactive by Fiat–Shamir (random-oracle assumption the requirements avoid) and its trace
openings are not hidden (no masking, no ZK). It measures arithmetic and lookup throughput only. It becomes a
candidate for `COMPLETE_ZK_BACKEND` only when its commitment/privacy layer is replaced by (or shown equivalent to)
sections 3, 5 and 7 below and the accounting of section 9 is reproduced by
`verity_numerical.security.accounting.account_candidate_b`.

## 0. Sources (version read 2026-09-21)

| tag | document | used for |
|---|---|---|
| AHIV22 | Ames–Hazay–Ishai–Venkitasubramaniam, *Ligero*, ePrint 2022/1608 (2022-11-18; extended CCS 2017) | ZKIPCP for arithmetic circuits (4.7: linear and quadratic constraints over interleaved RS codes), ZK variants of the three tests (4.6, App. B.4–B.5), interactive variant via CRH (5.1), Lemma 4.15 (ZK needs \(k>l+t\)), Theorem 4.7 and Appendix C (soundness), amortisation remarks |
| libzk | IETF `draft-google-cfrg-libzk-02` (22 July 2026), section 4 | concrete tableau layout: blinding rows, witness rows with `NREQ` random symbols, aligned quadratic row triples (4.3), prove/verify (4.4–4.5) |
| Hab22 | Haböck, ePrint 2022/1530, revision 2025-02-20 | logUp with helper (inverse) columns: Lemma 5, Theorem 4, section 3.4 (vector-valued lookups), batch inversion cost (3 mult. per helper) |
| PH23 | Papini–Haböck, ePrint 2023/1284, revision 2025-02-20 | optional `gkr` lookup mode (section 4.3) |
| FS24 | Frigo–shelat, ePrint 2024/2010, revision 2026-04-27, §2.2 | the Ligero bound used for parameters (Appendix C form) |

## 1. Relation, shapes, public data

Same public data as Candidate A section 1 (profile, \(K\), \(B\), \(U=BK/16\), tables, census). Differences:

| item | Candidate B |
|---|---|
| committed per unit | **every** value of the checker: operands (decoded), state in/out, products, quotients/remainders, leading-bit fields, flags, plus Ligero alignment copies for quadratic operands |
| constraints per unit **[source, census]** | 104 linear, 146 quadratic (`mul`), 172 lookups into 14 public tables (range checks are lookups into `T_range*`) |
| alignment copies **[source, libzk 4.3]** | Ligero's quadratic test needs \(x,y,z\) of each product in the same column of three rows: \(3\cdot146=438\) quadratic slots per unit, values reused across products are copied and bound by linear equalities |
| committed elements per unit **[assumption]** | 512 (438 slots + ≤74 values occurring only in linear/lookup predicates) until the WP3 layout lands (`CandidateB.witness_per_unit`) |
| lookup helpers (mode `helper_columns`) | one \(F_c\) element per query and per table row, committed **after** the lookup challenge: \(172U+\sum|T|\) elements \(\times\deg(F_c/F_t)\) coordinates |
| multiplicities \(m_T\) | committed in round 1 (challenge-independent) |

Committed trace-field elements, \(K=1536\), \(F_c=F_{p^3}\) (calculator): \(B=1\): 0.75M; \(B=64\): 7.0M;
\(B=4096\): **404.9M** = 201.3M witness + 0.16M multiplicities + 203.4M helper coordinates → 105 440 rows of
\(l=3840\), \(1.73\)G encoded symbols at rate \(1/4\). Candidate A commits 100.9M for the same batch: the
\(4.0\times\) difference is handoff 6.2's "encoding traffic and auxiliary lookup witness", quantified.

## 2. Fields

Identical to Candidate A section 2: Goldilocks trace/code field (64-bit reference first), \(F_{p^2}\) challenge
field for \(2^{-80}\), \(F_{p^3}\) for \(2^{-128}\) (the logUp identity term forces \(|F_c|\gtrsim2^{160}\) for
\(2^{-128}\) at every \(B\)); small-field study with a degree-6 extension of a 31-bit field, capped by
\(\mathrm{char}\,F>N_T\) (Hab22 Lemma 5). Helper columns live in \(F_c\) and are stored as coordinates; the
constraint \(h_j\cdot(z-w_j)=1\) with \(w_j\in F_t\) expands to \(\deg\) coordinate products **[derivation]**.

## 3. Commitment interface (AHIV22 4.7 / libzk 4.2–4.3)

Same code and parameters as Candidate A section 3: \([n=2^{14},k=2^{12}]\) RS over \(F_t\), \(l=3840\),
\(t=192\) (\(2^{-128}\)) or \(121\) (\(2^{-80}\)), \(e=(n-k)/2\), \(k\ge l+t\), three blinding rows per tableau,
leaf \(=H(\text{column})\). Two tableaux (two Merkle roots) in `helper_columns` mode; the Ligero tests run once
over the vertical juxtaposition of both (the tests are linear in rows, AHIV22 4.7 uses \(U\in L^{4m}\) the same
way). Row classes:

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

Soundness: Hab22 Theorem 4 identity term \((N_T+|T|-1)/(|F_c|-|T|)\) plus, for vector-valued rows, the
\(\beta\)-degree term \((N_T+|T|-1)(\mathrm{cols}-1)/|F_c|\) **[derivation]**; the constraint-satisfaction part is
Ligero's. No sumcheck, no batching term outside Ligero. Prover cost: one batch inversion per table
(\(3(N_T-1)\) multiplications + 1 inversion, Hab22 §5) plus encoding of the helper rows.

### 4.3 Lookups, mode `gkr` (PH23) **[proposal]**

The fractional sumcheck of Candidate A 4.2 with **committed** query columns: leaf claims become Ligero linear
constraints directly (no claim merging), pads for the fractional-GKR transcript are committed in round 1 (their
number is public), and there is **no second commitment round**. Trades 203M helper coordinates for
\(\approx3\sum_T n_T(n_T-1)/2+4n_T\) padded messages (\(\approx11\)k \(F_c\) elements) and 3 417 sumcheck rounds
at \(B=4096\); committed elements drop to 201.5M. Not the default: it re-introduces the padded-transcript
machinery of Candidate A.

## 5. Numbered transcript (mode `helper_columns`)

~~~text
 0  C  [proposal, malicious-verifier ZK]  statistically hiding commitments to all verifier coins of steps
       2, 4, 6 (Goldreich-Kahan style).  Without step 0: honest-verifier ZK only (AHIV22 5.1).
 1  M  com_1 = MerkleRoot_H( Encode(W_1) ),  W_1 = witness rows || quadratic triples || m_T rows || blinding
       rows.  Fixed here: every checker value and every multiplicity; W_1 is challenge-independent.
 2  C  z_T, beta_T for all 14 tables (one message); public redraw if z_T in { t_v(beta_T) }.
 3  M  com_2 = MerkleRoot_H( Encode(W_2) ),  W_2 = helper rows (h_j, h_v as F_t coordinates) || blinding rows.
       This is the only challenge-dependent auxiliary; it is committed, never sent in the clear.
 4  C  r in F_c^{rows_1+rows_2},  r_add in F_c^{#linear},  r_q in F_c^{#quadratic}
 5  M  v = r^T [W_1;W_2] + u_0 in F_c^n ;  q_add(.) deg < k+l-1 ;  p_0(.) deg < 2k-1     (AHIV22 4.7 step 2)
 6  C  Q subset [n], |Q| = t
 7  M  columns j in Q of both tableaux, with Merkle paths to com_1 and com_2
 8  V  Merkle paths; IRS test on v; linear test (sum_c q_add(zeta_c) = 0, column consistency); quadratic
       test (p_0(zeta_c) = 0, column consistency); accept/reject.
~~~

Five verifier messages (six with step 0) regardless of \(B\); no per-layer interaction. Linear constraints in
step 4 include: the 104 `lin` per unit, alignment equalities, table-side helper equations, the 14 sums, and the
integer-width predicates realised as lookups; quadratic constraints: the 146 `mul` per unit and the \(N_T\)
helper products.

## 6. Masked values and how masks are bound

| value | mask | binding |
|---|---|---|
| every committed row (witness, triples, multiplicities, helpers) | \(k-l\) uniformly random symbols per row | any \(t\) opened columns of a row are uniform (AHIV22 Lemma 4.15; libzk `random_row(NREQ)`) |
| IRS response \(v\) | blinding row \(u_0\) (random codeword) | AHIV22 4.6.1 |
| linear response \(q_{add}\) | blinding row with payload summing to 0 | AHIV22 4.6.2 / App. B.4 |
| quadratic response \(p_0\) | blinding row with zero payload | AHIV22 4.6.3 / App. B.5 |
| helpers \(h_j\) | committed (round 2), never sent | Ligero constraints of 4.2 |
| Merkle roots / unopened columns | none beyond the above | section 7, leaf hiding |

No prover message other than the two roots, the three blinded test responses and the \(t\) opened columns is
ever sent; there is no transcript to pad in `helper_columns` mode.

## 7. Hash properties assumed, and where

Same table as Candidate A section 7: collision resistance for both Merkle trees (soundness, `commitment_hash`
term); leaf hiding — either per-column statistically hiding commitments from CRH (AHIV22 5.1) or a separate
hiding/PRG assumption on \(H\) for unsalted leaves (libzk `commit`); a PRG assumption for GPU-side generation of
row randomizers and blinding rows (ZK only; uniform at the ideal level); a statistically hiding, computationally
binding commitment for step 0. No random oracle, no correlation intractability, no extractability of \(H\).

## 8. Leakage rules

| leak channel | rule |
|---|---|
| branch-dependent layout | every unit occupies the same slots; both arms of every branch are materialised; selectors are committed values |
| batch partitioning by exponent / group state | forbidden; one tableau pair per batch; padding instances public and prescribed |
| multiplicities, helpers | committed only; never sent |
| logUp pole | honest queries lie in \(T\), so a pole ⇔ \(z_T\in\{t_v(\beta_T)\}\): public, redrawn publicly; probability \(\sum_T|T|/|F_c|\approx2^{-175}\) (calculator `completeness`); never a witness-dependent abort |
| restart on a bad challenge | forbidden; abort the session; a new session recommits with fresh randomizers; sessions counted by the outer accounting (handoff 7.3) |
| number of opened columns / row counts | public functions of \((K,B)\) |
| helper commitment after the challenge | round 2 is a fresh randomized tableau; its root reveals nothing beyond section 7 |
| verifier-adaptive \(Q\) or \(r\) | with step 0 fixed before com_1; without it, honest-verifier ZK |
| timing | fixed-shape kernels; encoding and inversion work independent of values |

## 9. Soundness error decomposition (calculator output)

\(K=1536\), Goldilocks trace field, \(n=2^{14},k=2^{12},l=3840\), 256-bit hash; `account_candidate_b`.

| component | term | \(2^{-128}\) set: \(F_{p^3}\), \(t=192\), \(q=2^{60}\) | \(2^{-80}\) set: \(F_{p^2}\), \(t=121\), \(q=2^{64}\) |
|---|---|---|---|
| batching | none outside Ligero | 0 | 0 |
| sumcheck | none (`helper_columns`) | 0 | 0 |
| lookup | \(\sum_T (N_T+|T|-1)\big(\tfrac1{|F_c|-|T|}+\tfrac{\mathrm{cols}-1}{|F_c|}\big)\) | \(2^{-164.8}\) | \(2^{-100.8}\) |
| encoding/opening | \((1-e/n)^t+((k+l)/n)^t+(2k/n)^t+(n+3)/|F_c|\) | \(2^{-130.2}\) | \(2^{-82.0}\) |
| commitment/hash | \(q^2/2^{256}\) (two roots, same bound) | \(2^{-136}\) | \(2^{-128}\) |
| **total** | | **\(2^{-130.2}\)** (\(B\in\{1,64,4096\}\)) | **\(2^{-82.0}\)** |

The dominant term is again Ligero's IRS query term \((1-e/n)^t=(5/8)^{192}\); the two candidates have the same
statistical budget at matched parameters, and differ in committed elements (4.0×), rounds (5 vs. ≈480), and
opened elements (20.2M vs. 5.0M \(F_t\) at \(B=4096\)). Ligero's amortisation **[source, AHIV22 abstract, 4.7]**:
the three tests and the \(t\) column openings are shared by all rows, so verifier work and proof size grow as
\(t\cdot\mathrm{rows}+O(n)\) while soundness is independent of the row count.

## 10. Unproved obligations (verbatim list for the coordinator)

1. Knowledge soundness of the two-round variant (com_1 before, com_2 after the lookup challenge): AHIV22's
   extractor (Theorem 4.7 / 5.1) is stated for a single oracle; the extension to a challenge-dependent second
   oracle with the same query set is standard but not cited for the interactive CRH variant.
2. Simulation: honest-verifier ZK inherited from AHIV22 Lemma 4.15; malicious-verifier ZK via step 0 is proposed,
   not proved, and its cost is not measured.
3. Hiding of unsalted Merkle leaves (section 7, option ii) is an assumption on \(H\) beyond collision resistance.
4. The IRS/proximity statement for a subfield RS code over \(F_t\) with combination coefficients in \(F_c\): the
   calculator charges the Appendix C field terms over \(|F_c|\); the lemma for this setting is not cited.
5. Extraction/composition with Verity's outer commitments (handoff 7.5): W_1 is a fresh commitment to the
   operands; equality with the values fixed by Verity's earlier commitments is a separate proof obligation, and
   the outer draft's extraction/composition requirements for the numerical backend remain unproved.
6. Integer semantics inside \(F_t\) (handoff 7.2): range widths for every congruence are a WP2 obligation.
7. Application-level accounting of sessions/batches (handoff 7.3) is not fixed here.
8. The layout assumption of 512 committed elements per unit (438 aligned quadratic slots) pending WP3.

## 11. Tensions to raise (questions, not decisions)

- Encoding traffic: \(1.73\)G encoded \(F_t\) symbols (13.8 GB) per batch of 4096 at rate \(1/4\); rate \(1/2\)
  is excluded by the quadratic test (\((2k/n)^t=1\)). Is a per-batch memory ceiling part of the benchmark
  contract, or may batches be split into independently committed sub-batches (each with its own error budget)?
- Helper columns are \(50\%\) of the committed elements; the `gkr` mode removes them at the price of ≈3 400
  interactive rounds. Which of round count and committed bytes is the binding constraint for the GPU end state?
- \(2^{-128}\) requires the cubic extension for the lookup identity at every \(B\); same question as Candidate A.
