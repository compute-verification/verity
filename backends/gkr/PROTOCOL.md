# Candidate A — data-parallel GKR + logUp-GKR lookups + interactive Longfellow-style ZK layer

Status: **specification / proposed research**. Nothing here is implemented or measured. The interactive
adaptation of the Longfellow ZK layer (section 5) and the merging of logUp-GKR claims into the checker GKR
(section 4.3) are *proposals*; they are not drop-in components with a published proof for this composition.
Any prover built from this document is `NON_ZK_PROOF_DIAGNOSTIC` until sections 5, 6 and 8 are implemented and
the accounting of section 9 is reproduced by `verity_numerical.security.accounting.account_candidate_a`.

Text is tagged as in the handoff: **[source]** read in a cited document, **[derivation]** follows from sources by
a calculation shown or referenced here, **[proposal]** a design choice of this lane, **[assumption]** a parameter
or shape not yet fixed by another lane.

## 0. Sources (version read 2026-09-21)

| tag | document | used for |
|---|---|---|
| Tha13 | Thaler, *Time-Optimal Interactive Proofs for Circuit Evaluation*, arXiv:1304.3812v4 (8 Feb 2017), section 7 (7.1–7.3) | data-parallel GKR: copy index as extra sumcheck variables, prover cost \(O(S\cdot B)\) after \(O(S\log S)\) circuit-only preprocessing (Theorem 2) |
| Hab22 | Haböck, *Multivariate lookups based on logarithmic derivatives*, ePrint 2022/1530, revision 2025-02-20 | logUp: Lemma 5 (multiset ⇔ rational identity, needs \(\mathrm{char}\,F>N\)), Theorem 4 (soundness), section 3.4 (vector-valued lookups) |
| PH23 | Papini–Haböck, *Improving logarithmic derivative lookups using GKR*, ePrint 2023/1284, revision 2025-02-20 | fractional-sum GKR (section 3), Proposition 1 (GKR error), Proposition 2 (logUp-GKR error); the note calls itself *informal* and contains no ZK composition |
| FS24 | Frigo–shelat, *Anonymous credentials from ECDSA*, ePrint 2024/2010, revision 2026-04-27 | Protocol 2.5 (ZK by padded sumcheck transcript + Ligero on the decryption/verification circuit), Theorem 6 (honest-verifier ZK; knowledge soundness), section 2.2 Ligero bound used for parameters |
| libzk | IETF `draft-google-cfrg-libzk-02` (22 July 2026), sections 4–6 | Ligero tableau layout and commitment interface (4.2–4.3), prove/verify (4.4–4.5), quad circuit representation and in-circuit assertions (6.3), transcript encryption and deferred verification (6.5–6.7) |
| AHIV22 | Ames–Hazay–Ishai–Venkitasubramaniam, *Ligero*, ePrint 2022/1608 (2022-11-18; extended CCS 2017) | ZKIPCP (4.7), interactive variant via CRH (5.1), zero-knowledge condition Lemma 4.15, Theorem 4.7 and Appendix C bounds |

Not read for this document (cited by name only, as standard constructions): Halevi–Micali statistically hiding
commitments from CRH; Goldreich–Kahan verifier-coin commitment. Both appear only in **[proposal]** items.

## 1. Relation, shapes, public data

| symbol | value / meaning | public? |
|---|---|---|
| profile | `AMPERE_BF16_M16N8K16`, align `div` (`verity.ml.tc.silicon`) | yes |
| \(K\) | dot-product length; decision workload \(K=1536\) | yes |
| \(B\) | instances per proof (padded batch; real count also public) | yes |
| \(U=B\cdot K/16\) | transition units (one 16-product update each); \(K=1536\): \(U=96B\) | yes |
| \(b=\lceil\log_2 U\rceil\) | copy bits; \(B\in\{1,64,4096\}\Rightarrow b\in\{7,13,19\}\) | yes |
| \(C\) | fixed per-unit checker, layered quad circuit (libzk 6.3.2), layers \(L_0\) (assertions) … \(L_D\) (inputs), per-unit widths \(2^{s_i}\) | yes |
| \(x_u\in F_t^{2^{s_D}}\) | per-unit private inputs: 32 decoded operands, incoming state, outgoing state, hints (quotients/remainders, leading-bit positions, exponent, flags) | **no** |
| \(T_1..T_{14}\) | public lookup tables of the census (below) | yes |
| \(m_T\in\mathbb Z_{\ge0}^{|T|}\) | multiplicities of table \(T\) over the whole batch | **no** |
| census per unit | 146 `mul`, 104 `lin`, 172 lookups (`verity.ml.tc.relation.census`, instance-independent) **[source]** | yes |

Lookups per unit by table **[source]**: `T_bf16` 32 (65536 rows, 4 columns), `T_range16` 46, `T_range10` 36,
`T_pow26` 18 (512 rows, 3 cols), `T_pcls` 16 (64 rows, 5 cols), `T_any9` 6, `T_iszero12` 4, and 2 each for
`T_exp25`, `T_norm30`, `T_range14`, `T_range13`, `T_range8`, `T_range5`, `T_range3`. Total lookups at
\(K=1536,B=4096\): \(172\cdot 393216 = 67.6\)M, the largest table column `T_range16` has \(N=18.1\)M entries.

**[assumption]** Checker shape used by the calculator until the WP3 census lands: inputs \(2^8\) per unit, three
computed layers of widths \(2^7,2^6,2^5\), assertion layer \(2^8\) (`DEFAULT_LAYERS_A`). The message shape of every
protocol step depends only on the public data of this table; no message length or branch depends on a witness.

## 2. Fields

| role | choice | reason |
|---|---|---|
| trace field \(F_t\) (inputs, multiplicities, Reed–Solomon code) | Goldilocks \(p=2^{64}-2^{32}+1\) (**64-bit reference first**, handoff 7.2) | \(2^{32}\)-smooth multiplicative group gives RS length up to \(2^{32}\); every integer of the relation (\(|S|<2^{30}\), products \(<2^{16}\cdot2^{16}\), shifted magnitudes \(<2^{35}\)) embeds without wrap **[derivation]** |
| challenge field \(F_c\) (all sumcheck/logUp/Ligero-combination randomness, pads) | \(F_{p^2}\) (\(\approx2^{128}\)) for the \(2^{-80}\) target; \(F_{p^3}\) (\(\approx2^{192}\)) for \(2^{-128}\) | section 9: the logUp identity term is \((N+|T|-1)/(|F_c|-|T|)\approx 2^{-101}\) over \(F_{p^2}\) at \(B=4096\); \(2^{-128}\) needs \(|F_c|\gtrsim2^{160}\) **[derivation]** |
| small-field study | Baby Bear \(15\cdot2^{27}+1\) / Mersenne-31 trace field, degree-6 extension for \(2^{-128}\) (degree 5 misses: lookup terms \(\approx2^{-127}\)) | \(|S|<2^{30}\) motivates it; logUp's \(\mathrm{char}\,F>N\) (Hab22 Lemma 5) caps total lookups per table at \(<2^{31}\): fine at \(B=4096\) (\(N\le1.8\cdot10^7\)), fails at \(B=2^{20}\) (`test_small_field_study_limits`) |

Rule: \(F_c\) is an extension of \(F_t\); pads are \(F_c\) elements stored as \(\deg(F_c/F_t)\) trace-field
coordinates; an \(F_c\)-linear constraint on pads is \(\deg\) \(F_t\)-linear constraints on coordinates, an
\(F_c\)-product is \(\deg^2\) coordinate products plus linear combinations **[derivation]**.

## 3. Commitment interface (Ligero tableau, interactive, AHIV22 4.7 / libzk 4.2–4.3)

| parameter | value (\(2^{-128}\) set) | constraint (AHIV22 / libzk 4.2.1) |
|---|---|---|
| RS code | \([n,k]\) over \(F_t\), \(n=2^{14}\), \(k=2^{12}\), rate \(1/4\), distance \(d=n-k+1\) | \(n>2k\) (quadratic test), \(|F_t|\ge n+l\) |
| payload per row \(l\) | 3840 witness symbols | \(k\ge l+t\) so that any \(t\) opened columns are uniform (AHIV22 Lemma 4.15: \(k>l+t\); libzk: BLOCK = WR + NREQ with queries outside message positions) |
| opened columns \(t\) | 192 (121 for the \(2^{-80}\) set) | section 9 |
| proximity \(e\) | \((n-k)/2=6144\) (Appendix C regime \(e<d/2\)) | |
| blinding rows | one random codeword row per test (LDT row; linear-test row with payload summing to 0; quadratic-test row with zero payload) — libzk `layout_zk_rows` | |
| leaf \(j\) | \(H(T[0][j]\,\|\cdots\|\,T[\mathrm{NROW}][j])\) over the \(n\) codeword columns | hiding of unopened columns: see section 7 |
| committed elements, \(K=1536,B=4096\) | \(100.9\)M \(F_t\) elements = \(393216\cdot2^8\) inputs \(+\) 162k multiplicities \(+\) 10 991 pads \(\times3\) coordinates → 26 269 rows, \(430\)M encoded symbols | calculator `parameters` |

Quadratic constraints are laid out libzk-style as aligned triples of rows (`layout_quadratic_rows`); Candidate A
has only the deferred products of section 6 (\(\approx4\) per checker layer, 3 per fractional-GKR layer, times
\(\deg^2\) coordinates), so alignment copies are negligible here (they are not for Candidate B).

## 4. Interactive reductions

### 4.1 Data-parallel GKR over the checker (Tha13 §7, libzk 6.3)

Layer \(i\) has per-unit wires \(V_i(g,c)\), \(g\in\{0,1\}^{s_i}\), \(c\in\{0,1\}^{b}\) the copy index. With the
libzk quad predicate \(Q_i(g,l,r)\) (additions encoded as quads with the constant-1 wire, so one predicate per layer):

\[
\widetilde V_i(z,\gamma)=\sum_{g,c,l,r} \mathrm{eq}(z,g)\,\mathrm{eq}(\gamma,c)\,\widetilde Q_i(g,l,r)\,\widetilde V_{i+1}(l,c)\,\widetilde V_{i+1}(r,c).
\]

**[derivation]** variable degrees \(g:2\), \(c:3\), \(l:2\), \(r:2\); rounds per layer \(s_i+b+2s_{i+1}\);
sumcheck error \((2s_i+3b+4s_{i+1})/|F_c|\). The wiring predicate is copy-independent (Tha13 §7.3), so the prover
tables of \(\widetilde Q_i\) are built once per circuit and shared by all \(U\) copies (Theorem 2 preprocessing).
Binding order **[proposal]**: \(g\) first, then \(c\), then \(l\), \(r\) — the \(c\) rounds are the ones with
\(U\)-parallel prover work, the \(l,r\) rounds are per-circuit only.

Two evaluation claims on layer \(i+1\) (\(\widetilde V_{i+1}(l^*,c^*)\), \(\widetilde V_{i+1}(r^*,c^*)\)) are
merged into one with a fresh \(\alpha\) (libzk 6.3.2; error \(1/|F_c|\)); in-circuit assertions of a layer are
folded with a fresh \(\beta\) (libzk 6.3.3; \(1/|F_c|\)). The top claim is \(\widetilde V_0(z_0)=0\) at a uniform
\(z_0\in F_c^{s_0+b}\): all \(2^{s_0}U\) assertion wires are zero (Schwartz–Zippel, \((s_0+b)/|F_c|\)).

### 4.2 Lookups: logUp-GKR per table (Hab22, PH23)

For table \(T\) with \(N_T=q_T U\) queries \(w_1..w_{N_T}\) (checker wires, **not committed**) and multiplicities
\(m_T\) (committed), the identity **[source, Hab22 Lemma 5]**

\[
\sum_{j=1}^{N_T}\frac{1}{z-w_j}=\sum_{v\in T}\frac{m_T(v)}{z-t_v}
\]

holds as rational functions iff the multisets agree (requires \(\mathrm{char}\,F_c>N_T\)). Vector-valued rows
(columns \(>1\)) are compressed with powers of a second challenge \(\beta_T\) (Hab22 3.4).

**[source, PH23 §3]** The fractional sum is proved by GKR over a hypercube of dimension
\(n_T=\lceil\log_2(N_T+|T|)\rceil\) with leaves \((p,q)\): witness side \((1,\,z-w_j(\beta))\), table side
\((-m_T(v),\,z-t_v(\beta))\), padding \((0,1)\). Layer \(k\) combines \(2^{k+1}\) fractions into \(2^k\) by
\((p,q)\mapsto(p_0q_1+p_1q_0,\;q_0q_1)\); each layer is a degree-3 sumcheck of \(k\) variables (error \(3k/|F_c|\))
followed by a two-point-to-one-point reduction and a \(\lambda\) combination of the \(p\) and \(q\) claims
(\(2/|F_c|\) per layer). The root value is the **cleared numerator** \(p_{\mathrm{root}}(z)=\sum_j p_j\prod_{i\ne j}q_i\);
the verifier asserts \(p_{\mathrm{root}}=0\). **[derivation]** \(p_{\mathrm{root}}(X)\) is a nonzero polynomial
of degree \(\le N_T+|T|-1\) whenever the multisets differ, so no separate \(q_{\mathrm{root}}\ne0\) check is
needed; \(q_{\mathrm{root}}\) is still sent (padded) because layer 0 needs both values.

Multiplicities are challenge-independent (counts of witness values) and are committed in message M1 **before**
any challenge. No inverse or product intermediate is committed (PH23's point); all \((p,q)\) layers are
computed by the prover on the fly in \(F_c\).

### 4.3 Linking lookups to the checker **[proposal]**

The final fractional-GKR claim for table \(T\) at \(\rho_T\in F_c^{n_T}\) splits linearly into
(a) a claim on \(\widetilde m_T\) at the table-side coordinates → one Ligero linear constraint on the committed
multiplicities (section 6), and (b) a claim
\(\sum_{g,c}L_T(g,c)\,V_{i(T)}(g,c)=\text{value}\) on the checker layer \(i(T)\) holding all query wires of
\(T\), where \(L_T\) is public (products of \(\mathrm{eq}\) factors and \(\beta_T\) powers). Layout rule: every
query wire of table \(T\) is an output wire of a single layer \(i(T)\) (identity gates copy where needed);
queries on inputs (`T_bf16` decode of operands) become Ligero linear constraints directly. Claim (b) has the
same form as the layer claim with \(\mathrm{eq}(z,\cdot)\mathrm{eq}(\gamma,\cdot)\) replaced by \(L_T\); it is
merged into layer \(i(T)\)'s sumcheck with a fresh coefficient (\(1/|F_c|\) each) without changing degrees.
This is the mechanism that satisfies handoff 6.2's "witnesses must agree on the shared query values": the
lookup argument never has its own copy of a query value.

## 5. ZK layer: padded transcript + deferred Ligero verification, made interactive

FS24 Protocol 2.5 is already an interactive protocol (Fiat–Shamir is applied afterwards in FS24 §2.4); libzk
specifies only the non-interactive form. **[proposal]** We run Protocol 2.5's structure with live challenges and
three extensions: data-parallel GKR with \(F_t\) witness / \(F_c\) challenges, logUp-GKR fractional sumchecks
inside the padded transcript, and (optionally) verifier-coin commitments for malicious-verifier ZK. This is
proposed research: FS24 Theorem 6 is stated for honest-verifier ZK of the plain circuit case; nothing below has
a written proof for this composition (section 10).

### 5.1 Numbered transcript (\(K,B\), profile, \(C\), tables, code parameters, \(H\) fixed publicly beforehand)

Notation: `M` prover message, `C` verifier challenge (uniform in \(F_c\) unless stated). `⟨v⟩` denotes a padded
value \(v+\mathrm{pad}\) with a fresh committed pad; `open(coin)` is an opening of the verifier's coin commitment
(step 0) — without step 0 the coin is simply sent.

~~~text
 0  C  [proposal, malicious-verifier ZK]  V samples every coin of steps 2-6 and sends statistically hiding
       commitments c_V (one per challenge slot).  Skipping step 0 gives honest-verifier ZK only.
 1  M  com_1 = MerkleRoot_H( Encode(W_1) ),   W_1 = (x_u)_{u<U} || (m_T)_T || coords(pad) || blinding rows.
       Everything in W_1 is challenge-independent: inputs+hints, ALL multiplicities, ALL pads for steps 2-4,
       Ligero row randomizers and the three blinding rows.  Fixed here: multiplicities, pad count P (public).
 2  --- lookups, one fractional GKR per table T (14 tables; independent challenges, may be interleaved) ---
 2a C  z_T (and beta_T if columns(T) > 1); public rule: if z_T in { t_v(beta_T) } redraw (public data only).
 2b M  <q_root>;  p_root := 0 asserted (nothing sent).
 2c    for k = 1 .. n_T-1:
         M  <p_k(rho,0)>, <p_k(rho,1)>, <q_k(rho,0)>, <q_k(rho,1)>      (rho = current point, |rho| = k-1)
         C  mu_k, lambda_k                                                 (line reduction + p/q combination)
         k rounds of degree-3 sumcheck:  M <s_j(0)>, <s_j(2)>, <s_j(3)>  ;  C r_j
 2d    leaf claims at rho_T: (a) m~_T(rho_T|table)  -> deferred linear constraint (step 6)
                             (b) sum_c beta^c w~^{(c)}(rho_T|witness) -> pending claim on checker layer i(T)
 3  --- checker GKR, layer by layer ---
 3a C  z_0 in F_c^{s_0+b};  claim V~_0(z_0) = 0.
 3b    for i = 0 .. D-1:
         C  alpha_i (two incoming claims -> one), beta_i (assertions), one coefficient per pending lookup
            claim on layer i
         s_i + b + 2 s_{i+1} rounds:  M <s_j(0)>, <s_j(2)> (+<s_j(3)> in the b degree-3 rounds);  C r_j
         M  <v_l> = <V~_{i+1}(l*,c*)>,  <v_r> = <V~_{i+1}(r*,c*)>
 4  C  gamma: input-layer claims (v_l, v_r, lookup claims on inputs) -> deferred linear constraints.
 5  --- deferred verification = Ligero ZKIPCP, interactive (AHIV22 4.7 steps 1-3) ---
 5a C  r in F_c^{rows}, r_add in F_c^{#linear}, r_q in F_c^{#quadratic}
 5b M  v = r^T U + u_0 in F_c^n ;  q_add(.) of degree < k+l-1 ;  p_0(.) of degree < 2k-1
 5c C  Q subset [n], |Q| = t
 5d M  columns T[.][j], j in Q, with Merkle paths
 6  V  checks: Merkle paths; IRS test on v; sum_c q_add(zeta_c)=0 and column consistency (all linear
       constraints of section 6); p_0(zeta_c)=0 and column consistency (quadratic constraints); accept/reject.
~~~

What is fixed when: inputs, hints, multiplicities, pads, blinding rows — step 1, before any challenge. No
challenge-dependent value is ever committed; every challenge-dependent prover message is a padded transcript
element or a Ligero test response (blinded by the step-1 blinding rows).

Rounds (sequential verifier messages), \(K=1536\), \(B=4096\): calculator total 3 571 sumcheck rounds
(checker 154; fractional GKRs \(\sum_T n_T(n_T-1)/2 + (n_T-1)\)); the 14 fractional GKRs are independent and can
be interleaved (the longest, `T_range16`, has 324 rounds), giving \(\approx 480\) sequential round trips plus 3
for Ligero **[derivation]**.

### 5.2 Masked values and how masks are bound

| value | mask | binding |
|---|---|---|
| every sumcheck evaluation \(s_j(\cdot)\) (steps 2c, 3b) | one \(F_c\) pad each, committed in W_1 | libzk 6.5: verifier's check \(s_j(0)+s_j(1)=s_{j-1}(r_{j-1})\) becomes a linear constraint on pads with public coefficients (Lagrange weights at \(r_{j-1}\)); accumulated symbolically per layer (FS24 §2.3) |
| layer evaluations \(v_l,v_r\); fractional-GKR \(p_k,q_k\) at two points; \(q_{\mathrm{root}}\) | one pad each | final-round identity \(s_{\text{last}}(r)=\widetilde Q\cdot v_l v_r\) (resp. \(p_0q_1+p_1q_0+\lambda q_0q_1\)) → FS24 Claim: linear constraints plus one quadratic constraint per product on pads (section 6) |
| leaf claims on \(\widetilde m_T\), input claims | pad each | linear constraint: ⟨public eq-vector, committed \(m_T\) or \(x\)⟩ − pad = public value |
| committed rows (inputs, multiplicities, pads) | \(k-l\) random symbols per row | any \(t\) columns uniform (AHIV22 Lemma 4.15) |
| Ligero responses \(v, q_{add}, p_0\) | blinding rows \(u_0, u_{add}, u_q\) | AHIV22 4.6.1–4.6.3 (affine tests, Appendix B.4–B.5) |
| Merkle root / unopened columns | none beyond the above | see section 7 (hash hiding property) |

Pads are never reused: one per transcript element, indexed by (step, table/layer, round, evaluation point);
the index map is public and fixed by \((K,B,C)\). A pad reused across two elements would leak their difference.

### 5.3 Deferred checks (all evaluated inside the Ligero tests of step 5–6)

| check | Ligero constraint type | count (\(K=1536,B=4096\)) **[derivation]** |
|---|---|---|
| sumcheck round consistency \(s_j(0)+s_j(1)=s_{j-1}(r_{j-1})\) | linear over \(F_c\) → \(\deg\) linear over \(F_t\) | 3 571 |
| final-round identity per checker layer (\(\widetilde Q\, v_l v_r\)) | 1 quadratic + linear | 4 |
| final-round identity per fractional-GKR layer (3 products) | 3 quadratic + linear | \(3\sum_T(n_T-1)\approx 900\) |
| claim merges (\(\alpha,\beta\), lookup coefficients, \(\gamma\), \(\mu,\lambda\)) | linear | one per coefficient |
| \(\widetilde m_T\) leaf claims | linear on committed \(m_T\) | 14 |
| input-layer claims | linear on committed \(x\) | 2 + input-layer lookups |
| \(p_{\mathrm{root}}=0\) | folded into layer-0 identity | 14 |

The verifier's own work: Merkle path checks for \(t\) columns, the IRS/linear/quadratic tests on \(t\) columns
(\(t\cdot\mathrm{rows}\) = 5.0M \(F_t\) elements opened at \(B=4096\)), evaluation of \(\widetilde Q_i\) at the
random points, and the public eq/Lagrange coefficient vectors. Proof size and verifier time are secondary
(handoff 2.1) and are reported, not optimised.

## 6. Constraint generation (libzk 6.7 adapted)

Public inputs to constraint generation: all challenges, all padded values, \(\widetilde Q_i\) at the challenge
points, table evaluations \(\widetilde t_v(\beta_T)\) at \(\rho_T\). Output: matrices \(A\) (linear) and triples
(quadratic) over \(F_c\) on the committed vector \(W_1\), expanded to \(F_t\) coordinates. Both parties compute
them; the prover needs them only for step 5b.

## 7. Hash properties assumed, and where

| property of \(H\) | used where | consequence if it fails |
|---|---|---|
| **collision resistance** (family chosen publicly before step 1, per AHIV22 5.1) | Merkle tree over columns: binding of com_1; step 5d openings | soundness: a prover opening inconsistent columns; the `commitment_hash` term is the generic \(q^2/2^h\) bound for \(q\) hash evaluations (a modelling choice, section 9) |
| **hiding of leaf hashes** (unopened columns) | leaf \(=H(\text{column})\) with no salt (libzk `commit`) | ZK: distinguishing \(H\) of correlated high-entropy columns. **Two options** **[proposal]**: (i) CRH-only statement: per-column statistically hiding commitment from CRH (Halevi–Micali; AHIV22 5.1 phrasing) at extra cost; (ii) practical: assume \(H\) is hiding on uniform-looking inputs (a PRG/random-oracle-style property, **separate** from collision resistance). The first implementation shall state which it uses. |
| **PRG** (hash-based expansion of a seed into pads, row randomizers, blinding rows on the GPU) | prover randomness generation | ZK only; at the ideal level all randomness is uniform (handoff 7.4). Recorded as a separate assumption. |
| statistically hiding, computationally binding commitment for verifier coins (step 0) | malicious-verifier ZK | hiding failure → soundness (prover learns coins); binding failure → ZK (verifier adapts coins) |
| **not assumed**: random oracle, correlation intractability, extractability of \(H\) | — | Fiat–Shamir is not used; knowledge soundness of the argument is via AHIV22's extractor for the interactive variant, which uses collision resistance only (section 10 records it as unproved for *this* composition) |

## 8. Leakage rules

| leak channel | rule |
|---|---|
| branch-dependent message shapes (e.g. shift direction, zero/overflow flags) | every checker branch is fixed-size (selectors + both arms evaluated); number and degree of every sumcheck message is a function of \((K,B,C)\) only |
| batch partitioning by exponent / group state | forbidden; one batch, one GKR, one commitment; padding instances are public and prescribed (handoff 9.1) |
| lookup multiplicities | committed, never sent; only \(\widetilde m_T(\rho)\) padded |
| logUp pole (\(z_T-w_j=0\)) | honest query values lie in \(T\), so a pole exists iff \(z_T\in\{t_v(\beta_T)\}\): a **public** condition, rejected and redrawn publicly; the prover never aborts on witness data (calculator term `completeness`, \(\sum_T|T|/|F_c|\approx2^{-175}\)) |
| restart on a bad challenge | forbidden. A session is aborted, never restarted with the same W_1; a new session recommits with fresh pads and randomizers (and the outer protocol counts sessions, handoff 7.3) |
| timing / message length | all messages fixed length; prover work per round data-independent (fixed-shape kernels) |
| verifier-adaptive challenges | with step 0, challenges are fixed before com_1; without it, only honest-verifier ZK is claimed |
| Merkle root of a deterministic layout | rows randomized (\(k-l\) fresh symbols); leaf hiding per section 7 |
| Ligero responses \(v,q_{add},p_0\) | blinded (section 5.2); the linear/quadratic response polynomials have degree \(<k+l-1\), \(<2k-1\) and hide the payload by the blinding rows' randomness |

## 9. Soundness error decomposition (calculator output)

Union bound of statistical terms plus one computational term. \(K=1536\), Goldilocks trace field, Ligero
\(n=2^{14},k=2^{12},l=3840\), hash 256-bit. Figures from `account_candidate_a` (tests in
`backends/numerical/tests/security/test_accounting.py`).

| component | term | \(2^{-128}\) set: \(F_c=F_{p^3}\), \(t=192\), \(q=2^{60}\) hash evaluations | \(2^{-80}\) set: \(F_c=F_{p^2}\), \(t=121\), \(q=2^{64}\) |
|---|---|---|---|
| batching | \(z_0\) Schwartz–Zippel \((s_0+b)/|F_c|\); \(\alpha,\beta\) per layer; \(\gamma\); 14 lookup merges | \(2^{-186.4}\) | \(2^{-122.4}\) |
| sumcheck | \(\sum_i(2s_i+3b+4s_{i+1})/|F_c|\), 154 rounds | \(2^{-183.4}\) | \(2^{-119.4}\) |
| lookup | \(\sum_T\big[(N_T+|T|-1)(\tfrac1{|F_c|-|T|}+\tfrac{\mathrm{cols}-1}{|F_c|}) + \tfrac{2(n_T-1)+1+\sum_{k<n_T}3k}{|F_c|}\big]\) | \(2^{-164.8}\) | \(2^{-100.8}\) |
| encoding/opening | \((1-e/n)^t+((k+l)/n)^t+(2k/n)^t+(n+3)/|F_c|\) (AHIV22 App. C, FS24 §2.2) | \(2^{-130.2}\) | \(2^{-82.0}\) |
| commitment/hash | \(q^2/2^{256}\) | \(2^{-136}\) | \(2^{-128}\) |
| **total** | | **\(2^{-130.2}\)** (\(B\in\{1,64,4096\}\) all pass) | **\(2^{-82.0}\)** |

Dominant term: the Ligero query terms; the field terms are far below. Observations **[derivation]**:
\(F_{p^2}\) cannot reach \(2^{-128}\) at any \(B\) (lookup identity \(\approx2^{-109}\) even at \(B=1\), set by
the \(2^{16}\)-row tables); the sumcheck component depends on \(B\) only through \(b\). Rounds and committed
elements scale linearly in \(U\); soundness terms only logarithmically.

The calculator refuses `security_bits(field)`; only a complete `Budget` has a security level. The hash term is a
modelling choice (adversary work \(q\)), reported separately as `computational`.

## 10. Unproved obligations (verbatim list for the coordinator)

1. Knowledge soundness of the interactive composition (data-parallel GKR + logUp-GKR + padded transcript +
   Ligero): FS24 Theorem 6 covers a plain layered circuit with the Ligero extractor; the fractional-GKR
   sub-transcripts, the lookup-claim merging of 4.3 and the \(F_t\)/\(F_c\) coordinate expansion are not covered.
2. Simulation (zero knowledge): only honest-verifier ZK is inherited (AHIV22 Lemma 4.15, FS24 Theorem 6). The
   verifier-coin-commitment transformation of step 0 for malicious-verifier ZK is proposed, not proved, for this
   protocol; its round/latency cost is not measured.
3. Hiding of the Merkle leaves without per-column statistically hiding commitments (section 7, option ii) is an
   assumption on \(H\) beyond collision resistance.
4. The proximity-gap / IRS statement for a subfield RS code (\(F_t\)) tested with combination coefficients in
   the extension \(F_c\): the calculator charges the Appendix C field terms over \(|F_c|\); the corresponding
   lemma for this exact setting is not cited.
5. Extraction/composition with Verity's outer commitments (handoff 7.5): W_1 is a fresh commitment to the
   operands; equality with the values fixed by Verity's earlier commitments is a separate proof obligation, and
   the outer draft's extraction/composition requirements for the numerical backend remain unproved.
6. Integer semantics inside \(F_t\) (handoff 7.2): every range bound that makes a field congruence an integer
   equality is a checker obligation (WP2), not established here.
7. Application-level accounting: the number of sessions/batches an adversary may open multiplies the error
   budget (handoff 7.3); not fixed here.
8. The checker shape (`DEFAULT_LAYERS_A`, \(2^8\) inputs per unit) is an assumption pending the WP3 census.

## 11. Tensions to raise (questions, not decisions)

- Interactivity vs. batching latency: \(\approx480\) sequential round trips per batch (\(\approx3\,600\) without
  interleaving); at 1 ms RTT that is \(0.5\)–\(3.6\) s of pure latency per batch on top of proving. Is a
  round-trip budget part of the benchmark contract (handoff 2.1 "report rounds, communication, latency effects")?
- Malicious-verifier ZK (step 0) doubles the verifier's messages and adds a hash-based commitment scheme; is
  honest-verifier ZK acceptable for the first complete backend, with step 0 as a documented gap?
- The \(2^{-128}\) target forces \(|F_c|\ge2^{160}\) (cubic Goldilocks extension) for *every* \(B\) because of
  the 65 536-row tables; is \(2^{-128}\) a hard target or a sweep point (handoff 7.3 "a label is not a bound")?
