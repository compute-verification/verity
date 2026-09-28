# A-GKR — data-parallel GKR + logUp-GKR lookups + interactive Longfellow-style ZK layer

Status: **specification**. A-GKR is frozen; `README.md` says which parts are built and measured. The interactive
adaptation of the Longfellow ZK layer (section 5), the merging of logUp-GKR claims into the checker GKR
(section 4.3), the depth reductions of section 5.4 marked **[proposal]** and the verifier-coin commitment of
section 5.5 are *proposals*; none is a drop-in component with a published proof for this composition.

**Label rule** (`archive/AGENTS.md`). A prover built from this document is `NON_ZK_PROOF_DIAGNOSTIC` until
sections 5, 6 and 8 are implemented and section 9 is reproduced by
`verity_numerical.security.accounting.account_candidate_a`. With sections 5.1–5.3 and 6–8 implemented but
**without step 0** (section 5.5) it is `COMPLETE_HVZK_BACKEND` — honest-verifier privacy, a milestone, never
`COMPLETE_ZK_BACKEND`. `COMPLETE_ZK_BACKEND` requires step 0 (or another justified mechanism) implemented, its
transcript order / commitments / binding / timing / abort rules / privacy argument / costs written down (5.5) and
red-team reviewed.

Text is tagged as in the handoff: **[source]** read in a cited document, **[derivation]** follows from sources by
a calculation shown or referenced here, **[proposal]** a design choice of this lane, **[assumption]** a parameter
or shape not yet fixed by another lane.

## 0. Sources (version read 2026-09-21)

| tag | document | used for |
|---|---|---|
| Tha13 | Thaler, *Time-Optimal Interactive Proofs for Circuit Evaluation*, arXiv:1304.3812v4 (8 Feb 2017), section 7 (7.1–7.3) | data-parallel GKR: copy index as extra sumcheck variables, prover cost \(O(S\cdot B)\) after \(O(S\log S)\) circuit-only preprocessing (Theorem 2) |
| Hab22 | Haböck, *Multivariate lookups based on logarithmic derivatives*, ePrint 2022/1530, revision 2025-02-20 | logUp: Lemma 5 (multiset ⇔ rational identity, needs \(\mathrm{char}\,F>N\)), Theorem 4 (soundness, challenge drawn from a set of size \(\ge|F|-|H|\)), Remark 3 (sampling outside the table range is for completeness only), section 3.4 (vector-valued lookups) |
| PH23 | Papini–Haböck, *Improving logarithmic derivative lookups using GKR*, ePrint 2023/1284, revision 2025-02-20 | fractional-sum GKR (section 3), Proposition 1 (GKR error), Proposition 2 (logUp-GKR error); the note calls itself *informal* and contains no ZK composition |
| FS24 | Frigo–shelat, *Anonymous credentials from ECDSA*, ePrint 2024/2010, revision 2026-04-27 | Protocol 2.5 (ZK by padded sumcheck transcript + Ligero on the decryption/verification circuit), Theorem 6 (honest-verifier ZK; knowledge soundness), section 2.2 Ligero bound used for parameters |
| libzk | IETF `draft-google-cfrg-libzk-02` (22 July 2026), sections 4–6 | Ligero tableau layout and commitment interface (4.2–4.3), prove/verify (4.4–4.5), quad circuit representation and in-circuit assertions (6.3), transcript encryption and deferred verification (6.5–6.7) |
| AHIV22 | Ames–Hazay–Ishai–Venkitasubramaniam, *Ligero*, ePrint 2022/1608 (2022-11-18; extended CCS 2017) | ZKIPCP (4.7), interactive variant via CRH (5.1), zero-knowledge condition Lemma 4.15, Theorem 4.7 and Appendix C bounds |
| HM96 | Halevi–Micali, *Practical and Provably-Secure Commitment Schemes from Collision-Free Hashing*, CRYPTO 1996, LNCS 1109 pp. 201–215 (author's PDF, sections 1.2, 3, 4 read) | statistically hiding, computationally binding string commitment from a collision-free hash: first scheme (section 3: \(L=4k+2n+4\) random bits \(x\), universal hash \(h\) with \(h(x)=m\), commit string \((h,\mathrm{MD}(x))\); Theorem 1: statistical distance \(<2^{-k}\)); \(O(k)\)-bit variant (section 4: commit to \(\mathrm{MD}(m)\), \(9k\)-bit commit string) |

Not read for this document (cited by name only): Goldreich–Kahan, *How to construct constant-round zero-knowledge
proof systems for NP*, J. Cryptology 9(3) 1996 — the verifier-coin-commitment technique and its simulator; the
simulator of section 5.5 is written out so that the argument does not rest on the citation.

## 1. Relation, shapes, public data

| symbol | value / meaning | public? |
|---|---|---|
| profile | `AMPERE_BF16_M16N8K16`, align `div` (`verity.ml.circuits.silicon.silicon`) | yes |
| \(K\) | dot-product length; decision workload \(K=1536\) | yes |
| \(B\) | instances per proof (padded batch; real count also public); `N_subbatches` independently committed sub-batches → union bound (`union_over_proofs`) | yes |
| \(U=B\cdot K/16\) | transition units (one 16-product update each); \(K=1536\): \(U=96B\) | yes |
| \(b=\lceil\log_2 U\rceil\) | copy bits; \(B\in\{1,64,4096\}\Rightarrow b\in\{7,13,19\}\) | yes |
| \(C\) | fixed per-unit checker, layered quad circuit (libzk 6.3.2), layers \(L_0\) (assertions) … \(L_D\) (inputs), per-unit widths \(2^{s_i}\) | yes |
| \(x_u\in F_t^{2^{s_D}}\) | per-unit private inputs: the 34 words, the 225 hints (decode digits, flags, selectors, quotients/remainders, leading-bit positions, clamp/overflow flags), the 62 looked-up row columns; with `commit_query_wires` (default) also the 140 query wires | **no** |
| \(T_1..T_5\) | public lookup tables of the census (below): `T_range16`, `ALIGN`, `POW`, `LEAD`, `NORM` | yes |
| \(m_T\in\mathbb Z_{\ge0}^{|T|}\) | multiplicities of table \(T\) over the whole batch | **no** |
| census per unit (WP2 checker contracts, `verity_numerical.security.census.census_unit`) **[derivation from source]** | 358 `mul` (114 of them booleans), 137 `lin` (115 identities + 22 range-chunk splits), 306 lookups (284 sixteen-bit range chunks + 22 fixed-table rows); committed columns 321 = 34 words (\(c\), 32 operands, \(y\)) + 225 hints + 62 looked-up row columns; 244 products + 100 materialised range expressions as computed wires; multiplicative depth 5; 140 of the 306 lookup queries are on computed wires | yes |

**Census source (2026-09-21).** The counts are taken from the WP2 checker (`verity_numerical.checker`: `step.py`'s
composed unit executed once on symbolic values through the five predicates `rng`/`eq`/`boolean`/`lookup`/`lt_pow`,
hint widths from every gadget's `contract()`), **not** from `verity.ml.circuits.silicon.relation.census`, which is forgeable as
written (red team: `magW` never range-checked; WP2 F1/F2: remainders and quotient bounds) and counts none of the
repairs. Conventions (`security.census` docstring): a product of two non-constant expressions is one `mul`;
`boolean` is one `mul`; a \(w\)-bit range check is \(\lceil w/16\rceil\) lookups into the one 16-bit table (a
\(w<16\) value is checked as \(x\cdot2^{16-w}\)) plus one `lin` split when \(w>16\); `lt_pow` is the reference
realisation \(x\cdot\mathrm{ipow}<2^B\) (one `mul` + one \(B\)-bit range; sound over Goldilocks since
\(p>2^{60}\ge2^{2B}\), checker F1 — a small field needs the direct realisation, not counted). Identical products
formed twice (`q*sr` in `div` and `quot`) are counted twice (upper count).

Lookups per unit by table: `T_range16` 284 (65 536 rows, 1 column), `ALIGN` 16 (512 rows, 4 columns), `POW` 2
(512, 3), `LEAD` 2 (31, 3), `NORM` 2 (32, 4). Multiplicity columns \(\sum_T|T|=66\,623\) (relation: 162 715). Range
checks per unit by width: 7-bit 36, 8-bit 66, 9-bit 26, 16-bit 112, 23-bit 2, 24-bit 5, 25-bit 8, 30-bit 6,
32-bit 1. Total lookups at \(K=1536,B=4096\): \(306\cdot393216=120.3\)M; `T_range16` has \(N=111.7\)M entries
(\(n_T=27\)). Hint widths (contract ranges): 114 one-bit (flags, selectors, signs), 2 five-bit, 34 seven-bit,
33 eight-bit, 2 nine-bit, 32 sixteen-bit, 2 twenty-three-bit, 2 twenty-four-bit, 4 twenty-five-bit.

**Diff against `relation.vu_census` (`census.diff_vs_relation`, `test_census.py`)**, checker − relation:

| | relation `mul`/`lin`/`lookup` | checker `mul`/`lin`/`lookup` | Δ |
|---|---|---|---|
| unit (16 products) | 146 / 104 / 172 | 358 / 137 / 306 | +212 / +33 / +134 |
| VU \(K=1536\) | 14 027 / 9 992 / 16 521 | 34 370 / 13 059 / 29 379 | +20 343 / +3 067 / +12 858 |

Not in the relation's count: the direct range of every remainder (`align_state.r`, `align_product.r`,
`normalise.r`) and of the normalised magnitude (`normalise.M`, the `magW` repair); the tight quotient bounds
(`quot` products); the decode of an arbitrary incoming accumulator word in every unit; the pack of the output
word in every unit; operand decode by hints + range checks (the checker uses no `T_bf16`, `T_pcls`, `T_exp`,
`T_iszero` tables — those lookups are replaced by 4 hints, 3 range checks, 2 `mul`, 2 `lin` per operand).

**Two unit variants (`census_unit`, `census_vu`).** The backend transition unit takes an arbitrary finite word
\(c\) (decode + pack inside; the row above). The VU (`check_vu_expr`) is 96 word-chained units from the constant
\(c_0=0\) plus the epilogue: 30 631 committed columns (A), 34 370 `mul`, 13 059 `lin`, 29 379 lookups, i.e.
\(96\times\)unit \(+2\) `mul` \(+6\) hints \(+3\) lookups: the variants differ by \(<0.01\%\) because `step.py`
decodes and packs the word in every unit. A state-chained VU (decode once, pass \((s,e,M)\)) would save
\(\approx14\) `mul`, 7 hints, 9 `lin`, 10 range chunks per inner unit (\(\approx4\%\)); it is not what `step.py`
composes and is not counted.

**Checker shape [derivation from the census; widths are estimates]** (`census.layer_shape`, `DEFAULT_LAYERS_A`):
libzk 6.3 layers are strict, so a value used \(j\) layers later is carried by copy gates; the width of the layer at
multiplicative depth \(d\) is taken as its products plus every value of lower depth (upper bound).

| layer (output first) | products | carried width per unit | \(2^{s_{out}}\) | reads \(2^{s_{in}}\) | holds lookup queries |
|---|---|---|---|---|---|
| assertions (137 `lin`) | 0 | 137 | \(2^8\) | \(2^{10}\) | no |
| depth 5 | 6 | 679 | \(2^{10}\) | \(2^{10}\) | yes |
| depth 4 | 4 | 673 | \(2^{10}\) | \(2^{10}\) | yes |
| depth 3 | 22 | 669 | \(2^{10}\) | \(2^{10}\) | yes |
| depth 2 | 114 | 647 | \(2^{10}\) | \(2^{10}\) | yes |
| depth 1 | 212 | 533 | \(2^{10}\) | \(2^9\) (321 inputs) | yes |

The message shape of every protocol step depends only on the public data of this table; no message length or
branch depends on a witness.

## 2. Fields

| role | choice | reason |
|---|---|---|
| trace field \(F_t\) (inputs, multiplicities, Reed–Solomon code) | Goldilocks \(p=2^{64}-2^{32}+1\) (**64-bit reference first**, handoff 7.2) | \(2^{32}\)-smooth multiplicative group gives RS length up to \(2^{32}\); every integer of the checker (\(|S|<2^{30}\), `lt_pow` products \(<2^{2B}\le2^{60}\), words \(<2^{32}\)) embeds without wrap: checker F1–F4 are satisfied by Goldilocks with the product realisation of `lt_pow` (`checker/__init__.py` section 3) **[source]** |
| challenge field \(F_c\) (all sumcheck/logUp/Ligero-combination randomness, pads) | \(F_{p^2}\) (\(\approx2^{128}\)) first meets \(2^{-80}\) and \(2^{-96}\); \(F_{p^3}\) (\(\approx2^{192}\)) first meets \(2^{-112}\) and \(2^{-128}\) | section 9.2: the logUp identity term \((N+|T|-1)/(|F_c|-|T|)\) floors at \(2^{-111.3}\) (\(B=1\)) … \(2^{-100.9}\) (\(B=4096\)) over \(F_{p^2}\) **[derivation]** |
| small-field study | Baby Bear \(15\cdot2^{27}+1\) / Mersenne-31 trace field, degree-6 extension for \(2^{-128}\) (degree 5 misses: lookup terms \(\approx2^{-127}\)) | \(|S|<2^{30}\) motivates it; logUp's \(\mathrm{char}\,F>N\) (Hab22 Lemma 5) caps total lookups per table at \(<2^{31}\): fine at \(B=4096\) (`T_range16` \(N=1.1\cdot10^8\)), fails at \(B=2^{20}\) (`test_small_field_study_limits`) |

Rule: \(F_c\) is an extension of \(F_t\); pads are \(F_c\) elements stored as \(\deg(F_c/F_t)\) trace-field
coordinates; an \(F_c\)-linear constraint on pads is \(\deg\) \(F_t\)-linear constraints on coordinates, an
\(F_c\)-product is \(\deg^2\) coordinate products plus linear combinations **[derivation]**.

## 3. Commitment interface (Ligero tableau, interactive, AHIV22 4.7 / libzk 4.2–4.3)

| parameter | value (\(2^{-128}\) set) | constraint (AHIV22 / libzk 4.2.1) |
|---|---|---|
| RS code | \([n,k]\) over \(F_t\), \(n=2^{14}\), \(k=2^{12}\), rate \(1/4\), distance \(d=n-k+1\) | \(n>2k\) (quadratic test), \(|F_t|\ge n+l\) |
| payload per row \(l\) | 3840 witness symbols | \(k\ge l+t\) so that any \(t\) opened columns are uniform (AHIV22 Lemma 4.15: \(k>l+t\); libzk: BLOCK = WR + NREQ with queries outside message positions) |
| opened columns \(t\) | 192 reference set (minimum 191, section 9.2); 121 reference / 120 minimum for \(2^{-80}\) | section 9 |
| proximity \(e\) | \((n-k)/2=6144\) (Appendix C regime \(e<d/2\)) | |
| blinding rows | one random codeword row per test (LDT row; linear-test row with payload summing to 0; quadratic-test row with zero payload) — libzk `layout_zk_rows` | |
| leaf \(j\) **[proposal, default]** | HM96 \(O(k)\)-bit commitment to column \(j\): \(s_j=H(T[0][j]\|\cdots\|T[\mathrm{NROW}][j])\), random \(x_j\in\{0,1\}^{6k+4}\), universal \(h_j\) with \(h_j(x_j)=s_j\), leaf \(=(h_j,\,H(x_j))\), \(k=160\); opening \(=\) column \(\|\,x_j\) | hiding of unopened columns is **statistical**, \(n\cdot2^{-160}=2^{-146}\) (HM96 Theorem 1, union over \(n\) leaves; calculator `zk` term); binding by collision resistance of \(H\). Alternative (unsalted \(H(\text{column})\), libzk `commit`) needs a hiding assumption on \(H\) beyond CRH (section 7) |
| committed elements, \(K=1536,B=4096\) | \(181.4\)M \(F_t\) elements = \(393216\cdot(321+140)\) inputs, hints, row columns and query wires \(+\) 66 623 multiplicities \(+\) 4 674 pads \(\times3\) coordinates → 47 231 rows, \(774\)M encoded symbols; \(126.3\)M without the query wires (`commit_query_wires=False`, depth 630) | calculator `parameters` |

Quadratic constraints are laid out libzk-style as aligned triples of rows (`layout_quadratic_rows`); A-GKR
has only the deferred products of section 6 (\(\approx4\) per checker layer, 3 per fractional-GKR layer, times
\(\deg^2\) coordinates), so alignment copies are negligible here (they are not for B-Ligero).

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

For table \(T\) with \(N_T=q_T U\) queries \(w_1..w_{N_T}\) (checker wires or inputs, **not separately
committed**) and multiplicities \(m_T\) (committed), the identity **[source, Hab22 Lemma 5]**

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
the verifier asserts \(p_{\mathrm{root}}=0\). **No division occurs anywhere in the protocol**: the tree is
built from products and sums only (section 4.4).

**Challenge space and the conditioned bound.** \(z_T\) is uniform over \(F_c\setminus\mathrm{Img}_T\),
\(\mathrm{Img}_T=\{t_v(\beta_T):v\in T\}\) (public; \(|\mathrm{Img}_T|\le|T|\)), realised by the public redraw
rule of 5.1 step 2a. **[derivation]** \(p_{\mathrm{root}}(X)\) is a nonzero polynomial of degree
\(\le d_T=N_T+|T|-1\) whenever the multisets differ (Hab22 Lemma 5), so it has at most \(d_T\) roots in
\(F_c\setminus\mathrm{Img}_T\), and

\[
\Pr_{z\leftarrow F_c\setminus\mathrm{Img}_T}[p_{\mathrm{root}}(z)=0\mid\text{multisets differ}]
\;\le\;\frac{d_T}{|F_c|-|\mathrm{Img}_T|}\;\le\;\frac{N_T+|T|-1}{|F_c|-|T|}.
\]

This is the bound the calculator charges (`logup_identity_error(..., challenge_space="conditioned")`); it is
also the sampling set of Hab22 Protocol 2 (condition \(x\notin -t(H)\), Remark 3), so Theorem 4's
\(((M+1)|H|-1)/(|F|-|H|)\) is the same statement, not a whole-field theorem reused. The whole-field variant
(\(z\) uniform over \(F_c\), Hab22 Remark 3's overwhelmingly-complete protocol) gives \(d_T/|F_c|\); the two differ
by exactly \(d_T|T|/(|F_c|(|F_c|-|T|))\): at \(K=1536,B=4096\), `T_range16` (\(N_T=111.7\)M), \(F_c=F_{p^3}\), that is \(2^{-341.3}\)
absolute, a relative \(|T|/(|F_c|-|T|)=2^{-176}\) (`test_conditioned_challenge_space_vs_whole_field_theorem`).
For vector-valued rows the \(\beta\)-degree term \(d_T(\mathrm{cols}-1)/|F_c|\) is added **[derivation]**
(conditioning on \(\beta_T\), union bound).

Multiplicities are challenge-independent (counts of witness values) and are committed in message M1 **before**
any challenge. No inverse or product intermediate is committed (PH23's point); all \((p,q)\) layers are
computed by the prover on the fly in \(F_c\).

**Variable order of the per-layer sumcheck (fixed 2026-09-22, lane a-protocol-diff).** The identity above does
not fix which bit of the layer index a sumcheck round binds, but the *message values* depend on it, so it is part
of the transcript specification. Layer \(k\) (size \(2^k\), children at layer \(k+1\) indexed \(2x+b\)) is
reduced from the claim \(\sum_x eq(\rho,x)\,[p_0q_1+p_1q_0+\lambda q_0q_1](x)\), \(p_b(x)=p_{k+1}(2x+b)\), by
\(k\) rounds where **round \(i\) binds bit \(i\) of \(x\) (low bit first**: the round-\(i\) pairs are
\((2y,2y+1)\) of the current table). After the four claims \(p_{k+1}(2x^*),p_{k+1}(2x^*+1),q_{k+1}(2x^*),
q_{k+1}(2x^*+1)\) and \(\mu_k\), the next point is \(\rho'=(\mu_k,x^*_0,\dots,x^*_{k-1})\): \(\mu_k\) on bit 0
(the children pairing), \(x^*_i\) on bit \(i+1\); \(eq(\rho',x^{*\prime})\) at the next layer pairs coordinate
\(j\) with coordinate \(j\). The leaf point \(\rho_T\) is handed to the linear-constraint step (4.3, 5.1 step
2d) in the top-bit-first convention of `eq_table` (coordinate 0 on the leaf index's top bit), i.e. reversed.
This is the schedule of the GPU prover (`gpu/logup.py`, `packed/logup_reference.py`) and of both verifiers
(`gpu/logup.py::verify_table`, `verifier/src/verify.rs::verify_table`). The Goldilocks crate (`src/logup.rs`)
and the a-gpu prover up to v2e bound the **top** bit first with \(\rho'=(x^*,\mu_k)\): the same protocol under
bit-reversal of the layer index, with different message values from layer 2 on (the first layer whose
\(eq(\rho,x^*)\) has two factors). The two are **not** transcript-compatible; a verifier written to one rejects
the other at `LogUp <table> level 2: final check` (`note:r20-proof/a-protocol-diff/20260922T1047Z-report-a-protocol-diff`).

### 4.3 Linking lookups to the checker **[proposal]**

The final fractional-GKR claim for table \(T\) at \(\rho_T\in F_c^{n_T}\) splits linearly into
(a) a claim on \(\widetilde m_T\) at the table-side coordinates → one Ligero linear constraint on the committed
multiplicities (section 6), and (b) a claim
\(\sum_{g,c}L_T(g,c)\,V_{i(T)}(g,c)=\text{value}\) on the checker layer \(i(T)\) holding all query wires of
\(T\), where \(L_T\) is public (products of \(\mathrm{eq}\) factors and \(\beta_T\) powers).

**Where the queries are (census).** Of the 306 queries per unit, 166 are input-layer values (hints such as
`q`, `r`, `t`, `f`, `M`, `low`, the 33 words) and **140 are computed wires**: the `lt_pow` products
(\(r\cdot\mathrm{ipow}\), \(q\cdot\mathrm{pow}\), \((v-\mathrm{lo})\cdot\mathrm{ispan}\), \(r\cdot\mathrm{idiv}\)), the
`checked` expressions (\(t-nz\), \(M+z-1\), \(u+z-1\), \(v\), \(d_i=E-\mathrm{cand}_i\), `clamp_proof`,
`ovf_proof`) and the table keys \(d\), \(k\). They sit on every computed layer (depths 1–5).

**Layout rule [proposal], default `commit_query_wires`**: each of the 140 computed query wires is *also*
committed as an input column, bound to its expression by the identity the checker already states (the product
or linear identity is a gate of its layer either way), so every claim (b) is a Ligero linear constraint on the
committed \(x\) and the lookups never wait for the checker GKR. Cost: \(+140\) committed elements per unit
(\(321\to461\), \(+55\)M at \(B=4096\)). This satisfies handoff 6.2's "witnesses must agree on the shared query
values": the lookup argument reads the committed copy, and the checker binds the copy to the wire.
**Without it**, claim (b) is merged into the sumcheck of the layer holding the query wire (fresh coefficient,
\(1/|F_c|\) each, degrees unchanged), and every layer that holds a query — all five computed layers — follows the
lookups on the critical path (section 5.4: depth 630 instead of 381).

### 4.4 Every denominator, and why none can abort on witness data

| where | denominator | status |
|---|---|---|
| logUp identity (4.2), leaves \(q_j=z_T-w_j(\beta_T)\), \(q_v=z_T-t_v(\beta_T)\) | never inverted: the fractional GKR computes \(p_{\mathrm{root}},q_{\mathrm{root}}\) by products and sums (PH23 §3) | no division. Honest prover: \(w_j\in T\) and \(z_T\notin\mathrm{Img}_T\) ⇒ every \(q\ne0\), so \(p_{\mathrm{root}}=0\) exactly (completeness). Dishonest \(w_j=z_T\): some \(q_j=0\); the identity check \(p_{\mathrm{root}}=0\) is then the polynomial statement bounded in 4.2 (soundness), nothing to catch separately |
| verifier's evaluation of \(s_j(r_j)\) from \(s_j(0),s_j(1),s_j(2)[,s_j(3)]\) | Lagrange denominators \(\prod_{a\ne b}(a-b)\) over \(\{0,1,2,3\}\) | public constants, nonzero (\(\mathrm{char}>3\)) |
| \(\mathrm{eq}(z,\cdot)\), \(L_T\), \(\beta_T\) powers, \(\mathrm{Lagrange}\) weights of the padded transcript (FS24 §2.3) | none / public constants | public data only |
| Reed–Solomon encoding and the verifier's IRS/linear/quadratic tests | Vandermonde / NTT twiddles over the public evaluation domain (\(\eta_j\), \(\zeta_c\) distinct) | public constants |
| HM96 commitments (leaves, step-0 coins) | none | — |
| two-claims-to-one line reduction, \(\alpha,\beta,\gamma,\mu,\lambda\) combinations | none (multiplications by challenges) | — |
| deferred final-round identities (6) | none (products of padded values) | — |

There is no witness-dependent division in A-GKR; the only zero-test on witness data is the public
\(z_T\in\mathrm{Img}_T\) rule of step 2a, whose inputs (\(z_T\), \(\beta_T\), the table) are all public.

## 5. ZK layer: padded transcript + deferred Ligero verification, made interactive

FS24 Protocol 2.5 is already an interactive protocol (Fiat–Shamir is applied afterwards in FS24 §2.4); libzk
specifies only the non-interactive form. **[proposal]** We run Protocol 2.5's structure with live challenges and
three extensions: data-parallel GKR with \(F_t\) witness / \(F_c\) challenges, logUp-GKR fractional sumchecks
inside the padded transcript, and (optionally) verifier-coin commitments for malicious-verifier ZK. This is
proposed research: FS24 Theorem 6 is stated for honest-verifier ZK of the plain circuit case; nothing below has
a written proof for this composition (section 10).

### 5.1 Numbered transcript (\(K,B\), profile, \(C\), tables, code parameters, \(H\) fixed publicly beforehand)

Notation: `M` prover message, `C` verifier challenge (uniform in \(F_c\) unless stated). `⟨v⟩` denotes a padded
value \(v+\mathrm{pad}\) with a fresh committed pad. With step 0 every `C` line is an *opening* of the
corresponding coin commitment (5.5); without step 0 the coin is simply sent. Challenge **slots** are the `C`
lines instantiated (each \(r_j\), each \(\mu_k,\lambda_k\), …); `rounds.protocol` counts slots,
`rounds.sequential_depth` counts verifier messages on the critical path (5.4).

~~~text
 0  C  [proposal, malicious-verifier ZK; section 5.5]  V samples every coin of steps 2-5 and sends one HM96
       commitment per challenge slot (R = rounds.protocol slots).  Without step 0: COMPLETE_HVZK_BACKEND at most.
 1  M  com_1 = MerkleRoot_H( HM96-leaves( Encode(W_1) ) ),
       W_1 = (x_u)_{u<U} || (m_T)_T || coords(pad) || row randomizers || blinding rows.
       Everything in W_1 is challenge-independent: inputs+hints, ALL multiplicities, ALL pads for steps 2-4,
       Ligero row randomizers and the three blinding rows.  Fixed here: multiplicities, pad count P (public).
 2  --- lookups: one fractional GKR per table T (5 tables, independent challenges, run in parallel) ---
 2a C  for every T at once: z_T^(1), z_T^(2) (redraw slots, S = 2), beta_T if columns(T) > 1; z_0 of step 3a.
       Public rule: z_T := the first z_T^(i) not in Img_T = { t_v(beta_T) }; if all S lie in Img_T the session
       ends (public abort, probability (|T|/|F_c|)^S per table, calculator `completeness`).  No round trip.
 2b M  <q_root>;  p_root := 0 asserted (nothing sent).
 2c    for k = 1 .. n_T-1:
         M  <p_k(rho,0)>, <p_k(rho,1)>, <q_k(rho,0)>, <q_k(rho,1)>      (rho = current point, |rho| = k-1)
         C  mu_k, lambda_k               (must follow the four claims: the prover may not know them earlier)
         k rounds of degree-3 sumcheck:  M <s_j(0)>, <s_j(2)>, <s_j(3)>  ;  C r_j
 2d    leaf claims at rho_T: (a) m~_T(rho_T|table)  -> deferred linear constraint (step 6)
                             (b) sum_c beta^c w~^{(c)}(rho_T|witness) -> deferred linear constraint on x when
                                 the queries are committed (layout rule 4.3); else pending claim on layer i(T)
 3  --- checker GKR, layer by layer (in parallel with step 2 when no pending claim is outstanding) ---
 3a    claim V~_0(z_0) = 0, z_0 in F_c^{s_0+b} (issued in 2a).
 3b    for i = 0 .. D-1:
         C  alpha_i (two incoming claims -> one), beta_i (assertions), one coefficient per pending lookup
            claim on layer i                        (must follow the previous layer's <v_l>, <v_r>)
         s_i + b + 2 s_{i+1} rounds:  M <s_j(0)>, <s_j(2)> (+<s_j(3)> in the b degree-3 rounds);  C r_j
         M  <v_l> = <V~_{i+1}(l*,c*)>,  <v_r> = <V~_{i+1}(r*,c*)>
 4  C  gamma: input-layer claims (v_l, v_r, lookup claims on inputs) -> deferred linear constraints.
 5  --- deferred verification = Ligero ZKIPCP, interactive (AHIV22 4.7 steps 1-3) ---
 5a C  r in F_c^{rows} (explicit vector), r_add in F_c^{#linear}, r_q in F_c^{#quadratic}
 5b M  v = r^T U + u_0 in F_c^n ;  q_add(.) of degree < k+l-1 ;  p_0(.) of degree < 2k-1
 5c C  Q subset [n], |Q| = t
 5d M  columns T[.][j], j in Q, with HM96 openings x_j and Merkle paths
 6  V  checks: HM96 openings and Merkle paths; IRS test on v; sum_c q_add(zeta_c)=0 and column consistency
       (all linear constraints of section 6); p_0(zeta_c)=0 and column consistency (quadratic constraints);
       every step-0 opening verified as it arrived; accept/reject.
~~~

What is fixed when: inputs, hints, multiplicities, pads, randomizers, blinding rows — step 1, before any
challenge. No challenge-dependent value is ever committed; every challenge-dependent prover message is a padded
transcript element or a Ligero test response (blinded by the step-1 blinding rows). Verifier coins (with step
0) are fixed before step 1.

### 5.2 Masked values and how masks are bound

| value | mask | binding |
|---|---|---|
| every sumcheck evaluation \(s_j(\cdot)\) (steps 2c, 3b) | one \(F_c\) pad each, committed in W_1 | libzk 6.5: verifier's check \(s_j(0)+s_j(1)=s_{j-1}(r_{j-1})\) becomes a linear constraint on pads with public coefficients (Lagrange weights at \(r_{j-1}\)); accumulated symbolically per layer (FS24 §2.3) |
| layer evaluations \(v_l,v_r\); fractional-GKR \(p_k,q_k\) at two points; \(q_{\mathrm{root}}\) | one pad each | final-round identity \(s_{\text{last}}(r)=\widetilde Q\cdot v_l v_r\) (resp. \(p_0q_1+p_1q_0+\lambda q_0q_1\)) → FS24 Claim: linear constraints plus one quadratic constraint per product on pads (section 6) |
| leaf claims on \(\widetilde m_T\), input claims | pad each | linear constraint: ⟨public eq-vector, committed \(m_T\) or \(x\)⟩ − pad = public value |
| committed rows (inputs, multiplicities, pads) | \(k-l\) random symbols per row | any \(t\) columns uniform (AHIV22 Lemma 4.15) |
| Ligero responses \(v, q_{add}, p_0\) | blinding rows \(u_0, u_{add}, u_q\) | AHIV22 4.6.1–4.6.3 (affine tests, Appendix B.4–B.5) |
| unopened columns | HM96 leaf commitment, \(k=160\) | statistical, \(2^{-146}\) over all leaves (section 3) |

Pads are never reused: one per transcript element, indexed by (step, table/layer, round, evaluation point);
the index map is public and fixed by \((K,B,C)\). A pad reused across two elements would leak their difference.

### 5.3 Deferred checks (all evaluated inside the Ligero tests of step 5–6)

| check | Ligero constraint type | count (\(K=1536,B=4096\)) **[derivation]** |
|---|---|---|
| sumcheck round consistency \(s_j(0)+s_j(1)=s_{j-1}(r_{j-1})\) | linear over \(F_c\) → \(\deg\) linear over \(F_t\) | 1 569 |
| final-round identity per checker layer (\(\widetilde Q\, v_l v_r\)) | 1 quadratic + linear | 6 |
| final-round identity per fractional-GKR layer (3 products) | 3 quadratic + linear | \(3\sum_T(n_T-1)=315\) |
| claim merges (\(\alpha,\beta\), lookup coefficients, \(\gamma\), \(\mu,\lambda\)) | linear | one per coefficient |
| \(\widetilde m_T\) leaf claims | linear on committed \(m_T\) | 5 |
| input-layer claims | linear on committed \(x\) | 2 + input-layer lookups |
| \(p_{\mathrm{root}}=0\) | folded into layer-0 identity | 5 |

The verifier's own work: HM96 openings and Merkle paths for \(t\) columns, the IRS/linear/quadratic tests on
\(t\) columns (\(t\cdot\mathrm{rows}\) = 9.1M \(F_t\) elements opened at \(B=4096\)), evaluation of
\(\widetilde Q_i\) at the random points, and the public eq/Lagrange coefficient vectors. Proof size and
verifier time are secondary (handoff 2.1) and are reported, not optimised.

### 5.4 Protocol rounds vs. sequential challenge depth (latency axis)

Definitions (contract names). `rounds.protocol` = number of challenge slots of the transcript (every `C` line
instantiated; a naive one-challenge-per-message implementation needs this many verifier messages).
`rounds.sequential_depth` = verifier messages on the critical path after every available parallelisation: a
verifier message costs one network round trip only if the next prover message depends on it. The network floor
is `depth × RTT`; the final prover message (5d) adds a one-way latency not counted.

**Known techniques, always applied [source / immediate]:**
1. Independent sumchecks run in parallel: all 5 fractional GKRs and the checker GKR are independent instances
   (the lookups touch the checker only through the deferred constraints or a merged claim), so one verifier
   message carries all their round-\(j\) challenges and one prover message all their round-\(j\) polynomials
   (Tha13 §7 / libzk batch the *instances*; running independent protocols in lock-step is the standard
   observation that a public-coin verifier's coins for independent sub-protocols may be sent together).
2. Coins that do not depend on a prover message travel with the previous challenge: all \(z_T,\beta_T,z_0\) and
   the \(S\) redraw slots go in one message (2a); the redraw costs no trip.
3. Coins that **must** follow a prover message stay separate: \(\mu_k,\lambda_k\) after the four claims (else
   the prover chooses claims knowing the combination point — the two-claims-to-one reduction is sound only for
   a coin chosen after the claims), \(\alpha_i,\beta_i\) after \(v_l,v_r\), each \(r_j\) after \(s_j\), \(Q\)
   after the Ligero responses.

**Derivation of the baseline depth [derivation]**, \(K=1536,B=4096\), \(c=1\) variable per message, query wires
committed (layout rule 4.3), no step 0:

~~~text
fractional GKR of dimension n (per table, all tables in parallel):
    1 (mu_1, lambda_1 after the layer-1 claims) + sum_{k=1}^{n-1} (k sumcheck challenges + 1 (mu,lambda))
    = 1 + n(n-1)/2 + (n-1)                       n = 27 (T_range16, N = 111,673,344 + 65,536 <= 2^27):  378
checker GKR (6 layers, rounds s_out + 19 + 2 s_in = 47, 49, 49, 49, 49, 47):
    sum (rounds_i + 1) + 1 (gamma) = 296 + 1 = 297                                             (in parallel)
depth = 1 (message 2a: all z, beta, z_0) + max(378, 297) + 2 (Ligero r, Q) = 381
~~~

Without committed query wires every computed layer holds queries and waits for the lookups: depth
\(=1+\max(378,\,\text{assertion layer }48)+(5\text{ layers: }49+50+50+50+48)+1+2=630\).

**Proposals [proposal] (calculator `DepthOptions`), each with its cost:**

| reduction | mechanism | depth at \(B=4096\) | cost | status |
|---|---|---|---|---|
| \(c\) variables per message | prover sends the \(c\)-variate round polynomial (\((d+1)^c-1\) padded values); verifier binds \(c\) variables per message; per-message Schwartz–Zippel error \(cd/|F_c|\), total unchanged **[derivation]** | \(c=2\): 212; \(c=3\): 156 | pads \(\times(4^c-1)/(3c)\) (\(c=2\): \(2.5\times\), \(+7\)k pads); prover work in the sumcheck rounds \(\times\approx(d+1)^{c}/((d+1)c)\) (\(c=2\): \(\approx2\times\)) | folklore generalisation of the sumcheck round (no source read for it; derivation only) |
| table sharding \(s\) | split each table's \(N_T\) queries into \(2^s\) independent fractional GKRs with their own committed multiplicity vectors (challenge-independent); dimension \(\lceil\log_2(N_T/2^s+|T|)\rceil\); identity error charged per shard | \(s=5\): 300 (the checker chain, 297, is now the floor) | \((2^s-1)|T|\) extra committed multiplicities per sharded table (\(s=5\): \(+2.3\)M elements, \(+1.3\%\)); \(2^s\times\) the (padded) fractional transcript for that table (\(+3.3\) MB communication) | this lane's proposal |
| both | | \(c=2,s=5\): **158**; \(c=3,s=5\): 110 | sum of the above | |
| commit the 140 query wires (4.3) | default; listed here because it is the depth lever with the largest committed-element cost | 381 vs 630 | \(+55\)M committed elements (\(126.3\)M → \(181.4\)M, \(+44\%\)) | this lane's proposal |

Step 0 adds exactly one verifier message before com_1 (+1 depth) in every row.

**RTT sweep [derivation], \(K=1536,B=4096\), prover and verifier compute treated as given (\(t_{\text{compute}}\)),
network floor `depth × RTT` in seconds (`e2e_sweep`):**

| configuration | `rounds.protocol` | `rounds.sequential_depth` | `t.e2e.rtt_0ms` | `rtt_0.1ms` | `rtt_1ms` | `rtt_10ms` | `rtt_50ms` |
|---|---|---|---|---|---|---|---|
| A baseline (known techniques, query wires committed) | 1 589 | 381 | \(t_c\) | \(t_c+0.038\) | \(t_c+0.38\) | \(t_c+3.81\) | \(t_c+19.1\) |
| A without committed query wires (5 layers after the lookups) | 1 589 | 630 | \(t_c\) | \(+0.063\) | \(+0.63\) | \(+6.30\) | \(+31.5\) |
| A + \(c=2\) [proposal] | 1 589 | 212 | \(t_c\) | \(+0.021\) | \(+0.21\) | \(+2.12\) | \(+10.6\) |
| A + \(s=5\) [proposal] | 25 548 | 300 | \(t_c\) | \(+0.030\) | \(+0.30\) | \(+3.00\) | \(+15.0\) |
| A + \(c=2\), \(s=5\) [proposal] | 25 548 | 158 | \(t_c\) | \(+0.016\) | \(+0.16\) | \(+1.58\) | \(+7.9\) |
| A + \(c=3\), \(s=5\) [proposal] | 25 548 | 110 | \(t_c\) | \(+0.011\) | \(+0.11\) | \(+1.10\) | \(+5.5\) |
| A + step 0 (any row) | +1 | +1 | | | | | |
| B `helper_columns` (for comparison) | 7 | 3 | \(t_c\) | \(+0.0003\) | \(+0.003\) | \(+0.03\) | \(+0.15\) |
| B `gkr` mode | 1 569 | 381 | as A baseline | | | | |

At \(B=1\) and \(B=64\) the A baseline depth is 228 and 264 (`rounds_and_depth_a`): there the **checker chain**
(225, 261: six layers whose sumchecks have \(b=7,13\) copy variables) dominates the lookup chain (153, 231;
\(n_T=17,21\)). At \(B=4096\) the lookup chain dominates (378 vs 297). Whether the depth reductions are adopted is a
measurement question (`archive/AGENTS.md`: "no hard round budget yet").

### 5.5 Step 0: verifier-coin commitment for malicious-verifier privacy **[proposal]**

An implementation **without** this step is `COMPLETE_HVZK_BACKEND`, never `COMPLETE_ZK_BACKEND`.

**Commitment scheme.** HM96 first scheme with security parameter \(k_V=160\) for every coin slot \(i\) of
length \(\ell_i\) bits: V draws \(x_i\in\{0,1\}^{L_i}\), \(L_i=4k_V+2\ell_i+4\), a universal hash
\(h_i:\{0,1\}^{L_i}\to\{0,1\}^{\ell_i}\) with \(h_i(x_i)=\mathrm{coin}_i\), and sends
\(c_i=(h_i,\,H(x_i))\); the opening is \(x_i\) (the receiver recomputes \(\mathrm{coin}_i=h_i(x_i)\) and checks
\(H(x_i)\)). For long coins (the Ligero row-combination vector \(r\in F_c^{\mathrm{rows}}\), 630 KB; the query
set \(Q\)) the \(O(k)\) variant is used: commit to \(H(\mathrm{coin}_i)\) and open with \((\mathrm{coin}_i,x_i)\)
(HM96 section 4; 9k-bit commit string, 180 B). Statistically hiding: distance \(<2^{-k_V}\) per commitment
(HM96 Theorem 1), \(R\cdot2^{-160}=2^{-149.4}\) over the \(R=1589\) slots of the baseline (calculator term
`commitment_hash / verifier coin commitments`, statistical, **added to the soundness total**: it bounds what a
prover can learn about the coins). Computationally binding: two openings of one \(c_i\) are a collision of
\(H\) (HM96 section 3.2). Only collision resistance of \(H\) is used; no new assumption enters.

**What is committed.** Every coin of steps 2–5 in transcript order: for each table and redraw slot
\(z_T^{(i)}\), \(\beta_T\); \(z_0\); every \(\mu_k,\lambda_k\); every sumcheck challenge \(r_j\) (or the
\(c\)-tuple); every \(\alpha_i,\beta_i\) and lookup-merge coefficient; \(\gamma\); the Ligero \(r\) vector;
\(r_{add},r_q\) (as one uniform element \(\rho\) each when the powers form is used, section 6); \(Q\). The
slot list is a public function of \((K,B,C)\); its length is `rounds.protocol`.

**Transcript order.**

~~~text
 0   V -> P   (c_1, ..., c_R)                                  all coin commitments, before anything else
 1   P -> V   com_1                                            as 5.1 step 1
 i   V -> P   x_i  (opening of slot i, in the order of 5.1)    P checks H(x_i) = y_i, sets coin_i = h_i(x_i);
                                                               on failure P aborts (publicly attributable to V)
     P -> V   the prover message that depends on coin_i        never sent before the opening of coin_i verified
 ... every C line of 5.1 is such an opening; parallel slots (2a; the 5 tables' round-j coins) open together
 5d  P -> V   opened columns                                   after the opening of Q verified
~~~

**Binding requirement and why it suffices against an adaptive verifier.** A verifier who wants coin \(i\) to
depend on the prover's messages must open \(c_i\) to a value other than the one fixed in step 0, i.e. find
\(x_i'\ne x_i\) with \(H(x_i')=H(x_i)\). Under collision resistance this happens with negligible probability,
so every coin the prover accepts equals the one sampled before com_1: the verifier's only remaining freedom is
**whether** to open the next slot (abort) — the sampled coins themselves are a fixed function of the verifier's
initial randomness and are independent of every prover message. Nothing about the *distribution* of the
verifier's coins is required for privacy (a malicious verifier may pick them arbitrarily); privacy needs only
that they are fixed before the first prover message. Soundness needs the coins to be unpredictable to the
prover, which is what statistical hiding gives.

**Abort / restart rules.**
- P aborts iff an opening fails to verify (\(H(x_i)\ne y_i\)), a coin is malformed (not in its domain), or all
  \(S\) redraw slots of some table lie in \(\mathrm{Img}_T\). Each condition is computed from public data
  (\(c_i,x_i\), tables, \(\beta_T\)) only; none reads the witness. The prover **never** decides to abort on any
  other condition and never restarts within a session.
- V aborts iff a prover message fails a check. A new session recommits with fresh pads, randomizers and fresh
  coins; sessions are counted by the outer accounting (handoff 7.3).
- No restart-on-bad-challenge exists: the redraw of 2a is resolved inside the pre-issued slots.

**Privacy argument (sketch; the proof is an unproved obligation, section 10).** Let \(V^*\) be any verifier.
The view of \(V^*\) is (com_1, then padded/blinded prover messages, then \(t\) opened columns) for the coins it
committed to, or a prefix of it if it aborts. Simulator: (1) run \(V^*\) to obtain \((c_1..c_R)\); (2) send a
simulated com_1 (a commitment to a random tableau: rows of random symbols — indistinguishable from the real one
by the statistical hiding of the HM96 leaves and the uniformity of the \(t\) opened columns, AHIV22 Lemma
4.15); (3) receive \(V^*\)'s openings slot by slot; if \(V^*\) aborts, output the prefix; else all coins are now
known; (4) rewind \(V^*\) to after step 0, run the **honest-verifier simulator** of the underlying protocol
(AHIV22 Lemma 4.15 for the tableau and tests, FS24 Theorem 6 for the padded sumcheck transcript: every padded
value is uniform, the deferred constraints are satisfied by construction of the simulated pads) on the known
coins to produce com_1' and all prover messages, and replay; \(V^*\) must open the same coins (binding) or
abort; (5) if it aborts in the replay but not in the first run, repeat step 4 with fresh simulator randomness;
the expected number of repetitions is bounded as in Goldreich–Kahan (the abort probability depends on com_1
only through a statistically hiding commitment, so it differs negligibly between the first run and the
replay). Inherited from the HVZK proof: everything except steps (1)–(3) and (5); the new ingredients are the
binding of HM96 (computational) and the hiding of com_1 against the abort decision. The resulting notion is
computational zero knowledge under collision resistance, with statistical loss \(n\cdot2^{-160}\) (leaves) +
Ligero's \(0\) (Lemma 4.15 is perfect for \(k>l+t\)).

**Costs (\(K=1536,B=4096\), calculator `census_estimate_a(coin_commitment_bits=160)`):**

| item | value |
|---|---|
| additional verifier messages / depth | +1 (message 0); every other opening rides on the challenge message it replaces |
| additional communication | \(R\) commit strings + \(R\) openings \(\approx R\cdot(144+100)\) B \(=0.39\) MB for \(R=1589\) (74.7 MB → 75.1 MB, \(+0.5\%\)); long coins (\(r\), \(Q\)) add only their \(x_i\) |
| additional prover work | verify \(R\) openings: \(R\) hashes of \(\le 900\)-bit strings and \(R\) universal-hash evaluations — \(<10^7\) bit operations, below \(10^{-6}\) of the batch's \(1.2\cdot10^{11}\) field operations (section 9.4); no change to encoding, GKR or Ligero work |
| additional verifier work | sample coins up front (unchanged total), \(R\) commitments |
| soundness | \(+R\cdot2^{-160}=2^{-149.4}\), total unchanged at \(2^{-130.2}\) (`test_coin_commitment_step0_terms_and_label`) |

## 6. Constraint generation (libzk 6.7 adapted)

Public inputs to constraint generation: all challenges, all padded values, \(\widetilde Q_i\) at the challenge
points, table evaluations \(\widetilde t_v(\beta_T)\) at \(\rho_T\). Output: matrices \(A\) (linear) and triples
(quadratic) over \(F_c\) on the committed vector \(W_1\), expanded to \(F_t\) coordinates. Both parties compute
them; the prover needs them only for step 5b. Combination coefficients: \(r\) (IRS test) is an explicit uniform
vector; \(r_{add}\), \(r_q\) may be explicit (A-GKR has \(\approx10\)k linear and \(\approx1\)k quadratic
constraints) or powers of one uniform element each (`ligero_error(combination="powers")`, error
\((\#\text{constraints}-1)/|F_c|\) by Schwartz–Zippel **[derivation]**; required for B-Ligero).

## 7. Hash properties assumed, and where

| property of \(H\) | used where | consequence if it fails |
|---|---|---|
| **collision resistance** (family chosen publicly before step 1, per AHIV22 5.1) | Merkle tree over HM96 leaves: binding of com_1; step 5d openings; binding of the step-0 coin commitments | soundness: a prover opening inconsistent columns (`commitment_hash` term, generic \(q^2/2^h\) for \(q\) hash evaluations — a modelling choice, section 9); ZK with step 0: a verifier re-opening a coin |
| **hiding of unopened columns** | HM96 leaf commitments (default, section 3): statistical \(<2^{-160}\) each, from collision resistance alone | ZK only; if the unsalted alternative \(H(\text{column})\) is used instead, a hiding assumption on \(H\) beyond CRH is required and must be stated by the implementation |
| **PRG** (hash-based expansion of a seed into pads, row randomizers, blinding rows, HM96 \(x_j\) on the GPU) | prover randomness generation | ZK only; at the ideal level all randomness is uniform (handoff 7.4). Recorded as a separate assumption |
| statistically hiding, computationally binding commitment for verifier coins (step 0) | HM96 from the same \(H\) | hiding failure → soundness (prover learns coins; charged \(R\,2^{-160}\)); binding failure → ZK (verifier adapts coins) |
| **not assumed**: random oracle, correlation intractability, extractability of \(H\) | — | Fiat–Shamir is not used; knowledge soundness of the argument is via AHIV22's extractor for the interactive variant, which uses collision resistance only (section 10 records it as unproved for *this* composition) |

## 8. Leakage rules

| leak channel | rule |
|---|---|
| branch-dependent message shapes (e.g. shift direction, zero/overflow flags) | every checker branch is fixed-size (selectors + both arms evaluated); number and degree of every sumcheck message is a function of \((K,B,C)\) only |
| batch partitioning by exponent / group state | forbidden; one batch, one GKR, one commitment; padding instances are public and prescribed (handoff 9.1). Sub-batches, if used, are a public split with `N_subbatches` reported and union-bounded |
| lookup multiplicities | committed, never sent; only \(\widetilde m_T(\rho)\) padded |
| logUp pole (\(z_T-w_j=0\)) | honest query values lie in \(T\), so a pole exists iff \(z_T\in\mathrm{Img}_T\): a **public** condition resolved by the pre-issued redraw slots (2a); the transcript records which slot was used deterministically; the prover never aborts on witness data (calculator `completeness`: redraw \(\sum_T|T|/|F_c|\approx2^{-176}\), abort \(\sum_T(|T|/|F_c|)^2\approx2^{-352}\)) |
| any other zero denominator | none exists (section 4.4) |
| restart on a bad challenge | forbidden. A session is aborted, never restarted with the same W_1; a new session recommits with fresh pads and randomizers (and the outer protocol counts sessions, handoff 7.3) |
| timing / message length | all messages fixed length; prover work per round data-independent (fixed-shape kernels) |
| verifier-adaptive challenges | with step 0, challenges are fixed before com_1 (5.5); without it, only honest-verifier ZK is claimed (`COMPLETE_HVZK_BACKEND`) |
| Merkle root of a deterministic layout | rows randomized (\(k-l\) fresh symbols); leaves are HM96 commitments (section 3) |
| Ligero responses \(v,q_{add},p_0\) | blinded (section 5.2); the linear/quadratic response polynomials have degree \(<k+l-1\), \(<2k-1\) and hide the payload by the blinding rows' randomness |

## 9. Soundness error decomposition (calculator output)

Soundness is a parameter (`archive/AGENTS.md`): the per-proof target is derived from the application budget
(`per_proof_target`), sub-batches are union-bounded (`union_over_proofs`). All figures: \(K=1536\), Goldilocks
trace field, Ligero \(n=2^{14},k=2^{12},l=3840\), hash 256-bit, adversary \(q=2^{60}\) hash evaluations unless
stated. Tests in `benchmarks/numerical/tests/security/test_accounting.py`.

### 9.1 Reference decomposition, \(B=4096\)

| component | term | \(2^{-128}\) set: \(F_c=F_{p^3}\), \(t=192\), \(q=2^{60}\) | \(2^{-80}\) set: \(F_c=F_{p^2}\), \(t=121\), \(q=2^{64}\) |
|---|---|---|---|
| batching | \(z_0\) Schwartz–Zippel \((s_0+b)/|F_c|\); \(\alpha,\beta\) per layer; \(\gamma\); 5 lookup merges | \(2^{-186.7}\) | \(2^{-122.7}\) |
| sumcheck | \(\sum_i(2s_i+3b+4s_{i+1})/|F_c|\), 290 rounds (6 census layers) | \(2^{-182.6}\) | \(2^{-118.6}\) |
| lookup | \(\sum_T\big[(N_T+|T|-1)(\tfrac1{|F_c|-|T|}+\tfrac{\mathrm{cols}-1}{|F_c|}) + \tfrac{2(n_T-1)+1+\sum_{k<n_T}3k}{|F_c|}\big]\), conditioned challenge space; `T_range16` \(N=111.7\)M dominates | \(2^{-164.9}\) | \(2^{-100.9}\) |
| encoding/opening | \((1-e/n)^t+((k+l)/n)^t+(2k/n)^t+(n+3)/|F_c|\) (AHIV22 App. C, FS24 §2.2) | \(2^{-130.2}\) | \(2^{-82.0}\) |
| commitment/hash | \(q^2/2^{256}\); with step 0 also \(R\cdot2^{-160}=2^{-149.4}\) | \(2^{-136}\) | \(2^{-128}\) |
| **total** | | **\(2^{-130.2}\)** (\(B\in\{1,64,4096\}\) all pass) | **\(2^{-82.0}\)** |
| zk (not soundness) | HM96 leaves \(n\cdot2^{-160}\) | \(2^{-146}\) | \(2^{-146}\) |
| completeness (not soundness) | redraw \(\sum_T|T|/|F_c|\); abort \(\sum_T(|T|/|F_c|)^2\) | \(2^{-176.0}\); \(2^{-352}\) | \(2^{-112.0}\); \(2^{-224}\) |

The WP2 census moved the lookup term by \(2^{-0.1}\) (`T_range16` \(N\): 18.1M → 111.7M, but the 13 other
relation tables are gone) and the sumcheck term by \(2^{+0.8}\) (290 rounds instead of 154); no operating point
moved (9.2).

### 9.2 Operating points: which configuration first meets each (`first_configuration`, both candidates identical)

Sweep order: extension degree 2, then 3, then 4; within a degree the smallest \(t<k-l\). \(q=2^{60}\).

| operating point | first configuration | \(B=1\) | \(B=64\) | \(B=4096\) | why the smaller extension fails |
|---|---|---|---|---|---|
| \(2^{-80}\) (diagnostic) | **quadratic** \(F_{p^2}\), \(t=120\) | \(2^{-81.4}\) | \(2^{-81.4}\) | \(2^{-81.4}\) | — |
| \(2^{-96}\) (sweep) | **quadratic** \(F_{p^2}\), \(t=144\) | \(2^{-97.6}\) | \(2^{-97.6}\) | \(2^{-97.5}\) (A), \(2^{-97.2}\) (B) | — (lookup \(2^{-100.9}\) at \(B=4096\) still below) |
| \(2^{-112}\) (sweep) | **cubic** \(F_{p^3}\), \(t=167\) | \(2^{-113.2}\) | \(2^{-113.2}\) | \(2^{-113.2}\) | quadratic: lookup identity \(2^{-111.3}\) (\(B=1\)) … \(2^{-100.9}\) (\(B=4096\)) alone exceeds the target |
| \(2^{-128}\) (primary) | **cubic** \(F_{p^3}\), \(t=191\) (reference set \(t=192\)) | \(2^{-129.5}\) | \(2^{-129.5}\) | \(2^{-129.5}\) | quadratic: same floor, \(2^{-111.3}\) at best |

The quadratic Goldilocks extension cannot reach \(2^{-128}\) at any \(B\): the lookup identity term is set by the
\(65\,536\)-row range table (\(d_T/(|F_c|-|T|)\ge 2^{16}/2^{128}\)). The WP2 census did not move any first
configuration (`test_first_configuration_per_operating_point`, unchanged parameters). Rounds and committed elements scale linearly in
\(U\); soundness terms only logarithmically. The hash term is a modelling choice (adversary work \(q\)),
reported separately as `computational`; `security_bits(field)` is refused.

### 9.3 Sub-batches

A batch of 4096 proved as `N_subbatches` = 16 independent proofs of 256 multiplies every term by 16: a per-proof
\(2^{-130.2}\) becomes \(2^{-126.2}\) and misses \(2^{-128}\) (`test_sub_batches_are_union_bounded_mechanically`);
the per-VU cost includes every sub-batch.

### 9.4 Census, \(K=1536\) (`census_estimate_a` / `census_estimate_b`; **every row marked (est.) is an estimate
from the stated cost model**, `archive/AGENTS.md` columns)

Cost model (est.): \(F_{p^3}\) multiplication = 6 \(F_t\) multiplications; NTT butterfly = 1 mult-equivalent;
checker GKR 12 \(F_c\) mults per (wire, copy, layer); fractional GKR 15 \(F_c\) mults per leaf-equivalent;
Ligero tests \(3\cdot\mathrm{rows}\cdot n\cdot\deg\); memory passes: 3 over the committed matrix (read witness,
write codewords, hash) + 2 per sumcheck table; peak memory under the *retained* policy (codewords + witness +
largest live sumcheck table). B's rows: helpers by batch inversion, 3 \(F_c\) mults each, 2 passes.

Per-unit inputs to the model (WP2 census, section 1): A commits \(321+140\) elements per unit and computes 344
wires; B commits 1 064 per unit (3 aligned slots per `mul` + one per other column; layout **estimate**) plus 306
helper columns per unit in \(F_c\) (helper-column mode), with \(137+2\cdot358=853\) linear and 358 quadratic
constraints per unit. Primary configuration \(F_c=F_{p^3}\), \(t=192\) (\(2^{-128}\)); the \(2^{-80}\) diagnostic
(\(F_{p^2}\), \(t=121\)) is the last row group.

| column | A \(B=1\) | B \(B=1\) | A \(B=64\) | B \(B=64\) | A \(B=4096\) | B \(B=4096\) |
|---|---|---|---|---|---|---|
| materialized witness elements (est.; A: inputs + all wires + fractional-tree nodes; B: committed + helpers) | 1.68M | 0.46M | 31.4M | 12.4M | 2.01G | 780M |
| committed elements (`census.committed_elements`) | 116k | 457k | 2.91M | 12.4M | **181M** | **780M** |
| bytes encoded (`census.bytes_encoded`) | 4.5 MB | 16 MB | 100 MB | 425 MB | 6.19 GB | 26.6 GB |
| hash/commit work (`census.hash_bytes`, est.) | 7.9 MB | 23 MB | 103 MB | 432 MB | 6.19 GB | 26.6 GB |
| field operations (`census.field_ops`, \(F_t\) mult-eq., est.) | 45M | 39M | 1.90G | 1.03G | 121G | 64G |
| memory passes / bytes moved (`census.memory_passes` / `census.bytes_moved`, est.) | 25 / 43 MB | 5 / 56 MB | 25 / 1.53 GB | 5 / 1.47 GB | 25 / 97 GB | 5 / 92 GB |
| sequential challenge depth (`rounds.sequential_depth`; `rounds.protocol`) | 228; 579 | 3; 7 | 264; 968 | 3; 7 | 381; 1 589 | 3; 7 |
| communication (`transcript.bytes`, est.; P→V / V→P) | 0.99 MB (0.98 / 0.015) | 1.18 MB (1.18 / 0.003) | 2.16 MB (2.12 / 0.04) | 6.05 MB (5.97 / 0.08) | 74.7 MB (73.5 / 1.2) | 318 MB (313 / 4.9) |
| peak memory (retained, est.) | 6.9 MB | 20 MB | 223 MB | 525 MB | 14.0 GB | 32.8 GB |
| \(2^{-80}\) diagnostic (\(F_{p^2}\), \(t=121\)): committed / bytes encoded / field ops / peak | 114k / 4.3 MB / 25M / 6.3 MB | 361k / 12.7 MB / 26M / 15.6 MB | 2.90M / 100 MB / 1.02G / 190 MB | 10.5M / 359 MB / 0.71G / 443 MB | 181M / 6.19 GB / 65G / 11.9 GB | 659M / 22.5 GB / 45G / 27.8 GB |

Against the relation-based table (101M vs 405M at \(B=4096\)): A's committed elements rose \(1.8\times\) (321 + 140
query wires per unit instead of the assumed 256), B's \(1.9\times\) (1 064 slots + 306 helpers per unit instead of
512 + 172). The committed-elements ratio A:B is \(1:4.3\) at \(B=4096\) (\(1:4.0\) before), \(1:6.2\) without the
committed query wires (126M vs 780M, depth 630); it stays an explanatory variable of the encoding, hashing,
communication and memory columns. The field-operation column now favours B (A: \(2\times\) B): the checker GKR
runs over \(\approx3\,300\) carried values per unit per layer under the carry-everything width bound of section 1,
which is an upper bound a real layout would shrink; the depth column separates them in the other direction
(381 vs 3). No verdict is drawn here.

## 10. Unproved obligations (verbatim list for the coordinator)

1. Knowledge soundness of the interactive composition (data-parallel GKR + logUp-GKR + padded transcript +
   Ligero): FS24 Theorem 6 covers a plain layered circuit with the Ligero extractor; the fractional-GKR
   sub-transcripts, the lookup-claim merging of 4.3 and the \(F_t\)/\(F_c\) coordinate expansion are not covered.
2. Simulation (zero knowledge): only honest-verifier ZK is inherited (AHIV22 Lemma 4.15, FS24 Theorem 6). The
   verifier-coin-commitment step 0 (5.5) for malicious-verifier ZK is proposed with a simulator sketch, not
   proved, for this protocol; the Goldreich–Kahan expected-time analysis is cited, not read; its cost is
   estimated (5.5), not measured. An implementation without step 0 is `COMPLETE_HVZK_BACKEND`.
3. If the unsalted leaf hash is used instead of HM96 leaves (section 3), hiding of the Merkle leaves is an
   assumption on \(H\) beyond collision resistance.
4. The proximity-gap / IRS statement for a subfield RS code (\(F_t\)) tested with combination coefficients in
   the extension \(F_c\): the calculator charges the Appendix C field terms over \(|F_c|\); the corresponding
   lemma for this exact setting is not cited. When \(r_{add},r_q\) are powers of one element, the Schwartz–Zippel
   replacement of the linear/quadratic field terms is this lane's derivation, and the IRS test keeps uniform
   coefficients.
5. Extraction/composition with Verity's outer commitments (handoff 7.5): W_1 is a fresh commitment to the
   operands; equality with the values fixed by Verity's earlier commitments is a separate proof obligation, and
   the outer draft's extraction/composition requirements for the numerical backend remain unproved.
6. Integer semantics inside \(F_t\) (handoff 7.2): every range bound that makes a field congruence an integer
   equality is a checker obligation (WP2), not established here.
7. Application-level accounting: the number of sessions/batches an adversary may open multiplies the error
   budget (handoff 7.3); `per_proof_target` computes the per-proof allowance once that number is fixed.
8. The checker shape (`DEFAULT_LAYERS_A`) is derived from the WP2 census with carry-everything layer widths (an
   upper bound; the true widths depend on the WP3 layout); the layout rule of 4.3 (commit the 140 query wires) is a
   proposal with a stated cost, and the depth without it (630) applies if it is not adopted. The census itself
   counts the checker's *reference* realisation (`lt_pow` as a product, operand decode by hints); a backend that
   realises `Bf16Decode` by a 65 536-row table or shares the duplicated `q*sr` products changes the counts.
9. The depth reductions of 5.4 (\(c\) variables per message; table sharding) are this lane's proposals: their
   soundness derivations are in the table, their prover-cost factors are estimates, and neither has been measured.
10. The census figures of 9.4 marked (est.) come from the stated cost model, not from measurement.

## 11. Open items (questions, not decisions)

- Sequential depth: baseline 381 at \(B=4096\) (network floor 0.38 s at 1 ms RTT, 19.1 s at 50 ms) against
  B-Ligero's 3; the proposals reach 158 (\(c=2,s=5\)) at \(+1.3\%\) committed elements and \(\approx2\times\)
  sumcheck-round prover work; the checker chain (297) is the floor of sharding alone. The measurements decide; no
  round budget is set.
- Whether to commit the 140 computed query wires per unit (depth 381, 181M committed) or not (depth 630, 126M).
- Whether the checker should realise `Bf16Decode` by one 65 536-row lookup per operand (the relation's `T_bf16`)
  instead of 4 hints + 3 range checks + 3 `mul` + 2 `lin`: per unit −96 `mul`, −96 range chunks, −64 `lin`,
  −32 committed columns, at the cost of a second 65 536-row multiplicity column and 32 more fixed-table
  lookups — a checker-lane question, not decided here.
- Whether the first complete backend ships as `COMPLETE_HVZK_BACKEND` (no step 0) with 5.5 as the documented
  path to `COMPLETE_ZK_BACKEND`.

## 12. The VU chain: 96 units + epilogue as one proof (implemented 2026-09-22, lane a-chain)

The Verity VU is `GemmCoordinate<1536>`: 96 word-chained transition units from the constant \(c_0=0\), then the
epilogue `Epilogue.check` (RN-even cast of the last FP32 word to the BF16 output word). Its public endpoints are
\(c_0=0\) and the output word \(y16\); the 95 intermediate words and every hint are private. Three ways to wire
the 96 units of one VU into the data-parallel GKR of section 4, with their costs at \(K=1536, B=4096\)
(\(U=393\,216\) units, \(468\) exported columns per unit, \(826\) wires, 6 checker layers) **[derivation]**:

| design | sequential depth (checker chain) | committed elements / VU | prover work (wire evaluations per layer sweep; phase-1 tables) | new code |
|---|---|---|---|---|
| (a) **deep circuit**: one circuit of 96 unit-blocks + epilogue, \(6\cdot96+2\) strict layers, data-parallel over \(B\) VUs (\(b=12\)) | \(\approx 578\) layers \(\times\) 47 rounds \(=27\,075\) | \(44\,747\) (the 95 intermediate words are wires, not columns: \(-190\)) | \(578\times4096\times79\,296=187\)G wire evaluations (strict layering carries every value of lower depth: width \(96\times826\)); tables 15.6 GB | new exporter for a 45k-column circuit, per-VU wiring tables |
| (b) **data-parallel units + linkage** *(implemented)*: \(U\) unit copies and \(B\) epilogue copies over **one** commitment; the linkage \(c_{v,0}=0\), \(c_{v,i}=y_{v,i-1}\), \(y32_v=y_{v,95}\) and the public words \(y16_v=\) recorded word are **linear constraints on the committed vector**, batched into the Ligero opening | \(+0\): checker chain 243 (6 layers, \(b=19\)), lookup chain 435 (`R16` has \(2^{28}\) leaves, below) — depth is set by the lookups exactly as before | \(96\cdot468+9=44\,937\) (the words are already the `x.c` / `y` columns of each unit; \(+9\) epilogue columns) | \(6\times393\,216\times826=1.95\)G; tables 15.6 GB; one extra \(O(U)\) pass for the linkage functional; the epilogue GKR (11 wires, 2 layers, \(B\) copies) is negligible | epilogue circuit export; multi-segment instance; one functional; no change to `gkr.rs`/`logup.rs` semantics |
| (c) **checkpoints every \(m\)**: commit every \(m\)-th word, prove the \(m\) units in between as a deep sub-circuit (\(6m\) layers) data-parallel over \(U/m\) copies | \(m=2\): 495; 4: 1 011; 8: 2 067; 16: 4 227; 32: 8 643; 96: (a) | \(44\,937-2\cdot96(1-1/m)\): \(m=2\): 44 841; 8: 44 769; 96: 44 747 | wire evaluations \(\times m\) (strict layers carry the sub-chain's \(m\times826\) values): 3.9G (m=2) … 187G (m=96) | as (a) plus a checkpoint layout |

The decisive fact is that the committed *word* is 1 element in a row of 468: checkpointing saves at most
\(190/44\,937=0.42\%\) of the committed elements and costs \(m\times\) the checker's prover work and depth.
(b) is the \(m=1\) point of (c), it needs the least new code, and its soundness argument is the one the Ligero
opening already carries. **(b) is implemented first**; (a)/(c) are not built (section 12.4 reports their cost
model; no measured point is expected to beat \(m=1\)).

### 12.1 Layout **[proposal, implemented]**

Two data-parallel *segments* over one committed vector \(X\):

~~~text
X = [ unit rows: U = B·96 rows × 468 columns, row v·96+i = unit i of VU v (VU-major) ]
    [ epilogue rows: B rows × 9 columns: one, y32, y16, hi, lo, hi2, lsb, carry, rem ]
    [ multiplicities of ALIGN, POW, LEAD, NORM, R16 over BOTH segments ]
~~~

Segment 0 is the exported unit circuit (section 1), segment 1 the exported epilogue circuit
(`gkr_export.vu.build_epilogue_circuit`: `Epilogue.check` verbatim — 5 sixteen-bit range chunks into `R16`
(`hi`, `lo`, `hi2·2`, `rem`, `hi+carry`), 2 booleans, 3 linear identities, and the assertion \(y16=hi+carry\)).
Each segment runs its own checker GKR (its own \(z_0,\gamma\), \(b_0=\lceil\log_2 U\rceil\), \(b_1=\lceil\log_2 B\rceil\));
the two are independent and run in lock-step (one verifier message carries both segments' round challenges), so
the checker depth is the unit segment's. The five logUp instances take the witness side of a table from both
segments concatenated (unit queries, then epilogue queries, then the table rows, then padding): the epilogue's
five `R16` queries per VU join the range-check multiset; `R16` therefore has \(352U+5B+65\,536=138\,498\,048\)
leaves at \(B=4096\), i.e. \(n_T=28\) (the census's 284 range queries per unit gave \(n_T=27\); the exporter's
352 come from two one-sided checks per non-power-of-two range and the materialised query wires).

### 12.2 Linkage and public endpoints as linear constraints **[proposal, implemented]**

Every claim the verifier is left with after the GKRs is a linear functional of \(X\) (section 4.3 layout rule:
input claims, lookup leaf claims). The chain adds \(U+2B\) more, all with **public** coefficients:

\[
X[(v,0),\,c]=0,\qquad X[(v,i),\,c]-X[(v,i-1),\,y]=0\ (1\le i<96),\qquad
X_{\mathrm{epi}}[v,\,y32]-X[(v,95),\,y]=0,\qquad X_{\mathrm{epi}}[v,\,y16]=\mathrm{word}_v .
\]

They are combined with geometric coefficients \(\rho_{\mathrm{geo}}^k\) (\(k\) the constraint index) into one
functional, which is batched with the \(2\cdot2+5\) GKR/lookup functionals by fresh \(\rho_f\) and opened once
(section 3, `ligero::prove_open`). Transcript: the \(\rho_f\) and \(\rho_{\mathrm{geo}}\) are drawn after the last
GKR message; the public words are absorbed into the transcript before any challenge (they are part of the
statement). Soundness **[derivation]**: if any linkage or endpoint constraint fails, the combined value is a
nonzero polynomial of degree \(<U+2B\) in \(\rho_{\mathrm{geo}}\): error \((U+2B-1)/|F_c|=2^{-173.4}\) at
\(B=4096\) over \(F_{p^3}\), plus \(1/|F_c|\) for its batching coefficient; the Ligero linear test then binds the
batched functional to the committed rows as before (the `encoding_opening` term is unchanged). Committing the
words twice (as `y` of unit \(i\) and `x.c` of unit \(i+1\)) is what makes the linkage linear; a state-chained
variant (pass \((s,e,M)\)) would save the decode/pack of the word (\(\approx4\%\), section 1) but is not what
`step.py` composes and is not built.

**Endpoints.** \(c_0=0\) is a constraint with constant 0 (no committed constant column is needed); the output words
are verifier inputs (\(B\) sixteen-bit words) and the only witness-independent data besides the circuits and the
batch shape. A `vu-k1536-neg` instance is a *claimed* word: with honest hints the epilogue's \(y16\) column must
equal the cast of \(c_{96}\) (assertion `y16`) **and** the public word (endpoint constraint), so the prover has no
accepting witness; an out-of-domain negative (non-finite operand, saturated intermediate) is rejected by the
unit circuit's range checks (`acc.t < 255`, `a[i].t < 255`) or already by `gen_step`/`check_step` at witness
generation.

### 12.3 Depth, rounds, what changed in the transcript

`rounds.sequential_depth` \(=1+\max(\text{lookup chain},\text{checker chain})+2\) (message 2a; the Ligero \(r\)
and \(Q\) coins; the batching coefficients ride on the \(r\) message). At \(B=4096\): lookup chain 435 (\(n_T=28\)),
checker chain 243 → **438**; the linkage adds no round trip. `rounds.protocol` grows by the epilogue segment's
sumcheck slots and by \(2\cdot2+5+1+1=11\) batching coins. Sharding `R16` (\(s=1\), section 5.4) would return
\(n_T\) to 27 (depth 409) for \(+65\,536\) committed multiplicities; not done.

### 12.4 Checkpoint interval \(m\): cost model (no measured points; 12.5 has the measured \(m=1\))

\(m\) = units between committed words (\(m=1\): every word committed = design (b); \(m=96\): design (a)).
Strict layering (libzk 6.3) carries every value of lower depth, so a sub-chain of \(m\) units is a layer of width
\(m\times826\); copies \(=U/m\). **[derivation, cost model of section 9.4]**

| \(m\) | committed / VU | checker layers | copies (\(b\)) | width / layer | rounds / layer | checker depth | wire evaluations | phase-1 tables |
|---|---|---|---|---|---|---|---|---|
| 1 | 44 937 | 6 | 393 216 (19) | 826 | 40 | 243 | 1.95G | 15.6 GB |
| 2 | 44 841 | 12 | 196 608 (18) | 1 652 | 41 | 495 | 3.90G | 15.6 GB |
| 4 | 44 793 | 24 | 98 304 (17) | 3 304 | 42 | 1 011 | 7.80G | 15.6 GB |
| 8 | 44 769 | 48 | 49 152 (16) | 6 608 | 43 | 2 067 | 15.6G | 15.6 GB |
| 16 | 44 757 | 96 | 24 576 (15) | 13 216 | 44 | 4 227 | 31.2G | 15.6 GB |
| 32 | 44 751 | 192 | 12 288 (14) | 26 432 | 45 | 8 643 | 62.4G | 15.6 GB |
| 96 | 44 747 | 576 | 4 096 (12) | 79 296 | 47 | 27 075 | 187G | 15.6 GB |

Committed elements move by \(<0.5\%\); prover work and depth grow linearly in \(m\). No point of the sweep
changes the picture, so none is implemented; the recorded runs of 12.5 are all \(m=1\).

### 12.5 Recorded runs

vy-cpu2, 12 threads, PCS mode **[measured]**: B=64 r20260922-053451-03f9 (6.33 s, depth 279); B=4096
r20260922-055358-9f6b (headline, commit 2229c4b): prove 297.6 s, verify 33.0 s, 69.6 MB, `rounds.protocol` 1923,
`rounds.sequential_depth` **438** (the 12.3 prediction), peak RSS 33.9 GB, 184,128,575 committed elements, all 4096
recorded words accepted; first B=4096 run r20260922-053757-3b76 (293.4 s, overlapped a duplicate launch).
Negatives: `vu-k1536-neg` 52/52 rejected (8 at witness generation, 44 by the verifier; r20260922-054550-62b1),
transition-unit `fixtures/negatives` 52/52 rejected (r20260922-055135-ed34). Table, labels and the per-unit
comparison: `README.md` "Full VU runs".

## 13. BabyBear instantiation (implemented 2026-09-22, lane a-babybear; `--features babybear`)

The same protocol (sections 1–4, 12) over the trace field \(F_t=\) BabyBear \(p=2^{31}-2^{27}+1=2\,013\,265\,921\)
with the challenge field \(F_c=F_p[X]/(X^6-22)\) (\(|F_c|\approx2^{185.4}\)). Nothing in the relation changes
(section 1); the *embedding* of the 109 WP2 predicates into a 31-bit field does (13.2), and the field forces three
byte-level protocol changes (13.4). The Goldilocks build (`field.rs`) is untouched: `src/field_babybear.rs` exposes
the same `Fp`/`Ext` API and `#[path]`-swaps in behind the feature; `gkr.rs`, `logup.rs`, `ligero.rs`, `transcript.rs`
and `main.rs` are field-generic (constants `field::{P, DEGREE, NAME, EXT_NAME, DOMAIN, FIXTURES}`, `Fp::BYTES`,
`Fp::rejection_sample`, `Fp::root_of_unity`).

### 13.1 Field and extension **[source: `tensor/reference.py`, `field_babybear.rs` tests]**

| | Goldilocks (sections 2, 12) | BabyBear (this section) |
|---|---|---|
| \(p\) | \(2^{64}-2^{32}+1\) | \(2^{31}-2^{27}+1\); \(p-1=2^{27}\cdot3\cdot5\), two-adicity 27 (RS length up to \(2^{27}\); \(n=2^{14}\) used) |
| base element | canonical `u64`, 8 bytes LE | canonical `u32` in \([0,p)\), **4 bytes LE** |
| extension | \(X^3-X-1\), degree 3 | \(X^6-22\), degree 6; 22 is the smallest non-square non-cube of \(F_p\) (\(2,3\mid p-1\), so \(X^6-w\) is irreducible iff \(w\) is neither: `tests::w_is_the_smallest_non_square_non_cube`), the polynomial `archive/gkr/tensor/reference.py` (`_find_w()`) and the a-gpu lane use, so the two provers agree coordinate by coordinate |
| extension element | \((c_0,c_1,c_2)\), 24 bytes | \((c_0,\dots,c_5)\), \(x=\sum c_iX^i\), **24 bytes** (6 × 4): the same proof-message size |
| multiplication | schoolbook + reduction by \(X^3=X+1\) | schoolbook (36 base products, `u128` accumulation, one reduction) + \(X^6=22\) |
| generator / roots of unity | 7 | 31 (\(31^{(p-1)/2}=-1\), \(31^{(p-1)/3}\neq1\), \(31^{(p-1)/5}\neq1\)); \(\omega_{2^k}=31^{15(2^{27-k})}\) |
| challenge sampling | 8-byte words of the SHA-256 digest, reject \(\ge p\) | 4-byte words of the digest, reject \(\ge p\) (rejection rate \(1-p/2^{32}=6.25\%\)); 6 coordinates per challenge, each its own digest (`ctr` increments per coordinate) |

Unit tests pin `Fp`/`Ext` against values generated by `tensor/reference.py` (`ext_mul_matches_reference_py`,
`field_basics_against_python`, `roots_of_unity`).

### 13.2 Where the six rules land in the circuit export (`gkr_export.babybear`, `--field babybear`) **[implemented]**

`note:r20-proof/red-team-babybear/20260922T0605Z-report-babybear-rules` derives the rules from the predicate intervals; the exporter realises each as columns, gates and
lookups of the layered circuit. `BabyBearCircuitCtx` is `CircuitCtx` with the window discipline: a range
\([lo,hi)\) of \(n\) values that is not a power of two is two one-sided \(w\)-bit chunk checks (\(x-lo\) and
\(hi-1-x\)), **sound iff \(2^{w+1}-n\le p\)** (an \(x-lo=t\in[n,2^w)\) makes \(hi-1-x\) read as
\(p-(t-n+1)\ge p-(2^w-n)\ge2^w\)) — a range without a sound form is refused at export time, none occurs in the unit.

| rule | predicate(s) | in the circuit | census effect per unit |
|---|---|---|---|
| **LIMB16** | `fp32_decode.acc.word`, `pack.y`, `epilogue.cast.split` | the word columns `x.c`, `y` (unit) and `y32` (epilogue) become limb pairs `_lo`/`_hi` (public, 16-bit ranged); hints `acc.f_lo/f_hi` (16 + 7 bit) and `pack.low_lo/low_hi` replace the 23-/32-bit identities by per-limb identities; the epilogue's `hi`/`lo` hints *are* the limbs (−2 hints, −1 identity, −2 ranges) | +4 hint columns, +2 public columns; chain linkage is per limb (`Chain.limbs`, 13.3) |
| **LT_DIRECT** | `align_state.rem/quot`, `align_prod.rem/quot`, `lead.window` | the variable bound \(x<2^B/\mathrm{pow}\) is `rng(bnd - 1 - x, 0, 2^B)` against the table's bound column (POW `ipow`, ALIGN `isr`, LEAD's **new column `span`**: LEAD is 4 columns) on a *linear* wire; no product wire | −36 product wires (the `lt_pow` products), +1 LEAD column; the 25/30-bit ranges are two 16-bit chunks each |
| **QUOT_TIGHT** | `align_state.quot`, `align_prod.quot` | the quotient's direct bound keeps \(q\cdot\mathrm{pow}+r<2^B<p\): the division identity is an in-window identity, no extra column | 0 |
| **NORM_MBND** | `norm.div` | NORM gains column `mbnd = 2^30 / div`; `rng(mbnd - 1 - M', 0, 2^30)` per group (two 16-bit chunks) | +1 NORM column, +2 groups × 1 range |
| **NORM_VBND** | — | redundant given `leading_bit` (z3): the 5-column NORM table of the embedding loses `vbnd` in the export (`BabyBearCircuitCtx.lookup`) | 0 |
| **V_PREMISE** | `lead.window`, `norm.div` | no gate: \(v\) is the signed-sum wire, a linear form of range-checked hints with \(|v|\le s_{\max}<2^{29.1}\); `leading_bit`/`normalise` are stated for \(v\in[0,s_{\max}]\). The battery adapter's *gadget-only* circuits emit the premise as an explicit range (`premise.v`, sound by the criterion above) | 0 |

**Census** (`fixtures/babybear/manifest.json` vs `fixtures/manifest.json`, per transition unit):

| | Goldilocks | BabyBear |
|---|---|---|
| committed columns | 468 | **444** (+6 limb/hint columns, −30 materialised `lt_pow` query wires) |
| wires | 826 | 766 |
| product wires (mul) | 358 | **322** |
| assertions (lin) | 353 | 322 |
| lookup queries | 374 | **382** (R16 139, R14 10, R9 36, R8 137, R7 38; ALIGN 16, POW 2, LEAD 2, NORM 2) |
| tables | POW 3×512, ALIGN 4×512, LEAD 3×31, NORM 4×32, R7…R16 | POW 3×512, ALIGN 4×512, **LEAD 4×31, NORM 5×32**, R7…R16 |
| checker layers / max depth | 6 / 4 | 6 / 4 |

(The brief's 358/137/306 were the census estimate's mul/lin/lookups; the exporter's Goldilocks circuit is 358 /
353 / 374 because every non-power-of-two range is two chunks and query wires are materialised, section 12.1.)

**Range tables (soundness fix, both fields).** The exporter used to check a \(w<16\)-bit range as the single
`R16` query \(y\cdot2^{16-w}\). That is **forgeable in any prime field**: for every \(r<2^{16}\) the element
\(y=(r+kp)/2^{16-w}\) with \(k\equiv-rp^{-1}\pmod{2^{16-w}}\) satisfies \(y\cdot2^{16-w}\equiv r\) and lies in
\([p/2^{16-w},p)\), far outside \([0,2^w)\) (Goldilocks: \(y=p-(p-1)/2^{15}\) passes a bit check). Sub-16-bit ranges
now query the exact-width **implicit tables `R<w>`** = \([0,2^w)\) (R7, R8, R9, R14 in the unit; materialised by
`circuit.rs` like `R16`, narrowest first after the declared tables). The query count is unchanged (a chunk is a
chunk); the number of logUp instances grows from 5 to **9**, each small table adding a lookup sumcheck over
\(2^{\lceil\log_2(N+2^w)\rceil}\) leaves. The Goldilocks circuit was regenerated with the fix (commit 445cbca); its
numbers in section 12.5 / README predate it and are re-measured in 13.5.

### 13.3 Chain linkage per limb **[implemented]**

Section 12.2's functional links one word column per unit. Over BabyBear a word is two limbs, so `Chain.limbs` holds
\([\,c,\;y,\;y32\,]\) column triples per limb (`x.c_lo/y_lo/y32_lo`, `x.c_hi/y_hi/y32_hi`) and every constraint of
12.2 is stated per limb with its own geometric coefficient: \(c_{v,0}^{(\ell)}=0\), \(c_{v,i}^{(\ell)}=y_{v,i-1}^{(\ell)}\),
\(y32_v^{(\ell)}=y_{v,95}^{(\ell)}\), then \(y16_v=\) the public word. Linkage constraints per VU: \(2\cdot(96+1)+1=195\)
(Goldilocks 98). A "+p shadow" accumulator (the battery's `chain` class) has no limb form: the limbs of unit \(i+1\)'s
incoming word *are* unit \(i\)'s outgoing limbs, and unit \(i+1\)'s decode hints must match them.

### 13.4 Protocol changes forced by the field (each byte-exact in `fixtures/babybear-transcripts/README.md`)

1. **Transcript domain** `verity-gkr/candidate-a/babybear6/v1` (Goldilocks: `verity-gkr/candidate-a/v1`).
2. **Base-field serialisation is 4 bytes LE** everywhere a base element is absorbed or sent: the public `y16`
   words in the preamble, the opened Ligero columns (\(\text{rows}\times4\) bytes per column before the Merkle path),
   Merkle leaves (SHA-256 of the column's 4-byte values). Extension elements are 6 × 4 bytes, \(c_0\) first.
3. **Challenge squeeze** reads 4-byte words (13.1); one challenge = 6 squeezes.

Everything else — message order, round polynomials (degree 3 for logUp and phase 1, degree 2 for phases 2a/2b),
the batching functional, `open_set` (column index = \(c_0 \bmod n\) of a fresh challenge, repeats skipped) — is
the Goldilocks protocol verbatim, **except** the logUp sumcheck variable order: the GPU prover since a-gpu2 binds
the low bit first (4.2, "Variable order"), the Goldilocks crate the top bit first. Not forced by the field (a
kernel-layout choice of `packed/`), but a transcript change all the same.

### 13.5 Ligero parameters and soundness at \(2^{-128}\) **[derivation: `security.accounting`]**

Ligero over the 31-bit trace field with the section-3 tableau: \(k=4096\), \(n=16\,384\) (rate 1/4, RS over the
\(2^{14}\)-th roots of unity: \(\omega=31^{15\cdot2^{13}}\)), \(t=192\) opened columns — **the Goldilocks parameters
unchanged**: the query terms \((1-e/n)^t,((k+l)/n)^t,(2k/n)^t\) do not depend on the field, and the field terms
\((n+1)/|F_c|\) are \(2^{-171}\) at degree 6. `explore.fields.first_small_field_configuration` (census tables,
\(K=1536\), \(B=4096\), \(2^{60}\) hash queries):

| configuration | batching | sumcheck | lookup | encoding/opening (\(t=192\)) | commitment hash | **total** |
|---|---|---|---|---|---|---|
| Goldilocks, degree 3 (sections 9, 12) | \(2^{-180.1}\) | \(2^{-176.0}\) | \(2^{-158.3}\) | \(2^{-130.2}\) | \(2^{-136.0}\) | \(2^{-130.2}\) (\(2^{-127.7}\) with the chain functional's terms, README) |
| **BabyBear, degree 6** | \(2^{-180.1}\) | \(2^{-176.0}\) | \(2^{-158.3}\) | \(2^{-130.2}\) | \(2^{-136.0}\) | **\(2^{-130.2}\)** (`note:r20-proof/tensor-cost/20260922T0450Z-report-tensor`'s first-configuration sweep: \(2^{-129.5}\)); label **\(2^{-128}\)** |
| BabyBear, degree 4 | \(2^{-118.3}\) | \(2^{-114.2}\) | \(2^{-96.5}\) | \(2^{-109.6}\) | \(2^{-136.0}\) | \(2^{-96.5}\) at \(t=192\); `note:r20-proof/tensor-cost/20260922T0450Z-report-tensor`: \(2^{-81.4}\) at its first configuration — a \(2^{-80}\) point, not built |

Degree 5 misses \(2^{-128}\) (lookup term \(\approx2^{-127}\), section 2). logUp's \(\mathrm{char}\,F>N\) (Hab22
Lemma 5) holds for every table: the largest, `R16`, has \(139\cdot96\cdot4096+\text{epilogue chunks}+2^{16}\approx5.5\cdot10^7\)
leaves at \(B=4096\) (\(n_T=26\); it was 28 before the R7–R14 chunks left `R16`, for both fields), \(<2^{31}\); the
study's \(B=2^{20}\) failure point stands. The chain functional's Schwartz–Zippel term is \(195B/|F_c|<2^{-165}\).

### 13.6 Recorded runs

See `README.md` "BabyBear" (gate: `a-babybear.gate`; battery 83/83; measurements B = 64 / 4096 side by side with
Goldilocks on the same binary revision) — filled in by the lane as the runs land.

## 14. Checker v2 (lookup-centred arithmetisation; implemented 2026-09-22, lane a-v2; `--variant v2`)

The relation is unchanged (`note:r20-proof/checker-min/20260922T0727Z-report-checker-min`: the checker-min lane's differential and negatives are the evidence
that `checker.v2` accepts exactly the WP2 triples); what changes is the *shape* the prover sees. Exporter:
`gkr_export.v2` (`build_unit_circuit_v2`, `build_epilogue_circuit_v2`, `chain_spec`, `export_circuit_v2`); the Rust
side is the section-12 machinery with two additions — implicit tables (14.2) and a general link (14.3). Everything
below is for K = 1536 (96 units of 16 products, two groups of 8) over Goldilocks; BabyBear is **not** instantiated
(`state_decode` / `epilogue` are SAT as stated on 31-bit fields without the limb split, `note:r20-proof/red-team-babybear/20260922T0605Z-report-babybear-rules`).

### 14.1 Shape **[derivation: `gkr_export.v2`; checked by `tests/gkr_export/test_v2_export.py`]**

| | v1 (section 12) | v2 |
|---|---|---|
| committed columns / unit | 468 | **264** (4 incoming state, 32 operands, 4 outgoing state, 12 hints, 86 lookup outputs, 126 materialised keys) |
| wires / unit; multiplicative depth | 826; 4 (6 checker layers) | 290; **1** (2 checker layers: every product feeds a key or an output assertion) |
| products / lin / lookups per unit | 358 / 137 / 306 (census) | 26 / 9 / 86 (census 33 / 9 / 89: `state_decode`'s 3 mul / 4 lin / 5 lookups are gone — the state is chained, not decoded — Pack's 4 mul move to the epilogue; +2 one-sided halves per group of `attained`) |
| lookups by table / unit | `R16` 352, `ALIGN` 16, `POW`/`LEAD`/`NORM` 2 | `T_OP` 32, `ALIGN4` 18, `SHIFT` 16, `SSHIFT_HI`/`LO` 2+2, `LEAD` 4, `LEADNORM` 2, `TNORM` 2, `R12` 4, `R4` 4 |
| epilogue columns / wires | 9 / 11 | 16 / 22 (`Pack.check` on (s, e, M, z, ovf[g], sgn[g]) then `Epilogue.check`; 4 `R16` + 1 `R15`) |
| fixed table rows / proof | 66 623 | **5 143 328** (`SHIFT` 3 538 944, `TNORM` 1 048 576, `SSHIFT_HI`/`LO` 221 184 each, `T_OP` 65 280, `LEAD` 32 768, `R16` 65 536, `R15` 32 768, `LEADNORM` 12 152, `R12` 4 096, `ALIGN4` 3 120, `R4` 16) |
| linkage constraints / VU | 98 | **583** (14.3) |
| committed elements / VU (B = 4096) | 44 937 + tables/B | 96·264 + 16 = **25 360** + tables/B |

### 14.2 Tables: implicit, not committed **[decision]**

A LogUp table is public data: the prover commits only its *multiplicities* (one column of |T| elements), and the
verifier needs the table's rows only to evaluate the table-side multilinear at the final point (`logup.rs`:
`Σ m_t/(z − t)` over the table leaves, an O(|T|) pass with no commitment). So the question "committed per proof or
implicit" is really "listed row by row in `circuit.txt` or materialised from a definition": the *cost* is the same
either way — |T| multiplicity elements in the Ligero payload and |T| table leaves in the fractional-sum tree — and
only the circuit text (≈ 60 MB for `SHIFT` alone) and the verifier's evaluation differ. Decision: the four big
tables are **implicit** (`itable NAME cols rows kind params`, `circuit.rs::materialise`; the small ones stay as
rows):

| table | rows | definition (`gkr_export.v2.implicit_table_header`) |
|---|---|---|
| `SHIFT` | 2^16 · 54 | `shift 16 10 54`: key \(x\cdot54+2d+s\mapsto(1-2s)\lfloor x\,2^{10}/2^{d}\rfloor\), \(x<2^{16}\), \(d<27\) |
| `SSHIFT_HI` | 2^12 · 54 | `shift 12 13 54` |
| `SSHIFT_LO` | 2^12 · 54 | `shift 12 1 54` |
| `TNORM` | 2^15 · 32 | `tnorm 15 32 -7`: key \(v+2^{15}\,k\mapsto\lfloor v\,2^{k-7}\rfloor\), \(v<2^{15}\), \(k<32\) |

Both conventions are cross-checked: the exporter spot-checks `checker.v2.tables` against a Python transcription of
`materialise` before writing, and `circuit.rs`'s tests compare materialised rows with exported samples. Implicit
tables are `dense` (key = row index): `Table::find` is an array index, no hash map, so witness generation is O(1)
per query.

What it costs: `SHIFT` contributes 3.54M table leaves next to \(16U=6.29\)M query leaves at B = 4096 (\(n_T=24\)
against v1's `R16` at \(2^{28}\), \(n_T=28\): the lookup chain — and so the depth — *shortens*); at B = 64 the 5.1M
fixed rows are 84 % of the leaves and 76 % of the committed elements. The **split-u `SHIFT` variant** of
`note:r20-proof/checker-min/20260922T0727Z-report-checker-min` (two tables of 2^8·54) is the fallback if the fixed rows dominated at the target B; the
measurements in 14.5 decide (they do not, at B = 4096). Multiplicity columns: `committed_elements` includes all
5.14M per proof.

### 14.3 Chain linkage: state instead of words **[proposal, implemented]**

The v1 link is \(c_{v,i}=y_{v,i-1}\) on one column pair. In v2 the unit carries the *state* \((s,e,M,z)\) (sign,
exponent, magnitude, zero flag) and the group saturation flags; there is no decoded word until Pack. `chain.txt`
(`gkr_export.v2.write_chain`, `main.rs::read_chain`) has one line per link
`link <in|-> <out> <epi> <init> <zero_nonlast>`; `Chain.limbs: Vec<Link>` generalises the two v1 links:

| link | in → out (unit columns) | epilogue column | init at unit 0 | non-final units | constraints / VU |
|---|---|---|---|---|---|
| state ×4 | `x.s`←`y.s`, `x.e`←`y.e`, `x.M`←`y.M`, `x.z`←`y.z` | `s, e, M, z` | \((0,\ \mathrm{floor},\ 0,\ 1)\) = `zero_state` (the constant \(c_0=0\), no decode) | — | 4·(1 + 95 + 1) = 388 |
| ovf ×2 | — (`g<g>.ovf`, the group's saturation flag) | `ovf[g]` | — | **must be 0** (a saturated intermediate accumulator is outside the VU's domain; the exporter refuses such a witness) | 2·(95 + 1) = 192 |
| sgn ×2 | — (`g<g>.sgn`) | `sgn[g]` | — | — | 2 |
| public word | — | `y16` | — | — | 1 |

Total **583** per VU (v1: 98), \(2.39\)M at B = 4096 — still one linear functional with geometric coefficients,
batched into the opening exactly as 12.2; Schwartz–Zippel term \(583B/|F_c|<2^{-163}\). The `zero_nonlast`
constraints are what makes the chain *sound* against a prover who saturates mid-VU and un-saturates later: without
them Pack would see only the last unit's flags.

### 14.4 Security **[derivation: `security.accounting.account_candidate_a` on the exported shape]**

Layers `(s_out=5, s_in=9, assertions)`, `(9, 9)`; the 12 tables of 14.1 with the exported query counts; Ligero
\(n=16\,384,k=4096,l=4095,t=192\). **2^-127.71 at both B = 64 and B = 4096** (v1: 2^-127.7). The lookup term grows
with |T| as the brief warned — `SHIFT` is the worst single lookup identity at B = 64 (\(2^{-169.2}\), \(N=98\,304,\
|T|=3.54\)M) and `T_OP` at B = 4096 (\(2^{-166.4}\), \(N=12.6\)M) — but the lookup bucket is \(2^{-165}\) in total
and the budget is set by the commitment hash (\(2^{-128}\)) and the Ligero query test (\(2^{-130.2}\)) as before.
logUp's \(\mathrm{char}\,F>N\): the largest multiset is `T_OP` with \(32U+65\,280=12.6\)M leaves, far below
\(2^{64}\).

### 14.5 Recorded runs

See `README.md` "Checker v2" — gate (`gkr.v2.gate`), `vu-k1536-neg`, and the B = 64 / 4096 measurements beside the
v1 Goldilocks control on the same binary revision and box.

### 14.6 Hand-off to `lane/a-gpu` (layer descriptors)

* The unit circuit has **2 checker layers** (depth 1) instead of 6: the phase-1/2 kernels run 3× fewer sweeps; the
  layer widths are 264 inputs → 290 wires (v1 468 → 826). Per-layer descriptors come from the same `circuit.txt`
  format; nothing in the gate grammar changed.
* **Lookups dominate the tree work**: 86 queries / unit into 10 tables (v1: 374 into 5). The trees are shallower
  (\(n_T\le24\)) but there are twice as many; the 5.14M table-side leaves are a fixed per-proof cost (materialise
  them on device from the `itable` parameters — `shift`/`tnorm` are one shift and one multiply per row — rather
  than uploading 60 MB of rows).
* `Table::find` is dense for implicit tables (key = index): witness generation for the lookup outputs is a gather.
* The chain has 8 links (14.3), two of them with the `zero_nonlast` rule: the linkage functional is a strided
  read of 12 columns (v1: 2) — still one O(U) pass.
* Committed payload: 264 columns / unit + 16 / VU + 5.14M multiplicities; Ligero rows shrink proportionally
  (see README for the measured `committed_elements`).
* **Done (`lane/a-gpu-v2`, `gpu/DESIGN.md` §5b, `gpu/README.md` "Checker v2").**  The GPU prover runs this circuit
  over BabyBear^6: full VU ×4096 in 5.73 s on the A100 (v1 same box 7.93 s; 109.1 M committed elements, depth 328,
  24.4 GB), 4096/4096 accepted, 52/52 + 34/34 + 15/15 negatives rejected.  Over BabyBear the epilogue's word-form
  Pack (14.2) is SAT as stated (`note:r20-proof/checker-min/20260922T0727Z-report-checker-min` 3.2); the GPU export uses `lane/b-air-v2`'s limb-form Pack + cast
  (`gpu/v2/epilogue_limbs.py`: `M = M_hi8 2^16 + M_lo16`, high limb an expression, epilogue per limb as §13 LIMB16;
  19 epilogue columns, `R8` added; same function over Z, `chain.txt` unchanged), z3 UNSAT directly over BabyBear;
  the VU body's 14.x queries re-run with the BabyBear modulus are UNSAT (`notes-asset:campaigns/r20-proof/assets/a-gpu-v2/results/z3_babybear_v2.json`).

## 15. Batched logUp over the table union (implemented 2026-09-22, lane a-logup-union; `--logup union`)

**This is a protocol change to A-GKR's lookup argument and to nothing else.** The relation (`checker.v2`),
the circuit export, the committed columns, the chain linkage and the Ligero opening are those of sections 12–14; only
*how the lookups are proved* changes: the ten (v2: twelve, counting the epilogue's `R16`/`R15`) per-table
fractional-GKR instances of section 4 become **one** instance over the union of the tables, every element tagged by
its table. The per-table path stays the default; `--logup union` selects this section (the mode is part of the
statement: the transcript absorbs `logup-union-v1` and the tagged table order before any challenge, so a per-table
proof does not verify as a union proof or vice versa — `main.rs::tampered_proof_is_rejected`).

Motivation (`note:r20-proof/a-fusion/20260922T0853Z-report-a-kernels-h100-v2`): on the H100 the ten v2 instances cost 221.6 ms sequential / 84.7 ms on
ten streams against 35 ms modelled, "each instance pays ~14 ms of small-level latency"; one instance was estimated at
25–35 ms but flagged as a protocol change that the accounting did not cover. This section covers it.

### 15.1 Construction **[implemented: `logup.rs::union_*`, `main.rs::add_union_claim`]**

Tables \(T_1,\dots,T_m\) (v2: \(m=12\); \(\sum_t|T_t| = 5\,245\,624\) rows — the sum of the row counts listed in
14.1; the "5 143 328" total printed there is a miscount of the same list), columns \(c_t\le c_{\max}=3\). Tag
\(\mathrm{tag}(t)=1+\text{index of }t\) in `Instance::tables()` order (segments' tables in order of first use,
deduplicated), a base-field constant. One challenge message \((z,\beta,\gamma)\in F_c^3\) replaces the per-table
\((z_T,\beta_T)\) messages; it is drawn **after the commitment** (the multiplicities are part of the committed vector
\(X\), 12.1), at the point where the per-table coins were drawn (5.1 step 2a).

Encoding: a vector \(w=(w_0,\dots,w_{c-1})\) with tag \(t\) becomes the field element
\[\phi(w,t)=\sum_{k<c} w_k\,\beta^k+\gamma\,\mathrm{tag}(t).\]
The instance is the ordinary fractional sum \(\sum_j \frac{1}{z-\phi(q_j,t_j)} - \sum_v \frac{m_v}{z-\phi(r_v,t_v)} = 0\)
with leaves, in this order: for each segment (unit circuit, then epilogue) and each unit, its lookups in circuit
order — \(p=1,\ q=z-\phi(\text{query},t)\); then for each table in tag order its rows —
\(p=-m_v,\ q=z-\phi(r_v,t)\); padding \(p=0,q=1\) to \(2^n\), \(n=\lceil\log_2(N+|T|)\rceil\) with
\(N=\sum_t N_t\) the total number of lookups. One \(\mu,\lambda\)-chain (section 4.3) of \(n\) levels, one leaf
claim. The multiplicities are computed per table exactly as before (`union_multiplicities` is the per-table count
with the tables walked once); the committed layout (\(X\) = segments then tables' multiplicities) is unchanged.

The leaf claim is, as in 4.4, one linear functional of \(X\): the query leaves' \(q\) are affine in the committed
columns with coefficients \(\lambda\,\mathrm{eq}(\rho,j)\,\beta^k\cdot\)(the query's linear form); the table leaves'
\(p\) read the multiplicity column with coefficient \(-\lambda\,\mathrm{eq}(\rho,v)\); \(z-\gamma\,\mathrm{tag}(t)\)
and the constant parts of the linear forms go to the public constant. It joins the batched opening
(`add_union_claim`) in place of the per-table functionals: **one** functional instead of twelve.

### 15.2 Soundness **[derivation: `security.accounting.logup_union_error`; pinned by `test_union_lookup_meets_2_to_minus_128_on_the_exported_v2_shape`]**

Let \(A\) be the multiset of (query vector, tag) pairs the witness produces and \(B\) the multiset of (row, tag)
pairs with the committed multiplicities (arbitrary field elements). The lookup constraint is \(A\subseteq\)
rows-of-own-table, i.e. \(A=B\) for the honest \(m\). Suppose some query \(x=(w,t)\) is **not** a row of \(T_t\);
then \(A(x)\ge1\) and \(B(x)=0\) whatever \(m\) is. Let \(S\) be the support of \(A\cup B\), \(|S|\le N+|T|\).

* **Collision term** (the encoding is not injective at \(x\)): some \(y\in S,\ y\ne x\), has \(\phi(y)=\phi(x)\).
  - *Same tag* \(t\): \(\phi(y)-\phi(x)=(w'-w)(\beta)\) is a nonzero polynomial of degree \(\le c_t-1\) in
    \(\beta\); \(\le (c_t-1)/|F_c|\) per \(y\), at most \(N_t+|T_t|\) such \(y\).
  - *Different tag* \(t'\): for every \(\beta\) exactly one \(\gamma=(w(\beta)-r(\beta))/(\mathrm{tag}(t')-\mathrm{tag}(t))\);
    \(1/|F_c|\) per \(y\), at most \(N+|T|\) such \(y\). This is the brief's "\(n_{\text{rows}}\cdot n_{\text{tables}}/|F|\)"
    with \(|S|\) counted once rather than per table: the union bound over the pairs \((x,y)\) with \(x\) fixed.
  The accountant does not know which table the forged query targets, so it charges
  \(\big(\max_t (N_t+|T_t|)(c_t-1) + (N+|T|)\big)/|F_c|\). **[derivation]**
* **Identity term**: if \(\phi\) is injective at \(x\), the encoded multisets \(\phi(A),\phi(B)\) differ at
  \(\phi(x)\) — count \(A(x)\ne0\) in \(F_c\), which needs \(\mathrm{char}\,F>N\) (Goldilocks: \(2^{64}>3.4\cdot10^7\);
  BabyBear: \(2^{31}-2^{27}+1=2.01\cdot10^9>3.4\cdot10^7\), so the union does not break the characteristic
  condition either) — so the cleared rational identity is a nonzero polynomial of degree \(\le N+|T|-1\) in \(z\)
  (Haböck Lemma 5 with arbitrary \(m\)) vanishing at the conditioned \(z\) with probability
  \(\le (N+|T|-1)/(|F_c|-|T|)\). This is the per-table identity term with the **new denominator count**:
  \(N+|T|\) in one term instead of \(\sum_t\) of terms with \(N_t+|T_t|\) each — the same sum, so the bucket does
  not grow (at B = 4096 over Goldilocks\(^3\): \(2^{-166.8}\) against per-table's twelve terms summing to
  \(2^{-165.2}\)).
* **Fractional GKR**: Proposition 1 over \(n=\lceil\log_2(N+|T|)\rceil\) levels (\(2^{-182.0}\) at \(n=26\)).
* \(z\)-conditioning (5.1 step 2a) applies to the union: the redraw slots \(z^{(1)},z^{(2)}\) travel in the same
  message as \(\beta,\gamma\) and the condition \(z\notin\{\phi(r_v,t_v)\}\) is computed from public data once the
  message is known (no extra trip). Completeness: two rows of different tables that happen to collide under
  \(\gamma\) add their multiplicities on both sides — no honest abort.

Numbers on the exported v2 shape (`V2_EXPORT_TABLES`, 86 lookups/unit + 5 epilogue lookups/VU;
`hash_queries = 2^60` as in the accounting tests):

| | per-table (§14) | union (§15) |
|---|---|---|
| total, Goldilocks\(^3\), B = 4096 | \(2^{-130.16}\) (\(2^{-127.71}\) with the default hash budget) | **\(2^{-130.16}\)** (\(2^{-127.71}\)) — unchanged, set by Ligero + hash |
| total, BabyBear\(^6\), B = 4096 | \(2^{-130.16}\) | **\(2^{-130.16}\)** |
| lookup bucket, GL\(^3\) / BB\(^6\), B = 4096 | \(2^{-165.16}\) / \(2^{-158.60}\) | \(2^{-165.21}\) / \(2^{-158.65}\) |
| union collision / identity / GKR, GL\(^3\), B = 4096 | — | \(2^{-165.8}\) / \(2^{-166.8}\) / \(2^{-182.0}\) |
| \(N\), \(|T|\), \(n\) at B = 4096 | max per table: `T_OP` 12.6M + 65k, \(n_T\le24\) | 33 837 056 + 5 245 624 = 39.1M, \(n=26\) |
| at B = 64 | \(n_T\le22\) (`SHIFT`) | 528 704 + 5 245 624 = 5.77M, \(n=23\) |

**The union costs nothing in the bound at either field** (`meets(-128)` holds at B = 64 and 4096). The lane's
stop condition ("if the union term breaks \(2^{-128}\), stop") is not triggered.

### 15.3 Depth, rounds, verifier **[derivation: `rounds_and_depth_a`; measured `sequential_depth` in README]**

* **Coins**: one 3-element message replaces \(\sum_t(1+[c_t>1])\) per-table coins; the round challenges of one
  \(\mu,\lambda\)-chain of \(n\) levels replace twelve chains. `rounds.protocol` (coins) drops from 2825 to 448 at
  B = 4096 (2029 → 361 at B = 64).
* **Sequential depth**: the per-table chains run in parallel, so the per-table depth is the *deepest* table
  (\(n_T=24\) at B = 4096); the union is one chain of \(n=26\) levels. Depth = \(1+\sum_{k=1}^{n-1}(k+1)\)
  trips: 351 (union) vs 300 (per-table) at B = 4096, **+51**; 276 vs 253 at B = 64, **+23**. The checker chain
  (v2: 2 layers) is far shorter, so the lookup chain is the critical path in both modes: `rounds.sequential_depth`
  303 → 354 (B = 4096), 256 → 279 (B = 64). This is the honest cost of the union: two more levels of a
  quadratic-depth chain. The 4.6 mitigations (variables-per-round, sharding) apply unchanged; sharding by table is
  what the union undoes, so `table_shard_bits` is refused in union mode.
* **Verifier**: draws \((z,\beta,\gamma)\) once; runs one fractional-GKR verification (\(n\) levels, one final
  \((\rho,\lambda)\)); evaluates the table side \(\sum_v -m_v/(z-\phi(r_v,t_v))\) as one pass over the 5.2M tagged
  rows (`check_leaves` in clear mode; the multiplicity functional in PCS mode) instead of twelve passes; checks one
  leaf functional inside the batched opening. Work is the same O(N+|T|) as before, fewer fixed per-instance costs.
* Proof bytes: twelve (root claim, \(n_T\) levels of sumcheck polynomials) become one of \(n\) levels, but the
  opening dominates the proof: measured 3 152 048 → 2 992 520 B at B = 64 (0.95×) and 41 574 896 → 41 347 304 B at
  B = 4096 (0.995×), README rows.

### 15.4 GPU shape for `GraphLogUp` (hand-off: `note:r20-proof/a-logup-union/20260922T1033Z-handoff-handoff-union`)

* Leaves \(2^n\), \(n=\lceil\log_2(N+|T|)\rceil\): **\(2^{26}\) at B = 4096** (39.1M live of 67.1M; 42 % padding),
  \(2^{23}\) at B = 64 (5.77M of 8.39M). Layout as 15.1: segment 0's \(86\cdot 96B\) query leaves, the epilogue's
  \(5B\), then the tables in tag order (`T_OP`, `ALIGN4`, `SHIFT`, `SSHIFT_HI`, `SSHIFT_LO`, `LEAD`, `LEADNORM`,
  `TNORM`, `R12`, `R4`, `R16`, `R15` for the v2 export; the exact order is `Instance::tables()` = the transcript's
  absorbed order), padding \((0,1)\).
* Leaf kernels: query side — \(p=1\), \(q=z-\gamma\,\mathrm{tag}(t)-\sum_k\beta^k w_k\) with \(w\) a gather of
  \(\le3\) committed columns (the linear forms of `circuit.txt`); table side — \(p=-m_v\), \(q\) from the row
  materialised on device (`itable` parameters, 14.6) plus \(\gamma\,\mathrm{tag}(t)\). Both are embarrassingly
  parallel; \(\beta^k\) and \(\gamma\,\mathrm{tag}(t)\) are 3 + 12 constants.
* Tree: **one** fractional-sum tree of \(n\) levels and one \(\mu,\lambda\) sumcheck chain, i.e. the per-level
  latency (~14 ms/instance on the H100 for the small levels) is paid once. The 12 multiplicity columns are
  unchanged (already in \(X\)).
* Transcript: after the Ligero root, `label("union logup over 12 tables: z, beta, gamma (tag(t) = 1 + table index)")`,
  three `challenge()` calls in that order; then the segments' \((z_0,\gamma_{\text{copy}})\) points as before
  (`start_transcript`).

### 15.5 Red-team **[`redteam.logup_union`, `fixtures/redteam-3/logup-union/`, `tests/redteam/test_logup_union.py`]**

* **Tagged-union forgery** (query into \(t\) explained by a row of \(t'\)): z3 over \(F_{1000003}\) with 3 tables,
  free \(\gamma\) — UNSAT when \(\gamma\) is excluded from the finite collision set of 15.2, SAT only on it
  (the found \(\gamma\) is in the set). The set has \(\le\) (bad queries) \(\times\) (foreign rows) elements.
* **Multiplicity manipulation across the merged table**: with \(\gamma\) fixed outside the collision set the identity
  at \(N+|T|+1\) points is a linear system in the committed \(m\) over \(F_p\) — UNSAT for *every* \(m\); on a
  collision \(\gamma\) it is SAT with exactly the honest \(m\) plus one on the foreign row. So multiplicity freedom
  buys the prover nothing beyond the collision term.
* Fixtures: `cross_table/` (two tables sharing a column value; queries into \(T_A\) satisfied only by rows of
  \(T_B\), and vice versa — must reject in both modes, `main.rs::redteam3_cross_table_fixtures`), and
  `forged_epilogue_hint/` — **not a union attack**: the pre-fix `main.rs` proved lookups only for the tables of
  segment 0, so the epilogue's `R16`/`R15` range checks had no logUp instance at all and any public word was
  forgeable (`note:r20-proof/a-logup-union/20260922T0909Z-finding-redteam-urgent-epilogue-tables`; fixed in `Instance::new`, which now unions every segment's
  tables; regression `v2_forged_public_word_via_epilogue_hint_is_rejected`).

### 15.6 Recorded runs

README "Batched logUp over the table union": gate r20260922-093758-9aac (17 cargo tests; 30 + 4 v2 negatives rejected
in both modes), `vu-k1536-neg` r20260922-095946-5638 (52/52 in both modes), B = 64 r20260922-101533-20c4 and
B = 4096 r20260922-102013-2931 / r20260922-102556-cdcc on vy-cpu with `LOGUP=both` — per-table and union on the same
binary, box and instances. **CPU verdict: a wash** — union / per-table total 0.96× at B = 64, 1.08× at B = 4096
(two-run mean; lookup bucket 0.90× / 1.11×), bytes 0.95× / 0.995×, coins 2525 → 642 / 3393 → 761, depth +23 / +51.
The one \(2^{26}\) tree does 1.25× the leaf work of the twelve padded per-table trees at B = 4096; the union's case
is the GPU's per-instance latency, to be measured by the GPU lanes (`note:r20-proof/a-logup-union/20260922T1033Z-handoff-handoff-union`).

## 16. Operands bound (implemented 2026-09-25, lane agkr-bound; `bench_result.py --bind`)

Without it the GPU statement is the circuits plus the public words: an accepted proof shows that SOME operands give
each `y`.  Operands bound, the statement also holds the frozen operands, and the proof shows that the committed operand
columns of every unit ARE those operands (Table 2's "same full relation", user decision 2026-09-24).

### 16.1 Statement **[implemented: `gpu/bind.py`, `verifier/src/main.rs::read_bind`, `verifier/src/instances.rs`]**

* `bind.txt` (Params-only): `bind <col> <arr>:<stride>:<off>:<coef> ...` per bound unit column, meaning
  \(X[(v,i),\mathrm{col}] = \sum \mathrm{coef}\cdot \mathrm{ARR}[v, \mathrm{stride}\cdot i + \mathrm{off}]\) for every VU \(v\) and step
  \(i\).  Checker v2 (BF16, E4M3): the \(k\) x and \(k\) w word columns of the unit (\(k\) = 16 / 32).  fp4-nvf4:
  `x.p{i}` \(= 16a + b\) (64 E2M1 code pairs) and `x.s{g}` \(= 128a + b\) (4 UE4M3 scale pairs) of the 68-byte step.  The
  Rust verifier re-derives this text from the unit circuit's column layout and rejects any other `bind.txt`.
* `instances.txt` (`relation`, `lo`, `vus`, the word widths) names the frozen set; `x.bin`, `w.bin` hold its operand
  words, little-endian, VU-major (for `vu-k1536` byte-identical to the tier's `.x.u16` / `.w.u16`).
* The Rust verifier reads `x.bin` / `w.bin` itself and requires SHA-256 of both and of the public words (at the
  `y` width) to equal its pinned entry for (relation, lo, vus), whose model must be the manifest's (`instances.rs`;
  `bf16-ampere+bound` = the committed frozen tier `vu-k1536`, checked by a cargo test; the recipe sets by
  `python -m gpu.bind pin`).
* A bound statement is its own relation, `R+bound` for a relation `R` of `verifier/pins.txt`: `verify --relation R+bound`
  requires its circuit files to be a pinned set of `R` and `instances.txt` to name `R+bound`, whose entry pins the
  instance files.  `--relation R` refuses a statement with a `bind.txt`, `--relation R+bound` one without.
* Not the full relation (user, 2026-09-25): the verifier holds the operands.  The target statement commits x, W and y
  (`verity.ml.commitments` leaf/v2h) and binds the commitments in the proof; this section is its public-input intermediate.

### 16.2 Protocol **[implemented: `gpu/prover.py::add_bind`, `verify.rs` `bind_u` / `bind_mats`]**

The transcript absorbs SHA-256 of `bind.txt`, `instances.txt`, `x.bin`, `w.bin` after `chain.txt`'s.  After the chain
challenge `geo` it draws \(\rho_b, g_b\); with \(n_b\) bound columns the functional gains
\(\rho_b g_b^{u n_b + j}\) on unit \(u\)'s bound column \(j\) and \(b\) gains
\(\sum_u \rho_b g_b^{u n_b} \sum_j g_b^j\,\mathrm{val}(u,j)\), computed by the verifier from `x.bin` / `w.bin`.  The
prover's side is one rank-1 update of segment 0 (\(\rho_b (g_b^{n_b})^u \otimes g_b^j\)); it never reads the values.
Nothing is committed beyond the unbound statement, so the Ligero rows and every other term are unchanged.

### 16.3 Soundness **[derivation: `soundness.py` "operand binding"]**

If any bound column differs from its value, \(\rho_b \sum_c g_b^c r_c\) is a nonzero polynomial of degree
\(\le N_b = U n_b\) in \((\rho_b, g_b)\), both drawn after the commitment root and every message: error
\(\le N_b / |F|\), \(|F| = p^6\) (\(2^{-161.9}\) at \(U n_b = 393216 \cdot 32 = 196608 \cdot 64\), BF16 / E4M3; \(2^{-162.8}\) for fp4-nvf4), far below the
Ligero query term (\(2^{-130.19}\)), so the achieved bound is unchanged at 2 decimals.

## 17. The bit link, prime side (implemented 2026-09-25, lane agkr-bound; `link.txt`; drill-down only)

Route (a) of the standard-hash workstream: a binary proof (Flock, GF(2^128)) proves the SHA-256 / BLAKE3 row digests over
the operand **bits** z, and this proof shows that the operand words its units consume ARE those bits.  Sigma form (the
link red team's S1, `lanes/red-team-link` §4): the prover sends its 256 plane sums in the clear, so this is a
**NON_ZK_PROOF**-class construction only; no second commitment, no masking vector.  Results stay in the drill-down
(reason L) until the red team's C1 (one GF(2^256) point: done here), C3 (Flock at 2^-128, interactive coins) and C4
(chain glue across Flock blocks) hold.

### 17.1 Statement **[implemented: `tools/link_stub.extend_unit`, `gpu/link.py::derive`, `verifier/src/link.rs::{derive, load}`]**

* The unit circuit carries, after every other column, `link.b{5+j}.{i}` for each operand column `5 + j` (j < 2k: the k x
  words, then the k W words of the step) and bit i < bits (16 BF16, 8 E4M3), with the booleanity assertion
  b (b - 1) = 0 and the recomposition assertion col = sum_i 2^i b_i.  The verifier checks the names, the order, and both
  constraints for every bit (red team C7), from the circuit it verifies.
* Lambda_F, derived by the verifier and never read: cell jb = j bits + i of unit u = v steps + s goes to position
  pos = ((port vus + v) K + k s + (j mod k)) bits + i, port = j div k (x rows, then W columns; words in order; bit i
  LSB first).  It requires steps k = K and n_cells = 2 vus steps k bits = n_pos = 2 vus K bits <= p - 1 (C6: at most
  40,960 VUs at K = 1536 for BF16), and checks injectivity with a bitmap over [0, n_pos) (C5).  The binary side's
  Lambda_B must map its hash's message-bit order (SHA-256: big-endian words) onto these positions (red team F2); that
  map belongs to the binary proof.
* `link.txt`: `link sigma-gf2_256/v1`, `root_b <hex>`; the statement must be operands-committed (§16's successor,
  `commitment.txt`).  STAND-IN until the binary proof is wired: root_b = SHA-256(TAG || "/root-b-standin/" ||
  commitment.txt), TAG = `verity-gkr/link/sigma-gf2_256/v1`, and the binary side's claim y is computed by the verifier
  from `x.bin` / `w.bin`, which it first requires to hash (`sha256/row/v1`) to the public digest limbs of every VU.

### 17.2 Protocol **[implemented: `gpu/link.py::{prove, verify, add}`, `verify.rs` (`link_term`, `unit_row`)]**

After the GKR segments, before the functional's rho's (so after the Ligero root and every GKR message; C2):

1. absorb TAG || root_b; draw r_0 .. r_{m-1}, m = log2ceil(n_pos) (28 for BF16 at 4096 VUs), each one whole digest
   (`challenge_bytes`: its own coin slot), a GF(2^256) element little-endian.  GF(2^256) = GF(2)[x]/(x^256 + x^10 +
   x^5 + x^2 + 1), bit t = the coefficient of x^t; eq(r, p) = prod_j (p_j ? r_j : 1 + r_j), bit j of p against r_j.
2. y = z^(r) = sum over the positions p with z_p = 1 of eq(r, p) (binary side; stand-in above); absorb "link/y" || y.
3. The prover sends sigma_t = sum_q bit_t(eq(r, pos(q))) b_q for t < 256, each as the extension element (sigma_t, 0, ..,
   0).  The verifier requires every sigma_t canonical with 0 <= sigma_t <= n_cells and sigma_t = bit_t(y) mod 2.
4. absorb the 256 messages; draw rho.  The functional gains a[q] += sum_t rho^t bit_t(eq(r, pos(q))) on every link cell
   of segment 0, and b += sum_t rho^t sigma_t; it is opened with everything else (one Ligero opening, no new rows).

Cost: 256 messages (6,144 bytes), m + 1 coin slots.  The verifier's term of a is one GF(2^256) product
eq_hi(p >> 14) eq_lo(p mod 2^14) per cell and 32 byte-table lookups of rho powers; the prover's plane sums and term of
a are int8 tensor-core products of the bit planes of the eq table (2^m x 32 bytes) with 0/1 vectors and with the 7-bit
limbs of the rho powers.

### 17.3 Soundness **[derivation; superseded by 17.5, whose terms `soundness.py` books]**

Suppose some committed bit differs from z at its position (the bits are boolean by the unit's assertions, and Lambda_F
is a bijection, so b viewed in position order is a boolean vector b').  Then b'^ - z^ is a nonzero multilinear
polynomial over GF(2^256) in m variables and b'^(r) = y with probability at most m / 2^256 (2^-251.2 at m = 28;
r is drawn after both roots).  Otherwise b'^(r) != y differ in some bit t, i.e. sum_q bit_t(eq(r, pos(q))) b_q and
sigma_t differ in parity, so they differ as integers; both lie in [0, n_cells] with n_cells <= p - 1, so they differ
mod p, and the 256 linear constraints fail.  rho batches them with error <= 255 / p^6, and the Ligero linear test on
the batched functional is the existing term.  Total added error <= m / 2^256 + 255 / p^6, far below Ligero's 2^-130.19.
Not covered here: the binary proof (Flock's bound, C3), the chain glue between Flock blocks (C4), and the whole-proof
accountant (C8).

### 17.4 Recorded runs

A100 80GB, 4,096 VUs, in-unit bits with and without the link, 1 warm-up + 3 reps, medians; the binary side is the
stand-in, so these are drill-down figures (L). BF16 (bf16-ampere+sha256, r20260925-113047-d490, art:bd3d8b2c /
art:fc687ece): prove 1.199 → 1.529 s (link 0.333 s: eq table 0.137, plane sums 0.062, term of a 0.124), Python verify
1.261 → 1.564 s, Rust verify 3.414 → 4.70 s plus 0.436 s derivation at load. FP8 (fp8-hopper+sha256, unpinned
commitment, r20260925-114925-2f5c, art:60cabb96): prove 0.607 → 0.778 s (link 0.178 s), Python verify 0.489 → 0.648 s,
Rust verify 1.706 → 2.35 s plus 0.219 s. Proof +6,144 bytes on both. Every §17 negative (bit flip, non-boolean bit,
altered operand with its own bits, sigma out of range, sigma off by 2, root_b changed, z differs, VU remap, duplicate
link cell, dropped booleanity) is rejected by both verifiers.

### 17.5 Route (a) in one live session (v2, lane agkr-flock-cell; supersedes 17.1's stand-in and 17.2's point)

The binary side is Flock (`verity/ml/flock/live`, `flock-link cell-prove` / `cell-serve`, flock-128-r2) in the same live
session as this proof; the red-team-flock third audit's P1-P3 and E0-E1 are what this section implements.

* **Scheme (P3):** `bf16-ampere+blake3`, frame-v3 with `blake3-keyed/row/v2` row leaves, the leaf Flock proves (its chunk
  chain starts from the role's key, KEYED_HASH on every compression, keyed parents natively). Chosen over a SHA-256
  chunk chain in Flock: it is two constants in the existing BLAKE3 circuit, needs no big-endian Λ_B, and keeps Flock's
  faster hash; nothing else used the plain-BLAKE3 leaf. The instance commitment is pinned (`commitments.rs`), and so is
  the linked circuit set (`pins.txt`, `python -m tools.cell circuits`), which the verifier refuses without a `link.txt`.
* **Order (P1):** after the GKR segments the prover publishes root_F = SHA-256(`verity-gkr/link/root-f/v2` || transcript
  state || ctr) (it binds the Ligero root and every GKR message). The session's `Commit` carries root_F with Flock's
  root_B; the verifier draws the link points from the OS; Flock sends y; only then Flock's coins. The prime transcript
  absorbs TAG || Σ || SHA-256(root_B) || points || y (TAG = `verity-gkr/link/sigma-2xgf2_128/v2`), then the sigma
  messages, then rho. The prime verifier takes Σ, SHA-256(root_F), SHA-256(root_B), the points and y from the session
  record (`link.txt`: `link session-2xgf2_128/v2`, written from the record by `tools/cell_gate.py`), recomputes root_F
  from its own replay and refuses a record whose root_F differs.
* **Points (P2):** two GF(2^128) points (Flock's field, x^128 + x^7 + x^2 + x + 1), m = 28 coordinates each. A plane pair
  (point 0, point 1) is one 256-bit value, so the 256 sigma messages are 128 planes of each point; everything after the
  eq tables is 17.2 unchanged. Σ (`sigma.txt`, SHA-256 = the session's Σ) names the point choice, the scheme, both Λ,
  N, m, the Flock statement, SHA-256(commitment.txt) and SHA-256 of the leaf digests; the prime verifier checks every
  line it can derive from its own statement.
* **Evidence (E1):** `tools/cell_gate.py` admits a record only with `link_mode` exchange, `require_link` true, the cell's
  Σ, an accepted verdict, two points and two y, the proofs kept beside the record with their hashes, a verifier operator
  that is not a producer lane or pod, and (with `--rust`) the prime verifier accepting the prime proof against the
  record's link with the circuit set and the commitment pinned. E0 is in the session library: `SessionConfig::check`
  refuses `link: Some` with `require_link: false`, and the R5 gate applies whenever there is a link.
* **Soundness:** `soundness.py` adds 255/p^6 (the sigma batching) and (m/2^128)^2 (Schwartz-Zippel over the two
  session points); the Flock side's terms are its own (red-team-flock third audit: about 2^-195.5 for the two reps,
  2^-243.9 for the link claims' reduction). The composed bound is set by A-GKR's Ligero queries, 2^-130.19.

### 17.6 Live prime coins (lane route-a-live; the red-team-flock fourth audit's class condition and G2)

Under 17.5 the prime half drew every challenge from its own SHA-256 transcript: runner-derived Fiat–Shamir coins, so
the cell was NON_ZK_PROOF_DIAGNOSTIC (under the campaign's ×2^60 rule that half is 2^-70.2). Here the same live session
serves them, so the whole cell is the interactive object of §0 and the bound is the interactive one, 2^-130.19.

* **Round (R2's discipline):** the prime transcript keeps its absorb chain (`state <- SHA-256(state || msg)`). A challenge
  with an absorb since the previous round opens a round: the prover sends `Prime{state, n}` (the state commits every
  message absorbed so far and every earlier coin), and the verifier answers with `6 n` OS-drawn words below p, the
  round's n BabyBear^6 challenges; both sides then chain `state <- SHA-256(state || "verity-gkr/live-coins/v1" || words
  LE)`. A challenge with no absorb since takes the round's next unused coin; an absorb discards the rest. `reserve(n)`
  makes one round of n known-consecutive challenges, at four places mirrored by prover, Python verifier and Rust
  verifier: the transcript start (every table's z, beta and every segment's z0, gamma), each LogUp level's (mu, lam),
  the link's rho with the batching coins after it (`open_coins`: the functionals, geo, the bind pair, Ligero's row
  coefficient), and the Ligero query set (t + 8). Nothing is derived under live coins: `challenge_bytes` is refused, and
  the graphed LogUp tree (device-side Fiat–Shamir) is off.
* **Order (server, `verity/ml/flock/live`):** `Prime` is refused before `Hello`, and between `Commit` and y (R5 for the
  prime half: rho and the Ligero queries follow the points and y); `Commit` is refused before any prime round (root_F
  must follow live GKR coins); `Hello` pins `prime: live-coins/v1` (R7). No coin follows the verdict. The record keeps
  every round (state, n, words) and `before_commit`; the verdict requires at least one round on each side of `Commit`.
* **Replay (the prime verifier):** `prime_coins.txt` (from the record, `tools/cell_gate.py`) beside `link.txt`; the
  verifier's transcript must reproduce each recorded state before it takes that round's words, draw exactly the
  recorded counts, compute root_F after exactly `before_commit` rounds, and consume every round
  (`verifier/src/transcript.rs`, `gpu/transcript.py::ReplayTranscript`). `--require-live-coins` refuses a linked proof
  without them.
* **Transport:** one connection per session. `cell-prove --prime-sock` forwards the prime prover's `Prime` frames on the
  session's connection after `Hello`, and sends the proofs and `Finish` only after the prime prover closed its socket.
* **Cost:** one RTT per prime round, sequential. Before root_F they are on the critical path (Flock's `Commit`, and so
  all of Flock's rounds, wait for root_F); after y they overlap Flock. Each round is 45 bytes up and 13 + 24 n down.
  `times.json` / the result record `session.prime_rounds`, `session.prime_rounds_before_root_f`, the prime bytes each
  way, `prime.network_wait_seconds` and `prime.wait_before_root_f_seconds`; `session.rounds` counts both provers' rounds.
* **G2 (offline Flock replay):** the record also keeps root_F, every root_B and the committed public words.
  `flock-link replay --session DIR` rebuilds the session from `session.json` under the verifier's own configuration
  and statement (`Server::from_record`: the hello, C4 on the public words, each rep's root against root_B, the link
  context's shape, the prime order, the proofs' hashes, `link_sha256`), then re-runs the verdict without using the
  recorded one. `tools/cell_gate.py --flock` adds it as the `flock_replay` check. The record is evidence only under the
  verifier's custody; nothing in it authenticates its writer.
