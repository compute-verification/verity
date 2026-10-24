---
id: training-20261024T2310Z-draft-tvi-e1-sections
campaign: covering-bound-deck
lane: training
kind: draft
status: open
repo: compute-verification/verity
origin: E1 (statement), worker of @training (bc-b43da88b-5d25-4463-9b8c-da4df6f84207), phase 2 of crux 6
---

# E1, phase 2: paste-ready text for "Distinguishing inference from frontier training"

Numbering follows the Notion page as fetched at 22:50Z: Proposition 4 is partial sums, Lemma 5 recovery, Proposition 6
matrix outputs and Proposition 7 small state. New items are Lemma 8 (splitting), Lemma 9 (replay), Precondition P0,
Corollary 3b, Proposition 3′, Corollary 4a, Lemma 6a, Proposition 6b and Proposition 7b. Every number is reproduced by
`tvi/e1_phase2.py` on branch `cursor/tvi-e1-4207`. The shape constants are Kimi K2's: $d=7168$, and fan-out
$e=37{,}248$ products per MLP-input activation ($2\cdot2048\cdot9+384$). Elsewhere $\varphi=6.7\times10^9$, $\kappa=6$
and $h_{\rm tok}=17$.

**What changes, in brief.**

1. **§5.1 is missing its strongest bound, the fan-out face, and needs no new premise to get it.** Each activation
   occurrence is used in at most $e$ counted products. So a block's priced products are at most
   $\min\{h_w,P_{\rm col}\}\,h_x/\beta_x$, where $P_{\rm col}$ is the largest total price of the slots one activation
   meets. This is about $14$ kB at FP8, below $h_w$ for every $k\gtrsim14$ kB.
   - With uniform prices this gives $\rho\ge c\beta_x/(\kappa e h_x)$, which does not depend on weight prices.
   - In Lean it is `FanOutCoverBound`. The formula $\alpha_0 s f/(e k)$ reproduces its docstring's $8{,}900$ for
     Kimi K2.
   - At $2.5$ MB the joint premise then adds a factor of at most $h/(h-P_{\rm col})\approx1.006$, not $4$.
2. **§5.1's activation rate has to be replay-consistent.** A block holding $Q$ tokens recomputes $\theta$'s activations
   for free and adds the fresh residual. So any joint premise over a token's coordinates has
   $\beta_x\le(h_{\rm tok}+F)/d$: $0.39$–$0.87$ bits at FP8 for LoRA-16 and full fine-tuning, not $2.6$–$4.9$. The fan-out
   face at that rate is $\rho\approx1.2$–$2.6\times10^4$ at matched $c$.
3. **Lemma 8 makes splitting a stated result.** A product split inside its multiplication region costs one boundary gate
   (Lean's `completes_or_meets`). A value split whose pieces meet at an exact nonlinear consumer costs at least the
   smaller piece's entropy. In §5.1, splitting therefore changes the bound by at most a factor $1+2/\gamma$. Linear
   segments are the stated exception.
4. **P0 is the replay precondition, and it fails exactly where §5.4 packs steps.** With seeded or self-generated data,
   P0 fails for the first $\lfloor\tau/F\rfloor$ steps after each registration. That is $670$ steps of GRPO on
   Qwen2.5-1.5B, and about $24$ for frontier small-batch RL.
5. **§5.3: conditioning on $A$ is what caps the rank argument.** The derived bound holds only for
   $k<\beta_vd\approx0.16$–$0.56$ kB at FP8. The token-proportional count at production $k$ needs an output premise
   given $R$ alone (Proposition 6b), and that premise carries a cap, $k\le k^*$.
6. **§5.4: Proposition 7 is tight.** A data lower bound (Proposition 7b) matches it within
   $\tfrac{k}{k-s}\cdot\tfrac{u}{u_{\rm eff}}$: at $1$ MB it is $1.92$ against $\ge1.82$.

---

## Precondition P0 (insert in §3, after the replay paragraph, or as §4.1)

Every lower bound below assumes some description length is large. Replay is the explicit program that makes it
small, so the first check on any class premise is that replay doesn't fit.

**Lemma 9 (replay upper bound).** Take a training run between the latest registration and a counted item $y$:
$n_0$ steps, persistent unregistered state $s_0$ at registration ($s_0=0$ when $\theta$ is the whole state), new inputs
of at most $u$ bits per step, and at most $F$ work per step. Suppose $y$ is extractable from the replayed trace within
time $t_x$. If $n_0F+t_x\le\tau$, then

$$
\operatorname{LDL}_{\tau,R}(y)\le s_0+n_0u+\lceil\log_2 n_0\rceil+a_{\rm rep}.
$$

*Proof.* The program holds the state, the inputs and the step count, runs the recipe, and extracts $y$. ∎

**Precondition P0$(\tau)$.** For every collection $y$ that a class premise quantifies over, no program of length at most
$k'+\delta$ outputs $y$ given $R$ within time $\tau$. Here $\tau$ is the premise's time allowance: $t$ in §§5.1–5.2, the
recovery allowance in §5.3.

**Corollary 9a.** A premise $\operatorname{LDL}_{\tau,R}(y)\ge\beta n(y)-\delta$ with $\beta n(y)-\delta>k'$ implies P0
for $y$. By Lemma 9, P0, and with it the premise, fails for every counted step at which both of these hold:

$$
s_0+n_0u\le k'+\delta-\lceil\log_2n_0\rceil-a_{\rm rep}
\qquad\text{and}\qquad
n_0F+t_x\le\tau .
$$

So P0 holds at a step when one of three things is true: its inputs since registration don't fit in a block, its replay
work exceeds $\tau$, or no faster replay than the recipe is known. Fast-forwards replace $n_0F$ by their own time; they
include exact quadratic dynamics, a non-rotating update direction, and a computable fixed point. These conditions are
the complement of Proposition 7's packing conditions ($s+gu\le k$, $gF\le c$, with $\tau\approx c$). **The lower-bound
classes apply only at the steps the small-state construction cannot pack.**

**Status.** In the exact-covering toys (`benchmarks/covering_toy/`, `art:2f1eae928df7…`), every decoding failure occurred
where replay fit in a block, and none occurred elsewhere. The production cap failed only on step-0 values, that is,
registered-model inference.

**Numbers** (at $\tau=c=1.34\times10^{17}$, with seeded or self-generated data so that $u$ is a seed):

| Workload | Work per step $F$ | Steps after registration for which P0 fails ($n_0\le\tau/F$) |
| --- | --- | --- |
| GRPO, Qwen2.5-1.5B (§5.4's estimate) | $2\times10^{14}$ | $670$ |
| Frontier small-batch RL ($19{,}200$ tokens per step, 49B active) | $5.6\times10^{15}$ | about $24$ |
| Pythia-160m pretraining step | $2\times10^{15}$ | $67$ by work. One step's data is $4.5$ MB raw, so the input clause holds at $2.5$ MB unless the data codes below $k$ |
| Frontier pretraining step | $1.2\times10^{18}$ | none: P0 holds from step 1 |

**Registration.** P0 is relative to the *latest* registration. With seeded data, re-registering every $\lfloor\tau/F\rfloor$
steps makes each segment a one-block execution. A registration policy must bound or charge re-registration; §3 already
notes that registration can hard-code an update.

---

## 5.1. Explicit weight–activation products

### Class

As in the draft: required products $(i,j)\in\mathcal J$, each performed whole in its assigned block, with slots
$U_I$ and activation occurrences $V_I$. Per-family premises hold within time $t$:

$$
\operatorname{LDL}_{t,R}(w_{U_I})\ge\sum_{i\in U_I}p_i-\lambda_w,
\qquad
\operatorname{LDL}_{t,R}(a_{V_I})\ge\beta_x|V_I|-\lambda_x .
$$

Add two quantities that the recipe fixes; they are structure, not premises:

$$
e:=\max_j\#\{i:(i,j)\in\mathcal J\},
\qquad
P_{\rm col}:=\max_j\sum_{i:(i,j)\in\mathcal J}p_i .
$$

For a matrix, $j$ is an input coordinate at one token. Its products are the slots of that coordinate's column in every
matrix that reads it, so $e$ is the stacked output width and $P_{\rm col}\le e\,p_{\max}$.

### Bound

**Proposition 3 (slowdown lower bound, sharpened).** Every execution in the class satisfies

$$
\boxed{\rho\ge\frac{c\,\beta_x}{T_0\,\min\{h_w,P_{\rm col}\}\,h_x}\sum_{(i,j)\in\mathcal J}p_i .}
$$

Write $\bar p$ for the use-weighted mean price, and let $P_{\rm col}=e\bar p\le h_w$ (true for $k\gtrsim e\bar p$). Then
the prices cancel:

$$
\boxed{\rho\ge\frac{c\,\beta_x}{\kappa\,e\,h_x}.}
$$

This is `FanOutCoverBound`'s form, $\rho\ge\alpha_0 s f/(e k)$, with $\alpha_0=1/\kappa$ and $f=\beta_x$. At $f=1$ it
follows from `ColumnsIncompressible` (`KeyedActivationReads`).

**Corollary 3a (joint operand budget, column-capped).** Under the draft's joint premise
$\operatorname{LDL}_{t,R}(w_{U_I},a_{V_I})\ge\sum_{U_I}p_i+\beta_x|V_I|-\lambda$ with $h=k'+\lambda$, let
$m:=\min\{h/2,\,P_{\rm col}\}$. A block's priced products are then at most $m(h-m)/\beta_x$, so

$$
\boxed{\rho\ge\frac{c\,\beta_x}{T_0\,m(h-m)}\sum_{(i,j)}p_i .}
$$

The two regimes:
- For $P_{\rm col}\ge h/2$ this is the draft's $4c\bar p\beta_x/(\kappa h^2)$.
- For $P_{\rm col}<h/2$ it is $c\beta_x/(\kappa e(h-e\bar p))$ under uniform prices. That improves Proposition 3 only by
  the factor $h/(h-P_{\rm col})$.

So the joint premise is worth a factor of $4$ only while $P_{\rm col}\ge h$.

**Tightness.** In both regimes the bound is attained by a *column tile*. Take one activation coordinate's slots of
total price $m$, together with $(h-m)/\beta_x$ occurrences of that coordinate at different tokens. At unit prices,
$h=k$ and $P_{\rm col}\ge k/2$, this is the toy's optimum of $\lfloor k/2\rfloor\lceil k/2\rceil$ products. Under these
premises, no better per-block bound exists.

### Proof

Lemma 1 gives $P_I:=\sum_{U_I}p_i\le h_w$ and $|V_I|\le h_x/\beta_x$. Each assigned product $(i,j)$ has $j\in V_I$, so
its total price is

$$
\sum_{j\in V_I}\ \sum_{i\in U_I:(i,j)\in\mathcal J}p_i\le\min\{P_I,P_{\rm col}\}\,|V_I|
\le\min\{h_w,P_{\rm col}\}\frac{h_x}{\beta_x}.
$$

For the corollary, $P_I+\beta_x|V_I|\le h$ gives $\min\{P_I,P_{\rm col}\}(h-P_I)/\beta_x\le m(h-m)/\beta_x$. The left side
increases in $P_I$ up to $\min\{h/2,P_{\rm col}\}$ and is at most $P_{\rm col}(h-P_{\rm col})$ beyond $P_{\rm col}$. Apply
Theorem 2 with product weight $p_i$. ∎

### Splitting products (replaces "requires an additional accounting argument")

**Lemma 8 (splitting).**

*(i) Region split.* Suppose each counted product $(i,j)$ has a multiplication region $G_{ij}$: the gates between its
operand gates and its output gate $o_{ij}$. Suppose the regions are pairwise disjoint. If block $I$ computes $o_{ij}$
but not all of $G_{ij}$, then $\mathrm{in}(I)$ contains a gate of $G_{ij}$. Hence a block has at most $k$ such
products.

*(ii) Value split at an exact consumer.* Colour the blocks into sides $\mathcal A\sqcup\mathcal B$.
- Let $Z_A$ and $Z_B$ be the unregistered sources read only by $\mathcal A$-blocks and only by $\mathcal B$-blocks; fix
  every other unregistered source.
- Let the crossings $X$ be the gates in $\mathrm{in}(I)\setminus\theta$, for $I$ on one side, that no block of that side
  computes. Then $\mathrm{bits}(X)\le Bk$.
- Suppose that for tokens $j\in J$, the circuit computes $y_j=\|v_j\|^2\bmod2^{\ell+1}$ exactly, where
  $v_j=\alpha_j(z_A)+\beta_j(z_B)\bmod2^b$. Suppose the low $\ell$ bit-planes of $(\alpha_j)_{j\in J}$ and
  $(\beta_j)_{j\in J}$ range over $S_A$ and $S_B$.

Then

$$
\mathrm{bits}(X)\ \ge\ \log_2|S_A|+\log_2|S_B|-n|J|\ell .
$$

There are analogous bounds for the other exact consumers:
- An elementwise nonlinearity with a generic table gives $\ge\log_2|S_A|$ when the other piece is full range.
- The softmax normalizer is inner-product-hard: $\Omega(T)$ per query row.
- Adam's $G^2$ costs, per weight, the entropy of the smaller gradient piece.

*Proof.* (i) Walk back from $o_{ij}$ inside $G_{ij}$. A computed gate's predecessors are computed or are inputs, so the
first uncomputed gate is an input. Disjoint regions give distinct inputs.

(ii) Order the gates topologically. Each side's gates are functions of its own sources and of the crossings' values
$\xi$, so the inputs consistent with $\xi$ form a rectangle. The matrix $M[z_A,z_B]=\omega^{\sum_jy_j}$, with
$\omega=e^{2\pi i/2^{\ell+1}}$, is then a sum of at most $2^{\mathrm{bits}(X)}$ rank-one blocks. Since
$x^2\equiv(x\bmod2^b)^2\pmod{2^{\ell+1}}$, $M=D_A H D_B$, with $D_A,D_B$ diagonal unitaries and $H$ the character table of
$\mathbb Z_{2^\ell}^{n|J|}$ restricted to $S_A\times S_B$. Then
$\operatorname{rank}H\ge\|H\|_F^2/\|H\|_{\rm op}^2\ge|S_A||S_B|/2^{n|J|\ell}$. ∎
(Source: `note:training-20261024T2134Z-splitting-locality`; `bound_check.py` checks the identity.)

Part (ii) is information-theoretic. The class premises are time-bounded, so using it there takes a named premise:

**Premise (H$_\gamma^t$).** Call an activation occurrence *split* if no block holds it whole while its products'
contributions are computed in pieces. For a uniformly random 2-colouring of the blocks, the expected crossing bits
are at least $\tfrac{\gamma}{2}\beta_x|V_{\rm split}|-\lambda_s$.

At the information level, Lemma 8(ii) gives this when each split activation reaches an exact nonlinear consumer through
linear maps that are injective on its pieces' range. The $K^t$ form is a conjecture of the same kind as the class
premises.

**Proposition 3′ (explicit products, splitting allowed).** Drop the whole-product condition. Each counted product is now
one of three kinds:
- whole in its block;
- region-split, as in Lemma 8(i), at price at most $p_{\max}$;
- value-split on its activation, under (H$_\gamma^t$).

Then

$$
\boxed{\rho\ \ge\ \frac{c}{T_0}\sum_{(i,j)}p_i\ \Big/\ \Big(\min\{h_w,P_{\rm col}\}\frac{h_x}{\beta_x}+k\,p_{\max}
+\frac{2}{\gamma}\,\frac{k\,P_{\rm col}}{\beta_x}\Big).}
$$

At production $k$ with $P_{\rm col}\le h_w$ and $h_x\approx k$, the denominator is
$(1+2/\gamma)P_{\rm col}k/\beta_x+kp_{\max}$. **Splitting costs the bound at most a factor $1+2/\gamma$**, which is $3$
at $\gamma=1$. The exact coverings measured $1.1$–$1.5\times$.

*Proof.* Whole products obey Proposition 3's per-block bound. Region-split products number at most $k$ per block, by
Lemma 8(i). For split activations, $\mathrm{bits}(X)\le Bk$ and (H$_\gamma^t$) give
$\beta_x|V_{\rm split}|\le\tfrac{2}{\gamma}(Bk+\lambda_s)$, and each split activation's products cost at most
$P_{\rm col}$ in total. Add the three kinds and apply Theorem 2. ∎

**Scope (the linear-segment exception).** (H$_\gamma^t$) gives nothing for activations whose products feed only linear
maps before the next product:
- attention's value path given $p$;
- SwiGLU's $u$;
- LoRA's $B(Ax)$;
- the backward pass given the forward activations;
- the weight-gradient sum under SGD;
- linearized training.

Those products must be whole, which keeps them in this class, or be counted on accumulators (§5.2). Weight-split
products (a weight held in shares across blocks) are not covered by Proposition 3′.

### Application to training

**The activation rate must survive replay** (Lemma 9 with $n_0=0$, applied to one selection).
- A block holding $Q$ tokens computes $\theta$'s activations within $t$ and adds the coded fresh residual, $F$ bits per
  token. So $\operatorname{LDL}(a_V)\le Q(h_{\rm tok}+F)+O(1)$ for $V$ = all $d$ coordinates of $Q$ tokens.
- A premise valid for every selected set therefore needs
  $\beta_x\le\beta_x^{\rm rep}:=\min\{\beta_x^{\rm marg},(h_{\rm tok}+F)/d\}$. The held-subnetwork route adds a third
  term when the update below the layer fits in a block.
- The per-coordinate rates in the activation table, $\beta_x^{\rm marg}$, are not this.

**Cheap inputs.** Layers whose inputs are cheap have $F\approx0$ and $\beta_x^{\rm rep}\approx h_{\rm tok}/d=0.0024$:
the first layer, a frozen prefix, linearized training. There the bound is negligible, as it should be, and §5.2's
Corollary 4a applies instead.

**Token-aware upgrade (optional).** A premise that prices a token's $n$ selected coordinates at
$\min\{\beta_x^{\rm marg}n,\ h_{\rm tok}+F\}$ is still replay-consistent. Under it, the binding attack at $1$–$2.5$ MB
is the full-column tile at $\beta_x^{\rm marg}$: it does $20$–$110\times$ more products than the full-token tile at
measured $F$. That gives $\rho\ge c\beta_x^{\rm marg}/(\kappa e h)$, about $0.8$–$1.5\times10^5$ at FP8. This is a
stronger premise than the draft's.

### Parameters

At matched $c=k\varphi$ and $h_x\approx k$:

| Bound | Rate used | FP8, $1$ MB | FP8, $2.5$ MB |
| --- | --- | --- | --- |
| Proposition 3 as drafted, $\varphi\bar p\beta_x/(\kappa k)$ | draft $\beta_x^{\rm marg}=2.6$–$4.9$, $\bar p=2.8$–$5.5$ | $1.0$–$3.6\times10^3$ | $410$–$1{,}440$ |
| Same | $\beta_x^{\rm rep}=0.39$–$0.87$ | $150$–$670$ | $61$–$267$ |
| **Fan-out face (Proposition 3, sharpened)**, $\varphi\beta_x/(\kappa e)$ | $\beta_x^{\rm rep}=0.39$–$0.87$ | $\mathbf{1.2}$–$\mathbf{2.6\times10^4}$ | $\mathbf{1.2}$–$\mathbf{2.6\times10^4}$ |
| Fan-out face, token-aware premise | $\beta_x^{\rm marg}=2.6$–$4.9$ | $0.8$–$1.5\times10^5$ | $0.8$–$1.5\times10^5$ |

Notes on the inputs:
- $\beta_x^{\rm rep}$ uses B1's $F=580$–$1{,}320$ bits per token at the cheapest boundary for LoRA-16 and full
  fine-tuning, at Qwen's $d=1536$. For small adapters it is $1.9$–$3.2$, but those runs belong to §5.4.
- C1 scaled $F$ by $d/1536$ for K2 and V4, which leaves $F/d$ unchanged.
- The fan-out face does not depend on $\bar p$, and at matched $c$ it does not depend on $k$ either.
- With the same $e$, it reproduces `FanOutCoverBound`'s $8{,}900$ at the Lean operating point
  ($0.147\times4.52\times10^{16}/(37{,}248\times2\times10^7)$).

### What remains

- The activation premise at $\beta_x^{\rm rep}$ for every selected set: a joint lower bound, against which B1's coders
  are only upper bounds.
- (H$_\gamma^t$), the time-bounded rectangle bound, is a conjecture. The information-complexity route goes through
  Braverman–Weinstein.
- Weight-split products.
- $F$ is measured on Qwen2.5-1.5B RL checkpoints, short runs. It grows with the steps since registration, so a long run
  needs its own average.

---

## 5.2. Matrix products with explicit accumulation

### Class

These are the circuits with Lean's `Keyed` and `Cube` structure (`Definitions/Nci/CoverPartialSum.lean`):
- every matrix–row–column–token cell holds at most one counted product;
- matrices sharing an activation are stacked;
- each output entry has a disjoint accumulation region whose root is the entry's gate;
- the token and activation-path conditions (`Keyed`) hold.

The structural constants are $M$ products, a weight price $b$, an entry width $a$, a fan-out $e$ (`aOut`) and an
activation-read rate $f$. Two premises hold at block size $s$ and read cap $k$:
- `UpdateHardness`: $b$ bits per distinct weight whose multiplication region is computed whole.
- `ActivationReads`$(f,a)$: a block handed fewer than $fa$ gates of token $t$ completes no entry of $t$, and it is
  handed $f$ gates per distinct activation of $t$ it uses whole.

### Bound

**Proposition 4 (partial-sum slowdown lower bound; `PartialSumCoverBound`).** For every $\mu\in[0,1]$, with
$r=\operatorname{psRate}(b,f,a,e,2k,\mu)$,

$$
r\,(M-kB)\le2kB,
\qquad
\boxed{\rho\ge\frac{M}{T_0}\,\frac{c}{k}\,\frac{r}{r+2}.}
$$

**At $f=1$ the activation premise is a theorem.** `KeyedActivationReads` gives `ActivationReads`$(1,\min\{d-E,n\})$ from
`ColumnsIncompressible`$(b_{\rm col})$, `Wide`$(n)$ and $k<b_{\rm col}V(E+1)$. `PartialSumKeyedCoverBound` is then
Proposition 4 on structure, `UpdateHardness` and `ColumnsIncompressible` alone. At $f>1$, `ActivationReads` is a named
premise.

### Proof (sketch; the full proof is `Proofs/Nci/Cover/PartialSumCover.lean`)

1. *Wiring (`PartialSumsRead`, no premise).* Assign each product to a block that computes its output gate. Then
   $\sum_E(m_E-1)\le\sum_I\#(\mathrm{in}(I)\cap\text{accumulation gates})$. To see it, label each accumulation from its
   root downward: a block computing a gate has all its predecessors, so each new label is a handed gate.
2. *Per block.* Products completed at tokens the block reads $\ge fa$ gates of number at most $(k/b)(k/(fa))$. At the
   other tokens, Loomis–Whitney gives at most $\sqrt{p\,Q\,Y}$, with $bp\le k$, $fQ\le k$ and $Y$ the crossings. The
   fan-out face gives at most $e$ per activation. Products finished without their region total at most $k$ (Lemma
   8(i)).
3. *Across blocks.* Step 1 gives $\sum Y\le3\sum$(accumulation reads). Weight $\mu$ the reads, apply AM–GM per block,
   and sum. ∎

**Corollary 4a (cheap-input accumulators; slowdown lower bound).** Count the weight-gradient products of
$G_{oi}=\sum_tg_{ot}a_{it}$ at layers whose inputs are cheap. Assume premise (R): a block that finishes a product of
token $t$ reads at least $h_{\rm eff}$ bits about $t$, namely $h_{\rm tok}$ plus the fresh backward signal. Charge
$b_{\rm ps}$ bits per partial-sum crossing ($b_{\rm ps}=1$ needs no premise). Then, with $n_E$ the number of entries,

$$
\boxed{\rho\ \ge\ \frac{c\,h_{\rm eff}\,b_{\rm ps}}{\kappa\,k^2}-\frac{c\,b_{\rm ps}\,n_E}{T_0\,k}.}
$$

*Proof.* Block $I$ finishes at most $|E_I|Q_I$ products with $Q_Ih_{\rm eff}\le k$, so
$\sum_I|E_I|\ge Mh_{\rm eff}/k$. By step 1, the crossings number at least $\sum_I|E_I|-n_E$. Each costs $b_{\rm ps}$ of
some block's $k$. ∎

The leaf-and-tree covering attains it. At matched $c$, $h_{\rm eff}=17$ and $b_{\rm ps}=1$ it gives $2.4\times10^3$ at
$1$ MB and $950$ at $2.5$ MB. At $b_{\rm ps}=16$ it gives $3.8\times10^4$ and $1.5\times10^4$.

### Application to training: the linear-segment caveat

`PartialSumsRead` charges crossings of accumulation gates that *exist in $C$*. Accumulation is linear, so by Lemma 8's
exception a circuit computing the same recipe can keep an entry in pieces through the next linear map
($W\sum_cp_c=\sum_cWp_c$) and never materialize it. The class covers circuits that materialize their entries:
containment in Lean's `CoverTraining`, or exactness in its strong form. For a circuit that doesn't, the charge moves to
the entry's next exact nonlinear consumer, where Lemma 8(ii) prices the meeting at the smaller piece's entropy.

What that means by optimizer:
- **Adam** ($G^2$) and **Muon** (a polynomial in $G$) are exact nonlinear consumers, so gradient accumulators are priced
  under them.
- **SGD** is linear: the entries of $G$ can flow into $W'=W-\eta G$ and the next forward pass in pieces, and nothing is
  charged.
- The same holds for attention's value path, SwiGLU's $u$, residual adds and the MoE combine given the gates.

### Parameters

Kimi K2 at $k=2\times10^7$ bits (the module docstring), rescaled from $s=4.52\times10^{16}$ to $c=1.34\times10^{17}$ by
$2.96$:
- $f=1$: $1.6\times10^4\to4.7\times10^4$;
- $f=4$: $6.1\times10^4\to1.8\times10^5$.

The fan-out face alone, `FanOutCoverBound`, gives $8{,}900\to2.6\times10^4$. So partial sums add $1.8\times$ at $f=1$.

### What remains

- `ActivationReads` at $f>1$.
- The conversion from Boolean gates to FLOPs ($s$ and $\alpha_0$ in gates; red2's factor $g_{\rm serve}/g_{\rm train}$).
- Dark accumulations: a dark fraction of about $10^{-4}$–$10^{-3}$ erases the gain over fan-out.
- (R) and $b_{\rm ps}>1$ in Corollary 4a.
- The linear-segment caveat under SGD.

---

## 5.3. Specified matrix outputs

### Class and bound

As drafted: Lemma 5 (recovery), and Proposition 6 with $\operatorname{LDL}_{t,R}(Y_S\mid A)\ge b\,e(S)-\delta-a_{\rm rec}$
and $h=k'+\delta+a_{\rm rec}$. The rounded version uses $\beta_v$ and $h_v=h+\ell_0$. One addition makes the rank
dichotomy explicit:

**Lemma 6a (blockwise rank count).** Suppose every $d$ columns of $A$ are independent. Then
$e(S)=\sum_o\min\{|Q_o|,d\}$, and every block has
- at most $h_v/(\beta_vd)$ rows with $|Q_o|\ge d$, and
- $|S_I|\le h_v/\beta_v$ when $h_v<\beta_vd$, or $|S_I|\le(h_v/(\beta_vd))N+h_v/\beta_v$ otherwise.

Hence

$$
\rho\ \ge\
\begin{cases}
\dfrac{c\,\beta_v}{\kappa\,d\,h_v}, & h_v<\beta_vd\quad(\text{token-proportional}),\\[2ex]
\dfrac{c\,m\,d\,\beta_v}{T_0\,h_v}\,(1+d/N)^{-1}, & h_v\ge\beta_vd\quad(\text{update size only}).
\end{cases}
$$

*Proof.* Each full row contributes $d$ to $e(S_I)\le h_v/\beta_v$, and then carries all $N$ of its tokens. Apply
Theorem 2. ∎

**Proposition 6b (output premise given $R$ alone; slowdown lower bound).** Assume
$\operatorname{LDL}_{t,R}(Y_S)\ge\beta_v^R|S|-\delta$ for every selected set: $A$ is *not* given. Then
$\rho\ge c\beta_v^R/(\kappa dh)$ at every $k$. Replay bounds the premise:
- *Token route.* $\beta_v^Re_{\rm out}\le h_{\rm tok}+F_{\rm out}$ per token, where $e_{\rm out}$ is the stack's output
  width and $F_{\rm out}$ the coded fresh residual at the output boundary.
- *Tile route.* A block holding $R$ rows of $\Delta W$ and $Q$ tokens produces $RQ$ outputs from
  $Rd\beta_w+Q(h_{\rm tok}+F_{\rm in})$ bits. The premise therefore needs
  $k\le k^*:=4d\beta_w(h_{\rm tok}+F_{\rm in})/\beta_v^R$.

This is C1's recommendation (count at $\beta_{gx}$) stated as a theorem. Since $\operatorname{LDL}(Y\mid R)\ge\operatorname{LDL}(Y\mid R,A)$,
the conditional rates are a conservative stand-in for $\beta_v^R$.

### Parameters

The second rank condition, ignoring overheads ($h_v\approx k$), is $k<\beta_vd$:

| Rate | $d=7168$ (V4, K2) | $d=1536$ (Qwen, measured) |
| --- | --- | --- |
| bf16, given $A$: $0.47$–$3.0$ | $0.42$–$2.7$ kB | $0.09$–$0.58$ kB |
| **FP8, given $A$: $0.18$–$0.63$** | $\mathbf{0.16}$–$\mathbf{0.56}$ **kB** | $0.035$–$0.12$ kB |
| NVFP4, given $A$: $0.06$–$0.37$ | $0.05$–$0.33$ kB | $0.012$–$0.071$ kB |
| FP8 raw, wrong conditioning: $5.3$–$11$ | $4.7$–$9.9$ kB | $1.0$–$2.1$ kB |
| FP8 fine-tuning sweep, given $A$: up to $13$ | up to $11.6$ kB | up to $2.5$ kB |

- **Below threshold.** The estimate $\varphi\beta_v/(\kappa d)=2.8$–$9.8\times10^4$ (FP8) holds only at
  $k\le0.56$ kB, where an inference session holds at most about $260$ tokens.
- **What the rates show.** Conditioning on $\theta$'s map on $\theta'$'s inputs removes $94$–$97\%$ of the raw output
  bits ($5.3$–$11\to0.18$–$0.63$). Rounding to FP8 keeps $21$–$38\%$ of bf16's conditional bits, so **rounding removes
  $62$–$79\%$**.
- **The "$\approx90\%$ of the update" comparison doesn't hold.** It sets these rates against $3$–$5$-bit weight codes
  from other runs (Pythia and OLMo, 1,000-step drift). On the same A2 checkpoints $\beta_w$ is $0.06$–$0.19$ at FP8.
- **Proposition 6b's cap.** With $\beta_w=0.06$–$0.19$, $F_{\rm in}=2.7$–$6.2$ kbit (V4-scaled) and
  $\beta_v^R=0.18$–$0.63$, $k^*\approx0.9$–$23$ MB. At $2.5$ MB the premise may or may not survive; this is to be measured.

### What remains

- Either the derived bound below $0.16$–$0.56$ kB, or Proposition 6b's assumed output premise with its cap $k\le k^*$.
- The recovery description $\ell(S)$ under implemented rounding.
- A source lower bound that survives conditioning on $A$.

---

## 5.4. Small persistent state

### Class

As drafted: $n$ steps, a complete retained state of at most $s$ bits (trainable values, optimizer moments, seeds and
generator state), at most $u$ new bits per step, and at most $F$ work per step.

### Bound

**Proposition 7 (execution upper bound, with the packing lemma).** Let

$$
g^*:=\min\Big\{\Big\lfloor\frac{k-s}{u}\Big\rfloor,\ \Big\lfloor\frac{c}{F}\Big\rfloor\Big\}\ \ge1 .
$$

Then there is a compliant execution with $\boxed{\rho=c\lceil n/g^*\rceil/T_0}$, and so, for $T_0\approx nF$ and $n\gg g^*$,

$$
\rho\approx\frac{c}{g^*F}.
$$

There are two cases:
- *Input-bound* ($g^*=\lfloor(k-s)/u\rfloor$): $\rho\approx cu/((k-s)F)$.
- *Work-bound*: $\rho\approx1$.

*Proof.* A block receives the state ($s$ bits) and $g^*$ steps' inputs ($g^*u$ bits), computes $g^*$ steps with the
intermediate states inside, and passes the final state on. Its work is $g^*F\le c$. ∎

**Proposition 7b (data lower bound; slowdown lower bound, every class).** Suppose that for every block, the external
inputs of the steps whose inputs it holds satisfy $\operatorname{LDL}_{t,R}(D_I)\ge u_{\rm eff}\,n_I-\delta_D$ (fresh
data). Then

$$
\boxed{\rho\ \ge\ \frac{c\,n\,u_{\rm eff}}{T_0\,(k'+\delta_D)}\ \approx\ \frac{c\,u_{\rm eff}}{k\,F}.}
$$

So in the input-bound case **Proposition 7 is tight within $\frac{k}{k-s}\cdot\frac{u}{u_{\rm eff}}$**. With seeded or
self-generated data $u_{\rm eff}$ is a seed, and only the work bound applies.

*Proof.* Lemma 1 gives $n_I\le(k'+\delta_D)/u_{\rm eff}$; apply Theorem 2 with steps as items. ∎

### Parameters

Here $s=49$ kB, $u=2.6$ kB and $F=2\times10^{14}$ (9.8 kB arm), or $s=6$ kB (1.2 kB arm):

| Setting | Input-bound $g$ | Work-bound $g$ | $g^*$ | $\rho\approx c/(g^*F)$ | Proposition 7b, $cu/(kF)$ |
| --- | --- | --- | --- | --- | --- |
| $2.5$ MB, matched $c=1.34\times10^{17}$, 9.8 kB arm | $942$ | $670$ | $670$ | $1.00$ | $0.70$ (so $\rho\ge1$) |
| $1$ MB, $c=1.4\times10^{17}$, 9.8 kB arm | $365$ | $700$ | $365$ | $1.92$ | $1.82$ |
| $10$ kB, matched $c$, 1.2 kB arm | $1$ | $2$ | $1$ | $2.7$ | $0.70$ |
| $10$ kB, $c=1.4\times10^{17}$, 1.2 kB arm | $1$ | $700$ | $1$ | $700$ | $182$ |

The draft's "about 300 steps per block, $6\times10^{16}$ useful FLOPs" at $1$ MB is $365$ steps and $7.3\times10^{16}$
FLOPs; $\rho=1.92$ is unchanged.

**When the state does not fit** (the draft's toggle). The $225$–$384$ is the cliff at $S=k^+$:
$4\eta\kappa'(1+F/h)$ with $\kappa'=1/3$ and $\eta=1$, two crossings. It grows about as $8\eta\kappa'(S/k)F/h$ above $k$.
$F$ was measured at $d=1536$; scaled by $d/1536$ for V4 or K2, as C1 does, the cliff for small adapters at FP8 is about
$1{,}050$–$1{,}800$.

### What remains

- A boundary audit of the complete retained state: Adam moments, generator state, and any regenerated values with their
  work.
- The seeded evolution-strategies arm, which tests new-skill acquisition at small $s$.
- P0's registration clause: re-registration resets $n_0$.

---

## Corrected formula list (against the draft as of 22:50Z)

| Draft location | Draft states | Correction | Arithmetic |
| --- | --- | --- | --- |
| At a glance, §5.1 row | $\approx400$–$1{,}400\times$; $1{,}600$–$5{,}800\times$ under the joint premise | **Fan-out face: $\approx1.2$–$2.6\times10^4$**, conditional on the replay-consistent activation premise. The joint premise adds $\le h/(h-P_{\rm col})\approx1.006$ at $2.5$ MB | $\varphi\beta_x/(\kappa e)=6.7\times10^9\cdot(0.39..0.87)/(6\cdot37{,}248)$ |
| §5.1, Proposition 3 | $q=h_wh_x/\beta_x$ | $q=\min\{h_w,P_{\rm col}\}h_x/\beta_x$: no new premise, $72\times$ stronger at $1$ MB and $179\times$ at $2.5$ MB | $h_w/P_{\rm col}=k/(e\bar p)$, $\bar p=3$ |
| §5.1, Corollary 3a | "improves the bound by a factor of four" | Only while $P_{\rm col}\ge h$, i.e. $k\lesssim13$–$26$ kB at FP8. Beyond that, $m(h-m)/\beta_x$ with $m=\min\{h/2,P_{\rm col}\}$ | $P_{\rm col}=e\bar p=1.0$–$2.0\times10^5$ bits |
| §5.1, checks on the assumptions | the column tile matches the joint leading term | True for $h\le2P_{\rm col}$ ($26$–$51$ kB at FP8). Above, the optimal tile is a full column and the joint form overstates a block by $h/(4P_{\rm col})$: $18\times$ at $1$ MB, $45\times$ at $2.5$ MB | as stated |
| §5.1, activation table | $\beta_x=2.6$–$4.9$ (FP8, RL) feeds the estimate | A premise over all selections needs $\beta_x\le(h_{\rm tok}+F)/d=0.39$–$0.87$ (LoRA-16 and full fine-tuning) | B1's $F=580$–$1{,}320$ bits per token at $d=1536$ |
| §5.1, conditional estimates | $1$ MB FP8 $1$–$3.6\times10^3$; $10$ kB $1$–$3.6\times10^5$ | At $\beta_x^{\rm rep}$ the separate form gives $150$–$670$ at $1$ MB; superseded by the fan-out row | `e1_phase2.py` |
| §5.1, application, splitting | "requires an additional accounting argument" | Lemma 8 and Proposition 3′: a factor of at most $1+2/\gamma$, with linear segments and weight splits excepted | as stated |
| §5.2, parameters | $47{,}000$ and $181{,}000$ by rescaling | Arithmetic correct ($\times2.96$). Add that the fan-out face alone gives $2.6\times10^4$ at the same point | $8{,}919\times2.96$ |
| §5.2, application | gradient accumulators "natural", applicability not automatic | Corollary 4a, under (R): $950$ ($b_{\rm ps}=1$) to $1.5\times10^4$ ($b_{\rm ps}=16$) at $2.5$ MB. Priced only under an exact nonlinear optimizer consumer (Adam, Muon), not under SGD | $c\,h_{\rm eff}b_{\rm ps}/(\kappa k^2)$ |
| §5.3, consequences | "rounding hiding about 90% of the update" | Conditioning removes $94$–$97\%$ of the raw output bits. FP8 rounding removes $62$–$79\%$ of bf16's conditional bits. The weight codes compared are from other runs | $0.18/0.47$, $0.63/3.0$ |
| §5.3, threshold | $0.2$–$0.6$ kB | Correct: $0.16$–$0.56$ kB at $d=7168$; $0.035$–$0.12$ kB at the measured $d=1536$ | $\beta_vd/8000$ |
| §5.3, estimate | $3\times10^4$–$10^5$ at FP8 | Holds only below $0.56$ kB (Lemma 6a). At production $k$ it needs Proposition 6b and its cap $k^*\approx0.9$–$23$ MB | $\varphi\beta_v/(\kappa d)=2.8$–$9.8\times10^4$ |
| §5.4, $1$ MB row | about 300 steps, $6\times10^{16}$ FLOPs, $\rho\approx2$ | $365$ steps (input-bound), $7.3\times10^{16}$, $\rho=1.92$; Proposition 7b gives $\ge1.82$ | $\lfloor(8\times10^6-3.92\times10^5)/2.08\times10^4\rfloor$ |
| §5.4, split-state toggle | $\rho\approx225$–$384$ at FP8 | The cliff at $S=k^+$ with $\kappa'=1/3$, at $d=1536$. About $1{,}050$–$1{,}800$ at $d=7168$, growing with $S/k$ | $\tfrac43(1+F/17)$, $F\times7168/1536$ |
| §3, replay paragraph | replay as "a necessary check" | State it as P0 (Lemma 9, Corollary 9a). It fails for the first $\tau/F$ steps after each registration with seeded data: 670 (GRPO, 1.5B), about 24 (frontier small-batch RL) | $1.34\times10^{17}/F$ |
| Lemma 1, Theorem 2, Propositions 4, 6 and 7 | | Correct as stated | |
