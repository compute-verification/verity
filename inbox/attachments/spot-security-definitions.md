Fetched Tue 10 Nov 23:16Z from https://app.notion.com/p/SPOT-security-definitions-413399515d9e8162bbb7f8de48ca23e4 (Daniel's draft, Notion-flavored Markdown as the connector returns it).

Here is the result of "fetch" for the Page with URL https://app.notion.com/p/413399515d9e8162bbb7f8de48ca23e4 as of 2026-11-10T23:16:30.265Z:
<page url="https://app.notion.com/p/413399515d9e8162bbb7f8de48ca23e4">
<ancestor-path></ancestor-path>
<properties>
{"title":"SPOT security definitions"}
</properties>
<iconMetadata>null</iconMetadata>
<content>
This document defines <b>SPOT (Sample-and-PrOve Transform)</b>, which makes it practical to use zero-knowledge proof systems to achieve many important AI security goals by using them to prove the relatively weak statement that a computation was performed <b>mostly correctly</b>.

<table-of-contents color="gray" />

## 1. Overview

Existing argument systems prove that a computation was performed correctly, and to do so the prover must do cryptographic work on every step of it. For computations as large as AI inference at scale, that is too expensive. The idea behind SPOT is that some applications need less. If the harm a cheating prover can do grows with the number of steps it corrupts, as it does when the concern is how much the prover can leak through its outputs or how much work it can skip, then it is enough to know that few steps were corrupted. A random sample can establish that.

In more detail, we divide the computation into small pieces called <i>blocks</i>. The prover commits to every value that passes from one block to another, and we call these values a <i>trace</i>. The verifier then picks a random sample of blocks, and the prover uses an existing argument system to prove that each sampled block maps its committed inputs to its committed outputs. The prover does not know the sample when it commits, so it cannot keep its incorrect blocks out of the sample. If each block is sampled independently with probability $p$, a prover with $k$ incorrect blocks is accepted with probability at most $(1-p)^k$, up to a negligible term. At $p=1\%$, a prover with 500 incorrect blocks is accepted less than $1\%$ of the time, and a prover with five is accepted about $95\%$ of the time.

The saving is that the prover runs the argument system only on the sample, and otherwise only has to commit. The price is a weaker statement. SPOT does not show that the computation is correct. It shows that the prover is committed to a trace in which few blocks are incorrect. Such a trace is an honest execution in which the prover has overwritten the outputs of those blocks, so even one incorrect block can change the final output (Lemma 2.5, Remark 3.17).

Two further points:

1. <b>Interaction is strongly recommended.</b> When the verifier chooses the sample after the prover commits, the prover gets one attempt, and a small sample is enough. SPOT can be made non-interactive by computing the sample from the commitment, but the prover can then recommit in private until the sample misses its incorrect blocks. Making that hopeless takes about 19 times the sample that a $1\%$ error needs (Section 3.4).
2. <b>Zero-knowledge is preserved.</b> If the argument system it builds on is zero-knowledge and the commitment is hiding, the verifier learns nothing about the trace of a prover whose computation ran without error (Claim 3.20).

Below, we:

1. define circuits, blocks and traces (Sections 1 and 2);
2. analyze the sampling check, and define the statement that SPOT proves and the security notions we use (Sections 3.1 to 3.3);
3. present the protocol and explain what interaction buys (Section 3.4);
4. state our security theorems and sketch their proofs, then give a two-stage variant and the costs (Sections 3.5 to 3.7);
5. show what the guarantee means for a cheating prover (Section 4).

<p>
Terminology follows Thaler, <i>Proofs, Arguments, and Zero-Knowledge</i>, and Goldreich, <i>Foundations of Cryptography</i>, Volume 1, with adaptations recorded in Appendix C. All of the math is machine-checked in Lean (link to follow). 
</p>

## 2. Preliminaries

### 2.1. Circuits

<b>Definition 1.1 (circuit).</b> Fix a finite alphabet $\Sigma$. A circuit $C$ is a finite directed acyclic graph of nodes, each taking a value in $\Sigma$. A node with no in-neighbors is an input. Every other node is a gate. A gate $v$ with $d_v$ in-neighbors computes a function $f_v:\Sigma^{d_v}\to\Sigma$ of their values, taken in a fixed order. Some gates are designated as output gates. Given values $z$ at the inputs, evaluating the gates in topological order gives every gate a value, and $C(z)$ denotes the values at the output gates.

For a set of gates $S\subseteq C$:

- $\mathrm{in}(S)$ is the set of nodes outside $S$ that are in-neighbors of a gate in $S$.
- $\mathrm{out}(S)$ is the set of gates in $S$ that are output gates or in-neighbors of a gate outside $S$.
- $C_S$, the subcircuit induced by $S$, is the circuit with gates $S$, inputs $\mathrm{in}(S)$, and output gates $\mathrm{out}(S)$. Its gates keep their in-neighbors and functions.

For a collection $T$ of sets of gates, $\mathrm{out}(T)$ denotes the union of $\mathrm{out}(S)$ over $S\in T$.

### 2.2. Blocks and traces

Sampling is useful only if a small part of the computation can be checked without the rest. We therefore partition the gates into blocks and have the prover claim, in a trace, every value that passes into or out of a block. Given the trace, a block can be checked on its own, by evaluating it on the values claimed at its inputs and comparing the result with the values claimed at its outputs.

<b>Definition 2.1 (blocks).</b> Fix a circuit and a partition of its gates. The parts are called blocks. From here on, $C$ denotes a circuit together with its blocks. A node is a boundary node if it lies in $\mathrm{in}(S)\cup\mathrm{out}(S)$ for some block $S$. Every output gate is a boundary node, and so is every input that some block reads.

<b>Definition 2.2 (trace).</b> A trace $\tau$ assigns a value in $\Sigma$ to every boundary node. $\tau_G$ denotes its values on a set $G$ of boundary nodes. A partial trace assigns values to some of the boundary nodes.

<b>Definition 2.3 (correct).</b> A block $S$ is correct in $\tau$ if $C_S$, evaluated on the values at $\mathrm{in}(S)$, gives the values at $\mathrm{out}(S)$:

$$
C_S\big(\tau_{\mathrm{in}(S)}\big)=\tau_{\mathrm{out}(S)}
$$

Otherwise $S$ is incorrect in $\tau$. In a partial trace, $S$ is also incorrect if a value at $\mathrm{in}(S)\cup\mathrm{out}(S)$ is missing. $T_\tau$ denotes the set of blocks that are incorrect in $\tau$. A trace is fully correct if it has no incorrect blocks.

Note that a block is judged on the inputs the trace claims for it, even when those are wrong. For example, let one block compute $a=x$ and a second compute $b=\neg a$. In the trace with $x=0$, $a=1$ and $b=0$, only the first block is incorrect, because the second correctly negates the value claimed for $a$. One incorrect block can thus change every value computed after it, so the number of incorrect blocks does not measure how far a trace is from the honest execution. A trace is fully correct, however, exactly when it is the trace of the evaluation of $C$ on its values at the inputs.

Partial traces arise in Section 3, where the trace recovered from a prover may be missing values. Such a trace extends to a full one with no more incorrect blocks: give each missing node an arbitrary value, and a block that was correct, having had no value missing, stays correct.

We now show that all a prover gains from incorrect blocks is the freedom to choose their outputs. To state this, we need the circuit in which those outputs have become inputs.

<b>Definition 2.4 (overwriting).</b> For a set $T$ of blocks, $C[T]$ is $C$ with every gate in $\mathrm{out}(T)$ replaced by an input. A replaced gate that was an output gate remains a designated output.

<b>Lemma 2.5.</b> If every block outside $T$ is correct in $\tau$, then $\tau$ agrees at every boundary node with the evaluation of $C[T]$ on $\tau$'s values at the inputs and at $\mathrm{out}(T)$.

That is, a trace whose incorrect blocks are $T$ is an honest evaluation of $C[T]$. Once the prover has chosen $T$, the circuit's inputs and the $|\mathrm{out}(T)|$ values at $\mathrm{out}(T)$, every other value in the trace is determined. In particular, an incorrect block's output is counted once, where it is produced, and not again at the blocks that read it.

## 3. SPOT

SPOT turns a base argument, an argument of knowledge that one block is correct, into an argument of knowledge that fewer than $k$ blocks of a whole trace are incorrect. It does so in the usual two steps. The first is a check that needs no cryptography but assumes that the prover's trace is fixed in advance: the verifier samples some blocks, reads the values on their boundaries, and recomputes them. Section 3.1 analyzes this check as a PCP whose proof string is the trace. A real prover's trace is not fixed, and when asked about a block it could invent values that pass. The second step therefore has the prover commit to its trace with a string-commitment scheme before it learns the sample, and prove with the base argument that the sampled blocks are correct in the trace it committed to (Sections 3.2 to 3.4).

<b>Conventions.</b> The statements of this section use a security parameter $\lambda$ and public parameters:

1. <b>Efficiency.</b> Efficient means polynomial time in $\lambda$ and the length of the instance, and negligible means smaller than the reciprocal of every polynomial in $\lambda$.
2. <b>Public parameters.</b> Those of the commitment scheme and of the base argument are sampled first. They are known to every party, and to the extractors and simulators of the definitions below, and they are part of every view.
3. <b>Order and probabilities.</b> The circuit $C$ is fixed independently of the public parameters, and the prover sees them before it chooses its commitment. All probabilities are taken over their choice as well.

### 3.1 The PCP

<b>Definition 3.1 (sampling design).</b> A sampling design $\mathcal D$ is an efficiently sampleable distribution over sets $Q$ of blocks. It is public, like $C$.

<b>Definition 3.2 (sampling PCP).</b> The proof string is a trace $\tau$. The verifier $\mathcal V$ is given $C$ and oracle access to $\tau$. It draws $Q\sim\mathcal D$, queries $\tau$ at $\mathrm{in}(S)\cup\mathrm{out}(S)$ for each $S\in Q$, and accepts iff every $S\in Q$ is correct in $\tau$. Hence, for every trace,

$$
\Pr\big[\mathcal V^{\tau}(C)=\text{accept}\big]=\Pr_{Q\sim\mathcal D}\big[Q\cap T_\tau=\varnothing\big]
$$

<b>Proposition 3.3 (completeness and soundness of the PCP).</b> The sampling PCP has perfect completeness: it accepts a fully correct trace with probability 1. For every $k$, it accepts a trace with at least $k$ incorrect blocks with probability at most $\delta_s(k)$, where

$$
\delta_s(k)=\max_{|T|=k}\ \Pr_{Q\sim\mathcal D}\big[Q\cap T=\varnothing\big]
$$

and $\delta_s(k)=0$ when $k$ exceeds the number of blocks. If $\mathcal D$ includes each block independently with probability $p$, then $\delta_s(k)=(1-p)^k$ for $k$ up to the number of blocks.

A PCP's soundness is usually stated for instances that have no valid proof. Every circuit has a fully correct trace, so here it is stated for proof strings.

The trace is not encoded, so an incorrect block is caught only if that block is sampled. Proof systems usually avoid this weakness by checking an encoding of the values, in which one altered value changes almost every symbol, as Thaler explains for the GKR protocol (§4.6.5 of his book). We accept the weakness, and with it a $\delta_s(k)$ that is small only for large $k$, because the prover then never has to encode the trace.

<b>Remark 3.4 (how large </b><b>$k$</b><b> must be).</b> Suppose $\mathcal D$ includes each block independently with probability $p$. Then $\delta_s(k)=(1-p)^k\le e^{-pk}$, so $\delta_s(k)\le 2^{-b}$ once $k\ge b\ln 2/p$. An error of $2^{-128}$ needs $k\approx 89/p$, and an error of $1\%$ needs $k\approx 4.6/p$. So at a fixed sampling rate, a cryptographically small error means tolerating about 19 times as many incorrect blocks as a $1\%$ error. Since the expected sample is $p$ times the number of blocks, excluding a fraction $\varphi$ of incorrect blocks with error $2^{-128}$ takes an expected sample of about $89/\varphi$ blocks, whatever the size of the computation. Section 3.4 explains when the smaller error is needed.

### 3.2 Commitments

The prover commits to its trace with a string-commitment scheme, which is more commonly called a vector commitment. The string is the trace, with one position for each boundary node.

<b>Definition 3.5 (string-commitment scheme).</b> A string-commitment scheme consists of three efficient algorithms:

- Commit takes a string $s$ over $\Sigma$ and returns a commitment $c$ and opening information $d$. It is randomized, and the committer keeps $d$.
- Open takes $d$ and a position $i$, and returns the value $s_i$ with an opening $o$.
- Verify takes $c$, a position $i$, a value $v$ and an opening $o$, and accepts or rejects.

The scheme is correct if Verify accepts whatever Open returns for an honestly generated commitment. An opening of $c$ at a set of positions consists of an opening at each of them, and it is accepted if Verify accepts each one.

<b>Definition 3.6 (binding).</b> A string-commitment scheme is binding if no efficient algorithm outputs, except with negligible probability, a commitment $c$, a position $i$ and accepted openings of $c$ at $i$ to two different values.

<b>Definition 3.7 (hiding).</b> A string-commitment scheme is statistically hiding if, for any two strings of the same length, the public parameters together with a commitment to the first are within negligible statistical distance of the public parameters together with a commitment to the second. In explicit form, for fixed public parameters, it is hiding with error $\varepsilon_h$ if the commitments to any two such strings are within statistical distance $\varepsilon_h$.

Hiding is needed only for zero-knowledge. Appendix A describes the scheme we use, a Merkle tree whose leaves are hiding commitments.

### 3.3 The relation and the security definitions

What statement should the protocol prove? The obvious candidate, that $C$ has a trace with fewer than $k$ incorrect blocks, is vacuous: every circuit has a fully correct trace, and anyone can compute one by evaluating $C$ on inputs of their choice. The statement has to concern the particular trace that the prover committed to, so the commitment must be part of the instance.

Nor is it enough that a suitable opening of the commitment exists, because a commitment may have openings that no efficient party can find. What matters is that the prover knows one, so we define a relation and ask for an argument of knowledge.

<b>Definition 3.8 (the relation </b><b>$\mathcal R_k$</b><b>).</b> An instance is a pair $(C,c)$: a circuit with its blocks, and a commitment $c$. A witness is a set $o$ of accepted openings of $c$ at distinct boundary nodes. The opened values form a partial trace $\tau_o$. For an integer $k\ge 1$, the relation $\mathcal R_k$ is the collection of pairs in which that partial trace has fewer than $k$ incorrect blocks:

$$
\mathcal R_k=\big\{\big((C,c),o\big)\ :\ |T_{\tau_o}|<k\big\}
$$

Throughout, $(\mathcal P,\mathcal V)$ is an argument system for a relation $\mathcal R$. Definitions 3.9 to 3.11 follow the standard ones, and Appendix C records each departure.

<b>Definition 3.9 (completeness).</b> $(\mathcal P,\mathcal V)$ has completeness error $\delta_c$ if, for every $(x,w)\in\mathcal R$, the verifier accepts with probability at least $1-\delta_c$ when it interacts with the prescribed prover $\mathcal P$ holding $w$. Perfect completeness is the case $\delta_c=0$.

<b>Definition 3.10 (knowledge-soundness).</b> Let $\kappa$ assign to each instance a number in $[0,1]$. $(\mathcal P,\mathcal V)$ is knowledge-sound with knowledge error $\kappa$ if there is a probabilistic algorithm $\mathcal E$, the extractor, with the following property. For every efficient prover strategy $\mathcal P'$ there is a negligible function $\nu$ such that, for every instance $x$, the probability that the verifier accepts $x$ while $\mathcal E$'s output is not a witness for $x$ is at most $\kappa(x)+\nu(\lambda)$. The extractor is given $x$ and the verifier's internal randomness. It may rewind $\mathcal P'$: that is, rerun it on verifier messages of its choice, with the prover's internal randomness from the execution in question held fixed. It runs in expected polynomial time, counting each rerun as one step. The probability is over the public parameters and the randomness of the prover, the verifier and the extractor. The prover may choose part of the instance after seeing the public parameters, as it chooses the commitment here. A knowledge-sound argument system is an argument of knowledge.

The knowledge error of Protocol 1 will be large, because a sample can miss every incorrect block. Protocol 1 will also be complete for one relation and knowledge-sound for another, because an honest prover has a fully correct trace, whereas a cheating prover can be held only to fewer than $k$ incorrect blocks. For relations $\mathcal R\subseteq\mathcal R'$, $(\mathcal P,\mathcal V)$ is an argument of knowledge for the pair $(\mathcal R,\mathcal R')$ if it has negligible completeness error for $\mathcal R$ and is knowledge-sound for $\mathcal R'$.

<b>Definition 3.11 (zero-knowledge).</b> $(\mathcal P,\mathcal V)$ is statistical zero-knowledge with respect to auxiliary input if for every efficient verifier strategy $\hat{\mathcal V}$ there are an algorithm $\mathrm{Sim}$, the simulator, that runs in expected polynomial time, and a negligible function $\nu$, such that for every $(x,w)\in\mathcal R$ and every string $z$, the statistical distance between the view $\mathrm{View}_{\hat{\mathcal V}}\big(\mathcal P(x,w),\hat{\mathcal V}(x,z)\big)$ and the output $\mathrm{Sim}(x,z)$ is at most $\nu(\lambda)$. Here $\mathrm{View}_{\hat{\mathcal V}}$ denotes the verifier's internal randomness together with the messages it receives from $\mathcal P$, and $z$ is an auxiliary input known to the verifier. Running times are measured in $|x|$ and $\lambda$, whatever the length of $z$.

The definition quantifies over every efficient verifier strategy, so it covers dishonest verifiers, and the auxiliary input is what lets it survive sequential composition. Protocol 1 needs both: it is interactive, and it runs the base argument once for each sampled block.

### 3.4 The protocol

The base argument proves a commit-and-prove statement about one block $S$: the prover knows openings of $c$ on the boundary of $S$ under which $S$ is correct.

<b>Definition 3.12 (base argument).</b> Fix $C$. The block relation $\mathcal R_{\mathrm{blk}}$ is the collection of pairs $\big((c,S),(v,o)\big)$ in which $c$ is a commitment, $S$ is a block, $o$ is an accepted opening of $c$ at the nodes of $\mathrm{in}(S)\cup\mathrm{out}(S)$ to the values $v$, and $S$ is correct on those values:

$$
C_S\big(v_{\mathrm{in}(S)}\big)=v_{\mathrm{out}(S)}
$$

The base argument is an argument system for $\mathcal R_{\mathrm{blk}}$ that has negligible completeness error and is knowledge-sound with negligible knowledge error. For Claim 3.20 it is also statistical zero-knowledge with respect to auxiliary input (Definitions 3.9 to 3.11).

<b>Protocol 1.</b> The public input is $C$ and a sampling design $\mathcal D$. The prover holds a trace $\tau$.

1. The prover commits to $\tau$ and sends the commitment $c$.
2. The verifier draws $Q\sim\mathcal D$ and sends it.
3. For each $S\in Q$ in turn, the prover and verifier run the base argument on the instance $(c,S)$, with fresh randomness for each run.

The verifier accepts iff every run of the base argument accepts.

Once the prover has sent $c$, the pair $(C,c)$ is an instance of $\mathcal R_k$, and steps 2 and 3 are an argument system for it.

Protocol 1 is Kilian's argument system applied to the sampling PCP, with one change. Kilian's prover would reveal the queried values with their openings, and the verifier would recompute each sampled block. Our prover instead proves, with the base argument, that it knows such openings. The verifier therefore never sees the values and never recomputes a block, and its cost for each sampled block is that of the base argument's verifier.

<b>Interaction.</b> Protocol 1 is interactive in two places, and the reasons differ.

1. <b>The sample in step 2 comes from the verifier, so that a small sample suffices.</b> A prover whose trace has $k$ incorrect blocks escapes a sample with probability at most $\delta_s(k)$. Because the verifier draws the sample after it holds the commitment, the prover gets a single attempt, made in the verifier's sight. If the sample were computed from the commitment instead, as in the Fiat-Shamir transformation, the prover could recommit in private until the sample missed its incorrect blocks, and after $T$ attempts it would succeed with probability about $T\,\delta_s(k)$. Since $T=2^{64}$ is within reach, only bounds with $\delta_s(k)$ near $2^{-128}$ would then mean anything, and by Remark 3.4 those take about 19 times the sample that an error of $1\%$ needs.
2. <b>The base argument in step 3 need not be interactive.</b> Its error is already negligible, so private attempts gain the prover nothing, and a non-interactive argument could take its place, leaving three messages. We keep it interactive, at the cost of running its rounds once for each sampled block, because the alternative would be proved secure only in the random oracle model. Parties such as states want as few assumptions as possible and can afford to interact.

### 3.5 Security guarantees

Completeness and zero-knowledge are for a prover with a fully correct trace, and soundness tolerates up to $k-1$ incorrect blocks, for every $k$. We state soundness twice, because the standard statement is not the one we use. The standard statement, Theorem 3.18, is that an extractor outputs a witness for $\mathcal R_k$. That witness may depend on the verifier's randomness, so the theorem does not show that the prover was held to anything before the sample was drawn. Theorem 3.15 and Corollary 3.16 do show this. They concern a trace that is fixed before the sample is drawn, and Section 4 and the bound for repeated audits rest on them.

$\mathcal V^{\mathcal P'}(C,r)$ denotes the verifier's decision when its internal randomness is $r$ and it interacts with a prover strategy $\mathcal P'$.

<b>Proposition 3.13 (completeness).</b> Suppose the base argument has completeness error $\delta_c$. A prover holding a fully correct trace is accepted by Protocol 1 except with probability at most $q\,\delta_c$, where $q$ is the largest number of blocks $\mathcal D$ can sample. So Protocol 1 has completeness error at most $q\,\delta_c$ for $\mathcal R_1$.

A hardware fault can leave an honest prover with an incorrect block, which no verifier can tell from a cheat. A prover with $f$ faulted blocks is therefore rejected with probability $1-(1-p)^f$ under independent sampling, and what follows is policy outside the protocol.

To say that the prover is bound to a trace before the sample is drawn, we need a trace to point to. The commitment does not supply one: binding says that the prover cannot open a position in two ways, not that $c$ determines a string. So we define the trace by what the prover would open. Fix the prover's strategy and its randomness, and take, for each boundary node, the value to which the prover is most likely to open that node, over the verifier's randomness. The prover opens nothing in Protocol 1, so "open" has to mean that the base argument's extractor recovers an opening.

<b>Definition 3.14 (committed trace).</b> Fix $C$, the public parameters, a prover strategy $\mathcal P'$ for Protocol 1 and a value $\omega$ of its internal randomness. For verifier randomness $r$, apply the base argument's extractor, with its own randomness $\zeta$, to each run of the base argument in step 3. A boundary node $u$ is opened to $\sigma$ on $(r,\zeta)$ if, for some run that accepts, the extraction outputs an accepted opening of the commitment $c$ at $u$ to $\sigma$. The execution as a whole need not accept. The committed trace $\hat\tau_\omega$ assigns to each boundary node $u$ the value $\sigma$ that maximizes $\Pr_{r,\zeta}[u\text{ is opened to }\sigma\text{ on }(r,\zeta)]$, with ties broken by a fixed order on $\Sigma$, and no value if that probability is zero for every $\sigma$. So $\hat\tau_\omega$ is a partial trace.

Neither party computes the committed trace. It matters because it depends only on the public parameters, the prover's strategy and $\omega$, all of which are fixed before the sample is drawn.

<b>Theorem 3.15 (the prover is bound to the committed trace).</b> Suppose the commitment scheme is binding and the base argument is knowledge-sound, both in the explicit forms of Definition B.1 and with negligible errors. Let $Q(r)$ be the sample the verifier draws when its internal randomness is $r$. Then for every efficient prover strategy $\mathcal P'$ for Protocol 1 there is a negligible $\varepsilon$ such that

$$
\Pr_{\omega,r}\Big[\mathcal V^{\mathcal P'}(C,r)=\text{accept}\ \wedge\ Q(r)\cap T_{\hat\tau_\omega}\neq\varnothing\Big]\le\varepsilon
$$

That is, except with probability $\varepsilon$, Protocol 1 accepts only if the PCP verifier would accept the committed trace on the same sample. The reason is that an accepting run of the base argument on $S$ yields, by knowledge-soundness, openings on the boundary of $S$ under which $S$ is correct, and binding forces those openings to agree with the committed trace. Theorem B.2 gives the concrete bound for the instantiation of Appendix A.

<b>Corollary 3.16 (the main guarantee).</b> For every $k$,

$$
\Pr_{\omega,r}\Big[\mathcal V^{\mathcal P'}(C,r)=\text{accept}\ \wedge\ \big|T_{\hat\tau_\omega}\big|\ge k\Big]\le\delta_s(k)+\varepsilon
$$

The corollary follows because the committed trace does not depend on the verifier's randomness. If the trace has at least $k$ incorrect blocks, the sample misses them all with probability at most $\delta_s(k)$, and by Theorem 3.15 the sample contains one while the verifier accepts with probability at most $\varepsilon$. Since $k$ plays no part in Protocol 1, one execution gives the bound for every $k$ at once.

<b>Remark 3.17 (no value is enforced).</b> $\mathcal R_k$ bounds the number of incorrect blocks and says nothing about the value at any node. A check computed inside the circuit can therefore fail without any block being incorrect, because a block that correctly computes a failing check is a correct block. To enforce that a boundary node takes a given value, the verifier must check that value against the commitment in every execution, by an opening or by a proof.

A design that samples a block in every execution does enforce its correctness: by Theorem 3.15, the block is correct in the committed trace whenever the verifier accepts, except with probability $\varepsilon$. It is correct on its claimed inputs, however, and to enforce those, the blocks that produce them must always be sampled as well.

<b>Theorem 3.18 (SPOT yields an argument of knowledge).</b> Suppose the hypotheses of Theorem 3.15, and that the base argument has negligible completeness error. Then for every $k$, Protocol 1 is an argument of knowledge for the pair $(\mathcal R_1,\mathcal R_k)$, with knowledge error $\min\{1,\alpha^2\,\delta_s(k)\}$, where $\alpha^2\approx 1.00135$ is a constant from the extractor's analysis.

The extractor rewinds the prover with its internal randomness held fixed, so that the commitment $c$ is the same in every rerun. It applies the base argument's extractor to each accepted rerun and outputs the openings of $c$ that it obtains. Theorem B.3 gives the concrete bound and the extractor's cost for the instantiation of Appendix A.

Now suppose that Protocol 1 is run many times against a prover that adapts to what it has seen. The next result bounds the number of incorrect blocks the prover can accumulate before it is first caught.

<b>Claim 3.19 (repeated audits).</b> An audit is one execution of Protocol 1. Fix the circuits and sampling designs of audits $1,\dots,n$ in advance. One prover strategy plays all $n$ audits in order, so each audit may depend on everything before it, and the audits do not stop at a rejection. Audit $t$ is hit if its sample contains a block that is incorrect in its committed trace. Suppose every design misses each set of $m$ blocks with probability at most $\theta^m$, for some $\theta\in[0,1]$. Then the probability that some hit audit is accepted, or that the committed traces of the audits before the first hit one have at least $M$ incorrect blocks in total, is at most

$$
\sum_{t=1}^{n}\varepsilon_t+\theta^{M}
$$

where $\varepsilon_t$ is the error of Theorem 3.15 for audit $t$.

Here the prover is caught at the first hit audit, which rejects except with probability $\sum_t\varepsilon_t$. The bound $\theta^M$ on accumulating $M$ incorrect blocks before then is the bound that a single audit gives for $M$ incorrect blocks, so spreading them over many audits, or choosing them adaptively, does not help. With independent sampling at rate $p$, $\theta=1-p$.

<b>Claim 3.20 (SPOT preserves zero-knowledge).</b> Suppose the commitment scheme is statistically hiding, and the base argument is statistical zero-knowledge with respect to auxiliary input. Then so is Protocol 1, in this sense: when the prover holds a fully correct trace, the view of any efficient verifier strategy, including the commitment $c$, can be simulated from $C$ and the verifier's auxiliary input. Explicitly, if the commitment scheme is hiding with error $\varepsilon_h$ and each run of the base argument is zero-knowledge with error $\varepsilon_b$, the simulation error is at most $\varepsilon_h$ plus $\varepsilon_b$ for each run. Nothing is claimed for a prover whose trace has incorrect blocks.

<b>Proof sketch.</b> The simulator commits to a fully correct trace of its own, computed on arbitrary inputs. It lets the verifier choose its sample, and for each sampled block in turn it runs the base argument's simulator, giving it the verifier's state so far as auxiliary input. To compare its output with the real view, change the real execution in two hybrid steps, in this order. First replace each run of the base argument by its simulation, leaving the real commitment in place. Each replacement changes the view by at most $\varepsilon_b$. After that, nothing beyond the commitment depends on the witness. Then replace the real commitment by a commitment to the simulator's trace, which changes the view by at most $\varepsilon_h$. The sample is part of the state that is passed on, so a verifier that chooses it after seeing $c$ is covered.

The restriction to a fully correct trace is the usual scope of zero-knowledge, which protects a prover that follows the protocol and whose computation ran without error. Zero-knowledge hides the witness and not the instance, so the circuit and its blocks are public.

### 3.6 Two stages

In Protocol 1 the prover commits to every value on every block boundary, which costs time proportional to the whole computation however few blocks are sampled. The two-stage variant reduces this cost. The prover first commits only to the boundaries of larger units, called replay blocks, each of which is a set of blocks. Once the verifier has sampled some replay blocks, the prover replays them, recomputing the inputs and outputs of the blocks inside, and commits to those values as well. The verifier then samples blocks inside the sampled replay blocks, as in Protocol 1, which is the special case in which every replay block is a single block. The price is a weaker guarantee, described after Claim 3.23.

<b>Definition 3.21 (two-stage design).</b> A two-stage design consists of a partition of the blocks into replay blocks, a distribution $\mathcal D_0$ over sets of replay blocks, and for each replay block $R$ a distribution $\mathcal D_R$ over sets of blocks of $R$. The interior of $R$ is the set of boundary nodes that lie in $R$ but not in $\mathrm{out}(R)$. The two-stage sampling IOP is the following two-round public-coin IOP:

1. The prover sends the trace at the nodes in $\mathrm{in}(R)\cup\mathrm{out}(R)$ for each replay block $R$.
2. The verifier draws a set of replay blocks from $\mathcal D_0$.
3. The prover sends the trace at the interiors of the sampled replay blocks.
4. For each sampled replay block $R$, the verifier draws $Q_R\sim\mathcal D_R$, independently across replay blocks. It accepts iff every block in $Q=\bigcup_R Q_R$ is correct.

Each $\mathcal D_R$ is fixed in advance: it does not depend on the sample of step 2 or on the prover's messages. For a two-stage design, $\mathcal D$ denotes the distribution of the set $Q$ of step 4, over the randomness of both steps 2 and 4, and $\delta_s(k)$ is computed from it as in Proposition 3.3.

<b>Claim 3.22.</b> In the two-stage sampling IOP, choosing interiors after step 2 does not increase the prover's best acceptance probability. For a fixed first message, that probability is attained by fixing the best interiors in advance. This holds because each $\mathcal D_R$ is fixed in advance and step 4 samples independently across replay blocks.

Claim 3.22 is what allows $\delta_s(k)$ to be computed from $\mathcal D$ as if the whole trace were fixed in advance.

<b>Protocol 2.</b> The public input is $C$ and a two-stage design. The prover holds a trace $\tau$.

1. The prover commits to $\tau$ at the nodes in $\mathrm{in}(R)\cup\mathrm{out}(R)$ for each replay block $R$, and sends the commitment $c_1$.
2. The verifier draws a set of replay blocks from $\mathcal D_0$ and sends it.
3. For each sampled replay block $R$, the prover commits to $\tau$ at the interior of $R$ and sends the commitment $c_R$.
4. The verifier draws $Q$ as in step 4 of Definition 3.21 and sends it.
5. For each $S\in Q$ in turn, the prover and verifier run the base argument on the instance $\big((c_1,c_R),S\big)$, where $R$ is the replay block containing $S$.

The verifier accepts iff every run of the base argument accepts. In step 5 the block relation of Definition 3.12 is used with a pair of commitments, and each node is opened under the one that commits to it.

<b>Claim 3.23 (guarantees of Protocol 2).</b>

1. A prover holding a fully correct trace is accepted except with probability at most $q\,\delta_c$, as in Proposition 3.13.
2. The extractor of Theorem 3.18, run on Protocol 2, outputs the openings of $c_1$ that it obtains, at nodes in $\mathrm{in}(R)\cup\mathrm{out}(R)$ for replay blocks $R$. It need not obtain one at every such node, and it discards openings of the commitments $c_R$. Let $w$ be the partial trace that the opened values form. For every $k$, the probability that the verifier accepts while $\min_\iota|T_{w\cup\iota}|\ge k$ is at most $\min\{1,\alpha^2\,\delta_s(k)\}$ plus a negligible term, with $\delta_s(k)$ computed from the two-stage design.

The guarantee of Protocol 2 is weaker than that of Protocol 1 in four respects.

1. Interiors are existentially quantified. The minimum is over assignments $\iota$ of values to the interiors of all replay blocks. So the statement is that the opened boundary values are consistent with some interiors under which fewer than $k$ blocks are incorrect. It says nothing about an actual execution inside a replay block that was not sampled, because the prover never produces one. A replay-block boundary node with no opening makes every block that reads or writes it incorrect.
2. The relation cannot be checked efficiently. In the terms of Definition 3.10, Protocol 2 is knowledge-sound, with knowledge error $\min\{1,\alpha^2\,\delta_s(k)\}$, for the relation whose instances are pairs $(C,c_1)$ and whose witnesses are such sets of openings. Whether a given set of openings is a witness cannot in general be decided efficiently, because the condition asks for the existence of interiors.
3. There is no analogue of Theorem 3.15. The extracted openings may depend on the verifier's randomness, so we have no bound for repeated audits.
4. Zero-knowledge is not claimed.

### 3.7 Costs

1. <b>Prover.</b> Produce the trace, commit to it, and run the base argument on each sampled block. With Protocol 1 the first two costs are proportional to the whole computation, whatever the sampling rate. Protocol 2 still commits to every replay-block boundary, and confines the rest to the sampled replay blocks.
2. <b>Verifier.</b> Draw the sample and verify one run of the base argument per sampled block.
3. <b>Communication.</b> One commitment in Protocol 1. In Protocol 2, one commitment for the replay-block boundaries and one for each sampled replay block. In both, the transcripts of the base argument.
4. <b>Rounds.</b> One for the sample, then those of the base argument for each sampled block.

If proving block $S$ with the base argument costs $t_S$, the expected proving cost is $\sum_S\Pr[S\in Q]\,t_S$, which is $p\sum_S t_S$ when each block is sampled independently with probability $p$. So the sampling design sets both the soundness error and the expected work.

The sampling design and the blocks are chosen to minimize these costs for a target $\delta_s(k)$. Block size matters because smaller blocks make each incorrect block worth less to the prover. If the base argument's proving cost is roughly linear in block size, smaller blocks can therefore give a stronger statement for the same proving budget, up to the point where the cost of proving a block's openings dominates.

## 4. Applications

By Corollary 3.16, the verifier in Protocol 1 is unlikely to accept unless the prover's committed trace has fewer than $k$ incorrect blocks. Such a trace extends to a full trace with no more incorrect blocks (Section 2), which by Lemma 2.5 is an honest execution in which the prover chose the outputs of fewer than $k$ blocks. This limits three things a cheating prover might want: what it can encode in its outputs, what it can inject into a part of the computation, and how much work it can skip.

The limits hold for one fixed $C$, so any influence the prover has over $C$ must be accounted for separately. They also concern the input and output nodes of the trace, and applying them to the inputs and outputs of a real system takes a mechanism outside SPOT that ties those values to those nodes.

<b>Proposition 4.1 (exfiltration capacity).</b> Fix the values at the inputs. If $\tau$ has those input values and incorrect blocks $T$, its values at the output gates are among the at most $|\Sigma|^{|\mathrm{out}(T)|}$ outputs of $C[T]$. So if $\tau$ has fewer than $k$ incorrect blocks, the output values lie in a fixed set of size at most

$$
\sum_{|T|<k}|\Sigma|^{|\mathrm{out}(T)|}
$$

The logarithm of this sum bounds the exfiltration capacity at threshold $k$: the number of bits the prover can encode in the outputs, for fixed inputs, through its choice of incorrect blocks and of their values. Roughly, this is the choice of $T$ plus $\log_2|\Sigma|$ bits for each overwritten value. The bound does not limit what the outputs reveal about the inputs, because one incorrect block can change which function of the inputs the correct blocks after it compute, and that function may reveal more than the honest one does. It also covers output gates only, so every value that can be observed from outside the system must be an output gate.

In an honest execution, a set $I$ of gates depends on the rest of the circuit only through the values $\mathrm{in}(I)$ that it reads. The next proposition bounds how many more values a cheating prover can feed into $I$.

<b>Proposition 4.2 (infiltration capacity).</b> For every set $T$ of blocks and every set of gates $I$, the part of $I$ that remains in $C[T]$ reads at most $|\mathrm{out}(T)\cap I|$ more values than $I$ reads in $C$:

$$
\big|\mathrm{in}_{C[T]}\big(I\setminus\mathrm{out}(T)\big)\big|\ \le\ |\mathrm{in}(I)|+|\mathrm{out}(T)\cap I|
$$

where $\mathrm{in}_{C[T]}$ is taken in $C[T]$. The largest value of $|\mathrm{out}(T)\cap I|$ over sets $T$ of fewer than $k$ blocks is the infiltration capacity of $I$ at threshold $k$: an upper bound on the number of values a prover can inject into $I$ beyond what the circuit lets it read.

<b>Remark 4.3 (skippable work).</b> Assign each block a cost, and suppose that producing correct outputs for any set of blocks costs at least the sum of their costs. Suppose also that every block's boundary is committed while the computation runs, which Protocol 1 does not itself require. Then the only blocks whose work the prover can have skipped are the incorrect ones, so the skippable work at threshold $k$ is at most the largest total cost of fewer than $k$ blocks. This reasoning does not extend to Protocol 2, in which interiors are committed only after the sample is drawn.

## Appendix A. The instantiation we use

SPOT is stated for any string-commitment scheme and any base argument with the properties of Section 3. This appendix describes the ones we use, as far as the guarantees depend on them.

<b>The commitment scheme.</b> It is a Merkle tree over a hash function $H$ whose leaves are hiding commitments. The positions of the string are grouped into rows, and each leaf commits to one row under the first scheme of Halevi and Micali (1996), which is built from $H$ and a public key fixed in advance. The root is the commitment to the string, and an opening of a position consists of an opening of its row's leaf commitment and the hashes on the path from that leaf to the root.

<b>Binding.</b> Two accepted openings of one position to different values yield a collision in $H$, whatever the key. So the scheme is binding when $H$ is collision-resistant.

<b>Hiding.</b> Hiding is statistical and does not rest on any property of $H$. The key is a public matrix $A$ over the field with two elements. Each leaf masks its row's digest with $As$, where $s$ is a fresh, uniformly random salt, and $As$ is close to uniform even given the hash of $s$. Halevi and Micali draw the matrix afresh for each commitment, and their bound on hiding is an average over it. Here the matrix is fixed, so we use the bound in a different form: it holds for all but a small fraction of matrices, and we assume that the fixed one is not among them.

With our parameters the digest has 512 bits, the salt has 1536 bits, and $A$ has 512 rows and 1536 columns. The hash of the salt reveals at most 512 bits about it, so by the leftover hash lemma the mask is within $2^{-257}$ of uniform on average over $A$. By Markov's inequality, all but a $2^{-64}$ fraction of matrices keep any two leaves within $2^{-192}$ of each other, and the hiding error is then at most $L\cdot 2^{-192}$ for $L$ leaves.

<b>Zero-knowledge of the base argument.</b> The base argument we use masks every message it sends with fresh randomness, following VEIL (Dalal, Hemo, Rabinovich and Rothblum, 2026), and has the verifier commit to its randomness in advance, following Goldreich and Kahan (1996). That it is statistical zero-knowledge is a separate result about the base argument, and it is verified (Appendix D). Because the simulation is statistical, given uniformly random salts and pads, the guarantee holds against an unbounded distinguisher. The one hardness assumption is that the verifier cannot open its commitment in two ways. It constrains the verifier while the protocol runs, and not whoever examines the view afterwards.

## Appendix B. Explicit bounds

This appendix gives the results of Section 3.5 in explicit form. The public parameters are fixed, including the hash function $H$ of Appendix A, and there is no security parameter and no random oracle. No asymptotic assumption can be made about a fixed $H$, so each bound is stated in terms of the probability that an explicit algorithm, built from the prover, outputs a collision in $H$. The verifier's randomness is uniform and hidden from the prover until sent.

<b>Definition B.1 (explicit forms).</b> The explicit bounds, and the theorems of Section 3.5 that rest on them, assume three quantitative properties. The commitment scheme is binding by collisions: two accepted openings of one position to different values yield a collision in $H$, as for the scheme of Appendix A. The base argument has the two properties below, which ask more of it than Definitions 3.10 and 3.11 do.

- <b>Knowledge-soundness with parameters </b><b>$(\kappa,\gamma)$</b><b>.</b> There is an extractor $\mathcal E$ such that, for every instance $x$ and every prover strategy $\mathcal P'$, the probability that the verifier accepts while $\mathcal E$'s output is not a witness is at most $\kappa+\sqrt{\gamma\,\beta}$. The probability is over the randomness of the prover, the verifier and the extractor. The extractor is given $x$ and the verifier's internal randomness $r$. It may rewind $\mathcal P'$, rerunning it on verifier randomness of its choice with the prover's internal randomness held fixed. It is not given the prover's internal randomness, and its cost is counted in reruns. $\kappa$ and $\gamma$ do not depend on $\mathcal P'$. $\beta$ is the probability that the fork finder outputs a collision in $H$. The fork finder is an explicit algorithm that runs $\mathcal P'$ twice from a shared prefix of the verifier's randomness and looks for a collision among the verifier's hash evaluations.
- <b>Zero-knowledge with error </b><b>$\varepsilon_{\mathrm{zk}}$</b><b>.</b> There is a simulator $\mathrm{Sim}$ that is given $x$ and may run and rewind the verifier strategy, such that for every verifier strategy $\hat{\mathcal V}$, every $z$ and every $(x,w)\in\mathcal R$, the view and the simulator's output are within statistical distance $\varepsilon_{\mathrm{zk}}$.

In this appendix the base argument is knowledge-sound with parameters $(\kappa,\gamma)$, taken as the largest over the blocks, and $\rho$ is the number of rounds of the verifier in the protocol in question.

<b>From the explicit forms to the definitions.</b> The explicit form of knowledge-soundness gives Definition 3.10 with knowledge error $\kappa$ under two conditions: the extractor runs in expected polynomial time under the cost convention of Definition 3.10, and $\sqrt{\gamma\,\beta}$ is negligible for every efficient prover strategy. The second holds when $\gamma$ is polynomial and $H$ is drawn from a collision-resistant family. The explicit form of zero-knowledge is a black-box simulation. It gives Definition 3.11 when $\varepsilon_{\mathrm{zk}}$ is negligible and the simulator runs in expected polynomial time.

Goldreich's formulation of knowledge-soundness asks for more: an extractor that keeps trying until it finds a witness, in expected time inversely proportional to the gap between the acceptance probability and the knowledge error. It would follow by repetition if the bound on failure held for each fixed choice of public parameters and commitment, and not just on average over them. We do not claim this.

<b>Theorem B.2 (explicit form of Theorem 3.15).</b> Fix $C$ before the prover's randomness $\omega$ and the verifier's randomness $r$ are drawn. For every prover strategy $\mathcal P'$ for Protocol 1,

$$
\Pr_{\omega,r}\Big[\mathcal V^{\mathcal P'}(C,r)=\text{accept}\ \wedge\ Q(r)\cap T_{\hat\tau_\omega}\neq\varnothing\Big]\le\kappa+\sqrt{\rho\,\gamma\,\beta}+N\sqrt{\beta'}+\beta'
$$

Here $\beta$ is the probability that the fork finder of Definition B.1, built from $\mathcal P'$, outputs a collision in $H$, and $N$ is the number of gates of $C$. $\beta'$ is the probability that a second explicit algorithm outputs a collision in $H$. It runs the prover twice, independently, at one value of the prover's randomness, applies the base argument's extractor to both runs, and looks for a collision among the hash evaluations made. The first two terms come from the base argument, and the last two from the binding of the commitment.

<b>Theorem B.3 (explicit form of Theorem 3.18).</b> There is one extractor such that, for every $k$, every $C$ and every prover strategy $\mathcal P'$, the probability that the verifier accepts while the extractor's output is not a witness for $(C,c)$ in $\mathcal R_k$ is at most

$$
\alpha^2\big(\delta_s(k)+\kappa\big)+\eta+\sqrt{\alpha^4\rho\,\gamma\,\beta}+\beta_{\mathrm{ext}}
$$

where the symbols are as follows:

- $\alpha=2^{2^{-10}}$, the slack that the extractor's analysis allows for its union bounds;
- $\eta=(1+\alpha)\,2^{-127}+e^{-m_o/8}+\alpha\,e^{-m_i/8}$, where $m_o$ and $m_i$ are the target numbers of accepted reruns that the extractor collects at its outer and inner levels. They are $m_o=2^{10}(2n_o+1)$ and $m_i=2^{10}(2n_i+1)$, where $n_o$ and $n_i$ are parameters of the extractor's analysis and $n_o$ is at least the number of blocks. So $m_o$ is at least about $2048$ times the number of blocks;
- $\beta_{\mathrm{ext}}$ is the probability that an explicit algorithm, which runs the extractor once, finds a collision among its hash evaluations. Its running time is bounded in expectation only.

The extractor's expected number of reruns of the prover is at most $1+m_o+m_o\,m_i\,(1+q\mu)$, where $\mu$ is the expected number of reruns made by the base argument's extractor. The expectation is over the execution in question as well as the extractor, which reruns the prover only when that execution accepted. When it accepts and neither level reaches its limit on attempts, the extractor collects $m_o\,m_i$ accepted reruns, each with a freshly drawn sample. These are separate from the reruns made inside the base argument's extractor.

The term $2^{-127}$ in $\eta$ reflects a fixed limit on the extractor's attempts. Theorem 3.18 assumes that this limit, $m_o$ and $m_i$ all grow with $\lambda$, so that $\eta$ is negligible while the rerun bound stays polynomial.

The same bound holds for Protocol 2 and the condition of Claim 3.23, with $\delta_s(k)$ computed from the two-stage design.

## Appendix C. Terminology and precedents

For each term, the table gives the closest precedent and how this note differs from it. A reference with no author is to Thaler's book. "Goldreich" is <i>Foundations of Cryptography</i>, Volume 1, cited from the 1998 revised draft of its Chapter 4. Numbering and wording may differ in the 2001 Cambridge edition, where only Definition 4.7.2 has been confirmed. Terms that are not listed are defined in this note.

The following are used as in Thaler's book: relation, instance and witness (§12.2.1); IOP, round and public-coin (§10.1, §3.3); prover strategy, verifier strategy and prescribed prover (Definitions 3.1 and 11.1); internal randomness, completeness error and perfect completeness (Definition 3.1); soundness error (Definitions 3.1 and 9.1); statistical and dishonest-verifier (§11.1); efficient and negligible (§5.3).

| Term in this note | Closest precedent | Differences |
| --- | --- | --- |
| circuit, gate, input, output gate, in-neighbor | §4.2, §4.6 | Thaler calls every vertex a gate, including inputs. We reserve "gate" for nodes that compute something, and allow any finite alphabet and any gate functions. |
| trace | "transcript", §8.2.1.1 | Renamed, because "transcript" also means a protocol's messages (§3.1). Thaler notes the name "trace" in §6.5.3. Ours covers boundary nodes only. |
| value | "value" of a gate, §8.2.1.1; "symbol" of a proof string, Definition 9.1 | One word for both, because the proof string is a trace. |
| correct | "correct transcript", §8.2.1.1 | Applied to one block. |
| the relations $\mathcal R_{\mathrm{blk}}$ and $\mathcal R_k$ | A footnote in Chapter 13, on commit-and-prove SNARKs | Thaler mentions arguments which show that the prover knows an opening of a commitment with some property. $\mathcal R_{\mathrm{blk}}$ is a statement of exactly that kind. $\mathcal R_k$ is one that tolerates fewer than $k$ incorrect blocks. |
| PCP, proof string, query | Definition 9.1 | Soundness is stated for proof strings, because every circuit has a fully correct trace. |
| argument system | Definition 3.2; Goldreich, Definition 4.8.1 | Goldreich also requires the prescribed prover to run in polynomial time given an auxiliary input. |
| knowledge-soundness, extractor, knowledge error, argument of knowledge | Goldreich, Definitions 4.7.1 and 4.7.2, after Bellare and Goldreich. Thaler's account in §7.4 and §9.2.1 is informal. | Goldreich's Definition 4.7.2 is the source of the knowledge error. Ours differs in form. Goldreich bounds the expected time for the extractor to find a witness. We bound the probability that the verifier accepts and extraction fails, over the public parameters as well as the parties' randomness. That is the usual form for arguments with public parameters, and the form of our bounds. Goldreich's follows from it by repetition only if the bound holds for fixed public parameters. Four further changes: we cover only efficient prover strategies, with a negligible slack; the prover may choose part of the instance; the knowledge error may depend on the instance; and completeness may hold for a smaller relation. A large knowledge error has a precedent: Goldreich's 3-coloring proof has knowledge error $1-1/\lvert E\rvert$. |
| argument of knowledge for a pair of relations | None in Thaler or Goldreich | Complete for the smaller relation and knowledge-sound for the larger. Here the smaller is $\mathcal R_1$ and the larger is $\mathcal R_k$. |
| zero-knowledge, simulator, View, auxiliary input | Goldreich, Definition 4.3.8, with the view of Definition 4.3.3. Thaler's Definition 11.1 is informal. | Adapted from Goldreich's Definition 4.3.8, with three changes. Goldreich asks for computational indistinguishability, and we ask for negligible statistical distance, a variant he gives as an exercise. Goldreich's simulator runs in strict polynomial time, and ours in expected polynomial time, which he treats as a liberal formulation (Definition 4.3.5). Goldreich's definition is for an interactive proof for a language, and ours for an argument for a relation. The explicit form in Definition B.1 is a black-box simulation (Goldreich's Definition 4.5.7) with an explicit error. |
| sequential composition of zero-knowledge | Goldreich, Lemma 4.3.9; Goldreich and Oren | Goldreich's lemma repeats one prover on one common input, with fresh prover randomness each time. We compose runs on different instances, which Goldreich and Oren treat. Claim 3.20 gives the argument for this protocol directly. |
| statistical zero-knowledge argument | Brassard, Chaum and Crépeau (1988); Naor, Ostrovsky, Venkatesan and Yung (1998); Nguyen, Ong and Vadhan (2006) | They obtain it from a commitment that is statistically hiding and computationally binding, part of which the prover opens. Here no value of the trace commitment is opened to the verifier: the base argument proves facts about the committed values, so it must be statistical zero-knowledge as well. |
| string-commitment scheme, opening | §7.3.2.2, §12.3; Halevi and Micali (1996), Section 2.6 | Thaler's commitment scheme has KeyGen, Commit and Verify (§12.3). Ours has no key generation: its public parameters are fixed in advance. It opens single positions, as Thaler's string commitment does (§7.3.2.2). Halevi and Micali define a non-interactive string commitment by a sender algorithm and a receiver algorithm, and open the whole message at once. |
| binding, hiding | §12.3; Goldreich, Definition 4.8.2; Halevi and Micali (1996), Section 2.6 | Goldreich says "unambiguity" and "secrecy", and Halevi and Micali say "non-ambiguity" and "secrecy". All three combine statistical hiding with computational binding, as we do. Halevi and Micali's secrecy is an explicit bound on the statistical difference between commitments to any two messages, which is our hiding with error. |
| Merkle tree | §7.3.2.2 | Leaves are hiding commitments under the scheme of Halevi and Micali (1996), with its key fixed in advance where theirs is drawn afresh for each commitment. |
| Kilian's argument system | §9.2, §9.2.1 | Protocol 1 proves the verifier's checks through the base argument instead of revealing the queried values. |
| base argument | "base IP", §5.4; "base SNARK", chapter 18 | Thaler's word for the protocol that a construction repeats or composes. |
| transform | "transformation", §5.2 | Short for transformation, used as in "the Fiat-Shamir transformation": a construction that applies to any input of the right kind. |
| explicit bounds for a fixed hash function | Rogaway (2006), "Formalizing human ignorance" | Each bound is stated in terms of the probability that an explicit algorithm, built from the prover, finds a collision. |

## Appendix D. Status of formal verification

The explicit bounds of Appendix B are being formally verified in Lean. Verified means that a statement is proved in Lean under its stated hypotheses. The table lists what is still being worked on, and every other explicit bound is verified. The asymptotic statements of Section 3 are derived from the explicit bounds on paper, as Appendix B describes. That derivation is not formalized.

| Result | Verified so far | Still being worked on |
| --- | --- | --- |
| Claim 3.19, repeated audits | The bound for one concrete one-stage system. | The general statement given here. |
| Claim 3.20, zero-knowledge of Protocol 1 | Zero-knowledge of a single run of the base argument, against every verifier strategy. Hiding of a leaf commitment for a random key, with the bound on the fraction of bad keys. | The step that derives the claim from these two, which still rests on open premises. |
| Claim 3.22, choosing interiors late | Not yet confirmed. | To be confirmed. |
| Claim 3.23, the two-stage guarantee | The bound in a form stated for every acceptance threshold, under one premise that is still open. | That premise. The count form given here is derived from the verified form by hand. |
| A two-stage analogue of Theorem 3.15 | Nothing yet. | The statement and its proof. Repeated audits for two stages depend on it. |
| Zero-knowledge of Protocol 2 | Nothing yet. | The statement and its proof. |

Two further assumptions concern setup and randomness: that the fixed key of the leaf commitments is a good one, and that the salts are uniformly random.
</content>
</page>