# inference_only: the inference-only policy of Llama 3.2 1B, and its one number

One command prints the inference-only policy of Llama 3.2 1B as served, and one number for it: the least slowdown of
training run inside the policy's inference blocks, under the min-cut bound
(`verity/protocols/inference_only/rho/rho/min_cut.py`, whose `HE` names its Lean statement).

~~~sh
uv run verity bench inference_only --stated-only     # the served policy and its number, from its stated numbers alone
uv run verity bench inference_only --stated-only --uncredited-share 0.1%   # and the served class under the value tie
research run --on auto --mem-gb 16 --cwd clone -- bash -lc \
  'uv sync -q --all-packages --extra torch-cpu && uv run verity bench inference_only --out "$OUT"'
~~~

The second form also builds the class this tree can build and runs the Lean count on it (`lake build` of the check's
executable, then the executable on the class's instance). That is heavy work and it builds inside the tree, so it goes
through `research run` in the run's own clone.

## What it prints

1. **The served policy**, first, and its number, which is the command's number and the JSON's `value`. The policy is
   **stated, not built**: `served.toml` holds the numbers the serve states for its class before it starts, this tree
   builds no circuit of that class, and so no check here runs on it. `--policy FILE` prices another stated policy.
2. **The class this tree builds**, labelled as not served: the exact-int8 checkpoint with its layers' linears inside the
   circuit, which is what registering the linears inside the circuit would give. Its policy is computed by the
   integration's code (`verity_vllm`'s `window_circuit.class_circuit` and `block_counts`), the Lean count
   (the check's executable with `--count --detail`, `built.lean_count`) runs on it and its verdict and cost are
   printed, then its number,
   then the record's three divisions (the session, the position, the proof block). `--model`, `--class
   SLOTS,PROMPT,OUTPUTS` and `--division` choose the built class and which of its blocks the built number is of.
3. **Both over monitoring periods**: for one week, one month and one year, on one datacenter and on the world's compute
   (the slowdown calculator's fleets and periods), the policy's blocks the fleet holds at capacity, the training MACs
   they hold at most, and that as a share of the compute they cover (100% says the bound leaves all of it to
   training); with one line each on what is proved, what is a premise and what is scaled.
4. **The served class under the value tie** (`tied.toml`, below), only at a share of a block's work that no PoUW check
   credits: `--uncredited-share S`, a number from 0 to 1 or a percentage. It has no default, and without it the command
   prints no figure of the tied policy and says why. With it the section opens with its label: the class is not
   served, and a share typed on the command line rests on no PoUW verdict, so every figure of the section holds only
   where PoUW's checks credit all but that share of every block's work.
5. **One JSON object**, the last line, also written to `--out DIR` (default `$OUT`) as `inference_only.json`: `value`,
   `unit`, `check`, `rests_on`, `work_is`, `policies` (each with its `c`, `k`, `n` and their `units`), `divisions`,
   `periods`, `tied` (the tied section's figures, `null` without a share), `commit`, and `run` where the environment
   names one. `check` is the Lean count's verdict on the policy the value is
   of, `refused` or `accepted` by its executable. For a stated policy it is `not run`, which is no verdict: this tree
   has no instance of the stated class to give the count. The count's instance names no θ and carries no wire, so it
   counts every weight and wired row a block reads as fresh input: on the built class it counts `c` exactly and
   refuses on `k`, and an instance of any other class has the same form.

For each policy it prints what `k` is made of. For a stated policy: the outputs of the linears the class leaves outside
the circuit, from the model's shape (layers × positions × widths × bits a value), and the class's scan state row, with
what they leave of `k` printed as unexplained. For the built class: the rows a block reads, by source.

## What the number means

A policy is `(c, k, n)`: blocks of at most `c` Boolean gates, each reading at most `k` fresh input bits, at most `n` of
them. `c` counts two-input Boolean gates, as the Lean count counts a block Definition's by the gate table; the
integration's `inference_blocks.budgets` also calls a number c, the block's gates at their widths, which is not this
one. The number is `min_cut.bound`'s slowdown on one block: `max(1, N / limit)`, where `N` is the block's own work in
MACs and `limit` is the most training work the count lets one block of the policy finish (`Bound.macs_a_block`). The benchmark
passes the bound `N`, the output wires `O = N × o × b` (`o` the output values a MAC of the model's own forward matmuls,
`b` a BF16 value's 16 wires) and the circuit's size `c`. With the circuit's size `c`, `c` cancels: the number is the blocks the count needs for one
block's work. A number of 1 says the bound shows no slowdown: one block's `k` lets it hold all of its own work as
training.

The served class leaves the layers' linears outside the circuit, so a block reads their outputs as fresh input and its
`k` is large; the class with the linears inside reads only its scan state row. `k` is the lever.

**Which work.** Every figure is for training in bf16 words on the model's own matmul shapes, with every weight of the
work and its embedding table outside the registered weights θ. (H) is false of a work that reads a registered weight,
which has no path from a fresh input, or a registered table's row, which is a function of θ and a few fresh gates. So
no figure covers such a work, and the print says so beside each policy's bound and in the JSON's `work_is`.

## The served class under the value tie

`tied.toml` is the served class with every tied value right: PoUW binds each output of the layers' linears to the
linear's operands, so a block reads as fresh input only its scan state row and the execution phase's salt. Nothing
serves it and this tree builds no circuit of it. Its cut number with every word tied is far above the served policy's
1, and is never printed alone. Two things stand between it and a claim, and the section prints each figure with both
(`tied.py`):

- **The untied words**: outputs of the linears that PoUW's sample accepted without tying. A node's accepted execution
  phases hold at most the file's `words` of them for each `blocks` blocks of one phase, so a fleet's period holds that
  many a block of a phase, over the phases the period has. They enter the count over the period
  (`min_cut.bound_charged`) under three readings, each printed under its name (`Charged.reading`): one `β_x` with the
  threshold at which the figure is largest, one `β_x` at the top budget with no block charged, and a `β_x` a block.
- **The uncredited share** `s`: a developer gets that share of a block's work whole, so each cut number `ρ` is printed
  with its limit at `s`, `1 / (s + 1/ρ)` (`min_cut.capped`), and the pooled total in gates is at that limit. The
  words that work computes are tied by nothing either: a fourth reading counts them as fresh input, `k + 32 u` with
  `u` the most words the share's MACs feed (a word of a linear's output is fed by at least the least fan-in of the
  model's linears).

It prints one block, then one week, one month and one year on one datacenter and on the world's compute: the blocks,
their gates and the untied words; then under each reading the blocks charged at most, the cut number, its limit at
`s`, and the training the period's blocks hold at most at that limit, in gates (the gates `c` counts) and as a share
of the blocks' gates. Three lines say what is proved (each count's statement with the standing the locks give it,
`rho.lean.status`; a `β_x` a block is no Lean statement, and no limit is proved, being arithmetic on the typed share),
what is a premise (the share, the tie, the bound on the untied words, (H) and (E)), and what is scaled (every row).
`--policy tied.toml` is refused: it would print the cut number alone.

## What it does not mean

The number is a block's ratio under two assumptions, on a check that Lean does not yet accept for any class.

- **Two assumptions.** The bound holds under (H), the cut's hardness, and (E), its exports, conjectures
  (`min_cut.HARDNESS` and `min_cut.EXPORTS` are their Lean names). The values of (H)'s β by `k` are rho's
  (`verity/protocols/inference_only/rho/data/premises.toml`).
- **No class passes the check.** The Lean count is given the session Program with no θ and no dataflow, so it counts
  every row a unit reads as fresh and refuses the policy on `k`. A built policy's `c` and `k` are the Python count of
  the record's description, which the Lean check does not establish; a stated policy's are not checked here at all.
- **A block, not a deployment.** The periods table multiplies one block's ratio through the blocks a fleet holds at its
  data sheet's rate. What a deployment registers, what it leaves free and what its proofs miss are the slowdown
  calculator's (`uv run python verity/protocols/inference_only/rho/periods.py`).
- **Units, not converted.** `N` and `limit` are in the policy's own MACs, which count a MAC of the layers' linears as one
  and a BF16 MAC (attention, the LM head) as two, the card's dense int8 rate against its BF16 rate. The bound is at
  `F = 1`, which charges at most two of a training step's three kinds of product; the two thirds that leaves of a step
  where the weight-gradient products are not charged is not applied.

## The policy file

`served.toml`, and any `--policy FILE`: `source`, one sentence saying where the numbers come from; `model`, a checkpoint
the integration's manifests pin; `class`, `[slots, prompt ids, outputs]`; `c` in Boolean gates a block, `k` in fresh
input bits a block, `n` in blocks; `[work]`, one block's MACs by kind (`linears`, `attention`, `lm_head`) and `total =
linears + 2 × (attention + lm_head)`; optionally `[outside] linears`, the bits a value of the layers' linears'
outputs that a block reads as fresh input; and optionally `[fresh]`, what else a block reads as fresh input, each key
what it is and its value the bits. The tests hold `served.toml`'s work and its `k` by kind to what the tree
computes for its class; its `c` is held to nothing here, and the number does not read it.

`tied.toml` has the same keys, with no `[outside]` and the salt under `[fresh]`, and one table more, `[untied]`: `words`,
a node's bound on the untied words of its accepted execution phases for a phase of `blocks` blocks; `phase_seconds`;
`macs_a_word`, the MACs that feed one untied word at most; and `word_bits`, its fresh bits in the block that reads it.
The tests hold its `c`, `n` and work to the served policy's and its `k` to the scan state row and the salt. The
uncredited share is never in a file: it is the command line's.
