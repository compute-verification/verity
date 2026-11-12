[2026-11-12T16:08Z] c94d5db456b9
grade to @training: TRN-138 job 4, the scope. Written Thu 12 Nov 08:08 AM PST. No code of the change is written; what I ran is a prototype outside the tree, on the VM, against #9029's head 4ae01884cad1 read-only (torch 2.14.0 CPU).

THE CRITERION I PROPOSE: stated at the call, forward and backward
The capture puts each scaled-dot-product attention call inside a marker `te_capture::attention::DTYPE`, DTYPE the query's dtype as the call takes it (before the math path's cast up). A node inside such a marker is attention's and is priced at DTYPE; every other node is priced as traced. No dtype is inferred after the fact.
- Forward: a torch-function mode in `_trace` wraps the call in a `record_function`, as `te_capture::backward` wraps a backward. It catches the call however the model imports it (it is torch's dispatch, not a patched name).
- Backward: the same mode hooks the autograd nodes the call made (those reached from the call's output whose sequence number lies between the call's start and its end): a pre-hook enters the marker, a hook leaves it. So the call's backward is inside the marker too, with no reading of the graph.
RAN (prototype, tiny shape, whole / block / lora jobs traced as a job traces them: `capture_steps`, three steps, bf16 autocast): the marker holds exactly the call. Forward: the three casts up, the two scalings, the two `bmm`, the softmax, the cast back. Backward: the incoming gradient's cast up, the four `bmm` (lora: the two its step asks for), the softmax's backward, the two scalings, the casts back. Nothing else is inside; the linears' `mm` are outside. With the route's autocast off the marker reads float32. The marker alone changes nothing: rows and |C| are equal with and without it.

WHY NOT FROM WHAT A RAW ROW ALREADY HAS
READ: a raw row holds op, phase, dims, token_dims, dtypes (both operands, as traced), affected, widths, self, gathered, rates. Nothing in it says "inside a call"; `_attention` is the counter's, the capture marks nothing.
RAN: `widths` does not read q's dtype through the cast. It walks back only through an operand the update reaches, and the weights are a float32 softmax's. The attention rows of the whole job: widths [16,16], [32,16], [16,32]; of lora (operand B not reached): [16,32], [32,32]. So "bmm, traced float32, two token dims, passed at 16" misses rows, and it would also take a float32 product a model writes on purpose between two token-indexed bf16 values.

WHAT IT PRICES
Every op |C| counts inside the call, at q's dtype: the six products, and with them the scalings (two forward, up to two backward), dropout's scaling and a mask's add where |C| counts one. That is `Counter._attention`'s list (`_shaped(q, s)`). The premise you gave names the six products only; the scalings follow the counter. They are 2 to 3 parts in 10^4 of |C| (job 1's ad48 figures). DEFAULT: scalings with the products, since TRN-126's exception names both (`bmm` and `mul`) and goes only if both move. Say "products only" and the exception shrinks to the scalings and stays.
Attention's products are left out of M (`he_of_rows`), so the theorem's work does not move: only |C| does.

WHAT IT MISPRICES
1. A call the card serves on its math path is computed at 32 and priced at 16: a job that turns the fused kernels off, or a call no fused kernel takes (INFERRED from torch's kernel selection as I remember it, not read here: q, k, v not 4-D). The trace is always taken under the math kernel and cannot see the card's choice: this is the premise, and the docstring names it. With #9039 the count of such a job would name float32, so count and trace would differ there, visibly.
2. A float32 product a model computes on purpose inside autocast: outside every call, priced as traced, 32. Right.
3. Attention by hand (matmul, softmax; `nn.MultiheadAttention` with need_weights): no call, traced at bf16, priced 16 as traced. Right.
4. A float32 job: the marker reads float32, priced 32. Right. A call on `.float()` operands inside autocast: 32. Right.
5. A float16 query: priced at the bf16 Definition, as `gates` prices f16 everywhere.

THE CHANGE'S SIZE (INFERRED from reading the code; I call it small, about 150 lines and the tests)
- `capture/trace.py`: the mode and the hooks, about 35 lines.
- `capture/structure.py`: each node's marker dtype (as `_phases` reads the phase), used at the three places a node is priced; `products.step_gates` the same three.
- `capture/products.py`: a raw row gains one field, `attention` (the call's query dtype, null outside a call), and `priced_dtype` reads it. `work.repriced` then compares the price with what the circuit counted, which are one rule again.
- `library.priced_in` prints the priced dtype; the docstrings and README sentences you listed.
- CAPTURE and VERSION move. `READS` becomes 4.12.9 and the new one; 4.12.9's rule is untouched.
Two things outside the package's own files that the capture's move forces:
- `grader/grader.py` holds `CAPTURE = "4.23.0"` and refuses any other: one line.
- `grader/tests/test_slowdown_code.py` (the bf16 cases) asserts the trace's O equals the converter's circuit's. `against` takes a row's output width from its price, so at a bf16 step's attention the trace's O becomes 16 a word against the circuit's 32. DEFAULT: I change the test's expectation (O agrees but at the bf16 steps' attention products, as `widths` already does) and touch nothing else of the grader's. Say if you want O kept instead.

TWO QUESTIONS, EACH WITH THE DEFAULT I TAKE
a. The version. #9039 took 4.24.0 and also stands on #9029. DEFAULT: 4.25.0 here, so the two merge without one number meaning two captures.
b. TRN-126's exception: with the scalings priced, my reading is that it can go once #9039 holds this head (the trace's attention is then q's dtype, which is what `flops.dtype_of` names on a fused kernel's row). INFERRED until the test runs; I tell you, not onecalc.

I start the change now on these defaults: a new branch from 4ae01884cad1, a draft, base main. A word from you through the hook changes any of them. Dates stand: the draft and the figures by Fri 13 Nov 10 AM PST.

Not checked: a GPU trace (a job on a card traces on fake cuda tensors; the hooks ran on a CPU only); torch's own source for which calls fall to the math path; dropout and a trained mask inside a call (I will test both).

