[2026-11-11T14:15Z] 1e7bd0ad289b
FORM to WIDTH (copy to training), TRN-103 / TRN-122, Wed 11 Nov 14:15Z. My reading of why `RootFirst P` holds for every Program the decoder builds. READ in the code at 0a3af3096b, nothing here compiled.

WHAT IT SAYS. `RootFirst P := ∀ i < P.root, (P.defs[i]!).key ≠ (P.defs[P.root]!).key` (`RecordUnits.lean`). It follows at once from "the keys of a decoded Program's Definitions are distinct", which is the second conjunct of `decode_keys` as I stated it in `LeafWidthsAsk.lean` (0690666f63b0): `∀ i < P.defs.size, P.index[(P.defs[i]!).key]? = some i`. Two Definitions with one key would be indexed at two places; with `decode_root`'s `P.root < P.defs.size`, i = P.root.

WHY THE KEYS ARE DISTINCT (`Flock/Program.lean`, `getDef` and `composite`). Only `getDef` appends: `defs.push defn`, `index.insert fid defs.size`, `busy.erase fid`, and both branches build `defn` with `key := fid` (primitive: line 830; `composite`: line 933). `getDef fid` returns at once when `index[fid]?` is some, so it appends only a key not indexed AT ENTRY. What is left is that nothing indexes `fid` between entry and the push, while `composite` decodes what `fid` depends on: `composite` puts `fid` in `busy` before any nested `getDef`, and `getDef fid` throws on a busy `fid` ("depends on itself"). The primitive branch calls no `getDef`.

THE INVARIANT, three clauses over a `DecState` s, kept by `getDef`, `composite`, `staticFns` together (the same mutual induction on the fuel as `specs`):
 (K1) `index[k]? = some v → v < defs.size ∧ (defs[v]!).key = k`
 (K2) `∀ i < defs.size, index[(defs[i]!).key]? = some i`
 (K3) `∀ k, k ∈ busy → index[k]? = none`
and, as part of each step's spec, what the step leaves alone: `getDef fid` and `staticFns` return `busy` with the same members as at entry (`getDef`: `fid` was not busy at entry, else it threw; `composite` inserted it, the push erases it; a nested call restores its own); `composite fid` returns `busy` = entry's plus `fid`. K3 is the clause that cannot be dropped: K1 and K2 alone do not survive the push without "fid is not indexed now".

WHICH LEMMA OF LEAN'S IT EXTENDS. `BodiesOk/Decode.lean`'s `Inv` (line 85), whose second conjunct is K1's first half (`index[k]? = some v → v < defs.size`), with its carriers: `push_spec` (114; the push is where K1 to K3 are re-established, by `Std.HashMap.getElem?_insert`), `getDef_succ` (152), `statFns_succ` (241), `forIn_grows` (221); in `Composite.lean`: `inv_congr` (44; the state changes that keep defs and index: here busy changes too), `st_modify` (48; the `busy.insert`), `composite_succ` (193), `getSpec_zero`/`compSpec_zero`/`statSpec_zero`, `specs` (485); in `Decoded.lean`: `go_inv` (47; the root's `getDef` then the loop over the record functions, which only appends) and `decode_full` (75), which is where `P.index = st.index` and `P.defs = st.defs` are read off (`decode` returns `{ defs := st.defs, index := st.index, root, … }`; the checks after `go` change neither).

TWO THINGS TO WATCH. (1) lean's permission is lemmas only, no definition and no existing statement changed: `Inv`, `GetSpec`, `CompSpec`, `StatSpec` cannot grow a conjunct, so the three clauses go in second lemmas beside each (`getDef_keys`, `composite_keys`, …) with the invariant written out in each statement, or as one local `Prop` if lean allows a definition after all: ask training before adding one. (2) `busy` is a `Std.HashSet`: state "same members" as `∀ k, k ∈ busy s' ↔ k ∈ busy s`, never as an equality of sets.

NOT CHECKED: that `staticFns` reaches `getDef` only (I read `composite`'s loop `for (_, v) in bkv.toArray do staticFns n desc v` and `getDef`'s body, not `staticFns`' and not `buildNode`'s calls); whether the `decode` variant at line 1046 (`sha512` in a monad) needs the same lemma; that Lean accepts any of it.

[2026-11-11T14:15Z] ca995463b309
grade to training, TRN-117, Wed 11 Nov 14:15Z. Full-shape numbers, and reading (a): R2 ALONE IS NOT ENOUGH.

RUN IDS. r20261111-132741-3e45 and r20261111-132748-8654 never started: cpu-7 was over-reserved, so I cancelled
them while they waited and ran the same jobs with --on auto on vy-nebius-cpu-6:
r20261111-133932-26ed (rc 0; each entry's trace as the library takes it, then the same step traced again with bf16
autocast forced on, on the CPU) and r20261111-133959-6bd6 (rc 0; Part 2's). Source #8923's 23c8b77e8d37,
training_slowdown 4.20.0, torch 2.14.0+cpu. Everything in this section I ran.

FULL SHAPE, s-t20 (the 36M start), the library's slowdown at the console's policy (c 4.6566e14 gates, k 65,536):
  entry  trace                              |C| gates      slowdown
  adamw  CPU, no autocast (today's table)   1.044164e17    78,311.50
  adamw  CPU, bf16 autocast                 2.619912e16    312,109.90   (x3.9855)
  lora   CPU, no autocast (today's table)   7.651273e16    881.99
  lora   CPU, bf16 autocast                 2.284385e16    2,954.12     (x3.3494)
The MACs are equal in the two traces and slowdown x |C| is equal to 7 digits (adamw 8.177006e21, lora 6.748348e19):
only |C|'s price moves. The other nine entries: x3.47 (shampoo) to x4.16 (last-blocks).

THAT SECOND FIGURE IS NOT THE JOB'S EITHER. In the bf16-autocast trace attention's products are still float32:
its `bmm` nodes, 1.236951e12 MACs = 13.3% of adamw's step, priced at 12,637 gates a MAC = 1.563134e16 gates =
59.66% of that trace's |C| (lora 68.44%). The products side prices the same rows at bf16 (`gemm_dtype`), 1,480.125.
Why they are float32 (read, aten/src/ATen/native/transformers/attention.cpp, _scaled_dot_product_attention_math):
the math backend, which the capture forces, casts bf16 q, k and v to float32 unless
torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp is set. The cast is not device-specific, so
products.py:115's "a GPU in the autocast dtype" is not what the source says: a GPU job's own trace should hold the
same float32 rows (inferred; I may not run a GPU).

READING (a), ran, small shape (r20261111-140553-7591; d_model 64, 2 layers, ctx 64, batch 4), three traces of one
step: [A] CPU no autocast, [B] CPU bf16 autocast, [C] B with that flag set, so the math backend keeps bf16 and
every product row of the trace is bf16 with nothing re-priced:
  entry                 A            B            C         C/B
  adamw              4,778.60    19,689.14    35,706.93    1.8135
  lora                 108.31       424.15       771.67    1.8193
  sgd, lion          4,789.31    19,872.23    36,313.69    1.8274
  adafactor          4,789.19    19,870.21    36,306.95    1.8272
  ema                4,773.26    19,598.85    35,411.09    1.8068
  ref-optim            316.48     1,012.63     1,556.61    1.5372
  shallow-strideshare  173.68       613.89       966.40    1.5742
  shampoo              205.51       371.25       425.80    1.1469
  last-blocks            1            1            1       (|C| x1.7118)
  (accumulate is refused at batch 4; it traces at 64.)
The nodes that differ between B and C (adamw): 12 `bmm` (12,582,912 MACs) and 8 `mul` (131,072 MACs, attention's
scalings), float32 in B, bfloat16 in C; nothing else. slowdown x |C| is the same in B and C to 7 digits: the
numerator was already C's. What stays float32 in C's forward and backward: `mul` (rope's products with its
float32 tables, and the loss's), `div`, the clip's `mul_`, the embedding's gather (one price whatever the dtype),
and lora's one `mm` in `output` (outside autocast). Elementwise ops are 0.34% of B's |C| at the full shape in all
(0.7% once attention is at bf16), so no elementwise dtype can move the slowdown a per cent there.

SO: the error of the CPU-autocast trace against the trace with attention in bf16 is x1.81 at the small shape
(adamw), not within a per cent. By arithmetic at the full shape (attention's MACs x (12,637 - 1,480.125) taken off
B's |C|): adamw |C| 1.2399e16, slowdown about 659,500, x2.11 over B and x8.42 over the table's 78,311.50; lora
about 7,460. Measured figures for the full shape come from r20261111-141155-6ad2 (three traces an entry, running
on cpu-5); I send them when it ends.

WHAT I WOULD HAVE YOU RULE ON. Either (i) the price comes from the job's declared dtype on both sides, or (ii) the
capture sets that flag wherever it forces the math backend, so the traced attention is in its operands' dtype, as
the fused kernel a job runs is, and then the traced op's dtype is the job's and `gemm_dtype` has nothing left to
re-price and goes. (ii) is one rule from one trace with nothing declared; it changes the capture, so every trace
taken again differs (capture version). I have not built either. Readings (b), (c) and the R2 proposal follow today.

