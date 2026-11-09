# Leverage

What one incorrect proof block's served output tells a decoder about the model's weights beyond what honest queries
tell it. A fault sets one coordinate of one activation at one position to a chosen value; everything after it is honest.
The models are small open ones (GPT-2 small, Qwen2.5-0.5B, Qwen2.5-1.5B), run on the CPU in plain torch (`lm.py`).

- `check.py`: `lm.LM` against Hugging Face transformers' logits, and its weight edits against changed weights; run it
  first on each model.
- `clamp.py`: the range clamps' boxes. `--part hosts` lists the tokens that host the attention sink at position 0, and
  `--part calibrate` fits one box per coordinate and position class, read from positions and token ids alone (where
  the sink forms: every position before a sequence's first host, the run of its token and `--window` − 1 positions
  after it; the tokenizer's special tokens after them; the rest; and with `--run R` everything from the R-th copy of a
  host token in a row on, since a long run forms the sink again and puts sinks on the ordinary text after it), with
  `--headroom` powers of two over the honest maximum, one value or one a class; a linear input's box goes through its
  FP8 per-128-group UE8M0 scale. Its traffic is `traffic.py`'s hand-written texts and, with `--mixed N`, the pinned
  datasets' chat, web and multi-script texts; `--first-tokens -1` adds every vocabulary token at position 0, and
  `--run` the calibration run texts. `--shard i/n` splits a part across runs, and `--part merge` joins calibrate
  shards. `--part complete` serves the held-out texts under the boxes and counts the honest values they cut and the
  served tokens that change; `--part explain` names each value outside its box with its token, context and class, and
  each text's sink-like positions. With `--runs` both add the held-out run texts, and with `--long N --cap C` they
  check N held-out Pile documents of C tokens instead.
- `reach.py`: one fault's reach in the served tokens by site, layer and value. Raw reach is the tokens that differ
  from the honest continuation; direct reach is the tokens no honest query on the served prefix reproduces. It
  reports their Shannon bound, their rank code and the per-position KL. `--temperature` samples under a shared-noise
  coupling. Under `--boxes`, `--kinds` adds the in-range faults: a value at its box (at a linear input, the most a
  block writing the whole group can put in one coordinate), one FP8 code at its own group scale, a group's scale at
  its bound, and all 128 codes at their group's box.
- `extract.py`: the weight bits concrete attacks read a fault from top-k log-probabilities. The parts are `final`
  (the LM head, from the final norm's output), `rows` (an output-side row at any layer), `block` (under `--boxes`, a
  fault that writes an aligned block of `--widths` coordinates of the final norm's output, read by least squares
  over `--read-over-w` faults a coordinate: the bits a fault as the block a faulting proof block writes widens) and
  `honest` (Carlini et al.'s SVD attack with the same observables). `--fp4` quantizes the LM head to NVFP4; `--boxes` clamps every read
  and faults at the box amplitudes; `--differential` subtracts an honest query on the same prompt, which reads
  in-range faults the faulted response alone doesn't. `--peel` has the rows' decoder estimate the LM head and the
  final norm's g and b from faults on the last layer instead of knowing them: `saturate` unclamped, `box` or `own`
  under `--boxes` at the final site's box or one FP8 code's reach, repeated over `--peel-contexts` prompts.
- `attack.py`: a tokens-only attack with chosen in-range interior faults. A decoder that knows every weight but one
  down-projection entry, or one unit's whole column, picks each fault (prompt and value inside the box) to split its
  posterior and keeps the candidates that agree with the served tokens. It reports exact weight bits a fault.
  `--range own` holds each fault to one FP8 code at its token's own group scale (charged as one element), `box` lets
  it go anywhere in the box (beyond one code, a block writing the whole group); `--obs` is the served tokens it reads.
- `report.py`: Markdown tables from their outputs, with λ − 1 = B/(c + ℓ) at the charges `--charge` names.

Each writes under `--out`; under `research run`, give it `"$OUT"`:

~~~text
uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/check.py --model gpt2 --out "$OUT"
uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/reach.py --model gpt2 --new 256 --out "$OUT"
uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/extract.py --model qwen2.5-0.5b --part final --fp4 --out "$OUT"
uv run python benchmarks/leverage/report.py --reach "$OUT"/summary.json --extract "$OUT"/extract-*.json

uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/clamp.py --model gpt2 --part hosts --window 4 --out "$OUT"/clamp
uv run --extra torch-cpu --with tokenizers --with pyarrow python benchmarks/leverage/clamp.py --model gpt2 --part calibrate --mixed 200 --window 4 --hosts "$OUT"/clamp/hosts.pt --first-tokens -1 --out "$OUT"/clamp
uv run --extra torch-cpu --with tokenizers --with pyarrow python benchmarks/leverage/clamp.py --model gpt2 --part complete --mixed 100 --first-tokens 0 --boxes "$OUT"/clamp/boxes.pt --out "$OUT"/clamp
uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/extract.py --model gpt2 --part block --boxes "$OUT"/clamp/boxes.pt --ks 20 --out "$OUT"/block
uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/reach.py --model gpt2 --boxes "$OUT"/clamp/boxes.pt --kinds box,own,scale,codes --sites every --new 64 --out "$OUT"/reach-clamped
uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/extract.py --model gpt2 --part rows --boxes "$OUT"/clamp/boxes.pt --differential --row-site mlph --pick widest --values box own scale codes --out "$OUT"/rows
uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/attack.py --model qwen2.5-0.5b --boxes "$OUT"/clamp/boxes.pt --layers -2 -4 --out "$OUT"/attack
~~~
