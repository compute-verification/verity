# Leverage

What one incorrect proof block's served output tells a decoder about the model's weights beyond what honest queries
tell it. A fault sets one coordinate of one activation at one position to a chosen value; everything after it is honest.
The models are small open ones (GPT-2 small, Qwen2.5-0.5B, Qwen2.5-1.5B), run on the CPU in plain torch (`lm.py`).

- `check.py`: `lm.LM` against Hugging Face transformers' logits, and its weight edits against changed weights; run it
  first on each model.
- `clamp.py`: the range clamps' boxes. `--part calibrate` fits one box per coordinate and position class (position 0,
  the attention sink, and the rest) on `traffic.py`'s calibration texts, one power of two over the honest maximum;
  a linear input's box goes through its FP8 per-128-group UE8M0 scale. `--part complete` serves the held-out texts
  under the boxes and counts the honest values they cut and the served tokens that change.
- `reach.py`: one fault's reach in the served tokens by site, layer and value. Raw reach is the tokens that differ
  from the honest continuation; direct reach is the tokens no honest query on the served prefix reproduces. It
  reports their Shannon bound, their rank code and the per-position KL. `--temperature` samples under a shared-noise
  coupling. Under `--boxes`, `--kinds` adds the in-range faults: a value at its box (at a linear input, the most a
  block writing the whole group can put in one coordinate), one FP8 code at its own group scale, a group's scale at
  its bound, and all 128 codes at their group's box.
- `extract.py`: the weight bits concrete attacks read a fault from top-k log-probabilities. The parts are `final`
  (the LM head, from the final norm's output), `rows` (an output-side row at any layer) and `honest` (Carlini et
  al.'s SVD attack with the same observables). `--fp4` quantizes the LM head to NVFP4; `--boxes` clamps every read
  and faults at the box amplitudes; `--differential` subtracts an honest query on the same prompt, which reads
  in-range faults the faulted response alone doesn't.
- `attack.py`: a tokens-only attack with chosen in-range interior faults. A decoder that knows every weight but one
  down-projection entry, or one unit's whole column, picks each fault (prompt and value inside the box) to split its
  posterior and keeps the candidates that agree with the served tokens. It reports exact weight bits a fault.
- `report.py`: Markdown tables from their outputs, with λ − 1 = B/(c + ℓ) at the charges `--charge` names.

Each writes under `--out`; under `research run`, give it `"$OUT"`:

~~~text
uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/check.py --model gpt2 --out "$OUT"
uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/reach.py --model gpt2 --new 256 --out "$OUT"
uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/extract.py --model qwen2.5-0.5b --part final --fp4 --out "$OUT"
uv run python benchmarks/leverage/report.py --reach "$OUT"/summary.json --extract "$OUT"/extract-*.json

uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/clamp.py --model gpt2 --part calibrate --out "$OUT"/clamp
uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/clamp.py --model gpt2 --part complete --boxes "$OUT"/clamp/boxes.pt --out "$OUT"/clamp
uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/reach.py --model gpt2 --boxes "$OUT"/clamp/boxes.pt --kinds box,own,scale,codes --sites every --new 64 --out "$OUT"/reach-clamped
uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/extract.py --model gpt2 --part rows --boxes "$OUT"/clamp/boxes.pt --differential --row-site mlph --pick widest --values box own scale codes --out "$OUT"/rows
uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/attack.py --model qwen2.5-0.5b --boxes "$OUT"/clamp/boxes.pt --layers -2 -4 --out "$OUT"/attack
~~~
