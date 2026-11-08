# Leverage

What one incorrect proof block's served output tells a decoder about the model's weights beyond what honest queries
tell it. A fault sets one coordinate of one activation at one position to a chosen value; everything after it is honest.
The models are small open ones (GPT-2 small, Qwen2.5-0.5B), run on the CPU in plain torch (`lm.py`).

- `check.py`: `lm.LM` against Hugging Face transformers' logits; run it first on each model.
- `reach.py`: one fault's reach in the served tokens by site, layer and value. Raw reach is the tokens that differ
  from the honest continuation; direct reach is the tokens no honest query on the served prefix reproduces. It
  reports their Shannon bound and the per-position KL.
- `extract.py`: the weight bits concrete attacks read a fault from top-k log-probabilities. The parts are `final`
  (the LM head, from the final norm's output), `rows` (an output-side row at any layer, under saturation) and
  `honest` (Carlini et al.'s SVD attack with the same observables). `--fp4` quantizes the LM head to NVFP4.
- `report.py`: Markdown tables from their outputs, with λ − 1 = B/(c + ℓ) at the charges `--charge` names.

Each writes under `--out`; under `research run`, give it `"$OUT"`:

~~~text
uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/check.py --model gpt2 --out "$OUT"
uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/reach.py --model gpt2 --new 256 --out "$OUT"
uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/extract.py --model qwen2.5-0.5b --part final --fp4 --out "$OUT"
uv run python benchmarks/leverage/report.py --reach "$OUT"/summary.json --extract "$OUT"/extract-*.json
~~~
