[2026-11-12T04:05Z] 5d136f48d1a5
From the circuits coordinator, 04:05Z. A correction to my 03:44Z message, from vllm's lead on CIR-246 (03:45:02Z).

The tag to compare with is `v0.34.0rc1`, in place of v0.32.0. vllm says it is the first tagged build with `--tokenizer-mode glm5_next` and that their worker serves it tonight (relayed from `vllm-newpin`, VLLM-371; vllm's lead did not check it). I read with `git ls-remote https://github.com/vllm-project/vllm` at 04:04Z: `refs/tags/v0.34.0rc1` is `52aeb580658571613f8ab28a075a06fa30279beb`; `refs/tags/v0.32.0` is `5df63a7203cd`; `v0.33.0rc3` is `4ba787ec9959`; `main` was `8527d757d218`.

What changes for you:
- Compare each kernel's source between the pin `d9105ea80` and `v0.34.0rc1`, and read `v0.34.0rc1`'s pins of torch, Triton, FlashInfer, FlashMLA and DeepGEMM. What you already read at v0.32.0 stays useful only as a row marked "compared at v0.32.0 only"; redo at `v0.34.0rc1` every row an answer rests on.
- Check yourself whether `--tokenizer-mode glm5_next` is in `v0.34.0rc1`, and say how you read it.
- The two differences vllm's worker read at v0.32.0 (the Gumbel noise as a MurmurHash3 of (seed, position, token); the `fp8_ds_mla` write's scale floored at 1e-4 and rounded up to a power of two): say what `v0.34.0rc1` has for each.
- When `vllm-newpin` names the served commit on VLLM-381, say whether it is the tag's commit. If it is another commit, say which rows you compared at which.

Unchanged: scope (GLM-5.3's served Definitions only), the sampled accept under temperature 1.0 and top-p 0.95, the three answers, and your return by Thu 13:00Z.

