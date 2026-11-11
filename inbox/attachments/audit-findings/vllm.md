**Title:** The legacy sample seeds are 8 or 4 bytes of the run root

**Team:** vllm (VLLM). Not filed: `research msg linear create` refuses a worker. Body for the coordinator to file. `research msg owners` names @circuits for `integrations/vllm/verity_vllm/commit/challenge.py`, so the coordinator may route row 1 there.

**Why.** Daniel ordered an audit of every concrete-security number we quote (PRF-523); the bar is $2^{-100}$ at a prover of $T = 2^{80}$ SHA-512 compressions. Your 00:15Z comment on PRF-523 says the integration quotes no security probability; a worker of proofs (PRF-537 AUDIT-RESOURCE) found none either. The one row is AUDIT-CRYPTO's (PRF-536), with the lines reread by me at `2b869ed35db5`. No second reader has checked it. Where your own reading differs, yours is the one to state, with why.

**Done when:** the quote is corrected, or the seed is changed, or a reply says our reading is wrong, with why. A first reply is asked for by Wed 11 Nov 6 PM PST.

**Open decision this row depends on (Daniel's, not decided):** whether $T$ counts SHA-512 calls or compressions. It moves the figure by nothing here: both widths are far below $80$ bits.

**1. Sample seeds of 64 and 32 bits.**
- Quoted: "The legacy seed with no verifier challenge, sampled after commitment: the run root's first 8 bytes, big-endian." (`integrations/vllm/verity_vllm/commit/challenge.py:118-120`, `root_seed`); "sha256(challenge || "|" || run_root)[:8]" (`:87-90`, `legacy_replay_seed`); "the run root's first 4 bytes" (`:129-131`, `kernel_check_seed`); `--replay-seed`'s default "derived from the run root after commitment" (`pipeline/commit.py:1025`).
- Evaluated: a sample chosen by 64 bits of a root the prover commits is steered within $2^{64}$ recommitments, and one chosen by 32 bits within $2^{32}$. At $T = 2^{80}$ such a seed gives $0$ bits; AGENTS.md's live-coin rule allows a seed from a committed root at two named rates, and these are not among them. `legacy_replay_seed` also reads SHA-256, beside the two named exceptions to "One hash".
- From: no theorem or lock record; these are code paths. AUDIT-CRYPTO did not trace whether a record of record takes any of them; the docstrings call them "legacy".
- How known: relayed from AUDIT-CRYPTO; the four quotes read by me. Not run.
- What would change it: a statement of which served paths, if any, read these seeds. If none does, the row is a docstring saying so.

**Not in it.** Two numbers you relayed on PRF-523 at 00:15Z are in other owners' rows: PoUB's headline read rate (MEM-248) and the inference-only slowdown (training). The tick bound (VLLM-270) and the reading of one request's $N$ and $n_{PU}$ (VLLM-302) are yours and in flight.

**Pointers:** PRF-536, `note:proofs/20261111T0013Z-finding-audit-crypto-primitives-at-2-80`; PRF-537.
